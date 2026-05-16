import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import solve_z_vector, make_intermediates, Hessian
from functools import reduce

def run_grad_decomp():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    x_y = td.xy[0]
    td_grad_obj = tdrhf_grad.Gradients(td)
    nocc = int(mf.mo_occ.sum() // 2); nvir = mf.mo_energy.size - nocc
    x, y = x_y; xpy = (x + y).reshape(nocc, nvir).T; xmy = (x - y).reshape(nocc, nvir).T
    orbv = cp.asarray(mf.mo_coeff[:, nocc:]); orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    xpy = cp.asarray(xpy); xmy = cp.asarray(xmy)
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T); doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) + cp.dot(orbv, cp.dot(dvv, orbv.T))
    z1 = solve_z_vector(td_grad_obj, x_y); z1ao = cp.dot(orbv, cp.dot(z1, orbo.T))
    dmz1doo = z1ao + z1ao.T + dmzoo
    
    # Difference density P_I = dmz1doo.
    # Total gradient excitation part:
    # 1. Hcore
    h1 = cp.asarray(mf.nuc_grad_method().get_hcore(mol))
    from gpu4pyscf.df import int3c2e
    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)
    g_hcore = float(cp.trace(dmz1doo @ h1[2])) + float(dh1e_td[0,2])
    print(f"H-core Part: {g_hcore:.8f}")

    mo_energy_gpu = cp.asarray(mf.mo_energy)
    mo_coeff_gpu = cp.asarray(mf.mo_coeff)
    nmo = mo_energy_gpu.size
    # im0 construction (partial)
    im0 = cp.zeros((nmo, nmo))
    from gpu4pyscf.lib.cupy_helper import contract
    vresp = td_grad_obj.base.gen_response(singlet=True, hermi=1)
    dmxpy = orbv @ xpy @ orbo.T
    veff0doo = vresp(dmzoo)
    veff_xpy = vresp(dmxpy + dmxpy.T)
    veff0mop = mo_coeff_gpu.T @ veff_xpy @ mo_coeff_gpu
    # simplified for TDA H2
    im0[:nocc, :nocc] = orbo.T @ (veff0doo + veff_xpy) @ orbo
    
    zeta = (mo_energy_gpu[:,None] + mo_energy_gpu) * 0.5
    zeta[nocc:, :nocc] = mo_energy_gpu[:nocc]
    zeta[:nocc, nocc:] = mo_energy_gpu[nocc:]
    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1

    im0_full = im0 + zeta * dm1
    im0_ao = reduce(cp.dot, (mo_coeff_gpu, im0_full, mo_coeff_gpu.T))

    s1 = cp.asarray(mf.nuc_grad_method().get_ovlp(mol))
    g_ovlp = -float(cp.trace(im0_ao @ s1[2]))
    print(f"Overlap Part: {g_ovlp:.8f}")
    print(f"im0 norm:     {float(cp.linalg.norm(im0_ao)):.6f}")

run_grad_decomp()
