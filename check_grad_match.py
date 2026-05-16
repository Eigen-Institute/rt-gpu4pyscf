import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import solve_z_vector, make_intermediates, Hessian
from gpu4pyscf.hessian import rhf as rhf_hess
from functools import reduce

def run_grad_match():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.nstates = 1; td.kernel()
    
    # We must use unscaled amplitudes (X^2 = 0.5) to match the grad engine's internal math
    state = 0
    x_y = tuple([cp.asarray(v) for v in td.xy[state]])
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td)
    inter = make_intermediates(h_obj, x_y, z1)
    
    # Extract Reference Intermediates from grad engine logic
    nmo = mf.mo_energy.size
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = nmo - nocc
    x, y = x_y
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = cp.asarray(mf.mo_coeff[:, nocc:])
    orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T)
    doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T))
    dmzoo += cp.dot(orbv, cp.dot(dvv, orbv.T))
    z1ao = cp.dot(orbv, cp.dot(z1, orbo.T))
    dmz1doo_ref = z1ao + z1ao.T + dmzoo
    
    print(f"P_I_prime norm:    {float(cp.linalg.norm(inter['P_I_prime'])):.6f}")
    print(f"dmz1doo_ref norm: {float(cp.linalg.norm(dmz1doo_ref)):.6f}")
    
    # Trace check
    S = cp.asarray(mf.get_ovlp())
    print(f"Tr(P_I_prime S):    {float(cp.trace(inter['P_I_prime'] @ S)):.6f}")
    print(f"Tr(dmz1doo_ref S): {float(cp.trace(dmz1doo_ref @ S)):.6f}")
    
    # MO basis check
    mo_coeff_gpu = cp.asarray(mf.mo_coeff)
    P_mo = mo_coeff_gpu.T @ S @ inter['P_I_prime'] @ S @ mo_coeff_gpu
    print(f"P_I_prime MO diag (occ): {cp.diag(P_mo[:nocc]).real}")
    print(f"P_I_prime MO diag (vir): {cp.diag(P_mo[nocc:]).real}")
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    from gpu4pyscf.df import int3c2e
    dh1e_ref = int3c2e.get_dh1e(mol, dmz1doo_ref)
    g_hcore_ref = float(cp.trace(dmz1doo_ref @ h1[2])) + float(dh1e_ref[0,2])
    
    # My assembly (used in diagnostic_grad and Hessian cross terms)
    aoslices = mol.aoslice_by_atom()
    h1ao_x = cp.zeros((3, nmo, nmo))
    p0, p1 = aoslices[0][2:]
    h1ao_x[:, p0:p1] += h1[:, p0:p1]
    h1ao_x[:, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    with mol.with_rinv_at_nucleus(0):
        vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
    h1ao_x += vrinv
    
    g_hcore_my = float(cp.trace(inter['P_I_prime'] @ h1ao_x[2]))
    print(f"H-core (Ref): {g_hcore_ref:.8f}")
    print(f"H-core (My):  {g_hcore_my:.8f}")
    print(f"Ratio:        {g_hcore_my / g_hcore_ref:.4f}x")

run_grad_match()
