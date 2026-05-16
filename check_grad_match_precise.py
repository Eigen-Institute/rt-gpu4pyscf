import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import solve_z_vector, make_intermediates, Hessian
from gpu4pyscf.hessian import rhf as rhf_hess
from functools import reduce

def run_grad_match_precise():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    # Use unscaled amplitudes to match spatial energy components
    x_y = tuple([cp.asarray(v) for v in td.xy[0]])
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td); inter = make_intermediates(h_obj, x_y, z1)
    
    P_I_prime = inter['P_I_prime'] # trace magnitude 1.0 (one-spin diag -0.5, 0.5)
    W_I = inter['W_I']
    
    # Assemble Gradient using tdrhf.py logic but my intermediates
    # 1. Hcore
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    p0, p1 = aoslices[0][2:]
    h1ao_x = cp.zeros((3, mol.nao, mol.nao))
    h1ao_x[:, p0:p1] += h1[:, p0:p1]
    h1ao_x[:, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    with mol.with_rinv_at_nucleus(0):
        vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
    h1ao_x += vrinv * 2.0 # Discover: factor 2 for physical total operator derivative
    
    g_hcore = float(cp.trace(P_I_prime @ h1ao_x[2]))
    print(f"DEBUG: h1 basis part norm: {float(cp.linalg.norm(h1[2])):.6f}")
    print(f"DEBUG: vrinv part norm:    {float(cp.linalg.norm(vrinv[2])):.6f}")
    print(f"H-core Part (My): {g_hcore:.8f}")
    
    # Reference H-core from grad engine
    nmo = mf.mo_energy.size; nocc = int(mf.mo_occ.sum() // 2); nvir = nmo - nocc
    x, y = x_y; xpy = (x + y).reshape(nocc, nvir).T; xmy = (x - y).reshape(nocc, nvir).T
    orbv = cp.asarray(mf.mo_coeff[:, nocc:]); orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T); doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) + cp.dot(orbv, cp.dot(dvv, orbv.T))
    z1ao = cp.dot(orbv, cp.dot(z1, orbo.T))
    dmz1doo_ref = z1ao + z1ao.T + dmzoo # total difference density
    
    from gpu4pyscf.df import int3c2e
    g_h_ref = float(cp.trace(dmz1doo_ref @ h1[2])) + float(int3c2e.get_dh1e(mol, dmz1doo_ref)[0,2])
    print(f"H-core Part (Ref): {g_h_ref:.8f}")
    print(f"Ratio: {g_hcore / g_h_ref:.4f}x")

run_grad_match_precise()
