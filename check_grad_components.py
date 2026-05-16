import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import solve_z_vector, make_intermediates, Hessian

def run_component_check():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    state = 0
    x_y_orig = td.xy[state]
    
    # 1. Grad Engine Components
    td_grad_obj = tdrhf_grad.Gradients(td)
    g_ref = td_grad_obj.grad_elec(x_y_orig)[0,2]
    
    # Extract components from grad engine logic
    nmo = mf.mo_coeff.shape[1]
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = nmo - nocc
    x, y = x_y_orig
    xpy = cp.asarray((x + y).reshape(nocc, nvir).T)
    xmy = cp.asarray((x - y).reshape(nocc, nvir).T)
    orbv = cp.asarray(mf.mo_coeff[:, nocc:])
    orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T)
    doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) * 2.0
    dmzoo += cp.dot(orbv, cp.dot(dvv, orbv.T)) * 2.0
    
    # dmz1doo from grad engine
    z1 = solve_z_vector(td_grad_obj, x_y_orig)
    z1ao = cp.dot(orbv, cp.dot(z1, orbo.T))
    dmz1doo = dmzoo + z1ao + z1ao.T
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    s1 = cp.asarray(-mol.intor('int1e_ovlp', comp=3))
    
    # h-core part (unsymmetrized)
    # dh_td from grad engine
    dh_td = float(cp.trace(dmz1doo @ h1[2]))
    # No operator part? Wait, dmz1doo is just doo + z1ao.
    # In grad/tdrhf.py, there is dh1e which is int3c2e.get_dh1e(mol, dmz1doo)
    from gpu4pyscf.df import int3c2e
    dh1e_td = int3c2e.get_dh1e(mol, dmz1doo)
    dh_td += float(dh1e_td[0,2])
    print(f"dh_td (Ref):      {dh_td:.8f}")
    
    # 2. My Assembly Components
    # Scaling factor check: do we scale or not?
    # Case A: Scaled amplitudes sqrt(2)
    x_y_scaled = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
    h_obj = Hessian(td)
    # P_I_prime = orbv @ X.T @ X @ orbv.T ...
    # Wait, my P_I_prime in make_intermediates used xpy = X, not X+Y.
    # In RHF, it should be the same for TDA.
    inter = make_intermediates(h_obj, x_y_scaled, cp.zeros((nvir, nocc)))
    P_I_prime = inter['P_I_prime']
    
    natm = mol.natm
    nao = mol.nao
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
        
    g_h_my = float(cp.trace(P_I_prime @ h1ao_x[0,2]))
    print(f"H-core (My, Scaled): {g_h_my:.8f}")
    
    # Case B: Unscaled amplitudes
    inter_un = make_intermediates(h_obj, [cp.asarray(v) for v in x_y_orig], cp.zeros((nvir, nocc)))
    P_I_prime_un = inter_un['P_I_prime']
    g_h_my_un = float(cp.trace(P_I_prime_un @ h1ao_x[0,2]))
    print(f"H-core (My, Unscaled): {g_h_my_un:.8f}")

run_component_check()
