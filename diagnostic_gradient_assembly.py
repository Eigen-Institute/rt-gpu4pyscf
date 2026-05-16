import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, make_intermediates, solve_z_vector
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def run_gradient_diagnostic():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    state = 0
    # SCALE AMPLITUDES to Liu-Liang convention
    x_y_orig = td.xy[state]
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td)
    inter = make_intermediates(h_obj, x_y, z1)
    
    P_I_prime = inter['P_I_prime']
    
    # Check dmzoo from grad engine
    nmo = mf.mo_coeff.shape[1]
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = nmo - nocc
    x, y = x_y_orig
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = cp.asarray(mf.mo_coeff[:, nocc:])
    orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    x, y = [cp.asarray(v) for v in x_y_orig]
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T)
    doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) * 2.0
    dmzoo += cp.dot(orbv, cp.dot(dvv, orbv.T)) * 2.0
    
    print(f"P_I_prime norm: {float(cp.linalg.norm(P_I_prime)):.6e}")
    print(f"dmzoo norm:     {float(cp.linalg.norm(dmzoo)):.6e}")
    
    # 2. Assemble gradient from intermediates
    # g^x = Tr(P_I' H^x) + Tr(Gamma_I' Pi^x) + Tr(W_I S^x)
    
    # H^x term
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    nao = mol.nao
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        # Nuclear attraction operator derivative
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3))
            vrinv *= -mol.atom_charge(atm_id)
        # Fix: operator derivative is already symmetric, don't double it
        h1ao_x[atm_id] = vrinv
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)

        print(f"\nDEBUG Atom 0, Direction X (0):")
        print(f"  h1 part norm: {float(cp.linalg.norm(h1[0])):.6e}")
        print(f"  vrinv part norm: {float(cp.linalg.norm(vrinv[0])):.6e}")
        print(f"  s1ao_x norm: {float(cp.linalg.norm(s1ao_x[0,0])):.6e}")
    g_hcore = cp.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            g_hcore[i,j] = cp.trace(P_I_prime @ h1ao_x[i,j])
    
    # S^x term
    s1 = cp.asarray(-mol.intor('int1e_ovlp', comp=3))
    s1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao_x[atm_id, :, p0:p1] += s1[:, p0:p1]
        s1ao_x[atm_id, :, :, p0:p1] += s1[:, p0:p1].transpose(0, 2, 1)
        
    g_ovlp = cp.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            # factor of 2 because W_I is for one spin? 
            # In grad/tdrhf.py, wvo is multiplied by 2.
            # In rhf_grad, it's Tr(W S^x) * 2? No, for RHF it's 2 * Tr(W_alpha S^x).
            g_ovlp[i,j] = cp.trace(W_I @ s1ao_x[i,j]) * 2.0
            
    # JK term (integral derivatives part)
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_PI, vk_PI = _get_jk_ip1(mol, P_I_prime)
    vj_PI = vj_PI.reshape(natm, 3, nao, nao)
    vk_PI = vk_PI.reshape(natm, 3, nao, nao)
    g_jk = (vj_PI * 2 - vk_PI) * 0.0 # Placeholder for correct contraction with P_total?
    # Actually, Tr(P_I' G^x(P)) = Tr(P G^x(P_I'))
    vj_P, vk_P = _get_jk_ip1(mol, inter['P'])
    vj_P = vj_P.reshape(natm, 3, nao, nao)
    vk_P = vk_P.reshape(natm, 3, nao, nao)
    G_P_x = vj_P - 0.5 * vk_P
    
    g_jk = cp.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            g_jk[i,j] = cp.trace(P_I_prime @ G_P_x[i,j])
            
    # Interaction integral part: Tr(X V^x(X))
    vj_R, vk_R = _get_jk_ip1(mol, R_I + R_I.T)
    vj_R = vj_R.reshape(natm, 3, nao, nao)
    vk_R = vk_R.reshape(natm, 3, nao, nao)
    # Gp_R_x = vj_R - 0.5 * vk_R?
    for i in range(natm):
        for j in range(3):
            # Prefactor 0.25 as in Hessian ejk_PI?
            g_jk[i,j] += 0.25 * cp.trace((R_I + R_I.T) @ (vj_R[i,j] - 0.5 * vk_R[i,j]))
            
    # T term
    _, vk_T = _get_jk_ip1(mol, T_I - T_I.T)
    vk_T = vk_T.reshape(natm, 3, nao, nao)
    for i in range(natm):
        for j in range(3):
            g_jk[i,j] -= 0.25 * cp.trace((T_I - T_I.T) @ (-0.5 * vk_T[i,j]))
            
    print(f"Assembled Gradient (elec):\n{g_hcore + g_ovlp + g_jk}")
    print(f"  H-core: {g_hcore[0,2]:.6f}")
    print(f"  Overlap: {g_ovlp[0,2]:.6f}")
    print(f"  JK: {g_jk[0,2]:.6f}")

run_gradient_diagnostic()
