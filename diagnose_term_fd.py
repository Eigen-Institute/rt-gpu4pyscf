import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def diagnose_omega_grad():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.kernel()
    
    x_y = td.xy[0]
    nocc = int((mf.mo_occ > 0).sum()); nvir = mf.mo_coeff.shape[1] - nocc
    xpy = cp.asarray(x_y[0]).reshape(nocc, nvir)
    mo_coeff = cp.asarray(mf.mo_coeff)
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    
    # 1. P_I difference density
    P_I_MO = cp.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    P_I_MO[nocc:, nocc:] = xpy.T @ xpy
    P_I_MO[:nocc, :nocc] = -xpy @ xpy.T
    P_I = mo_coeff @ P_I_MO @ mo_coeff.T
    
    # 2. Transition densities
    R_I = orbo @ xpy @ orbv.T
    
    # Target: Analytical omega_grad
    g_total_ref = tdrhf.omega_grad(td, 0)
    
    # Component 1: Tr(H^x P_I)
    h1ao_x = tdrhf._get_h1ao_x(mol)
    g_hcore = contract('kxpq,pq->kx', h1ao_x, P_I)
    
    # Component 2: Tr(G^x[P_GS] P_I)
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vhfopt = mf._opt_gpu.get(mol.omega)
    dm0 = mf.make_rdm1()
    vj_x_raw, vk_x_raw = _get_jk_ip1(mol, dm0)
    # Unsort logic
    from gpu4pyscf.df.int3c2e import VHFOpt as VHFOpt3c
    intopt = VHFOpt3c(mol, gto.fakemol_for_charges(mol.atom_coords()), 'int2e')
    intopt.build(1e-14, aosym=False)
    P_inv = cp.argsort(cp.asarray(intopt._ao_idx))
    atm_inv = cp.argsort(cp.asarray(intopt._aux_ao_idx))
    vj_x = vj_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_x = vk_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_P_GS = vj_x - 0.5 * vk_x
    g_ejk_cross = contract('kxpq,pq->kx', G_x_P_GS, P_I)
    
    # Component 3: 0.5 Tr(G^x[R] R^T + ...)
    # For TDA R^T = R, and we use R+R.T
    vj_R_raw, vk_R_raw = _get_jk_ip1(mol, R_I + R_I.T)
    vj_R = vj_R_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_R = vk_R_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_RI = vj_R - 0.5 * vk_R
    g_ejk_ri = 0.5 * contract('kxpq,pq->kx', G_x_RI, R_I + R_I.T)

    # Component 4: -Tr(S^x W_I)
    # W_I = -0.5 (P_I F + F P_I)
    F_AO = cp.asarray(mf.get_fock())
    W_I = -0.5 * (P_I @ F_AO + F_AO @ P_I)
    from gpu4pyscf.grad import rhf as grad_rhf
    s1a = cp.asarray(grad_rhf.Gradients(mf).get_ovlp(mol))
    # s1a is (3, nao, nao)
    g_ovlp = cp.zeros((mol.natm, 3))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(mol.natm):
        p0, p1 = aoslices[atm_id][2:]
        g_ovlp[atm_id] = 2.0 * contract('xpq,pq->x', s1a[:, p0:p1], W_I[p0:p1])

    g_sum = g_hcore + g_ejk_cross + g_ejk_ri - g_ovlp
    
    print("--- Excitation Gradient Decomposition (atom 0, Z) ---")
    print(f"Ref Total:    {g_total_ref[0, 2]:.6f}")
    print(f"Manual Total: {g_sum[0, 2]:.6f}")
    print(f"  - Hcore:    {g_hcore[0, 2]:.6f}")
    print(f"  - ERI Cross: {g_ejk_cross[0, 2]:.6f}")
    print(f"  - ERI RI:    {g_ejk_ri[0, 2]:.6f}")
    print(f"  - Overlap:   {g_ovlp[0, 2]:.6f}")
    
    print("\nRatio (Manual/Ref):", g_sum[0,2]/g_total_ref[0,2])

if __name__ == "__main__":
    diagnose_omega_grad()
