import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
import os

def run_diagnose():
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'sto-3g'
    mol.verbose = 0
    mol.build()

    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()

    state = 0
    hess_obj = tdrhf.Hessian(td)
    
    # Manually compute e1_perturbed components
    x_y_orig = td.xy[state]
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf.solve_z_vector(td_grad_obj, x_y)
    print(f"Norm of z1: {cp.linalg.norm(z1)}")
    
    # Get all intermediates
    inter = tdrhf.make_intermediates(hess_obj, x_y, z1)
    
    # Get ground state MO responses
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    
    from gpu4pyscf.hessian import rhf as rhf_hess_gpu
    from gpu4pyscf.grad import rhf as grad_rhf
    mf_grad = grad_rhf.Gradients(mf)
    _, _, s1a_basis = rhf_hess_gpu.get_ovlp(mol)
    s1a_basis = cp.asarray(s1a_basis)
    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()
    
    s1ao = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
            
    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
                
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)
    
    Ux = cp.zeros((natm, 3, nao, nao))
    Ux[:,:,:,:nocc] = mo1
    Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
    Ux[:,:,:nocc,:nocc] = -0.5 * s1mo[:,:,:nocc,:nocc]
    Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]

    # CP-TDDFT responses
    x1, y1 = tdrhf.solve_cptddft(hess_obj, x_y, td.e[state], mo1, mo_e1)
    
    # Perturbed Intermediates
    perturbed_inter = tdrhf.make_perturbed_intermediates(hess_obj, inter, x_y, x1, y1, Ux, z1)
    
    # Integral Derivatives F_x_MO and G_x_PI_MO
    # (Copied from tdrhf.py)
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    from gpu4pyscf.df import int3c2e
    from gpu4pyscf.lib.cupy_helper import contract
    
    h1ao_x_eval = cp.zeros((natm, 3, nao, nao))
    h1_eval = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        h1ao_x_eval[atm_id, :, p0:p1] += h1_eval[:, p0:p1]
        h1ao_x_eval[atm_id, :, :, p0:p1] += h1_eval[:, p0:p1].transpose(0, 2, 1)
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
    eye_ao = cp.eye(nao)
    eye_ao_sorted = intopt.sort_orbitals(eye_ao, axis=[0])
    dh1e_ao = cp.zeros([natm, 3, nao, nao])
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e_ao[k0:k1, :, j0:j1, :] += contract('xkji,io->kxjo', int3c_blk, eye_ao_sorted[i0:i1])
        dh1e_ao[k0:k1, :, i0:i1, :] += contract('xkji,jo->kxio', int3c_blk, eye_ao_sorted[j0:j1])
    dh1e_ao = contract('kxjo,k->kxjo', dh1e_ao, -charges)
    P_sort = cp.asarray(intopt.sort_orbitals(np.eye(nao), axis=[0]))
    h1ao_x_eval += contract('pj,kxjo->kxpo', P_sort.T, dh1e_ao)
    
    orbo = mo_coeff[:, :nocc]
    dm0_eval = orbo @ orbo.T * 2
    vj_x_eval, vk_x_eval = _get_jk_ip1(mol, dm0_eval)
    vj_x_eval = vj_x_eval.reshape(natm, 3, nao, nao)
    vk_x_eval = vk_x_eval.reshape(natm, 3, nao, nao)
    G_px_D0 = vj_x_eval + vj_x_eval.transpose(0,1,3,2) - 0.5 * (vk_x_eval + vk_x_eval.transpose(0,1,3,2))
    
    Co_x_eval = cp.zeros((natm, 3, nao, nocc))
    for i in range(natm):
        for j in range(3):
            Co_x_eval[i, j] = mo_coeff @ Ux[i, j, :, :nocc]
    Dx_eval = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            tmp_e = Co_x_eval[i, j] @ orbo.T
            Dx_eval[i, j] = 2 * (tmp_e + tmp_e.T)
    vj_Dx, vk_Dx = mf.get_jk(mol, Dx_eval.reshape(-1, nao, nao))
    vj_Dx = vj_Dx.reshape(natm, 3, nao, nao)
    vk_Dx = vk_Dx.reshape(natm, 3, nao, nao)
    G_p_Dx = vj_Dx - 0.5 * vk_Dx
    F_x_AO = h1ao_x_eval + G_px_D0 + G_p_Dx
    F_x_MO = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff
            
    vj_PI_x, vk_PI_x = _get_jk_ip1(mol, inter['P_I_prime'] + inter['P_I_prime'].T)
    vj_PI_x = vj_PI_x.reshape(natm, 3, nao, nao)
    vk_PI_x = vk_PI_x.reshape(natm, 3, nao, nao)
    G_x_PI = vj_PI_x + vj_PI_x.transpose(0,1,3,2) - 0.5 * (vk_PI_x + vk_PI_x.transpose(0,1,3,2))
    G_x_PI_MO = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            G_x_PI_MO[i, j] = mo_coeff.T @ G_x_PI[i, j] @ mo_coeff

    # Components
    L_I_prime_y_MO = perturbed_inter['L_I_prime_y_MO']
    W_I_y_MO = perturbed_inter['W_I_y_MO']
    P_I_prime_y_MO = perturbed_inter['P_I_prime_y_MO']
    P_y_MO = perturbed_inter['P_y_MO']
    
    Z_MO = inter['Z_MO']
    P_z1_cross_MO = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            P_z1_cross_MO[i, j] = Ux[i, j] @ Z_MO + Z_MO @ Ux[i, j].T
            
    # G_px_D0_MO
    G_px_D0_MO = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            G_px_D0_MO[i, j] = mo_coeff.T @ G_px_D0[i, j] @ mo_coeff

    e1_term1 = cp.zeros((natm, natm, 3, 3))
    e1_term2 = cp.zeros((natm, natm, 3, 3))
    e1_term3 = cp.zeros((natm, natm, 3, 3))
    e1_term4 = cp.zeros((natm, natm, 3, 3))
    e1_term5 = cp.zeros((natm, natm, 3, 3))

    for i0 in range(natm):
        for j0 in range(natm):
            for x in range(3):
                for y in range(3):
                    tmp_U_S = 2 * Ux[i0, x] + s1mo[i0, x]
                    e1_term1[i0, j0, x, y] = cp.trace(L_I_prime_y_MO[j0, y] @ tmp_U_S)
                    e1_term2[i0, j0, x, y] = cp.trace(W_I_y_MO[j0, y] @ s1mo[i0, x])
                    e1_term3[i0, j0, x, y] = cp.trace(P_I_prime_y_MO[j0, y] @ F_x_MO[i0, x])
                    e1_term4[i0, j0, x, y] = cp.trace(P_y_MO[j0, y] @ G_x_PI_MO[i0, x])
                    e1_term5[i0, j0, x, y] = cp.trace(P_z1_cross_MO[j0, y] @ G_px_D0_MO[i0, x])

    def check_ti(mat, name):
        ti = cp.max(cp.abs(cp.sum(mat, axis=0))) # Sum over atoms i0
        print(f"TI error for {name}: {ti}")

    print("\n--- Individual Component TI Errors (Sum over i0) ---")
    check_ti(e1_term1, "term1 (L^y @ (2U^x+S^x))")
    check_ti(e1_term2, "term2 (W^y @ S^x)")
    check_ti(e1_term3, "term3 (P_I'^y @ F^x)")
    check_ti(e1_term4, "term4 (P^y @ G_PI^x)")
    check_ti(e1_term5, "term5 (P_z1_cross^y @ G_D0^x)")
    
    # Let's just look at the values for (0, 1, 2, 2)
    print("\n--- Component Values (atom 0, atom 1, Z, Z) ---")
    v1 = e1_term1[0,1,2,2]
    v2 = e1_term2[0,1,2,2]
    v3 = e1_term3[0,1,2,2]
    v4 = e1_term4[0,1,2,2]
    v5 = e1_term5[0,1,2,2]
    print(f"term1: {v1}")
    print(f"term2: {v2}")
    print(f"term3: {v3}")
    print(f"term4: {v4}")
    print(f"term5: {v5}")
    
    print("\n--- Combinations ---")
    print(f"2 + 3: {v2 + v3}")
    print(f"2 + 4: {v2 + v4}")
    print(f"2 + 3 + 4: {v2 + v3 + v4}")
    print(f"3 + 4: {v3 + v4}")
    print(f"0.5 * (2 + 3 + 4): {0.5 * (v2 + v3 + v4)}")
    
    print(f"2 + 3 + 5: {v2 + v3 + v5}")
    print(f"2 + 4 + 5: {v2 + v4 + v5}")
    print(f"2 + 3 + 4 + 5: {v2 + v3 + v4 + v5}")
    print(f"3 + 4 + 5: {v3 + v4 + v5}")
    print(f"0.5 * (2 + 3 + 4 + 5): {0.5 * (v2 + v3 + v4 + v5)}")
    
    print(f"3 + 5: {v3 + v5}")
    print(f"4 + 5: {v4 + v5}")
    print(f"2 + 5: {v2 + v5}")

if __name__ == "__main__":
    run_diagnose()
