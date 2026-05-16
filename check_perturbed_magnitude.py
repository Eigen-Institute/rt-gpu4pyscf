import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def check_perturbed():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.kernel()
    
    # 1. Analytical Perturbed part
    h_obj = tdrhf.Hessian(td)
    # We use the internal logic to get 'perturbed'
    x_y = td.xy[0]
    nocc = int((mf.mo_occ > 0).sum()); nvir = mf.mo_coeff.shape[1] - nocc
    mo_coeff = cp.asarray(mf.mo_coeff); mo_energy = cp.asarray(mf.mo_energy); mo_occ = cp.asarray(mf.mo_occ)
    
    from gpu4pyscf.hessian import rhf as rhf_hess_gpu
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, mf_hess.make_h1(mo_coeff, mo_occ), mf_hess.gen_vind(mo_coeff, mo_occ))
    mo1 = cp.asarray(h1mo)
    
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf.solve_z_vector(td_grad_obj, x_y)
    x1, y1 = tdrhf.solve_cptddft(h_obj, x_y, td.e[0], mo1, mo_e1)
    
    intermediates = tdrhf.make_intermediates(h_obj, x_y, z1)
    # This call calculates e1_perturbed
    # res = h_obj.analytical_omega_hessian(0)
    # Wait, I want to print the components from make_perturbed_intermediates
    from gpu4pyscf.grad import rhf as grad_rhf
    mf_grad = grad_rhf.Gradients(mf)
    s1a_basis = cp.asarray(mf_grad.get_ovlp(mol))
    natm = mol.natm; nao = mol.nao; aoslices = mol.aoslice_by_atom()
    s1ao = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
    Ux = cp.zeros((natm, 3, nao, nao))
    Ux[:,:,:,:nocc] = mo1
    Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
    Ux[:,:,:nocc,:nocc] = -0.5 * s1mo[:,:,:nocc,:nocc]
    Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]
    
    perturbed_intermediates = tdrhf.make_perturbed_intermediates(h_obj, intermediates, x_y, x1, y1, Ux, z1)
    
    F_AO = cp.asarray(mf.get_fock())
    F_x_AO = perturbed_intermediates['F_x_AO']
    F_x_MO = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff
            
    P_I_prime_y_MO = perturbed_intermediates['P_I_prime_y_MO']
    L_I_prime_y_MO = perturbed_intermediates['L_I_prime_y_MO']
    W_I_y_MO = perturbed_intermediates['W_I_y_MO']
    P_y_MO = perturbed_intermediates['P_y_MO']
    
    # Analytical term3: Tr(P_I^y F^x)
    idx = (0, 1, 2, 2)
    term3_ana = cp.trace(P_I_prime_y_MO[1, 2] @ F_x_MO[0, 2])
    
    # 2. FD derivative: d/dB Tr(P_I' F^A) where F^A is analytical gradient operator
    delta = 0.001
    gs = []
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        mf_p = gpu_scf.RHF(mol_p).run()
        td_p = gpu_tdscf.rhf.TDA(mf_p); td_p.kernel()
        
        # P_I' at perturbed geometry
        x_y_p = td_p.xy[0]
        td_grad_p = tdrhf_grad.Gradients(td_p)
        z1_p = tdrhf.solve_z_vector(td_grad_p, x_y_p)
        # intermediates_p = tdrhf.make_intermediates(h_obj, x_y_p, z1_p)
        # P_I_prime_p = intermediates_p['P_I_prime']
        
        # Actually, let's differencing the GRADIENT component Tr(P_I' F^x)
        # F^x at ORIGINAL geometry
        h1ao_x_ref = tdrhf._get_h1ao_x(mol) # atom 0, Z
        f_0_z = h1ao_x_ref[0, 2] # This is not full F^x
        # We need F^x = H^x + G^x[P]
        dm0 = cp.asarray(mf.make_rdm1())
        from gpu4pyscf.hessian.rhf import _get_jk_ip1
        # vj_x, vk_x = _get_jk_ip1(mol, dm0) ...
        # Use existing F_x_AO from analytical calc
        f_0_z_ao = F_x_AO[0, 2]
        
        # Tr(F^x_0 P_I'(B))
        # P_I' at perturbed B
        nocc_p = int((mf_p.mo_occ > 0).sum()); nvir_p = mf_p.mo_coeff.shape[1] - nocc_p
        xpy_p = cp.asarray(x_y_p[0]).reshape(nocc_p, nvir_p)
        P_I_MO_p = cp.zeros((mf_p.mo_coeff.shape[1], mf_p.mo_coeff.shape[1]))
        P_I_MO_p[nocc_p:, nocc_p:] = xpy_p.T @ xpy_p; P_I_MO_p[:nocc_p, :nocc_p] = -xpy_p @ xpy_p.T
        P_z_MO_p = cp.zeros_like(P_I_MO_p); P_z_MO_p[nocc_p:, :nocc_p] = z1_p; P_z_MO_p[:nocc_p, nocc_p:] = z1_p.T
        P_I_prime_p = cp.asarray(mf_p.mo_coeff) @ (P_I_MO_p + 0.5 * P_z_MO_p) @ cp.asarray(mf_p.mo_coeff).T
        
        gs.append(contract('pq,pq->', f_0_z_ao, P_I_prime_p))
        
    term3_fd = (gs[0] - gs[1]) / (2.0 * delta)
    
    print("--- Perturbed Component Verification (atom 0, atom 1, Z, Z) ---")
    print(f"Analytical Term3 Tr(P_I^y F^x): {term3_ana:.6f}")
    print(f"FD Term3 d/dB Tr(P_I' F^A):     {term3_fd:.6f}")
    print(f"Ratio:                          {term3_ana / term3_fd:.6f}")

if __name__ == "__main__":
    check_perturbed()
