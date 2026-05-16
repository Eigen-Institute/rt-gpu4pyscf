import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def diagnose_components():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    h_obj = tdrhf_hess.Hessian(td)
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
    omega = td.e[0]
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    
    # 1. MO1 and Ux
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    from gpu4pyscf.hessian import rhf as rhf_hess_gpu
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, 
                                  mf_hess.make_h1(mo_coeff, mo_occ), 
                                  mf_hess.gen_vind(mo_coeff, mo_occ))
    mo1 = cp.asarray(h1mo)
    
    # S1AO and S1MO
    from gpu4pyscf.grad import rhf as grad_rhf
    mf_grad = grad_rhf.Gradients(mf)
    res_ovlp = mf_grad.get_ovlp(mol)
    s1a_basis = cp.asarray(res_ovlp[-1] if isinstance(res_ovlp, tuple) else res_ovlp)
    natm = mol.natm; nao = mol.nao; nmo = mo_coeff.shape[1]
    s1ao = cp.zeros((natm, 3, nao, nao))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3): s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
        
    Ux = cp.zeros((natm, 3, nmo, nmo))
    nocc = int(mo_occ.sum()//2)
    Ux[:,:,:,:nocc] = mo1
    Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
    Ux[:,:,:nocc,:nocc] = -0.5 * s1mo[:,:,:nocc,:nocc]
    Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]
    
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y)
    x1, y1 = tdrhf_hess.solve_cptddft(h_obj, x_y, omega, mo1, mo_e1)
    
    inter = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
    pert_inter = tdrhf_hess.make_perturbed_intermediates(h_obj, inter, x_y, x1, y1, Ux, z1)
    
    # Components
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
    de_hcore = _e_hcore_generator(h_obj, inter['P_I_prime'])
    e1_hcore = de_hcore(0, 1)[2,2]
    
    vhfopt = mf._opt_gpu.get(mol.omega)
    ejk_PI_full = _partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt)
    ejk_PI_full -= _partial_ejk_ip2(mol, inter['P'], vhfopt)
    ejk_RI = _partial_ejk_ip2(mol, inter['R_I'] + inter['R_I'].T, vhfopt)
    ejk_TI = _partial_ejk_ip2(mol, inter['T_I'] - inter['T_I'].T, vhfopt, j_factor=0.0)
    ejk_PI = ejk_PI_full[0,1,2,2] + 0.5 * ejk_RI[0,1,2,2] - 0.5 * ejk_TI[0,1,2,2]
    
    s1aa, s1ab, _ = get_ovlp(mol)
    W_I = inter['W_I']
    p0, p1 = aoslices[0][2:]; q0, q1 = aoslices[1][2:]
    e1_ovlp = -contract('xypq,pq->xy', cp.asarray(s1ab[:,:,p0:p1,q0:q1]), W_I[p0:p1,q0:q1]) * 2
    e1_ovlp = e1_ovlp[2,2]
    
    # Perturbed
    term1 = cp.trace(pert_inter['L_I_prime_y_MO'][1, 2] @ (2 * Ux[0, 2] + s1mo[0, 2]))
    term2 = cp.trace(pert_inter['W_I_y_MO'][1, 2] @ s1mo[0, 2])
    F_x_MO = cp.zeros((nmo, nmo))
    F_x_MO[:] = mo_coeff.T @ pert_inter['F_x_AO'][0, 2] @ mo_coeff
    term3 = cp.trace(pert_inter['P_I_prime_y_MO'][1, 2] @ F_x_MO)
    
    G_x_PI_MO = mo_coeff.T @ pert_inter['G_x_PI_AO'][0, 2] @ mo_coeff
    term4 = cp.trace(pert_inter['P_y_MO'][1, 2] @ G_x_PI_MO)
    
    orbv = mo_coeff[:, nocc:]; orbo = mo_coeff[:, :nocc]
    G_x_RI_MO = orbv.T @ pert_inter['Gp_x_RI_AO'][0, 2] @ orbo
    G_x_TI_MO = orbv.T @ pert_inter['Gm_x_TI_AO'][0, 2] @ orbo
    term5 = cp.sum(x1[1, 2] * G_x_RI_MO.T)
    term6 = cp.sum(y1[1, 2] * G_x_TI_MO.T)
    
    e1_pert = term1 + term2 + term3 + term4 + term5 + term6
    
    print(f"Hcore:    {e1_hcore:.8f}")
    print(f"ERI:      {ejk_PI:.8f}")
    print(f"Overlap:  {e1_ovlp:.8f}")
    print(f"Perturbed: {e1_pert:.8f}")
    
    total = (e1_hcore + ejk_PI + e1_ovlp + e1_pert) * 2.0
    print(f"Total (x2): {total:.8f}")

diagnose_components()
