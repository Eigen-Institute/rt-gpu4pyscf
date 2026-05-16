import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def brute_force():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    h_obj = tdrhf_hess.Hessian(td)
    h_semi = h_obj.kernel()
    target = float(h_semi[0,2,0,2])
    
    x_y = td.xy[0]
    omega = td.e[0]
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    from gpu4pyscf.hessian import rhf as rhf_hess_gpu
    mf_hess = rhf_hess_gpu.Hessian(mf); h1mo, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, mf_hess.make_h1(mo_coeff, mo_occ), mf_hess.gen_vind(mo_coeff, mo_occ)); mo1 = cp.asarray(h1mo)
    from gpu4pyscf.grad import rhf as grad_rhf; mf_grad = grad_rhf.Gradients(mf); res_ovlp = mf_grad.get_ovlp(mol); s1a_basis = cp.asarray(res_ovlp[-1] if isinstance(res_ovlp, tuple) else res_ovlp)
    natm = mol.natm; nao = mol.nao; nmo = mo_coeff.shape[1]; nocc = int((mo_occ > 0).sum()); nvir = nmo - nocc; orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    s1ao = cp.zeros((natm, 3, nao, nao)); aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm): p0, p1 = aoslices[atm_id][2:]; s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]; s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3): s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
    Ux = cp.zeros((natm, 3, nmo, nmo)); Ux[:,:,:,:nocc] = mo1; Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2); Ux[:,:,:nocc,:nocc] = -0.5 * s1mo[:,:,:nocc,:nocc]; Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y); x1, y1 = tdrhf_hess.solve_cptddft(h_obj, x_y, omega, mo1, mo_e1)
    inter = tdrhf_hess.make_intermediates(h_obj, x_y, z1); pert_inter = tdrhf_hess.make_perturbed_intermediates(h_obj, inter, x_y, x1, y1, Ux, z1)
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
    hcore = float(_e_hcore_generator(h_obj, inter['P_I_prime'])(0,0)[2,2])
    vhfopt = mf._opt_gpu.get(mol.omega)
    
    h2 = cp.asarray(mol.intor('int1e_ipipkin', comp=9) + mol.intor('int1e_ipipnuc', comp=9)).reshape(3,3,mol.nao,mol.nao)
    # Total Hcore Static = Sum_{mu,nu} P_mu,nu * H''_mu,nu
    # H''_mu,nu = <mu''|h|nu> + <mu|h|nu''> + 2<mu'|h|nu'> + <mu|h''|nu>
    # The first 3 terms are basis function derivatives (Pulay).
    # The last term is operator derivative (HF).
    
    # HF part
    with mol.with_rinv_at_nucleus(0):
        ipiprinv = cp.asarray(mol.intor('int1e_ipiprinv', comp=9)).reshape(3,3,mol.nao,mol.nao)
        hf_curv = contract('xypq,pq->xy', ipiprinv, inter['P_I_prime']) * -mol.atom_charge(0)
    
    # Pulay part
    # int1e_ipipkin returns <mu''|T|nu>. 
    # Total T'' = <mu''|T|nu> + <mu|T|nu''> + 2<mu'|T|nu'>
    h1aa_kin = cp.asarray(mol.intor('int1e_ipipkin', comp=9)).reshape(3,3,mol.nao,mol.nao)
    h1ab_kin = cp.asarray(mol.intor('int1e_ipkinip', comp=9)).reshape(3,3,mol.nao,mol.nao)
    # ... this is getting complicated.
    
    print(f"Hcore Static (Generator): {hcore:.6f}")
    print(f"HF Curvature (Manual):    {hf_curv[2,2]:.6f}")
    ejk_PI = _partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt) - _partial_ejk_ip2(mol, inter['P'], vhfopt)
    ejk_RI = _partial_ejk_ip2(mol, inter['R_I'] + inter['R_I'].T, vhfopt)
    ejk_TI = _partial_ejk_ip2(mol, inter['T_I'] - inter['T_I'].T, vhfopt, j_factor=0.0)
    
    s1aa, s1ab, _ = get_ovlp(mol); W_I = inter['W_I']; p0, p1 = aoslices[0][2:]
    ovlp = -float(contract('xypq,pq->xy', cp.asarray(s1aa[:,:,p0:p1]), W_I[p0:p1])[2,2]) * 2.0
    
    term1 = cp.trace(pert_inter['L_I_prime_y_MO'][0, 2] @ (2 * Ux[0, 2] + s1mo[0, 2]))
    term2 = cp.trace(pert_inter['W_I_y_MO'][0, 2] @ s1mo[0, 2])
    F_x_MO = mo_coeff.T @ pert_inter['F_x_AO'][0, 2] @ mo_coeff
    term3 = cp.trace(pert_inter['P_I_prime_y_MO'][0, 2] @ F_x_MO)
    G_x_PI_MO = mo_coeff.T @ pert_inter['G_x_PI_AO'][0, 2] @ mo_coeff
    term4 = cp.trace(pert_inter['P_y_MO'][0, 2] @ G_x_PI_MO)
    G_x_RI_MO = orbv.T @ pert_inter['Gp_x_RI_AO'][0, 2] @ orbo
    G_x_TI_MO = orbv.T @ pert_inter['Gm_x_TI_AO'][0, 2] @ orbo
    term5 = cp.sum(x1[0, 2] * G_x_RI_MO.T)
    term6 = cp.sum(y1[0, 2] * G_x_TI_MO.T)
    pert = float(term1 + term2 + term3 + term4 + term5 + term6)
    
    # ERI parts
    e_static = float(ejk_PI[0,0,2,2])
    e_ri = float(ejk_RI[0,0,2,2])
    e_ti = float(ejk_TI[0,0,2,2])

    print(f"Hcore: {hcore:.6f}")
    print(f"E_static: {e_static:.6f}")
    print(f"E_RI: {e_ri:.6f}")
    print(f"E_TI: {e_ti:.6f}")
    print(f"Ovlp: {ovlp:.6f}")
    print(f"Pert: {pert:.6f}")
    print(f"Target: {target:.6f}")
    
    # Try common multipliers
    for m in [0.5]:
        for m_ri in [0.25, 0.5, 1.0]:
            val = m * (hcore + e_static + m_ri*e_ri - m_ri*e_ti + ovlp + pert)
            print(f"m={m:4.2f}, m_ri={m_ri:4.2f}: {val:.6f} (Diff: {val-target:.6f})")


if __name__ == "__main__":
    brute_force()
