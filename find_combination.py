import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def find_multipliers():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    h_obj = tdrhf_hess.Hessian(td)
    h_semi = h_obj.kernel()
    target = h_semi[0,2,0,2]
    print(f"Target: {target:.8f}")
    
    # Get components with unscaled amplitudes
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
    ejk = _partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt)
    ejk -= _partial_ejk_ip2(mol, inter['P'], vhfopt)
    ejk_static = float(ejk[0,0,2,2])
    ejk_RI = float(_partial_ejk_ip2(mol, inter['R_I'] + inter['R_I'].T, vhfopt)[0,0,2,2])
    
    s1aa, s1ab, _ = get_ovlp(mol); W_I = inter['W_I']; p0, p1 = aoslices[0][2:]
    ovlp = -float(contract('xypq,pq->xy', cp.asarray(s1aa[:,:,p0:p1]), W_I[p0:p1])[2,2]) * 2.0
    
    # Perturbed
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
    
    print(f"Hcore Static: {hcore:.8f}")
    print(f"ERI Static:   {ejk_static:.8f}")
    print(f"ERI RI:       {ejk_RI:.8f}")
    print(f"Overlap:      {ovlp:.8f}")
    print(f"Perturbed:    {pert:.8f}")
    
    # Try combinations
    print("\nTrying combinations (Hcore + m1*ERI_static + m2*ERI_RI + m3*Overlap + m4*Perturbed = target)")
    # We know Hcore is already doubled. 0.84 vs 0.21.
    # Wait, Hcore static FD was 0.84! So Hcore IS 1.0x.
    # Overlap static FD should be checked.
    
    res = hcore + ejk_static + 0.5*ejk_RI + ovlp + pert
    print(f"Sum (m=1): {res:.8f} (Ratio {res/target:.4f})")

if __name__ == "__main__":
    find_multipliers()
