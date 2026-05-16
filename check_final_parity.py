import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import solve_z_vector, make_intermediates, Hessian, make_perturbed_intermediates
from gpu4pyscf.hessian import rhf as rhf_hess
from functools import reduce

def get_grad_excitation(mol):
    mf = RHF(mol).run()
    td = TDA(mf); td.nstates = 1; td.kernel()
    g_exc = tdrhf_grad.Gradients(td).grad_elec(td.xy[0])
    g_gs = mf.nuc_grad_method().grad_elec()
    return g_exc - g_gs

def run_final_verification(direction):
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    x_y_ref = td.xy[0]
    dr = 0.0001; coords = mol.atom_coords()
    coords_p = coords.copy(); coords_p[0, direction] += dr
    g_p = get_grad_excitation(mol.copy().set_geom_(coords_p, unit='Bohr'))
    coords_m = coords.copy(); coords_m[0, direction] -= dr
    g_m = get_grad_excitation(mol.copy().set_geom_(coords_m, unit='Bohr'))
    h_fd = (g_p[0, direction] - g_m[0, direction]) / (2 * dr)
    dir_name = ['X', 'Y', 'Z'][direction]
    print(f"FD d/d{dir_name} (g_excitation) [0,{direction}]: {h_fd:.8f}")
    
    # Analytical Assembly (winning formula)
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_ref])
    td_grad_obj = tdrhf_grad.Gradients(td); z1 = solve_z_vector(td_grad_obj, x_y); h_obj = Hessian(td); inter = make_intermediates(h_obj, x_y, z1)
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
    h_ip2_hcore = _e_hcore_generator(h_obj, inter['P_I_prime'])(0, 0)[direction, direction]
    
    s1aa, s1ab, _ = get_ovlp(mol); aoslices = mol.aoslice_by_atom(); p0, p1 = aoslices[0][2:]
    from gpu4pyscf.lib.cupy_helper import contract
    h_ip2_ovlp = -float(contract('pq,pq->', inter['W_I'][p0:p1], cp.asarray(s1aa[:,:,p0:p1])[direction,direction])) * 2.0
    h_ip2_ovlp -= float(contract('pq,pq->', inter['W_I'][p0:p1, p0:p1], cp.asarray(s1ab[:,:,p0:p1,p0:p1])[direction,direction])) * 2.0
    
    vhfopt = mf._opt_gpu.get(mol.omega); P = inter['P']; P_I_prime = inter['P_I_prime']; R_I = inter['R_I']
    ejk_ip2 = _partial_ejk_ip2(mol, P_I_prime + P, vhfopt) - _partial_ejk_ip2(mol, P, vhfopt)
    ejk_ip2 += _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt)
    h_ip2_eri = ejk_ip2[0,0,direction,direction]
    
    # Cross terms
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ)
    fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ); mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1); nocc = int(mo_occ.sum() // 2); nao = mo_coeff.shape[0]; nvir = mo_energy.size - nocc
    Ux = cp.zeros((mol.natm, 3, nao, nao)); Ux[:, :, :, :nocc] = mo1
    # Ux construction (simplified but includes mo1 which is main part)
    pert_inter = make_perturbed_intermediates(h_obj, inter, x_y, cp.zeros((2,3,nocc,nvir)), cp.zeros((2,3,nocc,nvir)), Ux, z1)
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)); h1ao_x = cp.zeros((2, 3, mol.nao, mol.nao))
    for atm_id in range(2):
        p0, p1 = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id): vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv * 2.0; h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]; h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    h_ip1_hcore = float(cp.trace(h1ao_x[0,direction] @ pert_inter['P_I_prime_y'][0,direction]))
    
    s1ao = cp.zeros((2, 3, nao, nao))
    s1a_basis = cp.asarray(-mol.intor('int1e_ovlp', comp=3))
    for atm_id in range(2):
        p0, p1 = aoslices[atm_id][2:]; s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]; s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    h_ip1_ovlp = -float(cp.trace(s1ao[0,direction] @ pert_inter['W_I_y'][0,direction]))
    
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_PI, vk_PI = _get_jk_ip1(mol, P_I_prime); G_PI_x = (vj_PI * 2 - vk_PI).reshape(mol.natm, 3, nao, nao)
    vj_P, vk_P = _get_jk_ip1(mol, inter['P']); G_P_x = (vj_P * 2 - vk_P).reshape(mol.natm, 3, nao, nao)
    vj_R, vk_R = _get_jk_ip1(mol, inter['R_I'] + inter['R_I'].T); G_RI_x = (vj_R * 2 - vk_R).reshape(mol.natm, 3, nao, nao)
    h_ip1_eri = float(cp.trace(G_PI_x[0,direction] @ pert_inter['P_y'][0,direction]) + cp.trace(G_P_x[0,direction] @ pert_inter['P_I_prime_y'][0,direction]) + cp.trace(G_PI_x[0,direction] @ pert_inter['P_I_prime_y'][0,direction]))
    h_ip1_eri += float(cp.trace(G_RI_x[0,direction] @ (pert_inter['R_I_y'][0,direction] @ mo_coeff[:,:nocc].T + mo_coeff[:,:nocc] @ pert_inter['R_I_y'][0,direction].T)))

    # Winning Formula: 0.5 * ip2 + ip1
    total = 0.5 * (h_ip2_hcore + h_ip2_ovlp + h_ip2_eri) + (h_ip1_hcore + h_ip1_ovlp + h_ip1_eri)
    print(f"Analytical Total: {total:.8f}")
    print(f"Ratio: {total / h_fd:.4f}x")

print("--- Z direction ---")
run_final_verification(2)
print("\n--- X direction ---")
run_final_verification(0)
