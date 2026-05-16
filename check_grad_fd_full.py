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

def run_gradient_fd_check():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    x_y_ref = td.xy[0]
    dr = 0.0001; coords = mol.atom_coords()
    
    # FD Z
    coords_pz = coords.copy(); coords_pz[0,2] += dr
    g_pz = get_grad_excitation(mol.copy().set_geom_(coords_pz, unit='Bohr'))
    coords_mz = coords.copy(); coords_mz[0,2] -= dr
    g_mz = get_grad_excitation(mol.copy().set_geom_(coords_mz, unit='Bohr'))
    h_fd_z = (g_pz[0,2] - g_mz[0,2]) / (2 * dr)
    print(f"FD d/dz (g_excitation) [0,2]: {h_fd_z:.8f}")
    
    # FD X
    coords_px = coords.copy(); coords_px[0,0] += dr
    g_px = get_grad_excitation(mol.copy().set_geom_(coords_px, unit='Bohr'))
    coords_mx = coords.copy(); coords_mx[0,0] -= dr
    g_mx = get_grad_excitation(mol.copy().set_geom_(coords_mx, unit='Bohr'))
    h_fd_x = (g_px[0,0] - g_mx[0,0]) / (2 * dr)
    print(f"FD d/dx (g_excitation) [0,0]: {h_fd_x:.8f}")
    
    # 2. Analytical Assembly
    x_y = tuple([cp.asarray(v) for v in x_y_ref])
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td)
    inter = make_intermediates(h_obj, x_y, z1)
    
    # Response
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ)
    fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ); mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1); nocc = int(mo_occ.sum() // 2); nao = mo_coeff.shape[0]; nvir = mo_energy.size - nocc
    
    from gpu4pyscf.hessian.rhf import get_ovlp
    s1aa, s1ab, s1a_basis = get_ovlp(mol); s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab); s1a_basis = cp.asarray(s1a_basis)
    aoslices = mol.aoslice_by_atom()
    s1ao = cp.zeros((mol.natm, 3, nao, nao))
    for atm_id in range(mol.natm):
        p0, p1 = aoslices[atm_id][2:]; s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]; s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((mol.natm, 3, nao, nao))
    for i in range(mol.natm):
        for j in range(3): s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
    Ux = cp.zeros((mol.natm, 3, nao, nao)); Ux[:, :, :, :nocc] = mo1
    Ux[:, :, :nocc, nocc:] = (-s1mo[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2))
    Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]
    
    from gpu4pyscf.hessian.tdrhf import solve_cptddft
    x1, y1 = solve_cptddft(h_obj, x_y, td.e[0], mo1, mo_e1)
    pert_inter = make_perturbed_intermediates(h_obj, inter, x_y, x1, y1, Ux, z1)
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)); h1ao_x = cp.zeros((2, 3, mol.nao, mol.nao))
    for atm_id in range(2):
        p0, p1 = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id): vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv * 2.0; h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]; h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)

    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, _get_jk_ip1
    vhfopt = mf._opt_gpu.get(mol.omega)
    P = inter['P']; P_I_prime = inter['P_I_prime']; R_I = inter['R_I']
    vj_PI, vk_PI = _get_jk_ip1(mol, P_I_prime); G_PI_x = (vj_PI * 2 - vk_PI).reshape(mol.natm, 3, nao, nao)
    vj_P, vk_P = _get_jk_ip1(mol, inter['P']); G_P_x = (vj_P * 2 - vk_P).reshape(mol.natm, 3, nao, nao)
    vj_R, vk_R = _get_jk_ip1(mol, inter['R_I'] + inter['R_I'].T); G_RI_x = (vj_R * 2 - vk_R).reshape(mol.natm, 3, nao, nao)
    from gpu4pyscf.lib.cupy_helper import contract

    def compute_anal(ia, ix, ja, jx):
        e1_hcore = _e_hcore_generator(h_obj, inter['P_I_prime'])(ia,ja)[ix,jx]
        p0, p1 = aoslices[ia][2:]; q0, q1 = aoslices[ja][2:]
        if ia == ja:
            e1_ovlp = -float(contract('pq,pq->', inter['W_I'][p0:p1], cp.asarray(s1aa[ix,jx,p0:p1]))) * 2.0
            e1_ovlp -= float(contract('pq,pq->', inter['W_I'][p0:p1, p0:p1], cp.asarray(s1ab[ix,jx,p0:p1,p0:p1]))) * 4.0
        else:
            e1_ovlp = -float(contract('pq,pq->', inter['W_I'][p0:p1, q0:q1], cp.asarray(s1ab[ix,jx,p0:p1,q0:q1]))) * 4.0
        ejk_ip2 = _partial_ejk_ip2(mol, P_I_prime + P, vhfopt) - _partial_ejk_ip2(mol, P, vhfopt)
        ejk_ip2 += _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt); e1_eri = ejk_ip2[ia,ja,ix,jx]
        ip1_hcore = float(cp.trace(h1ao_x[ia,ix] @ pert_inter['P_I_prime_y'][ja,jx]))
        ip1_ovlp = -float(cp.trace(s1ao[ia,ix] @ pert_inter['W_I_y'][ja,jx])) * 2.0
        ip1_eri = float(cp.trace(G_PI_x[ia,ix] @ pert_inter['P_y'][ja,jx]) + cp.trace(G_P_x[ia,ix] @ pert_inter['P_I_prime_y'][ja,jx]) + cp.trace(G_PI_x[ia,ix] @ pert_inter['P_I_prime_y'][ja,jx]))
        ip1_eri += float(cp.trace(G_RI_x[ia,ix] @ (pert_inter['R_I_y_ao'][ja,jx] + pert_inter['R_I_y_ao'][ja,jx].T)))
        return (e1_hcore + e1_ovlp + e1_eri + 2*(ip1_hcore + ip1_ovlp + ip1_eri))

    total_z = compute_anal(0,2,0,2)
    print(f"Analytical Z Total: {total_z:.8f} (Ratio: {total_z / h_fd_z:.4f}x)")
    total_x = compute_anal(0,0,0,0)
    print(f"Analytical X Total: {total_x:.8f} (Ratio: {total_x / h_fd_x:.4f}x)")

run_gradient_fd_check()
