import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates, solve_cptddft
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.grad.tdrhf import Gradients

def diagnose_manual_z():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    x_y = tuple([cp.asarray(v) for v in td.xy[0]])
    h_obj = Hessian(td)
    z1 = solve_z_vector(Gradients(td), x_y)
    inter = make_intermediates(h_obj, x_y, z1)
    P = inter['P_I_prime']
    W = inter['W_I']
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, get_ovlp, _partial_ejk_ip2, _get_jk_ip1
    
    # 1. Manual ip2 H-core
    de_hcore = _e_hcore_generator(h_obj, P)
    h_hcore = float(de_hcore(0,0)[2,2])
    
    # 2. Manual ip2 Overlap
    s1aa, s1ab, s1a_basis = get_ovlp(mol); s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab); s1a_basis = cp.asarray(s1a_basis)
    aoslices = mol.aoslice_by_atom(); p0, p1 = aoslices[0][2:]
    val_ovlp = 0.0
    for j in range(2):
        if j == 0: continue
        q0, q1 = aoslices[j][2:]
        val_ovlp -= float(contract('pq,pq->', W[p0:p1, q0:q1], s1aa[2,2,p0:p1, q0:q1])) * 2.0
    
    # 2.5 ERI ip2
    vhfopt = mf._opt_gpu.get(mol.omega)
    h_eri_raw = _partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt) - _partial_ejk_ip2(mol, inter['P'], vhfopt)
    h_eri_raw += _partial_ejk_ip2(mol, inter['R_I'] + inter['R_I'].T, vhfopt)
    val_eri = float(h_eri_raw[0,0,2,2]) * 0.5

    # 2.7 ip1 cross terms
    mo_energy = mf.mo_energy; mo_coeff = mf.mo_coeff; mo_occ = mf.mo_occ
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ)
    fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ); mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1); nocc = int(mo_occ.sum() // 2); nao = mo_coeff.shape[0]; nvir = mo_energy.size - nocc
    
    s1ao_z = cp.zeros((2, 3, nao, nao))
    for atm_id in range(2):
        p0c, p1c = aoslices[atm_id][2:]; s1ao_z[atm_id, :, p0c:p1c] += s1a_basis[:, p0c:p1c]; s1ao_z[atm_id, :, :, p0c:p1c] += s1a_basis[:, p0c:p1c].transpose(0, 2, 1)
    s1mo_z = cp.zeros((2, 3, nao, nao))
    for i in range(2):
        for j in range(3): s1mo_z[i, j] = cp.asarray(mo_coeff.T) @ s1ao_z[i, j] @ cp.asarray(mo_coeff)
    Ux = cp.zeros((2, 3, nao, nao)); Ux[:, :, :, :nocc] = mo1
    Ux[:, :, :nocc, nocc:] = (-s1mo_z[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2))
    Ux[:, :, nocc:, nocc:] = -0.5 * s1mo_z[:, :, nocc:, nocc:]
    
    x1, y1 = solve_cptddft(h_obj, x_y, td.e[0], mo1, mo_e1)
    pert_inter = make_perturbed_intermediates(h_obj, inter, x_y, x1, y1, Ux, z1)
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)); h1ao_z = cp.zeros((2, 3, nao, nao))
    for atm_id in range(2):
        p0c, p1c = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id): vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(atm_id)
        h1ao_z[atm_id] = vrinv * 2.0; h1ao_z[atm_id, :, p0c:p1c] += h1[:, p0c:p1c]; h1ao_z[atm_id, :, :, p0c:p1c] += h1[:, p0c:p1c].transpose(0, 2, 1)

    ip1_hcore = float(cp.trace(h1ao_z[0,2] @ pert_inter['P_I_prime_y'][0,2]))
    ip1_ovlp = -float(cp.trace(s1ao_z[0,2] @ pert_inter['W_I_y'][0,2])) * 2.0
    
    vj_PI, vk_PI = _get_jk_ip1(mol, P); G_PI_x = (vj_PI * 2 - vk_PI).reshape(2, 3, nao, nao)
    vj_P, vk_P = _get_jk_ip1(mol, inter['P']); G_P_x = (vj_P * 2 - vk_P).reshape(2, 3, nao, nao)
    vj_R, vk_R = _get_jk_ip1(mol, inter['R_I'] + inter['R_I'].T); G_RI_x = (vj_R * 2 - vk_R).reshape(2, 3, nao, nao)
    
    ip1_eri = float(cp.trace(G_PI_x[0,2] @ pert_inter['P_y'][0,2]) + cp.trace(G_P_x[0,2] @ pert_inter['P_I_prime_y'][0,2]) + cp.trace(G_PI_x[0,2] @ pert_inter['P_I_prime_y'][0,2]))
    ip1_eri += float(cp.trace(G_RI_x[0,2] @ (pert_inter['R_I_y_ao'][0,2] + pert_inter['R_I_y_ao'][0,2].T)))

    # 3. Semi-analytical reference
    dr = 0.0001; coords = mol.atom_coords()
    def get_g(c):
        m = mol.copy(); m.set_geom_(c, unit='Bohr'); m.build()
        mf_c = RHF(m).run(); td_c = TDA(mf_c); td_c.nstates = 1; td_c.kernel()
        g_e = Gradients(td_c).grad_elec(td_c.xy[0])
        return g_e - mf_c.nuc_grad_method().grad_elec()

    c_p = coords.copy(); c_p[0,2] += dr
    g_p = get_g(c_p)
    c_m = coords.copy(); c_m[0,2] -= dr
    g_m = get_g(c_m)
    h_fd = (g_p[0,2] - g_m[0,2]) / (2 * dr)
    
    print(f"Hcore ip2: {h_hcore/2.0:.8f}")
    print(f"Ovlp ip2:  {val_ovlp:.8f}")
    print(f"ERI ip2:   {val_eri:.8f}")
    print(f"ip1 total: {ip1_hcore + ip1_ovlp + ip1_eri:.8f}")
    
    print(f"ip1 hcore: {ip1_hcore:.8f}")
    print(f"ip1 ovlp:  {ip1_ovlp:.8f}")
    print(f"ip1 eri:   {ip1_eri:.8f}")
    
    total_anal = (h_hcore/2.0 + val_ovlp + val_eri) + (ip1_hcore + ip1_ovlp + ip1_eri)
    print(f"Total Analytical Z-Z [0,0]: {total_anal:.8f}")
    print(f"Semi-Analytical Z-Z:         {h_fd:.8f}")
    print(f"Ratio: {total_anal / h_fd:.4f}x")

diagnose_manual_z()
