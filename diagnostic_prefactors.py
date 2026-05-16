import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates, solve_cptddft
from gpu4pyscf.lib.cupy_helper import contract

def print_term_breakdown():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    # 1. Physical amplitudes (Liu-Liang)
    x_y_orig = td.xy[0]; x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
    omega = td.e[0]
    
    h_obj = Hessian(td)
    z1 = solve_z_vector(h_obj, x_y)
    inter = make_intermediates(h_obj, x_y, z1)
    
    # 2. MO1/Ux
    from gpu4pyscf.hessian import rhf as rhf_hess
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ); fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ)
    mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx); mo1 = cp.asarray(mo1)
    
    # 3. CP-TDDFT
    x1, y1 = solve_cptddft(h_obj, x_y, omega, mo1, mo_e1)
    
    # 4. Ux construction
    from gpu4pyscf.hessian.rhf import get_ovlp as get_ovlp_hess
    _, _, s1a_basis = get_ovlp_hess(mol); s1a_basis = cp.asarray(s1a_basis); nao = mol.nao; natm = mol.natm
    s1ao_basis = cp.zeros((natm, 3, nao, nao)); aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm): p0, p1 = aoslices[atm_id][2:]; s1ao_basis[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]; s1ao_basis[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3): s1mo[i, j] = mo_coeff.T @ s1ao_basis[i, j] @ mo_coeff
    nocc = int(mf.mo_occ.sum() // 2)
    Ux = cp.zeros((natm, 3, nao, nao)); Ux[:, :, :, :nocc] = mo1; Ux[:, :, :nocc, nocc:] = (-s1mo[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2)); Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]
    
    pert_inter = make_perturbed_intermediates(h_obj, inter, x_y, x1, y1, Ux, z1)
    
    # 5. Terms for [0,2,0,2] (Stretch)
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2
    h_hcore_ip2 = float(_e_hcore_generator(h_obj, inter['P_I_prime'])(0,0)[2,2])
    
    vhfopt = mf._opt_gpu.get(mol.omega)
    h_eri_ip2 = float((_partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt) - _partial_ejk_ip2(mol, inter['P'], vhfopt))[0,0,2,2])
    h_eri_transit = float(_partial_ejk_ip2(mol, inter['R_I'] + inter['R_I'].T, vhfopt)[0,0,2,2])
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)); h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id): vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv * 2.0; h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]; h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    
    h_hcore_ip1 = float(contract('pq,qp->', h1ao_x[0,2], pert_inter['P_I_prime_y'][0,2]))
    
    print("Term breakdown for H2 [0,2,0,2] (Stretch):")
    print(f"  hcore_ip2:   {h_hcore_ip2:.8f}")
    print(f"  eri_ip2:     {h_eri_ip2:.8f}")
    print(f"  eri_transit: {h_eri_transit:.8f}")
    print(f"  hcore_ip1:   {h_hcore_ip1:.8f}")
    
    # Combined with current weights
    total = 0.5 * h_hcore_ip2 + (h_eri_ip2 + 0.5 * h_eri_transit) + 2.0 * (0.5 * h_hcore_ip1)
    print(f"  Total (calc): {total:.8f}")

print_term_breakdown()
