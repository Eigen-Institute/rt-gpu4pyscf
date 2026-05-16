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
    return g_exc[0,2] - g_gs[0,2]

def run_parity_check():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    # Liu-Liang Scale
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td); inter = make_intermediates(h_obj, x_y, z1)
    
    dr = 0.0001; coords = mol.atom_coords()
    coords_p = coords.copy(); coords_p[0,2] += dr
    g_p = get_grad_excitation(mol.copy().set_geom_(coords_p, unit='Bohr'))
    coords_m = coords.copy(); coords_m[0,2] -= dr
    g_m = get_grad_excitation(mol.copy().set_geom_(coords_m, unit='Bohr'))
    h_fd = (g_p - g_m) / (2 * dr)
    print(f"FD d/dz (g_excitation) [0,2]: {h_fd:.8f}")
    
    # 1. MO response
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ)
    fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ); mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1); nocc = int(mo_occ.sum() // 2); nao = mo_coeff.shape[0]; nvir = mo_energy.size - nocc
    aoslices = mol.aoslice_by_atom(); _, _, s1a_basis = rhf_hess.get_ovlp(mol); s1a_basis = cp.asarray(s1a_basis)
    s1ao = cp.zeros((mol.natm, 3, nao, nao))
    for atm_id in range(mol.natm):
        p0, p1 = aoslices[atm_id][2:]; s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]; s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
    s1mo = cp.zeros((mol.natm, 3, nao, nao))
    for i in range(mol.natm):
        for j in range(3): s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
    Ux = cp.zeros((mol.natm, 3, nao, nao)); Ux[:, :, :, :nocc] = mo1
    Ux[:, :, :nocc, nocc:] = (-s1mo[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2))
    Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]
    
    pert_inter = make_perturbed_intermediates(h_obj, inter, x_y, cp.zeros((2,3,nocc,nvir)), cp.zeros((2,3,nocc,nvir)), Ux, z1)
    
    # Assembly
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    h_ip2 = _e_hcore_generator(h_obj, inter['P_I_prime'])(0,0)[2,2]
    
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    h1ao_z = cp.zeros((nao, nao)); p0, p1 = aoslices[0][2:]
    h1ao_z[p0:p1] += h1[2, p0:p1]; h1ao_z[:, p0:p1] += h1[2, p0:p1].T
    with mol.with_rinv_at_nucleus(0): vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
    h1ao_z += vrinv[2] * 2.0
    
    h_ip1 = cp.trace(pert_inter['P_I_prime_y'][0,2] @ h1ao_z)
    
    # Energy Part
    print(f"Analytical Hcore Part: {float(h_ip2 + 2*h_ip1):.8f} (ip2={h_ip2:.4f}, ip1={h_ip1:.4f})")
    # Wait, the physical excitation energy is Tr(P_I H).
    # d^2 Omega / dR^2 = Tr(P_I H^zz) + 2 Tr(P_I^z H^z)
    # My h_ip2 is for P_I_prime (trace 1.0). Generator multiplies by 2.
    # So h_ip2 is physical total Tr(P_tot H^zz).
    # 2*h_ip1 is physical total 2 Tr(P^z H^z).
    # So (h_ip2 + 2*h_ip1) is the physical derivative of the excitation energy.
    
    print(f"Analytical Total (Hcore): {(h_ip2 + 2*h_ip1):.8f}")
    print(f"Ratio: {(h_ip2 + 2*h_ip1) / h_fd:.4f}x")

run_parity_check()
