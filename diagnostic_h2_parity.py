import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def get_h1ao_x(mol):
    natm = mol.natm
    nao = mol.nao
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        # Use int3c2e.get_dh1e equivalent for vrinv
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3))
            vrinv *= -mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    return h1ao_x

def get_s1ao_x(mol):
    natm = mol.natm
    nao = mol.nao
    s1 = cp.asarray(-mol.intor('int1e_ovlp', comp=3))
    aoslices = mol.aoslice_by_atom()
    s1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao_x[atm_id, :, p0:p1] += s1[:, p0:p1]
        s1ao_x[atm_id, :, :, p0:p1] += s1[:, p0:p1].transpose(0, 2, 1)
    return s1ao_x

def run_full_parity():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    state = 0
    x_y_orig = td.xy[state]
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td)
    inter = make_intermediates(h_obj, x_y, z1)
    P_I_prime = inter['P_I_prime']
    W_I = inter['W_I']
    print(f"W_I norm: {float(cp.linalg.norm(W_I)):.6f}, trace: {float(cp.trace(W_I)):.6f}")
    
    dr = 0.001
    coords = mol.atom_coords()
    
    # 1. H-core parity
    # Analytical (uses _e_hcore_generator which returns 2.0*de)
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    de_hcore_gen = _e_hcore_generator(h_obj, P_I_prime)
    # Total assembly has / 2.0, so we divide by 2.0 here
    hcore_anal = float(de_hcore_gen(0, 0)[2,2]) / 2.0
    
    # Numerical
    def get_g_hcore(mol_):
        h1ao_x = get_h1ao_x(mol_)
        return cp.trace(P_I_prime @ h1ao_x[0,2])
    coords_p = coords.copy(); coords_p[0,2] += dr
    g_p = get_g_hcore(mol.copy().set_geom_(coords_p, unit='Bohr'))
    coords_m = coords.copy(); coords_m[0,2] -= dr
    g_m = get_g_hcore(mol.copy().set_geom_(coords_m, unit='Bohr'))
    hcore_num = (g_p - g_m) / (2 * dr)
    
    # 2. Overlap parity
    # Analytical
    from gpu4pyscf.hessian.rhf import get_ovlp
    s1aa, s1ab, s1a = get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    p0, p1 = aoslices[0][2:]
    # e1_ovlp had * 2.0 factor. Total assembly has / 2.0. So net 1.0.
    ovlp_anal = float(cp.einsum('pq,xypq->xy', W_I[p0:p1], cp.asarray(s1aa[:,:,p0:p1,:]))[2,2]) * 2.0 / 2.0
    # Wait, the code has: e1_ovlp[i,i] += contract(s1aa, W_I[p0:p1]) * 2
    # Then final / 2.0. So it's 1.0.
    
    # Numerical
    def get_g_ovlp(mol_):
        s1ao_x = get_s1ao_x(mol_)
        if mol_.atom_coords()[0,2] > coords[0,2]: # Print only for g_p
            print(f"DEBUG s1ao_x[0,2,0,0]: {float(s1ao_x[0,2,0,0]):.6f}")
        return cp.trace(W_I @ s1ao_x[0,2]) * 2.0
    g_p_o = get_g_ovlp(mol.copy().set_geom_(coords_p, unit='Bohr'))
    g_m_o = get_g_ovlp(mol.copy().set_geom_(coords_m, unit='Bohr'))
    ovlp_num = (g_p_o - g_m_o) / (2 * dr)
    print(f"DEBUG Overlap: g_p {float(g_p_o):.6f}, g_m {float(g_m_o):.6f}")
    
    print(f"H-core:  Analytical {hcore_anal:10.6f}, Numerical {hcore_num:10.6f}, Ratio {hcore_anal/hcore_num:7.4f}x")
    print(f"Overlap: Analytical {ovlp_anal:10.6f}, Numerical {ovlp_num:10.6f}, Ratio {ovlp_anal/ovlp_num:7.4f}x")

run_full_parity()
