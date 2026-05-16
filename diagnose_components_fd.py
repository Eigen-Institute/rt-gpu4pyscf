import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates
from gpu4pyscf.lib.cupy_helper import contract

def get_grad_parts(mol, x_y_ref):
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    g_obj = tdrhf_grad.Gradients(td); x_y = x_y_ref
    z1 = solve_z_vector(g_obj, x_y)
    inter = make_intermediates(Hessian(td), x_y, z1)
    
    # 1. Hcore
    h1 = -cp.asarray(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
    p0, p1 = mol.aoslice_by_atom()[0][2:]
    gh = contract('xpq,pq->', h1[:, p0:p1], inter['P_I_prime'][p0:p1]) * 2.0
    with mol.with_rinv_at_nucleus(0):
        vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
    gh += cp.trace(vrinv[2] @ inter['P_I_prime'])
    
    # 2. ERI
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj1, vk1 = _get_jk_ip1(mol, inter['P_I_prime'])
    ge = float(cp.trace(inter['P'] @ (vj1 * 2.0 - vk1)[0,2])) 
    ge += float(cp.trace(inter['P_I_prime'] @ (vj1 * 2.0 - vk1)[0,2])) * 0.5
    
    # 3. Overlap
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    gs = contract('pq,pq->', cp.asarray(s1a[2, p0:p1]), inter['W_I'][p0:p1]) * 2.0
    
    return float(gh), float(ge), float(gs)

def diagnose():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    x_y_ref = td.xy[0]; dr = 0.0001; coords = mol.atom_coords()
    
    p_gh, p_ge, p_gs = get_grad_parts(mol.copy().set_geom_(coords.copy() + np.array([[0,0,dr],[0,0,0]]), unit='Bohr').build(), x_y_ref)
    m_gh, m_ge, m_gs = get_grad_parts(mol.copy().set_geom_(coords.copy() + np.array([[0,0,-dr],[0,0,0]]), unit='Bohr').build(), x_y_ref)
    
    hh = (p_gh - m_gh) / (2 * dr)
    he = (p_ge - m_ge) / (2 * dr)
    hs = (p_gs - m_gs) / (2 * dr)
    
    print(f"FD reference components:")
    print(f"  H-core: {hh:.8f}")
    print(f"  ERI:    {he:.8f}")
    print(f"  Ovlp:   {hs:.8f}")

    h_obj = Hessian(td)
    z1 = solve_z_vector(tdrhf_grad.Gradients(td), x_y_ref)
    inter = make_intermediates(h_obj, x_y_ref, z1)
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
    h_h_anal = float(_e_hcore_generator(h_obj, inter['P_I_prime'])(0,0)[2,2])
    
    vhfopt = mf._opt_gpu.get(mol.omega)
    ejk_PI = _partial_ejk_ip2(mol, inter['P_I_prime'], vhfopt)
    ejk_P = _partial_ejk_ip2(mol, inter['P'], vhfopt)
    ejk_total = _partial_ejk_ip2(mol, inter['P_I_prime'] + inter['P'], vhfopt)
    h_e_anal = float((ejk_total - ejk_PI - ejk_P)[0,0,2,2])
    
    s1aa, s1ab, _ = get_ovlp(mol); s1aa = cp.asarray(s1aa)
    p0, p1 = mol.aoslice_by_atom()[0][2:]
    h_s_anal = -float(contract('pq,pq->', inter['W_I'][p0:p1], s1aa[2,2,p0:p1])) * 2.0
    
    print(f"\nAnalytical raw components:")
    print(f"  H-core (raw): {h_h_anal:.8f} (Ratio: {h_h_anal/hh:.4f}x)")
    print(f"  ERI    (raw): {h_e_anal:.8f} (Ratio: {h_e_anal/he:.4f}x)")
    print(f"  Ovlp   (raw): {h_s_anal:.8f} (Ratio: {h_s_anal/hs:.4f}x)")

diagnose()
