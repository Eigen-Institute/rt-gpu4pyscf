import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates, solve_cptddft
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.grad.tdrhf import Gradients

def get_grad_parts(mol_curr, x_y_ref):
    mf = RHF(mol_curr).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    g_obj = Gradients(td)
    # Use REF amplitudes to stay at constant X
    x_y = x_y_ref
    
    # Gradient in AO basis: Tr[P H^x] + Tr[W S^x] + ...
    # Z-vector for this geometry
    z1 = solve_z_vector(g_obj, x_y)
    inter = make_intermediates(Hessian(td), x_y, z1)
    P = inter['P_I_prime']
    W = inter['W_I']
    
    # Core Force: Tr[P H^x]
    h1 = -cp.asarray(mol_curr.intor('int1e_ipkin', comp=3) + mol_curr.intor('int1e_ipnuc', comp=3))
    p0, p1 = mol_curr.aoslice_by_atom()[0][2:]
    f_hcore = float(contract('xpq,pq->', h1[:, p0:p1], P[p0:p1])) * 2.0
    with mol_curr.with_rinv_at_nucleus(0):
        vrinv = cp.asarray(mol_curr.intor('int1e_iprinv', comp=3)) * -mol_curr.atom_charge(0)
    f_hcore += float(cp.trace(vrinv[2] @ P))
    
    # Ovlp Force: Tr[W S^x]
    s1 = -cp.asarray(mol_curr.intor('int1e_ipovlp', comp=3))
    f_ovlp = float(contract('xpq,pq->', s1[:, p0:p1], W[p0:p1])) * 2.0
    
    return f_hcore, f_ovlp

def diagnose_scaling(system='H2'):
    if system == 'H2':
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    else:
        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto-3g', verbose=0)
        
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    x_y_ref = td.xy[0]
    
    dr = 0.0001; coords = mol.atom_coords()
    c_p = coords.copy(); c_p[0,2] += dr
    mol_p = mol.copy().set_geom_(c_p, unit='Bohr'); mol_p.build()
    fh_p, fs_p = get_grad_parts(mol_p, x_y_ref)
    
    c_m = coords.copy(); c_m[0,2] -= dr
    mol_m = mol.copy().set_geom_(c_m, unit='Bohr'); mol_m.build()
    fh_m, fs_m = get_grad_parts(mol_m, x_y_ref)
    
    h_hcore_fd = (fh_p - fh_m) / (2 * dr)
    h_ovlp_fd = (fs_p - fs_m) / (2 * dr)
    
    # Analytical
    z1 = solve_z_vector(Gradients(td), x_y_ref)
    inter = make_intermediates(Hessian(td), x_y_ref, z1)
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, get_ovlp
    de_hcore = _e_hcore_generator(Hessian(td), inter['P_I_prime'])
    h_hcore_anal = float(de_hcore(0,0)[2,2])
    
    s1aa, s1ab, _ = get_ovlp(mol); s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    p0, p1 = mol.aoslice_by_atom()[0][2:]
    h_ovlp_anal = -float(contract('pq,pq->', inter['W_I'][p0:p1], s1aa[2,2,p0:p1])) * 2.0
    h_ovlp_anal -= float(contract('pq,pq->', inter['W_I'][p0:p1, p0:p1], s1ab[2,2,p0:p1, p0:p1])) * 4.0
    
    print(f"System: {system}")
    print(f"  Hcore FD: {h_hcore_fd:.8f}, Anal: {h_hcore_anal:.8f}, Ratio: {h_hcore_anal/h_hcore_fd:.4f}x")
    print(f"  Ovlp  FD: {h_ovlp_fd:.8f}, Anal: {h_ovlp_anal:.8f}, Ratio: {h_ovlp_anal/h_ovlp_fd:.4f}x")

print("Scaling Diagnosis (Static Part Only)")
diagnose_scaling('H2')
diagnose_scaling('H2O')
