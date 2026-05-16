import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def diagnose_fd_z():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    x_y = tuple([cp.asarray(v) for v in td.xy[0]])
    h_obj = Hessian(td)
    from gpu4pyscf.grad.tdrhf import Gradients
    z1 = solve_z_vector(Gradients(td), x_y)
    inter = make_intermediates(h_obj, x_y, z1)
    P = inter['P_I_prime']
    W = inter['W_I']
    
    dr = 0.0001; coords = mol.atom_coords()
    
    def get_grad_parts(mol_curr):
        mf_c = RHF(mol_curr).run()
        h1 = mf_c.get_hcore()
        s1 = mf_c.get_ovlp()
        return h1, s1

    # FD Z-Z ip2
    c_p = coords.copy(); c_p[0,2] += dr
    mol_p = mol.copy(); mol_p.set_geom_(c_p, unit='Bohr'); mol_p.build()
    h_p, s_p = get_grad_parts(mol_p)
    
    c_m = coords.copy(); c_m[0,2] -= dr
    mol_m = mol.copy(); mol_m.set_geom_(c_m, unit='Bohr'); mol_m.build()
    h_m, s_m = get_grad_parts(mol_m)
    
    h_0 = mf.get_hcore(); s_0 = mf.get_ovlp()
    val_0 = float(cp.trace(P @ cp.asarray(h_0)) + cp.trace(W @ cp.asarray(s_0)))
    val_p = float(cp.trace(P @ cp.asarray(h_p)) + cp.trace(W @ cp.asarray(s_p)))
    val_m = float(cp.trace(P @ cp.asarray(h_m)) + cp.trace(W @ cp.asarray(s_m)))
    
    h_ip2_fd = (val_p - 2*val_0 + val_m) / (dr**2)
    print(f"FD ip2 Z-Z [0,0]: {h_ip2_fd:.8f}")
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, get_ovlp
    de_hcore = _e_hcore_generator(h_obj, P)
    h_anal_hcore = float(de_hcore(0,0)[2,2])
    
    s1aa, s1ab, _ = get_ovlp(mol); s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    aoslices = mol.aoslice_by_atom(); p0, p1 = aoslices[0][2:]
    
    # Physical Formula: only cross-terms survive for overlap
    h_anal_ovlp = 0.0
    for j in range(mol.natm):
        if j == 0: continue
        q0, q1 = aoslices[j][2:]
        h_anal_ovlp -= float(contract('pq,pq->', W[p0:p1, q0:q1], s1aa[2,2,p0:p1, q0:q1])) * 2.0
    
    print(f"Analytical Hcore: {h_anal_hcore:.8f}")
    print(f"Analytical Ovlp:  {h_anal_ovlp:.8f}")
    print(f"Ratio (Hcore/2 + Ovlp)/FD: {(h_anal_hcore/2.0 + h_anal_ovlp) / h_ip2_fd:.4f}x")

diagnose_fd_z()
