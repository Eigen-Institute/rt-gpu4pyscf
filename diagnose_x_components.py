import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates, make_perturbed_intermediates
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def diagnose_x():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    x_y = tuple([cp.asarray(v) for v in td.xy[0]])
    h_obj = Hessian(td)
    from gpu4pyscf.grad.tdrhf import Gradients
    z1 = solve_z_vector(Gradients(td), x_y)
    inter = make_intermediates(h_obj, x_y, z1)
    
    # ip2 hcore
    from gpu4pyscf.hessian.rhf import _e_hcore_generator, get_ovlp, _partial_ejk_ip2
    de_hcore = _e_hcore_generator(h_obj, inter['P_I_prime'])
    h2_hcore = de_hcore(0,0)[0,0]
    print(f"H-core ip2 X-X [0,0]: {h2_hcore:.8f}")
    
    # ip2 ovlp
    s1aa, s1ab, _ = get_ovlp(mol); s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    aoslices = mol.aoslice_by_atom(); p0, p1 = aoslices[0][2:]
    
    val_aa = -float(contract('pq,pq->', inter['W_I'][p0:p1], s1aa[0,0,p0:p1])) * 2.0
    val_ab = -float(contract('pq,pq->', inter['W_I'][p0:p1, p0:p1], s1ab[0,0,p0:p1,p0:p1])) * 4.0
    print(f"Overlap ip2 X-X [0,0] (s1aa part): {val_aa:.8f}")
    print(f"Overlap ip2 X-X [0,0] (s1ab part): {val_ab:.8f}")
    
    # ip2 eri
    vhfopt = mf._opt_gpu.get(mol.omega)
    P = inter['P']; P_I_prime = inter['P_I_prime']; R_I = inter['R_I']
    ejk_ip2 = _partial_ejk_ip2(mol, P_I_prime + P, vhfopt) - _partial_ejk_ip2(mol, P, vhfopt)
    ejk_ip2 += _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt)
    h2_eri = ejk_ip2[0,0,0,0]
    print(f"ERI ip2 X-X [0,0]:    {h2_eri:.8f}")

    # Total ip2
    total_ip2 = h2_hcore + val_aa + val_ab + h2_eri
    print(f"Total ip2 X-X [0,0]:  {total_ip2:.8f}")
    
    # Semi-analytical reference
    h_sa = h_obj.kernel()
    print(f"Semi-Analytical X-X [0,0]: {float(h_sa[0,0,0,0]):.8f}")

diagnose_x()
