import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates
from gpu4pyscf.lib.cupy_helper import contract

def diagnose():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    g_obj = tdrhf_grad.Gradients(td); x_y = td.xy[0]
    z1 = solve_z_vector(g_obj, x_y)
    inter = make_intermediates(Hessian(td), x_y, z1)
    P = inter['P_I_prime']
    
    # Check Tr[P H^zz] for atom 0
    with mol.with_rinv_at_nucleus(0):
        # Potential: V = -Z/|r-R0|. 
        # Gradient: V' = Z(r-R0)/|r-R0|^3.
        # Hessian: V'' = ...
        # This is exactly what ipiprinv computes.
        ipiprinv = cp.asarray(mol.intor('int1e_ipiprinv', comp=9)).reshape(3,3,mol.nao,mol.nao)
        hf_curv = contract('xypq,pq->xy', ipiprinv, P) * -mol.atom_charge(0)
    
    print(f"Hellmann-Feynman Curvature on Atom 0: {hf_curv[2,2]:.8f}")
    
    # Hcore ip2 from generator
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    h_hcore_gen = float(_e_hcore_generator(Hessian(td), P)(0,0)[2,2])
    print(f"Generator Hcore ip2 (H00): {h_hcore_gen:.8f}")

diagnose()
