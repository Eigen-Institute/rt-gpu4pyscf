"""
Test: Analytical Hessian coupling fix verification.

Compares analytical_omega_hessian() with method='analytical' vs 
method='semi-analytical' (FD on analytical gradient) to verify the missing
CP-TDDFT coupling term has been fixed.

Before fix: ~12.7x magnitude discrepancy  
After fix: should agree to within numerical precision (< 0.01 max abs diff)
"""

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, omega_grad

def test_H2():
    """H2 / STO-3G - very small system for quick testing"""
    print("=" * 60)
    print("Test: H2 / STO-3G")  
    print("=" * 60)
    
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0, max_memory=8000)
    mf = RHF(mol).run()
    
    td = TDA(mf)
    td.nstates = 2
    td.kernel()
    
    print(f"\n=== PySCF TDA Amplitude Normalization ===")
    x_orig = cp.asarray(td.xy[0][0])
    print(f"Tr[X_α^T @ X_α] (alpha spin) = {float(cp.sum(x_orig**2)):.6f}  (expected ~0.5)")
