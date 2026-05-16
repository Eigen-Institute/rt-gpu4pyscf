"""
Diagnostic script to track down the scaling factor discrepancy in analytical excited-state Hessian.

The issue stems from two normalization conventions:
  - PySCF TDA: 2(X^+*X) = 1 per spin channel → Tr[X_α^T @ X_α] = 0.5
  - Liu-Liang: X² = 1 for total (both spins combined)

The sqrt(2) conversion at line 548 of tdrhf.py converts PySCF amplitudes to Liu-Liang convention.

However, there is a missing coupling term between CP-TDDFT orbital rotations
(z1 from Z-vector response, x1py1/x1my1 from CP-TDDFT response) and the 
ground-state J/K field of D0: Tr{P_{z1}^{[y]} @ G[D0]^{[x]}}.

This term represents how excited-density orbital rotations couple through
the ground-state J/K field when taking second derivatives w.r.t. geometry.

The fix adds this coupling in e1_perturbed (tdrhf.py line 762):
    P_z1_cross[i, j] = Cv_i @ z1 @ Co_x_eval.T + orbv @ z1 @ Co_x_eval.T 
                     + Co_x_eval @ z1.T @ orbo.T + orbo @ z1.T @ Cv_i.T
    e1_perturbed += Tr{P_z1_cross[j0,y] @ G_D0_x[i0,x]}

To verify: compare analytical_omega_hessian() with method='analytical' vs 
method='semi-analytical' (FD on analytical gradient) for a simple system like H2O/STO-3G.
"""

import numpy as np
import cupy as cp
from pyscf import gto, scf
from gpu4pyscf import tdscf

# Small test systems to verify the fix

def test_H2():
    """H2 / STO-3G - very small system for quick testing"""
    print("=" * 60)
    print("Test: H2 / STO-3G")
    print("=" * 60)
    
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0, max_memory=8000)
    mf = scf.RHF(mol).run()
    
    td = tdscf.TDA(mf)
    td.nstates = 2
    td.kernel()
    
    print(f"\n=== PySCF TDA Amplitude Normalization ===")
    x_orig = td.xy[0][0]
    print(f"Tr[X_α^T @ X_α] (alpha spin) = {float(cp.asarray(x_orig)**2).sum():.6f}  (expected ~0.5)")

