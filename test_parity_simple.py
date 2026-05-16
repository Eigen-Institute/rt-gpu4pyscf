import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian

def run_verification():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    h_anal_obj = Hessian(td); h_anal_obj.method = 'analytical'
    h_anal = h_anal_obj.kernel()
    
    h_semi_obj = Hessian(td); h_semi_obj.method = 'semi-analytical'
    h_semi = h_semi_obj.kernel()
    
    print("H2 / STO-3G TDA (state 0) Parity Check")
    print(f"{'Component':15} {'Semi-Anal':>12} {'Analytical':>12} {'Ratio':>8}")
    print("-" * 50)
    
    components = [
        ((0,2), (0,2)), # Atom 0, Z; Atom 0, Z
        ((0,0), (0,0)), # Atom 0, X; Atom 0, X
        ((0,2), (1,2)), # Cross Z
    ]
    
    for (a0, d0), (a1, d1) in components:
        v_s = float(h_semi[a0, d0, a1, d1])
        v_a = float(h_anal[a0, d0, a1, d1])
        r = v_a / v_s if abs(v_s) > 1e-8 else 0.0
        name = f"[{a0},{d0}]-[{a1},{d1}]"
        print(f"{name:15} {v_s:12.8f} {v_a:12.8f} {r:8.4f}x")
    
    # TI check
    ti = float(cp.abs(h_anal.sum(axis=0)).max())
    print(f"\nTranslational Invariance Violation: {ti:.3e}")

run_verification()
