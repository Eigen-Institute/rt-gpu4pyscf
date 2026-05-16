#!/usr/bin/env python3
"""
Step 1A — Validate CP-TDDFT solver residual.

Before debugging the Hessian assembly, confirm that x1 from solve_cptddft 
actually solves (A − ω) x1 = −Delta.

Reference: Liu & Liang (2013), Eq. 20-21 for TDA Y=0 simplifies to:
    A * X^x - omega * X^x = Delta_x
    
or equivalently:
    (A - omega) x1 = -Delta
"""

import numpy as np
import cupy as cp
from pyscf import gto, scf
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess

H2_ATOM = '''H 0 0 0; H 0 0 1.4'''

def test_cptddft_residual():
    mol = gto.M(atom=H2_ATOM, basis='sto-3g', unit='Bohr', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()
    
    h = tdrhf_hess.Hessian(td)
    
    # Scale amplitudes to Liu-Liang convention
    x_y_orig = td.xy[0]
    x, y = [cp.asarray(v) * cp.sqrt(2) for v in x_y_orig]
    omega = float(td.e[0])
    nocc = int((mf.mo_occ > 0).sum())
    nvir = mf.mo_occ.shape[0] - nocc
    nov = nocc * nvir
    
    print(f"nocc={nocc}, nvir={nvir}, omega={omega:.6f}")
    
    # Build mo1/Ux for CP-TDDFT RHS (required by make_cptddft_rhs)
    from gpu4pyscf.hessian import rhf as rhf_hess_gpu
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)  # Convert to cupy array
    
    # Solve CP-TDDFT
    x1, y1 = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)
    print(f"x1 shape: {x1.shape}, norm: {float(cp.linalg.norm(x1)):.6e}")
    print(f"y1 shape: {y1.shape}, norm: {float(cp.linalg.norm(y1)):.6e}")
    
    # Build RHS (Delta) — requires mo1 for the F^x_AO term
    Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(h, (x, y), omega, mo1, mo_e1)
    
    # For TDA with Y=0, Eq. 20 becomes: (A - omega) X^x = Delta
    vind, hdiag = td.gen_vind()
    
    # Check residual for each atom-perturbation direction
    max_resid = 0.0
    for ia in range(mol.natm):
        for ix in range(3):
            x1_flat = x1[ia, ix].ravel()
            Delta_flat = Delta[ia, ix].ravel()
            
            # (A - omega) * x1 + Delta should be ~0
            Ax1 = vind(x1_flat).reshape(nocc, nvir)
            residual = Ax1 - omega * x1_flat.reshape(nocc, nvir) + Delta[ia, ix]
            rel = cp.linalg.norm(residual) / (cp.linalg.norm(Delta[ia, ix]) + 1e-30)
            
            if float(rel) > max_resid:
                max_resid = float(rel)
    
    print(f"\nCP-TDDFT residual: {max_resid:.3e}")
    if max_resid < 1e-6:
        print("PASS: Solver is converging correctly.")
    else:
        print("FAIL: Solver residual too large — x1 does NOT solve (A - omega)x1 = -Delta")

if __name__ == '__main__':
    from gpu4pyscf import tdscf as gpu_tdscf
    test_cptddft_residual()
