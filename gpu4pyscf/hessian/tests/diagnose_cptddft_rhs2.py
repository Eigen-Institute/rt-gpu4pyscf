#!/usr/bin/env python3
"""
Deeper diagnostic: Check what (A - omega) * x1 gives vs -Delta for H2/STO-3G TDA.
"""

import numpy as np
import cupy as cp
from pyscf import gto, scf
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu

H2_ATOM = '''H 0 0 0; H 0 0 1.4'''

def diagnose():
    mol = gto.M(atom=H2_ATOM, basis='sto-3g', unit='Bohr', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()
    
    h = tdrhf_hess.Hessian(td)
    
    x_y_orig = td.xy[0]
    x, y = [cp.asarray(v) * cp.sqrt(2) for v in x_y_orig]
    omega = float(td.e[0])
    nocc = int((mf.mo_occ > 0).sum())
    nvir = mf.mo_occ.shape[0] - nocc
    
    print(f"nocc={nocc}, nvir={nvir}, omega={omega:.6f}")
    
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)
    
    vind, hdiag = td.gen_vind()
    print(f"hdiag[0] (A diagonal for state 0): {float(hdiag.ravel()[0]):.6f}")
    
    # Build RHS and solve CP-TDDFT
    Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(h, (x, y), omega, mo1, mo_e1)
    x1, y1 = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)
    
    # For Z-direction atom 0: check manually
    ia, ix = 0, 2
    x1_flat = x1[ia, ix].ravel()
    Delta_flat = Delta[ia, ix].ravel()
    
    Ax1 = vind(x1_flat).reshape(nocc, nvir)
    print(f"\nAtom {ia} Z-direction:")
    print(f"  x1[ia,ix] = {cp.asnumpy(x1_flat)}")
    print(f"  Delta[ia,ix] = {cp.asnumpy(Delta_flat)}")
    
    # (A - omega) * x1 should equal -Delta
    diff_A_omega_x1 = Ax1 - omega * x1_flat.reshape(nocc, nvir)
    print(f"  (A - omega) * x1 = {cp.asnumpy(diff_A_omega_x1)}")
    print(f"  -(A - omega) * x1 = {cp.asnumpy(-diff_A_omega_x1)}")
    print(f"  Expected: -Delta = {-cp.asnumpy(Delta_flat)}")
    
    # What should x1 be? Delta / (hdiag - omega) in the diagonal approximation
    D_diag = hdiag.ravel()[0] - omega
    x1_expected = Delta_flat / D_diag
    print(f"\n  Diagonal approx: D = {float(D_diag):.6f}")
    print(f"  Expected x1 = Delta/D = {cp.asnumpy(x1_expected)}")
    
    # Check the actual solver's preconditioned equation
    Kx = Ax1 - hdiag.ravel()[0] * x1_flat.reshape(nocc, nvir)
    print(f"\n  Kx = A*x1 - hdiag*x1 = {cp.asnumpy(Kx)}")
    print(f"  Preconditioner: -Kx / D = {-float(Kx.ravel()[0]) / float(D_diag):.6f}")

if __name__ == '__main__':
    from gpu4pyscf import tdscf as gpu_tdscf
    diagnose()
