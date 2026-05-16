#!/usr/bin/env python3
"""
Diagnose CP-TDDFT RHS construction for H2/STO-3G TDA.

For TDA with Y=0, Eq. 20 simplifies to:
    (A - omega) X^x = Delta_x
    
where A is the TDA matrix and Delta_x comes from Eq. 20 expanded.
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
    
    # Scale amplitudes to Liu-Liang convention
    x_y_orig = td.xy[0]
    x, y = [cp.asarray(v) * cp.sqrt(2) for v in x_y_orig]
    omega = float(td.e[0])
    nocc = int((mf.mo_occ > 0).sum())
    nvir = mf.mo_occ.shape[0] - nocc
    nov = nocc * nvir
    
    print(f"nocc={nocc}, nvir={nvir}, omega={omega:.6f}")
    print(f"x norm: {float(cp.linalg.norm(x)):.6e}")
    
    # Build mo1/Ux for CP-TDDFT RHS
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)
    
    # Get the TDA matrix A for state 0
    vind, hdiag = td.gen_vind()
    print(f"hdiag (A diagonal): {cp.asnumpy(hdiag.ravel())}")
    
    # Build RHS
    Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(h, (x, y), omega, mo1, mo_e1)
    print(f"Delta norm: {float(cp.linalg.norm(Delta)):.6e}")
    print(f"Delta[0,2] (Z-direction atom 0): {cp.asnumpy(Delta[0, 2]).ravel()}")
    print(f"Delta[1,2] (Z-direction atom 1): {cp.asnumpy(Delta[1, 2]).ravel()}")
    
    # Solve CP-TDDFT
    x1, y1 = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)
    print(f"\nx1 norm: {float(cp.linalg.norm(x1)):.6e}")
    print(f"x1[0,2] (Z-direction atom 0): {cp.asnumpy(x1[0, 2]).ravel()}")
    print(f"x1[1,2] (Z-direction atom 1): {cp.asnumpy(x1[1, 2]).ravel()}")
    
    # Compute residual: (A - omega) x1 + Delta
    for ia in range(mol.natm):
        for ix in range(3):
            x1_flat = x1[ia, ix].ravel()
            Ax1 = vind(x1_flat).reshape(nocc, nvir)
            
            # For TDA: (A - omega)*X^x + Delta = 0 means residual should be ~0
            resid = (Ax1 - omega * x1_flat.reshape(nocc, nvir)) + Delta[ia, ix]
            abs_resid = float(cp.linalg.norm(resid))
            norm_delt = float(cp.linalg.norm(Delta[ia, ix]))
            
            if norm_delt > 0:
                rel_resid = abs_resid / norm_delt
            else:
                rel_resid = abs_resid
            
            dir_name = ['X', 'Y', 'Z'][ix]
            print(f"  Atom {ia} {dir_name}: resid={abs_resid:.3e}, Delta_norm={norm_delt:.3e}, "
                  f"rel={rel_resid:.3e}")

if __name__ == '__main__':
    from gpu4pyscf import tdscf as gpu_tdscf
    diagnose()
