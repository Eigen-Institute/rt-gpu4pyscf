#!/usr/bin/env python3
"""Check orthogonality of Delta with X_gs for H2O/STO-3G TDA."""

import numpy as np
import cupy as cp
from pyscf import gto
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess

H2O_ATOM = '''O   0.000000000000   0.000000000000   0.117790000000
H   0.000000000000   0.755453000000  -0.471160000000
H   0.000000000000  -0.755453000000  -0.471160000000'''

mol = gto.M(atom=H2O_ATOM, basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf)
td.nstates = 1
td.kernel()

state = 0
omega = float(td.e[state])
x_y_orig = td.xy[state]
nocc = int((mf.mo_occ > 0).sum())
nvir = mol.nao - nocc
nov = nocc * nvir

# Build mo1/Ux for CP-TDDFT RHS  
mo_coeff = cp.asarray(mf.mo_coeff)
mo_energy = cp.asarray(mf.mo_energy)
mo_occ = cp.asarray(mf.mo_occ)

from gpu4pyscf.hessian import rhf as rhf_hess_gpu
mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
fx = mf_hess.gen_vind(mo_coeff, mo_occ)
mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
mo1 = cp.asarray(mo1)

# Build Delta
h = tdrhf_hess.Hessian(td)
x_y_scaled = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(h, x_y_scaled, omega, mo1, mo_e1)

# Check: is Delta orthogonal to X_gs?
x_gs_np = np.asarray(x_y_orig[0])  # (nocc, nvir) — the excitation amplitude X (numpy already)

print("Checking <X_gs, Delta> for each perturbation direction:")
for ia in range(mol.natm):
    for ix in range(3):
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())  # (nocc, nvir)
        overlap = np.sum(x_gs_np * delta_ia_ix)
        delta_norm = np.linalg.norm(delta_ia_ix)
        
        if delta_norm > 1e-8:
            rel_overlap = abs(overlap) / delta_norm
        else:
            rel_overlap = abs(overlap)
        
        dir_name = ['X', 'Y', 'Z'][ix]
        print("  Atom %d Dir '%s': overlap=%.3e, |Delta|=%.3e, rel=%.2f" 
              % (ia, dir_name, overlap, delta_norm, rel_overlap))

# Now check the specific component: is Delta[4,0] non-zero?
print("\nChecking Delta at component 8 (occ=4,virt=0) — the singular direction:")
for ia in range(mol.natm):
    for ix in range(3):
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())
        print("  Atom %d Dir '%s': Delta[4,0]=%.6e" 
              % (ia, ['X','Y','Z'][ix], delta_ia_ix[4,0]))

# Check: what is the overlap of X_gs with each component of Delta?
print("\nComponent-by-component overlap <X_gs, Delta>:")
for ia in range(mol.natm):
    for ix in range(3):
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())
        x_gs = x_gs_np
        
        # Overlap component by component
        total_overlap = 0.0
        for i in range(nocc):
            for j in range(nvir):
                contrib = x_gs[i,j] * delta_ia_ix[i,j]
                if abs(contrib) > 1e-8:
                    print("    Atom %d Dir '%s': X[%d,%d]=%.6f * Delta=%.6e = %.3e" 
                          % (ia, ['X','Y','Z'][ix], i, j, x_gs[i,j], delta_ia_ix[i,j], contrib))
                total_overlap += contrib
