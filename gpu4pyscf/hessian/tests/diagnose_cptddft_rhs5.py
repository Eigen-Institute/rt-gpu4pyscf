#!/usr/bin/env python3
"""Check if Delta is orthogonal to the ground-state eigenvector."""

import numpy as np
import cupy as cp
from pyscf import gto
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu

H2O_ATOM = '''O   0.000000000000   0.000000000000   0.117790000000
H   0.000000000000   0.755453000000  -0.471160000000
H   0.000000000000  -0.755453000000  -0.471160000000'''

mol = gto.M(atom=H2O_ATOM, basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf)
td.nstates = 1
td.kernel()

x_y_orig = td.xy[0]
x, y = [cp.asarray(v) * cp.sqrt(2) for v in x_y_orig]
omega = float(td.e[0])
nocc = int((mf.mo_occ > 0).sum())
nvir = mf.mo_occ.shape[0] - nocc
nov = nocc * nvir

print("H2O: nocc=%d, nvir=%d, omega=%.6f" % (nocc, nvir, omega))

mo_coeff = cp.asarray(mf.mo_coeff)
mo_energy = cp.asarray(mf.mo_energy)
mo_occ = cp.asarray(mf.mo_occ)

mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
fx = mf_hess.gen_vind(mo_coeff, mo_occ)
mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
mo1 = cp.asarray(mo1)

Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(
    td_hess=tdrhf_hess.Hessian(td), x_y=(x, y), omega=omega, mo1=mo1, mo_e1=mo_e1)
Delta_np = Delta_cupy.get()

# Build A matrix and find its eigenvector corresponding to omega (ground state)
vind, hdiag = td.gen_vind()
A_mat = np.zeros((nov, nov))
for i in range(nov):
    Ax_i = vind(np.eye(nov)[i]).reshape(nocc, nvir).get()
    A_mat[i] = Ax_i.ravel()

# Get the eigenvector for omega (ground state) — it's at index 0 of sorted eigenvalues
eigvals, eigvecs = np.linalg.eigh(A_mat)
print("\nA eigenvalues:", eigvals)
print("omega =", omega)
print("Ground-state eigenvector indices (smallest):", np.argsort(eigvals)[:3])

# The ground state corresponds to the first eigenvector
X_gs = eigvecs[:, 0].reshape(nocc, nvir)
print("\nGround-state X^T:", X_gs.ravel())

# Check orthogonality: Tr[X_gs^† * Delta] should be 0 for each perturbation direction
print("\nOrthogonality check (should be ~0):")
for ia in range(mol.natm):
    for ix in range(3):
        # Inner product: <X_gs, Delta[ia,ix]> = sum_kl X_gs[k,l] * Delta[ia,ix][k,l]
        overlap = np.sum(X_gs * Delta_np[ia, ix])
        delta_norm = np.linalg.norm(Delta_np[ia, ix])
        if delta_norm > 1e-8:
            rel_overlap = abs(overlap) / delta_norm
        else:
            rel_overlap = abs(overlap)
        
        dir_name = ['X', 'Y', 'Z'][ix]
        print("  Atom %d %s: overlap=%.3e, |overlap|/|Delta|=%.3e" 
              % (ia, dir_name, overlap, rel_overlap))

# Now let me check: what does the solver actually return?
h = tdrhf_hess.Hessian(td)
x1_solver, y1_solver = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)

# Check if x1 has a huge component in direction 8 (the singular direction)
for ia in range(mol.natm):
    for ix in range(3):
        delta_norm = np.linalg.norm(Delta_np[ia, ix])
        if delta_norm < 1e-8:
            continue
        
        x1_ia_ix = cp.asnumpy(x1_solver[ia, ix].get()).ravel()
        
        # Project onto ground-state eigenvector
        proj_gs = np.sum(X_gs.ravel() * x1_ia_ix) / np.linalg.norm(X_gs.ravel())
        
        # Also check: what is (A - omega)*x1?
        Ax1_cupy = vind(cp.asarray(x1_ia_ix)).reshape(nocc, nvir)
        resid = cp.asnumpy(Ax1_cupy.get() - omega * x1_ia_ix.reshape(nocc, nvir)) + Delta_np[ia, ix]
        
        dir_name = ['X', 'Y', 'Z'][ix]
        print("\n  Atom %d %s:" % (ia, dir_name))
        print("    |x1| = %.6e" % np.linalg.norm(x1_ia_ix))
        print("    Projection onto X_gs: %.6e" % proj_gs)
        print("    |resid|/|Delta| = %.3e" % (np.linalg.norm(resid) / delta_norm))
