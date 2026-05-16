#!/usr/bin/env python3
"""Compare scipy direct solve vs Krylov solver for H2O/STO-3G TDA."""

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

h = tdrhf_hess.Hessian(td)

x_y_orig = td.xy[0]
x, y = [cp.asarray(v) * cp.sqrt(2) for v in x_y_orig]
omega = float(td.e[0])
nocc = int((mf.mo_occ > 0).sum())
nvir = mf.mo_occ.shape[0] - nocc

print("H2O: nocc=%d, nvir=%d, omega=%.6f" % (nocc, nvir, omega))

mo_coeff = cp.asarray(mf.mo_coeff)
mo_energy = cp.asarray(mf.mo_energy)
mo_occ = cp.asarray(mf.mo_occ)

mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
fx = mf_hess.gen_vind(mo_coeff, mo_occ)
mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
mo1 = cp.asarray(mo1)

Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(h, (x, y), omega, mo1, mo_e1)
Delta_np = Delta_cupy.get()  # Convert to numpy for scipy solve

# Build A matrix explicitly for state 0
vind, hdiag = td.gen_vind()
nov = nocc * nvir
A_mat = np.zeros((nov, nov))
for i in range(nov):
    Ax_i = vind(np.eye(nov)[i]).reshape(nocc, nvir).get()
    A_mat[i] = Ax_i.ravel()

print("A diagonal:", np.diag(A_mat))

# Solve (A - wI) x1 = Delta_flat for a non-zero direction using scipy  
ia, ix = 0, 0  # Oxygen X-direction  
Delta_flat_np = Delta_np[ia, ix].ravel()
dir_name = ['X', 'Y', 'Z'][ix]
print("\nSolving for Atom %d dir '%s':" % (ia, dir_name))

A_omega = A_mat - omega * np.eye(nov)
from scipy.linalg import solve
x1_scipy = solve(A_omega, Delta_flat_np)
print("  scipy x1:", x1_scipy)

# Check residual with cupy  
Ax1_cupy = vind(x1_scipy.ravel()).reshape(nocc, nvir)
resid = Ax1_cupy - omega * cp.asarray(x1_scipy.reshape(Delta_cupy[ia, ix].shape)) + Delta_cupy[ia, ix]
rel_resid = float(cp.linalg.norm(resid)) / float(cp.linalg.norm(Delta_cupy[ia, ix]))
print("  scipy residual:", rel_resid)

# Now compare with the Krylov solver's output  
x1_solver, y1_solver = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)
x1_solver_flat_np = np.asarray(x1_solver[ia, ix].get()).ravel()
print("\n  Solver x1:", x1_solver_flat_np)

# Check solver's residual  
Ax1_s_cupy = vind(cp.asarray(x1_solver_flat_np)).reshape(nocc, nvir)
resid_s = Ax1_s_cupy - omega * cp.asarray(x1_solver_flat_np).reshape(nocc, nvir) + Delta_cupy[ia, ix]
rel_resid_s = float(cp.linalg.norm(resid_s)) / float(cp.linalg.norm(Delta_cupy[ia, ix]))
print("  Solver residual:", rel_resid_s)

# Check (A - omega)*solver_x1  
resid_check = Ax1_s_cupy - omega * cp.asarray(x1_solver_flat_np).reshape(nocc, nvir)
print("\n  Solver (A-w)*x1:", np.asarray(resid_check.get()))

# Jacobi iteration  
D_np = np.asarray(hdiag.ravel().get()) - omega
Delta_jac_flat = Delta_np.reshape(-1, nov)  # shape (9, 10) for H2O
x_jac = Delta_jac_flat / D_np[np.newaxis, :]

for it in range(50):
    Ax_jac_cupy = vind(x_jac.ravel()).reshape(nocc, nvir)
    r_jac = Delta_np.reshape(-1, nov) - (np.asarray(Ax_jac_cupy.get()) - omega * x_jac)
    x_jac += r_jac / D_np[np.newaxis, :]

# Check for each perturbation direction
print("\nJacobi check:")
for ia in range(mol.natm):
    for ix in range(3):
        idx = ia * 3 + ix
        Delta_idx_np = Delta_np[ia, ix].ravel()
        if np.linalg.norm(Delta_idx_np) < 1e-8:
            continue
        
        x_jac_this = x_jac[idx]
        Ax_jac_cupy = vind(x_jac.ravel())[idx*nov:(idx+1)*nov].reshape(nocc, nvir)
        resid_jac = cp.asarray(Ax_jac_cupy.get()) - omega * cp.asarray(x_jac_this.reshape(nocc, nvir)) + Delta_cupy[ia, ix]
        rel_jac = float(cp.linalg.norm(resid_jac)) / float(cp.linalg.norm(Delta_cupy[ia, ix]))
        
        dir_name2 = ['X', 'Y', 'Z'][ix]
        print("  Atom %d %s: jacobi_rel=%.3e" % (ia, dir_name2, rel_jac))

print("\nDone!")
