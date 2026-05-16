#!/usr/bin/env python3
"""Diagnose CP-TDDFT solver behavior more carefully."""

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

print("H2O: nocc=%d, nvir=%d, omega=%.6f" % (nocc, nvir, omega))

mo_coeff = cp.asarray(mf.mo_coeff)
mo_energy = cp.asarray(mf.mo_energy)
mo_occ = cp.asarray(mf.mo_occ)

mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
fx = mf_hess.gen_vind(mo_coeff, mo_occ)
mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
mo1 = cp.asarray(mo1)

Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(td_hess=tdrhf_hess.Hessian(td), x_y=(x, y), omega=omega, mo1=mo1, mo_e1=mo_e1)
Delta_np = Delta_cupy.get()  # Convert to numpy for manual checks

# Build A matrix explicitly for state 0 and compute its eigenvalues
vind, hdiag = td.gen_vind()
nov = nocc * nvir
A_mat = np.zeros((nov, nov))
for i in range(nov):
    Ax_i = vind(np.eye(nov)[i]).reshape(nocc, nvir).get()
    A_mat[i] = Ax_i.ravel()

eigvals = np.linalg.eigvalsh(A_mat)
print("\nA eigenvalues:", eigvals)
print("omega =", omega)
print("Distance |lambda - omega|:", np.min(np.abs(eigvals - omega)))

# The Krylov solver uses a preconditioned iteration. Let me trace it:
from gpu4pyscf.lib.cupy_helper import krylov

D = hdiag.ravel().get() - omega
print("\nPreconditioner D = hdiag - omega:")
for i in range(nov):
    print("  D[%d] = %.6f" % (i, D[i]))

# Initial guess for TDA: x0 = Delta / D
Delta_flat = Delta_cupy.reshape(-1, nov).get()
x0 = Delta_flat / D[np.newaxis, :]
print("\nInitial guess norm:", float(cp.linalg.norm(cp.asarray(x0))))

# Now let me manually trace the Krylov iteration for one perturbation direction
ia, ix = 0, 0  # Oxygen X-direction
idx = ia * 3 + ix  # index in flattened (natm*3) dimension

b_vec = Delta_flat[idx]  # shape (nov,) - this is Delta for (ia=0, ix=0)
print("\nTarget: solve (A - omega)*x1[ia=%d,ix=%d] = Delta" % (ia, ix))
print("  b_norm:", float(cp.linalg.norm(cp.asarray(b_vec))))

# Check what (A - omega)*x0 gives  
Ax0_cupy = vind(x0.ravel()).reshape(nocc, nvir)
r0_np = np.asarray(Ax0_cupy.get()) - omega * x0.reshape(-1, nov)[idx]
print("  r0 = A*x0 - w*x0 =", float(cp.linalg.norm(cp.asarray(r0_np))))

# Now run the Krylov solver and check what it actually solves
h = tdrhf_hess.Hessian(td)
x1_solver, y1_solver = tdrhf_hess.solve_cptddft(h, (x, y), omega, mo1, mo_e1)
x1_ia_ix_np = np.asarray(x1_solver[ia, ix].get()).ravel()

# What does the solver output give for (A - omega)*x1?  
Ax1_cupy = vind(cp.asarray(x1_ia_ix_np)).reshape(nocc, nvir)
resid = Ax1_cupy - omega * cp.asarray(x1_ia_ix_np).reshape(nocc, nvir) + Delta_cupy[ia, ix]
rel_resid = float(cp.linalg.norm(resid)) / float(cp.linalg.norm(Delta_cupy[ia, ix]))

print("\nSolver output:")
print("  x1:", x1_ia_ix_np)
print("  (A-w)*x1 + Delta norm:", float(cp.linalg.norm(resid)))
print("  relative residual:", rel_resid)

# The Krylov solver is solving: vind(x1) - hdiag*x1 = -Kx/D where Kx = Ax - hdiag*x
# For TDA, the iteration is: x_{n+1} = x_n - (A_diag - omega)^{-1}(Ax_n - hdiag*x_n - r_n)
# where r_n = Delta - A*x_n + omega*x_n

# Let me check: does the solver satisfy: vind(x1) = something specific?
vind_x1 = vind(cp.asarray(x1_ia_ix_np))
print("\n  vind(x1):", np.asarray(vind_x1.get().ravel()))
print("  hdiag*x1:", np.asarray(hdiag.ravel().get() * cp.asarray(x1_ia_ix_np).reshape(-1, nov)[0]))

# Let me check the Krylov VOP more carefully  
# The solver builds: Kx = vind(V) - hdiag*V where V is in the subspace
# Then returns -Kx / D as the search direction

# For the initial vector x0:
Ax0 = cp.asarray(Ax0_cupy.ravel())
print("\n  A*x0:", np.asarray(Ax0[:5]))
print("  hdiag*x0:", np.asarray((hdiag.ravel().get() * cp.asarray(x0.ravel()))[:5]))
