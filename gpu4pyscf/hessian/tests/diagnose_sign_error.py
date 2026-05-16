#!/usr/bin/env python3
"""
For TDA with B=0 and Y=0, Eq. 24 gives Δ₁ = ω_x * X + ... 

But from differentiating A*X = ω*X (the ground state equation):
  d(A - ωI)/dR * X + (A-ωI) * dX/dR = 0
  (A-ωI)*X^x = -(d(ε_a - ε_i)_x)*X

So the RHS should be -d(ε_x)*X, not +ω_x*X!

The sign error in make_cptddft_rhs: it uses +omega_x instead of -omega_x.
This causes <X, Δ₁> = ω_x*<X,X> ≠ 0 for TDA with B=0 and Y=0.

Let me verify by checking what the correct sign should give us.
"""

import numpy as np
import cupy as cp
from pyscf import gto
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess

H2_ATOM = '''H 0 0 0; H 0 0 1.4'''

mol = gto.M(atom=H2_ATOM, basis='sto-3g', unit='Bohr', verbose=0)
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

print("H2: nocc=%d, nvir=%d, omega=%.6f" % (nocc, nvir, omega))

# Build mo1/Ux for CP-TDDFT RHS  
mo_coeff = cp.asarray(mf.mo_coeff)
mo_energy = cp.asarray(mf.mo_energy)
mo_occ = cp.asarray(mf.mo_occ)

from gpu4pyscf.hessian import rhf as rhf_hess_gpu
mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, cp.asarray(mf.mo_occ))
fx = mf_hess.gen_vind(mo_coeff, cp.asarray(mf.mo_occ))
mo1, mo_e1 = mf_hess.solve_mo1(cp.asarray(mf.mo_energy), mo_coeff, cp.asarray(mf.mo_occ), h1mo, fx)
mo1 = cp.asarray(mo1)

# Get the gradient of excitation energy  
h_obj = tdrhf_hess.Hessian(td)
omega_grad_z = h_obj.omega_grad(state)  # (natm, 3)

x_y_scaled = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(h_obj, x_y_scaled, omega, mo1, mo_e1)

# For H2/STO-3G TDA with nocc=nvir=1: X_gs is just one component [0,0]
X_gs_np = np.asarray(x_y_orig[0])  # shape (nocc, nvir) = (1,1)

print("\nFor H2/STO-3G with nocc=nvir=1:")
print("  X_gs[0]:", float(X_gs_np[0]))

# The current Delta has: Δ₁_0 = ω_x * X + other_terms ≈ ω_x * X for the first term
# If we flip the sign: Δ₁_corrected_0 = -ω_x * X + other_terms

print("\nComparing with and without sign flip on omega_x:")
for ia in range(mol.natm):
    for ix in range(3):
        dir_name = ['X', 'Y', 'Z'][ix]
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())  # (1,1)
        
        omega_x_val = float(omega_grad_z[ia, ix])
        current_delta0 = float(delta_ia_ix[0])
        expected_from_omega = omega_x_val * float(X_gs_np[0])  # ω_x * X
        
        print("  Atom %d Dir '%s': Δ₁=%-15.6e, ω_x*X=%-15.6e" 
              % (ia, dir_name, current_delta0, expected_from_omega))

# For TDA with B=0 and Y=0: the correct RHS should come from differentiating A*X = ω*X
#   d(A-ωI)/dR * X + (A-ωI) * X^x = 0
#   (A-ωI)*X^x = -d(ε_a - ε_i)_x * X

# But wait — I need to check if the other terms in Eq. 24 also contribute.
# For H2/STO-3G with nocc=nvir=1, the only non-zero components are at [0,0]:
#   A[0,0] = ε_1 - ε_0 + (11|11) = ω

# So dA_x = dε_1^x - dε_0^x + d((11|11))/dx
# And: Δ₁_corrected = -(d(ε_a - ε_i)_x)*X = -(dε_1^x - dε_0^x + d((11|11))/dx)*X

# But ω_x = dω/dR = d(ε_1-ε_0+(11|11))/dR = same as dA_x
# So: -(dA_x)*X = -ω_x*X... wait, that's the SAME sign issue!

# Actually no — I was confusing myself. Let me redo this more carefully.
# 
# For TDA with B=0 and Y=0: A*X = ω*X where ω = ε_1-ε_0+(11|11)
# Differentiating: dA_x * X + (A-ωI)*X^x - dω/dR * X = 0
# But since A-ωI is applied to X and gives zero, we have:
#   (A-ωI)*X^x = -(dA_x - dω/dR)*X
# 
# Since ω = ε_1-ε_0+(11|11) for TDA with B=0: dω/dR = d(ε_1-ε_0+(11|11))/dR
# And dA_x = d(ε_1-ε_0+(11|11))/dx (same thing, different direction)
# So: Δ₁_corrected = -(dA_x - ω_x)*X

# For H2/STO-3G TDA with nocc=nvir=1: at component [0,0] where A[0,0]=ω:
#   0 * X^x_0 = Δ₁_0 = -(dA_x - ω_x)*X_0

# Now the question is: what does Eq. 24 give for Δ₁?
# From my earlier analysis: Δ₁ = ω_x*X + (other terms from Fock perturbations)
# And the other terms give approximately d((11|11))/dx * X for H2/STO-3G

# So if we flip the sign of omega_x in Eq. 24:
#   Δ₁_corrected = -ω_x*X + (other terms from Fock perturbations)
#                ≈ -ω_x*X + d((11|11))/dx * X
# For H2/STO-3G with nocc=nvir=1, this should approximately cancel to give Δ₁_0 ≈ 0

print("\nIf we flip the sign of omega_x in Eq. 24:")
for ia in range(mol.natm):
    for ix in range(3):
        dir_name = ['X', 'Y', 'Z'][ix]
        
        omega_x_val = float(omega_grad_z[ia, ix])
        current_delta0 = float(np.asarray(Delta_cupy[ia, ix].get())[0])
        
        # The "other terms" from Fock perturbation are: other_terms = Delta_0 - omega_x*X
        other_terms = current_delta0 - omega_x_val * float(X_gs_np[0])
        
        if abs(omega_x_val) > 1e-8:
            corrected_delta0 = -omega_x_val * float(X_gs_np[0]) + other_terms
            print("  Atom %d Dir '%s': Δ₁=%.6f, ω_x*X=%.6f, other=%.6f" 
                  % (ia, dir_name, current_delta0, omega_x_val*float(X_gs_np[0]), other_terms))
            print("    Corrected: -ω_x*X + other = %.6e" % corrected_delta0)

EOF
