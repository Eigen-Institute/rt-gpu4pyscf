#!/usr/bin/env python3
"""
Diagnose Delta construction for H2/STO-3G TDA - check individual terms.

For H2/STO-3G with nocc=nvir=1, the only component [0,0] IS the ground state:
  A[0,0] = ω = 0.947423
So (A-ωI)X^x = Δ₁ means we need Δ₁_0 = 0 for consistency.

Let me compute what make_cptddft_rhs gives and compare with the expected value.
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
natm = mol.natm

print("H2: nocc=%d, nvir=%d, omega=%.6f" % (nocc, nvir, omega))

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

# Get the gradient of excitation energy  
from gpu4pyscf.hessian.tdrhf import omega_grad
omega_grad_z = omega_grad(td, state)

print("ω_z (excitation energy gradient in Z):", np.asarray(omega_grad_z[:, 2]))

# For H2: atoms are at z=0 and z=1.4, so omega gradient should be equal/opposite
# This means dω/dR_0,Z = -dω/dR_1,Z

# Now let me manually compute what Δ₁_0 should be for TDA with B=0 and Y=0:
# From differentiating (A-ωI)*X = 0:
#   (A-ωI)X^x + d(ε_a - ε_i)_x * X = 0
# So: Δ₁_0 = -d(ε_a - ε_i)_x * X_0

# For H2/STO-3G with nocc=nvir=1:
#   A[0,0] = (ε_1 - ε_0) + (11|11) = 0.9474... = ω
# So d(ε_1 - ε_0)_x = mo_e1[nocc:, nocc:] derivative for the virtual MO

# Actually, in a diagonal TDA approximation: A[0,0] = ε_1 - ε_0 + (11|11) 
# where ε_0 is the occupied orbital energy and ε_1 is the virtual.
# The perturbation of this gives dA_x = dε_1^x - dε_0^x

print("\nExpected Δ₁ for TDA with B=0, Y=0:")
for ia in range(natm):
    for ix in range(3):
        dir_name = ['X', 'Y', 'Z'][ix]
        
        # The perturbed MO energy derivative for the occupied orbital (nocc=1)
        d_eps_occupied_x = float(mo_e1[ia, ix])  # mo_e1 has shape (natm,3,nao,nocc)
        print("  Atom %d Dir '%s': d(ε_occ)_x = %.6f" % (ia, dir_name, d_eps_occupied_x))

# Actually wait — mo_e1 is from solve_mo1 which gives the perturbed orbital energies
# For nocc=1: mo_e1 has shape (natm,3,nao,nocc) so it's (2,3,2,1) for H2/STO-3G
print("\nmo_e1 shape:", mo1.shape, "(should be natm,3,nao,nocc)")

# For the TDA equation (A-ωI)X^x = -d(ε_a - ε_i)_x * X:
# At the singular component [0,0]: 0*x1_0 = Δ₁_0 where Δ₁_0 = -d(ε_1 - ε_0)_x * X_0

# But wait — I need to be more careful about what mo_e1 represents.
# In PySCF's solve_mo1:
#   mo_e1 is the perturbed orbital energy derivative for each MO direction
# It's used in the CP-HF equations, not directly in TDHF/TDA

# Let me just check what Delta actually gives vs what it should give  
x_y_scaled = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
Delta_cupy, Upsilon_cupy = tdrhf_hess.make_cptddft_rhs(h_obj, x_y_scaled, omega, mo1, mo_e1)

# For H2/STO-3G TDA: X_0 is the excitation amplitude at component [0,0]
X_gs_np = np.asarray(x_y_orig[0])  # shape (nocc, nvir) = (1,1)
print("\nX_gs[0]:", float(X_gs_np[0]))

# For TDA with Y=0 and B=0: Δ₁ should be -d(ε_a-ε_i)_x * X_0 at the [0,0] component
# But make_cptddft_rhs uses a different construction (Eq. 24 from Liu & Liang)

print("\nActual Delta for each perturbation direction:")
for ia in range(natm):
    for ix in range(3):
        dir_name = ['X', 'Y', 'Z'][ix]
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())  # (nocc,nvir) = (1,1)
        
        # For H2/STO-3G TDA with nocc=nvir=1: Delta is just a scalar at [0,0]
        delta_0 = float(delta_ia_ix[0])
        
        print("  Atom %d Dir '%s': Δ₁[0] = %.6e" % (ia, dir_name, delta_0))

# Now let me compute the expected value using a different approach:
# For TDA with B=0 and Y=0, Eq. 24 gives:
#   Δ₁ = ω_x * X - Fock terms + J/K terms
# But since A is diagonal (no off-diagonal coupling), the J/K terms are zero.

print("\n\nFor H2/STO-3G with nocc=nvir=1 and diagonal A:")
print("  The only non-zero term should be from the Fock perturbation.")
print("  At component [0,0] where A[0,0]=ω: we need Δ₁_0 = 0")

# Let me check what omega_x * X gives for each direction  
for ia in range(natm):
    for ix in range(3):
        dir_name = ['X', 'Y', 'Z'][ix]
        delta_ia_ix = np.asarray(Delta_cupy[ia, ix].get())
        omega_x_val = float(omega_grad_z[ia, ix])
        
        # From Eq. 24: first term is ω_x * (X-Y) = ω_x * X for TDA with Y=0
        expected_omega_term = omega_x_val * float(X_gs_np[0])  # ω_x * X
        
        print("  Atom %d Dir '%s': ω_x=%.6f, ω_x*X=%.6e, Δ₁=%-15.6e" 
              % (ia, dir_name, omega_x_val, expected_omega_term, delta_ia_ix[0]))
