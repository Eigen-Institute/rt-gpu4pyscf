"""
Verify the hcore gradient formula by comparing analytical vs FD.

If Tr(P_I_prime * H^x) is the correct gradient formula, then FD of
the hcore component of the excited-state energy should match.

The excited-state energy = ground-state energy + omega.
The hcore contribution to d(omega)/dX should come from
  d/dX [Tr((P_GS + P_I_prime) * H_core)] - d/dX [Tr(P_GS * H_core)]
= Tr(P_I_prime * H^x)  [if P_I_prime is the response density]

But in the TDA gradient code, the actual expression might differ.
This script checks what grad/tdrhf.py uses for the hcore contribution.
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.tdrhf import _get_h1ao_x
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.grad import rhf as rhf_grad

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

def get_full_td_gradient(mol, mf, td):
    """Full analytical TDA gradient from gpu4pyscf."""
    td_g = tdrhf_grad.Gradients(td)
    return np.asarray(td_g.kernel(state=1))   # (natm, 3) numpy

def get_hcore_grad_term(mol, mf, td, state=0):
    """Compute Tr(P_I_prime * H^x) for all atoms/dirs."""
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    from gpu4pyscf.hessian import tdrhf as tdrhf_hess
    x_y = td.xy[state]
    h_obj = tdrhf_hess.Hessian(td)
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y)
    ints = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
    P_I_prime = ints['P_I_prime']
    h1ao_x = _get_h1ao_x(mol)  # (natm, 3, nao, nao)
    natm = mol.natm
    grad = cp.zeros((natm, 3))
    for ia in range(natm):
        for ix in range(3):
            grad[ia, ix] = cp.einsum('pq,pq->', P_I_prime, h1ao_x[ia, ix])
    return grad.get()

coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
mol0, mf0, td0 = build_system(coords0)

# Analytical TDA gradient (full)
grad_full = get_full_td_gradient(mol0, mf0, td0)
print("Full TDA gradient (analytical):")
print(grad_full)

# Ground-state gradient (to subtract and get excited-state correction)
gs_grad = np.asarray(rhf_grad.Gradients(mf0).kernel())
print("Ground-state gradient:")
print(gs_grad)
print("Excited - GS:")
print(grad_full - gs_grad)

# H_core contribution Tr(P_I_prime * H^x) for each atom/direction
hcore_grad = get_hcore_grad_term(mol0, mf0, td0)
print("\nTr(P_I_prime * H^x) for each atom/direction:")
print(hcore_grad)

# FD of Tr(P_I_prime * H^x) w.r.t. Z displacement of H0 (H[0,2])
print("\n--- FD vs analytical for H0-Z gradient ---")
results = []
for d in [FD_DELTA, -FD_DELTA]:
    c = coords0.copy(); c[0, 2] += d
    mol_d, mf_d, td_d = build_system(c)
    g_d = get_hcore_grad_term(mol_d, mf_d, td_d)
    results.append(g_d[0, 2])  # H0, Z direction
fd_hcore_Z = (results[0] - results[1]) / (2*FD_DELTA)
print(f"FD d/dZ_H0 [Tr(P_I_prime * H^Z_H0)] = {fd_hcore_Z:.8f}")
print(f"(This should equal h_hcore[H0,Z,H0,Z] = {0:.8f} ???)")

# What does the generator give for (H0,Z,H0,Z)?
from gpu4pyscf.hessian.rhf import _e_hcore_generator
from gpu4pyscf.grad import tdrhf as tdrhf_grad0
x_y0 = td0.xy[0]
h_obj0 = tdrhf_hess.Hessian(td0)
td_g0 = tdrhf_grad.Gradients(td0)
z1_0 = tdrhf_hess.solve_z_vector(td_g0, x_y0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, x_y0, z1_0)
P_I_prime0 = ints0['P_I_prime']
gen = _e_hcore_generator(h_obj0, P_I_prime0)
print(f"Generator h_hcore[H0,Z,H0,Z] = {float(gen(0,0)[2,2]):.8f}")

# Check: what is the h_core contribution in the full gradient?
# Use the grad code directly to extract hcore contribution
print("\n--- Checking what tdrhf gradient uses for hcore ---")
# The full gradient includes: hcore + jk + ovlp + other terms
# Let's check the h1ao_x computation matches the gradient's hcore term
# by computing FD of just the h_core energy Tr(P_total * H_core)
def hcore_energy(mol, mf, td):
    """Tr((P_GS + P_I_prime) * H_core) — hcore part of total energy."""
    from gpu4pyscf.hessian import tdrhf as th
    from gpu4pyscf.grad import tdrhf as tg
    h_obj = th.Hessian(td)
    z1 = th.solve_z_vector(tg.Gradients(td), td.xy[0])
    ints = th.make_intermediates(h_obj, td.xy[0], z1)
    P_I_prime = ints['P_I_prime']
    P_GS = ints['P']  # orbo @ orbo.T
    H_core = cp.asarray(mf.get_hcore())
    return float(cp.trace((P_GS + P_I_prime) @ H_core))

def hcore_energy_gs(mol, mf):
    """Tr(P_GS * H_core)."""
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum())
    orbo = mo_coeff[:, :nocc]
    P_GS = orbo @ orbo.T
    H_core = cp.asarray(mf.get_hcore())
    return float(cp.trace(P_GS @ H_core))

print(f"Hcore energy (total) at ref: {hcore_energy(mol0, mf0, td0):.8f}")
print(f"Hcore energy (GS)    at ref: {hcore_energy_gs(mol0, mf0):.8f}")
print(f"Hcore omega contribution:    {hcore_energy(mol0, mf0, td0) - hcore_energy_gs(mol0, mf0):.8f}")
print(f"Tr(P_I_prime * H_core):     {float(cp.trace(P_I_prime0 @ cp.asarray(mf0.get_hcore()))):.8f}")

# FD of hcore energy w.r.t. Z displacement of H0
results_e = []
for d in [FD_DELTA, -FD_DELTA]:
    c = coords0.copy(); c[0, 2] += d
    mol_d, mf_d, td_d = build_system(c)
    results_e.append(hcore_energy(mol_d, mf_d, td_d))

fd_hcore_energy_Z = (results_e[0] - results_e[1]) / (2*FD_DELTA)
print(f"\nFD d/dZ_H0 [hcore_energy] = {fd_hcore_energy_Z:.8f}")
print(f"hcore_grad[H0, Z] from Tr formula = {hcore_grad[0, 2]:.8f}")
