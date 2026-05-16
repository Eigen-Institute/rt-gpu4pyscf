"""
Check whether _e_hcore_generator applied to P_I_prime gives the correct
static contribution Tr(P_I_prime * H^{xy}).

The FD in diagnose_term_by_term.py mixes in the response term
Tr(dP_I_prime/dR * H^x). This script separates the two:
  FD_total = static + response
  static   = Tr(P_I_prime_ref * H^{xy}_ref)     <- what generator computes
  response = Tr(dP_I_prime/dX * H^x_ref)        <- FD with frozen H^{xy}

Also computes static directly from raw 2nd-derivative integrals (no generator)
to verify the generator is internally consistent.
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.lib.cupy_helper import contract

FD_DELTA = 2e-3

def build_system(atom_coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if atom_coords is None:
        atom_coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(atom_coords, unit='Bohr')
    mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()
    return mol, mf, td

def get_P_I_prime(mol, mf, td, state=0):
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    x_y = td.xy[state]
    h_obj = tdrhf_hess.Hessian(td)
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y)
    ints = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
    return ints['P_I_prime'].get()  # numpy

coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
mol0, mf0, td0 = build_system(coords0)

P_ref = get_P_I_prime(mol0, mf0, td0)

# ── 1. Static contribution from raw integrals ──────────────────────────────
# h^{xy}[A,B,x,y,p,q] = d²H_core_{pq}/dR_A^x dR_B^y
# For A=B=H0, x=y=0 (X,X): the same-atom block uses h1aa and h1ab
# We want: Tr(P_ref * H^{XX}_{H0,H0})
mol = mol0
nao = mol.nao
aoslices = mol.aoslice_by_atom()
p0, p1 = aoslices[0][2:]   # H0 AO indices

# Get raw 2nd derivative integrals (same as get_hcore in rhf.py)
h1aa = mol.intor('int1e_ipipkin', comp=9).reshape(3,3,nao,nao)
h1aa += mol.intor('int1e_ipipnuc', comp=9).reshape(3,3,nao,nao)
h1ab = mol.intor('int1e_ipkinip', comp=9).reshape(3,3,nao,nao)
h1ab += mol.intor('int1e_ipnucip', comp=9).reshape(3,3,nao,nao)

# diagonal block (H0,H0), x=0,y=0
de_aa = np.einsum('pq,pq->', h1aa[0,0,p0:p1], P_ref[p0:p1])   # <d²mu/dr²|h|nu>
de_ab = np.einsum('pq,pq->', h1ab[0,0,p0:p1,p0:p1], P_ref[p0:p1,p0:p1])  # cross-Pulay same atom
static_pulay = 2.0 * (de_aa + de_ab)   # factor 2 for bra/ket

# nuclear-electron part from hess_nuc_elec
P_ref_cp = cp.asarray(P_ref)
from gpu4pyscf.hessian.rhf import hess_nuc_elec
de_ne = hess_nuc_elec(mol, P_ref_cp)  # (3,3,natm,natm)
static_nuc_elec = float(de_ne[0, 0, 0, 0])  # X,X for H0,H0

static_total = static_pulay + static_nuc_elec
print(f"Static Tr(P_I_prime * H^{{XX}}_{{H0,H0}}) via raw integrals:")
print(f"  Pulay part (2*de_aa+2*de_ab): {static_pulay:.8f}")
print(f"  NucElec part:                 {static_nuc_elec:.8f}")
print(f"  TOTAL static:                 {static_total:.8f}")

# ── 2. Via _e_hcore_generator ─────────────────────────────────────────────
h_obj0 = tdrhf_hess.Hessian(td0)
from gpu4pyscf.hessian.rhf import _e_hcore_generator
de_hcore_fn = _e_hcore_generator(h_obj0, P_ref_cp)
generator_result = float(de_hcore_fn(0, 0)[0, 0])   # [X,X] for H0,H0
print(f"\n_e_hcore_generator result [H0,H0,X,X]: {generator_result:.8f}")
print(f"Ratio generator/static_raw:            {generator_result/static_total:.6f}")

# ── 3. FD of grad_hcore, frozen P_I_prime (static only) ──────────────────
# Compute H^x at displaced geometries, contract with P_I_prime at reference
from gpu4pyscf.hessian.tdrhf import _get_h1ao_x

h1_plus = None; h1_minus = None
for sign, d in [(+1, FD_DELTA), (-1, -FD_DELTA)]:
    c = coords0.copy(); c[0, 0] += d
    mol_d = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol_d.set_geom_(c, unit='Bohr'); mol_d.build()
    h1x_d = _get_h1ao_x(mol_d)   # (natm, 3, nao, nao)
    grad_h0x = float(cp.einsum('pq,pq->', P_ref_cp, h1x_d[0, 0]))   # H0, X direction
    if sign > 0: h1_plus = grad_h0x
    else: h1_minus = grad_h0x

fd_frozen_P = (h1_plus - h1_minus) / (2 * FD_DELTA)
print(f"\nFD with FROZEN P_I_prime (static only):  {fd_frozen_P:.8f}")

# ── 4. FD of grad_hcore, full (P_I_prime also updated) ───────────────────
results_full = []
for d in [FD_DELTA, -FD_DELTA]:
    c = coords0.copy(); c[0, 0] += d
    mol_d, mf_d, td_d = build_system(c)
    P_d = get_P_I_prime(mol_d, mf_d, td_d)
    h1x_d = _get_h1ao_x(mol_d)
    grad_h0x = float(np.einsum('pq,pq->', P_d, h1x_d[0, 0].get()))
    results_full.append(grad_h0x)

fd_full = (results_full[0] - results_full[1]) / (2 * FD_DELTA)
print(f"FD with FULL P_I_prime update (total):   {fd_full:.8f}")
print(f"Response contribution (total - frozen):  {fd_full - fd_frozen_P:.8f}")
print(f"\nSummary:")
print(f"  Generator  = {generator_result:.8f}  (should equal static)")
print(f"  FD frozen  = {fd_frozen_P:.8f}  (static, P fixed)")
print(f"  FD full    = {fd_full:.8f}  (static + response)")
print(f"  Gen/frozen = {generator_result/fd_frozen_P:.6f}  (should be 1.000 if correct)")
print(f"  Gen/full   = {generator_result/fd_full:.6f}  (was 1.971 in previous diag)")
