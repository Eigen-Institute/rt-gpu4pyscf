"""
Extract im0 directly from the gradient code by monkey-patching contract_h1e_dm,
then compute W_I = im0 - dme0_gs and verify e1_ovlp against FD.
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import get_ovlp
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,0.74*1.8897259886]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='Angstrom', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

mol0, mf0, td0 = build_system()

# ── Extract im0 from the gradient code by intercepting contract_h1e_dm ──
captured_im0 = {}

import gpu4pyscf.grad.rhf as rhf_grad_mod
_orig_contract = rhf_grad_mod.contract_h1e_dm

def _capturing_contract(mol, h1, dm, hermi=0):
    result = _orig_contract(mol, h1, dm, hermi=hermi)
    # Detect the overlap call: h1 is s1 (anti-symmetric structure)
    if dm.shape == (mol.nao, mol.nao) and not captured_im0.get('done'):
        try:
            # s1 from get_ovlp returns (3,nao,nao) of bra derivatives
            # We want to capture dm when it's im0 (the TDA call)
            # The TDA gradient calls contract_h1e_dm(mol, s1, im0, hermi=0)
            # We track all calls and pick the one with hermi=0 that uses im0
            captured_im0.setdefault('calls', []).append(
                (h1.shape if hasattr(h1,'shape') else None,
                 cp.asarray(dm).get() if hasattr(dm,'get') else np.array(dm),
                 hermi))
        except Exception:
            pass
    return result

rhf_grad_mod.contract_h1e_dm = _capturing_contract

# Actually, let's use a simpler approach: temporarily override make_rdm1 in the gradient
# and capture im0 directly by patching the gradient's kernel function.

# Restore original
rhf_grad_mod.contract_h1e_dm = _orig_contract

# ── Better approach: use the gradient's internal function directly ──
# Run the gradient and capture im0 by patching at a lower level.
# The gradient code calls: ds = rhf_grad.contract_h1e_dm(mol, s1, im0, hermi=0)
# We can get im0 by running gradient's solve + intermediates.

def get_im0_from_grad(mol, mf, td, state=0):
    """Run the gradient kernel and return im0 (TDA energy-weighted density in AO)."""
    from gpu4pyscf.grad import tdrhf as tdrhf_grad_mod
    import gpu4pyscf.grad.rhf as rhf_grad_mod2

    _captured = {}
    _orig = rhf_grad_mod2.contract_h1e_dm

    def _intercept(mol_in, h1, dm, hermi=0):
        # The overlap call has hermi=0 and h1 = s1 (antisymmetric-ish)
        # We capture all hermi=0 calls - the last one with the full im0 shape
        res = _orig(mol_in, h1, dm, hermi=hermi)
        if hermi == 0 and hasattr(h1, 'shape') and h1.shape[0] == 3:
            _captured['im0'] = cp.asarray(dm).get()
            _captured['h1'] = h1
        return res

    rhf_grad_mod2.contract_h1e_dm = _intercept
    td_g = tdrhf_grad_mod.Gradients(td)
    td_g.kernel(state=state+1)
    rhf_grad_mod2.contract_h1e_dm = _orig

    if 'im0' not in _captured:
        raise RuntimeError("im0 not captured — check interception logic")
    return cp.asarray(_captured['im0'])

im0_from_grad = get_im0_from_grad(mol0, mf0, td0)
print("im0 from gradient (AO basis):")
print(im0_from_grad.get())

mo_coeff = cp.asarray(mf0.mo_coeff)
mo_occ   = cp.asarray(mf0.mo_occ)
mo_energy = cp.asarray(mf0.mo_energy)
nocc = int((mo_occ > 0).sum())
orbo = mo_coeff[:, :nocc]

dme0_gs = orbo @ cp.diag(mo_energy[:nocc] * 2) @ orbo.T
W_I_correct = im0_from_grad - dme0_gs
print("\ndme0_gs:")
print(dme0_gs.get())
print("\nW_I_correct = im0 - dme0_gs:")
print(W_I_correct.get())

# ── Compare with current W_I from make_intermediates ──
td_g0 = tdrhf_grad.Gradients(td0)
z1_hess = tdrhf_hess.solve_z_vector(td_g0, td0.xy[0])
h_obj0 = tdrhf_hess.Hessian(td0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, td0.xy[0], z1_hess)
W_I_current = ints0['W_I']
print("\nW_I from make_intermediates (current):")
print(W_I_current.get())

print(f"\nMax |W_correct - W_current|: {float(cp.abs(W_I_correct - W_I_current).max()):.6f}")

# ── Compute e1_ovlp with each W and compare to FD ──
s1aa, s1ab, _ = get_ovlp(mol0)
s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
natm = mol0.natm; aoslices = mol0.aoslice_by_atom()

def compute_e1_ovlp(W):
    e1 = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        e1[i0, i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W[p0:p1]) * 2
        for j0 in range(i0+1):
            q0, q1 = aoslices[j0][2:]
            e1[i0, j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W[p0:p1,q0:q1]) * 2
            e1[j0, i0] = e1[i0, j0].T
    return e1

def sym_h(h):
    h2 = h.get().transpose(0,2,1,3)
    return 0.5*(h2 + h2.transpose(2,3,0,1))

e1_correct = sym_h(compute_e1_ovlp(W_I_correct))
e1_current = sym_h(compute_e1_ovlp(W_I_current))

# ── FD of just the overlap contribution ──
def get_ovlp_contribution(mol, mf, td, state=0):
    """Compute omega_grad - hcore_grad - jk_grad = ovlp contribution."""
    from gpu4pyscf.grad import tdrhf as tg, rhf as rg
    from gpu4pyscf.hessian import tdrhf as th

    td_g = tg.Gradients(td)
    de_tda = td_g.kernel(state=state+1)
    mf_g = rg.Gradients(mf)
    de_gs = mf_g.kernel()
    omega_grad = de_tda - de_gs

    # Extract hcore contribution: dh_td + dh1e_td
    # We do this by computing separately
    import gpu4pyscf.grad.rhf as rg2
    _captured2 = {}
    _orig2 = rg2.contract_h1e_dm

    _calls = []
    def _track(mol_in, h1, dm, hermi=0):
        res = _orig2(mol_in, h1, dm, hermi=hermi)
        _calls.append((h1.shape if hasattr(h1,'shape') else None, hermi, res.copy()))
        return res

    # Actually just compute analytically via known formulas
    # The simplest approach: use make_intermediates P_I_prime for hcore and JK
    from gpu4pyscf.hessian.rhf import _partial_ejk_ip2
    z1 = th.solve_z_vector(td_g, td.xy[state])
    h_obj = th.Hessian(td)
    ints = th.make_intermediates(h_obj, td.xy[state], z1)

    # But these are the reference-geometry intermediates. For FD we need them at displaced geometry.
    # For the FD, just use: ovlp = omega - hcore - jk directly
    # But we don't have hcore/jk split at displaced geometry...

    # Use the simplest approach: return omega_grad (we'll extract ovlp by subtraction at the hessian level)
    return np.array(omega_grad)

# Direct FD of omega_grad components is complex. Instead, let's use a cleaner approach:
# FD of the overlap term in omega_grad directly.
def get_omega_ovlp_grad(mol, mf, td, state=0):
    """Compute the overlap part of omega_grad directly from gradient internals."""
    from gpu4pyscf.grad import tdrhf as tg_mod, rhf as rg_mod
    import gpu4pyscf.grad.rhf as rg_mod2

    _captured3 = {'tda_ovlp': None, 'gs_ovlp': None}
    _orig3 = rg_mod2.contract_h1e_dm
    _call_count = [0]

    def _intercept3(mol_in, h1, dm, hermi=0):
        res = _orig3(mol_in, h1, dm, hermi=hermi)
        if hermi == 0 and hasattr(h1, 'shape') and h1.shape[0] == 3:
            _captured3['tda_ovlp'] = np.array(res)
        return res

    rg_mod2.contract_h1e_dm = _intercept3
    td_g2 = tg_mod.Gradients(td)
    td_g2.kernel(state=state+1)

    # Capture GS overlap
    _captured_gs = {}
    def _intercept_gs(mol_in, h1, dm, hermi=0):
        res = _orig3(mol_in, h1, dm, hermi=hermi)
        if hermi == 1 and hasattr(h1, 'shape') and h1.shape[0] == 3:
            _captured_gs['gs_ovlp'] = np.array(res)
        return res

    rg_mod2.contract_h1e_dm = _intercept_gs
    mf_g2 = rg_mod2.Gradients(mf)
    mf_g2.kernel()
    rg_mod2.contract_h1e_dm = _orig3

    tda_ds = _captured3['tda_ovlp']
    gs_ds  = _captured_gs.get('gs_ovlp')
    if tda_ds is None: raise RuntimeError("TDA overlap not captured")
    if gs_ds is None: raise RuntimeError("GS overlap not captured")
    return -tda_ds + gs_ds  # omega overlap contribution

print("\n=== FD of overlap contribution ===")
coords0 = mol0.atom_coords().copy()
results_ovlp = {}
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0.copy(); c[ia, ix] += d
            mol_d, mf_d, td_d = build_system(c)
            results_ovlp[(ia, ix, d > 0)] = get_omega_ovlp_grad(mol_d, mf_d, td_d)

h_ovlp_fd = np.zeros((2, 3, 2, 3))
for i0 in range(2):
    for x in range(3):
        fd = (results_ovlp[(i0,x,True)] - results_ovlp[(i0,x,False)]) / (2*FD_DELTA)
        for j0 in range(2):
            for y in range(3):
                h_ovlp_fd[j0,y,i0,x] = fd[j0,y]

print("\nFD ovlp nonzero elements:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                if abs(h_ovlp_fd[i,x,j,y]) > 1e-4:
                    print(f"  [{i},{x},{j},{y}] = {h_ovlp_fd[i,x,j,y]:.6f}")

print("\nComparison at nonzero elements:")
print(f"{'Element':12s} {'FD':>12s} {'W_correct':>12s} {'W_current':>12s}")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                vfd = h_ovlp_fd[i,x,j,y]
                vc = e1_correct[i,x,j,y]
                vcur = e1_current[i,x,j,y]
                if abs(vfd) > 1e-4 or abs(vc) > 1e-4 or abs(vcur) > 1e-4:
                    print(f"[{i},{x},{j},{y}]     {vfd:12.6f} {vc:12.6f} {vcur:12.6f}")
