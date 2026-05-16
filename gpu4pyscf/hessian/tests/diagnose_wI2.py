"""
Determine the correct W_I for e1_ovlp by:
1. Constructing im0 directly from gradient-code formulas (using hessian's z-vector convention)
2. Checking W_I = im0 - dme0_gs against FD of the overlap gradient term.
"""
import numpy as np
import cupy as cp
import pyscf
from functools import reduce
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import get_ovlp
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import cphf

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

def compute_im0_direct(mol, mf, td, state=0):
    """
    Reconstruct im0 exactly as grad/tdrhf.py computes it (lines 62-152).
    Uses the gradient's z-vector convention: z1 shape (nvir, nocc).
    """
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ   = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum()); nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]

    x, y = [cp.asarray(v) for v in td.xy[state]]
    # Gradient convention: xpy is (nvir, nocc)
    xpy = (x + y).reshape(nocc, nvir).T   # (nvir, nocc)
    xmy = xpy  # TDA

    dvv = xpy @ xpy.T + xmy @ xmy.T          # (nvir, nvir)
    doo = -(xpy.T @ xpy + xmy.T @ xmy)       # (nocc, nocc)
    dmzoo = orbo @ doo @ orbo.T + orbv @ dvv @ orbv.T  # (nao, nao), = 2*P_I * 2

    dmxpy = orbv @ xpy @ orbo.T  # R_I in gradient convention
    dmxmy = dmxpy                 # TDA

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=1)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=1)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    print(f"DEBUG GRAD: vk2:\n{vk2}")
    vj = cp.stack([cp.asarray(vj0), cp.asarray(vj1), cp.asarray(vj2)])
    vk = cp.stack([cp.asarray(vk0), cp.asarray(vk1), cp.asarray(vk2)])

    veff0doo = vj[0] * 2 - vk[0]   # gradient uses *2 for spin doubling
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2   # (nvir, nocc)

    veff = vj[1] * 2 - vk[1]   # singlet
    veff0mop = mo_coeff.T @ veff @ mo_coeff
    wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2

    veff = -vk[2]
    veff0mom = mo_coeff.T @ veff @ mo_coeff
    print(f"DEBUG GRAD: veff0mom MO oo: {veff0mom[:nocc, :nocc]}")
    wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    # Gradient's fvind — factor of 2 on dm for double occupancy
    vresp = mf.gen_response(singlet=None, hermi=1)
    def fvind_grad(x_):
        dm = reduce(cp.dot, (orbv, x_.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1_grad = cphf.solve(fvind_grad, mo_energy, mo_occ, wvo,
                         max_cycle=200, tol=1e-12)[0].reshape(nvir, nocc)

    z1ao = reduce(cp.dot, (orbv, z1_grad, orbo.T))
    veff_z1 = vresp(z1ao + z1ao.T)

    # Build im0_MO exactly as in gradient code
    im0_MO = cp.zeros((nmo, nmo))
    im0_MO[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo + veff_z1, orbo))
    im0_MO[:nocc, :nocc] += contract("ak,ai->ki", veff0mop[nocc:, :nocc], xpy)
    im0_MO[:nocc, :nocc] += contract("ak,ai->ki", veff0mom[nocc:, :nocc], xmy)
    im0_MO[nocc:, nocc:]  = contract("ci,ai->ac", veff0mop[nocc:, :nocc], xpy)
    im0_MO[nocc:, nocc:] += contract("ci,ai->ac", veff0mom[nocc:, :nocc], xmy)
    im0_MO[nocc:, :nocc]  = contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    im0_MO[nocc:, :nocc] += contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    print(f"DEBUG GRAD: im0_MO vo: {im0_MO[nocc:, :nocc]}")

    zeta = (mo_energy[:, None] + mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]

    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo + cp.eye(nocc) * 2  # includes GS
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1_grad

    im0_AO = reduce(cp.dot, (mo_coeff, im0_MO + zeta * dm1, mo_coeff.T))
    return im0_AO, z1_grad

im0_grad, z1_grad_ref = compute_im0_direct(mol0, mf0, td0)
dme0_gs = (lambda o: o @ cp.diag(cp.asarray(mf0.mo_energy)[:int((cp.asarray(mf0.mo_occ)>0).sum())] * 2) @ o.T)(cp.asarray(mf0.mo_coeff)[:, :int((cp.asarray(mf0.mo_occ)>0).sum())])
W_I_grad = im0_grad - dme0_gs

print("im0 (from gradient formulas):")
print(im0_grad.get())
print("\ndme0_gs:"); print(dme0_gs.get())
print("\nW_I_grad = im0 - dme0_gs:"); print(W_I_grad.get())

# ── Compare z-vectors ──
td_g0 = tdrhf_grad.Gradients(td0)
z1_hess = tdrhf_hess.solve_z_vector(td_g0, td0.xy[0])
print(f"\nz1 from gradient: {z1_grad_ref.get()}")
print(f"z1 from hessian's solve_z_vector (nocc,nvir): {cp.asarray(z1_hess).get()}")
print(f"z1_hess.T (nvir,nocc): {cp.asarray(z1_hess).T.get()}")
print(f"Ratio hess.T / grad: {(cp.asarray(z1_hess).T / (z1_grad_ref + 1e-30)).get()}")

# ── Compare W_I_grad with current W_I ──
h_obj0 = tdrhf_hess.Hessian(td0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, td0.xy[0], z1_hess)
W_I_current = ints0['W_I']
print(f"\nW_I_current (from make_intermediates): {W_I_current.get()}")
print(f"Max |W_grad - W_current|: {float(cp.abs(W_I_grad - W_I_current).max()):.6f}")

# ── Also compute 2*P_I_prime vs gradient's (dmz1doo + dmz1doo.T)*0.5 ──
mo_coeff = cp.asarray(mf0.mo_coeff); mo_occ = cp.asarray(mf0.mo_occ)
nocc = int((mo_occ>0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
orbo = mo_coeff[:,:nocc]; orbv = mo_coeff[:,nocc:]
x, y = [cp.asarray(v) for v in td0.xy[0]]
xpy_h = x.reshape(nocc, nvir)  # hessian convention (nocc,nvir)
xpy_g = xpy_h.T  # gradient convention (nvir,nocc)
dvv = xpy_g @ xpy_g.T + xpy_g @ xpy_g.T  # = 2*X^T X (gradient)
doo = -(xpy_g.T @ xpy_g + xpy_g.T @ xpy_g)  # = -2*X X^T
dmzoo_g = orbo @ doo @ orbo.T + orbv @ dvv @ orbv.T  # gradient's dmzoo
z1ao_grad = orbv @ z1_grad_ref @ orbo.T
dmz1doo = z1ao_grad + dmzoo_g  # gradient's dmz1doo

P_prime_grad_sym = (dmz1doo + dmz1doo.T) * 0.5
P_I_prime_hess = ints0['P_I_prime']
print(f"\n2*P_I_prime (hessian) vs grad P_prime_sym:")
print(f"  2*P_I_prime: {(2*P_I_prime_hess).get()}")
print(f"  grad P_prime_sym: {P_prime_grad_sym.get()}")
print(f"  Diff: {float(cp.abs(2*P_I_prime_hess - P_prime_grad_sym).max()):.6f}")

# ── Compute e1_ovlp with W_I_grad ──
s1aa, s1ab, _ = get_ovlp(mol0)
s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
natm = mol0.natm; aoslices = mol0.aoslice_by_atom()

def compute_e1_ovlp(W):
    e1 = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        e1[i0,i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W[p0:p1]) * 2
        for j0 in range(i0+1):
            q0, q1 = aoslices[j0][2:]
            e1[i0,j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W[p0:p1,q0:q1]) * 2
            e1[j0,i0] = e1[i0,j0].T
    return e1

def sym_h(h):
    h2 = h.get().transpose(0,2,1,3)
    return 0.5*(h2 + h2.transpose(2,3,0,1))

e1_grad_W = sym_h(compute_e1_ovlp(W_I_grad))
e1_curr_W = sym_h(compute_e1_ovlp(W_I_current))

# ── FD of the overlap term in omega_grad ──
def get_ovlp_term(mol, mf, td, state=0):
    """Return -ds_TDA + ds_GS (overlap contribution to omega_grad)."""
    from gpu4pyscf.grad import tdrhf as tg_mod
    import gpu4pyscf.grad.rhf as rg_mod

    _captured = {}
    _orig = rg_mod.contract_h1e_dm

    def _intercept(mol_in, h1, dm, hermi=0):
        res = _orig(mol_in, h1, dm, hermi=hermi)
        # The s1 call in TDA gradient: hermi=0, h1 shape (3,nao,nao)
        if hermi == 0 and hasattr(h1, 'shape') and len(h1.shape)==3 and h1.shape[0]==3:
            _captured['tda'] = np.array(res)
        return res

    rg_mod.contract_h1e_dm = _intercept
    td_g2 = tg_mod.Gradients(td)
    td_g2.kernel(state=state+1)
    rg_mod.contract_h1e_dm = _orig

    # GS gradient overlap: in rhf.py, the Pulay term is computed differently
    # Simplest: compute omega_grad and subtract known hcore+jk terms analytically
    # For FD purposes, just return the captured TDA overlap term and GS separately
    from gpu4pyscf.grad import rhf as rg_mod2
    _captured2 = {}
    _orig2 = rg_mod2.contract_h1e_dm

    def _intercept2(mol_in, h1, dm, hermi=0):
        res = _orig2(mol_in, h1, dm, hermi=hermi)
        if hermi == 1 and hasattr(h1, 'shape') and len(h1.shape)==3 and h1.shape[0]==3:
            _captured2['gs'] = np.array(res)
        return res

    rg_mod2.contract_h1e_dm = _intercept2
    rg_mod2.Gradients(mf).kernel()
    rg_mod2.contract_h1e_dm = _orig2

    tda_ds = _captured.get('tda')
    gs_ds  = _captured2.get('gs')
    if tda_ds is None: raise RuntimeError("TDA ovlp not captured")
    if gs_ds is None: raise RuntimeError("GS ovlp not captured")
    return -tda_ds + gs_ds

print("\n=== FD of overlap gradient term ===")
coords0_arr = mol0.atom_coords().copy()
results = {}
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0_arr.copy(); c[ia,ix] += d
            md, mfd, tdd = build_system(c)
            results[(ia,ix,d>0)] = get_ovlp_term(md, mfd, tdd)

h_fd = np.zeros((2,3,2,3))
for i0 in range(2):
    for x in range(3):
        fd = (results[(i0,x,True)] - results[(i0,x,False)]) / (2*FD_DELTA)
        for j0 in range(2):
            for y in range(3):
                h_fd[j0,y,i0,x] = fd[j0,y]

print(f"\n{'Element':12s} {'FD':>12s} {'W_grad':>12s} {'W_current':>12s}")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                vfd = h_fd[i,x,j,y]
                vg  = e1_grad_W[i,x,j,y]
                vc  = e1_curr_W[i,x,j,y]
                if abs(vfd)>1e-5 or abs(vg)>1e-5 or abs(vc)>1e-5:
                    print(f"[{i},{x},{j},{y}]     {vfd:12.6f} {vg:12.6f} {vc:12.6f}")
