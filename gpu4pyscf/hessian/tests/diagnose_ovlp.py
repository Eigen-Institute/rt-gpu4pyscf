"""
Diagnose the e1_ovlp term: check if W_I = im0_TDA - dme0_GS.
The gradient overlap contribution is -Tr(im0_TDA * S^x) (TDA)
                             and     +Tr(dme0_GS * S^x) (GS subtract)
So for omega Hessian, the static overlap term needs W = im0_TDA - dme0_GS.
"""
import numpy as np
import cupy as cp
from functools import reduce
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import get_ovlp, _e_hcore_generator
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.scf import cphf
from opt_einsum import contract

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

mol0, mf0, td0 = build_system()

def compute_im0_tda(mol, mf, td, state=0):
    """Reconstruct im0 from gradient code (in AO basis)."""
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]

    x, y = [cp.asarray(v) for v in td.xy[state]]
    # Gradient uses xpy shape (nvir, nocc)
    xpy_g = (x + y).reshape(nocc, nvir).T  # (nvir, nocc)
    xmy_g = (x - y).reshape(nocc, nvir).T  # same for TDA
    # Hessian uses xpy shape (nocc, nvir)
    xpy_h = xpy_g.T  # (nocc, nvir) = X_h

    doo = -xpy_g.T @ xpy_g - xmy_g.T @ xmy_g  # (nocc, nocc); doo = -2*X_h@X_h.T for TDA
    dvv = xpy_g @ xpy_g.T + xmy_g @ xmy_g.T   # (nvir, nvir); dvv = 2*X_h.T@X_h for TDA
    dmzoo = orbo @ doo @ orbo.T + orbv @ dvv @ orbv.T  # 2*P_I in AO

    R_I = orbo @ xpy_h @ orbv.T  # (nao, nao); R_I = orbo @ X_h @ orbv.T

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=1)
    vj1, vk1 = mf.get_jk(mol, R_I + R_I.T, hermi=1)
    dmxmy = orbo @ xmy_g.T @ orbv.T  # T_I in AO = R_I for TDA
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    veff0doo = cp.asarray(vj0) * 2 - cp.asarray(vk0)  # G(2*P_I) with factor 2 for alpha+beta
    veff0mop_AO = cp.asarray(vj1) * 2 - cp.asarray(vk1)  # G_singlet(R_I+R_I.T)
    veff0mom_AO = -cp.asarray(vk2)  # K(T_I - T_I.T) in gradient convention

    veff0mop = mo_coeff.T @ veff0mop_AO @ mo_coeff  # Gp_RI in MO
    veff0mom = mo_coeff.T @ veff0mom_AO @ mo_coeff  # K(T_I-T_I.T) in MO

    # Solve z-vector using gradient convention (nvir, nocc)
    td_g = tdrhf_grad.Gradients(td)
    wvo = orbo.T @ veff0doo @ orbv * 2.0  # (nocc, nvir)
    wvo -= contract("ki,ka->ia", veff0mop[:nocc, :nocc], xpy_h) * 2
    wvo += contract("ca,ia->ic", veff0mop[nocc:, nocc:], xpy_h) * 2
    wvo -= contract("ki,ka->ia", veff0mom[:nocc, :nocc], xpy_h) * 2  # TDA: xmy = xpy
    wvo += contract("ca,ia->ic", veff0mom[nocc:, nocc:], xpy_h) * 2

    vresp = mf.gen_response(singlet=None, hermi=1)

    def fvind(x_):
        dm = orbo @ x_.reshape(nocc, nvir) @ orbv.T
        v1ao = vresp(dm + dm.T)
        return (orbo.T @ v1ao @ orbv).ravel()

    z1_nocc_nvir = cphf.solve(fvind, mo_energy, mo_occ, wvo.T,
                              max_cycle=200, tol=1e-10)[0]
    z1_nvir_nocc = z1_nocc_nvir.reshape(nvir, nocc)  # gradient convention
    z1ao = orbv @ z1_nvir_nocc @ orbo.T
    veff_z1 = vresp(z1ao + z1ao.T)
    veff_z1_MO = mo_coeff.T @ veff_z1 @ mo_coeff

    # Build im0 in MO basis (gradient convention)
    im0_MO = cp.zeros((nmo, nmo))
    im0_MO[:nocc, :nocc] = orbo.T @ (veff0doo + veff_z1) @ orbo
    im0_MO[:nocc, :nocc] += contract("ak,ai->ki", veff0mop[nocc:, :nocc], xpy_g)
    im0_MO[:nocc, :nocc] += contract("ak,ai->ki", veff0mom[nocc:, :nocc], xmy_g)
    im0_MO[nocc:, nocc:] = contract("ci,ai->ac", veff0mop[nocc:, :nocc], xpy_g)
    im0_MO[nocc:, nocc:] += contract("ci,ai->ac", veff0mom[nocc:, :nocc], xmy_g)
    im0_MO[nocc:, :nocc] = contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy_g) * 2
    im0_MO[nocc:, :nocc] += contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy_g) * 2

    # Add zeta * dm1 (full, including GS 2*I_oo)
    zeta = (mo_energy[:, cp.newaxis] + mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]
    dm1 = cp.zeros((nmo, nmo))
    dm1[:nocc, :nocc] = doo + cp.eye(nocc) * 2  # full (includes GS)
    dm1[nocc:, nocc:] = dvv
    dm1[nocc:, :nocc] = z1_nvir_nocc

    im0_AO = mo_coeff @ (im0_MO + zeta * dm1) @ mo_coeff.T
    return im0_AO

def compute_dme0_gs(mol, mf):
    """Ground state energy-weighted density: dme0 = 2 * sum_i eps_i |phi_i><phi_i|."""
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    nocc = int((mo_occ > 0).sum())
    orbo = mo_coeff[:, :nocc]
    return orbo @ cp.diag(mo_energy[:nocc] * 2) @ orbo.T  # factor 2 for double occupancy

im0_0 = compute_im0_tda(mol0, mf0, td0)
dme0_0 = compute_dme0_gs(mol0, mf0)
W_needed = im0_0 - dme0_0

print("W_needed (im0_TDA - dme0_GS):")
print(W_needed.get())

# Compare to W_I from make_intermediates
td_g0 = tdrhf_grad.Gradients(td0)
z1_hess = tdrhf_hess.solve_z_vector(td_g0, td0.xy[0])
h_obj0 = tdrhf_hess.Hessian(td0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, td0.xy[0], z1_hess)
W_I0 = ints0['W_I']

print("\nW_I from make_intermediates:")
print(W_I0.get())

print("\nDifference W_needed - W_I:")
diff = W_needed - W_I0
print(diff.get())
print(f"Max |diff|: {float(cp.abs(diff).max()):.8f}")
print(f"Max |diff / W_needed| (where W_needed > 1e-6): {float((cp.abs(diff) / (cp.abs(W_needed) + 1e-10)).max()):.6f}")

# Now compute e1_ovlp with W_needed and compare with FD
def get_ovlp_grad(mol, mf, td, state=0):
    """Compute the overlap contribution to omega_grad."""
    td_g = tdrhf_grad.Gradients(td)
    from gpu4pyscf.grad import rhf as rhf_grad
    mf_grad = mf.nuc_grad_method()
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    im0 = compute_im0_tda(mol, mf, td, state)
    dme0 = compute_dme0_gs(mol, mf)
    # The omega gradient overlap = -Tr((im0_TDA - dme0_GS) * S^x)
    W = im0 - dme0
    de = cp.zeros((mol.natm, 3))
    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        # s1 has shape (3, nao, nao), p0:p1 is the bra slice
        de[ia] += cp.einsum('xpq,pq->', s1[:, p0:p1, :], W[p0:p1, :])
        de[ia] += cp.einsum('xpq,qp->', s1[:, p0:p1, :], W[:, p0:p1])
    de *= -1  # the gradient has -ds
    return de.get()

ovlp_grad0 = get_ovlp_grad(mol0, mf0, td0)
print("\nOmega overlap gradient (analytical from W_needed):")
print(ovlp_grad0)

# FD of omega overlap gradient
results = {}
coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0.copy(); c[ia, ix] += d
            mol_d, mf_d, td_d = build_system(c)
            results[(ia, ix, d > 0)] = get_ovlp_grad(mol_d, mf_d, td_d)

h_ovlp_fd = np.zeros((2, 3, 2, 3))
for i0 in range(2):
    for x in range(3):
        fd = (results[(i0, x, True)] - results[(i0, x, False)]) / (2 * FD_DELTA)
        for j0 in range(2):
            for y in range(3):
                h_ovlp_fd[j0, y, i0, x] = fd[j0, y]

print("\nFD h_ovlp nonzero elements:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                if abs(h_ovlp_fd[i,x,j,y]) > 1e-5:
                    print(f"  [{i},{x},{j},{y}] = {h_ovlp_fd[i,x,j,y]:.8f}")

# Now compute e1_ovlp using W_needed at reference geometry
s1aa, s1ab, _ = get_ovlp(mol0)
s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
natm = mol0.natm
aoslices = mol0.aoslice_by_atom()

def compute_e1_ovlp(W, s1aa, s1ab, natm, aoslices):
    e1 = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        e1[i0, i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W[p0:p1]) * 2
        for j0 in range(i0+1):
            q0, q1 = aoslices[j0][2:]
            e1[i0, j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W[p0:p1,q0:q1]) * 2
            e1[j0, i0] = e1[i0, j0].T
    return e1

e1_ovlp_W_needed = compute_e1_ovlp(W_needed, s1aa, s1ab, natm, aoslices)
e1_ovlp_W_I = compute_e1_ovlp(W_I0, s1aa, s1ab, natm, aoslices)

# Symmetrize: transpose(0,2,1,3), then average with transpose
def sym_h(h):
    h2 = h.get().transpose(0,2,1,3)
    return 0.5*(h2 + h2.transpose(2,3,0,1))

e1_W_needed_sym = sym_h(e1_ovlp_W_needed)
e1_W_I_sym = sym_h(e1_ovlp_W_I)

print("\nComparison of e1_ovlp (after symmetrization):")
print(f"{'Element':12s} {'FD':>12s} {'W_needed':>12s} {'ratio_W':>8s}  {'W_I':>12s} {'ratio_WI':>8s}")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                vfd = h_ovlp_fd[i,x,j,y]
                vw = e1_W_needed_sym[i,x,j,y]
                vi = e1_W_I_sym[i,x,j,y]
                if abs(vfd) > 1e-5 or abs(vw) > 1e-5 or abs(vi) > 1e-5:
                    rw = vw/vfd if abs(vfd) > 1e-8 else float('nan')
                    ri = vi/vfd if abs(vfd) > 1e-8 else float('nan')
                    print(f"[{i},{x},{j},{y}]     {vfd:12.6f} {vw:12.6f} {rw:8.4f}  {vi:12.6f} {ri:8.4f}")
