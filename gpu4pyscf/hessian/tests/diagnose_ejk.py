"""
FD Hessian of just the JK gradient contribution (dvhf) from the TDA gradient.
Compare against the current ejk_PI formula in analytical_omega_hessian.
"""
import numpy as np
import cupy as cp
import pyscf
from functools import reduce
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import _partial_ejk_ip2
from gpu4pyscf.grad import tdrhf as tdrhf_grad

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

def get_jk_grad_contribution(mol, mf, td, state=0):
    """JK gradient contribution (dvhf) from TDA gradient."""
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]

    x, y = [cp.asarray(v) for v in td.xy[state]]
    xpy = (x + y).reshape(nocc, nvir).T  # (nvir, nocc); for TDA = X^T
    xmy = (x - y).reshape(nocc, nvir).T  # same for TDA

    dvv = cp.einsum('ai,bi->ab', xpy, xpy) + cp.einsum('ai,bi->ab', xmy, xmy)  # 2*X^T X
    doo = -cp.einsum('ai,aj->ij', xpy, xpy) - cp.einsum('ai,aj->ij', xmy, xmy)  # -2*X X^T
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T)) + reduce(cp.dot, (orbv, dvv, orbv.T))  # 2*P_I

    from gpu4pyscf.hessian import tdrhf as tdrhf_hess
    td_g = tdrhf_grad.Gradients(td)
    z1_int = cp.asarray(tdrhf_hess.solve_z_vector(td_g, td.xy[state]))  # (nocc, nvir)
    z1_grad = z1_int.reshape(nocc, nvir).T  # (nvir, nocc)
    z1ao = reduce(cp.dot, (orbv, z1_grad, orbo.T))
    dmz1doo = z1ao + dmzoo  # (dmz1doo+dmz1doo.T)/2 = 2*P_I_prime

    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))   # X in AO = R_I
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))   # same for TDA = T_I
    oo0 = orbo @ orbo.T * 2  # ground state

    # dvhf (JK gradient per atom)
    td_g2 = tdrhf_grad.Gradients(td)
    dvhf = td_g2.get_veff(mol, (dmz1doo + dmz1doo.T)*0.5 + oo0, hermi=1)
    dvhf -= td_g2.get_veff(mol, (dmz1doo + dmz1doo.T)*0.5, hermi=1)
    dvhf += 2 * td_g2.get_veff(mol, (dmxpy + dmxpy.T))
    dvhf -= 2 * td_g2.get_veff(mol, (dmxmy - dmxmy.T), 0.0, 1.0, hermi=2)

    return np.asarray(dvhf)

coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
mol0, mf0, td0 = build_system(coords0)

jk_grad0 = get_jk_grad_contribution(mol0, mf0, td0)
print("JK gradient contribution at ref:", jk_grad0)

# FD Hessian of JK gradient
results = {}
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0.copy(); c[ia, ix] += d
            mol_d, mf_d, td_d = build_system(c)
            g_d = get_jk_grad_contribution(mol_d, mf_d, td_d)
            results[(ia, ix, d > 0)] = g_d

h_ejk_fd = np.zeros((2, 3, 2, 3))
for i0 in range(2):
    for x in range(3):
        g_plus = results[(i0, x, True)]
        g_minus = results[(i0, x, False)]
        fd = (g_plus - g_minus) / (2 * FD_DELTA)
        for j0 in range(2):
            for y in range(3):
                h_ejk_fd[j0, y, i0, x] = fd[j0, y]

print("\nFD h_ejk nonzero elements:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                if abs(h_ejk_fd[i,x,j,y]) > 1e-5:
                    print(f"  [{i},{x},{j},{y}] = {h_ejk_fd[i,x,j,y]:.8f}")

# Compare against current ejk_PI formula
from gpu4pyscf.grad import tdrhf as tdrhf_grad0
x_y0 = td0.xy[0]
h_obj0 = tdrhf_hess.Hessian(td0)
td_g0 = tdrhf_grad.Gradients(td0)
z1_0 = tdrhf_hess.solve_z_vector(td_g0, x_y0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, x_y0, z1_0)
P_I_prime0 = ints0['P_I_prime']
P_GS0 = ints0['P']
R_I0 = ints0['R_I']
T_I0 = ints0['T_I']

vhfopt = mf0._opt_gpu.get(mol0.omega)
ejk_cur = _partial_ejk_ip2(mol0, P_I_prime0 + P_GS0, vhfopt)
ejk_cur -= _partial_ejk_ip2(mol0, P_GS0, vhfopt)
ejk_cur += 0.5 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
ejk_cur -= 0.5 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)

# Also try with 2*P_I_prime
ejk_2p = _partial_ejk_ip2(mol0, 2*P_I_prime0 + P_GS0, vhfopt)
ejk_2p -= _partial_ejk_ip2(mol0, P_GS0, vhfopt)
ejk_2p += 0.5 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
ejk_2p -= 0.5 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)

# And try: _partial_ejk_ip2(P_GS + 2*P_I) - _partial_ejk_ip2(2*P_I)
ejk_alt = _partial_ejk_ip2(mol0, P_GS0 + 2*P_I_prime0, vhfopt)
ejk_alt -= _partial_ejk_ip2(mol0, 2*P_I_prime0, vhfopt)
ejk_alt += 0.5 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
ejk_alt -= 0.5 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)

print("\nComparison with analytical formulas (symmetrize for H0-X, H0-X = [0,0,0,0]):")
print("(after transpose(0,2,1,3) and symmetrize)")
# Convert (natm,natm,3,3) → (natm,3,natm,3) and symmetrize
def sym(h):
    h = h.get().transpose(0,2,1,3)
    return 0.5*(h + h.transpose(2,3,0,1))

ec = sym(ejk_cur); e2p = sym(ejk_2p); ea = sym(ejk_alt)
print(f"{'Element':12s} {'FD':>12s} {'cur':>12s} {'cur/fd':>8s}  {'2P':>12s} {'2P/fd':>8s}  {'alt':>12s} {'alt/fd':>8s}")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                vfd = h_ejk_fd[i,x,j,y]
                vc = ec[i,x,j,y]; v2p = e2p[i,x,j,y]; va = ea[i,x,j,y]
                if abs(vfd) > 1e-5 or abs(vc) > 1e-5:
                    rc = vc/vfd if abs(vfd) > 1e-8 else float('nan')
                    r2p = v2p/vfd if abs(vfd) > 1e-8 else float('nan')
                    ra = va/vfd if abs(vfd) > 1e-8 else float('nan')
                    print(f"[{i},{x},{j},{y}]     {vfd:12.6f} {vc:12.6f} {rc:8.4f}  {v2p:12.6f} {r2p:8.4f}  {va:12.6f} {ra:8.4f}")
