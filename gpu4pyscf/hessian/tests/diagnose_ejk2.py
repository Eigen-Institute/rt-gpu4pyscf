"""
Compute FD Hessian of omega_grad JK contribution (2*dvhf - 2*dvhf_GS)
and test different ejk_PI formulas.
"""
import numpy as np
import cupy as cp
import pyscf
from functools import reduce
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import _partial_ejk_ip2
from gpu4pyscf.grad import tdrhf as tdrhf_grad, rhf as rhf_grad

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

def get_jk_omega_grad(mol, mf, td, state=0):
    """JK contribution to omega_grad = (2*dvhf from TDA) - (2*dvhf from GS)."""
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]

    x, y = [cp.asarray(v) for v in td.xy[state]]
    xpy = (x + y).reshape(nocc, nvir).T  # (nvir, nocc)
    xmy = (x - y).reshape(nocc, nvir).T  # same for TDA

    dvv = cp.einsum('ai,bi->ab', xpy, xpy) + cp.einsum('ai,bi->ab', xmy, xmy)
    doo = -cp.einsum('ai,aj->ij', xpy, xpy) - cp.einsum('ai,aj->ij', xmy, xmy)
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T)) + reduce(cp.dot, (orbv, dvv, orbv.T))

    from gpu4pyscf.hessian import tdrhf as tdrhf_hess
    td_g = tdrhf_grad.Gradients(td)
    z1_int = cp.asarray(tdrhf_hess.solve_z_vector(td_g, td.xy[state]))
    z1_grad = z1_int.reshape(nocc, nvir).T
    z1ao = reduce(cp.dot, (orbv, z1_grad, orbo.T))
    dmz1doo = z1ao + dmzoo  # = 2*P_I_prime

    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))
    oo0 = orbo @ orbo.T * 2

    td_g2 = tdrhf_grad.Gradients(td)
    # TDA JK gradient: 2*dvhf
    dvhf_tda = td_g2.get_veff(mol, (dmz1doo + dmz1doo.T)*0.5 + oo0, hermi=1)
    dvhf_tda -= td_g2.get_veff(mol, (dmz1doo + dmz1doo.T)*0.5, hermi=1)
    dvhf_tda += 2 * td_g2.get_veff(mol, (dmxpy + dmxpy.T))
    dvhf_tda -= 2 * td_g2.get_veff(mol, (dmxmy - dmxmy.T), 0.0, 1.0, hermi=2)
    dvhf_tda = dvhf_tda * 2  # factor 2 from `de += ... 2*dvhf ...`

    # GS JK gradient: GS uses dm0 = oo0 = 2*P_GS
    dvhf_gs = td_g2.get_veff(mol, oo0, hermi=1) * 2  # factor 2

    return np.asarray(dvhf_tda) - np.asarray(dvhf_gs)

coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
mol0, mf0, td0 = build_system(coords0)

jk_omega_grad0 = get_jk_omega_grad(mol0, mf0, td0)
print("JK omega_grad contribution:", jk_omega_grad0)

# FD Hessian
results = {}
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0.copy(); c[ia, ix] += d
            mol_d, mf_d, td_d = build_system(c)
            results[(ia, ix, d > 0)] = get_jk_omega_grad(mol_d, mf_d, td_d)

h_ejk_fd = np.zeros((2, 3, 2, 3))
for i0 in range(2):
    for x in range(3):
        fd = (results[(i0, x, True)] - results[(i0, x, False)]) / (2 * FD_DELTA)
        for j0 in range(2):
            for y in range(3):
                h_ejk_fd[j0, y, i0, x] = fd[j0, y]

print("\nFD h_ejk (omega_grad JK) nonzero elements:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                if abs(h_ejk_fd[i,x,j,y]) > 1e-5:
                    print(f"  [{i},{x},{j},{y}] = {h_ejk_fd[i,x,j,y]:.8f}")

# Expected: h_semi - h_hcore_correct - h_ovlp_correct
print("\nExpected (from semi - hcore_fix - ovlp):")
# h_semi from FD of omega_grad; h_hcore_correct from generator(2P_I'); h_ovlp = -0.04689
# For [0,0,0,0]: 0.43827 - (-0.99785) - (-0.04689) = 1.48301
print("  [0,0,0,0] should = 0.43827 + 0.99785 + 0.04689 = 1.48301")

# Now test formulas
from gpu4pyscf.grad import tdrhf as tdrhf_grad0
x_y0 = td0.xy[0]
h_obj0 = tdrhf_hess.Hessian(td0)
td_g0 = tdrhf_grad.Gradients(td0)
z1_0 = tdrhf_hess.solve_z_vector(td_g0, x_y0)
ints0 = tdrhf_hess.make_intermediates(h_obj0, x_y0, z1_0)
P_I_prime0 = ints0['P_I_prime']; P_GS0 = ints0['P']; R_I0 = ints0['R_I']; T_I0 = ints0['T_I']
vhfopt = mf0._opt_gpu.get(mol0.omega)

def sym(h):
    h = h.get().transpose(0,2,1,3)
    return 0.5*(h + h.transpose(2,3,0,1))

def try_formula(label, ejk):
    es = sym(ejk)
    print(f"\n{label}:")
    for i in range(2):
        for j in range(2):
            for x in range(3):
                for y in range(3):
                    vfd = h_ejk_fd[i,x,j,y]
                    v = es[i,x,j,y]
                    if abs(vfd) > 1e-5 or abs(v) > 1e-5:
                        r = v/vfd if abs(vfd) > 1e-8 else float('nan')
                        print(f"  [{i},{x},{j},{y}] analyt={v:.6f}  fd={vfd:.6f}  ratio={r:.4f}")

# Formula 0: current
e0 = _partial_ejk_ip2(mol0, P_I_prime0 + P_GS0, vhfopt) - _partial_ejk_ip2(mol0, P_GS0, vhfopt)
e0 += 0.5 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
e0 -= 0.5 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)
try_formula("Formula 0 (current)", e0)

# Formula 1: 2*PI cross, 4x R_I/T_I
e1 = 2 * (_partial_ejk_ip2(mol0, P_I_prime0 + P_GS0, vhfopt) - _partial_ejk_ip2(mol0, P_GS0, vhfopt))
e1 += 4 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
e1 -= 4 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)
try_formula("Formula 1: 2*(PI cross), 4x RI/TI", e1)

# Formula 2: subtract GS self-interaction
e2 = 2 * (_partial_ejk_ip2(mol0, P_I_prime0 + P_GS0, vhfopt) - _partial_ejk_ip2(mol0, P_GS0, vhfopt))
e2 -= _partial_ejk_ip2(mol0, P_GS0, vhfopt)  # subtract GS self
e2 += 4 * _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)
e2 -= 4 * _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)
try_formula("Formula 2: 2*cross - GS + 4x RI/TI", e2)

# Print raw values for each term
print("\nRaw term values (sym, [0,0,0,0] and [0,2,0,2]):")
terms = [
    ("ejk(PI'+PGS) - ejk(PGS)", _partial_ejk_ip2(mol0, P_I_prime0+P_GS0, vhfopt) - _partial_ejk_ip2(mol0, P_GS0, vhfopt)),
    ("ejk(PGS)", _partial_ejk_ip2(mol0, P_GS0, vhfopt)),
    ("ejk(PI')", _partial_ejk_ip2(mol0, P_I_prime0, vhfopt)),
    ("ejk(RI+RI.T)", _partial_ejk_ip2(mol0, R_I0 + R_I0.T, vhfopt)),
    ("ejk(TI-TI.T, j=0)", _partial_ejk_ip2(mol0, T_I0 - T_I0.T, vhfopt, j_factor=0.0)),
]
for lbl, t in terms:
    ts = sym(t)
    print(f"  {lbl}: [0,0,0,0]={ts[0,0,0,0]:.6f}  [0,2,0,2]={ts[0,2,0,2]:.6f}")

print(f"\n  FD ejk: [0,0,0,0]={h_ejk_fd[0,0,0,0]:.6f}  [0,2,0,2]={h_ejk_fd[0,2,0,2]:.6f}")
