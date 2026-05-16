"""
The TDA gradient uses dmz1doo for hcore, not P_I_prime.
This diagnostic:
1. Computes dmz1doo from the gradient code
2. Computes the hcore Hessian using FD of the full hcore gradient term
3. Compares with _e_hcore_generator(P_I_prime) and _e_hcore_generator(dmz1doo_symm)
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.hessian.rhf import _e_hcore_generator
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e

FD_DELTA = 2e-3

def build_system(coords=None):
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    if coords is None: coords = coords0
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mol.set_geom_(coords, unit='Bohr'); mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf); td.nstates = 1; td.kernel()
    return mol, mf, td

def get_hcore_grad_contribution(mol, mf, td, state=0):
    """Get just the hcore part of the TDA gradient (dh_td + dh1e_td)."""
    from gpu4pyscf.grad import tdrhf as tdrhf_grad, rhf as rhf_grad
    from gpu4pyscf.df import int3c2e
    from functools import reduce

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum())
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]

    x, y = [cp.asarray(v) for v in td.xy[state]]
    xpy = (x + y).reshape(nocc, nvir).T  # (nvir, nocc)
    xmy = (x - y).reshape(nocc, nvir).T  # (nvir, nocc) = xpy for TDA

    dvv = cp.einsum('ai,bi->ab', xpy, xpy) + cp.einsum('ai,bi->ab', xmy, xmy)  # 2*X^T X
    doo = -cp.einsum('ai,aj->ij', xpy, xpy) - cp.einsum('ai,aj->ij', xmy, xmy)  # -2*X X^T

    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T)) + reduce(cp.dot, (orbv, dvv, orbv.T))  # 2*P_I in AO

    # z1 in gradient convention (nvir, nocc)
    td_g = tdrhf_grad.Gradients(td)
    # Actually we need to get z1 from the gradient kernel...
    # Use solve_z_vector from hessian code but convert
    from gpu4pyscf.hessian import tdrhf as tdrhf_hess
    z1_int = cp.asarray(tdrhf_hess.solve_z_vector(td_g, td.xy[state]))  # (nocc, nvir)
    # gradient convention: z1 shape (nvir, nocc)
    z1_grad = z1_int.reshape(nocc, nvir).T  # convert from (nocc,nvir) to (nvir,nocc)
    z1ao = reduce(cp.dot, (orbv, z1_grad, orbo.T))  # (nao, nao)

    dmz1doo = z1ao + dmzoo  # this is what gradient uses

    mf_grad = mf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))

    dh_td = rhf_grad.contract_h1e_dm(mol, h1, dmz1doo, hermi=0)  # (natm, 3)
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1doo + dmz1doo.T) * 0.5)  # nuclear attraction 1/r

    dh_td_np = np.asarray(dh_td) if not hasattr(dh_td, 'get') else dh_td.get()
    dh1e_np = np.asarray(dh1e_td) if not hasattr(dh1e_td, 'get') else dh1e_td.get()
    return dh_td_np + dh1e_np, dmz1doo.get()

coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
mol0, mf0, td0 = build_system(coords0)

hcore_grad0, dmz1doo0 = get_hcore_grad_contribution(mol0, mf0, td0)
print("hcore gradient contribution (dh_td + dh1e_td):")
print(hcore_grad0)

# FD of hcore gradient contribution w.r.t. Z displacement of H0
print("\nFD Hessian of hcore gradient w.r.t. H0-Z:")
results = {}
for ia in range(2):
    for ix in range(3):
        for d in [FD_DELTA, -FD_DELTA]:
            c = coords0.copy(); c[ia, ix] += d
            mol_d, mf_d, td_d = build_system(c)
            g_d, _ = get_hcore_grad_contribution(mol_d, mf_d, td_d)
            key = (ia, ix, d > 0)
            results[key] = g_d

# Extract h_hcore[j0,y,i0,x] from FD
h_hcore_fd = np.zeros((2, 3, 2, 3))
for i0 in range(2):
    for x in range(3):
        g_plus = results[(i0, x, True)]
        g_minus = results[(i0, x, False)]
        fd = (g_plus - g_minus) / (2 * FD_DELTA)  # (natm, 3)
        for j0 in range(2):
            for y in range(3):
                h_hcore_fd[j0, y, i0, x] = fd[j0, y]

print("\nFD h_hcore (all nonzero elements):")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                if abs(h_hcore_fd[i,x,j,y]) > 1e-5:
                    print(f"  h_hcore_fd[{i},{x},{j},{y}] = {h_hcore_fd[i,x,j,y]:.8f}")

# Compare with _e_hcore_generator(P_I_prime) and _e_hcore_generator(2*P_I_prime)
from gpu4pyscf.hessian import tdrhf as tdrhf_hess0
x_y0 = td0.xy[0]
h_obj0 = tdrhf_hess0.Hessian(td0)
td_g0 = tdrhf_grad.Gradients(td0)
z1_0 = tdrhf_hess0.solve_z_vector(td_g0, x_y0)
ints0 = tdrhf_hess0.make_intermediates(h_obj0, x_y0, z1_0)
P_I_prime0 = ints0['P_I_prime']

print("\n_e_hcore_generator with P_I_prime:")
gen_PI = _e_hcore_generator(h_obj0, P_I_prime0)
h_gen_PI = np.zeros((2, 2, 3, 3))
for i0 in range(2):
    for j0 in range(i0+1):
        h_gen_PI[i0, j0] = gen_PI(i0, j0).get()
        h_gen_PI[j0, i0] = h_gen_PI[i0, j0].T

print("_e_hcore_generator(P_I_prime) nonzero elements vs FD:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                v_gen = h_gen_PI[i,j,x,y]
                v_fd = h_hcore_fd[i,x,j,y]
                if abs(v_fd) > 1e-5 or abs(v_gen) > 1e-5:
                    print(f"  [{i},{x},{j},{y}] gen={v_gen:.6f}  fd={v_fd:.6f}  ratio={v_gen/v_fd if abs(v_fd)>1e-8 else float('nan'):.4f}")

# Now try with dmz1doo_symm = (dmz1doo + dmz1doo.T)/2
dmz1doo0_symm = cp.asarray((dmz1doo0 + dmz1doo0.T) / 2)
print("\n_e_hcore_generator with (dmz1doo+dmz1doo.T)/2:")
gen_dm = _e_hcore_generator(h_obj0, dmz1doo0_symm)
h_gen_dm = np.zeros((2, 2, 3, 3))
for i0 in range(2):
    for j0 in range(i0+1):
        h_gen_dm[i0, j0] = gen_dm(i0, j0).get()
        h_gen_dm[j0, i0] = h_gen_dm[i0, j0].T

print("_e_hcore_generator(dmz1doo_symm) vs FD:")
for i in range(2):
    for j in range(2):
        for x in range(3):
            for y in range(3):
                v_gen = h_gen_dm[i,j,x,y]
                v_fd = h_hcore_fd[i,x,j,y]
                if abs(v_fd) > 1e-5 or abs(v_gen) > 1e-5:
                    print(f"  [{i},{x},{j},{y}] gen={v_gen:.6f}  fd={v_fd:.6f}  ratio={v_gen/v_fd if abs(v_fd)>1e-8 else float('nan'):.4f}")

print("\nRelation between P_I_prime and dmz1doo:")
print(f"  max|P_I_prime|                    = {float(cp.abs(P_I_prime0).max()):.6f}")
print(f"  max|(dmz1doo+dmz1doo.T)/2|        = {float(cp.abs(dmz1doo0_symm).max()):.6f}")
print(f"  max|2*P_I_prime - dmz1doo_symm|   = {float(cp.abs(2*P_I_prime0 - dmz1doo0_symm).max()):.6f}")
