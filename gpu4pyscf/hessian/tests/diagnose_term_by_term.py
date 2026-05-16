"""
For H2/STO-3G TDA, isolate e1_hcore, ejk_PI, and e1_ovlp individually
and compare each against FD of the corresponding gradient term.

The goal is to find which term carries the 0.777x static X/Y error.
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.lib.cupy_helper import contract
from pyscf import gto

FD_DELTA = 2e-3  # Bohr

def build_system(atom_coords=None):
    if atom_coords is None:
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    else:
        mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
        mol.set_geom_(atom_coords, unit='Bohr')
        mol.build()
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()
    return mol, mf, td


def get_components_at(mol, mf, td, state=0):
    """Return e1_hcore, ejk_PI, e1_ovlp at current geometry."""
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    nocc = int((mo_occ > 0).sum())
    nmo = mo_coeff.shape[1]
    nao = mol.nao
    natm = mol.natm

    x_y_raw = td.xy[state]
    x_y = tuple([cp.asarray(v) for v in x_y_raw])

    h_obj = tdrhf_hess.Hessian(td)

    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)

    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y)
    intermediates = tdrhf_hess.make_intermediates(h_obj, x_y, z1)

    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp

    # e1_hcore
    de_hcore = _e_hcore_generator(h_obj, intermediates['P_I_prime'])
    e1_hcore = cp.zeros((natm, natm, 3, 3))
    aoslices = mol.aoslice_by_atom()
    for i0 in range(natm):
        for j0 in range(i0+1):
            e1_hcore[i0, j0] += de_hcore(i0, j0)
            e1_hcore[j0, i0] = e1_hcore[i0, j0].T

    # ejk_PI
    vhfopt = mf._opt_gpu.get(mol.omega)
    P_I_prime = intermediates['P_I_prime']
    P_GS = intermediates['P']
    R_I = intermediates['R_I']
    T_I = intermediates['T_I']
    ejk_PI = _partial_ejk_ip2(mol, P_I_prime + P_GS, vhfopt)
    ejk_PI -= _partial_ejk_ip2(mol, P_GS, vhfopt)
    ejk_RI = _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt)
    ejk_PI += 0.5 * ejk_RI
    ejk_TI = _partial_ejk_ip2(mol, T_I - T_I.T, vhfopt, j_factor=0.0)
    ejk_PI -= 0.5 * ejk_TI

    # e1_ovlp
    s1aa, s1ab, _ = get_ovlp(mol)
    s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    W_I = intermediates['W_I']
    e1_ovlp = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        e1_ovlp[i0, i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1]) * 2
        for j0 in range(i0+1):
            q0, q1 = aoslices[j0][2:]
            e1_ovlp[i0, j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1]) * 2
            e1_ovlp[j0, i0] = e1_ovlp[i0, j0].T

    return e1_hcore, ejk_PI, e1_ovlp, intermediates


def get_gradient_terms(mol, mf, td, state=0):
    """Return each static gradient term: h_core, jk, ovlp contributions."""
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum())
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    natm = mol.natm

    x_y_raw = td.xy[state]
    x, y = [cp.asarray(v) for v in x_y_raw]
    xpy = x.reshape(nocc, nvir)
    xmy = xpy  # TDA

    h_obj = tdrhf_hess.Hessian(td)
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y_raw)
    intermediates = tdrhf_hess.make_intermediates(h_obj, x_y_raw, z1)

    P_I_prime = intermediates['P_I_prime']
    W_I = intermediates['W_I']
    R_I = intermediates['R_I']
    T_I = intermediates['T_I']

    from gpu4pyscf.hessian.tdrhf import _get_h1ao_x
    h1ao_x = _get_h1ao_x(mol)  # (natm, 3, nao, nao) — H^x

    # grad_hcore = Tr(P_I_prime @ H^x) for each atom, direction
    grad_hcore = cp.zeros((natm, 3))
    for ia in range(natm):
        for ix in range(3):
            grad_hcore[ia, ix] = cp.trace(P_I_prime @ h1ao_x[ia, ix])

    # grad_ovlp = -Tr(W_I @ S^x) for each atom, direction
    from gpu4pyscf.grad import rhf as rhf_grad
    mf_grad = rhf_grad.Gradients(mf)
    res_ovlp = mf_grad.get_ovlp(mol)
    s1a = cp.asarray(res_ovlp[-1] if isinstance(res_ovlp, tuple) else res_ovlp)
    aoslices = mol.aoslice_by_atom()
    # s1a: (3, nao, nao) where s1a[x, mu, nu] = -<dmu/dR_A | nu> for AO mu on atom A
    s1ao_full = cp.zeros((natm, 3, mol.nao, mol.nao))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        s1ao_full[ia, :, p0:p1] = s1a[:, p0:p1]
        s1ao_full[ia, :, :, p0:p1] = s1a[:, p0:p1].transpose(0, 2, 1)

    grad_ovlp = cp.zeros((natm, 3))
    for ia in range(natm):
        for ix in range(3):
            grad_ovlp[ia, ix] = -cp.trace(W_I @ s1ao_full[ia, ix])

    # grad_jk: JK part — from ERI first derivatives acting on P_I_prime and R_I, T_I
    # This is the hard part; use FD of the JK contribution to get reference
    # We approximate it here as: total gradient - hcore - ovlp - z_relax
    # Instead, let's just return hcore and ovlp so we can FD each independently
    return grad_hcore, grad_ovlp


def fd_of_grad_term(atom_idx, direction, get_fn, delta=FD_DELTA):
    """FD second derivative of get_fn(mol,mf,td) w.r.t. atom_idx displacement."""
    coords0 = np.array([[0,0,0],[0,0,1.4]], dtype=float)
    results = []
    for d in [delta, -delta]:
        c = coords0.copy(); c[atom_idx, direction] += d
        mol_p, mf_p, td_p = build_system(c)
        results.append(get_fn(mol_p, mf_p, td_p))
    return (results[0] - results[1]) / (2*delta)


# ── Main ──────────────────────────────────────────────────────────────────
mol0, mf0, td0 = build_system()
e1_hcore, ejk_PI, e1_ovlp, intermediates = get_components_at(mol0, mf0, td0)

# Symmetrize and reshape to (natm,3,natm,3)
def symm(h):
    h = h.transpose(0, 2, 1, 3)  # (natm,natm,3,3) -> (natm,3,natm,3)
    return 0.5 * (h + h.transpose(2, 3, 0, 1))

h_hcore = symm(e1_hcore)
h_ejk = symm(ejk_PI)
h_ovlp = symm(e1_ovlp)
h_static = h_hcore + h_ejk + h_ovlp

print("=== Component breakdown for H2/STO-3G TDA (unscaled amplitudes) ===")
print(f"{'Element':12s} {'Hcore':10s} {'EJK':10s} {'Ovlp':10s} {'Static':10s}")
natm = mol0.natm
aoslices = mol0.aoslice_by_atom()
for i in range(natm):
    for j in range(natm):
        for x in range(3):
            for y in range(3):
                vh = float(h_hcore[i,x,j,y])
                ve = float(h_ejk[i,x,j,y])
                vo = float(h_ovlp[i,x,j,y])
                vs = float(h_static[i,x,j,y])
                if abs(vs) > 1e-5:
                    print(f"[{i},{x},{j},{y}]    {vh:10.6f} {ve:10.6f} {vo:10.6f} {vs:10.6f}")

# Now FD: for the (0, 0) atom X-direction perturbation, get FD of hcore and ovlp grad
print("\n=== FD validation of gradient terms ===")
print("Computing FD of grad_hcore w.r.t. H0-X...")
dg_hcore_dX = fd_of_grad_term(0, 0, lambda m,f,t: float(get_gradient_terms(m,f,t)[0][0, 0]))
print(f"  FD d/dX_H0 [ grad_hcore[H0,X] ] = {float(dg_hcore_dX):.8f}")
print(f"  Analytical h_hcore[0,0,0,0]     = {float(h_hcore[0,0,0,0]):.8f}")
print(f"  Ratio                            = {float(h_hcore[0,0,0,0])/float(dg_hcore_dX):.4f}")

print("\nComputing FD of grad_ovlp w.r.t. H0-X...")
dg_ovlp_dX = fd_of_grad_term(0, 0, lambda m,f,t: float(get_gradient_terms(m,f,t)[1][0, 0]))
print(f"  FD d/dX_H0 [ grad_ovlp[H0,X] ] = {float(dg_ovlp_dX):.8f}")
print(f"  Analytical h_ovlp[0,0,0,0]     = {float(h_ovlp[0,0,0,0]):.8f}")
if abs(float(dg_ovlp_dX)) > 1e-10:
    print(f"  Ratio                           = {float(h_ovlp[0,0,0,0])/float(dg_ovlp_dX):.4f}")

# Semi-analytical reference
h_semi_obj = tdrhf_hess.Hessian(td0)
h_semi_obj.method = 'semi-analytical'
h_semi = h_semi_obj.kernel()
print(f"\nSemi-analytical h[0,0,0,0] = {float(h_semi[0,0,0,0]):.8f}")
print(f"Static total   h[0,0,0,0] = {float(h_static[0,0,0,0]):.8f}")
print(f"Static/Semi ratio         = {float(h_static[0,0,0,0])/float(h_semi[0,0,0,0]):.4f}")

if __name__ == '__main__':
    pass
