"""
FD each gradient TERM individually to determine correct hessian assembly factors.
Strategy: at +FD and -FD geometries, compute the individual gradient contributions
(hcore, JK, overlap) and take (f(+) - f(-))/2h to get the static hessian element.
Compare these FD-derived elements to what _e_hcore_generator, ejk_PI, e1_ovlp give.
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
from functools import reduce

mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf); td.kernel()

h_obj = tdrhf_hess.Hessian(td)
h_obj.method = 'analytical'
h_semi = h_obj.kernel()
# Revert to semi-analytical for reference
h_obj.method = 'semi-analytical'
h_semi = h_obj.kernel()

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
natm = mol.natm; nao = mol.nao; aoslices = mol.aoslice_by_atom()

x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
td_g = tdrhf_grad.Gradients(td)
z1 = tdrhf_hess.solve_z_vector(td_g, x_y)
ints = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
P_I_prime = ints['P_I_prime']
W_I = ints['W_I']
R_I = ints['R_I']; T_I = ints['T_I']
vhfopt = mf._opt_gpu.get(mol.omega)

# Compute analytical terms at reference geometry
de_hcore_fn = _e_hcore_generator(h_obj, P_I_prime)
e1h_ref = {}
for i0 in range(natm):
    for j0 in range(i0+1):
        v = de_hcore_fn(i0, j0)
        e1h_ref[(i0,j0)] = v
        e1h_ref[(j0,i0)] = v.T

dm0f = 2 * ints['P']
ejk_cross = (_partial_ejk_ip2(mol, dm0f + P_I_prime, vhfopt)
             - _partial_ejk_ip2(mol, dm0f, vhfopt)
             - _partial_ejk_ip2(mol, P_I_prime, vhfopt))
ejk_RI = _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt, j_factor=2.0)
ejk_TI = _partial_ejk_ip2(mol, T_I - T_I.T, vhfopt, j_factor=0.0)

s1aa, s1ab, _ = get_ovlp(mol)
s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
e1_ovlp_ref = cp.zeros((natm, natm, 3, 3))
for i0 in range(natm):
    p0, p1 = aoslices[i0][2:]
    e1_ovlp_ref[i0, i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1])
    for j0 in range(i0+1):
        q0, q1 = aoslices[j0][2:]
        e1_ovlp_ref[i0,j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1])
        e1_ovlp_ref[j0,i0] = e1_ovlp_ref[i0,j0].T

# Now FD: displace atom0 in z direction and compute gradient TERMS
FD = 2e-3
coords0 = mol.atom_coords().copy()

def grad_terms_at(coords):
    """Return hcore, JK, overlap contributions to the OMEGA gradient at displaced geometry."""
    mol_d = mol.copy(); mol_d.set_geom_(coords, unit='Bohr'); mol_d.build()
    mf_d = gpu_scf.RHF(mol_d).run()
    td_d = gpu_tdscf.rhf.TDA(mf_d); td_d.kernel()

    x_y_d = tuple([cp.asarray(v) * cp.sqrt(2) for v in td_d.xy[0]])
    td_g_d = tdrhf_grad.Gradients(td_d)
    h_d = tdrhf_hess.Hessian(td_d)
    z1_d = tdrhf_hess.solve_z_vector(td_g_d, x_y_d)
    ints_d = tdrhf_hess.make_intermediates(h_d, x_y_d, z1_d)

    mo_c = cp.asarray(mf_d.mo_coeff)
    mo_o = cp.asarray(mf_d.mo_occ)
    no = int((mo_o > 0).sum()); nm = mo_c.shape[1]; nv = nm - no
    orbo_d = mo_c[:, :no]; orbv_d = mo_c[:, no:]
    na = mol_d.natm; nao_d = mol_d.nao; asl = mol_d.aoslice_by_atom()

    P_Ip = ints_d['P_I_prime']
    W_Id = ints_d['W_I']

    # Hcore gradient of omega: Tr((dmzoo+z1ao+z1ao.T) * dh/dR)
    # (= 2*Tr(P_I_prime * dh/dR) for symmetric h)
    # Get it from actual gradient - GS gradient
    from gpu4pyscf.grad import rhf as rhf_grad_m
    de_tda = np.array(td_g_d.kernel(state=1))
    de_gs = np.array(rhf_grad_m.Gradients(mf_d).kernel())
    de_omega = de_tda - de_gs  # full omega gradient (natm, 3)

    # Hcore only: Tr(dmz1doo * dh/dR) where dh is hcore derivative
    # Use h1 from mf: get_hcore derivative. Use int3c2e for nuc part.
    h1 = cp.asarray(mf_d.nuc_grad_method().get_hcore(mol_d))  # (3, nao, nao) = -dh/dR
    # h1[x, p, q] = d<p|h|q>/dR_x (AO derivative, sign convention: rhf grad uses -h1)
    # Actually in pyscf, get_hcore for gradient returns the derivative integrals
    # The gradient contribution: -Tr(P * h1) where h1 = dh/dR_A (one-sided)
    # Full: sum both sides. Let's use the complete gradient formula.

    # Easier: compute omega gradient from existing function, and extract JK from dvhf
    # Actually, let's just use the full omega gradient components.

    # What we can extract: the full omega gradient de_omega[atm, x].
    # Let's compute the overlap contribution: -Tr(W_I * dS/dR)
    s1_d = cp.asarray(mf_d.nuc_grad_method().get_ovlp(mol_d))  # (3, nao, nao)
    # s1_d[x, p, q] = d<p|q>/dR
    # Need: for each atom A and direction x, the gradient contribution from overlap
    from gpu4pyscf.df import int3c2e as int3c2e_mod

    # Overlap gradient: -Tr(W_I * dS_full/dR)
    # dS_full[A,x,p,q] = d<p|q>/dR_{A,x}
    # Full: sum over AOs on atom A (bra) + AOs on atom A (ket) = 2*sum over bra
    grad_ovlp = np.zeros((na, 3))
    for ia in range(na):
        p0, p1 = asl[ia][2:]
        for x in range(3):
            # s1_d[x, p0:p1, :] = d<mu_A|nu>/dR_{A,x} (bra derivative)
            grad_ovlp[ia, x] = -float(contract('pq,pq->', s1_d[x, p0:p1, :], W_Id[p0:p1, :]))
            grad_ovlp[ia, x] -= float(contract('pq,qp->', s1_d[x, p0:p1, :], W_Id[:, p0:p1]))

    return de_omega.get() if hasattr(de_omega, 'get') else de_omega, grad_ovlp

print("Computing FD gradient terms...")
coords_p = coords0.copy(); coords_p[0, 2] += FD
coords_m = coords0.copy(); coords_m[0, 2] -= FD

de_p, govlp_p = grad_terms_at(coords_p)
de_m, govlp_m = grad_terms_at(coords_m)

# FD of full omega gradient and overlap gradient
fd_full = (de_p - de_m) / (2*FD)   # shape (natm, 3) -> hessian h[j,y,0,z]
fd_ovlp = (govlp_p - govlp_m) / (2*FD)  # overlap part of hessian

print("\n=== FD of full omega gradient (displacing atom0 z) ===")
print("FD hessian h[0,z,0,z] =", fd_full[0,2])
print("FD hessian h[1,z,0,z] =", fd_full[1,2])
print("Reference h_semi[0,2,0,2] =", float(h_semi[0,2,0,2]))
print("Reference h_semi[1,2,0,2] =", float(h_semi[1,2,0,2]))

print("\n=== FD of overlap gradient ===")
print("FD overlap h[0,z,0,z] =", fd_ovlp[0,2])
print("FD overlap h[1,z,0,z] =", fd_ovlp[1,2])
print("e1_ovlp[0,0,z,z] =", float(e1_ovlp_ref[0,0,2,2]), " (x2=", float(e1_ovlp_ref[0,0,2,2]*2), ")")
print("e1_ovlp[1,0,z,z] =", float(e1_ovlp_ref[1,0,2,2]), " (x2=", float(e1_ovlp_ref[1,0,2,2]*2), ")")

# The FD hessian for the overlap should match e1_ovlp_ref * factor
# factor_ovlp = fd_ovlp[0,2] / float(e1_ovlp_ref[0,0,2,2])
print("Factor for e1_ovlp[0,0,z,z]:", fd_ovlp[0,2] / float(e1_ovlp_ref[0,0,2,2]))
print("Factor for e1_ovlp[1,0,z,z]:", fd_ovlp[1,2] / float(e1_ovlp_ref[1,0,2,2]))

print("\n=== e1_hcore_ref ===")
print("e1_hcore[0,0,z,z] =", float(e1h_ref[(0,0)][2,2]))
print("e1_hcore[1,0,z,z] =", float(e1h_ref[(1,0)][2,2]))

print("\n=== ejk at reference ===")
print("ejk_cross[0,0,z,z] =", float(ejk_cross[0,0,2,2]))
print("ejk_RI[0,0,z,z] =", float(ejk_RI[0,0,2,2]))
print("ejk_TI[0,0,z,z] =", float(ejk_TI[0,0,2,2]))
print("ejk_PI[0,0,z,z] =", float((ejk_cross+ejk_RI+ejk_TI)[0,0,2,2]))

print("\n=== X-X case ===")
print("e1_hcore[0,0,x,x] =", float(e1h_ref[(0,0)][0,0]))
print("ejk_cross[0,0,x,x] =", float(ejk_cross[0,0,0,0]))
print("ejk_RI[0,0,x,x] =", float(ejk_RI[0,0,0,0]))
print("ejk_TI[0,0,x,x] =", float(ejk_TI[0,0,0,0]))
print("e1_ovlp[0,0,x,x] =", float(e1_ovlp_ref[0,0,0,0]))
print("h_semi[0,x,0,x] =", float(h_semi[0,0,0,0]))
