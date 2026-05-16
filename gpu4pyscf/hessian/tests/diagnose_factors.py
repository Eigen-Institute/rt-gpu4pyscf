"""
Determine the correct factor for each term in the TDA hessian assembly.
Strategy: FD each term's gradient to get the second-derivative contribution.
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

# First check: ground state hessian - does _e_hcore and _partial_ejk_ip2 work correctly?
mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf); td.kernel()

mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mo_energy = cp.asarray(mf.mo_energy)
nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
natm = mol.natm; nao = mol.nao; aoslices = mol.aoslice_by_atom()

# Ground state hessian for reference
mf_hess = rhf_hess_gpu.Hessian(mf)
h_gs = mf_hess.kernel()
print("Ground state hessian [0,2,0,2] (z-z atom0-atom0):", float(h_gs[0, 2, 0, 2]))
print("Ground state hessian [0,0,0,0] (x-x atom0-atom0):", float(h_gs[0, 0, 0, 0]))

# TDA semi-analytical hessian
h_obj = tdrhf_hess.Hessian(td)
h_semi = h_obj.kernel()
print("\nTDA semi-analytical hessian [0,2,0,2]:", float(h_semi[0, 2, 0, 2]))
print("TDA semi-analytical hessian [0,0,0,0]:", float(h_semi[0, 0, 0, 0]))

# FD of individual contributions
FD = 2e-3

def build(coords):
    mol_d = mol.copy(); mol_d.set_geom_(coords, unit='Bohr'); mol_d.build()
    mf_d = gpu_scf.RHF(mol_d).run()
    td_d = gpu_tdscf.rhf.TDA(mf_d); td_d.kernel()
    return mol_d, mf_d, td_d

coords0 = mol.atom_coords().copy()

def get_static_contributions(mol_d, mf_d, td_d):
    """Compute static e1_hcore, ejk_PI, e1_ovlp at displaced geometry."""
    x_y_d = tuple([cp.asarray(v) * cp.sqrt(2) for v in td_d.xy[0]])
    td_g_d = tdrhf_grad.Gradients(td_d)
    h_d = tdrhf_hess.Hessian(td_d)
    z1_d = tdrhf_hess.solve_z_vector(td_g_d, x_y_d)
    ints_d = tdrhf_hess.make_intermediates(h_d, x_y_d, z1_d)

    mo_c = cp.asarray(mf_d.mo_coeff)
    mo_o = cp.asarray(mf_d.mo_occ)
    no = int((mo_o > 0).sum()); nm = mo_c.shape[1]; nv = nm - no
    orbo = mo_c[:, :no]; orbv = mo_c[:, no:]
    na = mol_d.natm; nao_d = mol_d.nao
    asl = mol_d.aoslice_by_atom()

    # e1_hcore
    de_h = _e_hcore_generator(h_d, ints_d['P_I_prime'])
    e1h = cp.zeros((na, na, 3, 3))
    for i0 in range(na):
        for j0 in range(i0+1):
            e1h[i0,j0] = de_h(i0, j0); e1h[j0,i0] = e1h[i0,j0].T

    # ejk_PI
    vhf = mf_d._opt_gpu.get(mol_d.omega)
    P_I_p = ints_d['P_I_prime']; R_I = ints_d['R_I']; T_I = ints_d['T_I']
    dm0f = 2 * ints_d['P']
    ejk = (_partial_ejk_ip2(mol_d, dm0f + P_I_p, vhf) - _partial_ejk_ip2(mol_d, dm0f, vhf) - _partial_ejk_ip2(mol_d, P_I_p, vhf))
    ejk += _partial_ejk_ip2(mol_d, R_I + R_I.T, vhf, j_factor=2.0)
    ejk += _partial_ejk_ip2(mol_d, T_I - T_I.T, vhf, j_factor=0.0)

    # e1_ovlp
    s1aa, s1ab, _ = get_ovlp(mol_d)
    W_I = ints_d['W_I']
    e1ov = cp.zeros((na, na, 3, 3))
    for i0 in range(na):
        p0, p1 = asl[i0][2:]
        e1ov[i0,i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1])
        for j0 in range(i0+1):
            q0, q1 = asl[j0][2:]
            e1ov[i0,j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1])
            e1ov[j0,i0] = e1ov[i0,j0].T

    return e1h.get(), ejk.get(), e1ov.get()

def sym(h):
    h2 = h.transpose(0,2,1,3)
    return 0.5*(h2 + h2.transpose(2,3,0,1))

# FD only atom0, z direction to get the first column of hessian
print("\nComputing FD of static terms (displacing atom0 z by +-FD)...")
coords_p = coords0.copy(); coords_p[0, 2] += FD
coords_m = coords0.copy(); coords_m[0, 2] -= FD

mol_p, mf_p, td_p = build(coords_p)
mol_m, mf_m, td_m = build(coords_m)

e1h_p, ejk_p, e1ov_p = get_static_contributions(mol_p, mf_p, td_p)
e1h_m, ejk_m, e1ov_m = get_static_contributions(mol_m, mf_m, td_m)

# FD for the "hessian" = d/dR_z[atom0] of (static contribution at displaced geometry)
# These give h[j, y, 0, z] = d/dR[0,z] (term[j, y])
# term[j, y] is the gradient w.r.t. R[j, y]

fd_e1h = (e1h_p - e1h_m) / (2 * FD)  # shape (natm, natm, 3, 3)
fd_ejk = (ejk_p - ejk_m) / (2 * FD)
fd_e1ov = (e1ov_p - e1ov_m) / (2 * FD)

# fd_e1h[j, k, y, ?] = derivative of e1h[j,k,y,z] w.r.t. R[0,z] ...
# Actually FD of the (natm, natm, 3, 3) arrays gives how each element changes when atom0-z moves
# The hessian element we want is h[j0, y, 0, z] = d/dR[0,z] of (gradient component [j0,y])
# But e1h[j0,j0,y,?] is the second derivative contribution to the gradient of j0 w.r.t. some direction

# Let me just compare specific elements:
# For atom0-z hessian diagonal: h_semi[0,2,0,2] = 0.6877
# The contribution from displacing atom0-z:
# fd_term[0,0,2,2] = derivative of e1h[atom0,atom0,z,z] w.r.t. R[0,z]... no this is not right

# Actually, the static e1h as function of geometry is the GRADIENT component,
# not directly the hessian. Let me be more careful.

# The static e1_hcore[i0,j0,x,y] is a second derivative integral at fixed geometry.
# When we displace atom0 by dR_z, we get e1_hcore(R+dR) which is different from e1_hcore(R).
# But this static e1_hcore is NOT itself a gradient - it's a second derivative of the energy
# contracted with P_I_prime(R). The TOTAL hessian element H[i,x,j,y] = e1_hcore[i,j,x,y] * factor.

# So what I want is: for each "static" contribution, what is the correct factor?
# The FULL hessian is known (h_semi). The static contributions are parts of it.
# But the static contributions must match the semi-analytical.

# Actually, the hessian assembly in the code is:
# omega_xy = f * e1_hcore + g * ejk_PI + h * e1_ovlp + e1_perturbed
# (with some factors f, g, h)
# After sym(), this should give h_semi.

# The perturbed term at the reference geometry (with current factors) = 0.9008 for z-z.
# The static terms (with factor 2 each) = 0.3126 - 1.3743 + 0.0271 = -1.0346.
# Total = -1.0346 + 0.9008 = -0.1338. But h_semi = 0.6877.

# Let me now check what the perturbed term SHOULD be.
# h_semi - (static with some factors) = correct_perturbed
# If static factor=2: -1.0346 + correct_perturbed = 0.6877, correct_perturbed = 1.7223
# If static factor=1: -0.5173 + correct_perturbed = 0.6877, correct_perturbed = 1.2050

# For the x-x case, perturbed = 0:
# If static factor = f: f*(-0.9999 + 0.9701 - 0.1576) = -0.4396
#                       f*(-0.1874) = -0.4396
#                       f = 2.347 (not integer!)

# This suggests the factors are NOT uniform across the three static terms.

# Let me try to determine each factor separately using x-x and z-z:
# For x-x: fh*(-1.0) + fj*(0.9701) + fo*(-0.1576) = -0.4396
# For z-z: fh*(0.1563) + fj*(-0.6872) + fo*(0.0136) + correct_pert = 0.6877

# From x-x: fh = (fj*0.9701 - fo*0.1576 + 0.4396) / 1.0
# This is underdetermined with 3 unknowns.

# Let me try to isolate ejk by computing it at reference and checking against FD.
# Actually: compute the gradient of omega and see which terms contribute.
print("\n=== Gradient check ===")
print("Computing TDA gradient at reference...")
td_g = tdrhf_grad.Gradients(td)
de_tda = np.array(td_g.kernel(state=1))
from gpu4pyscf.grad import rhf as grad_rhf
de_gs = np.array(grad_rhf.Gradients(mf).kernel())
de_omega = de_tda - de_gs
print("omega gradient:", de_omega)

# Now compute gradient contributions analytically
x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
z1 = tdrhf_hess.solve_z_vector(td_g, x_y)
ints = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
P_I_prime = ints['P_I_prime']
W_I = ints['W_I']

# Gradient of hcore: Tr(P_I_prime * dh/dR)
# Gradient of JK: Tr(P_I_prime * dG(P_gs)/dR) + transition terms
# Gradient of overlap: -Tr(W_I * dS/dR)

# Hcore gradient
from gpu4pyscf.hessian.rhf import _get_h1ao_x
h1ao_x = _get_h1ao_x(mol)  # shape (natm*3, nao, nao)
grad_hcore = cp.zeros((natm, 3))
for ia in range(natm):
    p0, p1 = aoslices[ia][2:]
    for ix in range(3):
        # h1 = h1ao_x[ia*3+ix]
        h1 = h1ao_x[ia*3+ix]
        grad_hcore[ia, ix] = cp.trace(h1 @ P_I_prime + P_I_prime @ h1.T)

print("Hcore gradient contribution (2x for spin?):", grad_hcore.get())

# From the gradient code, the coupling Tr(P_I_prime * F^x):
# This includes both hcore and JK terms.
# The F^x includes h^x + G^x(dm0)

# Actually, let me just check: does the total gradient match?
print("Reference omega gradient:", de_omega)

# E1_ovlp gradient check
s1aa, s1ab, s1a = get_ovlp(mol)
grad_ovlp = cp.zeros((natm, 3))
for ia in range(natm):
    p0, p1 = aoslices[ia][2:]
    grad_ovlp[ia] -= contract('xpq,pq->', s1a[:, p0:p1, :], W_I[p0:p1, :])
    grad_ovlp[ia] -= contract('xpq,qp->', s1a[:, p0:p1, :], W_I[:, p0:p1])
print("\nOverlap gradient contribution (raw):", grad_ovlp.get())
print("Overlap gradient * 2 (for spin):", (2*grad_ovlp).get())
