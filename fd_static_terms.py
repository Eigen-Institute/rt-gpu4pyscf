"""
Test each static hessian term by doing INTEGRAL-FD (fixed density, move integrals).
This gives exactly the static part of the hessian (no density relaxation).
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf, tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp

mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf); td.kernel()

h_obj = tdrhf_hess.Hessian(td)
h_obj.method = 'semi-analytical'
h_semi = h_obj.kernel()

x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
td_g = tdrhf_grad.Gradients(td)
z1 = tdrhf_hess.solve_z_vector(td_g, x_y)
ints = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
P_I_prime = ints['P_I_prime']
W_I = ints['W_I']
R_I = ints['R_I']; T_I = ints['T_I']
dm0_full = 2 * ints['P']
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
nocc = int((mo_occ > 0).sum())
orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
natm = mol.natm; nao = mol.nao; aoslices = mol.aoslice_by_atom()
vhfopt = mf._opt_gpu.get(mol.omega)

# Analytical terms
de_hcore_fn = _e_hcore_generator(h_obj, P_I_prime)
e1h = de_hcore_fn(0, 0)
print(f"e1_hcore[0,0,z,z] = {float(e1h[2,2]):.6f}")
print(f"e1_hcore[0,0,x,x] = {float(e1h[0,0]):.6f}")

ejk_cross = _partial_ejk_ip2(mol, dm0_full+P_I_prime, vhfopt) - _partial_ejk_ip2(mol, dm0_full, vhfopt) - _partial_ejk_ip2(mol, P_I_prime, vhfopt)
ejk_RI = _partial_ejk_ip2(mol, R_I+R_I.T, vhfopt, j_factor=2.0)
ejk_TI = _partial_ejk_ip2(mol, T_I-T_I.T, vhfopt, j_factor=0.0)

print(f"\nCurrent ejk[0,0,z,z]: cross={float(ejk_cross[0,0,2,2]):.6f}, RI={float(ejk_RI[0,0,2,2]):.6f}, TI={float(ejk_TI[0,0,2,2]):.6f}")
print(f"Current ejk[0,0,x,x]: cross={float(ejk_cross[0,0,0,0]):.6f}, RI={float(ejk_RI[0,0,0,0]):.6f}, TI={float(ejk_TI[0,0,0,0]):.6f}")

s1aa, s1ab, _ = get_ovlp(mol)
s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
e1ov_00 = -contract('xypq,pq->xy', s1aa[:,:,0:1], W_I[0:1])
e1ov_00 -= contract('xypq,pq->xy', s1ab[:,:,0:1,0:1], W_I[0:1,0:1])
print(f"\ne1_ovlp[0,0,z,z] = {float(e1ov_00[2,2]):.6f}")
print(f"e1_ovlp[0,0,x,x] = {float(e1ov_00[0,0]):.6f}")

# Now FD the static contributions (fixed density matrices, varying integrals)
FD = 2e-3
coords0 = mol.atom_coords().copy()

def hcore_gradient_fixed_dm(coords, dm):
    """Compute Tr(dm * dh/dR) at displaced geometry with FIXED dm from reference."""
    mol_d = mol.copy(); mol_d.set_geom_(coords, unit='Bohr'); mol_d.build()
    # Get hcore gradient for dm (using the displaced molecule but reference dm)
    from pyscf import gto
    mf_d_cpu = pyscf.scf.RHF(mol_d)
    h1_d = cp.asarray(mf_d_cpu.get_hcore(mol_d))  # (nao, nao)

    # Also nuclear attraction from gradient
    from gpu4pyscf.df import int3c2e as int3c2e_mod
    # Use simple: Tr(dm * h1_d) as function of coords -> FD gives gradient
    # h1 at displaced = hcore at displaced
    return float(contract('pq,pq->', h1_d, dm))

def jk_energy_fixed_dm(coords, dm, j_factor=1.0, k_factor=1.0):
    """Compute 0.5*Tr(dm * G(dm)) at displaced geometry."""
    mol_d = mol.copy(); mol_d.set_geom_(coords, unit='Bohr'); mol_d.build()
    mf_d = gpu_scf.RHF(mol_d)
    vj, vk = mf_d.get_jk(mol_d, dm.get(), hermi=1)
    vj = cp.asarray(vj); vk = cp.asarray(vk)
    G = j_factor * vj - k_factor * 0.5 * vk
    return float(contract('pq,pq->', dm, G))

def ovlp_energy_fixed_dm(coords, W):
    """Compute -Tr(W * S) at displaced geometry with fixed W."""
    mol_d = mol.copy(); mol_d.set_geom_(coords, unit='Bohr'); mol_d.build()
    mf_d_cpu = pyscf.scf.RHF(mol_d)
    S = cp.asarray(mf_d_cpu.get_ovlp(mol_d))
    return -float(contract('pq,pq->', W, S))

print("\n=== FD of static terms (displacing atom0 z, then taking d^2/dz^2) ===")
print("This gives the PURE STATIC second derivative (fixed density matrices)")

# For second derivative along z: FD twice
# FD1: (f(+h) - f(-h)) / 2h gives df/dz at R0
# FD2: (f(+h) + f(-h) - 2*f(0)) / h^2 gives d^2f/dz^2 at R0
h = FD

# Hcore: Tr(P_I_prime * h(R))
# We need d^2/dz^2 Tr(P_I_prime * h(R)) = Tr(P_I_prime * d^2h/dz^2)
# This is exactly what _e_hcore_generator gives (with factor 2 for bra+ket)
# But _e_hcore_generator is called for the DIAGONAL (0,0) with the hessian's internal formula

# However, the relevant density for hcore is dmzoo (2*P_I_prime for z1=0)
# The hcore gradient in grad/tdrhf.py uses Tr(dmzoo * dh/dR) = 2*Tr(P_I_prime * dh/dR)
dmzoo = ints['dmzoo']

coords_p = coords0.copy(); coords_p[0, 2] += h
coords_m = coords0.copy(); coords_m[0, 2] -= h
coords_pp = coords0.copy(); coords_pp[0, 2] += 2*h
coords_mm = coords0.copy(); coords_mm[0, 2] -= 2*h

# FD2 of Tr(dmzoo * h(R)) and Tr(P_I_prime * h(R))
e_hcore_zz_dmzoo_ref = hcore_gradient_fixed_dm(coords0, dmzoo)
e_hcore_zz_dmzoo_p = hcore_gradient_fixed_dm(coords_p, dmzoo)
e_hcore_zz_dmzoo_m = hcore_gradient_fixed_dm(coords_m, dmzoo)
fd2_hcore_dmzoo = (e_hcore_zz_dmzoo_p + e_hcore_zz_dmzoo_m - 2*e_hcore_zz_dmzoo_ref) / h**2

e_hcore_zz_PI_ref = hcore_gradient_fixed_dm(coords0, P_I_prime)
e_hcore_zz_PI_p = hcore_gradient_fixed_dm(coords_p, P_I_prime)
e_hcore_zz_PI_m = hcore_gradient_fixed_dm(coords_m, P_I_prime)
fd2_hcore_PI = (e_hcore_zz_PI_p + e_hcore_zz_PI_m - 2*e_hcore_zz_PI_ref) / h**2

print(f"\nFD^2 Tr(dmzoo * h)/dz^2 = {fd2_hcore_dmzoo:.6f}")
print(f"FD^2 Tr(P_I_prime * h)/dz^2 = {fd2_hcore_PI:.6f}")
print(f"e1_hcore[0,0,z,z] = {float(e1h[2,2]):.6f}")
print(f"e1_hcore*2 = {float(e1h[2,2])*2:.6f}")

# Note: _e_hcore_generator includes de_nuc_elec (nuclear attraction part).
# fd2_hcore only includes the AO integral part since hcore() gives T+V_en.
# But mol.atom_coords change moves the NUCLEAR position too, so fd2 includes
# both the AO derivative AND the nuclear position derivative of V_en.
# The _e_hcore_generator() includes de_nuc_elec which IS the nuclear position part.
# So the comparison should be direct.

print(f"\nConclusion: FD^2 Tr(dmzoo*h) = {fd2_hcore_dmzoo:.4f}")
print(f"            2*e1_hcore = {float(e1h[2,2])*2:.4f}")
print(f"            These should be equal if the formula is correct")

# Overlap: -Tr(W_I * S(R))
e_ovlp_ref = ovlp_energy_fixed_dm(coords0, W_I)
e_ovlp_p = ovlp_energy_fixed_dm(coords_p, W_I)
e_ovlp_m = ovlp_energy_fixed_dm(coords_m, W_I)
fd2_ovlp = (e_ovlp_p + e_ovlp_m - 2*e_ovlp_ref) / h**2

print(f"\nFD^2 (-Tr(W_I * S))/dz^2 = {fd2_ovlp:.6f}")
print(f"e1_ovlp[0,0,z,z] = {float(e1ov_00[2,2]):.6f}")
print(f"e1_ovlp*2 = {float(e1ov_00[2,2])*2:.6f}")

# JK coupling: Tr(P_I_prime * G(dm0)) or Tr(dmzoo * G(dm0)) ?
# In gradient: -get_veff(dmzoo) contributes to omega. So the relevant energy is:
# coupling energy = -E_JK(dm0+dmzoo) + E_JK(dm0) + E_JK(dmzoo)
#                 = -bilinear_JK(dm0, dmzoo)
# And the omega JK has:
# omega_JK_coupling = -2*bilinear_JK(dm0, dmzoo)  [factor 2 from 2*dvhf and factor -1 from subtraction]
# Wait let me re-examine...

# The omega gradient JK (from grad/tdrhf.py with 2*dvhf):
# = 2*(get_veff(dm0+dmzoo) - get_veff(dm0) - get_veff(dmzoo)) + 4*get_veff(RI+RI.T) - 4*get_veff(TI-TI.T)
# The energy function for this gradient:
# = 2*(E_JK(dm0+dmzoo) - E_JK(dm0) - E_JK(dmzoo)) + 4*E_JK(RI+RI.T) - 4*E_JK(TI-TI.T, k_factor)

# where the last term uses j=0, k=1: E_JK(dm, j=0, k=1) = -K_energy(dm)

# So compute FD of each:
# 2*(E_JK(dm0+dmzoo) - E_JK(dm0) - E_JK(dmzoo)):
e_coupling_ref = jk_energy_fixed_dm(coords0, dm0_full+dmzoo) - jk_energy_fixed_dm(coords0, dm0_full) - jk_energy_fixed_dm(coords0, dmzoo)
e_coupling_p = jk_energy_fixed_dm(coords_p, dm0_full+dmzoo) - jk_energy_fixed_dm(coords_p, dm0_full) - jk_energy_fixed_dm(coords_p, dmzoo)
e_coupling_m = jk_energy_fixed_dm(coords_m, dm0_full+dmzoo) - jk_energy_fixed_dm(coords_m, dm0_full) - jk_energy_fixed_dm(coords_m, dmzoo)
fd2_coupling = 2 * (e_coupling_p + e_coupling_m - 2*e_coupling_ref) / h**2

# 4*E_JK(RI+RI.T, j=1, k=1):
RI_plus = (R_I + R_I.T)
e_RI_ref = jk_energy_fixed_dm(coords0, RI_plus, j_factor=1.0, k_factor=1.0)
e_RI_p = jk_energy_fixed_dm(coords_p, RI_plus, j_factor=1.0, k_factor=1.0)
e_RI_m = jk_energy_fixed_dm(coords_m, RI_plus, j_factor=1.0, k_factor=1.0)
fd2_RI = 4 * (e_RI_p + e_RI_m - 2*e_RI_ref) / h**2

# -4*E_JK(TI-TI.T, j=0, k=1) = 4*K_energy(TI-TI.T):
TI_minus = (T_I - T_I.T)
# E_JK(dm, j=0, k=1) = -0.5*Tr(dm*K(dm)), so -4*E_JK = 4*0.5*Tr = 2*Tr(dm*K)
# Actually: E_JK with j=0, k=1: 0.5*Tr(dm*(0*J - K)(dm)) = -0.5*Tr(dm*K(dm))
e_TI_ref = jk_energy_fixed_dm(coords0, TI_minus, j_factor=0.0, k_factor=1.0)
e_TI_p = jk_energy_fixed_dm(coords_p, TI_minus, j_factor=0.0, k_factor=1.0)
e_TI_m = jk_energy_fixed_dm(coords_m, TI_minus, j_factor=0.0, k_factor=1.0)
fd2_TI = -4 * (e_TI_p + e_TI_m - 2*e_TI_ref) / h**2

total_fd_jk = fd2_coupling + fd2_RI + fd2_TI

print(f"\n=== JK static FD contributions [z,z] ===")
print(f"FD^2 of 2*(JK cross) = {fd2_coupling:.6f}")
print(f"FD^2 of 4*JK(RI)     = {fd2_RI:.6f}")
print(f"FD^2 of -4*JK(TI)    = {fd2_TI:.6f}")
print(f"Total JK FD          = {total_fd_jk:.6f}")
print(f"Current ejk_PI[0,0,z,z]*2 = {float((ejk_cross+ejk_RI+ejk_TI)[0,0,2,2])*2:.6f}")
print(f"\nh_semi[0,z,0,z] = {float(h_semi[0,2,0,2]):.6f}")
print(f"e1_hcore*2 + jk_FD + e1_ovlp*2 = {float(e1h[2,2])*2 + total_fd_jk + float(e1ov_00[2,2])*2:.6f}")
print(f"(+ e1_perturbed needed)")

# Now check x-x (FD along x direction)
coords_px = coords0.copy(); coords_px[0, 0] += h
coords_mx = coords0.copy(); coords_mx[0, 0] -= h

e_hcore_xx_PI_p = hcore_gradient_fixed_dm(coords_px, P_I_prime)
e_hcore_xx_PI_m = hcore_gradient_fixed_dm(coords_mx, P_I_prime)
fd2_hcore_PI_xx = (e_hcore_xx_PI_p + e_hcore_gradient_fixed_dm(coords0, P_I_prime) - 2*e_hcore_xx_PI_m) / h**2
# redo properly:
e_hcore_PI_0_xx = hcore_gradient_fixed_dm(coords0, P_I_prime)
fd2_hcore_PI_xx = (e_hcore_xx_PI_p + e_hcore_xx_PI_m - 2*e_hcore_PI_0_xx) / h**2

e_ovlp_px = ovlp_energy_fixed_dm(coords_px, W_I)
e_ovlp_mx = ovlp_energy_fixed_dm(coords_mx, W_I)
fd2_ovlp_xx = (e_ovlp_px + e_ovlp_mx - 2*e_ovlp_ref) / h**2

e_coupling_px = jk_energy_fixed_dm(coords_px, dm0_full+dmzoo) - jk_energy_fixed_dm(coords_px, dm0_full) - jk_energy_fixed_dm(coords_px, dmzoo)
e_coupling_mx = jk_energy_fixed_dm(coords_mx, dm0_full+dmzoo) - jk_energy_fixed_dm(coords_mx, dm0_full) - jk_energy_fixed_dm(coords_mx, dmzoo)
fd2_coupling_xx = 2 * (e_coupling_px + e_coupling_mx - 2*e_coupling_ref) / h**2

e_RI_px = jk_energy_fixed_dm(coords_px, RI_plus)
e_RI_mx = jk_energy_fixed_dm(coords_mx, RI_plus)
fd2_RI_xx = 4 * (e_RI_px + e_RI_mx - 2*e_RI_ref) / h**2

e_TI_px = jk_energy_fixed_dm(coords_px, TI_minus, j_factor=0.0, k_factor=1.0)
e_TI_mx = jk_energy_fixed_dm(coords_mx, TI_minus, j_factor=0.0, k_factor=1.0)
fd2_TI_xx = -4 * (e_TI_px + e_TI_mx - 2*e_TI_ref) / h**2

total_fd_jk_xx = fd2_coupling_xx + fd2_RI_xx + fd2_TI_xx

print(f"\n=== X-X contributions ===")
print(f"FD^2 hcore (Tr(dmzoo*h)) = {fd2_hcore_PI_xx * 2:.6f}")
print(f"FD^2 JK_coupling * 2     = {fd2_coupling_xx:.6f}")
print(f"FD^2 JK_RI * 4           = {fd2_RI_xx:.6f}")
print(f"FD^2 JK_TI * (-4)        = {fd2_TI_xx:.6f}")
print(f"FD^2 ovlp * 2            = {fd2_ovlp_xx:.6f}")
total_static_xx = fd2_hcore_PI_xx*2 + total_fd_jk_xx + fd2_ovlp_xx
print(f"Total static for x-x     = {total_static_xx:.6f}")
print(f"h_semi[0,x,0,x] = {float(h_semi[0,0,0,0]):.6f}")
print(f"Missing (= correct e1_perturbed_xx) = {float(h_semi[0,0,0,0]) - total_static_xx:.6f}")
