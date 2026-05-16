"""
Decompose the analytical TDA hessian into individual terms and compare to semi-analytical.
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

mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf); td.kernel()

x_y_orig = td.xy[0]
x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
omega = float(td.e[0])
mo_coeff = cp.asarray(mf.mo_coeff)
mo_occ = cp.asarray(mf.mo_occ)
mo_energy = cp.asarray(mf.mo_energy)
nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
natm = mol.natm; nao = mol.nao; aoslices = mol.aoslice_by_atom()

h_obj = tdrhf_hess.Hessian(td)
h_semi = h_obj.kernel()

td_g = tdrhf_grad.Gradients(td)
mf_hess = rhf_hess_gpu.Hessian(mf)
h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
fx = mf_hess.gen_vind(mo_coeff, mo_occ)
mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
mo1 = cp.asarray(mo1)

s1aa_h, s1ab_h, s1a_basis = get_ovlp(mol)  # s1a_basis shape (3, nao, nao)
s1ao = cp.zeros((natm, 3, nao, nao))
for atm_id in range(natm):
    p0, p1 = aoslices[atm_id][2:]
    s1ao[atm_id, :, p0:p1, :] += s1a_basis[:, p0:p1, :]
    s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1, :].transpose(0, 2, 1)

s1mo = cp.zeros((natm, 3, nmo, nmo))
for i in range(natm):
    for j in range(3):
        s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff

Ux = cp.zeros((natm, 3, nmo, nmo))
Ux[:, :, :, :nocc] = mo1
Ux[:, :, :nocc, nocc:] = -s1mo[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2)
Ux[:, :, :nocc, :nocc] = -0.5 * s1mo[:, :, :nocc, :nocc]
Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]

z1 = tdrhf_hess.solve_z_vector(td_g, x_y)
x1, y1 = tdrhf_hess.solve_cptddft(h_obj, x_y, omega, mo1, mo_e1, s1mo)
intermediates = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
perturbed = tdrhf_hess.make_perturbed_intermediates(h_obj, intermediates, x_y, x1, y1, Ux, z1, s1mo)

de_hcore = _e_hcore_generator(h_obj, intermediates['P_I_prime'])
e1_hcore = cp.zeros((natm, natm, 3, 3))
for i0 in range(natm):
    for j0 in range(i0 + 1):
        e1_hcore[i0, j0] += de_hcore(i0, j0)
        e1_hcore[j0, i0] = e1_hcore[i0, j0].T

vhfopt = mf._opt_gpu.get(mol.omega)
P_I_prime = intermediates['P_I_prime']
R_I = intermediates['R_I']
T_I = intermediates['T_I']
_dm0_full = 2 * intermediates['P']
ejk_PI = (_partial_ejk_ip2(mol, _dm0_full + P_I_prime, vhfopt)
          - _partial_ejk_ip2(mol, _dm0_full, vhfopt)
          - _partial_ejk_ip2(mol, P_I_prime, vhfopt))
ejk_PI += _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt, j_factor=2.0)
ejk_PI += _partial_ejk_ip2(mol, T_I - T_I.T, vhfopt, j_factor=0.0)

W_I = intermediates['W_I']
e1_ovlp = cp.zeros((natm, natm, 3, 3))
for i0 in range(natm):
    p0, p1 = aoslices[i0][2:]
    e1_ovlp[i0, i0] -= contract('xypq,pq->xy', s1aa_h[:, :, p0:p1], W_I[p0:p1])
    for j0 in range(i0 + 1):
        q0, q1 = aoslices[j0][2:]
        e1_ovlp[i0, j0] -= contract('xypq,pq->xy', s1ab_h[:, :, p0:p1, q0:q1], W_I[p0:p1, q0:q1])
        e1_ovlp[j0, i0] = e1_ovlp[i0, j0].T

G_x_PI_AO = perturbed['G_x_PI_AO_integral']
G_x_PI_MO = cp.zeros((natm, 3, nmo, nmo))
for i in range(natm):
    for j in range(3):
        G_x_PI_MO[i, j] = mo_coeff.T @ G_x_PI_AO[i, j] @ mo_coeff

Gp_x_RI_AO = perturbed['Gp_x_RI_AO']
Gm_x_TI_AO = perturbed['Gm_x_TI_AO']
G_x_RI_MO = cp.zeros((natm, 3, nvir, nocc))
G_x_TI_MO = cp.zeros((natm, 3, nvir, nocc))
for i in range(natm):
    for j in range(3):
        G_x_RI_MO[i, j] = orbv.T @ Gp_x_RI_AO[i, j] @ orbo
        G_x_TI_MO[i, j] = orbv.T @ Gm_x_TI_AO[i, j] @ orbo

F_x_AO = perturbed['F_x_AO_integral']
F_x_MO = cp.zeros((natm, 3, nmo, nmo))
for i in range(natm):
    for j in range(3):
        F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff

P_I_prime_y_MO = perturbed['P_I_prime_y_MO']
W_I_y_MO = perturbed['W_I_y_MO']
P_y_MO = perturbed['P_y_MO']

e1_perturbed = cp.zeros((natm, natm, 3, 3))
for i0 in range(natm):
    for j0 in range(natm):
        for x in range(3):
            for y in range(3):
                t2 = -cp.trace(W_I_y_MO[j0, y] @ s1mo[i0, x]) * 2.0
                t3 = cp.trace(P_I_prime_y_MO[j0, y] @ F_x_MO[i0, x]) * 2.0
                t4 = cp.trace(P_y_MO[j0, y] @ G_x_PI_MO[i0, x]) * 2.0
                t5 = cp.sum(x1[j0, y] * G_x_RI_MO[i0, x].T) * 4.0
                t6 = cp.sum(y1[j0, y] * G_x_TI_MO[i0, x].T) * 4.0
                e1_perturbed[i0, j0, x, y] += t2 + t3 + t4 + t5 + t6


def sym(h):
    h = h.transpose(0, 2, 1, 3)
    return 0.5 * (h + h.transpose(2, 3, 0, 1))


# h_semi shape: (natm, 3, natm, 3), e1_* shape: (natm, natm, 3, 3)
# h_semi[atm_i, coord_x, atm_j, coord_y] vs e1[atm_i, atm_j, coord_x, coord_y]
for label, semi_idx, e1_idx in [
    ('atom0-z, atom0-z', (0, 2, 0, 2), (0, 0, 2, 2)),
    ('atom0-x, atom0-x', (0, 0, 0, 0), (0, 0, 0, 0)),
    ('atom0-z, atom1-z', (0, 2, 1, 2), (0, 1, 2, 2)),
]:
    ai, x, aj, y = semi_idx
    bi, bj, bx, by = e1_idx
    print(f'--- {label} ---')
    print(f'  e1_hcore     = {float(e1_hcore[bi,bj,bx,by]):10.6f}  (x2 = {float(e1_hcore[bi,bj,bx,by]*2):10.6f})')
    print(f'  ejk_PI       = {float(ejk_PI[bi,bj,bx,by]):10.6f}  (x2 = {float(ejk_PI[bi,bj,bx,by]*2):10.6f})')
    print(f'  e1_ovlp      = {float(e1_ovlp[bi,bj,bx,by]):10.6f}  (x2 = {float(e1_ovlp[bi,bj,bx,by]*2):10.6f})')
    print(f'  e1_perturbed = {float(e1_perturbed[bi,bj,bx,by]):10.6f}')
    tot_x2 = float(sym(e1_hcore*2 + ejk_PI*2 + e1_ovlp*2 + e1_perturbed)[ai, x, aj, y])
    tot_no2 = float(sym(e1_hcore + ejk_PI + e1_ovlp + e1_perturbed)[ai, x, aj, y])
    semi = float(h_semi[ai, x, aj, y])
    print(f'  Total (x2 1st3) = {tot_x2:10.6f}')
    print(f'  Total (no x2)   = {tot_no2:10.6f}')
    print(f'  h_semi          = {semi:10.6f}')
    print()
