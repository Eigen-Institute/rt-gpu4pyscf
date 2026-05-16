"""
Decompose the analytical Hessian into static vs perturbed components for H2/STO-3G.
Compares each piece against the semi-analytical (FD on gradient) reference.

Static  = e1_hcore + ejk_PI + e1_ovlp  (no CP-TDDFT response)
Perturbed = e1_perturbed                (CP-TDDFT response contribution)
"""
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf import scf as gpu_scf
from pyscf import gto

# ── system ─────────────────────────────────────────────────────────────────
mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
mf = gpu_scf.RHF(mol).run()
td = gpu_tdscf.rhf.TDA(mf)
td.nstates = 1
td.kernel()
state = 0

# ── reference: semi-analytical ─────────────────────────────────────────────
h_semi_obj = tdrhf_hess.Hessian(td)
h_semi_obj.method = 'semi-analytical'
h_semi = h_semi_obj.kernel()  # (natm, 3, natm, 3)
print("Semi-analytical Hessian:")
natm = mol.natm
for i in range(natm):
    for j in range(natm):
        for x in range(3):
            for y in range(3):
                v = float(h_semi[i, x, j, y])
                if abs(v) > 1e-6:
                    print(f"  h_semi[{i},{x},{j},{y}] = {v:12.8f}")

# ── build analytical Hessian components ────────────────────────────────────
h_obj = tdrhf_hess.Hessian(td)
mf_ = h_obj.base._scf
mol_ = mf_.mol

# Amplitudes — test both unscaled and scaled
for label, scale in [('unscaled (PySCF)', 1.0), ('scaled (sqrt2)', float(cp.sqrt(2)))]:
    print(f"\n{'='*60}")
    print(f"Amplitude normalization: {label}")

    x_y_raw = td.xy[state]
    x_y = tuple([cp.asarray(v) * scale for v in x_y_raw])
    omega = td.e[state]

    mo_coeff = cp.asarray(mf_.mo_coeff)
    mo_occ = cp.asarray(mf_.mo_occ)
    mo_energy = cp.asarray(mf_.mo_energy)
    nocc = int((mo_occ > 0).sum())
    nmo = mo_coeff.shape[1]
    nvir = nmo - nocc
    nao = mol_.nao

    # ground-state MO responses
    mf_hess = rhf_hess_gpu.Hessian(mf_)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)

    from gpu4pyscf.hessian.rhf import get_ovlp
    _, _, s1a_basis = get_ovlp(mol_)
    s1a_basis = cp.asarray(s1a_basis)
    aoslices = mol_.aoslice_by_atom()

    s1ao = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)

    s1mo = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff

    Ux = cp.zeros((natm, 3, nmo, nmo))
    Ux[:,:,:,:nocc] = mo1
    Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
    Ux[:,:,:nocc,:nocc] = -0.5 * s1mo[:,:,:nocc,:nocc]
    Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]

    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, x_y)
    x1, y1 = tdrhf_hess.solve_cptddft(h_obj, x_y, omega, mo1, mo_e1)
    intermediates = tdrhf_hess.make_intermediates(h_obj, x_y, z1)
    perturbed_intermediates = tdrhf_hess.make_perturbed_intermediates(
        h_obj, intermediates, x_y, x1, y1, Ux, z1)

    from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2
    de_hcore = _e_hcore_generator(h_obj, intermediates['P_I_prime'])
    e1_hcore = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        for j0 in range(i0+1):
            e1_hcore[i0, j0] += de_hcore(i0, j0)
            e1_hcore[j0, i0] = e1_hcore[i0, j0].T

    vhfopt = mf_._opt_gpu.get(mol_.omega)
    P_I_prime = intermediates['P_I_prime']
    P_GS = intermediates['P']
    R_I = intermediates['R_I']
    T_I = intermediates['T_I']
    ejk_PI = _partial_ejk_ip2(mol_, P_I_prime + P_GS, vhfopt)
    ejk_PI -= _partial_ejk_ip2(mol_, P_GS, vhfopt)
    ejk_RI = _partial_ejk_ip2(mol_, R_I + R_I.T, vhfopt)
    ejk_PI += 0.5 * ejk_RI
    ejk_TI = _partial_ejk_ip2(mol_, T_I - T_I.T, vhfopt, j_factor=0.0)
    ejk_PI -= 0.5 * ejk_TI

    s1aa, s1ab, _ = get_ovlp(mol_)
    s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    e1_ovlp = cp.zeros((natm, natm, 3, 3))
    W_I = intermediates['W_I']
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        e1_ovlp[i0, i0] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1]) * 2
        for j0 in range(i0+1):
            q0, q1 = aoslices[j0][2:]
            e1_ovlp[i0, j0] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1]) * 2
            e1_ovlp[j0, i0] = e1_ovlp[i0, j0].T

    F_x_AO = perturbed_intermediates['F_x_AO']
    F_x_MO = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff

    G_x_PI_AO = perturbed_intermediates['G_x_PI_AO']
    G_x_PI_MO = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            G_x_PI_MO[i, j] = mo_coeff.T @ G_x_PI_AO[i, j] @ mo_coeff

    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    Gp_x_RI_AO = perturbed_intermediates['Gp_x_RI_AO']
    Gm_x_TI_AO = perturbed_intermediates['Gm_x_TI_AO']
    G_x_RI_MO = cp.zeros((natm, 3, nvir, nocc))
    G_x_TI_MO = cp.zeros((natm, 3, nvir, nocc))
    for i in range(natm):
        for j in range(3):
            G_x_RI_MO[i, j] = orbv.T @ Gp_x_RI_AO[i, j] @ orbo
            G_x_TI_MO[i, j] = orbv.T @ Gm_x_TI_AO[i, j] @ orbo

    P_I_prime_y_MO = perturbed_intermediates['P_I_prime_y_MO']
    L_I_prime_y_MO = perturbed_intermediates['L_I_prime_y_MO']
    W_I_y_MO = perturbed_intermediates['W_I_y_MO']
    P_y_MO = perturbed_intermediates['P_y_MO']

    e1_perturbed = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        for j0 in range(natm):
            for x in range(3):
                for y in range(3):
                    tmp_U_S = 2 * Ux[i0, x] + s1mo[i0, x]
                    term1 = cp.trace(L_I_prime_y_MO[j0, y] @ tmp_U_S)
                    term2 = cp.trace(W_I_y_MO[j0, y] @ s1mo[i0, x])
                    term3 = cp.trace(P_I_prime_y_MO[j0, y] @ F_x_MO[i0, x])
                    term4 = cp.trace(P_y_MO[j0, y] @ G_x_PI_MO[i0, x])
                    term5 = cp.sum(x1[j0, y] * G_x_RI_MO[i0, x].T)
                    term6 = cp.sum(y1[j0, y] * G_x_TI_MO[i0, x].T)
                    e1_perturbed[i0, j0, x, y] += term1 + term2 + term3 + term4 + term5 + term6

    e1_static = e1_hcore + ejk_PI + e1_ovlp

    # symmetrize
    def symm(h):
        h = h.transpose(0, 2, 1, 3)
        return 0.5 * (h + h.transpose(2, 3, 0, 1))

    h_static = symm(e1_static)
    h_perturbed = symm(e1_perturbed)
    h_total = h_static + h_perturbed

    print("\nComponent breakdown vs semi-analytical reference:")
    print(f"{'Element':16s} {'Semi':>12s} {'Static':>12s} {'Perturbed':>12s} {'Total':>12s} {'Ratio(tot)':>10s}")
    for i in range(natm):
        for j in range(natm):
            for x in range(3):
                for y in range(3):
                    vs = float(h_semi[i, x, j, y])
                    vs_static = float(h_static[i, x, j, y])
                    vs_pert = float(h_perturbed[i, x, j, y])
                    vs_tot = float(h_total[i, x, j, y])
                    if abs(vs) > 1e-5 or abs(vs_tot) > 1e-5:
                        ratio = vs_tot / vs if abs(vs) > 1e-8 else float('nan')
                        print(f"[{i},{x},{j},{y}]         {vs:12.8f} {vs_static:12.8f} {vs_pert:12.8f} {vs_tot:12.8f} {ratio:10.4f}")

    print(f"\nTI violation: static={float(cp.abs(h_static.sum(axis=2)).max()):.3e}, "
          f"perturbed={float(cp.abs(h_perturbed.sum(axis=2)).max()):.3e}, "
          f"total={float(cp.abs(h_total.sum(axis=2)).max()):.3e}")

if __name__ == '__main__':
    pass
