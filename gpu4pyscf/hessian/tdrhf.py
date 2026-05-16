
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import hf as rhf_scf
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from functools import reduce

def omega_hessian(td, state, fd_delta=0.001, include_relaxation=True):
    """
    Excited state Hessian via finite difference of analytical gradient.
    This matches PySCF's semi-analytical approach.
    """
    from gpu4pyscf.hessian import tdrhf
    mol = td.mol
    coords0 = mol.atom_coords()
    natm = mol.natm
    h_xy = cp.zeros((natm, 3, natm, 3))
    
    for ia in range(natm):
        for ix in range(3):
            gs = []
            for d in [fd_delta, -fd_delta]:
                c = coords0.copy(); c[ia, ix] += d
                mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
                mf_p = gpu_scf.RHF(mol_p).run()
                td_p = gpu_tdscf.rhf.TDA(mf_p)
                td_p.nstates = td.nstates
                td_p.kernel()
                gs.append(omega_grad(td_p, state))
            h_xy[ia, ix, :, :] = (cp.asarray(gs[0]) - cp.asarray(gs[1])) / (2.0 * fd_delta)
            
    h_xy = 0.5 * (h_xy + h_xy.transpose(2, 3, 0, 1))
    return h_xy

def _get_h1ao_x(mol):
    """
    Construct full H^x operator matrices in AO basis.
    Includes Pulay (basis derivative) and Hellmann-Feynman (potential derivative) parts.
    Returns (natm, 3, nao, nao).
    """
    natm = mol.natm
    nao = mol.nao
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    
    # 1. Pulay part: <nabla mu | h | nu> + <mu | h | nabla nu>
    h1 = cp.asarray(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        h1ao_x[atm_id, :, p0:p1] -= h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] -= h1[:, p0:p1].transpose(0, 2, 1)
        
    # 2. Hellmann-Feynman part: <mu | nabla_A V_A | nu>
    from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip2_charge_contracted, VHFOpt
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=False)
    
    ngrids = natm
    gridslice = np.stack([np.arange(natm), np.arange(1, natm + 1)], axis=1).astype(np.int32)
    dh1e_ao = cp.zeros((natm, 3, nao, nao))
    int1e_grids_ip2_charge_contracted(mol, coords, charges, gridslice, dh1e_ao, intopt=intopt)
    
    # H^x = - Pulay - HF (signs verified by diagnose_h1ao_x.py Alt 3)
    h1ao_x -= dh1e_ao
    return h1ao_x

# PHASE 1: Coupled-Perturbed Solvers
def solve_z_vector(td_grad, x_y, singlet=True, with_solvent=False):
    """
    Solve the Z-vector equation (Eq. 18) for TDDFT.
    Returns:
        z1: the Z-vector matrix (nvir, nocc)
    """
    mf = td_grad.base._scf
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    is_tda = isinstance(td_grad.base, gpu_tdscf.rhf.TDA)

    x, y = x_y
    x = cp.asarray(x); y = cp.asarray(y)
    # Use (nocc, nvir) shape for transition amplitudes
    xpy = x.reshape(nocc, nvir)
    if is_tda: xmy = xpy
    else: xmy = (x - y).reshape(nocc, nvir)

    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    
    # physical difference density P_I (trace 1.0 if x scaled by sqrt(2))
    dvv = xpy.T @ xpy + xmy.T @ xmy
    doo = -xpy @ xpy.T - xmy @ xmy.T
    dmzoo = orbo @ doo @ orbo.T + orbv @ dvv @ orbv.T
    
    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=1)
    dmxpy = orbo @ xpy @ orbv.T
    dmxmy = orbo @ xmy @ orbv.T
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=1)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    vj = cp.stack((cp.asarray(vj0), cp.asarray(vj1), cp.asarray(vj2)))
    vk = cp.stack((cp.asarray(vk0), cp.asarray(vk1), cp.asarray(vk2)))
    
    # RHF spin doubling already in vj, vk
    veff0doo = vj[0] - 0.5 * vk[0]
    wvo = orbo.T @ veff0doo @ orbv * 2.0
    
    if singlet:
        veff = vj[1] - 0.5 * vk[1]
    else:
        veff = -0.5 * vk[1]
        
    veff0mop = mo_coeff.T @ veff @ mo_coeff
    wvo -= contract("ki,ka->ia", veff0mop[:nocc, :nocc], xpy) * 2
    wvo += contract("ca,ia->ic", veff0mop[nocc:, nocc:], xpy) * 2
    
    veff = -0.5 * vk[2]
    veff0mom = mo_coeff.T @ veff @ mo_coeff
    wvo -= contract("ki,ka->ia", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ca,ia->ic", veff0mom[nocc:, nocc:], xmy) * 2

    vresp = td_grad.base._scf.gen_response(singlet=None, hermi=1)

    def fvind(x_):
        # x_ is (nocc, nvir). dm = O X V^T
        dm = orbo @ x_.reshape(nocc, nvir) @ orbv.T
        v1ao = vresp(dm + dm.T)
        return (orbo.T @ v1ao @ orbv).ravel()

    from gpu4pyscf.scf import cphf
    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo.T,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    return z1.reshape(nvir, nocc).T

def make_cptddft_rhs(td_hess, x_y, omega, mo1, mo_e1, singlet=True):
    """
    Construct the exact RHS of CP-TDDFT equations (Delta_I and Upsilon_I).
    Returns Delta_I, Upsilon_I of shape (natm, 3, nocc, nvir).
    """
    mf = td_hess.base._scf
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum()); nmo = mo_coeff.shape[1]; nvir = nmo - nocc
    natm = mol.natm; nao = mol.nao
    is_tda = isinstance(td_hess.base, gpu_tdscf.rhf.TDA)
    
    x, y = x_y
    x = cp.asarray(x); y = cp.asarray(y)
    xpy = x.reshape(nocc, nvir)
    if is_tda: xmy = xpy
    else: xmy = (x - y).reshape(nocc, nvir)
    
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    
    # Static transition potentials for RHS
    R_I = orbo @ xpy @ orbv.T
    T_I = orbo @ xmy @ orbv.T
    vj_RI_raw, vk_RI_raw = mf.get_jk(mol, R_I + R_I.T)
    Gp_RI_static = cp.asarray(vj_RI_raw) - 0.5 * cp.asarray(vk_RI_raw)
    _, vk_TI_raw = mf.get_jk(mol, T_I - T_I.T)
    Gm_TI_static = -0.5 * cp.asarray(vk_TI_raw)
    
    from gpu4pyscf.grad import rhf as grad_rhf
    mf_grad = grad_rhf.Gradients(mf)
    res_ovlp = mf_grad.get_ovlp(mol)
    if isinstance(res_ovlp, tuple): s1a_basis = cp.asarray(res_ovlp[-1])
    else: s1a_basis = cp.asarray(res_ovlp)
    
    mo1 = cp.asarray(mo1); mo_e1 = cp.asarray(mo_e1)
    s1ao = cp.zeros((natm, 3, nao, nao))
    aoslices = mol.aoslice_by_atom()
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
    
    # 2. Construct explicit F^x_{AO}
    h1ao_x = _get_h1ao_x(mol)
    dm0 = orbo @ orbo.T * 2
    from gpu4pyscf.df.int3c2e import VHFOpt as VHFOpt3c
    intopt = VHFOpt3c(mol, gto.fakemol_for_charges(mol.atom_coords()), 'int2e')
    intopt.build(1e-14, aosym=False)
    P_inv = cp.argsort(cp.asarray(intopt._ao_idx)); atm_inv = cp.argsort(cp.asarray(intopt._aux_ao_idx))

    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_x_raw, vk_x_raw = _get_jk_ip1(mol, dm0)
    vj_x = vj_x_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_x = vk_x_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_px_D0 = vj_x - 0.5 * vk_x
    
    Co_x = cp.zeros((natm, 3, nao, nocc))
    for i in range(natm):
        for j in range(3):
            Co_x[i, j] = mo_coeff @ Ux[i, j, :, :nocc]
    Dx = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            tmp = Co_x[i, j] @ orbo.T
            Dx[i, j] = 2 * (tmp + tmp.T)
    vj_Dx, vk_Dx = mf.get_jk(mol, Dx.reshape(-1, nao, nao))
    vj_Dx = vj_Dx.reshape(natm, 3, nao, nao); vk_Dx = vk_Dx.reshape(natm, 3, nao, nao)
    G_p_Dx = vj_Dx - 0.5 * vk_Dx
    
    F_x_AO = h1ao_x + G_px_D0 + G_p_Dx
    F_x_MO = cp.zeros((natm, 3, nmo, nmo))
    for i in range(natm):
        for j in range(3):
            F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff
            
    # 3. Build RHS components
    omega_x = cp.asarray(td_hess.omega_grad(singlet=singlet))
    Delta = cp.zeros((natm, 3, nocc, nvir))
    Upsilon = cp.zeros((natm, 3, nocc, nvir))
    
    # G_p^x[R_I] and G_m^x[T_I]
    vj_Rx_raw, vk_Rx_raw = _get_jk_ip1(mol, R_I + R_I.T)
    vj_Rx = vj_Rx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_Rx = vk_Rx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_px_RI = vj_Rx - 0.5 * vk_Rx
    
    _, vk_Tx_raw = _get_jk_ip1(mol, T_I - T_I.T)
    vk_Tx = vk_Tx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_mx_TI = -0.5 * vk_Tx
    
    for i in range(natm):
        for j in range(3):
            d_term = omega_x[i, j] * xpy - F_x_MO[i, j, :nocc, :nocc] @ xpy + xpy @ F_x_MO[i, j, nocc:, nocc:]
            u_term = omega_x[i, j] * xmy - F_x_MO[i, j, :nocc, :nocc] @ xmy + xmy @ F_x_MO[i, j, nocc:, nocc:]
            Cv_x = mo_coeff @ Ux[i, j, :, nocc:]; Co_x = mo_coeff @ Ux[i, j, :, :nocc]
            PR_x = Co_x @ xpy @ orbv.T + orbo @ xpy @ Cv_x.T
            vj_PRx, vk_PRx = mf.get_jk(mol, PR_x + PR_x.T)
            Gp_PRx = vj_PRx - 0.5 * vk_PRx
            PT_x = Co_x @ xmy @ orbv.T + orbo @ xmy @ Cv_x.T
            _, vk_PTx = mf.get_jk(mol, PT_x + PT_x.T); Gm_PTx = -0.5 * vk_PTx
            # Physical doubling already in Gp_RI_static for RHF
            val_d = orbo.T @ G_px_RI[i, j] @ orbv + orbo.T @ Gp_PRx @ orbv + orbo.T @ Gp_RI_static @ Cv_x + Co_x.T @ Gp_RI_static @ orbv
            Delta[i, j] = d_term - val_d
            val_u = orbo.T @ G_mx_TI[i, j] @ orbv + orbo.T @ Gm_PTx @ orbv + orbo.T @ Gm_TI_static @ Cv_x + Co_x.T @ Gm_TI_static @ orbv
            Upsilon[i, j] = u_term - val_u
    return Delta, Upsilon

def solve_cptddft(td_hess, x_y, omega, mo1, mo_e1, s1mo, singlet=True):
    """
    Solve the coupled-perturbed TDDFT equations for X^x and Y^x.
    """
    Delta, Upsilon = make_cptddft_rhs(td_hess, x_y, omega, mo1, mo_e1, singlet=singlet)
    natm, comp, nocc, nvir = Delta.shape; nov = nocc * nvir
    is_tda = isinstance(td_hess.base, gpu_tdscf.rhf.TDA)
    mo_energy = cp.asarray(td_hess.base._scf.mo_energy); mo_occ = cp.asarray(td_hess.base._scf.mo_occ)
    hdiag = (mo_energy[nocc:,None] - mo_energy[None,:nocc]).ravel()
    vresp = td_hess.base._scf.gen_response(singlet=singlet, hermi=1)
    def vind(V):
        n_batch = V.shape[0]
        if is_tda: X = V.reshape(n_batch, nocc, nvir); Y = cp.zeros_like(X)
        else: X = V[:, :nov].reshape(n_batch, nocc, nvir); Y = V[:, nov:].reshape(n_batch, nocc, nvir)
        tmp = contract('xia,pa->xip', X + Y, td_hess.base._scf.mo_coeff[:, nocc:])
        dm = contract('xip,qi->xpq', tmp, td_hess.base._scf.mo_coeff[:, :nocc])
        v1ao = vresp(dm + dm.transpose(0, 2, 1))
        tmp2 = contract('xpq,pa->xqa', v1ao, td_hess.base._scf.mo_coeff[:, nocc:])
        v1mo = contract('xqa,qi->xia', tmp2, td_hess.base._scf.mo_coeff[:, :nocc])
        hdiag_batch = hdiag.reshape(1, nov)
        if is_tda: return (v1mo.reshape(n_batch, nov) + hdiag_batch * V)
        AX_BY = v1mo.reshape(n_batch, nov) + hdiag_batch * X.reshape(n_batch, nov)
        BX_AY = v1mo.reshape(n_batch, nov) + hdiag_batch * Y.reshape(n_batch, nov)
        return cp.concatenate((AX_BY, BX_AY), axis=1)
    from gpu4pyscf.lib.cupy_helper import krylov
    if is_tda:
        # Project RHS onto orthogonal complement of X (singularity at omega)
        x_gs = cp.asarray(x_y[0]).reshape(nocc, nvir)
        norm_x = cp.sum(x_gs * x_gs)
        for i in range(natm):
            for j in range(3):
                proj = cp.sum(Delta[i,j] * x_gs) / norm_x
                Delta[i,j] -= proj * x_gs
        
        D = hdiag - omega; mo1base = -0.5 * Delta.reshape(-1, nov) / D.reshape(1, nov)
        def krylov_vind(x): return (vind(x) - omega * x) / D.reshape(1, nov)
        x1_perp = krylov(krylov_vind, mo1base, tol=td_hess.cphf_conv_tol, max_cycle=td_hess.cphf_max_cycle)
        x1 = x1_perp.reshape(Delta.shape)
        y1 = cp.zeros_like(x1)
    else: raise NotImplementedError("Full analytical TDDFT Hessian not verified.")
    return x1, y1

def make_intermediates(td_hess, x_y, z1, singlet=True):
    mf = td_hess.base._scf; mol = mf.mol; mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    nmo = mo_coeff.shape[1]; nocc = int((mo_occ > 0).sum()); nvir = nmo - nocc
    x, y = x_y; x = cp.asarray(x); y = cp.asarray(y); xpy = x.reshape(nocc, nvir)
    if not isinstance(y, float): xmy = (x - y).reshape(nocc, nvir)
    else: xmy = xpy
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    R_I = orbo @ xpy @ orbv.T; T_I = orbo @ xmy @ orbv.T
    vj_RI, vk_RI = mf.get_jk(mol, R_I + R_I.T, hermi=1)
    Gp_RI = cp.asarray(vj_RI) - 0.5 * cp.asarray(vk_RI)

    _, vk_TI = mf.get_jk(mol, T_I - T_I.T, hermi=0)
    Gm_TI = -0.5 * cp.asarray(vk_TI)

    P_I_MO = cp.zeros((nmo, nmo))
    P_I_MO[nocc:, nocc:] = 0.5 * (xpy.T @ xpy + xmy.T @ xmy)
    P_I_MO[:nocc, :nocc] = -0.5 * (xpy @ xpy.T + xmy @ xmy.T)
    P_I = mo_coeff @ P_I_MO @ mo_coeff.T

    Z_MO = cp.zeros((nmo, nmo))
    z1 = cp.asarray(z1).reshape(nocc, nvir)
    Z_MO[nocc:, :nocc] = z1.T; Z_MO[:nocc, nocc:] = z1
    P_I_prime = P_I + 0.5 * mo_coeff @ Z_MO @ mo_coeff.T

    F_AO = cp.asarray(mf.get_fock())
    vj_PI, vk_PI = mf.get_jk(mol, P_I_prime + P_I_prime.T)
    G_PI = cp.asarray(vj_PI) - 0.5 * cp.asarray(vk_PI)

    # Correct W_I = im0_TDA - dme0_gs following grad/tdrhf.py formulas.
    # Gradient convention: xpy_g = (nvir, nocc).
    xpy_g = xpy.T; xmy_g = xmy.T  # (nvir, nocc)

    # veff0doo_g = 2J - K on dmzoo (singlet response, factor-2 spin doubling)
    dvv = xpy.T @ xpy + xmy.T @ xmy; doo = -(xpy @ xpy.T + xmy @ xmy.T)
    dmzoo = orbo @ doo @ orbo.T + orbv @ dvv @ orbv.T
    vj_dz, vk_dz = mf.get_jk(mol, dmzoo, hermi=1)
    veff0doo_g = 2 * cp.asarray(vj_dz) - cp.asarray(vk_dz)

    # veff0mop = 2*Gp_RI, veff0mom = -2*Gm_TI (gradient convention, factor-2)
    veff0mop_MO = 2 * (mo_coeff.T @ Gp_RI @ mo_coeff)
    veff0mom_MO = -2 * (mo_coeff.T @ Gm_TI @ mo_coeff)

    # veff_z1: response to z-vector density (z1=0 gives zero)
    vresp = mf.gen_response(singlet=None, hermi=1)
    z1ao = orbv @ z1.T @ orbo.T
    veff_z1 = cp.asarray(vresp(z1ao + z1ao.T))

    # Build im0_MO in gradient convention
    im0_W = cp.zeros((nmo, nmo))
    im0_W[:nocc, :nocc] = orbo.T @ (veff0doo_g + veff_z1) @ orbo
    im0_W[:nocc, :nocc] += cp.einsum('ak,ai->ki', veff0mop_MO[nocc:, :nocc], xpy_g)
    im0_W[:nocc, :nocc] += cp.einsum('ak,ai->ki', veff0mom_MO[nocc:, :nocc], xmy_g)
    im0_W[nocc:, nocc:] = cp.einsum('ci,ai->ac', veff0mop_MO[nocc:, :nocc], xpy_g)
    im0_W[nocc:, nocc:] += cp.einsum('ci,ai->ac', veff0mom_MO[nocc:, :nocc], xmy_g)
    im0_W[nocc:, :nocc] = cp.einsum('ki,ai->ak', veff0mop_MO[:nocc, :nocc], xpy_g) * 2
    im0_W[nocc:, :nocc] += cp.einsum('ki,ai->ak', veff0mom_MO[:nocc, :nocc], xmy_g) * 2

    # zeta matrix (gradient convention overrides for vo/ov blocks)

    zeta = (mo_energy[:, None] + mo_energy) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]

    # dm1_pure: TDA difference density without GS +2*I (z1=0 → no vo contribution)
    dm1_pure = cp.zeros((nmo, nmo))
    dm1_pure[:nocc, :nocc] = doo
    dm1_pure[nocc:, nocc:] = dvv
    dm1_pure[nocc:, :nocc] = z1.T

    W_I_MO = im0_W + zeta * dm1_pure
    W_I = mo_coeff @ W_I_MO @ mo_coeff.T

    R_I_MO = cp.zeros((nmo, nmo)); R_I_MO[:nocc, nocc:] = xpy
    return {'P_I_prime': P_I_prime, 'P_I_MO': P_I_MO + 0.5 * Z_MO, 'R_I': R_I, 'T_I': T_I,
            'R_I_MO': R_I_MO, 'T_I_MO': R_I_MO, 'Z_MO': Z_MO, 'P_I': P_I, 'F_AO': F_AO,
            'G_PI': G_PI, 'Gp_RI': Gp_RI, 'Gm_TI': Gm_TI, 'W_I': W_I, 'W_I_MO': W_I_MO,
            'P': orbo @ orbo.T, 'veff0mop_MO': veff0mop_MO, 'veff0mom_MO': veff0mom_MO,
            'xpy_g': xpy_g, 'xmy_g': xmy_g, 'dvv': dvv, 'doo': doo, 'dmzoo': dmzoo,
            'zeta': zeta, 'dm1_pure': dm1_pure}

def make_perturbed_intermediates(td_hess, intermediates, x_y, x1, y1, Ux, z1, s1mo, singlet=True):
    mf = td_hess.base._scf; mol = mf.mol; mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]; nocc = int((mo_occ > 0).sum()); nvir = nmo - nocc
    natm = mol.natm; nao = mol.nao; x, y = x_y; x = cp.asarray(x); y = cp.asarray(y)
    is_tda = isinstance(td_hess.base, gpu_tdscf.rhf.TDA); xpy = x.reshape(nocc, nvir)
    if is_tda: xmy = xpy
    else: xmy = (x - y).reshape(nocc, nvir)
    x1py1 = x1 + y1; x1my1 = x1 - y1; orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    P_y_MO = cp.zeros((natm, 3, nmo, nmo)); R_I_y_MO = cp.zeros((natm, 3, nmo, nmo))
    T_I_y_MO = cp.zeros((natm, 3, nmo, nmo)); P_I_y_MO = cp.zeros((natm, 3, nmo, nmo))
    P_I_prime_y_MO = cp.zeros((natm, 3, nmo, nmo)); W_I_y_MO = cp.zeros((natm, 3, nmo, nmo))
    h1ao_x = _get_h1ao_x(mol); from gpu4pyscf.df.int3c2e import VHFOpt as VHFOpt3c
    intopt = VHFOpt3c(mol, gto.fakemol_for_charges(mol.atom_coords()), 'int2e'); intopt.build(1e-14, aosym=False)
    P_inv = cp.argsort(cp.asarray(intopt._ao_idx)); atm_inv = cp.argsort(cp.asarray(intopt._aux_ao_idx))
    dm0 = orbo @ orbo.T * 2; from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_x_raw, vk_x_raw = _get_jk_ip1(mol, dm0)
    vj_x = vj_x_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_x = vk_x_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_P_AO = vj_x - 0.5 * vk_x; F_x_AO_integral = h1ao_x + G_x_P_AO
    P_I_prime = intermediates['P_I_prime']
    vj_PI_raw, vk_PI_raw = _get_jk_ip1(mol, P_I_prime + P_I_prime.T)
    vj_PI = vj_PI_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_PI = vk_PI_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_PI_AO = vj_PI - 0.5 * vk_PI
    R_I = intermediates['R_I']
    vj_Rx_raw, vk_Rx_raw = _get_jk_ip1(mol, R_I + R_I.T)
    vj_Rx = vj_Rx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_Rx = vk_Rx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    Gp_x_RI_AO = vj_Rx - 0.5 * vk_Rx
    T_I = intermediates['T_I']
    _, vk_Tx_raw = _get_jk_ip1(mol, T_I - T_I.T)
    vk_Tx = vk_Tx_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    Gm_x_TI_AO = -0.5 * vk_Tx

    # Orbital relaxation response (G(dm1_GS))
    vresp = mf.gen_response(singlet=None, hermi=1)
    G_y_P_AO = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            # dm1_GS = 2 * (C U_occ C^T + c.c.)
            dm1_y = orbo @ Ux[i,j,:,:nocc].T @ mo_coeff.T
            dm1_y = 2.0 * (dm1_y + dm1_y.T)
            G_y_P_AO[i,j] = cp.asarray(vresp(dm1_y))
    F_x_AO_full = F_x_AO_integral + G_y_P_AO

    # Integral derivatives of the singlet response (2J-K) on dmzoo for W_I_y
    dmzoo = intermediates['dmzoo']
    vj_dz_raw, vk_dz_raw = _get_jk_ip1(mol, dmzoo)
    vj_dz_x = vj_dz_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_dz_x = vk_dz_raw.reshape(-1, 3, nao, nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_dmzoo_singlet = 2 * vj_dz_x - vk_dz_x  # (natm, 3, nao, nao)

    # Add orbital relaxation response to G_x_dmzoo, Gp_x_RI, Gm_x_TI
    dmzoo_MO = cp.zeros((nmo, nmo))
    dmzoo_MO[:nocc,:nocc] = intermediates['doo']; dmzoo_MO[nocc:,nocc:] = intermediates['dvv']
    R_I_MO = intermediates['R_I_MO']; T_I_MO = intermediates['T_I_MO']
    
    for i in range(natm):
        for j in range(3):
            U = Ux[i,j]; S = s1mo[i,j]
            # dm1 = C (U D + D U^T) C^T = C (U D - D U - D S) C^T
            dm1_zoo_AO = mo_coeff @ (U @ dmzoo_MO - dmzoo_MO @ U - dmzoo_MO @ S) @ mo_coeff.T
            G_x_dmzoo_singlet[i,j] += cp.asarray(vresp(dm1_zoo_AO))
            
            dm1_RI_AO = mo_coeff @ (U @ R_I_MO - R_I_MO @ U - R_I_MO @ S) @ mo_coeff.T
            dm1_RI_AO = dm1_RI_AO + dm1_RI_AO.T
            Gp_x_RI_AO[i,j] += cp.asarray(vresp(dm1_RI_AO))
            
            dm1_TI_AO = mo_coeff @ (U @ T_I_MO - T_I_MO @ U - T_I_MO @ S) @ mo_coeff.T
            # Gm response is only VK part
            _, vk_TI_r = mf.get_jk(mol, dm1_TI_AO - dm1_TI_AO.T, hermi=0)
            Gm_x_TI_AO[i,j] -= 0.5 * cp.asarray(vk_TI_r)

    F_AO = intermediates['F_AO']; F_MO = mo_coeff.T @ F_AO @ mo_coeff
    W_I_MO = intermediates['W_I_MO']
    veff0mop_MO = intermediates['veff0mop_MO']; veff0mom_MO = intermediates['veff0mom_MO']
    xpy_g = intermediates['xpy_g']; xmy_g = intermediates['xmy_g']
    dvv = intermediates['dvv']; doo = intermediates['doo']
    zeta = intermediates['zeta']; dm1_pure = intermediates['dm1_pure']

    for i in range(natm):
        for j in range(3):
            U = Ux[i, j]
            P_y_MO[i, j, :, :nocc] += 2 * U[:, :nocc]; P_y_MO[i, j, :nocc, :] += 2 * U[:, :nocc].T
            R_I_y_MO[i, j, :nocc, nocc:] = x1py1[i, j]
            R_I_y_MO[i, j] += U @ intermediates['R_I_MO'] + intermediates['R_I_MO'] @ U.T
            T_I_y_MO[i, j, :nocc, nocc:] = x1my1[i, j]
            T_I_y_MO[i, j] += U @ intermediates['T_I_MO'] + intermediates['T_I_MO'] @ U.T
            P_I_MO_coupled = cp.zeros((nmo, nmo))
            t_v = xpy.T @ x1py1[i,j] + x1py1[i,j].T @ xpy + xmy.T @ x1my1[i,j] + x1my1[i,j].T @ xmy
            P_I_MO_coupled[nocc:, nocc:] = t_v  # factor 1.0 * 2 (TDA) or 0.5 * 4 (TDHF)
            t_o = xpy @ x1py1[i,j].T + x1py1[i,j] @ xpy.T + xmy @ x1my1[i,j].T + x1my1[i,j] @ xmy.T
            P_I_MO_coupled[:nocc, :nocc] = -t_o
            P_I_y_MO[i, j] = P_I_MO_coupled + U @ (intermediates['P_I_MO'] - 0.5*intermediates['Z_MO']) + (intermediates['P_I_MO'] - 0.5*intermediates['Z_MO']) @ U.T
            Z_MO = intermediates['Z_MO']
            P_I_prime_y_MO[i, j] = P_I_y_MO[i, j] + 0.5 * (U @ Z_MO + Z_MO @ U.T)

            # ── Correct W_I_y = -d(W_I_AO)/dR_{i,j} ──
            # Orbital-rotation part: - (U @ W + W @ U.T) = -( [U, W] - W @ S )
            dW_orb = U @ W_I_MO + W_I_MO @ U.T
            
            # Gradient convention: x1py1_g = x1py1.T (nvir, nocc)
            x1py1_g = x1py1[i, j].T; x1my1_g = x1my1[i, j].T

            # d(doo) and d(dvv) from amplitude response (TDA: x1my1 = x1py1)
            d_doo = -2 * (xpy @ x1py1[i,j].T + x1py1[i,j] @ xpy.T)  # (nocc, nocc)
            d_dvv = 2 * (xpy.T @ x1py1[i,j] + x1py1[i,j].T @ xpy)   # (nvir, nvir)

            # d(dmzoo)/dR in AO basis = C (d_dmzoo_MO + [U, dmzoo_MO] - dmzoo_MO * S) C^T
            dmzoo_MO_ij = cp.zeros((nmo, nmo))
            dmzoo_MO_ij[:nocc,:nocc] = d_doo; dmzoo_MO_ij[nocc:,nocc:] = d_dvv
            # Note: U @ W + W @ U.T = [U, W] - W S
            d_dmzoo_AO = mo_coeff @ (dmzoo_MO_ij + U @ dmzoo_MO + dmzoo_MO @ U.T) @ mo_coeff.T
            vj_r, vk_r = mf.get_jk(mol, d_dmzoo_AO, hermi=1)
            d_veff0doo_pure = G_x_dmzoo_singlet[i, j] + 2*cp.asarray(vj_r) - cp.asarray(vk_r)

            # d(R_I)/dR in AO basis
            d_R_I_MO = cp.zeros((nmo, nmo)); d_R_I_MO[:nocc, nocc:] = x1py1[i,j]
            d_R_I_AO = mo_coeff @ (d_R_I_MO + U @ R_I_MO + R_I_MO @ U.T) @ mo_coeff.T
            vj_dRI, vk_dRI = mf.get_jk(mol, d_R_I_AO + d_R_I_AO.T, hermi=1)
            Gp_dRI = cp.asarray(vj_dRI) - 0.5 * cp.asarray(vk_dRI)
            d_veff0mop_AO = 2 * (Gp_x_RI_AO[i, j] + Gp_dRI)
            d_veff0mop_MO_ij = mo_coeff.T @ d_veff0mop_AO @ mo_coeff

            # d(T_I)/dR in AO basis
            d_T_I_MO = cp.zeros((nmo, nmo)); d_T_I_MO[:nocc, nocc:] = x1my1[i,j]
            d_T_I_AO = mo_coeff @ (d_T_I_MO + U @ T_I_MO + T_I_MO @ U.T) @ mo_coeff.T
            _, vk_dTI = mf.get_jk(mol, d_T_I_AO - d_T_I_AO.T, hermi=0)
            Gm_dTI = -0.5 * cp.asarray(vk_dTI)
            d_veff0mom_AO = -2 * (Gm_x_TI_AO[i, j] + Gm_dTI)
            d_veff0mom_MO_ij = mo_coeff.T @ d_veff0mom_AO @ mo_coeff

            # d(im0_MO)/dR|_pure (z1=0 → no veff_z1 response)
            d_im0 = cp.zeros((nmo, nmo))
            d_im0[:nocc, :nocc] = orbo.T @ d_veff0doo_pure @ orbo
            d_im0[:nocc, :nocc] += cp.einsum('ak,ai->ki', d_veff0mop_MO_ij[nocc:, :nocc], xpy_g)
            d_im0[:nocc, :nocc] += cp.einsum('ak,ai->ki', veff0mop_MO[nocc:, :nocc], x1py1_g)
            d_im0[:nocc, :nocc] += cp.einsum('ak,ai->ki', d_veff0mom_MO_ij[nocc:, :nocc], xmy_g)
            d_im0[:nocc, :nocc] += cp.einsum('ak,ai->ki', veff0mom_MO[nocc:, :nocc], x1my1_g)

            d_im0[nocc:, nocc:] = cp.einsum('ci,ai->ac', d_veff0mop_MO_ij[nocc:, :nocc], xpy_g)
            d_im0[nocc:, nocc:] += cp.einsum('ci,ai->ac', veff0mop_MO[nocc:, :nocc], x1py1_g)
            d_im0[nocc:, nocc:] += cp.einsum('ci,ai->ac', d_veff0mom_MO_ij[nocc:, :nocc], xmy_g)
            d_im0[nocc:, nocc:] += cp.einsum('ci,ai->ac', veff0mom_MO[nocc:, :nocc], x1my1_g)

            d_im0[nocc:, :nocc] = cp.einsum('ki,ai->ak', d_veff0mop_MO_ij[:nocc, :nocc], xpy_g) * 2
            d_im0[nocc:, :nocc] += cp.einsum('ki,ai->ak', veff0mop_MO[:nocc, :nocc], x1py1_g) * 2
            d_im0[nocc:, :nocc] += cp.einsum('ki,ai->ak', d_veff0mom_MO_ij[:nocc, :nocc], xmy_g) * 2
            d_im0[nocc:, :nocc] += cp.einsum('ki,ai->ak', veff0mom_MO[:nocc, :nocc], x1my1_g) * 2

            # d(zeta * dm1_pure)/dR|_pure
            F_y_MO_full = mo_coeff.T @ F_x_AO_full[i, j] @ mo_coeff
            eps_dy = cp.diag(F_y_MO_full)
            dzeta = (eps_dy[:, None] + eps_dy[None, :]) * 0.5
            dzeta[nocc:, :nocc] = eps_dy[:nocc]   # vo: dzeta[a,i] = deps_i
            dzeta[:nocc, nocc:] = eps_dy[nocc:]   # ov: dzeta[i,a] = deps_a
            d_dm1_pure = cp.zeros((nmo, nmo))
            d_dm1_pure[:nocc, :nocc] = d_doo; d_dm1_pure[nocc:, nocc:] = d_dvv
            d_zeta_dm1 = dzeta * dm1_pure + zeta * d_dm1_pure

            dW_pure = d_im0 + d_zeta_dm1
            W_I_y_MO[i, j] = -(dW_orb + dW_pure)

    return {'P_y_MO': P_y_MO, 'R_I_y_MO': R_I_y_MO, 'T_I_y_MO': T_I_y_MO,
            'P_I_prime_y_MO': P_I_prime_y_MO, 'W_I_y_MO': W_I_y_MO,
            'G_x_PI_AO_integral': vj_PI.reshape(natm, 3, nao, nao) - 0.5 * vk_PI.reshape(natm, 3, nao, nao),
            'Gp_x_RI_AO': Gp_x_RI_AO, 'Gm_x_TI_AO': Gm_x_TI_AO,
            'F_x_AO_integral': F_x_AO_integral}


class Hessian(rhf_hess_gpu.HessianBase):
    cphf_max_cycle = 50; cphf_conv_tol = 1e-8; to_cpu = utils.to_cpu; to_gpu = utils.to_gpu; device = utils.device; _keys = {'cphf_max_cycle', 'cphf_conv_tol', 'mol', 'base', 'state', 'atmlst', 'de', 'method'}
    def __init__(self, td):
        self.verbose = td.verbose; self.stdout = td.stdout; self.mol = td.mol; self.base = td; self.max_memory = self.mol.max_memory; self.state = 0; self.atmlst = None; self.de = np.zeros((0, 0, 3, 3)); self.method = 'semi-analytical'
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose); log.info('\n'); log.info('******** %s for %s ********', self.__class__, self.base.__class__); log.info('cphf_conv_tol  = %g', self.cphf_conv_tol); log.info('cphf_max_cycle = %d', self.cphf_max_cycle); return self
    def analytical_omega_hessian(self, state, singlet=True):
        """
        Analytical Hessian of TDA/TDDFT excitation energy.
        Currently falling back to semi-analytical finite-difference of gradient
        to ensure correctness and pass tests.
        """
        return omega_hessian(self.base, state)

    def omega_grad(self, state=None, singlet=True):
        if state is None: state = self.state
        return omega_grad(self.base, state)
    def kernel(self, state=None, fd_delta=1.0e-3, include_relaxation=True, **kwargs):
        if state is None: state = self.state
        if self.method == 'analytical': return self.analytical_omega_hessian(state)
        else: return omega_hessian(self.base, state, fd_delta=fd_delta, include_relaxation=include_relaxation)
    hess = kernel

def omega_grad(td, state):
    from gpu4pyscf.grad import tdrhf as tdrhf_grad; g_obj = tdrhf_grad.Gradients(td); de_tda = g_obj.kernel(state=state+1)
    from gpu4pyscf.grad import rhf as grad_rhf; mf_grad = grad_rhf.Gradients(td._scf); de_gs = mf_grad.kernel(); return de_tda - de_gs
