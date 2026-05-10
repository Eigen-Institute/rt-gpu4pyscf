import numpy as np
import cupy as cp
from pyscf import lib
from pyscf import gto
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import hf as rhf_scf
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.scf import cphf
from functools import reduce

def omega_grad(td, state, atmlst=None, with_solvent=False, singlet=True):
    '''Verified analytical gradient from tdrhf_grad engine.'''
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    de_tda_elec = td_grad_obj.grad_elec(td.xy[state], singlet=singlet, atmlst=atmlst, with_solvent=with_solvent)
    mf_grad = td._scf.nuc_grad_method()
    de_gs_elec = mf_grad.grad_elec(atmlst=atmlst)
    return np.asarray(de_tda_elec) - np.asarray(de_gs_elec)

def omega_hessian(td, state, fd_delta=1.0e-3, include_relaxation=True):
    '''Robust semi-analytical Hessian (FD on analytical gradient).'''
    from gpu4pyscf import scf as gpu_scf
    from gpu4pyscf import tdscf as gpu_tdscf
    mf = td._scf; mol = td.mol; natm = mol.natm; coords0 = mol.atom_coords()
    h_xy = cp.zeros((natm, 3, natm, 3))
    
    for ia in range(natm):
        for ix in range(3):
            g_pm = []
            for d in [fd_delta, -fd_delta]:
                c = coords0.copy(); c[ia, ix] += d
                mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
                mf_p = gpu_scf.RHF(mol_p).run()
                td_p = gpu_tdscf.rhf.TDA(mf_p)
                td_p.nstates = td.nstates
                td_p.kernel()
                g_pm.append(omega_grad(td_p, state))
            h_xy[:, :, ia, ix] = (cp.asarray(g_pm[0]) - cp.asarray(g_pm[1])) / (2.0 * fd_delta)
            
    h_xy = 0.5 * (h_xy + h_xy.transpose(2,3,0,1))
    return h_xy

# PHASE 1: Coupled-Perturbed Solvers
def solve_z_vector(td_grad, x_y, singlet=True, with_solvent=False):
    """
    Solve the Z-vector equation (Eq. 18) for TDDFT.
    Returns:
        z1: the Z-vector matrix (nvir, nocc)
    """
    mf = td_grad.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]
    
    dvv = contract("ai,bi->ab", xpy, xpy) + contract("ai,bi->ab", xmy, xmy)  # 2 T_{ab}
    doo = -contract("ai,aj->ij", xpy, xpy) - contract("ai,aj->ij", xmy, xmy)  # 2 T_{ij}
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))  # (X+Y) in ao basis
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))  # (X-Y) in ao basis
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))  # T_{ij}*2 in ao basis
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))  # T_{ij}*2 + T_{ab}*2 in ao basis
    
    if with_solvent:
        # TODO: handle solvent properly if needed here
        pass

    vj0, vk0 = mf.get_jk(td_grad.mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(td_grad.mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(td_grad.mol, dmxmy - dmxmy.T, hermi=0)
    vj = cp.stack((cp.asarray(vj0), cp.asarray(vj1), cp.asarray(vj2)))
    vk = cp.stack((cp.asarray(vk0), cp.asarray(vk1), cp.asarray(vk2)))
    
    veff0doo = vj[0] * 2 - vk[0]
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2
    
    if singlet:
        veff = vj[1] * 2 - vk[1]
    else:
        veff = -vk[1]
        
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2
    
    veff = -vk[2]
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    vresp = td_grad.base.gen_response(singlet=None, hermi=1)
    def fvind(x_):
        dm = reduce(cp.dot, (orbv, x_.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=td_grad.cphf_max_cycle,
                    tol=td_grad.cphf_conv_tol)[0]
    return z1.reshape(nvir, nocc)

def make_cptddft_rhs(td_hess, x_y, omega, mo1, mo_e1, singlet=True):
    """
    Construct the exact RHS of CP-TDDFT equations (Delta_I and Upsilon_I).
    Returns Delta_I, Upsilon_I of shape (natm, 3, nocc, nvir).
    """
    mf = td_hess.base._scf
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    nocc = int((mo_occ > 0).sum())
    nvir = mo_coeff.shape[1] - nocc
    natm = mol.natm
    nao = mol.nao
    
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = x + y
    xmy = x - y
    
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    
    # 1. Build full U^x matrix (MO responses)
    from gpu4pyscf.grad import rhf as grad_rhf
    mf_grad = grad_rhf.Gradients(mf)
    s1ao = cp.asarray(mf_grad.get_ovlp(mol)) # (natm, 3, nao, nao)
    
    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            tmp = s1ao[i, j] @ mo_coeff
            s1mo[i, j] = mo_coeff.T @ tmp
            
    Ux = cp.zeros((natm, 3, nao, nao))
    Ux[:,:,:,:nocc] = mo1
    Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
    Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]
    
    # 2. Construct explicit F^x_{AO}
    from gpu4pyscf.df import int3c2e
    
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
        
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
                 
    eye_ao = cp.eye(nao)
    eye_ao_sorted = intopt.sort_orbitals(eye_ao, axis=[0])
    
    dh1e_ao = cp.zeros([natm, 3, nao, nao])
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e_ao[k0:k1, :, j0:j1, :] += contract('xkji,io->kxjo', int3c_blk, eye_ao_sorted[i0:i1])
        dh1e_ao[k0:k1, :, i0:i1, :] += contract('xkji,jo->kxio', int3c_blk, eye_ao_sorted[j0:j1])
        
    dh1e_ao = contract('kxjo,k->kxjo', dh1e_ao, -charges)
    P_sort = intopt.sort_orbitals(np.eye(nao), axis=[0])
    P_sort = cp.asarray(P_sort)
    dh1e_ao_unsorted = contract('pj,kxjo->kxpo', P_sort.T, dh1e_ao)
    h1ao_x += dh1e_ao_unsorted
    
    dm0 = orbo @ orbo.T * 2
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_x, vk_x = _get_jk_ip1(mol, dm0)
    vj_x = vj_x.reshape(natm, 3, nao, nao)
    vk_x = vk_x.reshape(natm, 3, nao, nao)
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
    vj_Dx = vj_Dx.reshape(natm, 3, nao, nao)
    vk_Dx = vk_Dx.reshape(natm, 3, nao, nao)
    G_p_Dx = vj_Dx - 0.5 * vk_Dx
    
    F_x_AO = h1ao_x + G_px_D0 + G_p_Dx
    F_x_MO = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff
            
    # 3. Build RHS components
    omega_x = cp.asarray(td_hess.omega_grad(singlet=singlet)) # (natm, 3)
    Delta = cp.zeros((natm, 3, nocc, nvir))
    Upsilon = cp.zeros((natm, 3, nocc, nvir))
    
    R_I = orbv @ xpy.T @ orbo.T
    T_I = orbv @ xmy.T @ orbo.T
    
    # G_p^x[R_I] and G_m^x[T_I]
    vj_Rx, vk_Rx = _get_jk_ip1(mol, R_I + R_I.T)
    vj_Rx = vj_Rx.reshape(natm, 3, nao, nao)
    vk_Rx = vk_Rx.reshape(natm, 3, nao, nao)
    vj_Rx = vj_Rx + vj_Rx.transpose(0,1,3,2)
    vk_Rx = vk_Rx + vk_Rx.transpose(0,1,3,2)
    G_px_RI = vj_Rx - 0.5 * vk_Rx
    
    vj_Tx, vk_Tx = _get_jk_ip1(mol, T_I + T_I.T)
    vj_Tx = vj_Tx.reshape(natm, 3, nao, nao)
    vk_Tx = vk_Tx.reshape(natm, 3, nao, nao)
    vj_Tx = vj_Tx + vj_Tx.transpose(0,1,3,2)
    vk_Tx = vk_Tx + vk_Tx.transpose(0,1,3,2)
    G_mx_TI = -0.5 * vk_Tx
    
    for i in range(natm):
        for j in range(3):
            # Base terms
            d_term = omega_x[i, j] * xmy.T - F_x_MO[i, j, nocc:, nocc:] @ xpy.T + xpy.T @ F_x_MO[i, j, :nocc, :nocc]
            u_term = omega_x[i, j] * xpy.T - F_x_MO[i, j, nocc:, nocc:] @ xmy.T + xmy.T @ F_x_MO[i, j, :nocc, :nocc]
            
            # Additional terms
            Cv_x = mo_coeff @ Ux[i, j, :, nocc:]
            Co_x = mo_coeff @ Ux[i, j, :, :nocc]
            
            PR_x = Cv_x @ xpy.T @ orbo.T + orbv @ xpy.T @ Co_x.T
            vj_PRx, vk_PRx = mf.get_jk(mol, PR_x + PR_x.T)
            Gp_PRx = vj_PRx - 0.5 * vk_PRx
            
            PT_x = Cv_x @ xmy.T @ orbo.T + orbv @ xmy.T @ Co_x.T
            _, vk_PTx = mf.get_jk(mol, PT_x + PT_x.T)
            Gm_PTx = -0.5 * vk_PTx
            
            # Add to Delta
            val_d = orbv.T @ G_px_RI[i, j] @ orbo + orbv.T @ Gp_PRx @ orbo + Cv_x.T @ (vj_Rx[i,j] - 0.5*vk_Rx[i,j]) @ orbo + orbv.T @ (vj_Rx[i,j] - 0.5*vk_Rx[i,j]) @ Co_x
            Delta[i, j] = d_term.T - val_d.T
            
            # Add to Upsilon
            val_u = orbv.T @ G_mx_TI[i, j] @ orbo + orbv.T @ Gm_PTx @ orbo + Cv_x.T @ (-0.5*vk_Tx[i,j]) @ orbo + orbv.T @ (-0.5*vk_Tx[i,j]) @ Co_x
            Upsilon[i, j] = u_term.T - val_u.T

    return Delta, Upsilon

def solve_cptddft(td_hess, x_y, omega, mo1, mo_e1, singlet=True):
    """
    Solve CP-TDDFT equations for X^x and Y^x. (Eqs. 20, 21)
    """
    # Uses make_cptddft_rhs to construct the right-hand sides
    Delta, Upsilon = make_cptddft_rhs(td_hess, x_y, omega, mo1, mo_e1, singlet=singlet)
    
    td = td_hess.base
    from gpu4pyscf.tdscf import rhf as tdscf_rhf
    is_tda = isinstance(td, tdscf_rhf.TDA)
    
    nocc = int((td._scf.mo_occ > 0).sum())
    nvir = td._scf.mo_occ.shape[0] - nocc
    nov = nocc * nvir
    
    vind, hdiag = td.gen_vind()
    
    from gpu4pyscf.lib.cupy_helper import krylov
    
    if is_tda:
        D = hdiag.ravel() - omega
        mo1base = Delta.reshape(-1, nov) / D
        
        def krylov_vind(x):
            Kx = vind(x) - hdiag.ravel() * x
            return -Kx / D
            
        x1 = krylov(krylov_vind, mo1base, tol=td_hess.cphf_conv_tol, max_cycle=td_hess.cphf_max_cycle)
        x1 = x1.reshape(Delta.shape)
        y1 = cp.zeros_like(x1)
    else:
        b1 = (Delta + Upsilon) / 2.0
        b2 = (Delta - Upsilon) / 2.0
        
        b1 = b1.reshape(-1, nov)
        b2 = b2.reshape(-1, nov)
        
        D_X = hdiag[:nov] - omega
        D_Y = hdiag[:nov] + omega
        
        mo1base = cp.hstack((b1 / D_X, b2 / D_Y))
        
        def krylov_vind(V):
            X = V[:, :nov]
            Y = V[:, nov:]
            v_out = vind(V)
            AX_BY = v_out[:, :nov]
            minus_BX_AY = v_out[:, nov:]
            
            KX = AX_BY - hdiag[:nov] * X
            KY = -minus_BX_AY - hdiag[:nov] * Y
            
            return cp.hstack((-KX / D_X, -KY / D_Y))

        sol = krylov(krylov_vind, mo1base, tol=td_hess.cphf_conv_tol, max_cycle=td_hess.cphf_max_cycle)
        
        x1 = sol[:, :nov].reshape(Delta.shape)
        y1 = sol[:, nov:].reshape(Upsilon.shape)
        
    return x1, y1

def make_intermediates(td_hess, x_y, z1, singlet=True):
    """
    Phase 2: Construct the unperturbed transition and density matrices
    """
    mf = td_hess.base._scf
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    
    nocc = int((mo_occ > 0).sum())
    
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = x + y
    xmy = x - y
    
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    
    # Eq 17
    # P_I
    P_I = 0.5 * orbv @ xpy.T @ xpy @ orbv.T + 0.5 * orbv @ xmy.T @ xmy @ orbv.T \
          - 0.5 * orbo @ xpy @ xpy.T @ orbo.T - 0.5 * orbo @ xmy @ xmy.T @ orbo.T
          
    R_I = orbv @ xpy.T @ orbo.T
    T_I = orbv @ xmy.T @ orbo.T
    
    # P_I'
    P_I_prime = P_I + 0.5 * orbv @ z1 @ orbo.T + 0.5 * orbo @ z1.T @ orbv.T
    
    # Lambda_I_prime
    F_AO = cp.asarray(mf.get_fock())
    P = orbo @ orbo.T * 2
    
    vj_PI, vk_PI = mf.get_jk(mol, P_I_prime + P_I_prime.T)
    G_PI = vj_PI - 0.5 * vk_PI
    
    vj_RI, vk_RI = mf.get_jk(mol, cp.stack((R_I.T + R_I, R_I + R_I.T)))
    Gp_RI_T = vj_RI[0] - 0.5 * vk_RI[0]
    Gp_RI = vj_RI[1] - 0.5 * vk_RI[1]
    
    _, vk_TI = mf.get_jk(mol, cp.stack((T_I.T + T_I, T_I + T_I.T)))
    Gm_TI_T = -0.5 * vk_TI[0]
    Gm_TI = -0.5 * vk_TI[1]
    
    Lambda_I_prime = P_I_prime @ F_AO + P @ G_PI + \
                     0.5 * R_I @ Gp_RI_T + 0.5 * R_I.T @ Gp_RI + \
                     0.5 * T_I @ Gm_TI_T + 0.5 * T_I.T @ Gm_TI
                     
    W_I = -0.5 * Lambda_I_prime - 0.5 * Lambda_I_prime.T
    
    return {'P_I': P_I, 'R_I': R_I, 'T_I': T_I, 'P_I_prime': P_I_prime, 
            'Lambda_I_prime': Lambda_I_prime, 'W_I': W_I, 'P': P, 'F_AO': F_AO}

def make_perturbed_intermediates(td_hess, intermediates, x_y, x1, y1, Ux, z1, singlet=True):
    """
    Phase 2: Construct the geometric derivatives of the intermediates
    Returns P_I_prime_y, P_y, Gamma_I_prime_y, W_I_y, L_I_prime_y
    """
    mf = td_hess.base._scf
    mol = mf.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    
    nocc = int((mo_occ > 0).sum())
    natm = mol.natm
    nao = mol.nao
    
    x, y = x_y
    x = cp.asarray(x)
    y = cp.asarray(y)
    xpy = x + y
    xmy = x - y
    
    x1py1 = x1 + y1
    x1my1 = x1 - y1
    
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    
    P_y = cp.zeros((natm, 3, nao, nao))
    R_I_y = cp.zeros((natm, 3, nao, nao))
    T_I_y = cp.zeros((natm, 3, nao, nao))
    P_I_prime_y = cp.zeros((natm, 3, nao, nao))
    L_I_prime_y = cp.zeros((natm, 3, nao, nao))
    W_I_y = cp.zeros((natm, 3, nao, nao))
    
    for i in range(natm):
        for j in range(3):
            C_y = mo_coeff @ Ux[i, j]
            Co_y = C_y[:, :nocc]
            Cv_y = C_y[:, nocc:]
            
            P_y[i, j] = 2 * (Co_y @ orbo.T + orbo @ Co_y.T)
            
            R_I_y[i, j] = Cv_y @ xpy.T @ orbo.T + orbv @ x1py1[i, j].T @ orbo.T + orbv @ xpy.T @ Co_y.T
            T_I_y[i, j] = Cv_y @ xmy.T @ orbo.T + orbv @ x1my1[i, j].T @ orbo.T + orbv @ xmy.T @ Co_y.T
            
            # P_I_y
            t1 = Cv_y @ xpy.T @ xpy @ orbv.T + orbv @ x1py1[i, j].T @ xpy @ orbv.T + orbv @ xpy.T @ x1py1[i, j] @ orbv.T + orbv @ xpy.T @ xpy @ Cv_y.T
            t2 = Cv_y @ xmy.T @ xmy @ orbv.T + orbv @ x1my1[i, j].T @ xmy @ orbv.T + orbv @ xmy.T @ x1my1[i, j] @ orbv.T + orbv @ xmy.T @ xmy @ Cv_y.T
            t3 = Co_y @ xpy @ xpy.T @ orbo.T + orbo @ x1py1[i, j] @ xpy.T @ orbo.T + orbo @ xpy @ x1py1[i, j].T @ orbo.T + orbo @ xpy @ xpy.T @ Co_y.T
            t4 = Co_y @ xmy @ xmy.T @ orbo.T + orbo @ x1my1[i, j] @ xmy.T @ orbo.T + orbo @ xmy @ x1my1[i, j].T @ orbo.T + orbo @ xmy @ xmy.T @ Co_y.T
            P_I_y = 0.5 * t1 + 0.5 * t2 - 0.5 * t3 - 0.5 * t4
            
            P_I_prime_y[i, j] = P_I_y + 0.5 * (Cv_y @ z1 @ orbo.T + orbv @ z1 @ Co_y.T + Co_y @ z1.T @ orbv.T + orbo @ z1.T @ Cv_y.T)
            
            # Form L_I^{\prime[\tilde{y}]} (Eq 19)
            F_AO = intermediates['F_AO']
            C_F_C_y = C_y.T @ F_AO @ mo_coeff + mo_coeff.T @ F_AO @ C_y 
            Cv_F_Cv_y = C_F_C_y[nocc:, nocc:]
            Co_F_Co_y = C_F_C_y[:nocc, :nocc]
            L_MO_y = 0.5 * Cv_F_Cv_y @ z1 - 0.5 * z1 @ Co_F_Co_y
            
            vj_PIy, vk_PIy = mf.get_jk(mol, P_I_prime_y[i, j] + P_I_prime_y[i, j].T)
            G_PIy = vj_PIy - 0.5 * vk_PIy
            
            # Add missing G_y[P_I] 
            vj_PI, vk_PI = mf.get_jk(mol, intermediates['P_I'] + intermediates['P_I'].T)
            G_PI = vj_PI - 0.5 * vk_PI
            
            L_MO_y += orbv.T @ G_PIy @ orbo
            
            # C_v^{\dagger y} G[P_I] C_o + C_v^\dagger G[P_I] C_o^y
            L_MO_y += Cv_y.T @ G_PI @ orbo + orbv.T @ G_PI @ Co_y
            
            # 1/2 { C_v^\dagger G_p[R_I] C_v X }^y
            # Since R_I = C_v X C_o^T, G_p[R_I] = Gp_RI
            vj_RI, vk_RI = mf.get_jk(mol, intermediates['R_I'] + intermediates['R_I'].T)
            Gp_RI = vj_RI - 0.5 * vk_RI
            
            vj_RIy, vk_RIy = mf.get_jk(mol, R_I_y[i, j] + R_I_y[i, j].T)
            Gp_RIy = vj_RIy - 0.5 * vk_RIy
            
            # term: 1/2 C_v^{\dagger y} G_p[R_I] C_v X
            term_R1 = 0.5 * Cv_y.T @ Gp_RI @ orbv @ xpy.T
            # term: 1/2 C_v^\dagger G_p[R_I^y] C_v X
            term_R2 = 0.5 * orbv.T @ Gp_RIy @ orbv @ xpy.T
            # term: 1/2 C_v^\dagger G_p[R_I] C_v^y X
            term_R3 = 0.5 * orbv.T @ Gp_RI @ Cv_y @ xpy.T
            
            L_MO_y += term_R1 + term_R2 + term_R3
            
            # - 1/2 { X C_o^\dagger G_p[R_I] C_o }^y
            term_R4 = -0.5 * xpy.T @ Co_y.T @ Gp_RI @ orbo
            term_R5 = -0.5 * xpy.T @ orbo.T @ Gp_RIy @ orbo
            term_R6 = -0.5 * xpy.T @ orbo.T @ Gp_RI @ Co_y
            
            L_MO_y += term_R4 + term_R5 + term_R6
            
            # Same for T_I
            _, vk_TI = mf.get_jk(mol, intermediates['T_I'] - intermediates['T_I'].T)
            Gm_TI = -0.5 * vk_TI
            
            _, vk_TIy = mf.get_jk(mol, T_I_y[i, j] - T_I_y[i, j].T)
            Gm_TIy = -0.5 * vk_TIy
            
            term_T1 = 0.5 * Cv_y.T @ Gm_TI @ orbv @ xmy.T
            term_T2 = 0.5 * orbv.T @ Gm_TIy @ orbv @ xmy.T
            term_T3 = 0.5 * orbv.T @ Gm_TI @ Cv_y @ xmy.T
            
            L_MO_y += term_T1 + term_T2 + term_T3
            
            term_T4 = -0.5 * xmy.T @ Co_y.T @ Gm_TI @ orbo
            term_T5 = -0.5 * xmy.T @ orbo.T @ Gm_TIy @ orbo
            term_T6 = -0.5 * xmy.T @ orbo.T @ Gm_TI @ Co_y
            
            L_MO_y += term_T4 + term_T5 + term_T6
            
            L_I_prime_y[i, j] = orbv @ L_MO_y @ orbo.T
            
            # Form W_I^{[\tilde{y}]} (Eq 19)
            Lambda_y = P_I_prime_y[i, j] @ F_AO + P_y[i, j] @ G_PIy # simplified
            W_I_y[i, j] = -0.5 * Lambda_y - 0.5 * Lambda_y.T
            
    return {'P_y': P_y, 'R_I_y': R_I_y, 'T_I_y': T_I_y, 'P_I_prime_y': P_I_prime_y, 
            'L_I_prime_y': L_I_prime_y, 'W_I_y': W_I_y}

class Hessian(rhf_hess_gpu.HessianBase):
    cphf_max_cycle = 50
    cphf_conv_tol = 1e-8
    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device
    _keys = {'cphf_max_cycle', 'cphf_conv_tol', 'mol', 'base', 'state', 'atmlst', 'de', 'method'}
    
    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.max_memory = self.mol.max_memory
        self.state = 1
        self.atmlst = None
        self.de = np.zeros((0, 0, 3, 3))
        self.method = 'semi-analytical' # Options: 'analytical', 'semi-analytical'

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********', self.__class__, self.base.__class__)
        log.info('cphf_conv_tol  = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('State          = %d', self.state)
        log.info('Method         = %s', self.method)
        return self
        
    def omega_grad(self, state=None, atmlst=None, with_solvent=False, singlet=True):
        if state is None: state = self.state - 1
        return omega_grad(self.base, state, atmlst=atmlst, with_solvent=with_solvent, singlet=singlet)
        
    def analytical_omega_hessian(self, state, singlet=True):
        """
        Full analytical Hessian of the excitation energy.
        """
        log = logger.new_logger(self)
        time0 = log.init_timer()
        
        mf = self.base._scf
        mol = mf.mol
        # SCALE AMPLITUDES TO MATCH LIU & LIANG CONVENTION (X^2 = 1)
        x_y_orig = self.base.xy[state]
        x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
        omega = self.base.e[state]
        
        from gpu4pyscf.grad import tdrhf as tdrhf_grad
        td_grad_obj = tdrhf_grad.Gradients(self.base)
        
        # 1. Ground state MO responses (U^x)
        mo_coeff = cp.asarray(mf.mo_coeff)
        mo_occ = cp.asarray(mf.mo_occ)
        mo_energy = cp.asarray(mf.mo_energy)
        from gpu4pyscf.hessian import rhf as rhf_hess_gpu
        mf_hess = rhf_hess_gpu.Hessian(mf)
        h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
        fx = mf_hess.gen_vind(mo_coeff, mo_occ)
        mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
        mo1 = cp.asarray(mo1)
        log.timer('Ground-state MO responses U^x', *time0)
        
        # Build full Ux
        from gpu4pyscf.grad import rhf as grad_rhf
        mf_grad = grad_rhf.Gradients(mf)
        _, _, s1a_basis = rhf_hess_gpu.get_ovlp(mol)
        s1a_basis = cp.asarray(s1a_basis)
        
        natm = mol.natm
        nao = mol.nao
        nocc = int((mo_occ > 0).sum())
        
        s1ao = cp.zeros((natm, 3, nao, nao))
        aoslices = mol.aoslice_by_atom()
        for atm_id in range(natm):
            p0, p1 = aoslices[atm_id][2:]
            s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
            s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)
            
        s1mo = cp.zeros((natm, 3, nao, nao))
        for i in range(natm):
            for j in range(3):
                s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff
                
        Ux = cp.zeros((natm, 3, nao, nao))
        Ux[:,:,:,:nocc] = mo1
        Ux[:,:,:nocc,nocc:] = -s1mo[:,:,:nocc,nocc:] - mo1[:,:,nocc:,:].transpose(0,1,3,2)
        Ux[:,:,nocc:,nocc:] = -0.5 * s1mo[:,:,nocc:,nocc:]
        
        # 2. Z-vector
        z1 = solve_z_vector(td_grad_obj, x_y, singlet=singlet)
        log.timer('Z-vector', *time0)
        
        # 3. CP-TDDFT equations
        x1, y1 = solve_cptddft(self, x_y, omega, mo1, mo_e1, singlet=singlet)
        log.timer('CP-TDDFT responses', *time0)
        
        # 4. Density Matrices and Intermediates
        intermediates = make_intermediates(self, x_y, z1, singlet=singlet)
        perturbed_intermediates = make_perturbed_intermediates(self, intermediates, x_y, x1, y1, Ux, z1, singlet=singlet)
        log.timer('Intermediates construction', *time0)
        
        # 5. Exact Integral Derivatives
        from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
        from gpu4pyscf.lib.cupy_helper import contract
        
        # H^xy * P_I'
        de_hcore = _e_hcore_generator(self, intermediates['P_I_prime'])
        e1_hcore = cp.zeros((natm, natm, 3, 3))
        for i0 in range(natm):
            for j0 in range(i0+1):
                e1_hcore[i0, j0] += de_hcore(i0, j0)
                e1_hcore[j0, i0] = e1_hcore[i0, j0].T
                
        # Gamma_I' * Pi^xy
        vhfopt = mf._opt_gpu.get(mol.omega)
        P_I_prime = intermediates['P_I_prime']
        P = intermediates['P']
        R_I = intermediates['R_I']
        T_I = intermediates['T_I']
        
        ejk_PI = _partial_ejk_ip2(mol, P_I_prime + P, vhfopt)
        ejk_PI -= _partial_ejk_ip2(mol, P_I_prime, vhfopt)
        ejk_PI -= _partial_ejk_ip2(mol, P, vhfopt)
        
        ejk_RI = _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt)
        ejk_PI += 0.5 * ejk_RI
        
        ejk_TI = _partial_ejk_ip2(mol, T_I - T_I.T, vhfopt, j_factor=0.0)
        ejk_PI -= 0.5 * ejk_TI
        
        # W_I * S^xy
        s1aa, s1ab, s1a_ovlp = get_ovlp(mol)
        s1aa = cp.asarray(s1aa)
        s1ab = cp.asarray(s1ab)
        e1_ovlp = cp.zeros((natm, natm, 3, 3))
        W_I = intermediates['W_I']
        
        for i0 in range(natm):
            p0, p1 = aoslices[i0][2:]
            e1_ovlp[i0, i0] += contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1]) * 2
            for j0 in range(i0+1):
                q0, q1 = aoslices[j0][2:]
                e1_ovlp[i0, j0] += contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1]) * 2
                e1_ovlp[j0, i0] = e1_ovlp[i0, j0].T
                
        # First order perturbed components
        from gpu4pyscf.hessian.rhf import _get_jk_ip1
        from gpu4pyscf.df import int3c2e
        
        h1ao_x_eval = cp.zeros((natm, 3, nao, nao))
        h1_eval = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
        for atm_id in range(natm):
            p0, p1 = aoslices[atm_id][2:]
            h1ao_x_eval[atm_id, :, p0:p1] += h1_eval[:, p0:p1]
            h1ao_x_eval[atm_id, :, :, p0:p1] += h1_eval[:, p0:p1].transpose(0, 2, 1)
        coords = mol.atom_coords()
        charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
        fakemol = gto.fakemol_for_charges(coords)
        intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=True, aosym=False, group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)
        eye_ao = cp.eye(nao)
        eye_ao_sorted = intopt.sort_orbitals(eye_ao, axis=[0])
        dh1e_ao = cp.zeros([natm, 3, nao, nao])
        for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
            dh1e_ao[k0:k1, :, j0:j1, :] += contract('xkji,io->kxjo', int3c_blk, eye_ao_sorted[i0:i1])
            dh1e_ao[k0:k1, :, i0:i1, :] += contract('xkji,jo->kxio', int3c_blk, eye_ao_sorted[j0:j1])
        dh1e_ao = contract('kxjo,k->kxjo', dh1e_ao, -charges)
        P_sort = cp.asarray(intopt.sort_orbitals(np.eye(nao), axis=[0]))
        h1ao_x_eval += contract('pj,kxjo->kxpo', P_sort.T, dh1e_ao)
        
        nao = mol.nao
        nocc = int((mo_occ > 0).sum())
        orbo = mo_coeff[:, :nocc]
        dm0_eval = orbo @ orbo.T * 2
        vj_x_eval, vk_x_eval = _get_jk_ip1(mol, dm0_eval)
        vj_x_eval = vj_x_eval.reshape(natm, 3, nao, nao)
        vk_x_eval = vk_x_eval.reshape(natm, 3, nao, nao)
        G_px_D0 = vj_x_eval + vj_x_eval.transpose(0,1,3,2) - 0.5 * (vk_x_eval + vk_x_eval.transpose(0,1,3,2))
        
        Co_x_eval = cp.zeros((natm, 3, nao, nocc))
        for i in range(natm):
            for j in range(3):
                Co_x_eval[i, j] = mo_coeff @ Ux[i, j, :, :nocc]
        Dx_eval = cp.zeros((natm, 3, nao, nao))
        for i in range(natm):
            for j in range(3):
                tmp_e = Co_x_eval[i, j] @ orbo.T
                Dx_eval[i, j] = 2 * (tmp_e + tmp_e.T)
        vj_Dx, vk_Dx = mf.get_jk(mol, Dx_eval.reshape(-1, nao, nao))
        vj_Dx = vj_Dx.reshape(natm, 3, nao, nao)
        vk_Dx = vk_Dx.reshape(natm, 3, nao, nao)
        G_p_Dx = vj_Dx - 0.5 * vk_Dx
        F_x_AO = h1ao_x_eval + G_px_D0 + G_p_Dx
        F_x_MO = cp.zeros((natm, 3, nao, nao))
        for i in range(natm):
            for j in range(3):
                F_x_MO[i, j] = mo_coeff.T @ F_x_AO[i, j] @ mo_coeff
                
        vj_PI_x, vk_PI_x = _get_jk_ip1(mol, P_I_prime + P_I_prime.T)
        vj_PI_x = vj_PI_x.reshape(natm, 3, nao, nao)
        vk_PI_x = vk_PI_x.reshape(natm, 3, nao, nao)
        G_x_PI = vj_PI_x + vj_PI_x.transpose(0,1,3,2) - 0.5 * (vk_PI_x + vk_PI_x.transpose(0,1,3,2))
        vj_RI_x, vk_RI_x = _get_jk_ip1(mol, R_I + R_I.T)
        vj_RI_x = vj_RI_x.reshape(natm, 3, nao, nao)
        vk_RI_x = vk_RI_x.reshape(natm, 3, nao, nao)
        G_px_RI = vj_RI_x + vj_RI_x.transpose(0,1,3,2) - 0.5 * (vk_RI_x + vk_RI_x.transpose(0,1,3,2))
        _, vk_TI_x = _get_jk_ip1(mol, T_I - T_I.T)
        vk_TI_x = vk_TI_x.reshape(natm, 3, nao, nao)
        G_mx_TI = -0.5 * (vk_TI_x + vk_TI_x.transpose(0,1,3,2))

        e1_perturbed = cp.zeros((natm, natm, 3, 3))
        L_I_prime_y = perturbed_intermediates['L_I_prime_y']
        W_I_y = perturbed_intermediates['W_I_y']
        P_I_prime_y = perturbed_intermediates['P_I_prime_y']
        P_y = perturbed_intermediates['P_y']
        R_I_y = perturbed_intermediates['R_I_y']
        T_I_y = perturbed_intermediates['T_I_y']
        
        for i0 in range(natm):
            for j0 in range(natm):
                for x in range(3):
                    for y in range(3):
                        # SWAPPED INDICES TO MATCH Eq 19
                        tmp_U_S = 2 * Ux[i0, x] + s1mo[i0, x]
                        e1_perturbed[i0, j0, x, y] += cp.trace(L_I_prime_y[j0, y] @ tmp_U_S)
                        e1_perturbed[i0, j0, x, y] += cp.trace(W_I_y[j0, y] @ s1mo[i0, x])
                        e1_perturbed[i0, j0, x, y] += cp.trace(P_I_prime_y[j0, y] @ F_x_MO[i0, x])
                        e1_perturbed[i0, j0, x, y] += cp.trace(P_I_prime_y[j0, y] @ G_px_D0[i0, x])
                        e1_perturbed[i0, j0, x, y] += cp.trace(P_y[j0, y] @ G_x_PI[i0, x])
                        e1_perturbed[i0, j0, x, y] += 0.5 * cp.trace((R_I_y[j0, y] + R_I_y[j0, y].T) @ G_px_RI[i0, x])
                        e1_perturbed[i0, j0, x, y] += 0.5 * cp.trace((T_I_y[j0, y] - T_I_y[j0, y].T) @ G_mx_TI[i0, x])
                        
        # Assembly
        omega_xy = e1_hcore + ejk_PI + e1_ovlp + e1_perturbed
        return omega_xy / 2.0

    def kernel(self, *args, fd_delta=1.0e-3, include_relaxation=True, **kwargs):
        state = self.state - 1
        if self.method == 'analytical':
            return self.analytical_omega_hessian(state)
        else:
            return omega_hessian(self.base, state, fd_delta=fd_delta, include_relaxation=include_relaxation)
            
    hess = kernel
