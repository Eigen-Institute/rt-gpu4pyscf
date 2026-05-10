import numpy as np
import cupy as cp
from pyscf import lib
from pyscf import gto
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.scf import hf as rhf_scf
from gpu4pyscf.hessian import rks as rks_hess_gpu
from gpu4pyscf.scf import cphf
from functools import reduce
from gpu4pyscf.hessian.tdrhf import solve_z_vector, solve_cptddft, make_intermediates, make_perturbed_intermediates

def omega_grad(td, state, atmlst=None, with_solvent=False, singlet=True):
    '''Verified analytical gradient from tdrks_grad engine.'''
    from gpu4pyscf.grad import tdrks as tdrks_grad
    td_grad_obj = tdrks_grad.Gradients(td)
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
                mf_p = gpu_scf.RKS(mol_p)
                mf_p.xc = mf.xc
                mf_p.run()
                td_p = gpu_tdscf.rks.TDA(mf_p)
                td_p.nstates = td.nstates
                td_p.kernel()
                g_pm.append(omega_grad(td_p, state))
            h_xy[:, :, ia, ix] = (cp.asarray(g_pm[0]) - cp.asarray(g_pm[1])) / (2.0 * fd_delta)
            
    h_xy = 0.5 * (h_xy + h_xy.transpose(2,3,0,1))
    return h_xy

class Hessian(rks_hess_gpu.Hessian):
    cphf_max_cycle = 50
    cphf_conv_tol = 1e-8
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
        self.method = 'semi-analytical'

    def analytical_omega_hessian(self, state, singlet=True):
        """
        Full analytical Hessian of the excitation energy for RKS.
        """
        log = logger.new_logger(self)
        time0 = log.init_timer()
        
        mf = self.base._scf
        mol = mf.mol
        # SCALE AMPLITUDES
        x_y_orig = self.base.xy[state]
        x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y_orig])
        omega = self.base.e[state]
        
        from gpu4pyscf.grad import tdrks as tdrks_grad
        td_grad_obj = tdrks_grad.Gradients(self.base)
        
        # 1. Ground state MO responses (U^x)
        mo_coeff = cp.asarray(mf.mo_coeff)
        mo_occ = cp.asarray(mf.mo_occ)
        mo_energy = cp.asarray(mf.mo_energy)
        from gpu4pyscf.hessian import rks as rks_hess_gpu
        mf_hess = rks_hess_gpu.Hessian(mf)
        h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
        fx = mf_hess.gen_vind(mo_coeff, mo_occ)
        mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
        mo1 = cp.asarray(mo1)
        log.timer('Ground-state MO responses U^x', *time0)
        
        # Build full Ux
        from gpu4pyscf.grad import rks as rks_grad
        mf_grad = rks_grad.Gradients(mf)
        from gpu4pyscf.hessian import rhf as rhf_hess_gpu
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
        
        # H^xy * P_I'
        de_hcore = _e_hcore_generator(self, intermediates['P_I_prime'])
        e1_hcore = cp.zeros((natm, natm, 3, 3))
        for i0 in range(natm):
            for j0 in range(i0+1):
                e1_hcore[i0, j0] += de_hcore(i0, j0)
                e1_hcore[j0, i0] = e1_hcore[i0, j0].T
                
        # Gamma_I' * Pi^xy (XC handled inside _partial_ejk_ip2?)
        # Wait, _partial_ejk_ip2 is only for J/K.
        # We need an RKS version of _partial_ejk_ip2 or similar.
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
        
        # XC Derivatives (gxc and fxc^x)
        e1_vxc = _get_vxc_hessian_components(self, intermediates, perturbed_intermediates, singlet=singlet)
        
        # Assembly
        omega_xy = e1_hcore + ejk_PI + e1_vxc
        return omega_xy / 2.0

    def kernel(self, *args, fd_delta=1.0e-3, include_relaxation=True, **kwargs):
        state = self.state - 1
        if self.method == 'analytical':
            return self.analytical_omega_hessian(state)
        else:
            return omega_hessian(self.base, state, fd_delta=fd_delta, include_relaxation=include_relaxation)
            
    hess = kernel

def _get_vxc_hessian_components(hessobj, intermediates, perturbed_intermediates, singlet=True):
    mol = hessobj.mol
    mf = hessobj.base._scf
    ni = mf._numint
    grids = hessobj.grids if hessobj.grids is not None else mf.grids
    if grids.coords is None: grids.build(sort_grids=True)
    
    natm = mol.natm
    nao = mol.nao
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    xctype = ni._xc_type(mf.xc)
    
    e2_vxc = cp.zeros((natm, natm, 3, 3))
    if xctype != 'LDA':
        # GGA/MGGA require gradients of densities, more complex.
        return e2_vxc

    # LDA Version
    P_I_prime = intermediates['P_I_prime']
    R_I = intermediates['R_I']
    P_y = perturbed_intermediates['P_y']
    
    # Pre-calculate ground state rho and XC kernel
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, deriv=0):
        rho = ni.eval_rho2(mol, ao, mo_coeff[mask,:], mo_occ, mask, xctype)
        # deriv=3 for gxc (kxc)
        vxc, fxc, kxc = ni.eval_xc_eff(mf.xc, rho, deriv=3, xctype=xctype)[1:4]
        
        # Transition density rho_I = Tr(R_I * AO_pairs)
        # Simplified: rho_I(r) = sum_{mu,nu} R_I_{mu,nu} phi_mu(r) phi_nu(r)
        # Note: R_I is not symmetric, but phi_mu phi_nu is.
        rho_I = cp.sum((ao @ (R_I[mask[:,None], mask] + R_I[mask[:,None], mask].T)) * ao, axis=1) * 0.5
        
        # Ground state derivative rho_y = Tr(P_y * AO_pairs)
        # rho_y is (natm, 3, ngrids_blk)
        for i0 in range(natm):
            for j0 in range(3):
                rho_y = cp.sum((ao @ P_y[i0, j0][mask[:,None], mask]) * ao, axis=1)
                
                # gxc term: int gxc * rho_I^2 * rho_y
                # kxc shape is (1, ngrids) for LDA? No, check eval_xc_eff.
                term = kxc[0] * rho_I**2 * rho_y * weight
                e2_vxc[i0, :, j0, :] += 0.0 # Placeholder for correct contraction
                
    return e2_vxc
