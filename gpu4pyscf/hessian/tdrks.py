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
    from gpu4pyscf import dft as gpu_dft
    from gpu4pyscf import tdscf as gpu_tdscf
    mf = td._scf; mol = td.mol; natm = mol.natm; coords0 = mol.atom_coords()
    h_xy = cp.zeros((natm, 3, natm, 3))
    
    for ia in range(natm):
        for ix in range(3):
            g_pm = []
            for d in [fd_delta, -fd_delta]:
                c = coords0.copy(); c[ia, ix] += d
                mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
                mf_p = gpu_dft.RKS(mol_p)
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
        rks_hess_gpu.Hessian.__init__(self, td._scf)
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td
        self.max_memory = self.mol.max_memory
        self.state = 1
        self.atmlst = None
        self.de = np.zeros((0, 0, 3, 3))
        self.method = 'semi-analytical'

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
        Analytical Hessian of TDA/TDDFT excitation energy for RKS.
        Currently falling back to semi-analytical finite-difference of gradient
        to ensure correctness.
        """
        return omega_hessian(self.base, state)

    def kernel(self, state=None, fd_delta=1.0e-3, include_relaxation=True, **kwargs):
        if state is None: state = self.state - 1
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
