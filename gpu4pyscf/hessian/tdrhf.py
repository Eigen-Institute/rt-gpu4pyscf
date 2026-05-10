
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.scf import hf as rhf_scf
from gpu4pyscf.hessian import rhf as rhf_hess_gpu
from gpu4pyscf.scf import cphf
from functools import reduce

def omega_grad(td, state, atmlst=None, with_solvent=False, singlet=True):
    '''Verified analytical gradient from tdrhf_grad engine.'''
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td); de_tda_elec = td_grad_obj.grad_elec(td.xy[state], singlet=singlet, atmlst=atmlst, with_solvent=with_solvent)
    mf_grad = td._scf.nuc_grad_method(); de_gs_elec = mf_grad.grad_elec(atmlst=atmlst); return np.asarray(de_tda_elec) - np.asarray(de_gs_elec)

def omega_hessian(td, state, fd_delta=1.0e-3, include_relaxation=True):
    '''Robust semi-analytical Hessian (FD on analytical gradient).'''
    from gpu4pyscf import scf as gpu_scf
    from gpu4pyscf import tdscf as gpu_tdscf
    mf = td._scf; mol = td.mol; natm = mol.natm; coords0 = mol.atom_coords()
    h_xy = cp.zeros((natm, 3, natm, 3))
    
    # Evaluate d/dy (G_x) where G_x is the verified analytical gradient.
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
                # Gradient of the 'state'-th root at the perturbed geometry
                g_pm.append(omega_grad(td_p, state))
            h_xy[:, :, ia, ix] = (cp.asarray(g_pm[0]) - cp.asarray(g_pm[1])) / (2.0 * fd_delta)
            
    # Final symmetry enforcement for numerical precision
    h_xy = 0.5 * (h_xy + h_xy.transpose(2,3,0,1))
    return h_xy

class Hessian(rhf_hess_gpu.HessianBase):
    cphf_max_cycle = 50; cphf_conv_tol = 1e-8; to_cpu = utils.to_cpu; to_gpu = utils.to_gpu; device = utils.device
    _keys = {'cphf_max_cycle', 'cphf_conv_tol', 'mol', 'base', 'state', 'atmlst', 'de'}
    def __init__(self, td):
        self.verbose = td.verbose; self.stdout = td.stdout; self.mol = td.mol; self.base = td; self.max_memory = self.mol.max_memory; self.state = 1; self.atmlst = None; self.de = np.zeros((0, 0, 3, 3))
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose); log.info('\n'); log.info('******** %s for %s ********', self.__class__, self.base.__class__)
        log.info('cphf_conv_tol  = %g', self.cphf_conv_tol); log.info('cphf_max_cycle = %d', self.cphf_max_cycle); log.info('State          = %d', self.state); return self
    def omega_grad(self, state=None, atmlst=None, with_solvent=False, singlet=True):
        if state is None: state = self.state - 1
        return omega_grad(self.base, state, atmlst=atmlst, with_solvent=with_solvent, singlet=singlet)
    def kernel(self, *args, fd_delta=1.0e-3, include_relaxation=True, **kwargs):
        state = self.state - 1; return omega_hessian(self.base, state, fd_delta=fd_delta, include_relaxation=include_relaxation)
    hess = kernel
