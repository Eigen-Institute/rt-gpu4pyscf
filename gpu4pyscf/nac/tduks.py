# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import cupy as cp
import numpy as np
from functools import reduce
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import tduks as tduks_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.scf import ucphf
from gpu4pyscf.lib.cupy_helper import contract, tag_array
from gpu4pyscf.df import int3c2e

def get_nacv_ge(td_nac, x_yI, EI, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate UKS non-adiabatic coupling vectors between ground and excited states.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for spin-conserving transitions')
    
    mol = td_nac.mol
    mf = td_nac.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    occidxa = cp.where(mo_occ[0] > 0)[0]
    occidxb = cp.where(mo_occ[1] > 0)[0]
    viridxa = cp.where(mo_occ[0] == 0)[0]
    viridxb = cp.where(mo_occ[1] == 0)[0]
    
    nocca, nvira = len(occidxa), len(viridxa)
    noccb, nvirb = len(occidxb), len(viridxb)
    
    orboa, orbva = mo_coeff[0][:, occidxa], mo_coeff[0][:, viridxa]
    orbob, orbvb = mo_coeff[1][:, occidxb], mo_coeff[1][:, viridxb]

    (xa, xb), (ya, yb) = x_yI
    xa = cp.asarray(xa).reshape(nocca, nvira).T
    xb = cp.asarray(xb).reshape(noccb, nvirb).T
    
    if isinstance(ya, (int, float)) and ya == 0: ya = cp.zeros_like(xa)
    else: ya = cp.asarray(ya).reshape(nocca, nvira).T
        
    if isinstance(yb, (int, float)) and yb == 0: yb = cp.zeros_like(xb)
    else: yb = cp.asarray(yb).reshape(noccb, nvirb).T
    
    # 1RDM components
    xpya, xpyb = xa + ya, xb + yb
    dmxpya = reduce(cp.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(cp.dot, (orbvb, xpyb, orbob.T))
    
    # RHS for CP-UKS
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    f1vo = tduks_grad._contract_xc_kernel(td_nac, mf.xc, cp.stack((dmxpya, dmxpyb)), None, True, False)[0]
    
    vj, vk = mf.get_jk(mol, cp.stack((dmxpya + dmxpya.T, dmxpyb + dmxpyb.T)), hermi=1)
    veff = vj[0] + vj[1] - hyb * vk + f1vo[:,0] * 2.0
    
    wvoa = reduce(cp.dot, (orbva.T, veff[0], orboa))
    wvob = reduce(cp.dot, (orbvb.T, veff[1], orbob))
    
    # Solve CP-UKS
    vresp = td_nac.base.gen_response(hermi=1)
    def fvind(x):
        dm1 = cp.empty((2, mol.nao, mol.nao))
        za = x[0, : nvira * nocca].reshape(nvira, nocca)
        zb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma, dmb = reduce(cp.dot, (orbva, za, orboa.T)), reduce(cp.dot, (orbvb, zb, orbob.T))
        dm1[0], dm1[1] = dma + dma.T, dmb + dmb.T
        v1 = vresp(dm1)
        v1a, v1b = reduce(cp.dot, (orbva.T, v1[0], orboa)), reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=50, tol=1e-8)[0]
    
    # Gradient terms
    z1ao = cp.stack((reduce(cp.dot, (orbva, z1a, orboa.T)), reduce(cp.dot, (orbvb, z1b, orbob.T))))
    dm_eff = (z1ao + z1ao.transpose(0,2,1)) * 0.5 + cp.stack((dmxpya, dmxpyb))
    
    mf_grad = mf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    
    de = rhf_grad.contract_h1e_dm(mol, h1, dm_eff[0] + dm_eff[1], hermi=0)
    # Background contributions for NACV are complex; simplified implementation here.
    return -de / EI

def get_nacv_ee(td_nac, x_yI, x_yJ, EI, EJ, singlet=True, atmlst=None, verbose=logger.INFO):
    """
    Calculate UKS non-adiabatic coupling vectors between excited states.
    """
    if singlet is False:
        raise NotImplementedError('Only supports for spin-conserving transitions')
    
    mol = td_nac.mol
    mf = td_nac.base._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    
    occidxa = cp.where(mo_occ[0] > 0)[0]
    occidxb = cp.where(mo_occ[1] > 0)[0]
    viridxa = cp.where(mo_occ[0] == 0)[0]
    viridxb = cp.where(mo_occ[1] == 0)[0]
    
    nocca, nvira = len(occidxa), len(viridxa)
    noccb, nvirb = len(occidxb), len(viridxb)
    
    orboa, orbva = mo_coeff[0][:, occidxa], mo_coeff[0][:, viridxa]
    orbob, orbvb = mo_coeff[1][:, occidxb], mo_coeff[1][:, viridxb]

    # Amplitudes for I and J
    (xaI, xbI), (yaI, ybI) = x_yI
    xaI = cp.asarray(xaI).reshape(nocca, nvira).T
    xbI = cp.asarray(xbI).reshape(noccb, nvirb).T
    if isinstance(yaI, (int, float)) and yaI == 0: yaI = cp.zeros_like(xaI)
    else: yaI = cp.asarray(yaI).reshape(nocca, nvira).T
    if isinstance(ybI, (int, float)) and ybI == 0: ybI = cp.zeros_like(xbI)
    else: ybI = cp.asarray(ybI).reshape(noccb, nvirb).T

    (xaJ, xbJ), (yaJ, ybJ) = x_yJ
    xaJ = cp.asarray(xaJ).reshape(nocca, nvira).T
    xbJ = cp.asarray(xbJ).reshape(noccb, nvirb).T
    if isinstance(yaJ, (int, float)) and yaJ == 0: yaJ = cp.zeros_like(xaJ)
    else: yaJ = cp.asarray(yaJ).reshape(nocca, nvira).T
    if isinstance(ybJ, (int, float)) and ybJ == 0: ybJ = cp.zeros_like(xbJ)
    else: ybJ = cp.asarray(ybJ).reshape(noccb, nvirb).T

    # 1RDM products for excited-excited coupling
    rIJooA = -contract('ai,aj->ij', xaJ, xaI) - contract('ai,aj->ij', yaI, yaJ)
    rIJvvA = contract('ai,bi->ab', xaI, xaJ) + contract('ai,bi->ab', yaJ, yaI)
    rIJooB = -contract('ai,aj->ij', xbJ, xbI) - contract('ai,aj->ij', ybI, ybJ)
    rIJvvB = contract('ai,bi->ab', xbI, xbJ) + contract('ai,bi->ab', ybJ, ybI)

    dmzooIJa = reduce(cp.dot, (orboa, (rIJooA + rIJooA.T)*0.5, orboa.T)) + reduce(cp.dot, (orbva, (rIJvvA + rIJvvA.T)*0.5, orbva.T))
    dmzooIJb = reduce(cp.dot, (orbob, (rIJooB + rIJooB.T)*0.5, orbob.T)) + reduce(cp.dot, (orbvb, (rIJvvB + rIJvvB.T)*0.5, orbvb.T))
    
    # Transition components (simplified)
    xpyaI, xpybI = xaI + yaI, xbI + ybI
    xpyaJ, xpybJ = xaJ + yaJ, xbJ + ybJ
    dmxpyaI = reduce(cp.dot, (orbva, xpyaI, orboa.T))
    dmxpybI = reduce(cp.dot, (orbvb, xpybI, orbob.T))
    dmxpyaJ = reduce(cp.dot, (orbva, xpyaJ, orboa.T))
    dmxpybJ = reduce(cp.dot, (orbvb, xpybJ, orbob.T))

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    
    # RHS for CP-UKS (relaxed coupling)
    f1voI = tduks_grad._contract_xc_kernel(td_nac, mf.xc, cp.stack((dmxpyaI, dmxpybI)), cp.stack((dmzooIJa, dmzooIJb)), True, False)[0]
    f1voJ = tduks_grad._contract_xc_kernel(td_nac, mf.xc, cp.stack((dmxpyaJ, dmxpybJ)), None, True, False)[0]
    
    vj_oo, vk_oo = mf.get_jk(mol, cp.stack((dmzooIJa, dmzooIJb)), hermi=1)
    vj_I, vk_I = mf.get_jk(mol, cp.stack((dmxpyaI + dmxpyaI.T, dmxpybI + dmxpybI.T)), hermi=1)
    vj_J, vk_J = mf.get_jk(mol, cp.stack((dmxpyaJ + dmxpyaJ.T, dmxpybJ + dmxpybJ.T)), hermi=1)
    
    veff0 = vj_oo[0] + vj_oo[1] - hyb * vk_oo + f1voI[:,0] * 2.0 # simplified
    wvoa = reduce(cp.dot, (orbva.T, veff0[0], orboa))
    wvob = reduce(cp.dot, (orbvb.T, veff0[1], orbob))
    
    # Solve Z-vector (relaxed part)
    vresp = td_nac.base.gen_response(hermi=1)
    def fvind(x):
        dm1 = cp.empty((2, mol.nao, mol.nao))
        za = x[0, : nvira * nocca].reshape(nvira, nocca)
        zb = x[0, nvira * nocca :].reshape(nvirb, noccb)
        dma, dmb = reduce(cp.dot, (orbva, za, orboa.T)), reduce(cp.dot, (orbvb, zb, orbob.T))
        dm1[0], dm1[1] = dma + dma.T, dmb + dmb.T
        v1 = vresp(dm1)
        v1a, v1b = reduce(cp.dot, (orbva.T, v1[0], orboa)), reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa, wvob), max_cycle=50, tol=1e-8)[0]
    z1ao = cp.stack((reduce(cp.dot, (orbva, z1a, orboa.T)), reduce(cp.dot, (orbvb, z1b, orbob.T))))
    
    # Total effective derivative transition DM
    dm_eff = (z1ao + z1ao.transpose(0,2,1)) * 0.5 * (EJ - EI) + cp.stack((dmzooIJa, dmzooIJb))
    
    mf_grad = mf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    de = rhf_grad.contract_h1e_dm(mol, h1, dm_eff[0] + dm_eff[1], hermi=0)
    
    return -de / (EJ - EI)

class NAC(lib.StreamObject):
    def __init__(self, td):
        self.base = td
        self.mol = td.mol
    
    def get_nacv_ge(self, x_yI, EI, singlet=True):
        return get_nacv_ge(self, x_yI, EI, singlet=singlet)

    def get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet=True):
        return get_nacv_ee(self, x_yI, x_yJ, EI, EJ, singlet=singlet)
