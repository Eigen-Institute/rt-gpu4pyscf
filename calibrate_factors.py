import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def get_hcore_grad(mol, dm):
    h1 = -mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)
    # dm is total density (Trace N)
    return np.einsum('xpq,pq->xp', h1, dm)

def get_eri_grad(mol, dm):
    mf = RHF(mol).run()
    vj, vk = mf.get_jk(mol, dm)
    # Singlet gradient uses 2J-K
    veff = vj * 2.0 - vk
    # Wait, PySCF grad uses get_veff which does this.
    # We want the derivative of Tr(P G[P]).
    # For RHF, it's Tr(P (2J-K)).
    # Derivative is 2 * Tr(P G'[P])? No, it's Tr(P G'[P]).
    # Actually, PySCF computes it via int3c2e.get_jk_ip1
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_x, vk_x = _get_jk_ip1(mol, dm)
    # vj_x is (natm, 3, nao, nao)
    return np.einsum('axpq,pq->ax', (vj_x * 2.0 - vk_x).get(), dm.get())

def calibrate():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    x_y = td.xy[0]
    
    # physical difference density P_I (both spins)
    nocc = int(mf.mo_occ.sum()//2)
    nvir = mf.mo_energy.size - nocc
    x, y = [cp.asarray(v) for v in x_y]
    xpy = (x + y).reshape(nocc, nvir)
    xmy = (x - y).reshape(nocc, nvir)
    dvv = (xpy.T @ xpy + xmy.T @ xmy) * 2.0
    doo = (-xpy @ xpy.T - xmy @ xmy.T) * 2.0
    dm_I = mf.mo_coeff[:,:nocc] @ doo @ mf.mo_coeff[:,:nocc].T + \
           mf.mo_coeff[:,nocc:] @ dvv @ mf.mo_coeff[:,nocc:].T
    
    dr = 0.001
    coords = mol.atom_coords()
    
    # 1. Hcore Static Calibration
    def hcore_grad(mol_):
        # Keep dm_I fixed in AO basis? No, better in MO basis.
        # But for Hcore static part, we only need Tr(P_I H'').
        return float(np.einsum('xpq,pq', -mol_.intor('int1e_ipkin', comp=3) - mol_.intor('int1e_ipnuc', comp=3), dm_I.get())[2])

    coords_p = coords.copy(); coords_p[0,2] += dr
    g_p = hcore_grad(mol.copy().set_geom_(coords_p, unit='Bohr'))
    coords_m = coords.copy(); coords_m[0,2] -= dr
    g_m = hcore_grad(mol.copy().set_geom_(coords_m, unit='Bohr'))
    h_hcore_fd = (g_p - g_m) / (2 * dr)
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    class Dummy:
        def __init__(self, mol):
            self.mol = mol
            self.base = RHF(mol)
        def get_hcore(self, mol):
            nao = mol.nao
            h1aa = mol.intor('int1e_ipipkin', comp=9) + mol.intor('int1e_ipipnuc', comp=9)
            h1ab = mol.intor('int1e_ipkinip', comp=9) + mol.intor('int1e_ipnucip', comp=9)
            return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)
    
    # Analytical Hcore Static
    # Note: P_I has Trace 1.0.
    hcore_gen = _e_hcore_generator(Dummy(mol), dm_I)
    h_hcore_ana = hcore_gen(0, 0)[2,2]
    
    print(f"Hcore Static FD: {h_hcore_fd:.8f}")
    print(f"Hcore Static Analytical: {h_hcore_ana:.8f}")
    print(f"Ratio: {h_hcore_ana / h_hcore_fd:.8f}")

calibrate()
