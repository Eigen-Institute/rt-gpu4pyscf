import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def calibrate_all():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = tdrhf_hess.solve_z_vector(td_grad_obj, td.xy[0])
    print(f"z1 max: {cp.abs(z1).max():.8f}")
    
    # physical gradient
    g_obj = tdrhf_hess.omega_grad(td, 0)
    print(f"Total Physical Grad [0,Z]: {g_obj[0,2]:.8f}")
    
    # Semi-analytical Hessian
    h_obj = tdrhf_hess.Hessian(td)
    h_semi = h_obj.kernel()
    print(f"Total Semi-Analytical Hessian [0,Z,0,Z]: {h_semi[0,2,0,2]:.8f}")
    
    # 1. Hcore Static
    nocc = int(mf.mo_occ.sum()//2)
    nvir = mf.mo_energy.size - nocc
    x, y = [cp.asarray(v) for v in td.xy[0]]
    # Physical P_I (Trace 1.0)
    xpy = (x + y).reshape(nocc, nvir).T
    xmy = (x - y).reshape(nocc, nvir).T
    dvv = xpy @ xpy.T + xmy @ xmy.T
    doo = -xpy.T @ xpy - xmy.T @ xmy
    P_I = mf.mo_coeff[:,nocc:] @ dvv @ mf.mo_coeff[:,nocc:].T + \
          mf.mo_coeff[:,:nocc] @ doo @ mf.mo_coeff[:,:nocc:].T
    
    h2 = cp.asarray(mol.intor('int1e_ipipkin', comp=9) + mol.intor('int1e_ipipnuc', comp=9)).reshape(3,3,mol.nao,mol.nao)
    # Manual Tr(P_I H'')
    # Actually, we need the total static Hcore Hessian for atom 0.
    # This involves both Pulay and Hellmann-Feynman parts.
    # PySCF generator handles this.
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    class Dummy:
        def __init__(self, mol): self.mol = mol; self.base = RHF(mol)
        def get_hcore(self, mol):
            nao = mol.nao
            h1aa = mol.intor('int1e_ipipkin', comp=9) + mol.intor('int1e_ipipnuc', comp=9)
            h1ab = mol.intor('int1e_ipkinip', comp=9) + mol.intor('int1e_ipnucip', comp=9)
            return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)
    
    # Note: Generator on Trace 1.0 returns Trace 2.0. We want Trace 1.0.
    hcore_gen = _e_hcore_generator(Dummy(mol), P_I)
    h_hcore_static = float(hcore_gen(0, 0)[2,2]) * 0.5
    print(f"Physical Hcore Static: {h_hcore_static:.8f}")
    
    # 2. Overlap Static: Tr(W S'')
    # W = ground state energy weighted DM + excited state energy weighted DM?
    # Liu-Liang Eq 22: (W_I + W_g) . S^xy
    # W_I = -0.5 (Lambda + Lambda.T)
    # Let's compute physical W_I for H2.
    # For H2, Lambda = P_I F + P G[P_I] + R G[R].
    # F = diag(eps). P_I_oo = -1.0. P_I_vv = 1.0.
    # Lambda_oo = -eps_o. Lambda_vv = eps_v.
    # W_I_oo = eps_o. W_I_vv = -eps_v.
    eps = mf.mo_energy
    w_oo = cp.diag(eps[:nocc])
    w_vv = -cp.diag(eps[nocc:])
    W_I = mf.mo_coeff[:,:nocc] @ w_oo @ mf.mo_coeff[:,:nocc].T + \
          mf.mo_coeff[:,nocc:] @ w_vv @ mf.mo_coeff[:,nocc:].T
    
    s2 = cp.asarray(mol.intor('int1e_ipipovlp', comp=9)).reshape(3,3,mol.nao,mol.nao)
    # Tr(W_I S'')
    h_ovlp_static = -float(cp.trace(W_I @ s2[2,2])) # Pulay only? 
    # Actually, RHF generator has 2.0 * de.
    print(f"Physical Overlap Static: {h_ovlp_static:.8f}")

calibrate_all()
