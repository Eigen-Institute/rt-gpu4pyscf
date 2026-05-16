import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def check_grad_components():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    g_obj = tdrhf_grad.Gradients(td)
    g_exc = g_obj.grad_elec(td.xy[0])
    g_gs = mf.nuc_grad_method().grad_elec()
    g_omega = g_exc - g_gs
    
    print(f"Total Omega Grad [0,Z]: {g_omega[0,2]:.8f}")
    
    # Analyze components of g_exc
    # PySCF TDA grad_elec:
    # de = hcore_deriv + eri_deriv + ovlp_deriv
    # It uses a difference density dm_exc = 2 * (X X^T - X^T X) ?
    # Let's check gpu4pyscf/grad/tdrhf.py
    
    nocc = int(mf.mo_occ.sum()//2)
    nvir = mf.mo_energy.size - nocc
    x, y = [cp.asarray(v) for v in td.xy[0]]
    xpy = (x + y).reshape(nocc, nvir)
    xmy = (x - y).reshape(nocc, nvir)
    
    # physical difference density (both spins)
    dvv = (xpy.T @ xpy + xmy.T @ xmy) * 2.0
    doo = (-xpy @ xpy.T - xmy @ xmy.T) * 2.0
    # Wait! PySCF uses factor 2.0 for spin doubling in dm_exc!
    
    dm_exc = mf.mo_coeff[:,:nocc] @ doo @ mf.mo_coeff[:,:nocc].T + \
             mf.mo_coeff[:,nocc:] @ dvv @ mf.mo_coeff[:,nocc:].T
    
    h1 = -mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3)
    g_hcore = np.einsum('xpq,pq->xp', h1, dm_exc.get())
    # ... nuclear part ...
    
    print(f"Manual Hcore Grad [0,Z]: {g_hcore[2,0]:.8f}") # index order (comp, atm)

if __name__ == "__main__":
    check_grad_components()
