import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.lib.cupy_helper import contract

def fix_ti():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    h_obj = tdrhf_hess.Hessian(td)
    h_obj.method = 'analytical'
    h_ana = h_obj.kernel()
    h_obj.method = 'semi-analytical'
    h_semi = h_obj.kernel()
    
    print(f"Original Analytical H[0,Z,0,Z]: {h_ana[0,2,0,2]:.8f}")
    print(f"Semi-Analytical     H[0,Z,0,Z]: {h_semi[0,2,0,2]:.8f}")
    print(f"Diff: {h_ana[0,2,0,2] - h_semi[0,2,0,2]:.8f}")
    
    # Manual HF curvature
    x_y = td.xy[0]
    nocc = int(mf.mo_occ.sum()//2)
    nvir = mf.mo_energy.size - nocc
    x, y = [cp.asarray(v) for v in x_y]
    xpy = (x + y).reshape(nocc, nvir)
    xmy = (x - y).reshape(nocc, nvir)
    dvv = xpy.T @ xpy + xmy.T @ xmy
    doo = -xpy @ xpy.T - xmy @ xmy.T
    P_I = mf.mo_coeff[:,:nocc] @ doo @ mf.mo_coeff[:,:nocc].T + \
          mf.mo_coeff[:,nocc:] @ dvv @ mf.mo_coeff[:,nocc:].T
    
    with mol.with_rinv_at_nucleus(0):
        ipiprinv = cp.asarray(mol.intor('int1e_ipiprinv', comp=9)).reshape(3,3,mol.nao,mol.nao)
        hf_curv = contract('xypq,pq->xy', ipiprinv, P_I) * -mol.atom_charge(0)
    
    print(f"Manual HF Curvature on Atom 0: {hf_curv[2,2]:.8f}")
    
    # Try adding it
    fixed = h_ana[0,2,0,2] + hf_curv[2,2]
    print(f"Fixed Analytical: {fixed:.8f}")
    print(f"New Diff:         {fixed - h_semi[0,2,0,2]:.8f}")

fix_ti()
