import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian import tdrhf as tdrhf_hess

def check_parity():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    print(f"Excitation Energy: {td.e[0]:.8f}")
    
    h_obj = tdrhf_hess.Hessian(td)
    
    h_obj.method = 'semi-analytical'
    h_semi = h_obj.kernel()
    
    h_obj.method = 'analytical'
    h_ana = h_obj.kernel()
    
    print(f"Semi-Analytical H[0,Z,0,Z]: {h_semi[0,2,0,2]:.8f}")
    print(f"Analytical      H[0,Z,0,Z]: {h_ana[0,2,0,2]:.8f}")
    print(f"Difference:                 {h_ana[0,2,0,2] - h_semi[0,2,0,2]:.8f}")
    
    print(f"Semi-Analytical H[0,Z,1,Z]: {h_semi[0,2,1,2]:.8f}")
    print(f"Analytical      H[0,Z,1,Z]: {h_ana[0,2,1,2]:.8f}")
    print(f"Difference:                 {h_ana[0,2,1,2] - h_semi[0,2,1,2]:.8f}")
    
    ti_semi = h_semi[0,2,0,2] + h_semi[0,2,1,2]
    ti_ana = h_ana[0,2,0,2] + h_ana[0,2,1,2]
    print(f"TI Semi: {ti_semi:.8f}")
    print(f"TI Ana:  {ti_ana:.8f}")

if __name__ == "__main__":
    check_parity()
