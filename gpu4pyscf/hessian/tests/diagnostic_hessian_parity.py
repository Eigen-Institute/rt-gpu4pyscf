import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess

def test_h2_parity():
    # H2 along z-axis
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=4)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 1
    td.kernel()

    print("H2/STO-3G TDA State 0 Parity Check")
    
    h_semi_obj = tdrhf_hess.Hessian(td)
    h_semi_obj.method = 'semi-analytical'
    h_semi_obj.verbose = 4
    h_semi = h_semi_obj.kernel()
    
    h_anal_obj = tdrhf_hess.Hessian(td)
    h_anal_obj.method = 'analytical'
    h_anal_obj.verbose = 4
    h_anal = h_anal_obj.kernel()
    
    natm = mol.natm
    for i in range(natm):
        for j in range(natm):
            for x in range(3):
                for y in range(3):
                    val_s = float(h_semi[i,x,j,y])
                    val_a = float(h_anal[i,x,j,y])
                    if abs(val_s) > 1e-4 or abs(val_a) > 1e-4:
                        ratio = val_a / val_s if abs(val_s) > 1e-8 else float('nan')
                        print(f"[{i},{x}]-[{j},{y}]: Semi={val_s:12.8f} Anal={val_a:12.8f} Ratio={ratio:6.3f}")

    # Check TI
    ti_s = cp.abs(h_semi.sum(axis=2)).max()
    ti_a = cp.abs(h_anal.sum(axis=2)).max()
    print(f"TI Violation: Semi={float(ti_s):.3e} Anal={float(ti_a):.3e}")

if __name__ == "__main__":
    test_h2_parity()
