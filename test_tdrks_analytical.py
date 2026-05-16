import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft.rks import RKS
from gpu4pyscf.tdscf.rks import TDA
from gpu4pyscf.hessian.tdrks import Hessian

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
mf = RKS(mol)
mf.xc = 'pbe'
mf.run()

td = TDA(mf)
td.nstates = 2
td.kernel()

# Semi-analytical (FD on analytical gradient)
h_semi = Hessian(td)
h_semi.method = 'semi-analytical'
h_semi_result = h_semi.kernel(state=0)

# Full analytical Hessian
h_anal = Hessian(td)
h_anal.method = 'analytical'
h_anal_result = h_anal.kernel(state=0)

print('H2 / STO-3G RKS-PBE TDA (state 0)')
val_semi = float(h_semi_result[0,0,0,0])
val_anal = float(h_anal_result[0,0,0,0])
ratio = val_anal / val_semi if abs(val_semi) > 1e-12 else 0.0
print(f'Semi-Analytical [0,0,0,0]: {val_semi:14.8f}')
print(f'Analytical      [0,0,0,0]: {val_anal:14.8f}')
print(f'Ratio:                     {ratio:14.4f}x')
