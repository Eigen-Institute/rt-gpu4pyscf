import numpy as np
from pyscf import gto, scf, tdscf
from pyscf.hessian import tdrhf
mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
td = tdscf.TDA(mf)
td.nstates = 1
td.kernel()
h = tdrhf.Hessian(td)
x_y = td.xy[0]
omega = td.e[0]
mo1, mo_e1 = mf.nuc_grad_method().Hessian().solve_mo1()
mo_coeff = mf.mo_coeff
# PySCF make_cptddft_rhs is not a separate function, we have to look at solve_cptddft
try:
    x1, y1 = tdrhf.solve_cptddft(h, x_y, omega, mo1, mo_e1)
    print("PySCF x1 norm:", np.linalg.norm(x1))
except Exception as e:
    print("Error:", e)
