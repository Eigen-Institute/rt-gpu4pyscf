import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.hessian.rhf import Hessian, _e_hcore_generator, _partial_ejk_ip2

def check_generator_ti(system='H2'):
    if system == 'H2':
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    else:
        mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto-3g', verbose=0)
    
    mf = RHF(mol).run()
    dm = mf.make_rdm1()
    h_obj = Hessian(mf)
    
    # 1. H-core TI
    de_hcore = _e_hcore_generator(h_obj, cp.asarray(dm))
    h_hcore = np.zeros((mol.natm, mol.natm, 3, 3))
    for i in range(mol.natm):
        for j in range(mol.natm):
            h_hcore[i,j] = de_hcore(i,j).get()
    
    print(f"System: {system}")
    ti_hcore = np.abs(h_hcore.sum(axis=1)).max()
    print(f"  H-core TI Violation: {ti_hcore:.2e}")
    
    # 2. ERI TI
    vhfopt = mf._opt_gpu.get(mol.omega)
    ejk_ip2 = _partial_ejk_ip2(mol, cp.asarray(dm), vhfopt).get()
    ti_eri = np.abs(ejk_ip2.sum(axis=1)).max()
    print(f"  ERI ip2 TI Violation: {ti_eri:.2e}")

check_generator_ti('H2')
check_generator_ti('H2O')
