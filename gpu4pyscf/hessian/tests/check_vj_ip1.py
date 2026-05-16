import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.rhf import _get_jk_ip1

def check_fock_grad():
    mol = pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    dm0 = mf.make_rdm1()
    
    vj, vk = _get_jk_ip1(mol, dm0)
    vj = vj.reshape(mol.natm, 3, mol.nao, mol.nao)
    
    for ia in range(mol.natm):
        for ix in range(3):
            mat = vj[ia, ix]
            asym = float(cp.max(cp.abs(mat - mat.T)))
            print(f"Atom {ia} Dir {ix}: vj asymmetry = {asym:.3e}")
            
    # Check TI of vj sum
    ti_vj = vj.sum(axis=0)
    print(f"vj Sum Max: {float(cp.abs(ti_vj).max()):.3e}")

if __name__ == "__main__":
    check_fock_grad()
