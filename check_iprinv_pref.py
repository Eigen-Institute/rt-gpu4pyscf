import cupy as cp
from pyscf import gto
from gpu4pyscf.df import int3c2e

mol = gto.M(atom='H 0 0 0; H 0 0 1.0', basis='sto-3g')
nao = mol.nao
dm = cp.zeros((nao, nao))
dm[1,1] = 1.0 # Basis on nucleus 1 (z=1.0), potential from nucleus 0 (z=0)

# Method 1: get_dh1e
val1 = int3c2e.get_dh1e(mol, dm)[0,2]

# Method 2: manual iprinv
with mol.with_rinv_at_nucleus(0):
    vrinv = mol.intor('int1e_iprinv', comp=3)
# Operator part of nucleus 0: -Z_0 * <mu | iprinv_0 | nu>
val2 = -1.0 * vrinv[2,1,1]

print(f"get_dh1e:   {float(val1):.8f}")
print(f"manual:     {float(val2):.8f}")
