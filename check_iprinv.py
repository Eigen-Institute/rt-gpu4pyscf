import cupy as cp
from pyscf import gto
from gpu4pyscf.df import int3c2e

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')
nao = mol.nao
dm = cp.random.rand(nao, nao)
dm = dm + dm.T

# Method 1: get_dh1e
dh1e = int3c2e.get_dh1e(mol, dm)
val1 = dh1e[0,2]

# Method 2: Manual iprinv
with mol.with_rinv_at_nucleus(0):
    vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
# Operator part only:
with mol.with_rinv_at_nucleus(0):
    vrinv_op = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * -mol.atom_charge(0)
val2 = cp.trace(dm @ vrinv_op[2])

print(f"get_dh1e:   {float(val1):.8f}")
print(f"iprinv trace: {float(val2):.8f}")
