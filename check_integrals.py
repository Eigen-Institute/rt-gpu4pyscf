import numpy as np
from pyscf import gto
mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')

ipipkin = mol.intor('int1e_ipipkin', comp=9).reshape(3,3,2,2)
print("T^{xx} [0,0]:", ipipkin[0,0,0,0])

ipipnuc = mol.intor('int1e_ipipnuc', comp=9).reshape(3,3,2,2)
print("V^{xx} [0,0]:", ipipnuc[0,0,0,0])

# Full H^{xx}
print("H^{xx} [0,0]:", ipipkin[0,0,0,0] + ipipnuc[0,0,0,0])
