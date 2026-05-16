import numpy as np
from pyscf import gto
mol = gto.M(atom='H 0 0 0; H 0 0 1.0', basis='sto-3g')
coords = mol.atom_coords()
dr = 0.001
coords_p = coords.copy(); coords_p[0,2] += dr
mol_p = mol.copy(); mol_p.set_geom_(coords_p, unit='Bohr'); mol_p.build()
coords_m = coords.copy(); coords_m[0,2] -= dr
mol_m = mol.copy(); mol_m.set_geom_(coords_m, unit='Bohr'); mol_m.build()

def get_nuc_pot(m):
    # Potential at origin from nucleus 0
    # E = -Z/r
    # We want integral <1| -Z/r_0 |1>
    # Simplest is just the value at a point
    return m.intor('int1e_nuc')

p0 = get_nuc_pot(mol_p)
m0 = get_nuc_pot(mol_m)
fd = (p0 - m0) / (2 * dr)
print(f"FD d/dz (V_nuc): {fd[0,0]:.8f}")

# Analytical ipnuc
ipnuc = mol.intor('int1e_ipnuc', comp=3)
print(f"ipnuc [2,0,0]:   {ipnuc[2,0,0]:.8f}")

# iprinv
with mol.with_rinv_at_nucleus(0):
    iprinv = mol.intor('int1e_iprinv', comp=3)
print(f"iprinv [2,0,0]:  {iprinv[2,0,0]:.8f}")
