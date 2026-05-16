"""Debug: Compare analytical vs semi-analytical Hessian with detailed component breakdown."""
import numpy as np
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)

mf = RHF(mol).run()
td = TDA(mf)
td.nstates = 1
td.kernel()

# Semi-analytical (FD on analytical gradient) - the baseline
h_semi = Hessian(td)
h_semi.method = 'semi-analytical'  
h_semi_result = h_semi.kernel(state=0)

# Full analytical Hessian with fix applied
h_anal = Hessian(td)
h_anal.method = 'analytical'
h_anal_result = h_anal.kernel(state=0)

print('H2 / STO-3G TDA (state 0)')
header = f'{"Component":<15} {"Semi-Analytical":>14} {"Analytical":>14} {"Diff":>14} {"Ratio":>8}'
print(header)
print('-' * len(header))

natm = mol.natm
# Compare all components
for i0 in range(natm):
    for j0 in range(3):
        for k0 in range(natm):
            for l0 in range(3):
                val_semi = float(h_semi_result[i0,j0,k0,l0])
                val_anal = float(h_anal_result[i0,j0,k0,l0])
                diff = abs(val_anal - val_semi)
                ratio = abs(val_anal / val_semi) if abs(val_semi) > 1e-12 else (float('inf') if abs(val_anal) > 1e-12 else 0.0)
                print(f'  [{i0},{j0}]-[{k0},{l0}]    {val_semi:>14.8f} {val_anal:>14.8f} {diff:>14.6e} {ratio:>7.3f}x')

max_diff = float(abs(h_anal_result - h_semi_result).max())
print(f'\nMax absolute difference: {max_diff:.6e}')
if abs(float(h_semi_result[0,0,0,0])) > 1e-12:
    print(f'Ratio (analytical/FD):  {abs(float(h_anal_result[0,0,0,0]) / float(h_semi_result[0,0,0,0])):.4f}x')

# Also check that the Hessian is symmetric
h_flat = h_anal_result.reshape(3 * natm, 3 * natm)
max_asym = float(abs(h_flat - h_flat.T).max())
print(f'Max asymmetry: {max_asym:.6e} (expected < 1e-7)')

# Check translational invariance for Semi-Analytical
print("\nTI check (Semi-Analytical):")
for i in range(natm):
    for x in range(3):
        for y in range(3):
            ti_sum = float(h_semi_result[i,x,:,y].sum())
            if abs(ti_sum) > 1e-4:
                print(f"  Atom {i} dir {x}-{y} violation: {ti_sum:.4e}")

# Check translational invariance for Analytical
print("\nTI check (Analytical):")
for i in range(natm):
    for x in range(3):
        for y in range(3):
            ti_sum = float(h_anal_result[i,x,:,y].sum())
            if abs(ti_sum) > 1e-4:
                print(f"  Atom {i} dir {x}-{y} violation: {ti_sum:.4e}")
