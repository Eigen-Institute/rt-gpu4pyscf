import numpy as np
import cupy
import sys, time
import json
import argparse
from pyscf import gto,symm
from pyscf.tools import cubegen,molden
from gpu4pyscf import dft,tdscf
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT
from gpu4pyscf.tdscf import rtutils as rtu

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Linear Response TDDFT')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

# Load Input Data
with open(args.input, 'r') as f:
    input_data = json.load(f)
properties = input_data.get("property", {})
mol_data = input_data.get('molecule', {})
theory_data = input_data.get('theory', {})
tddft_data = input_data.get('tddft', {})

# Print Input Parameters
print('')
print("*"*100)
print("Reading calculation input from:"+args.input)
print("")
print(input_data)
print("*"*100)

# Name of the calculation
calcName = input_data.get('calcName', 'tddft_calc')
calcName = calcName.replace(" ","_").lower()

# Define Molecule
mol = gto.M(
    atom=mol_data.get('atom', 'tet.xyz'),
    basis=mol_data.get('basis', '3-21g'),
    verbose=mol_data.get('verbose', 4),
    charge=mol_data.get('charge', 0),
    spin=mol_data.get('spin', 0),
    symmetry=mol_data.get('symmetry',False)
)

# Define Theory
if theory_data["shell"].lower() == "open":
    ks = dft.UKS(mol)
    isOs = True
else:
    ks = dft.RKS(mol)
    isCs = True
ks.xc = theory_data.get('xc', 'pbe0')
ks.chkfile=theory_data.get('initial guess',calcName+'.chk')
ks.init_guess = 'chkfile'
ks.kernel()

# Perform Analyze
if "analyze" in properties and properties["analyze"]:
    ks.analyze()
if properties["scf summary"]:
    ks.dump_scf_summary()

# Print Symmetry Information
if mol.symmetry:
    mo_coeff = ks.mo_coeff
    if theory_data["shell"].lower() == "open":
        # For UKS, mo_coeff is (2, nbasis, nmo) or [mo_a, mo_b]
        mo_a = mo_coeff[0].get() if hasattr(mo_coeff[0], 'get') else mo_coeff[0]
        mo_b = mo_coeff[1].get() if hasattr(mo_coeff[1], 'get') else mo_coeff[1]
        irreps_a = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_a)
        irreps_b = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_b)
        print("\n   Alpha MO irreps:", irreps_a)
        print("   Beta  MO irreps:", irreps_b, "\n")
    else:
        mo = mo_coeff.get() if hasattr(mo_coeff, 'get') else mo_coeff
        irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        print("\n   MO irreps:", irreps, "\n")


# TDDFT
if isOs:
    td = tdscf.uks.TDDFT(ks)
else:
    td = tdscf.rks.TDDFT(ks)

# Number of Excited States

nstates = tddft_data.get('nstates', 5)
td.kernel(nstates=nstates)
td.analyze()

# NTOs
if properties["NTO"]:
    for i in range(1,nstates+1):
        w, n = td.get_nto(state=i, verbose=4)
        
        # Convert CuPy arrays to NumPy for molden export
        if isinstance(n, (list, tuple)):
            n = [c.get() if hasattr(c, 'get') else c for c in n]
        elif hasattr(n, 'get'):
            n = n.get()

        # Check if n is a 3D array (UKS stacked), convert to list
        if hasattr(n, 'ndim') and n.ndim == 3:
            n = [n[0], n[1]]

        if isinstance(n, list):
            # UKS: write alpha and beta NTOs to separate files
            molden.from_mo(mol, f'nto-td-{i}-alpha.molden', n[0])
            molden.from_mo(mol, f'nto-td-{i}-beta.molden', n[1])
        else:
            molden.from_mo(mol, f'nto-td-{i}.molden', n)

# Generate Transition Density Cube for Excited States
# Single Excited State
cubeList=tddft_data.get('transition density cube',[1])
cubeSpin=tddft_data.get('transition density spin','total')
if cubeList is not None:
    if "all" in cubeList:
        cubeList = list(range(1,nstates+1))
    for i in range(len(cubeList)):
        # Write Excited State Cubes
        state_idx=cubeList[i]-1
        rtu.write_transition_density_cube(td, state_idx, "tdens-"+str(state_idx+1)+"_"+calcName+".cube",margin=4.0,spin=cubeSpin)
            
