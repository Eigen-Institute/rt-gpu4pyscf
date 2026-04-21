import numpy as np
import cupy
import sys, time
import json
import argparse
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft,tdscf
from pyscf.geomopt import geometric_solver
sys.path.append("/home/craig/research/templates/eigen")
import footer
startTime=time.time()

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Geometry Optimization')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

# Load Input Data
with open(args.input, 'r') as f:
    input_data = json.load(f)

# Name of the calculation
calcName = input_data.get('calcName', 'geom_opt')
calcName = calcName.replace(" ","_").lower()

# Define Molecule
mol_data = input_data.get('molecule', {})
mol = gto.M(
    atom=mol_data.get('atom', 'opt.xyz'),
    basis=mol_data.get('basis', '3-21g'),
    verbose=mol_data.get('verbose', 4),
    charge=mol_data.get('charge', 0),
    spin=mol_data.get('spin', 0)
)

# Define Theory
theory_data = input_data.get('theory', {})
if theory_data["shell"] == "open":
    ks = dft.UKS(mol)
    isOs = True
if theory_data["shell"] == "closed":
    ks =  dft.RKS(mol)
    isCs = True
ks.xc = theory_data.get('xc', 'pbe0')
ks.chkfile = theory_data.get("initial guess",calcName+'.chk')
ks.init_guess = 'chk'
ks.kernel()

# Geometry Optimization
geomopt_data = input_data.get('geomopt', {})
maxsteps = geomopt_data.get('maxsteps', 100)
optimizedFile = geomopt_data.get('output_file', calcName + '_opt.xyz')

mol_eq = geometric_solver.optimize(ks, maxsteps=maxsteps)
print(mol_eq.tostring())

# Get optimized coordinates in Angstroms
coords = mol_eq.atom_coords(unit='Ang')
symbols = [atom[0] for atom in mol_eq.atom]

# Print in XYZ format
print(len(symbols))
print("Optimized geometry")
for sym, (x, y, z) in zip(symbols, coords):
    print(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}")
# Write in XYZ format
with open(optimizedFile, 'w') as f:
    f.write(str(len(symbols))+'\n')
    f.write("Optimized geometry "+calcName+" @"+mol.basis+"/"+ks.xc+"\n")
    for sym, (x, y, z) in zip(symbols, coords):
        f.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

# Properties calculation (Hessian, Vib)
props = input_data.get('properties', {})

if props.get('Hessian', False) or props.get('vib', False):
    print("\n" + "="*50)
    print("--- Properties Calculation ---")
    print("="*50)
    
    # Ensure ks is consistent with optimized mol
    # Re-run kernel at optimized geometry for accurate Hessian
    ks.reset(mol_eq)
    ks.kernel()
    
    print("\nCalculating analytical Hessian...")
    h_start = time.time()
    hess_obj = ks.Hessian()
    hessian = hess_obj.kernel()
    h_end = time.time()
    print(f"Hessian calculation time: {h_end - h_start:.2f} s")
    
    if props.get('Hessian', False):
        print("\nAnalytical Hessian (Ha/Bohr^2):")
        # Print a small part if it's large, or just summary
        print(f"Hessian shape: {hessian.shape}")
        # Save Hessian to file
        hess_file = f"{calcName}_hessian.npy"
        np.save(hess_file, hessian)
        print(f"Full Hessian saved to {hess_file}")

    if props.get('vib', False):
        from pyscf.hessian import thermo
        print("\nPerforming Vibrational Analysis...")
        vib_info = thermo.harmonic_analysis(ks.mol, hessian)
        
        print("\nNormal Mode Summary:")
        print(f"{'Mode':>5} {'Freq (cm^-1)':>15} {'Red. Mass (AMU)':>15}")
        for i, (freq, rm) in enumerate(zip(vib_info['freq_wavenumber'], vib_info['reduced_mass'])):
            # Note: Negative frequencies in PySCF represent imaginary frequencies (TS or non-minimum)
            print(f"{i+1:5d} {freq:15.2f} {rm:15.4f}")
        
        # Save vibrational info to JSON
        vib_results = {
            "frequencies_cm1": vib_info['freq_wavenumber'].tolist(),
            "reduced_mass_amu": vib_info['reduced_mass'].tolist(),
            "normal_modes": vib_info['norm_mode'].tolist()
        }
        with open(f"{calcName}_vib.json", "w") as f:
            json.dump(vib_results, f, indent=2)
        print(f"\nVibrational analysis results saved to {calcName}_vib.json")

endTime=time.time()
print("\n\n     wall time:",str(endTime-startTime)+" s")
footer.print_footer()

