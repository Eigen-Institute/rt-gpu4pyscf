import numpy as np
import cupy
import time, sys
import json
import argparse
import re
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft,tdscf
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT
from gpu4pyscf.tdscf import rtutils as rtu
sys.path.append("/home/craig/research/templates/eigen")
import footer
startTime=time.time()
footer.print_footer()

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Real-Time TDDFT')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

# Load Input Data
with open(args.input, 'r') as f:
    input_data = json.load(f)

print('')
print("*"*100)
print("Reading calculation input from:"+args.input)
print("")
print(input_data)
print("*"*100)

# Parse main level inputs
theory_data = input_data.get('theory', {})
rttddft_data = input_data.get('rttddft', {})
field_data = rttddft_data.get('field', {})
properties = input_data.get('property', {})

# Name of the calculation
calcName = input_data.get('calcName', 'rttddft_calc')
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

# Define theory
if theory_data["shell"] == "open":
    ks = dft.UKS(mol)
    isOs = True
    isCs = False
elif theory_data["shell"] == "closed":
    ks = dft.RKS(mol)
    isCs = True
    isOs = False
ks.xc = theory_data.get('xc', 'pbe0')
ks.chkfile=theory_data.get("initial guess",calcName+'.chk')
ks.init_guess = 'chkfile'
ks.kernel()
if properties["analyze"]:
    ks.analyze()
if properties['scf summary']:
    ks.dump_scf_summary()


# Define RT-TDDFT simulation

rt = RTTDDFT(ks)
rt.verbose=4
rt.mu_spin='total'
rt.record_occ=True
dt = rttddft_data.get('dt', 0.2)
tmax = rttddft_data.get('tmax', 1100)
propagator = rttddft_data.get('propagator', 'magnus_interpol')

## Define Field Parameters
fieldType = field_data.get("type","gaussian")
E0 = field_data.get('E0', 0.01)
t0 = field_data.get('t0', 150)
sigma = field_data.get('sigma', 100)
freq = field_data.get('freq', 0.05)
phase = field_data.get('phase', 0)
polarization = field_data.get('polarization', 'x')
hand = field_data['hand']

# Dynamic Target Selection
target_state = rttddft_data.get('target')
if target_state is not None:
    tddft_file = rttddft_data.get('tddft file')
    if tddft_file:
        print(f"\nParsing TDDFT output from {tddft_file} for target state {target_state}...")
        parsed_states = rtu.parse_tddft_output(tddft_file)
        if target_state in parsed_states:
            state_info = parsed_states[target_state]
            
            # Auto-set Frequency (eV -> Ha) unless overridden in JSON
            if 'freq' not in field_data or type(freq) is str:
                freq_ev = state_info['energy_ev']
                freq = freq_ev / 27.211386
                print(f"  Auto-setting freq to {freq:.6f} Ha ({freq_ev} eV)")
            
            # Auto-set Polarization unless overridden
            if 'polarization' not in field_data:
                dip = state_info.get('dipole', [0,0,0])
                abs_dip = [abs(d) for d in dip]
                max_idx = abs_dip.index(max(abs_dip))
                polarization = ['x', 'y', 'z'][max_idx]
                print(f"  Auto-setting polarization to '{polarization}' (Dipole: {dip})")
        else:
            print(f"  Warning: State {target_state} not found in {tddft_file}.")
    else:
        print("  Warning: 'target' specified but 'tddft file' is missing.")

# Assign field type
match fieldType.lower():
    case "gaussian" | "gauss" :
        rt.field_fn = rtu.Field.gaussian_pulse(E0,t0,sigma,freq,phase,polarization,hand)
        rtu.Field.printField(fieldType=fieldType,E0=E0,t0=t0,sigma=sigma,freq=freq,phase=phase,polarization=polarization, hand=hand)
    case "cw":
        rt.field_fn = rtu.Field.cw_field(E0,freq,phase,polarization,hand)    
        rtu.Field.printField(fieldType=fieldType,E0=E0,freq=freq,phase=phase,polarization=polarization, hand=hand)

# Output file names
output_file = rttddft_data.get("output",'rt_data_'+calcName+'.dat')
mo_file = 'occ_'+calcName+'.dat'

# Logging callback 
logger = rtu.RTLogger(output_file,mo_file,rt.field_fn,isCs)

# Run Propagation
times = np.arange(0, tmax, dt) # Short test run

# Start Run
#rtu.printParams()
print(f"Starting RT-TDDFT. Data will be written to {output_file}...")
results = rt.kernel(times=times, dt=dt, propagator=propagator, callback=logger)

# Finished
print("\n\n   Propagation finished.\n")

#######################################################
#
#   Calculation Finished. Print Footer.
#
#######################################################

endTime=time.time()
print("\n   wall time: ",endTime-startTime," s \n\n")
footer.print_footer()

