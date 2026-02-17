#
#   Template script for RT-TDDFT
#   Use in conjunction with rttddft.json:
#
#       python rttddft_json.py -i rttddft.json
#
#
import numpy as np
import cupy
import time, sys
import json, re
import argparse
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft,tdscf
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT
from gpu4pyscf.tdscf import rtutils as rtu
startTime=time.time()

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
viz_data = rttddft_data.get('visualization')
mol_data = input_data.get('molecule', {})

# Name of the calculation
calcName = input_data.get('calcName', 'rttddft_calc')
calcName = calcName.replace(" ","_").lower()

# Define calculation parameters
params = rtu.parseInputJson(args.input)

# Define Molecule
mol = gto.M(
    atom=params[1]["atom"],
    basis=params[1]['basis'],
    verbose=params[1]['verbose'],
    charge=params[1]['charge'],
    spin=params[1]['spin']
)


# Define theory
if params[0]["shell"] == "open":
    ks = dft.UKS(mol)
    isOs = True
    isCs = False
elif params[0]["shell"] == "closed":
    ks = dft.RKS(mol)
    isCs = True
    isOs = False
ks.xc = params[0]['xc']
ks.chkfile=params[0]["initial guess"]
ks.init_guess = 'chkfile'
ks.kernel()
if params[2]["analyze"]:
    ks.analyze()
if params[2]['scf summary']:
    ks.dump_scf_summary()


# Define RT-TDDFT simulation

rt = RTTDDFT(ks,basis=params[3].get('propagation basis','MO'))
rt.verbose=4
rt.mu_spin='total'
rt.record_occ=True
dt = params[3]['dt']
tmax = params[3]['tmax']
propagator = params[3].get('propagator', 'magnus_interpol')

## Define Field Parameters
fieldType = params[4].get("type","gaussian")
E0 = params[4].get('E0', 0.01)
t0 = params[4].get('t0', 150)
sigma = params[4].get('sigma', 100)
freq = params[4].get('freq', 0.05)
phase = params[4].get('phase', 0)
polarization = params[4].get('polarization', 'x')
hand = params[4].get('hand','right')

# Dynamic Target Selection
target_state = params[3].get('target')
if target_state is not None:
    freq, polarization = rtu.getTargetStateFreq(params,target_state)

# Assign field type
match fieldType.lower():
    case "gaussian" | "gauss" :
        rt.field_fn = rtu.Field.gaussian_pulse(E0,t0,sigma,freq,phase,polarization,hand)
        rtu.Field.printField(fieldType=fieldType,E0=E0,t0=t0,sigma=sigma,freq=freq,phase=phase,polarization=polarization, hand=hand)
    case "cw":
        rt.field_fn = rtu.Field.cw_field(E0,freq,phase,polarization,hand)    
        rtu.Field.printField(fieldType=fieldType,E0=E0,freq=freq,phase=phase,polarization=polarization, hand=hand)

# Output file names
output_file = params[3].get("output",'rt_data.'+params[6]['name']+'.dat')
mo_file = 'occ_'+params[6]['name']+'.dat'

# Initialize Callbacks
callbacks = []
# Logging callback 
logger = rtu.RTLogger(output_file,mo_file,rt.field_fn,isCs)
callbacks.append(logger)
# 2. Visualization callback
viz_data = rttddft_data.get('visualization')
if params[5]:
    interval = params[5]['interval']
    treference = params[5]['treference']
    spin = params[5].get('spin', 'total')
    visualizer = rtu.CubeVisualizer(mol, interval=interval, prefix=params[6]['name'], treference=treference, spin=spin)
    callbacks.append(visualizer)
    if treference is not None:
        print(f"Visualization enabled ({spin} density). Writing difference cubes relative to t={treference} every {interval} steps.")
    else:
        print(f"Visualization enabled ({spin} density). Writing cubes every {interval} steps.")

# Combine
callback_fn = rtu.MultiCallback(callbacks)


# Run Propagation
times = np.arange(0, tmax, dt) # Short test run

# Start Run
#rtu.printParams()
print(f"Starting RT-TDDFT. Data will be written to {output_file}...")
results = rt.kernel(times=times, dt=dt, propagator=propagator, callback=callback_fn)

# Finished
print("\n\n   Propagation finished.\n")

#######################################################
#
#   Calculation Finished. Print Footer.
#
#######################################################

endTime=time.time()
print("\n   wall time: ",endTime-startTime," s \n\n")

