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
import h5py
from pyscf.lib import H5FileWrap
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
if "initial guess" in params[0]:
    ks.chkfile=params[0]["initial guess"]
    if h5py.is_hdf5(ks.chkfile):
        ks.init_guess = 'chkfile'
ks.kernel()
if params[2].get("analyze", False):
    ks.analyze()
if params[2].get('scf summary', False):
    ks.dump_scf_summary()
scfTime=time.time()
print("\n   scf wall time: ",scfTime-startTime," s \n\n")


# Define MD Simulation (Ehrenfest or BOMD)
ehrenfest_data = params[7]
qmd_data = input_data.get('qmd')
ehrenfest_enabled = ehrenfest_data and ehrenfest_data.get('enabled', False)
qmd_enabled = qmd_data is not None

from gpu4pyscf.tdscf.ehrenfest import EhrenfestMD, BOMD

if qmd_enabled:
    rt = BOMD(ks)
    md_data = qmd_data
    dt = md_data.get('dt_nucl', 0.05) 
    tmax = md_data.get('nstep_nucl', 10) * dt
    propagator = 'none'
    print("Born-Oppenheimer Molecular Dynamics (QMD) enabled.")
elif ehrenfest_enabled:
    rt = EhrenfestMD(ks, basis=params[3].get('propagation basis','OAO'))
    md_data = ehrenfest_data
    dt = params[3].get('dt', 0.05)
    tmax = params[3].get('tmax', 1.0)
    propagator = params[3].get('propagator', 'magnus_interpol')
    print("Ehrenfest Dynamics enabled.")
else:
    rt = RTTDDFT(ks,basis=params[3].get('propagation basis','MO'))
    md_data = None
    dt = params[3].get('dt', 0.05)
    tmax = params[3].get('tmax', 1.0)
    propagator = params[3].get('propagator', 'magnus_interpol')

# Configure MD Parameters (Velocities, Thermostats)
if qmd_enabled or ehrenfest_enabled:
    if 'velocities' in md_data:
        rt.velocities = np.array(md_data['velocities'])
    
    # Thermostat
    thermo = md_data.get('thermostat')
    if thermo:
        if isinstance(thermo, str):
             parts = thermo.split()
             rt.thermostat = parts[0].lower()
             if len(parts) > 1: rt.tau = float(parts[1])
        rt.target_temp = md_data.get('targ_temp', 298.15)
        print(f"Thermostat enabled: {rt.thermostat} at {rt.target_temp}K (tau={rt.tau} au)")

rt.verbose=4
rt.mu_spin='total'
rt.record_occ=True

## Define Field Parameters
fieldType = params[4].get("type","gaussian")
E0 = params[4].get('E0', 0.0)
t0 = params[4].get('t0', 0.0)
sigma = params[4].get('sigma', 1.0)
freq = params[4].get('freq', 0.0)
phase = params[4].get('phase', 0)
polarization = params[4].get('polarization', 'x')
hand = params[4].get('hand','right')
# ... (match fieldType.lower() block remains same)

# ... (getTargetStateFreq block remains same)

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
traj_file = 'traj_'+params[6]['name']+'.dat'
force_file = 'forces_'+params[6]['name']+'.dat'
xyz_file = params[6]['name']+'.xyz'

# Initialize Callbacks
callbacks = []

# 1. S2 callback (before logger so results['s2'] is populated for logging)
if params[3].get('S2', False):
    if not isCs:
        print("S2 reporting enabled for unrestricted system.")
        s2_cb = rtu.S2Callback(mol)
        callbacks.append(s2_cb)
    else:
        print("Warning: S2 reporting requested but system is closed-shell. Skipping.")

# 2. Force Logger (if enabled in RT but not full MD)
if params[3].get('forces', False) and not (ehrenfest_enabled or qmd_enabled):
    print(f"Force logging enabled. Data will be written to {force_file}")
    f_log = rtu.ForceLogger(force_file, rt)
    callbacks.append(f_log)

# 3. Ehrenfest/QMD Logger (trajectory)
if ehrenfest_enabled or qmd_enabled:
    print(f"Dynamics trajectory will be written to {traj_file} and {xyz_file}")
    e_log = rtu.EhrenfestLogger(traj_file, mol)
    callbacks.append(e_log)
    x_log = rtu.XYZLogger(xyz_file, mol)
    callbacks.append(x_log)

# 4. Logging callback 
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

# start Run
if qmd_enabled:
    print(f"Starting QMD (BOMD). Data will be written to {output_file}...")
    results = rt.kernel(times=times, dt=dt, callback=callback_fn)
else:
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
print("\n   rttddft wall time: ",endTime-scfTime," s \n\n")
print("\n   wall time: ",endTime-startTime," s \n\n")

