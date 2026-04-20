#
#   Template script for Ehrenfest Molecular Dynamics
#   Usage:
#       python 54-ehrenfest_json.py -i 54-ehrenfest_input.json
#

import numpy as np
import cupy
import time, sys
import json, argparse
from gpu4pyscf import dft, tdscf
from gpu4pyscf.tdscf import rtutils as rtu

startTime = time.time()

# Command line arguments
parser = argparse.ArgumentParser(description='GPU4PySCF Ehrenfest Dynamics')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file', required=True)
args = parser.parse_args()

# Load and Parse Input
params = rtu.parseInputJson(args.input)
with open(args.input, 'r') as f:
    input_data = json.load(f)

# 1. Setup Molecule
mol = gto.M(**params[1])

# 2. Setup Ground State SCF
if params[0]["shell"] == "open":
    ks = dft.UKS(mol)
    isCs = False
else:
    ks = dft.RKS(mol)
    isCs = True

ks.xc = params[0]['xc']
if "initial guess" in params[0]:
    ks.chkfile = params[0]["initial guess"]
    ks.init_guess = 'chkfile'
ks.kernel()
scfTime = time.time()

# 3. Setup Ehrenfest Simulation
ehrenfest_data = params[7]
if not (ehrenfest_data and ehrenfest_data.get('enabled', False)):
    print("Error: Ehrenfest is not enabled in the JSON input.")
    sys.exit(1)

from gpu4pyscf.tdscf.ehrenfest import EhrenfestMD
rt = EhrenfestMD(ks, basis=params[3].get('propagation basis','OAO'))

if 'velocities' in ehrenfest_data:
    rt.velocities = np.array(ehrenfest_data['velocities'])

rt.verbose = 4
dt = params[3]['dt']
tmax = params[3]['tmax']
n_electronic = ehrenfest_data.get('n_electronic', 1)
propagator = params[3].get('propagator', 'magnus_interpol')

# Output file names
calc_name = params[6]['name']
output_file = params[3].get("output", f'rt_data.{calc_name}.dat')
traj_file = f'traj_{calc_name}.dat'
xyz_file = f'{calc_name}.xyz'

# 4. Initialize Callbacks
callbacks = []

# Trajectory and XYZ Logging
print(f"Dynamics trajectory will be written to {traj_file} and {xyz_file}")
callbacks.append(rtu.EhrenfestLogger(traj_file, mol))
callbacks.append(rtu.XYZLogger(xyz_file, mol))

# Property Logging
callbacks.append(rtu.RTLogger(output_file, f'occ_{calc_name}.dat', rt.field_fn, isCs))

# Combined Callback
callback_fn = rtu.MultiCallback(callbacks)

# 5. Run Propagation
times = np.arange(0, tmax, dt)
print(f"Starting Ehrenfest Dynamics. Data will be written to {output_file}...")
results = rt.kernel(times=times, dt=dt, propagator=propagator, callback=callback_fn, n_electronic=n_electronic)

endTime = time.time()
print(f"\n   Ehrenfest wall time: {endTime-scfTime:.2f} s")
print(f"   Total wall time:     {endTime-startTime:.2f} s\n")
