#
#   Template script for FSSH
#   Use in conjunction with 58-fssh.json:
#
#       python 58-fssh_json.py -i 58-fssh.json
#
import numpy as np
import cupy
import time, sys
import json, re
import argparse
import h5py
from pyscf import gto
from gpu4pyscf import dft, tdscf
from gpu4pyscf.tdscf.fssh import FSSHMD
from gpu4pyscf.tdscf import rtutils as rtu

startTime=time.time()

# Parse Command Line Arguments
parser = argparse.ArgumentParser(description='GPU4PySCF FSSH Dynamics')
parser.add_argument('--input','-i', type=str, help='Path to input JSON file')
args = parser.parse_args()

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
    isCs = False
else:
    ks = dft.RKS(mol)
    isCs = True
ks.xc = params[0]['xc']
if "initial guess" in params[0]:
    ks.chkfile=params[0]["initial guess"]
    if h5py.is_hdf5(ks.chkfile):
        ks.init_guess = 'chkfile'
ks.kernel()
if params[2].get("analyze", False):
    ks.analyze()
scfTime=time.time()

# Define FSSH Dynamics
fssh_data = params[8]
if fssh_data is None or not fssh_data.get('enabled', False):
    print("FSSH not enabled in input. Exiting.")
    sys.exit()

# Build TDDFT object for FSSH
tda = bool(fssh_data.get('tda', True))
nstates = int(fssh_data.get('nstates', 5))
td = ks.TDA() if tda else ks.TDDFT()
td.nstates = nstates
td.kernel()

state = int(fssh_data.get('state', 1))
scheme = fssh_data.get('scheme', 'fssh')
decoherence = fssh_data.get('decoherence', 'idc')
seed = fssh_data.get('seed')

rt = FSSHMD(ks, td=td, state=state, scheme=scheme, decoherence=decoherence, seed=seed)

# Configure MD Parameters
rt.frozen = fssh_data.get('frozen', False)
if 'velocities' in fssh_data:
    vel_input = fssh_data['velocities']
    if isinstance(vel_input, str):
        print(f"Loading velocities from {vel_input}...")
        rt.velocities = rtu.load_velocities_from_xyz(vel_input)
    else:
        rt.velocities = np.array(vel_input)
elif 'init_vel_temperature' in fssh_data:
    T_init = float(fssh_data['init_vel_temperature'])
    rt.velocities = rtu.maxwell_boltzmann_velocities(rt.masses, T_init)
    rtu.remove_com_momentum(rt.masses, rt.velocities)

# Thermostat
thermo = fssh_data.get('thermostat')
if thermo:
    if isinstance(thermo, str):
         parts = thermo.split()
         rt.thermostat = parts[0].lower()
         if len(parts) > 1: rt.tau = float(parts[1])
    rt.target_temp = fssh_data.get('targ_temp', 298.15)

# Output files
output_name = params[6]['name']
traj_file = f'traj_{output_name}.dat'
xyz_file = f'{output_name}.xyz'
output_file = f'rt_data.{output_name}.dat'

# Callbacks
callbacks = []
print(f"Dynamics trajectory will be written to {traj_file} and {xyz_file}")
callbacks.append(rtu.EhrenfestLogger(traj_file, mol))
callbacks.append(rtu.XYZLogger(xyz_file, mol))
callbacks.append(rtu.RTLogger(output_file, None, None, isCs))

callback_fn = rtu.MultiCallback(callbacks)

# Run
dt = params[3].get('dt', 0.02)
tmax = params[3].get('tmax', 10.0)
times = np.arange(0, tmax, dt)

print(f"Starting FSSH MD (state={state})...")
results = rt.kernel(times=times, dt=dt, callback=callback_fn)

endTime=time.time()
print(f"\n   FSSH wall time: {endTime-scfTime:.2f} s")
print(f"   Total wall time: {endTime-startTime:.2f} s")
