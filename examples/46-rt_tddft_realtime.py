import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

# 1. Setup Molecule
mol = gto.M(
    atom='H 0 0 0; F 0 0 0.92',
    basis='6-31g',
    verbose=3
)

ks = dft.RKS(mol)
ks.xc = 'pbe0' 
ks.kernel()

# 2. Define RT-TDDFT simulation
rt = RTTDDFT(ks)

# Delta pulse
E0 = 0.005 
t0 = 0.01
sigma = 0.005
def field_fn(t):
    val = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
    return [0, 0, val]

rt.field_fn = field_fn

# 3. Define Real-Time Callback
output_file = 'rt_data.dat'

# Initialize file with header
with open(output_file, 'w') as f:
    f.write(f"# Time (au) | Energy (Ha) | Dipole X | Dipole Y | Dipole Z\n")

def realtime_writer(t, dm, results):
    '''
    Callback function to write data to file.
    '''
    # Extract latest values
    energy = results['energy'][-1]
    dip = results['dip'][-1] # [dx, dy, dz] 
    
    # Write to file (append mode)
    with open(output_file, 'a') as f:
        f.write(f"{t:12.6f} {energy:18.10f} {dip[0]:14.8f} {dip[1]:14.8f} {dip[2]:14.8f}\n")
    
    # Optional: Print simple status to stdout if not using logger
    # print(f" Step t={t:.4f} written.")

# 4. Run Propagation
dt = 0.05
times = np.arange(0, 2.0, dt) # Short test run

print(f"Starting RT-TDDFT. Data will be written to {output_file}...")
results = rt.kernel(times=times, dt=dt/2, callback=realtime_writer)

print("Propagation finished.")
print("First 5 lines of output data:")
with open(output_file, 'r') as f:
    for i in range(5):
        print(f.readline().strip())
