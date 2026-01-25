import numpy as np
import cupy
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

# 1. Setup Molecule
mol = gto.M(
    atom='H 0 0 0; F 0 0 0.92',
    basis='def2-svp',
    verbose=3
)

ks = dft.RKS(mol)
ks.xc = 'pbe0' 
ks.kernel()

# 2. Setup RT-TDDFT
rt = RTTDDFT(ks)

# Define a pulse
def field_fn(t):
    val = 0.01 * np.exp(-(t-5.0)**2 / 2.0)
    return [0, 0, val]

rt.field_fn = field_fn

# 3. Define Cube Generation Callback
cube_interval = 20  # Write cube every 20 steps
step_counter = 0

def visualize_density(t, dm, results):
    global step_counter
    step_counter += 1
    
    if step_counter % cube_interval == 0:
        fname = f'density_t{t:.2f}.cube'
        print(f"Writing cube file: {fname}")
        
        # Convert GPU density to CPU
        dm_cpu = cupy.asnumpy(dm)
        
        # Generate Cube File
        # standard PySCF cubegen works with the CPU molecule and CPU density
        cubegen.density(mol, fname, dm_cpu)

# 4. Run Simulation
dt = 0.05
times = np.arange(0, 10.0, dt)

print("Starting RT-TDDFT with Cube generation...")
rt.kernel(times=times, dt=dt, callback=visualize_density)
