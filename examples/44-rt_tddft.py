import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

# 1. Setup Molecule and Ground State
mol = gto.M(
    atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    basis='6-31g',
    verbose=3
)

# Use RKS with a hybrid functional
ks = dft.RKS(mol)
ks.xc = 'pbe0' 
ks.kernel()

# 2. Define RT-TDDFT simulation
rt = RTTDDFT(ks)

# Define a delta pulse at t=0
# In practice, for a delta pulse, we can just perturb the initial density matrix
# or use a very short, strong field.
# Here we use a field function that is a narrow Gaussian to simulate a pulse.
def field_fn(t):
    # Gaussian pulse at t=0.1 au
    t0 = 0.1
    sigma = 0.02
    E0 = 0.01 # Field strength in au
    # Pulse in x-direction
    val = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
    return [val, 0, 0]

rt.field_fn = field_fn

# 3. Run propagation
dt = 0.05
times = np.arange(0, 2.0, dt) # Short run for testing
results = rt.kernel(times=times, dt=dt/2)

print("\nRT-TDDFT Propagation Finished")
print(f"Final dipole: {results['dip'][-1]}")
print(f"Final energy: {results['energy'][-1]}")

# 4. Optional: Save results for analysis
# np.save('rt_results.npy', results)
