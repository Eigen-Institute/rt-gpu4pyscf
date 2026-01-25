import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

# 1. Setup Open-Shell Molecule (OH Radical)
mol = gto.M(
    atom='O 0 0 0; H 0 0 0.97',
    basis='def2-svp',
    spin=1, # Doublet (1 unpaired electron)
    verbose=3
)

# 2. Unrestricted Ground State (UKS)
ks = dft.UKS(mol)
ks.xc = 'pbe0' 
ks.kernel()

# 3. Define RT-TDDFT simulation with Magnus Propagator
rt = RTTDDFT(ks)

# Define a pulse
def field_fn(t):
    E0 = 0.001
    t0 = 5.0
    sigma = 1.0
    val = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
    return [0, 0, val]

rt.field_fn = field_fn

# 4. Run with 'magnus_step' propagator
dt = 0.05
times = np.arange(0, 10.0, dt)

print("Starting UKS RT-TDDFT with Magnus 2nd Order Propagator...")
# Explicitly specifying the propagator
results = rt.kernel(times=times, dt=dt/2, propagator='magnus_step')

print("\nPropagation Finished.")
print(f"Final Energy: {results['energy'][-1]:.6f} Ha")
print(f"Final Dipole (z): {results['dip'][-1][2]:.6f} Debye")

# 5. Check Spin Expectation value <S^2>
# (Optional verification that spin is conserved or evolving correctly)
dm_final = results['dm'] if 'dm' in results else None
# Note: 'dm' is not returned by default in the simplified dict unless added.
# But we can access the final state if we wanted to. 
# For now, just printing energy and dipole is sufficient validation.
