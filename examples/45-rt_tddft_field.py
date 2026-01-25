import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT
import matplotlib.pyplot as plt

# 1. Setup Molecule and Ground State
mol = gto.M(
    atom='H 0 0 0; H 0 0 0.74',
    basis='def2-svp',
    verbose=3
)

ks = dft.RKS(mol)
ks.xc = 'pbe0' 
ks.kernel()

# 2. Define RT-TDDFT simulation
rt = RTTDDFT(ks)

# Delta pulse at t=0.01 au
# Simulated by a narrow Gaussian
E0 = 0.001 
t0 = 0.01
sigma = 0.005

def field_fn(t):
    val = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
    return [0, 0, val] # Pulse in z-direction

rt.field_fn = field_fn

# 3. Run propagation for a longer time
dt = 0.05
total_time = 20.0
times = np.arange(0, total_time, dt)
results = rt.kernel(times=times, dt=dt/2)

# 4. Process results
t = np.array(results['times'])
dip_z = np.array(results['dip'])[:,2]
# Center dipole
dip_z -= dip_z[0]

# Fourier Transform to get spectrum
# Padding for better resolution
n_fft = 4096
freq = np.fft.rfftfreq(n_fft, d=dt) * 2 * np.pi # Energy in au
dip_f = np.fft.rfft(dip_z, n=n_fft)

# S(w) ~ Im(alpha(w)) ~ Im(mu(w) / E(w))
# E(w) is the FT of the pulse
pulse = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
E_f = np.fft.rfft(pulse, n=n_fft)

# Absorption spectrum
S_w = np.imag(dip_f / E_f)

plt.figure(figsize=(8,5))
plt.plot(freq * 27.2114, S_w) # Convert au to eV
plt.xlim(0, 30) # Typical UV-Vis range
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption (arb. units)')
plt.title('RT-TDDFT Absorption Spectrum (H2)')
plt.savefig('h2_spectrum.png')
print("Spectrum saved to h2_spectrum.png")
