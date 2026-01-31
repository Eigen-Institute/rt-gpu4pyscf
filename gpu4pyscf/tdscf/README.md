# Real-Time Time-Dependent Density Functional Theory (RT-TDDFT)

The `rt_tddft` module in `gpu4pyscf` provides a GPU-accelerated implementation of Real-Time TDDFT. It allows for the time-propagation of the electron density matrix under the influence of an optional external electric field (e.g., a laser pulse).

## Features

*   **GPU Acceleration:** All heavy tensor contractions (Fock build, matrix multiplication, exponentiation) are performed on the GPU using CuPy.
*   **Unitary Propagators:** Uses the **Magnus 2nd Order** propagator (equivalent to MMUT), a stable, unitary, second-order predictor-corrector scheme.
*   **Hamiltonian:** Supports standard DFT functionals (LDA, GGA) and **Hybrid Functionals** (e.g., B3LYP, PBE0) with exact exchange.
*   **Spin Support:** Fully supports both **Restricted (RKS)** and **Unrestricted (UKS)** systems for open-shell dynamics.
*   **Basis Orthogonalization:** Automatically handles non-orthogonal basis sets using Lowdin symmetric orthogonalization ($S^{-1/2}$).
*   **Numerical Stability:** Includes options for **McWeeny Purification** to maintain density matrix idempotency.
*   **External Fields:** Flexible interface for defining arbitrary time-dependent electric field vectors.
*   **Real-Time Analysis:** Callback interface for extracting properties (dipoles, energy, etc.) on-the-fly.
*   **Restart System:** Supports periodic checkpointing and resuming from saved states.

## Usage

### 1. Basic Setup

First, perform a ground-state DFT calculation using `gpu4pyscf.dft.RKS` or `gpu4pyscf.dft.UKS`.

```python
from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

# 1. Define Molecule
mol = gto.M(atom='H 0 0 0; F 0 0 0.92', basis='def2-svp')

# 2. Ground State Calculation
ks = dft.RKS(mol) # or dft.UKS(mol)
ks.xc = 'pbe0'
ks.kernel()
```

### 2. Initialize Propagator

Create an instance of `RTTDDFT` passing the converged Kohn-Sham object.

```python
rt = RTTDDFT(ks)

# Optional: Set purification interval (e.g., every 10 steps)
rt.purify_interval = 10

# Optional: Set checkpointing for long runs
rt.checkpoint_interval = 100
rt.checkpoint_file = 'rt_restart.npz'
```

### 3. Restarting a Calculation

If a simulation was interrupted, you can resume from the last saved checkpoint.

```python
rt = RTTDDFT(ks)
rt.restart_from = 'rt_restart.npz'
rt.kernel(times=times)
```

### 4. Define External Field (Optional)

You can define a time-dependent electric field function `field_fn(t)` that returns a vector `[Ex, Ey, Ez]` (in atomic units).

```python
import numpy as np

# Example: Gaussian Pulse in Z-direction
def laser_pulse(t):
    E0 = 0.01      # Field strength (au)
    t0 = 10.0      # Center time (au)
    sigma = 1.0    # Width (au)
    omega = 0.5    # Frequency (au)
    
    env = E0 * np.exp(-(t-t0)**2 / (2*sigma**2))
    field_z = env * np.cos(omega * t)
    
    return [0, 0, field_z]

rt.field_fn = laser_pulse
```

### 5. Run Propagation

Use the `kernel` method to propagate the density.

*   `times`: Array of time points to stop and record data.
*   `dt`: Integration time step (au). Usually 0.01-0.02 au is recommended.
*   `propagator`: Method name (default `'magnus_step'`).
*   `callback`: (Optional) Function called at every recording step.

```python
# Propagate for 20 atomic units
times = np.arange(0, 20, 0.1) 
dt = 0.02

results = rt.kernel(times=times, dt=dt)

print("Final Energy:", results['energy'][-1])
print("Final Dipole:", results['dip'][-1])
```

### 6. Visualization and Analysis

#### Real-Time Properties (Callback)
Use a callback to save data (e.g., dipoles, occupations) to a file or generate Cube files.

```python
from pyscf.tools import cubegen
import cupy

def log_step(t, dm, res):
    # Log Dipole
    dip = res['dip'][-1]
    print(f"t={t:.2f} Dipole={dip}")
    
    # Generate Cube File every 100 steps
    step = len(res['times'])
    if step % 100 == 0:
        dm_cpu = cupy.asnumpy(dm)
        cubegen.density(rt.mol, f'dens_t{t:.2f}.cube', dm_cpu)

rt.kernel(times=times, dt=dt, callback=log_step)
```

#### Spin-Resolved Properties (UKS)
For open-shell systems, you can access spin-specific properties.

```python
rt.mu_spin = 'alpha' # 'total', 'alpha', 'beta'
results = rt.kernel(times=times, dt=dt)

# Access all components regardless of mu_spin setting
dip_a = results['dip_alpha']
dip_b = results['dip_beta']
```

#### Transition Densities (Linear Response TDDFT)
You can also generate transition density cube files from standard Linear Response TDDFT calculations.

See **`examples/49-tddft_transition_density_cube.py`** for a complete example.

## API Reference

### `class RTTDDFT(ks)`

**Parameters:**
*   `ks`: A converged `gpu4pyscf.dft.RKS` or `UKS` object.

**Attributes:**
*   `field_fn`: Callable `f(t) -> [x, y, z]`. Default is `None`.
*   `purify_interval`: `int`. Step interval for McWeeny purification. Default is `None`.
*   `mu_spin`: `str`. For UKS, controls the main `results['dip']` output. Options: `'total'` (default), `'alpha'`, `'beta'`.
*   `record_occ`: `bool`. If `True`, tracks MO occupation numbers in `results['occ']` (RKS) or `results['occ_alpha']`/`results['occ_beta']` (UKS).
*   `checkpoint_interval`: `int`. Save density matrix every N steps.
*   `checkpoint_file`: `str`. File to save checkpoint data.
*   `restart_from`: `str`. File to load for restarting a calculation.

**Methods:**

#### `kernel(times, dm0=None, dt=0.02, propagator='magnus_step', callback=None)`
Runs the time propagation.

*   `times` (ndarray): Time points (au) at which to record results.
*   `dm0` (ndarray, optional): Initial density matrix. Defaults to `ks.make_rdm1()`.
*   `dt` (float): Integration time step (au).
*   `propagator` (str): `'magnus_step'`.
*   `callback` (callable, optional): `callback(t, dm, results)`.

**Returns:** `dict` containing lists:
*   `'times'`: Time points.
*   `'energy'`: Total energy (au).
*   `'dip'`: Dipole moment vector (Debye).
*   `'dm'`: Final density matrix (NumPy).
*   `'dip_alpha'`, `'dip_beta'`: (UKS only) Spin-resolved dipoles.
*   `'occ'`, `'occ_alpha'`, `'occ_beta'`: (If `record_occ=True`) MO occupation numbers.

## Examples

Ready-to-run scripts in the `examples/` directory:
*   `examples/44-rt_tddft.py`: Basic RKS propagation.
*   **`examples/45-rt_tddft_field.py`**: Linear response (absorption spectrum) using a delta pulse.
*   **`examples/46-rt_tddft_realtime.py`**: Logging data to a file in real-time.
*   **`examples/47-rt_tddft_magnus_uks.py`**: Open-shell UKS propagation.
*   **`examples/48-rt_tddft_cubegen.py`**: Generating density cube files during trajectory.
*   **`examples/49-tddft_transition_density_cube.py`**: Linear Response TDDFT transition densities.