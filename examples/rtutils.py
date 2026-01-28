import numpy as np
import cupy
from pyscf.tools import cubegen

import numpy as np
class Field:
    '''
    Helper class to generate common electric field functions for RT-TDDFT.
    '''
    @staticmethod
    def gaussian_pulse(E0=0.01, t0=10.0, sigma=1.0, freq=0.0, phase=0.0, polarization='xz'):
        '''
        Creates a Gaussian envelope pulse.
        Args:
            E0 (float): Peak field strength (au).
            t0 (float): Center time (au).
            sigma (float): Width (standard deviation) (au).
            freq (float): Carrier frequency (au). Default 0.0 (DC pulse).
            phase (float): Phase of carrier (radians).
            polarization (str or list): 'x', 'y', 'z' or [dx, dy, dz].
        '''
        dirs = {'x': 0, 'y': 1, 'z': 2}
        if isinstance(polarization, str):
            d_idx = dirs.get(polarization.lower(), 2)
            vec = np.zeros(3)
            if len(polarization) == 1:
                vec[d_idx] = 1.0
            elif len(polarization) == 2:
                dirs_list = list(polarization)
                d_id1 = dirs.get(dirs_list[0])
                d_id2 = dirs.get(dirs_list[1])
                vec[d_id1] = 1.0
                vec[d_id2] = 1.0
        else:
            vec = np.array(polarization) / np.linalg.norm(polarization)

        def _field(t):
            env = E0 * np.exp(-(t - t0)**2 / (2 * sigma**2))
            osc = np.cos(freq * t + phase) if freq > 0 else 1.0
            osc1 = np.sin(freq * t + phase) if freq > 0 else 1.0
            val = env * osc
            if len(polarization) == 1:
                return vec * val
            elif len(polarization) == 2:
                val1 = env * osc1
                tvec = np.zeros(3)
                tvec[d_id1] = val
                tvec[d_id2] = val1
                return vec * tvec
                            
        return _field

    @staticmethod
    def step_function(E0=0.01, t_start=0.0, direction='z'):
        '''Creates a step function field (constant after t_start).'''
        dirs = {'x': 0, 'y': 1, 'z': 2}
        d_idx = dirs.get(direction.lower(), 2)
        
        def _field(t):
            res = [0.0, 0.0, 0.0]
            if t >= t_start:
                res[d_idx] = E0
            return res
        return _field


class RTLogger:
    '''
    Stateful callback for logging RT-TDDFT properties to a file.
    Usage:
        logger = RTLogger('my_output.dat')
        rt.kernel(..., callback=logger)
    '''
    def __init__(self, filename, overwrite=True):
        self.filename = filename
        self.occfilename = "mo_occ.dat"
        mode = 'w' if overwrite else 'a'
        
        # Initialize file with header
        with open(self.filename, mode) as f:
            f.write(f"# Time (au) | Energy (Ha) | Dipole X | Dipole Y | Dipole Z")
            # We don't know yet if it is UKS or if occupations are recorded until the first call
            # So we defer writing the full header or just write generic columns
            f.write("\n")

    def __call__(self, t, dm, results):
        '''The actual callback function.'''
        # Extract latest values
        energy = results['energy'][-1]
        dip = results['dip'][-1] # [dx, dy, dz]
        
        # Format strings
        line = f"{t:12.6f} {energy:18.10f} {dip[0]:14.8f} {dip[1]:14.8f} {dip[2]:14.8f}"
        
        # Add spin-resolved dipoles if available
        if 'dip_alpha' in results and len(results['dip_alpha']) > 0:
            dipa = results['dip_alpha'][-1]
            dipb = results['dip_beta'][-1]
            line += f" {dipa[0]:14.8f} {dipa[1]:14.8f} {dipa[2]:14.8f} {dipb[0]:14.8f} {dipb[1]:14.8f} {dipb[2]:14.8f} {field[0]:14.8f} {field[1]:14.8f} {field[2]:14.8f}"

        
        if 'occ' in results and len(results['occ']) > 0:
            occs = results['occ_alpha'][-1]
            occ_str = " ".join([f"{x:14.8f}" for x in occs])
            with open(self.occfilename,'a') as f:
                f.write(f"{t:12.6f} {occ_str} \n")
        if 'occ_alpha' in results and len(results['occ_alpha']) > 0:
            occs_a = results['occ_alpha'][-1]
            occs_b = results['occ_beta'][-1]
            occ_a_str = " ".join([f"{x:14.8f}" for x in occs_a])
            occ_b_str = " ".join([f"{x:14.8f}" for x in occs_b])
            with open(self.occfilename,'a') as f:
                f.write(f"{t:12.6f} {occ_a_str} {occ_b_str}\n")

        # Add occupations if available (printing HOMO/LUMO region could be complex)
        # For this logger, we skip full occupations to keep the file readable.
        # Use a separate logger for occupations if needed.
        
        with open(self.filename, 'a') as f:
            f.write(line + "\n")


class CubeVisualizer:
    '''
    Callback for generating Cube files at specified intervals.
    '''
    def __init__(self, mol, interval=100, prefix='density'):
        self.mol = mol
        self.interval = interval
        self.prefix = prefix
        self.step = 0

    def __call__(self, t, dm, results):
        self.step += 1
        if self.step % self.interval == 0:
            fname = f"{self.prefix}_t{t:.2f}.cube"
            print(f"Writing cube: {fname}")
            dm_cpu = cupy.asnumpy(dm)
            cubegen.density(self.mol, fname, dm_cpu)


def write_transition_density_cube(td_obj, state_id, filename):
    '''
    Generates a cube file for the transition density of a specific excited state
    from a Linear Response TDDFT calculation.
    '''
    mol = td_obj.mol
    
    # Ensure coefficients are on CPU (NumPy) to match TDDFT amplitudes
    mo_coeff = cupy.asnumpy(td_obj._scf.mo_coeff)
    mo_occ = cupy.asnumpy(td_obj._scf.mo_occ)
    
    # Get X and Y amplitudes for the requested state (PySCF returns NumPy arrays)
    x, y = td_obj.xy[state_id]
    
    print(f"Generating transition density for State {state_id+1}...")
    
    if td_obj._scf.istype('UHF'):
        # UKS Case
        dm_trans = []
        for s in [0, 1]: # alpha, beta
            c = mo_coeff[s]
            occ_idx = mo_occ[s] > 0
            vir_idx = mo_occ[s] == 0
            
            xs = x[s]
            ys = y[s]
            
            # Transition DM in MO basis: X + Y
            t_mo = xs + ys
            
            # Transform to AO basis: C_occ @ T_mo @ C_vir.T
            t_ao = c[:, occ_idx] @ t_mo @ c[:, vir_idx].T
            
            # Symmetrize
            dm_trans_s = t_ao + t_ao.T
            dm_trans.append(dm_trans_s)
            
        # Total transition density
        dm_trans_tot = np.array(dm_trans[0] + dm_trans[1])
        
    else:
        # RKS Case
        occ_idx = mo_occ > 0
        vir_idx = mo_occ == 0
        c = mo_coeff
        
        t_mo = x + y
        t_ao = c[:, occ_idx] @ t_mo @ c[:, vir_idx].T
        
        dm_trans_tot = t_ao + t_ao.T

    # Generate Cube
    cubegen.density(mol, filename, dm_trans_tot)
    print(f"Written to {filename}")
