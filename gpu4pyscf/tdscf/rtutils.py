import numpy as np
import cupy
from pyscf.tools import cubegen
import re, json

#--------------------------------------------------------------
#
# Define MultiCallback wrapper
#
#--------------------------------------------------------------

class MultiCallback:
    def __init__(self, callbacks):
        self.callbacks = callbacks
    def __call__(self, t, dm, results):
        for cb in self.callbacks:
            cb(t, dm, results)

#--------------------------------------------------------------
#
# Parse JSON input. Returns array of dicts. 
#   Array elements: 
#   [0] - theory_data
#   [1] - mol_data
#   [2] - properties
#   [3] - rttddft_data
#   [4] - field_data
#   [5] - viz_data
#   [6] - names
#
#--------------------------------------------------------------

def parseInputJson(filename):
    opts = []
    with open(filename, 'r') as f:
        input_data = json.load(f)

    print('')
    print("*"*100)
    print("Reading calculation input from:"+filename)
    print("")
    print(input_data)
    print("*"*100)

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
    
    opts.append(theory_data)
    opts.append(mol_data)
    opts.append(properties)
    opts.append(rttddft_data)
    opts.append(field_data)
    opts.append(viz_data)
    opts.append({"name":calcName})
        
    return opts

def getTargetStateFreq(params,target_state=1):
    tddft_file = params[3]['tddft file']
    freq = params[4]['freq']
    polarization = params[4]['polarization']
    if tddft_file:
        print(f"\nParsing TDDFT output from {tddft_file} for target state {target_state}...")
        parsed_states = parse_tddft_output(tddft_file)
        if target_state in parsed_states:
            state_info = parsed_states[target_state]
            
            # Auto-set Frequency (eV -> Ha) unless overridden in JSON
            if 'freq' not in params[4] or type(freq) is str:
                freq_ev = state_info['energy_ev']
                freq = freq_ev / 27.211386
                print(f"  Auto-setting freq to {freq:.6f} Ha ({freq_ev} eV)")
            
            # Auto-set Polarization unless overridden
            if 'polarization' not in params[4] or params[4]['polarization'] in ['target','auto']:
                dip = state_info.get('dipole', [0,0,0])
                abs_dip = [abs(d) for d in dip]
                max_idx = abs_dip.index(max(abs_dip))
                polarization = ['x', 'y', 'z'][max_idx]
                print(f"  Auto-setting polarization to '{polarization}' (Dipole: {dip})")
        else:
            print(f"  Warning: State {target_state} not found in {tddft_file}.")
    else:
        print("  Warning: 'target' specified but 'tddft file' is missing.")

    return freq, polarization


#--------------------------------------------------------------
#
#   def parse_tddft_output: Parse tddft data from provided
#                           output file. 
#                           Returns: dict states[int][float 'dipole']
#
#--------------------------------------------------------------

def parse_tddft_output(filename):
    states = {}
    with open(filename, 'r') as f:
        content = f.read()
    
    # Parse Energies
    # Excited State   1:      0.41936 eV
    energy_pattern = re.compile(r"Excited State\s+(\d+):\s+([\d\.]+)\s+eV")
    for match in energy_pattern.finditer(content):
        idx = int(match.group(1))
        energy = float(match.group(2))
        if idx not in states: states[idx] = {}
        states[idx]['energy_ev'] = energy

    # Parse Dipoles
    # state          X           Y           Z
    #   1        -0.3062      0.1006     -0.2375
    dipole_section = re.search(r"\*\* Transition electric dipole moments \(AU\) \*\*(.*?)(\*\*|$)", content, re.DOTALL)
    if dipole_section:
        dip_lines = dipole_section.group(1).strip().split('\n')
        for line in dip_lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0].isdigit():
                idx = int(parts[0])
                try:
                    dip = [float(parts[1]), float(parts[2]), float(parts[3])]
                    if idx in states:
                        states[idx]['dipole'] = dip
                except ValueError:
                    continue
    return states

#--------------------------------------------------------------
#
#   class Field - electric field helper
#
#--------------------------------------------------------------
class Field:
    
    '''
    Print field parameters
    '''
    def printField(fieldType=None,E0=None,t0=None,sigma=None,freq=None,phase=0,polarization=None, hand=None) :
        print("")
        print(f'****    External Field Parameter:      ****')
        if E0 is not None:
            print(f'    Maximum Amplitude, E0 =     {E0}')
        if freq > 0:
            print(f'    Frequency (Ha) =            {freq:.6f}')
        if fieldType is not None:
            print(f'    Envelope Type =             {fieldType}')
        if t0 is not None:
            print(f'    Temporal Center (au) =      {t0}')
        if sigma is not None:
            print(f'    Temporal Width(au) =        {sigma}')
        if phase is not None:
            print(f'    Initial Phase (radians) =   {phase}')
        if polarization is not None:
            print(f'    Polarization Direction =    {polarization}')
        if hand is not None:
            print(f'    Polarization Chirality =    {hand}')
        print("")


    '''
    Helper class to generate common electric field functions for RT-TDDFT.
    '''
    @staticmethod
    def gaussian_pulse(E0=0.01, t0=10.0, sigma=1.0, freq=0.0, phase=0.0, polarization='z',hand=None):
        '''
        Creates a Gaussian envelope pulse.
        Args:
            E0 (float): Peak field strength (au).
            t0 (float): Center time (au).
            sigma (float): Width (standard deviation) (au).
            freq (float): Carrier frequency (au). Default 0.0 (DC pulse).
            phase (float): Phase of carrier (radians).
            polarization (str or list): 'x', 'y', 'z', 'xy', 'yz', 'xz'.
        '''
        dirs = {'x': 0, 'y': 1, 'z': 2}
        handMap = {'right':1.0,'left':-1.0} #CTC confirm this definition
        if isinstance(polarization, str):
            d_idx = dirs.get(polarization.lower(), 2)
            vec = np.zeros(3)
            if len(polarization) == 1:
                vec[d_idx] = 1.0
            # Circular polarization, polarization={"xy","xz","yz"}
            elif len(polarization) == 2:
                dirs_list = list(polarization)
                d_id1 = dirs.get(dirs_list[0])
                d_id2 = dirs.get(dirs_list[1])
                vec[d_id1] = 1.0
                vec[d_id2] = 1.0*handMap[hand.lower()]
        else:
            vec = np.array(polarization) / np.linalg.norm(polarization)

        def _field(t):
            env = E0 * np.exp(-(t - t0)**2 / (2 * sigma**2))
            osc = np.sin(freq * t + phase) if freq > 0 else 1.0
            osc1 = np.cos(freq * t + phase) if freq > 0 else 1.0
            val = env * osc
            # Linear polarization
            if len(polarization) == 1:
                return vec * val
            # Circular polarization
            elif len(polarization) == 2:
                val1 = env * osc1
                tvec = np.zeros(3)
                tvec[d_id1] = val
                tvec[d_id2] = val1
                return vec * tvec
        return _field

    @staticmethod
    def cw_function(E0=0.001, freq=0.01, phase=0.0, polarization='z', hand=None):
        '''Creates a CW sinusoidal field '''
        dirs = {'x': 0, 'y': 1, 'z': 2}
        d_idx = dirs.get(polarization.lower(), 2)
        handMap = {'right':1.0,'left':-1.0} #CTC confirm this definition
        if isinstance(polarization, str):
            d_idx = dirs.get(polarization.lower(), 2)
            vec = np.zeros(3)
            if len(polarization) == 1:
                vec[d_idx] = 1.0
            # Circular polarization, polarization={"xy","xz","yz"}
            elif len(polarization) == 2:
                dirs_list = list(polarization)
                d_id1 = dirs.get(dirs_list[0])
                d_id2 = dirs.get(dirs_list[1])
                vec[d_id1] = 1.0
                vec[d_id2] = 1.0*handMap[hand.lower()]
        else:
            vec = np.array(polarization) / np.linalg.norm(polarization)

        def _field(t):
            env = E0
            osc = np.sin(freq * t + phase) if freq > 0 else 1.0
            osc1 = np.cos(freq * t + phase) if freq > 0 else 1.0
            val = env * osc
            # Linear polarization
            if len(polarization) == 1:
                return vec * val
            # Circular polarization
            elif len(polarization) == 2:
                val1 = env * osc1
                tvec = np.zeros(3)
                tvec[d_id1] = val
                tvec[d_id2] = val1
                return vec * tvec
        return _field

       
    @staticmethod
    def step_function(E0=0.01, t_start=0.0, polarization='z'):
        '''Creates a step function field (constant after t_start).'''
        dirs = {'x': 0, 'y': 1, 'z': 2}
        d_idx = dirs.get(polarization.lower(), 2)
        
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
        logger = RTLogger('my_output.dat', field_fn=rt.field_fn) # Original usage
        rt.kernel(..., callback=logger)
    '''
    def __init__(self, filename, occfilename, field_fn=None, isCs=True,overwrite=True):
        self.filename = filename
        self.field_fn = field_fn
        self.occfilename = occfilename
        self.isCs=isCs
        mode = 'w' if overwrite else 'a'
        
        # Initialize file with header
        header = "# Time (au) | Energy Total (Ha)  | E_nuc (Ha) | E_coul (Ha) | E_core (Ha) | E_xc (Ha) | E_field (Ha) | Field X | Field Y | Field Z "
        if self.field_fn is not None:
            header += " | mu X | mu Y | mu Z"
        if not isCs:
            header += " | mu_alpha X | mu_alpha Y | mu_alpha Z | mu_beta X | mu_beta Y | mu_beta Z "
        
        with open(self.filename, mode) as f:
            f.write(header + "\n")

    def __call__(self, t, dm, results):
        '''The actual callback function.'''
        # Extract latest values
        energy = results['energy'][-1]
        energy_nuc = results['energy_nuc'][-1]
        energy_core = results['energy_core'][-1]
        energy_coul = results['energy_coul'][-1]
        energy_xc = results['energy_xc'][-1]
        energy_field = results['energy_field'][-1]

        dip = results['dip'][-1] # [dx, dy, dz] 
                # Add field if available
        if self.field_fn is not None:
            efield = self.field_fn(t)
            line = f"{t:12.6f} {energy:18.10f} {energy_nuc:18.10f} {energy_coul:18.10f} {energy_core:18.10f} {energy_xc:18.10f} {energy_field:18.10f} {efield[0]:14.8f} {efield[1]:14.8f} {efield[2]:14.8f}"

        # Format strings
        #if self.isCs:
        line += f" {dip[0]:14.8f} {dip[1]:14.8f} {dip[2]:14.8f}"
        
        # Add spin-resolved dipoles if available
        if 'dip_alpha' in results and len(results['dip_alpha']) > 0:
            dipa = results['dip_alpha'][-1]
            dipb = results['dip_beta'][-1]
            line += f" {dipa[0]:14.8f} {dipa[1]:14.8f} {dipa[2]:14.8f} {dipb[0]:14.8f} {dipb[1]:14.8f} {dipb[2]:14.8f}"
        
        # MO Occupation numbers
        if 'occ' in results and self.isCs > 0:
            occs = results['occ'][-1]
            occ_str = " ".join([f"{x:14.8f}" for x in occs])
            with open(self.occfilename,'a') as f:
                f.write(f"{t:12.6f} {occ_str} \n")
        if 'occ_alpha' in results and not self.isCs:
            occs_a = results['occ_alpha'][-1]
            occs_b = results['occ_beta'][-1]
            occ_a_str = " ".join([f"{x:14.8f}" for x in occs_a])
            occ_b_str = " ".join([f"{x:14.8f}" for x in occs_b])
            with open(self.occfilename,'a') as f:
                f.write(f"{t:12.6f} {occ_a_str} {occ_b_str}\n")
        
        with open(self.filename, 'a') as f:
            f.write(line + "\n")


class CubeVisualizer:
    '''
    Callback for generating Cube files at specified intervals.
    '''
    def __init__(self, mol, interval=100, prefix='density', margin=4.0, treference=None, spin='total'):
        self.mol = mol
        self.interval = interval
        self.prefix = prefix
        self.step = 0
        self.margin = margin
        self.treference = treference
        self.dm_ref = None
        self.spin = spin.lower()

    def __call__(self, t, dm, results):
        # Extract the desired spin component
        if dm.ndim == 3: # UKS
            if self.spin == 'alpha':
                dm_selected = dm[0]
            elif self.spin == 'beta':
                dm_selected = dm[1]
            elif self.spin == 'both':
                dm_selected = dm[0]
                dm_selected2 = dm[1]
            else: # total
                dm_selected = dm[0] + dm[1]
        else: # RKS
            dm_selected = dm

        # Capture reference density if at the target time
        if self.treference is not None and self.dm_ref is None:
            if abs(t - self.treference) < 1e-5:
                print(f"CubeVisualizer: Capturing {self.spin} reference density at t={t}")
                self.dm_ref = dm_selected.copy()
                if self.spin == "both":
                    self.dm_refBeta = dm_selected2.copy()

        self.step += 1
        if self.step % self.interval == 0:
            if self.dm_ref is not None:
                # Write Difference Density
                if not self.spin == "both":
                    fname = f"density_subgs.{self.prefix}.{self.spin}.{self.step:07d}.cube"
                    print(f"Writing {self.spin} difference cube: {fname}")
                else:
                    fname = f"density_subgs.{self.prefix}.alpha.{self.step:07d}.cube"
                    print(f"Writing alpha difference cube: {fname}")
                    fname_beta = f"density_subgs.{self.prefix}.beta.{self.step:07d}.cube"
                    print(f"Writing beta difference cube: {fname_beta}")
                    dm_to_write2 = dm_selected2 - self.dm_refBeta
                    dm_cpu = cupy.asnumpy(dm_to_write2)
                    cubegen.density(self.mol, fname_beta, dm_cpu, margin=self.margin)
                
                dm_to_write = dm_selected - self.dm_ref

            else:
                # Write Full Density
                fname = f"{self.prefix}_{self.spin}_t{t:.2f}.cube"
                print(f"Writing {self.spin} cube: {fname}")
                dm_to_write = dm_selected

            dm_cpu = cupy.asnumpy(dm_to_write)
            cubegen.density(self.mol, fname, dm_cpu, margin=self.margin)



def write_transition_density_cube(td_obj, state_id, filename, margin):
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
    cubegen.density(mol, filename, dm_trans_tot,margin=4.0)
    print(f"Written to {filename}")
