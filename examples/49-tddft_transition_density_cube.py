import numpy as np
import cupy
from pyscf import gto
from pyscf.tools import cubegen
from gpu4pyscf import dft, tdscf

# 1. Setup Molecule and Ground State
mol = gto.M(
    atom='H 0 0 0; F 0 0 0.92',
    basis='def2-svp',
    verbose=3
)

ks = dft.RKS(mol)
ks.xc = 'pbe0'
ks.kernel()

# 2. Run TDDFT
td = tdscf.uks.TDDFT(ks) if ks.istype('UHF') else tdscf.rks.TDDFT(ks)
td.nstates = 5
td.kernel()

def write_transition_density_cube(td_obj, state_id, filename):
    '''
    Manual construction of transition density matrix and cube generation.
    '''
    mol = td_obj.mol
    
    # Ensure coefficients are on CPU (NumPy) to match TDDFT amplitudes
    mo_coeff = cupy.asnumpy(td_obj._scf.mo_coeff)
    mo_occ = cupy.asnumpy(td_obj._scf.mo_occ)
    
    # Get X and Y amplitudes for the requested state
    # PySCF stores amplitudes as (X, Y) tuples in td.xy
    x, y = td_obj.xy[state_id]
    
    if td_obj._scf.istype('UHF'):
        # UKS Case
        # x is ((nocc_a, nvir_a), (nocc_b, nvir_b))
        # mo_coeff is (2, nao, nmo)
        # mo_occ is (2, nmo)
        
        dm_trans = []
        for s in [0, 1]: # alpha, beta
            c = mo_coeff[s]
            occ_idx = mo_occ[s] > 0
            vir_idx = mo_occ[s] == 0
            
            xs = x[s]
            ys = y[s]
            
            # Transition DM in MO basis: X + Y
            # (Note: For transition density \rho_0n = \psi_0 \psi_n, 
            # we use X+Y for the density part)
            t_mo = xs + ys
            
            # Transform to AO basis: C_occ @ T_mo @ C_vir.T
            t_ao = c[:, occ_idx] @ t_mo @ c[:, vir_idx].T
            
            # Transition density is \rho = \sum \phi_i \phi_a (Xia + Yia)
            # We symmetrize if we want a real-valued "density-like" plot
            dm_trans_s = t_ao + t_ao.T
            dm_trans.append(dm_trans_s)
            
        # Total transition density (alpha + beta)
        dm_trans_tot = np.array(dm_trans[0] + dm_trans[1])
        
    else:
        # RKS Case
        # x, y are (nocc, nvir)
        occ_idx = mo_occ > 0
        vir_idx = mo_occ == 0
        c = mo_coeff
        
        t_mo = x + y
        t_ao = c[:, occ_idx] @ t_mo @ c[:, vir_idx].T
        
        # Symmetrize for visualization
        dm_trans_tot = t_ao + t_ao.T

    # Generate Cube
    print(f"Writing transition density for state {state_id} to {filename}")
    cubegen.density(mol, filename, dm_trans_tot)

# 3. Generate Cube for the first excited state
# State indices are 0-based
write_transition_density_cube(td, 0, 'transition_density_s1.cube')

print("Finished.")
