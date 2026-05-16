import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def get_energy(mol):
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    return td.e[0]

def run_grad_check():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    # 1. FD Gradient
    dr = 0.001
    coords = mol.atom_coords()
    coords_p = coords.copy(); coords_p[0,2] += dr
    e_p = get_energy(mol.copy().set_geom_(coords_p, unit='Bohr'))
    coords_m = coords.copy(); coords_m[0,2] -= dr
    e_m = get_energy(mol.copy().set_geom_(coords_m, unit='Bohr'))
    g_fd = (e_p - e_m) / (2 * dr)
    print(f"FD Gradient (excitation): {g_fd:.8f}")
    
    # 2. Analytical Intermediates
    x_y_orig = td.xy[0]
    # Scaling factor check:
    # If we use unscaled amplitudes, do we match g_fd?
    td_grad_obj = tdrhf_grad.Gradients(td)
    g_anal = td_grad_obj.grad_elec(x_y_orig) # This computes d(E_excited)/dR
    
    mf_grad = mf.nuc_grad_method()
    g_gs = mf_grad.grad_elec() # d(E_ground)/dR
    
    g_diff = g_anal[0,2] - g_gs[0,2]
    print(f"Grad Engine (diff):      {g_diff:.8f}")
    print(f"Ratio Anal/FD:           {g_diff / g_fd:.4f}x")

run_grad_check()
