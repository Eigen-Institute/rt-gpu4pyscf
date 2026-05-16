import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.hessian.rhf import Hessian

def check_gs_parity():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    h_obj = Hessian(mf)
    h_anal = h_obj.kernel()
    
    # FD
    dr = 0.001; coords = mol.atom_coords(); h_fd = np.zeros((2,3,2,3))
    for ia in range(2):
        for ix in range(3):
            c_p = coords.copy(); c_p[ia,ix] += dr
            g_p = RHF(mol.copy().set_geom_(c_p, unit='Bohr').build()).run().nuc_grad_method().kernel()
            c_m = coords.copy(); c_m[ia,ix] -= dr
            g_m = RHF(mol.copy().set_geom_(c_m, unit='Bohr').build()).run().nuc_grad_method().kernel()
            h_fd[:,:,ia,ix] = (g_p - g_m) / (2 * dr)
            
    print("GS Hessian [0,2,0,2] (Stretch):")
    print(f"  FD:    {h_fd[0,2,0,2]:.8f}")
    print(f"  Anal:  {h_anal[0,0,2,2]:.8f}")
    print(f"  Ratio: {h_anal[0,0,2,2]/h_fd[0,2,0,2]:.4f}x")
    
    print("\nGS Hessian [0,0,0,0] (Rotation):")
    print(f"  FD:    {h_fd[0,0,0,0]:.8f}")
    print(f"  Anal:  {h_anal[0,0,0,0]:.8f}")
    print(f"  Ratio: {h_anal[0,0,0,0]/h_fd[0,0,0,0]:.4f}x")

check_gs_parity()
