import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.hessian.rhf import _partial_ejk_ip2

def check_k_scaling():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); dm = mf.make_rdm1()
    vhfopt = mf._opt_gpu.get(mol.omega)
    
    # 1. Analytical K part only
    ejk = _partial_ejk_ip2(mol, cp.asarray(dm), vhfopt, j_factor=0.0, k_factor=1.0)
    h_anal_k = float(ejk[0,0,2,2])
    print(f"Analytical K ip2 [0,0,2,2]: {h_anal_k:.8f}")
    
    # 2. FD of K Gradient
    dr = 0.0001; coords = mol.atom_coords()
    def get_k_force(c):
        m = mol.copy().set_geom_(c, unit='Bohr'); m.build()
        from gpu4pyscf.hessian.rhf import _get_jk_ip1
        vj, vk = _get_jk_ip1(m, cp.asarray(dm))
        # vk is G^x_K(P). Trace is Tr[P G^x_K(P)].
        return float(cp.trace(cp.asarray(dm) @ vk[0,2]))

    f_p = get_k_force(coords.copy() + np.array([[0,0,dr], [0,0,0]]))
    f_m = get_k_force(coords.copy() + np.array([[0,0,-dr], [0,0,0]]))
    h_fd_k = (f_p - f_m) / (2 * dr)
    print(f"FD d/dz (Tr[P K^z(P)]) [0,2]: {h_fd_k:.8f}")
    print(f"Ratio: {h_anal_k / h_fd_k:.4f}x")

check_k_scaling()
