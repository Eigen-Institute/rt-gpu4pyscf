import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.hessian.rhf import Hessian, _e_hcore_generator
from gpu4pyscf.lib.cupy_helper import contract

def check_hcore_doubling():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); dm = mf.make_rdm1()
    dr = 0.0001; coords = mol.atom_coords()
    
    # 1. Analytical generator
    h_obj = Hessian(mf)
    de_hcore_gen = _e_hcore_generator(h_obj, cp.asarray(dm))
    h_anal = float(de_hcore_gen(0,0)[2,2])
    print(f"Analytical de_hcore [0,0,2,2]: {h_anal:.8f}")
    
    # 2. FD of Hcore Force
    def get_hcore_force(c):
        m = mol.copy().set_geom_(c, unit='Bohr'); m.build()
        # Full gradient of H-core part
        h1 = -cp.asarray(m.intor('int1e_ipkin', comp=3) + m.intor('int1e_ipnuc', comp=3))
        # Total force on atom 0
        p0, p1 = m.aoslice_by_atom()[0][2:]
        g = cp.zeros(3)
        g += contract('xpq,pq->x', h1[:, p0:p1], cp.asarray(dm)[p0:p1]) * 2.0
        with m.with_rinv_at_nucleus(0):
            vrinv = cp.asarray(m.intor('int1e_iprinv', comp=3)) * -m.atom_charge(0)
        g += contract('xpq,pq->x', vrinv, cp.asarray(dm))
        return g

    f_p = get_hcore_force(coords.copy() + np.array([[0,0,dr], [0,0,0]]))
    f_m = get_hcore_force(coords.copy() + np.array([[0,0,-dr], [0,0,0]]))
    h_fd = (f_p[2] - f_m[2]) / (2 * dr)
    print(f"FD d/dz (g_hcore) [0,2]:       {h_fd:.8f}")
    print(f"Ratio: {h_anal / h_fd:.4f}x")

check_hcore_doubling()
