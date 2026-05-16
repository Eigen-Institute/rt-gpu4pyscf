import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates
from gpu4pyscf.hessian import rhf as rhf_hess

def check_asymmetry():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run(); td = TDA(mf); td.nstates = 1; td.kernel()
    
    # We need to reach into the analytical path logic
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in td.xy[0]])
    td_grad_obj = tdrhf_grad.Gradients(td); z1 = solve_z_vector(td_grad_obj, x_y); h_obj = Hessian(td); inter = make_intermediates(h_obj, x_y, z1)
    
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    h_ip2 = _e_hcore_generator(h_obj, inter['P_I_prime'])
    print("ip2 hcore [0,2,1,2]:", h_ip2(0, 1)[2,2])
    print("ip2 hcore [1,2,0,2]:", h_ip2(1, 0)[2,2])
    
    # Check MO response mo1
    mo_coeff = cp.asarray(mf.mo_coeff); mo_occ = cp.asarray(mf.mo_occ); mo_energy = cp.asarray(mf.mo_energy)
    gs_hess = rhf_hess.Hessian(mf); h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ); fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ)
    mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)
    print("mo1 norm atom 0:", float(cp.linalg.norm(mo1[0])))
    print("mo1 norm atom 1:", float(cp.linalg.norm(mo1[1])))

check_asymmetry()
