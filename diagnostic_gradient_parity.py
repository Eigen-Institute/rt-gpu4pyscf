import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def get_h1ao_x(mol):
    natm = mol.natm
    nao = mol.nao
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        from gpu4pyscf.df import int3c2e
        coords = mol.atom_coords()
        charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
        fakemol = gto.fakemol_for_charges(coords)
        intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
        intopt.build(1e-14, diag_block_with_triu=True, aosym=False)
        eye_ao = cp.eye(nao)
        eye_ao_sorted = intopt.sort_orbitals(eye_ao, axis=[0])
        dh1e_ao = cp.zeros([3, nao, nao])
        for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
            if k0 <= atm_id < k1:
                dh1e_ao += cp.einsum('xkji,io->xjo', int3c_blk[:,:,:,atm_id-k0:atm_id-k0+1], eye_ao_sorted[i0:i1]).sum(axis=2) # This is getting complex, I'll use the loop from tdrhf
        pass
    # I'll just use the working loop from tdrhf for h1ao_x
    return None

def run_grad_parity():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    state = 0
    x_y_orig = td.xy[state]
    # Use unscaled amplitudes for first check to match grad engine
    x_y = x_y_orig
    
    td_grad_obj = tdrhf_grad.Gradients(td)
    # Reference
    g_ref = td_grad_obj.grad_elec(x_y_orig)
    
    # Components from grad engine
    nmo = mf.mo_coeff.shape[1]
    nocc = int(mf.mo_occ.sum() // 2)
    nvir = nmo - nocc
    x, y = x_y
    xpy = cp.asarray((x + y).reshape(nocc, nvir).T)
    xmy = cp.asarray((x - y).reshape(nocc, nvir).T)
    orbv = cp.asarray(mf.mo_coeff[:, nocc:])
    orbo = cp.asarray(mf.mo_coeff[:, :nocc])
    
    dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T)
    doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
    dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) * 2.0
    dmzoo += cp.dot(orbv, cp.dot(dvv, orbv.T)) * 2.0
    
    # 1. H-core part
    mf_grad = mf.nuc_grad_method()
    h1 = cp.asarray(mf_grad.get_hcore(mol))
    g_h_basis = cp.asarray([cp.trace(dmzoo @ h1[i]) for i in range(3)])
    
    from gpu4pyscf.df import int3c2e
    dh1e = int3c2e.get_dh1e(mol, dmzoo)
    g_h_oper = dh1e[0]
    
    print(f"H-core part (atom 0, dir Z): {float(g_h_basis[2] + g_h_oper[2]):.6f}")
    
    # 2. Overlap part
    # im0 in grad_elec
    # I'll just check if dmzoo matches dmz1doo in grad_elec
    pass

run_grad_parity()
