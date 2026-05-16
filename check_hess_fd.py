import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.grad import tdrhf as tdrhf_grad

def get_grad(mol):
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    td_grad_obj = tdrhf_grad.Gradients(td)
    g_exc = td_grad_obj.grad_elec(td.xy[0])
    g_gs = mf.nuc_grad_method().grad_elec()
    return g_exc - g_gs

def run_hess_fd():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf); td.kernel()
    
    dr = 0.001
    coords = mol.atom_coords()
    coords_p = coords.copy(); coords_p[0,2] += dr
    coords_m = coords.copy(); coords_m[0,2] -= dr
    
    x_y_ref = td.xy[0]
    def get_terms(mol_):
        mf = RHF(mol_).run()
        nocc = int(mf.mo_occ.sum()//2)
        nvir = mf.mo_energy.size - nocc
        x, y = [cp.asarray(v) for v in x_y_ref]
        orbv = cp.asarray(mf.mo_coeff[:, nocc:])
        orbo = cp.asarray(mf.mo_coeff[:, :nocc])
        
        xpy = (x + y).reshape(nocc, nvir).T
        xmy = (x - y).reshape(nocc, nvir).T
        dvv = cp.dot(xpy, xpy.T) + cp.dot(xmy, xmy.T)
        doo = -cp.dot(xpy.T, xpy) - cp.dot(xmy.T, xmy)
        dmzoo = cp.dot(orbo, cp.dot(doo, orbo.T)) * 2.0
        dmzoo += cp.dot(orbv, cp.dot(dvv, orbv.T)) * 2.0
        
        h1 = cp.asarray(-mol_.intor('int1e_ipkin', comp=3) - mol_.intor('int1e_ipnuc', comp=3))
        from gpu4pyscf.df import int3c2e
        dh1e_td = int3c2e.get_dh1e(mol_, dmzoo)
        term_h_grad = cp.trace(dmzoo @ h1[2]) + dh1e_td[0,2]
        
        from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2
        class Dummy:
            def __init__(self, mol):
                self.mol = mol
                self.base = RHF(mol)
            def get_hcore(self, mol):
                nao = mol.nao
                h1aa = mol.intor('int1e_ipipkin', comp=9) + mol.intor('int1e_ipipnuc', comp=9)
                h1ab = mol.intor('int1e_ipkinip', comp=9) + mol.intor('int1e_ipnucip', comp=9)
                return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)
        
        dummy = Dummy(mol_)
        hcore_gen = _e_hcore_generator(dummy, dmzoo)
        term_h_static = float(hcore_gen(0, 0)[2,2])
        
        vhfopt = mf._opt_gpu.get(mol_.omega)
        ejk_static = _partial_ejk_ip2(mol_, dmzoo, vhfopt)
        term_eri_static = float(ejk_static[0, 0, 2, 2])
        
        return float(term_h_grad), float(term_h_static), float(term_eri_static)

    (tp_g, tp_s, tp_e), (tm_g, tm_s, tm_e) = get_terms(mol.copy().set_geom_(coords_p, unit='Bohr')), get_terms(mol.copy().set_geom_(coords_m, unit='Bohr'))
    
    print(f"FD H-core total [0,2,0,2]:  {(tp_g - tm_g)/(2*dr):.8f}")
    print(f"FD H-core static [0,2,0,2]: {tp_s:.8f}")
    print(f"FD ERI static [0,2,0,2]:    {tp_e:.8f}")

run_hess_fd()
