import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def check_eri_doubling():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.kernel()
    
    x_y = td.xy[0]
    nocc = int((mf.mo_occ > 0).sum()); nvir = mf.mo_coeff.shape[1] - nocc
    xpy = cp.asarray(x_y[0]).reshape(nocc, nvir)
    mo_coeff = cp.asarray(mf.mo_coeff)
    
    # physical difference density P_I (trace 0.5 for PySCF amplitudes)
    P_I_MO = cp.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    P_I_MO[nocc:, nocc:] = xpy.T @ xpy
    P_I_MO[:nocc, :nocc] = -xpy @ xpy.T
    P_I = mo_coeff @ P_I_MO @ mo_coeff.T
    
    # 1. Analytical ERI Hessian cross-part: Tr(P_I G^xy [P_GS])
    # P_GS = 2.0 * GS density
    dm0 = cp.asarray(mf.make_rdm1())
    from gpu4pyscf.hessian.rhf import _partial_ejk_ip2
    vhfopt = mf._opt_gpu.get(mol.omega)
    # _partial_ejk_ip2(P_I + dm0) - _partial_ejk_ip2(P_I) - _partial_ejk_ip2(dm0)
    # matches Tr(P_I G^xy [dm0])
    h_ana_cross = _partial_ejk_ip2(mol, P_I + dm0, vhfopt)
    h_ana_cross -= _partial_ejk_ip2(mol, P_I, vhfopt)
    h_ana_cross -= _partial_ejk_ip2(mol, dm0, vhfopt)
    
    # 2. FD derivative: d/dB Tr(P_I G^x [P_GS])
    delta = 0.001
    gs = []
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        # Potential derivative G^x[dm0]
        vj_x_raw, vk_x_raw = _get_jk_ip1(mol_p, dm0)
        # Unsort
        from gpu4pyscf.df.int3c2e import VHFOpt as VHFOpt3c
        intopt = VHFOpt3c(mol_p, gto.fakemol_for_charges(mol_p.atom_coords()), 'int2e')
        intopt.build(1e-14, aosym=False)
        P_inv = cp.argsort(cp.asarray(intopt._ao_idx))
        atm_inv = cp.argsort(cp.asarray(intopt._aux_ao_idx))
        vj_x = vj_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
        vk_x = vk_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
        G_x_dm0 = vj_x - 0.5 * vk_x
        # Grad component (atom 0, Z)
        gs.append(contract('pq,pq->', G_x_dm0[0, 2], P_I))
        
    h_fd_cross = (gs[0] - gs[1]) / (2.0 * delta)
    
    print("--- ERI Cross Verification (atom 0, atom 1, Z, Z) ---")
    print(f"Analytical Cross: {h_ana_cross[0, 1, 2, 2]:.6f}")
    print(f"FD Cross:         {h_fd_cross:.6f}")
    print(f"Ratio:            {h_ana_cross[0,1,2,2] / h_fd_cross:.6f}")

if __name__ == "__main__":
    check_eri_doubling()
