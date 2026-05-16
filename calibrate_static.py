import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def check_static_parts():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.kernel()
    
    # 1. Hcore part
    x_y = td.xy[0]
    nocc = int((mf.mo_occ > 0).sum()); nvir = mf.mo_coeff.shape[1] - nocc
    xpy = cp.asarray(x_y[0]).reshape(nocc, nvir)
    P_I_MO = cp.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    P_I_MO[nocc:, nocc:] = xpy.T @ xpy
    P_I_MO[:nocc, :nocc] = -xpy @ xpy.T
    P_I = cp.asarray(mf.mo_coeff) @ P_I_MO @ cp.asarray(mf.mo_coeff).T
    
    # Analytical: Tr(h^xy P_I)
    h_obj = tdrhf.Hessian(td)
    from gpu4pyscf.hessian.rhf import _e_hcore_generator
    de_hcore = _e_hcore_generator(h_obj, P_I)
    h_ana_hcore = de_hcore(0, 1) # atom 0, atom 1 cross derivative
    
    # FD: d/dB Tr(h^A P_I)
    delta = 0.001
    gs = []
    from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip2_charge_contracted, VHFOpt
    from gpu4pyscf.grad import rhf as grad_rhf
    
    aoslices = mol.aoslice_by_atom()
    p0, p1 = aoslices[0][2:]
    
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d # atom 1, Z
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        mf_p = gpu_scf.RHF(mol_p).run()
        
        # Pulay part: returns -h
        h1_pulay = grad_rhf.Gradients(mf_p).get_hcore(mol_p)
        # HF part
        intopt = VHFOpt(mol_p); intopt.build(1e-14, aosym=False)
        dh1e_ao = cp.zeros((mol_p.natm, 3, mol_p.nao, mol_p.nao))
        gridslice = np.stack([np.arange(2), np.arange(1, 3)], axis=1).astype(np.int32)
        int1e_grids_ip2_charge_contracted(mol_p, mol_p.atom_coords(), cp.asarray(mol_p.atom_charges(), dtype=np.float64), gridslice, dh1e_ao, intopt=intopt)
        
        # h^A Pulay part for atom 0
        h_0_z = cp.zeros((mol_p.nao, mol_p.nao))
        h_0_z[p0:p1, :] -= h1_pulay[2, p0:p1, :]
        h_0_z[:, p0:p1] -= h1_pulay[2, p0:p1, :].T
        # Add HF part for atom 0
        h_0_z += dh1e_ao[0, 2]
        
        gs.append(contract('pq,pq->', h_0_z, P_I))
        
    h_fd_hcore = (gs[0] - gs[1]) / (2.0 * delta)
    
    print("--- Static H-core Verification (atom 0, atom 1, Z, Z) ---")
    print(f"Analytical Hcore: {h_ana_hcore[2, 2]:.6f}")
    print(f"FD Hcore:         {h_fd_hcore:.6f}")
    print(f"Ratio:            {h_ana_hcore[2,2] / h_fd_hcore:.6f}")
    
    # 2. Overlap static part
    F_AO = cp.asarray(mf.get_fock())
    W_I = -0.5 * (P_I @ F_AO + F_AO @ P_I)
    from gpu4pyscf.hessian.rhf import get_ovlp
    s1aa, s1ab, _ = get_ovlp(mol)
    q0, q1 = aoslices[1][2:]
    # Analytical Tr(S^xy W)
    h_ana_ovlp = -contract('pq,pq->', cp.asarray(s1ab[2, 2, p0:p1, q0:q1]), W_I[p0:p1, q0:q1]) * 2.0
    
    # FD: d/dB -Tr(S^A W_I)
    gs_ovlp = []
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        s1ao = cp.asarray(mol_p.intor('int1e_ipovlp', comp=3))
        # S^A is <nabla mu | nu> + <mu | nabla nu> where mu on atom 0
        s_0_z = cp.zeros((mol.nao, mol.nao))
        s_0_z[p0:p1, :] += s1ao[2, p0:p1, :]
        s_0_z[:, p0:p1] += s1ao[2, p0:p1, :].T
        gs_ovlp.append(-contract('pq,pq->', s_0_z, W_I))
        
    h_fd_ovlp = (gs_ovlp[0] - gs_ovlp[1]) / (2.0 * delta)
    print("\n--- Static Overlap Verification (atom 0, atom 1, Z, Z) ---")
    print(f"Analytical Ovlp: {h_ana_ovlp:.6f}")
    print(f"FD Ovlp:         {h_fd_ovlp:.6f}")
    print(f"Ratio:           {h_ana_ovlp / h_fd_ovlp:.6f}")

if __name__ == "__main__":
    check_static_parts()
