import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def check_rhs():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.kernel()
    
    x_y = td.xy[0]
    # SCALE AMPLITUDES TO MATCH LIU & LIANG CONVENTION
    x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in x_y])
    nocc = int((mf.mo_occ > 0).sum()); nvir = mf.mo_coeff.shape[1] - nocc
    xpy = cp.asarray(x_y[0]).reshape(nocc, nvir)
    mo_coeff = cp.asarray(mf.mo_coeff); mo_energy = cp.asarray(mf.mo_energy); mo_occ = cp.asarray(mf.mo_occ)
    orbo = mo_coeff[:,:nocc]; orbv = mo_coeff[:,nocc:]

    # FD derivative: d/dB (A(B) - w(B)) X
    delta = 0.001
    gs = []
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        mf_p = gpu_scf.RHF(mol_p).run()
        td_p = gpu_tdscf.rhf.TDA(mf_p); td_p.kernel()
        vresp = mf_p.gen_response(singlet=True, hermi=1)
        hdiag = (mf_p.mo_energy[nocc:] - mf_p.mo_energy[:nocc,None]).T
        dm = mf_p.mo_coeff[:,:nocc] @ xpy @ mf_p.mo_coeff[:,nocc:].T
        v1ao = vresp(dm + dm.T)
        v1mo = mf_p.mo_coeff[:,:nocc].T @ v1ao @ mf_p.mo_coeff[:,nocc:]
        ax = hdiag * xpy + v1mo
        gs.append(ax - td_p.e[0] * xpy)
    ax_deriv = (gs[0] - gs[1]) / (2.0 * delta)

    omega_x = tdrhf.omega_grad(td, 0)
    term_omega = omega_x[1, 2] * xpy
    
    h1ao_x = tdrhf._get_h1ao_x(mol)
    dm0 = cp.asarray(mf.make_rdm1())
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    intopt = gto.fakemol_for_charges(mol.atom_coords())
    from gpu4pyscf.df.int3c2e import VHFOpt as VHFOpt3c
    intopt = VHFOpt3c(mol, gto.fakemol_for_charges(mol.atom_coords()), 'int2e')
    intopt.build(1e-14, aosym=False)
    P_inv = cp.argsort(cp.asarray(intopt._ao_idx)); atm_inv = cp.argsort(cp.asarray(intopt._aux_ao_idx))
    vj_x_raw, vk_x_raw = _get_jk_ip1(mol, dm0)
    vj_x = vj_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_x = vk_x_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_dm0 = vj_x - 0.5 * vk_x
    F_x_AO = h1ao_x + G_x_dm0
    F_x_MO = mo_coeff.T @ F_x_AO[1, 2] @ mo_coeff
    term_fock = - F_x_MO[:nocc, :nocc] @ xpy + xpy @ F_x_MO[nocc:, nocc:]
    
    vj_R_raw, vk_R_raw = _get_jk_ip1(mol, orbo @ xpy @ orbv.T)
    vj_R = vj_R_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    vk_R = vk_R_raw.reshape(-1, 3, mol.nao, mol.nao)[:, :, P_inv][:, :, :, P_inv][atm_inv]
    G_x_RI = vj_R * 2.0 - vk_R
    term_ax = orbo.T @ G_x_RI[1, 2] @ orbv

    # 4. Numerical components of A^x X
    # A X = [F, X] + G[X]
    gs_fock = []; gs_gpart = []
    for d in [delta, -delta]:
        c = mol.atom_coords().copy(); c[1, 2] += d
        mol_p = mol.copy(); mol_p.set_geom_(c, unit='Bohr'); mol_p.build()
        mf_p = gpu_scf.RHF(mol_p).run()
        vresp = mf_p.gen_response(singlet=True, hermi=1)
        
        # [F(B), X]
        f_p = cp.asarray(mf_p.get_fock())
        f_mo_p = mf_p.mo_coeff.T @ f_p @ mf_p.mo_coeff
        # [[F, X]]_ia = F_aa X_ia - X_ia F_ii
        # Wait, operator is (E_a - E_i) X_ia
        hdiag = (mf_p.mo_energy[nocc:] - mf_p.mo_energy[:nocc,None]).T
        gs_fock.append(hdiag * xpy)
        
        # G(B)[X]
        dm = mf_p.mo_coeff[:,:nocc] @ xpy @ mf_p.mo_coeff[:,nocc:].T
        v1ao = vresp(dm + dm.T)
        v1mo = mf_p.mo_coeff[:,:nocc].T @ v1ao @ mf_p.mo_coeff[:,nocc:]
        gs_gpart.append(v1mo)
        
    f_deriv = (gs_fock[0] - gs_fock[1]) / (2.0 * delta)
    g_deriv = (gs_gpart[0] - gs_gpart[1]) / (2.0 * delta)

    print("\n--- A^x Component Calibration (atom 1, Z) ---")
    print(f"Fock Part [0,0]: Analytical {term_fock[0,0]:.6f}, FD {-f_deriv[0,0]:.6f}")
    print(f"G[X] Part [0,0]: Analytical {-term_ax[0,0]:.6f}, FD {-g_deriv[0,0]:.6f}")
    
    print(f"\nTarget Total Delta [0,0]: {omega_x[1,2]*xpy[0,0] - f_deriv[0,0] - g_deriv[0,0]:.6f}")

if __name__ == "__main__":
    check_rhs()
