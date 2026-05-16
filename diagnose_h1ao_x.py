import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf.hessian import tdrhf
from gpu4pyscf.lib.cupy_helper import contract

def check_h1ao_x():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    dm0 = cp.asarray(mf.make_rdm1())
    
    # Pulay parts
    h1_kin = cp.asarray(mol.intor('int1e_ipkin', comp=3))
    h1_nuc = cp.asarray(mol.intor('int1e_ipnuc', comp=3))
    
    def get_g_pulay(h1):
        g = cp.zeros((mol.natm, 3))
        aoslices = mol.aoslice_by_atom()
        for atm_id in range(mol.natm):
            p0, p1 = aoslices[atm_id][2:]
            # Tr( (nabla i | j) P + (i | nabla j) P ) = 2 Tr( (nabla i | j) P )
            g[atm_id] = 2.0 * contract('xpq,pq->x', h1[:, p0:p1], dm0[p0:p1])
        return g

    g_pulay_kin = get_g_pulay(h1_kin)
    g_pulay_nuc = get_g_pulay(h1_nuc)
    
    # HF part
    from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip2_charge_contracted, VHFOpt
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=np.float64)
    intopt = VHFOpt(mol)
    intopt.build(1e-14, aosym=False)
    ngrids = mol.natm
    gridslice = np.stack([np.arange(ngrids), np.arange(1, ngrids + 1)], axis=1).astype(np.int32)
    dh1e_ao = cp.zeros((mol.natm, 3, mol.nao, mol.nao))
    int1e_grids_ip2_charge_contracted(mol, coords, charges, gridslice, dh1e_ao, intopt=intopt)
    
    g_hf = contract('kxpq,pq->kx', -dh1e_ao, dm0)
    
    # Nuc nuc
    from gpu4pyscf.grad import rhf as grad_rhf
    g_obj = grad_rhf.Gradients(mf)
    g_nuc = cp.asarray(g_obj.grad_nuc())
    
    # ERI part
    vj, vk = mf.get_jk(mol, dm0)
    # dvhf is Tr( (nabla ERI) P P )
    g_eri = cp.asarray(g_obj.get_veff(mol, dm0))
    
    # Overlap part
    s1 = cp.asarray(g_obj.get_ovlp(mol))
    # wdm = 2 \sum n_i epsilon_i |i><i|
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum())
    # W = orbo @ diag(epsilon_i) @ orbo.T * 2
    wdm = mo_coeff[:,:nocc] @ cp.diag(mo_energy[:nocc]) @ mo_coeff[:,:nocc].T * 2.0
    
    from gpu4pyscf.grad.rhf import contract_h1e_dm
    dh = cp.asarray(contract_h1e_dm(mol, h1_kin + h1_nuc, dm0, hermi=1))
    ds = cp.asarray(contract_h1e_dm(mol, s1, wdm, hermi=1))
    
    g_total_manual = dh - ds + 2.0 * g_eri + g_hf + g_nuc
    g_total_ref = cp.asarray(g_obj.kernel())
    
    # Try alternative signs
    g_alt1 = -dh - ds + 2.0 * g_eri + g_hf + g_nuc
    g_alt2 = dh - ds + 2.0 * g_eri - g_hf + g_nuc
    g_alt3 = -dh - ds + 2.0 * g_eri - g_hf + g_nuc
    
    print("--- Alternative Gradient Combinations ---")
    print(f"Manual (Original):\n{g_total_manual}")
    print(f"Alt 1 (-dh):\n{g_alt1}")
    print(f"Alt 2 (-hf):\n{g_alt2}")
    print(f"Alt 3 (-dh -hf):\n{g_alt3}")
    print(f"Ref Total:\n{g_total_ref}")
    
    print("\n--- Raw Integral Values (first 3) ---")
    print(f"h1_kin[2,0,0]: {h1_kin[2,0,0]}")
    print(f"h1_nuc[2,0,0]: {h1_nuc[2,0,0]}")
    
    ti_err = cp.abs(g_total_manual.sum(axis=0)).max()
    print(f"\nManual TI error: {ti_err}")

    
    # Pulay part from grad engine
    h1 = g_obj.get_hcore(mol) # Pulay part only usually?
    # Wait, Gradients.get_hcore returns <nabla mu | h | nu> part.
    
    # dh1e is Hellmann-Feynman part
    dh1e_ref = int1e_grids_ip2(mol, coords, charges=charges, dm=dm0).T
    
    print("--- Hellmann-Feynman Gradient Comparison ---")
    print("Manual HF grad (contracted):")
    # Pulay part of h1ao_x is:
    h1ao_x_pulay = cp.zeros_like(h1ao_x)
    h1_pulay = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    for atm_id in range(mol.natm):
        p0, p1 = aoslices[atm_id][2:]
        h1ao_x_pulay[atm_id, :, p0:p1] += h1_pulay[:, p0:p1]
        h1ao_x_pulay[atm_id, :, :, p0:p1] += h1_pulay[:, p0:p1].transpose(0, 2, 1)
    
    g_pulay = contract('kxpq,pq->kx', h1ao_x_pulay, dm0)
    g_hf = g_ana - g_pulay
    
    print(f"Manual HF Grad:\n{g_hf}")
    print(f"Ref dh1e Grad:\n{dh1e_ref}")
    
    print("\n--- Total Gradient Comparison ---")
    print(f"Manual Total Grad:\n{g_ana}")
    print(f"Ref Total Grad:\n{g_total}")
    
    ti_err = cp.abs(g_ana.sum(axis=0)).max()
    print(f"\nManual TI error: {ti_err}")

if __name__ == "__main__":
    check_h1ao_x()
