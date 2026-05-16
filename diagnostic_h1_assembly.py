import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.grad import rhf as rhf_grad

def get_h1ao_x(mol):
    natm = mol.natm
    nao = mol.nao
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    aoslices = mol.aoslice_by_atom()
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = cp.asarray(mol.intor('int1e_iprinv', comp=3))
            vrinv *= mol.atom_charge(atm_id)
        h1ao_x[atm_id] = vrinv * 2.0
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
    return h1ao_x

def run_h1_comparison():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    
    # 1. My assembly
    h1_my = get_h1ao_x(mol)
    
    # 2. Grad engine assembly
    # rhf_grad.get_hcore returns -(ipkin+ipnuc)
    # rhf_grad.get_dh1e returns the operator part
    g_obj = rhf_grad.Gradients(mf)
    h1_basis = cp.asarray(g_obj.get_hcore(mol))
    
    # Check TI
    sum_h1 = h1_my.sum(axis=0)
    violation = cp.abs(sum_h1).max()
    print(f"H1 assembly TI violation: {float(violation):.6e}")
    print("Sum H1 matrix (comp=2):")
    print(sum_h1[2])
    print("Sum H1 matrix (comp=0):")
    print(sum_h1[0])
    
    # Check components
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    print(f"h1 [2,0,0]: {float(h1[2,0,0]):.8f}")
    with mol.with_rinv_at_nucleus(1):
        vrinv_1 = cp.asarray(mol.intor('int1e_iprinv', comp=3)) * mol.atom_charge(1)
    print(f"vrinv_1 [2,0,0]: {float(vrinv_1[2,0,0]):.8f}")
    
    # For H2 along Z
    val_my = float(h1_my[0,2,0,0])
    print(f"H1_my [0,2,0,0]: {val_my:.8f}")
    
    # Manual comparison for atom 0
    p0, p1 = mol.aoslice_by_atom()[0][2:]
    h1_basis_part = h1_basis[:, p0:p1, p0:p1]
    print(f"H1_basis [0,2,0,0]: {float(h1_basis_part[2,0,0]):.8f}")
    
    from gpu4pyscf.df import int3c2e
    dm = cp.zeros((mol.nao, mol.nao))
    dm[0,0] = 1.0 # Probe DM
    dh1e = int3c2e.get_dh1e(mol, dm)
    print(f"dh1e [0,2]: {float(dh1e[0,2]):.8f}")
    print(f"dh1e [1,2]: {float(dh1e[1,2]):.8f}")

run_h1_comparison()
