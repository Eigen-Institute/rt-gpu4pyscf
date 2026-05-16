import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian.tests.test_tdrhf_hessian import _build_mo1_ux

def check_residual():
    # Setup H2O system
    mol = pyscf.M(atom='''
O   0.000000000000   0.000000000000   0.117790000000
H   0.000000000000   0.755453000000  -0.471160000000
H   0.000000000000  -0.755453000000  -0.471160000000
''', basis='sto-3g', unit='Angstrom', verbose=0)
    mf = gpu_scf.RHF(mol).run()
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = 3
    td.kernel()

    state = 0
    omega = float(td.e[state])
    x_y = td.xy[state]
    x_y = tuple([cp.asarray(v) for v in x_y])
    
    h_obj = tdrhf_hess.Hessian(td)
    h_obj.verbose = 4
    mo1, mo_e1, Ux = _build_mo1_ux(mf, mol)
    
    nocc = int((mf.mo_occ > 0).sum())
    nvir = mol.nao - nocc
    nov = nocc * nvir
    natm = mol.natm
    
    print(f"Checking CP-TDDFT residual for state {state}, omega = {omega:.6f}")
    
    # Get RHS
    Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(h_obj, x_y, omega, mo1, mo_e1)
    
    # Check orthogonality of Delta to X
    # X in PySCF is normalized as 2*tr(X.T @ X) = 1
    # The matrix A is the Hessian of the energy with respect to density variations.
    # The eigenvector for A is X_unit = X * sqrt(2)
    X_unit = x_y[0].reshape(-1) * cp.sqrt(2.0)
    
    print(f"X_unit norm: {float(cp.linalg.norm(X_unit)):.6f}")
    
    for i in range(natm):
        for j in range(3):
            proj = cp.dot(Delta[i,j].reshape(-1), X_unit)
            norm_d = cp.linalg.norm(Delta[i,j])
            print(f"Atom {i} Dir {j}: <X|Delta> = {float(proj):.6e}, Delta norm = {float(norm_d):.6e}, ratio = {float(cp.abs(proj)/norm_d):.3f}")
    
    # Check D
    vind, hdiag = td.gen_vind()
    D = hdiag.ravel() - omega
    print(f"Min |D|: {float(cp.abs(D).min()):.6e}")
    
    # Check mo1base norms
    mo1base = Delta.reshape(-1, nov) / D
    for i in range(natm):
        for j in range(3):
            norm = float(cp.linalg.norm(mo1base[i*3+j]))
            print(f"Atom {i} Dir {j}: mo1base norm = {norm:.6e}")
    
    # Check mo1base singular values
    u, s, vh = cp.linalg.svd(mo1base)
    print(f"mo1base singular values: {s.get()}")
    
    # Solve CP-TDDFT
    x1, y1 = tdrhf_hess.solve_cptddft(h_obj, x_y, omega, mo1, mo_e1)
    
    for i in range(natm):
        for j in range(3):
            norm = float(cp.linalg.norm(x1[i,j]))
            print(f"Atom {i} Dir {j}: x1 norm = {norm:.6e}")
            
    print(f"x1 shape:    {x1.shape}")
    print(f"Delta shape: {Delta.shape}")
    
    x1_flat = x1.reshape(-1, nov)
    Ax1 = vind(x1_flat).reshape(natm, 3, nocc, nvir)
    
    print(f"Ax1 shape:   {Ax1.shape}")
    
    res_m = Ax1 - omega * x1 - Delta
    res_p = Ax1 - omega * x1 + Delta
    
    for i in range(natm):
        for j in range(3):
            norm_m = float(cp.linalg.norm(res_m[i,j]))
            norm_p = float(cp.linalg.norm(res_p[i,j]))
            norm_d = float(cp.linalg.norm(Delta[i,j]))
            print(f"Atom {i} Dir {j}: res(-)={norm_m:.6e}, res(+)={norm_p:.6e}, Delta={norm_d:.6e}")
    
    norm_res_plus = cp.linalg.norm(res_p)
    norm_res_minus = cp.linalg.norm(res_m)
    norm_rhs = cp.linalg.norm(Delta)
    
    print(f"Total: res(-)={float(norm_res_minus):.6e}, res(+)={float(norm_res_plus):.6e}, Delta={float(norm_rhs):.6e}")
    
    if min(norm_res_plus, norm_res_minus) / norm_rhs < 1e-6:
        print("SUCCESS: CP-TDDFT solver converged correctly.")
    else:
        print("FAILURE: CP-TDDFT solver residual too large.")

if __name__ == "__main__":
    check_residual()
