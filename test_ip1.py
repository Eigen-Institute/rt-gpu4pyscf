import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.scf.hf import RHF
from gpu4pyscf.tdscf.rhf import TDA
from gpu4pyscf.hessian.tdrhf import Hessian, solve_z_vector, make_intermediates
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.hessian import rhf as rhf_hess
from functools import reduce
from gpu4pyscf.hessian.rhf import _e_hcore_generator, _partial_ejk_ip2, get_ovlp
from gpu4pyscf.grad.rhf import contract_h1e_dm

def run_ip1_test():
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', verbose=0)
    mf = RHF(mol).run()
    td = TDA(mf)
    td.nstates = 1
    td.kernel()
    
    # Scale amplitudes
    state = 0
    # Let's NOT scale them and see if it naturally converges to the right value.
    x_y = tuple([cp.asarray(v) for v in td.xy[state]])
    
    gs_hess = rhf_hess.Hessian(mf)
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mo_energy = cp.asarray(mf.mo_energy)
    
    # 1. solve_mo1
    h1mo = rhf_hess.make_h1(gs_hess, mo_coeff, mo_occ)
    fx = rhf_hess.gen_vind(gs_hess, mo_coeff, mo_occ)
    mo1, mo_e1 = rhf_hess.solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)
    
    natm = mol.natm
    nao = mol.nao
    nocc = int(mo_occ.sum() // 2)
    nvir = nao - nocc
    aoslices = mol.aoslice_by_atom()
    
    # 2. First-derivative integrals
    _, _, s1a_basis = get_ovlp(mol)
    s1a_basis = cp.asarray(s1a_basis)
    s1ao = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)

    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff

    Ux = cp.zeros((natm, 3, nao, nao))
    Ux[:, :, :, :nocc] = mo1
    Ux[:, :, :nocc, nocc:] = (-s1mo[:, :, :nocc, nocc:] - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2))
    Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]
    
    # h1ao_x
    h1 = cp.asarray(-mol.intor('int1e_ipkin', comp=3) - mol.intor('int1e_ipnuc', comp=3))
    h1ao_x = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        h1ao_x[atm_id, :, p0:p1] += h1[:, p0:p1]
        h1ao_x[atm_id, :, :, p0:p1] += h1[:, p0:p1].transpose(0, 2, 1)
        
    from gpu4pyscf.df import int3c2e
    coords = mol.atom_coords(); charges = cp.asarray(mol.atom_charges()); fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e'); intopt.build(1e-14, diag_block_with_triu=True, aosym=False)
    eye_ao = cp.eye(nao); eye_ao_sorted = intopt.sort_orbitals(eye_ao, axis=[0]); dh1e_ao = cp.zeros([natm, 3, nao, nao])
    from gpu4pyscf.lib.cupy_helper import contract
    for i0,i1,j0,j1,k0,k1,int3c_blk in int3c2e.loop_int3c2e_general(intopt, ip_type='ip1'):
        dh1e_ao[k0:k1, :, j0:j1, :] += contract('xkji,io->kxjo', int3c_blk, eye_ao_sorted[i0:i1])
        dh1e_ao[k0:k1, :, i0:i1, :] += contract('xkji,jo->kxio', int3c_blk, eye_ao_sorted[j0:j1])
    dh1e_ao = contract('kxjo,k->kxjo', dh1e_ao, -charges)
    P_sort = cp.asarray(intopt.sort_orbitals(np.eye(nao), axis=[0]))
    h1ao_x += contract('pj,kxjo->kxpo', P_sort.T, dh1e_ao)
    
    # 3. Intermediates
    td_grad_obj = tdrhf_grad.Gradients(td)
    z1 = solve_z_vector(td_grad_obj, x_y)
    h_obj = Hessian(td)
    inter = make_intermediates(h_obj, x_y, z1)
    
    # Check W_I against grad engine im0
    nmo = mo_coeff.shape[1]
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    # Re-calculate im0 from grad engine logic
    singlet = True
    x, y = [cp.asarray(v) for v in td.xy[state]]
    xpy_gr = (x + y).reshape(nocc, nvir).T
    xmy_gr = (x - y).reshape(nocc, nvir).T
    vresp = td_grad_obj.base.gen_response(singlet=singlet, hermi=0)
    def vresp_symm(dm):
        return vresp(dm + dm.T)
    dmxpy = orbv @ xpy_gr @ orbo.T
    veff0mop = mo_coeff.T @ vresp_symm(dmxpy) @ mo_coeff
    dmxmy = orbv @ xmy_gr @ orbo.T
    veff0mom = mo_coeff.T @ vresp(dmxmy - dmxmy.T) @ mo_coeff
    im0_gr = cp.zeros((nmo, nmo))
    im0_gr[:nocc, :nocc] = orbo.T @ (veff0mop[:nocc, :nocc] + veff0mom[:nocc, :nocc]) @ orbo # Simplified
    # Wait, the grad engine logic is complex. Let's just trust my W_I for now.
    
    W_I = inter['W_I']
    print(f"W_I norm: {float(cp.linalg.norm(W_I)):.6f}")
    
    # 4. Perturbed Intermediates
    x1, y1 = cp.zeros((natm, 3, nocc, nvir)), cp.zeros((natm, 3, nocc, nvir))
    x, y = x_y
    xpy = x.reshape(nocc, nvir) # TDA: y=0
    xmy = x.reshape(nocc, nvir)
    orbo = mo_coeff[:, :nocc]; orbv = mo_coeff[:, nocc:]
    
    P_I_prime_y = cp.zeros((natm, 3, nao, nao))
    R_I_y = cp.zeros((natm, 3, nao, nocc))
    T_I_y = cp.zeros((natm, 3, nao, nocc))
    P_y = cp.zeros((natm, 3, nao, nao))
    
    for i in range(natm):
        for j in range(3):
            C_y = mo_coeff @ Ux[i, j]
            Co_y = C_y[:, :nocc]
            Cv_y = C_y[:, nocc:]
            
            P_y[i, j] = Co_y @ orbo.T * 2 + orbo @ Co_y.T * 2
            
            R_I_y[i, j] = Cv_y @ xpy.T + orbv @ (x1[i,j] + y1[i,j]).T
            T_I_y[i, j] = Cv_y @ xmy.T + orbv @ (x1[i,j] - y1[i,j]).T
            
            t1 = Cv_y @ xpy.T @ xpy @ orbv.T + orbv @ (x1[i,j]+y1[i,j]).T @ xpy @ orbv.T + orbv @ xpy.T @ (x1[i,j]+y1[i,j]) @ orbv.T + orbv @ xpy.T @ xpy @ Cv_y.T
            t2 = Cv_y @ xmy.T @ xmy @ orbv.T + orbv @ (x1[i,j]-y1[i,j]).T @ xmy @ orbv.T + orbv @ xmy.T @ (x1[i,j]-y1[i,j]) @ orbv.T + orbv @ xmy.T @ xmy @ Cv_y.T
            t3 = Co_y @ xpy @ xpy.T @ orbo.T + orbo @ (x1[i,j]+y1[i,j]) @ xpy.T @ orbo.T + orbo @ xpy @ (x1[i,j]+y1[i,j]).T @ orbo.T + orbo @ xpy @ xpy.T @ Co_y.T
            t4 = Co_y @ xmy @ xmy.T @ orbo.T + orbo @ (x1[i,j]-y1[i,j]) @ xmy.T @ orbo.T + orbo @ xmy @ (x1[i,j]-y1[i,j]).T @ orbo.T + orbo @ xmy @ xmy.T @ Co_y.T
            P_I_y = t1 + t2 - t3 - t4
            
            P_I_prime_y[i, j] = P_I_y + (Cv_y @ z1 @ orbo.T + orbv @ z1 @ Co_y.T + Co_y @ z1.T @ orbv.T + orbo @ z1.T @ Cv_y.T)
            
    # Batch get_jk for G(P_y), G(P_I_prime_y), G(R_I_y + R_I_y.T)
    dms_y = cp.concatenate([
        P_y.reshape(-1, nao, nao),
        P_I_prime_y.reshape(-1, nao, nao),
        (R_I_y + R_I_y.transpose(0, 1, 3, 2)).reshape(-1, nao, nao)
    ], axis=0)
    vj_y, vk_y = mf.get_jk(mol, dms_y)
    G_y = vj_y - 0.5 * vk_y
    G_P_y = G_y[:natm*3].reshape(natm, 3, nao, nao)
    G_PI_prime_y = G_y[natm*3:2*natm*3].reshape(natm, 3, nao, nao)
    G_RI_y = G_y[2*natm*3:].reshape(natm, 3, nao, nao)
    
    L_I_prime_y = cp.zeros((natm, 3, nao, nao))
    W_I_y = cp.zeros((natm, 3, nao, nao))
    F_AO = inter['F_AO']
    P = inter['P']
    P_I_prime = inter['P_I_prime']
    R_I = inter['R_I']
    
    # Need G_PI and G_RI for ground state densities
    vj, vk = mf.get_jk(mol, cp.array([P_I_prime, R_I + R_I.T]))
    G_PI = vj[0] - 0.5 * vk[0]
    G_RI = vj[1] - 0.5 * vk[1]
    
    for i in range(natm):
        for j in range(3):
            C_y = mo_coeff @ Ux[i, j]
            # Orbital-rotation part of dF/dR^y: C_y^T F C + C^T F C_y ... wait!
            # The equations are in AO basis.
            # L_I_prime = P_I_prime @ F_AO + P @ G_PI + (R_I + R_I.T) @ G_RI
            # So L_I_prime_y = P_I_prime_y @ F_AO + P_I_prime @ G(P_y) + P_y @ G_PI + P @ G_PI_prime_y 
            #                  + (R_I_y + R_I_y.T) @ G_RI + (R_I + R_I.T) @ G_RI_y
            L_I_prime_y[i, j] = P_I_prime_y[i, j] @ F_AO + P_I_prime @ G_P_y[i, j] + P_y[i, j] @ G_PI \
                              + P @ G_PI_prime_y[i, j] + (R_I_y[i, j] + R_I_y[i, j].T) @ G_RI \
                              + (R_I + R_I.T) @ G_RI_y[i, j]
            W_I_y[i, j] = -0.5 * (L_I_prime_y[i, j] + L_I_prime_y[i, j].T)
            
    # 5. Assembly
    omega_xy = cp.zeros((natm, natm, 3, 3))
    
    # ip2 terms
    de_hcore = _e_hcore_generator(h_obj, P_I_prime)
    e1_hcore = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        for j0 in range(i0+1):
            e1_hcore[i0, j0] = de_hcore(i0, j0)
            e1_hcore[j0, i0] = e1_hcore[i0, j0].T
    omega_xy += e1_hcore
    
    vhfopt = mf._opt_gpu.get(mol.omega)
    ejk_PI = _partial_ejk_ip2(mol, P_I_prime + P, vhfopt)
    ejk_PI -= _partial_ejk_ip2(mol, P, vhfopt)
    ejk_PI += _partial_ejk_ip2(mol, R_I + R_I.T, vhfopt)
    omega_xy += ejk_PI
    
    s1aa, s1ab, _ = get_ovlp(mol)
    s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    W_I = inter['W_I']
    e1_ovlp = cp.zeros((natm, natm, 3, 3))
    for i in range(natm):
        p0, p1 = aoslices[i][2:]
        e1_ovlp[i, i] -= contract('xypq,pq->xy', s1aa[:,:,p0:p1], W_I[p0:p1]) * 2
        for j in range(i+1):
            q0, q1 = aoslices[j][2:]
            e1_ovlp[i, j] -= contract('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], W_I[p0:p1,q0:q1]) * 2
            e1_ovlp[j, i] = e1_ovlp[i, j].T
    omega_xy += e1_ovlp
    
    # ip1 terms (cross terms)
    # Hcore cross: Tr[h1ao_x[i] @ P_I_prime_y[j]]
    ip1_hcore = contract('ixpq,jyqp->ijxy', h1ao_x, P_I_prime_y)
    omega_xy += ip1_hcore + ip1_hcore.transpose(1, 0, 3, 2)
    
    # Overlap cross: -Tr[s1ao_atom[i] @ W_I_y[j]]
    ip1_ovlp = -contract('ixpq,jyqp->ijxy', s1ao, W_I_y)
    omega_xy += ip1_ovlp + ip1_ovlp.transpose(1, 0, 3, 2)
    
    # ERI cross:
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    vj_PI, vk_PI = _get_jk_ip1(mol, P_I_prime)
    G_PI_x = (vj_PI - 0.5 * vk_PI).reshape(natm, 3, nao, nao)
    
    vj_R, vk_R = _get_jk_ip1(mol, R_I + R_I.T)
    G_RI_x = (vj_R - 0.5 * vk_R).reshape(natm, 3, nao, nao)
    
    vj_P, vk_P = _get_jk_ip1(mol, P)
    G_P_x = (vj_P - 0.5 * vk_P).reshape(natm, 3, nao, nao)

    # Gradient: Tr(G_basis^x(P_total) P_total) - Tr(G_basis^x(P) P) + 0.5 Tr(G_basis^x(R_I) R_I)
    # Derivative wrt y:
    # Tr(G_basis^x(P_y) P_total) + Tr(G_basis^x(P_total) P_y) - Tr(G_basis^x(P_y) P) - Tr(G_basis^x(P) P_y) ...
    # Simplified cross term for difference:
    ip1_eri = contract('ixpq,jyqp->ijxy', G_PI_x, P_y) + contract('ixpq,jyqp->ijxy', G_P_x, P_I_prime_y)
    # Add the missing P_I_prime_y x G_PI_x term!
    ip1_eri += contract('ixpq,jyqp->ijxy', G_PI_x, P_I_prime_y)
    
    ip1_eri_RI = contract('ixpq,jyqp->ijxy', G_RI_x, R_I_y + R_I_y.transpose(0, 1, 3, 2))
    
    # ERI cross terms are symmetric by nature of the addition
    ip1_eri_total = ip1_eri + ip1_eri_RI
    omega_xy += ip1_eri_total + ip1_eri_total.transpose(1, 0, 3, 2)
    
    # Symmetrize and divide by 2
    omega_xy = (omega_xy + omega_xy.transpose(1, 0, 3, 2)) * 0.5
    omega_xy = omega_xy.transpose(0, 2, 1, 3) / 2.0
    
    # Semi-analytical FD
    h_semi = Hessian(td)
    h_semi_res = h_semi.kernel(state=0, method='semi-analytical')
    
    print("DEBUG Components [0,0,2,2]:")
    print(f"  ip2 hcore: {e1_hcore[0,0,2,2]:.6f}")
    print(f"  ip2 ejk:   {ejk_PI[0,0,2,2]:.6f}")
    
    # Overlap ip2
    # We added it directly to omega_xy, let's extract it
    print(f"  ip2 ovlp:  {e1_ovlp[0,0,2,2]:.6f}")
    
    print(f"  ip1 hcore: {ip1_hcore[0,0,2,2]:.6f} (x2 = {(ip1_hcore+ip1_hcore.transpose(1,0,3,2))[0,0,2,2]:.6f})")
    print(f"  ip1 ovlp:  {ip1_ovlp[0,0,2,2]:.6f} (x2 = {(ip1_ovlp+ip1_ovlp.transpose(1,0,3,2))[0,0,2,2]:.6f})")
    print(f"  ip1 eri:   {ip1_eri_total[0,0,2,2]:.6f} (x2 = {(ip1_eri_total+ip1_eri_total.transpose(1,0,3,2))[0,0,2,2]:.6f})")
    
    print("Analytical [0,2,0,2]:", omega_xy[0,2,0,2])
    print("Semi-Anal [0,2,0,2]: ", h_semi_res[0,2,0,2])
    print("Ratio:", omega_xy[0,2,0,2] / h_semi_res[0,2,0,2])

run_ip1_test()
