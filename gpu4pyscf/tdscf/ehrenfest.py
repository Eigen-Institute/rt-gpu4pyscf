# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cupy
from pyscf import lib, gto
from pyscf.gto.mole import intor_cross
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import tag_array

def _natural_orbitals(dm_re, s_mat):
    '''Natural orbitals for an arbitrary (possibly non-idempotent) real-symmetric
    density matrix P: returns (C, n) with P = C diag(n) C^T and C^T S C = I.
    Used to tag dm_re for PySCF gradient code, which needs an MO representation
    consistent with the *current* density — stale GS tags are a silent force bias.'''
    e, v = cupy.linalg.eigh(s_mat)
    mask = e > 1e-15
    e = e[mask]; v = v[:, mask]
    s_half = (v * cupy.sqrt(e)) @ v.conj().T
    s_neg_half = (v / cupy.sqrt(e)) @ v.conj().T
    p_tilde = s_half @ dm_re @ s_half
    p_tilde = 0.5 * (p_tilde + p_tilde.conj().T)
    n, u = cupy.linalg.eigh(p_tilde)
    idx = cupy.argsort(-n)
    return (s_neg_half @ u)[:, idx], n[idx]


def get_ehrenfest_force(rt_obj, dm_ao, t=0.0):
    '''
    Calculate the Ehrenfest force using PySCF component summation.
    Matches PySCF analytical gradient components.
    '''
    mol = rt_obj.mol
    mf = rt_obj.ks
    g = mf.Gradients()
    g.grid_response = True
    
    # 1. Hybrid Exchange Factor
    ni = mf._numint
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    if hybrid:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        c_k = hyb if rt_obj.is_uks else 0.5 * hyb
    else:
        c_k = 0.0
        
    hcore = cupy.asarray(mf.get_hcore())
    if rt_obj.is_uks and hcore.ndim == 2: hcore = cupy.stack([hcore, hcore])
    ints_mu = None
    if rt_obj.field_fn is not None:
        with mol.with_common_orig((0,0,0)): ints_mu = cupy.asarray(mol.intor('int1e_r'))
            
    fock = rt_obj.get_fock(dm_ao, hcore, ints_mu, t, c_k)
    dm_re = dm_ao.real
    
    # 2. Energy-weighted DM (W)
    if rt_obj.is_uks:
        w_re = (dm_ao @ fock @ dm_ao).real
    else:
        w_re = (0.5 * dm_ao @ fock @ dm_ao).real
        
    # 3. Component derivatives. Tag with natural orbitals of the *current*
    # dm_re at the *current* geometry so the XC grid-response (which uses
    # mo_coeff/mo_occ to eval rho on the grid) sees the right density.
    s_cur = cupy.asarray(mf.get_ovlp())
    if rt_obj.is_uks:
        c_a, n_a = _natural_orbitals(dm_re[0], s_cur)
        c_b, n_b = _natural_orbitals(dm_re[1], s_cur)
        no_coeff = cupy.stack([c_a, c_b])
        no_occ = cupy.stack([n_a, n_b])
    else:
        # RKS: make_rdm1 returns P = 2·Σ |i⟩⟨i|, so natural occupations are in [0,2].
        no_coeff, no_occ = _natural_orbitals(dm_re, s_cur)
    dm_re_tagged = tag_array(dm_re, mo_coeff=no_coeff, mo_occ=no_occ)
    dm_sf = dm_re[0] + dm_re[1] if rt_obj.is_uks else dm_re
    
    h1 = cupy.asarray(g.get_hcore(mol, exclude_ecp=True))
    s1 = cupy.asarray(g.get_ovlp(mol))
    if rt_obj.is_uks:
        h1 = cupy.stack([h1, h1])
        s1 = cupy.stack([s1, s1])

    # contract_h1e_dm with hermi=1 already multiplies by 2
    dh = rhf_grad.contract_h1e_dm(mol, h1, dm_re, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, w_re, hermi=1)
    
    dvhf = g.get_veff(mol, dm_re_tagged)
    dh1e = int3c2e.get_dh1e(mol, dm_sf)
    f_nuc = g.grad_nuc(mol)
    
    # 4. Imaginary Exchange Correction. dm_im is antisymmetric and contributes
    # no physical density, so tag with zero occupations to null the XC grid
    # response path (the remaining J/K derivatives are unaffected by mo_occ).
    if c_k > 0:
        dm_im = dm_ao.imag
        zero_occ = cupy.zeros_like(no_occ)
        dm_im = tag_array(dm_im, mo_coeff=no_coeff, mo_occ=zero_occ)
        de_im = -c_k * g.get_veff(mol, dm_im)
        dvhf += de_im.real
    
    # Extra forces (grid response etc)
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    dm0 = dm_re_tagged
    extra = np.zeros((mol.natm, 3))
    for i in range(mol.natm):
        extra[i] += np.asarray(g.extra_force(i, locals()))

    # Total gradient: dh - ds + 2.0*dvhf + dh1e
    de = dh - ds + 2.0 * cupy.asnumpy(dvhf) + cupy.asnumpy(dh1e) + f_nuc + extra
    return -de

class BaseMD(RTTDDFT):
    '''Base class for Molecular Dynamics.'''
    def __init__(self, ks, basis='OAO'):
        super().__init__(ks, basis=basis)
        self.velocities = None
        self.masses = np.array(self.mol.atom_mass_list()) * 1822.888486
        self.forces = None
        self.thermostat = None
        self.target_temp = 298.15
        self.tau = 1000.0
        self._keys.update({'velocities', 'masses', 'forces', 'thermostat', 'target_temp', 'tau'})

    def _apply_thermostat(self, dt):
        if self.thermostat != 'svr': return
        k_curr = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        ndof = 3 * self.mol.natm - 6
        if ndof <= 0: ndof = 3 * self.mol.natm
        kb = 3.166811e-6
        k_targ = 0.5 * ndof * kb * self.target_temp
        c = np.exp(-dt / self.tau)
        r = np.random.normal(0, 1)
        alpha2 = c + (1-c)*(k_targ/k_curr) + 2*np.sqrt(c*(1-c)*k_targ/k_curr)*r/np.sqrt(ndof)
        if alpha2 < 0: alpha2 = 0.0
        self.velocities *= np.sqrt(alpha2)

    def _record_md(self, t, dm, coords, results):
        results['times'].append(t)
        results['coords'].append(coords.copy())
        results['velocities'].append(self.velocities.copy())
        results['forces'].append(self.forces.copy())
        dm_re = dm.real
        veff = self.ks.get_veff(self.mol, dm_re)
        hcore = self.ks.get_hcore()
        if self.is_uks:
            if hcore.ndim == 2: hcore = cupy.stack([hcore, hcore])
            e1 = cupy.einsum('sij,sji->', hcore, dm_re).real
        else:
            e1 = cupy.einsum('ij,ji->', hcore, dm_re).real
        e_elec = e1 + veff.ecoul + veff.exc + self.mol.energy_nuc()
        # Hybrid correction for complex DM: +0.5*c_k*Tr(P_im*K[P_im]) (matches RTTDDFT._record).
        ni = self.ks._numint
        if ni.libxc.is_hybrid_xc(self.ks.xc):
            _, _, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
            if self.is_uks:
                vk_im_a = self.ks.get_k(self.mol, dm.imag[0], hermi=0)
                vk_im_b = self.ks.get_k(self.mol, dm.imag[1], hermi=0)
                vk_im = cupy.stack([vk_im_a, vk_im_b])
                e_elec = e_elec + 0.5 * c_k * cupy.einsum('sij,sji->', dm.imag, vk_im).real
            else:
                vk_im = self.ks.get_k(self.mol, dm.imag, hermi=0)
                e_elec = e_elec + 0.5 * c_k * cupy.einsum('ij,ji->', dm.imag, vk_im).real
        e_elec = float(e_elec.real)
        results['energy_elec'].append(e_elec)
        e_kin = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        results['energy_tot'].append(e_elec + e_kin)

class EhrenfestMD(BaseMD):
    '''Ehrenfest Molecular Dynamics implementation.'''
    def kernel(self, times, dm0=None, dt=0.02, propagator='magnus_interpol',
               callback=None, n_electronic=1):
        '''Propagate Ehrenfest dynamics.

        dt: nuclear time step. Verlet uses this size.
        n_electronic: number of electronic Magnus sub-steps per nuclear step.
            Electrons advance in steps of dt/n_electronic with geometry linearly
            interpolated between R(t) and R(t+dt). Forces on nuclei are still
            only evaluated at the endpoints (standard multi-scale Ehrenfest).
            n_electronic=1 reproduces the single-scale scheme.
        '''
        log = logger.new_logger(self, self.verbose)
        mol, mf = self.mol, self.ks
        if n_electronic < 1:
            raise ValueError("n_electronic must be >= 1")
        if dm0 is None: dm0 = mf.make_rdm1()
        dm_ao = cupy.asarray(dm0).astype(cupy.complex128)
        # Primary state is the density in the Löwdin-orthogonal basis. When the
        # nuclei move we hold dm_orth fixed across the basis change; since the
        # Löwdin basis tracks atoms smoothly, this is far less wrong than
        # holding dm_ao fixed (which reinterprets matrix elements in a different
        # AO basis at every step). Equivalent to a sudden approximation in the
        # Löwdin basis and is standard in Ehrenfest MD.
        dm_orth = self.to_orth(dm_ao)

        coords = mol.atom_coords()
        if self.velocities is None: self.velocities = np.zeros_like(coords)
        log.info(f"Ehrenfest MD: dt={dt}, n_electronic={n_electronic} "
                 f"(dt_e={dt/n_electronic:.4g}). Computing initial forces...")
        self.forces = get_ehrenfest_force(self, dm_ao, t=0.0)
        results = {'times': [], 'coords': [], 'velocities': [], 'forces': [], 'energy_elec': [], 'energy_tot': []}
        t_now = 0.0
        self._record_md(t_now, dm_ao, coords, results)
        if callback: callback(t_now, dm_ao, results)
        dt_e = dt / n_electronic
        for t_target in times:
            if t_target <= t_now + 1e-6: continue
            steps = int(np.round((t_target - t_now) / dt))
            for step in range(steps):
                # Predictor: Nuclear Half-step (nuclear-scale dt)
                accel = self.forces / self.masses[:, None]
                v_mid = self.velocities + 0.5 * accel * dt
                r_next = coords + v_mid * dt

                # Electronic sub-steps at R(τ) linearly interpolated between
                # R(t) and R(t+dt). For n_electronic=1 this reduces to the
                # single-step scheme with the move happening before the
                # electronic step.
                f_orth_prev = None
                mol_prev = mol.copy()           # mol at R(τ_{j}) for D_dt
                x_mat_prev = self.x_mat.copy()  # X at R(τ_{j})
                if propagator == 'magnus_interpol':
                    f_orth_prev = self._build_f_orth(dm_orth, t_now)
                for j in range(n_electronic):
                    tau_end = (j + 1) / n_electronic
                    R_tau = coords + tau_end * (r_next - coords)
                    mol.set_geom_(R_tau, unit='Bohr')
                    mf.reset(mol)
                    self._update_basis()

                    D_dt_sub = self._compute_D_dt(mol_prev, x_mat_prev)

                    t_sub = t_now + j * dt_e
                    dm_orth = self._electronic_step(
                        dm_orth, t_sub, dt_e, propagator,
                        f_orth_t=f_orth_prev, D_dt=D_dt_sub)

                    # Hand off F_orth and basis snapshots to the next sub-step.
                    if j < n_electronic - 1 and propagator == 'magnus_interpol':
                        f_orth_prev = self._build_f_orth(dm_orth, t_sub + dt_e)
                        mol_prev = mol.copy()
                        x_mat_prev = self.x_mat.copy()

                # AO representation at R(t+dt) for force computation & logging.
                dm_ao = self.to_ao(dm_orth)

                # New forces at t+dt (nuclear-endpoint only)
                new_forces = get_ehrenfest_force(self, dm_ao, t=t_now + dt)
                new_accel = new_forces / self.masses[:, None]

                # Corrector: Nuclear Full-step
                self.velocities = v_mid + 0.5 * new_accel * dt
                coords = r_next
                self.forces = new_forces

                self._apply_thermostat(dt)
                t_now += dt
            self._record_md(t_now, dm_ao, coords, results)
            if callback: callback(t_now, dm_ao, results)
            log.info(f"Time: {t_now:10.4f} au | Energy: {results['energy_tot'][-1]:20.12f}")
        return results

    def _update_basis(self):
        s = cupy.asarray(self.ks.get_ovlp())
        if self.x_mat.ndim == 3:
             c = cupy.asarray(self.ks.mo_coeff)
             self.x_mat = c
             self.x_inv = cupy.einsum('sji,jk->sik', c.conj(), s) if self.is_uks else c.conj().T @ s
        else:
            e, v = cupy.linalg.eigh(s)
            mask = e > 1e-15
            e, v = e[mask], v[:, mask]
            self.x_mat = v @ cupy.diag(e**(-0.5)) @ v.T
            self.x_inv = v @ cupy.diag(e**(0.5)) @ v.T

    def _compute_D_dt(self, mol_old, x_mat_old):
        '''Non-adiabatic basis-velocity coupling, integrated over one step.
        Returns D·dt ≈ antisym(X_old^T · M · X_new), where
            M_{μν} = ⟨χ_μ(R_old) | χ_ν(R_new)⟩.
        Derivation: ⟨φ̃_i(t) | φ̃_j(t+dt)⟩ = δ_ij + dt · D_ij + O(dt²) for a
        time-dependent orthonormal basis, and for real χ the anti-symmetric
        part of X_old^T M X_new is exactly D·dt to O(dt²) with the symmetric
        O(dt²) noise removed.  D is anti-Hermitian; F_eff = F - iD is
        Hermitian, so the modified propagator exp(-i F dt - D dt) is unitary.'''
        M = cupy.asarray(intor_cross('int1e_ovlp', mol_old, self.mol))
        XMX = x_mat_old.conj().T @ M @ self.x_mat
        return 0.5 * (XMX - XMX.conj().swapaxes(-1, -2))

    def _build_f_orth(self, dm_orth, t):
        '''Build the orthogonal-basis Fock F_orth at the *current* geometry
        from dm_orth. Used by kernel to snapshot F(t) at R(t) before the
        nuclear move, so the Magnus interpolator can symmetrically average
        F(t)/F(t+dt) even though each is built at a different geometry.'''
        dm_ao = self.to_ao(dm_orth)
        hcore = cupy.asarray(self.ks.get_hcore())
        if self.is_uks and hcore.ndim == 2: hcore = cupy.stack([hcore, hcore])
        ints_mu = None
        if self.field_fn:
            with self.mol.with_common_orig((0,0,0)):
                ints_mu = cupy.asarray(self.mol.intor('int1e_r'))
        ni = self.ks._numint
        c_k = 0.0
        if ni.libxc.is_hybrid_xc(self.ks.xc):
            _, _, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
        f_ao = self.get_fock(dm_ao, hcore, ints_mu, t, c_k)
        return self.to_orth_fock(f_ao)

    def _electronic_step(self, dm_orth, t, dt, propagator, f_orth_t=None, D_dt=None):
        '''Propagate dm_orth (Löwdin basis) from t to t+dt at the current
        geometry. Input and output are both in the orthogonal basis.

        f_orth_t: optional pre-built F_orth at t. If provided (Ehrenfest path),
        the Magnus interpolator will average it with F(t+dt) at the current
        geometry. If None (static-geometry path), F(t) is built here.

        D_dt: optional non-adiabatic coupling matrix integrated over dt. When
        provided, the propagator uses F_eff = F - i·D, yielding the unitary
        exp(-i F dt - D dt). D = D_dt / dt; we pre-compute -i·D_dt/dt once and
        add it to F before each propagator call.'''
        hcore = cupy.asarray(self.ks.get_hcore())
        if self.is_uks and hcore.ndim == 2: hcore = cupy.stack([hcore, hcore])
        ints_mu = None
        if self.field_fn:
            with self.mol.with_common_orig((0,0,0)): ints_mu = cupy.asarray(self.mol.intor('int1e_r'))
        ni = self.ks._numint
        hybrid = ni.libxc.is_hybrid_xc(self.ks.xc)
        c_k = 0.0
        if hybrid:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb

        max_iter, threshold = 15, 1e-7

        # Basis-velocity correction folded into an effective Fock:
        #   U = exp(-i (F - iD) dt) = exp(-i F dt - D dt).
        # We add F_couple := -i · D to the averaged F inside each propagator call.
        f_couple = None
        if D_dt is not None:
            f_couple = -1j * D_dt / dt  # shape (N,N); broadcasts over UKS spin

        def _apply(u, p):
            return u @ p @ u.conj().swapaxes(-1, -2) if self.is_uks else u @ p @ u.conj().T

        def _add_couple(f):
            return f if f_couple is None else f + f_couple

        if propagator == 'magnus_interpol':
            # NWChem-style F-averaging. If f_orth_t was pre-built at R(t) we
            # average in the orthogonal basis (geometry-consistent form); else
            # build both at the current geometry and average in AO.
            if f_orth_t is None:
                dm_ao = self.to_ao(dm_orth)
                f_ao_t = self.get_fock(dm_ao, hcore, ints_mu, t, c_k)
            dm_orth_next = dm_orth.copy()
            for _ in range(max_iter):
                dm_prev = dm_orth_next
                dm_ao_next = self.to_ao(dm_orth_next)
                f_ao_next = self.get_fock(dm_ao_next, hcore, ints_mu, t + dt, c_k)
                if f_orth_t is None:
                    f_orth_mid = self.to_orth_fock(0.5 * (f_ao_t + f_ao_next))
                else:
                    f_orth_next = self.to_orth_fock(f_ao_next)
                    f_orth_mid = 0.5 * (f_orth_t + f_orth_next)
                u = self.compute_propagator(_add_couple(f_orth_mid), dt)
                dm_orth_next = _apply(u, dm_orth)
                if cupy.linalg.norm(dm_orth_next - dm_prev) < threshold: break
            return dm_orth_next

        elif propagator == 'magnus_iter':
            # Self-consistent midpoint:  converge P(t+dt/2), then do full step
            dm_orth_mid = dm_orth.copy()
            f_orth_mid = None
            for _ in range(max_iter):
                dm_prev = dm_orth_mid
                dm_ao_mid = self.to_ao(dm_orth_mid)
                f_ao_mid = self.get_fock(dm_ao_mid, hcore, ints_mu, t + dt/2, c_k)
                f_orth_mid = self.to_orth_fock(f_ao_mid)
                u_half = self.compute_propagator(_add_couple(f_orth_mid), dt/2)
                dm_orth_mid = _apply(u_half, dm_orth)
                if cupy.linalg.norm(dm_orth_mid - dm_prev) < threshold: break
            u_full = self.compute_propagator(_add_couple(f_orth_mid), dt)
            return _apply(u_full, dm_orth)

        else:
            # Single-step: F(t,P(t))
            dm_ao = self.to_ao(dm_orth)
            f_ao_t = self.get_fock(dm_ao, hcore, ints_mu, t, c_k)
            f_orth = self.to_orth_fock(f_ao_t)
            u = self.compute_propagator(_add_couple(f_orth), dt)
            return _apply(u, dm_orth)

    def to_orth(self, dm_ao):
        xi = self.x_inv
        return xi @ dm_ao @ xi.conj().swapaxes(-1, -2) if self.is_uks else xi @ dm_ao @ xi.conj().T

class BOMD(BaseMD):
    '''Born-Oppenheimer Molecular Dynamics implementation.'''
    def kernel(self, times, dt=0.02, callback=None):
        log = logger.new_logger(self, self.verbose)
        mol, mf = self.mol, self.ks
        mf.kernel()
        dm_ao = cupy.asarray(mf.make_rdm1()).astype(cupy.complex128)
        coords = mol.atom_coords()
        if self.velocities is None: self.velocities = np.zeros_like(coords)
        log.info("Computing initial forces (BOMD)...")
        self.forces = -mf.Gradients().kernel()
        results = {'times': [], 'coords': [], 'velocities': [], 'forces': [], 'energy_elec': [], 'energy_tot': []}
        t_now = 0.0
        self._record_md(t_now, dm_ao, coords, results)
        if callback: callback(t_now, dm_ao, results)
        for t_target in times:
            if t_target <= t_now + 1e-6: continue
            steps = int(np.round((t_target - t_now) / dt))
            for step in range(steps):
                accel = self.forces / self.masses[:, None]
                self.velocities += 0.5 * accel * dt
                coords += self.velocities * dt
                mol.set_geom_(coords, unit='Bohr')
                mf.reset(mol)
                mf.kernel()
                new_forces = -mf.Gradients().kernel()
                new_accel = new_forces / self.masses[:, None]
                self.velocities += 0.5 * new_accel * dt
                self.forces = new_forces
                self._apply_thermostat(dt)
                t_now += dt
            dm_ao = cupy.asarray(mf.make_rdm1()).astype(cupy.complex128)
            self._record_md(t_now, dm_ao, coords, results)
            if callback: callback(t_now, dm_ao, results)
            log.info(f"Time: {t_now:10.4f} au | Energy: {results['energy_tot'][-1]:20.12f}")
        return results
