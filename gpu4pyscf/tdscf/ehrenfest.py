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
    '''Natural orbitals for tagging density matrices for XC grid response.'''
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
    Calculate the Ehrenfest force.
    Returns: (force, energy_elec)
    '''
    mol = rt_obj.mol
    mf = rt_obj.ks

    # Re-Hermitianize the density matrix.  Q-transport and unitary propagation
    # preserve Hermiticity only to float precision (~1e-14).  Tiny residual
    # non-(anti)symmetry in (dm_re, dm_im) is amplified to O(1) by the gpu4pyscf
    # gradient path, so we enforce exact (P + P†)/2 here.
    if dm_ao.ndim == 3:  # UKS: stack of (alpha, beta)
        dm_ao = 0.5 * (dm_ao + dm_ao.conj().swapaxes(-1, -2))
    else:
        dm_ao = 0.5 * (dm_ao + dm_ao.conj().T)

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
        
    # 3. Component derivatives
    s_cur = cupy.asarray(mf.get_ovlp())
    if rt_obj.is_uks:
        c_a, n_a = _natural_orbitals(dm_re[0], s_cur)
        c_b, n_b = _natural_orbitals(dm_re[1], s_cur)
        no_coeff = cupy.stack([c_a, c_b])
        no_occ = cupy.stack([n_a, n_b])
    else:
        no_coeff, no_occ = _natural_orbitals(dm_re, s_cur)
    dm_re_tagged = tag_array(dm_re, mo_coeff=no_coeff, mo_occ=no_occ)
    dm_sf = dm_re[0] + dm_re[1] if rt_obj.is_uks else dm_re
    
    h1 = cupy.asarray(g.get_hcore(mol, exclude_ecp=True))
    s1 = cupy.asarray(g.get_ovlp(mol))
    if rt_obj.is_uks:
        h1 = cupy.stack([h1, h1])
        s1 = cupy.stack([s1, s1])

    dh = rhf_grad.contract_h1e_dm(mol, h1, dm_re, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, w_re, hermi=1)
    
    # dvhf already includes full J/K part and grid response
    dvhf = g.get_veff(mol, dm_re_tagged)
    dh1e = int3c2e.get_dh1e(mol, dm_sf)
    f_nuc = g.grad_nuc(mol)
    
    # 4. Imaginary Exchange Correction
    if c_k > 0:
        dm_im = dm_ao.imag
        zero_occ = cupy.zeros_like(no_occ)
        dm_im = tag_array(dm_im, mo_coeff=no_coeff, mo_occ=zero_occ)
        de_im = -c_k * g.get_veff(mol, dm_im)
        dvhf += de_im.real
    
    # Extra forces
    extra = np.zeros((mol.natm, 3))
    # We need mo_coeff/occ in locals() for extra_force
    mo_coeff, mo_occ, dm0 = no_coeff, no_occ, dm_re_tagged
    for i in range(mol.natm):
        extra[i] += np.asarray(g.extra_force(i, locals()))

    # Total gradient
    de = dh - ds + 2.0 * cupy.asnumpy(dvhf) + cupy.asnumpy(dh1e) + f_nuc + extra
    
    # Energy computation (optimized: use components already built)
    # E = Tr(P Hcore) + 0.5*Tr(P Veff) + Vnuc
    # Note: dvhf.energy contains Ecoul + Exc.
    e1 = cupy.einsum('sij,sji->', hcore, dm_re).real if rt_obj.is_uks else cupy.einsum('ij,ji->', hcore, dm_re).real
    # get_veff returns a special object with energy attributes in gpu4pyscf
    veff = mf.get_veff(mol, dm_re)
    e_elec = e1 + veff.ecoul + veff.exc + mol.energy_nuc()
    
    # Imaginary exchange energy correction
    if c_k > 0:
        if rt_obj.is_uks:
            vk_im_a = mf.get_k(mol, dm_ao.imag[0], hermi=0)
            vk_im_b = mf.get_k(mol, dm_ao.imag[1], hermi=0)
            vk_im = cupy.stack([vk_im_a, vk_im_b])
            e_elec += 0.5 * c_k * cupy.einsum('sij,sji->', dm_ao.imag, vk_im).real
        else:
            vk_im = mf.get_k(mol, dm_ao.imag, hermi=0)
            e_elec += 0.5 * c_k * cupy.einsum('ij,ji->', dm_ao.imag, vk_im).real
            
    return -de, float(e_elec.real)

class BaseMD(RTTDDFT):
    '''Base class for Molecular Dynamics.'''
    def __init__(self, ks, basis='OAO'):
        super().__init__(ks, basis=basis)
        self.velocities = None
        self.masses = np.array(self.mol.atom_mass_list()) * 1822.888486
        self.forces = None
        self.energy_elec = 0.0
        self.thermostat = None
        self.target_temp = 298.15
        self.tau = 1000.0
        self.frozen = False
        self._keys.update({'velocities', 'masses', 'forces', 'thermostat', 'target_temp', 'tau', 'frozen', 'energy_elec'})

    def _apply_thermostat(self, dt):
        if self.thermostat != 'svr' or self.frozen: return
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
        results['energy_elec'].append(self.energy_elec)
        e_kin = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        results['energy_tot'].append(self.energy_elec + e_kin)

    def _compute_elec_energy(self, dm_ao):
        '''Cheaper than get_ehrenfest_force: energy only, no gradient.
        Used for mid-step records between nuclear boundaries.'''
        dm_re = dm_ao.real
        hcore = cupy.asarray(self.ks.get_hcore())
        if self.is_uks and hcore.ndim == 2:
            hcore = cupy.stack([hcore, hcore])
        veff = self.ks.get_veff(self.mol, dm_re)
        if self.is_uks:
            e1 = cupy.einsum('sij,sji->', hcore, dm_re).real
        else:
            e1 = cupy.einsum('ij,ji->', hcore, dm_re).real
        e_elec = e1 + veff.ecoul + veff.exc + self.mol.energy_nuc()
        ni = self.ks._numint
        if ni.libxc.is_hybrid_xc(self.ks.xc):
            _, _, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
            if self.is_uks:
                vk_a = self.ks.get_k(self.mol, dm_ao.imag[0], hermi=0)
                vk_b = self.ks.get_k(self.mol, dm_ao.imag[1], hermi=0)
                vk_im = cupy.stack([vk_a, vk_b])
                e_elec = e_elec + 0.5 * c_k * cupy.einsum('sij,sji->', dm_ao.imag, vk_im).real
            else:
                vk_im = self.ks.get_k(self.mol, dm_ao.imag, hermi=0)
                e_elec = e_elec + 0.5 * c_k * cupy.einsum('ij,ji->', dm_ao.imag, vk_im).real
        return float(e_elec.real)

class EhrenfestMD(BaseMD):
    '''Ehrenfest Molecular Dynamics implementation.

    Time stepping:
        dt           - electronic time step
        n_electronic - electronic sub-steps per nuclear step
        dt_nuclear = dt * n_electronic

    Recording fires at every user-supplied t_target in `times` that lies on
    an electronic sub-step boundary (t_target must be a multiple of dt).
    Forces and velocities are only updated at nuclear-step boundaries;
    between them, coords/velocities are from the last nuclear boundary
    and energy is recomputed cheaply from the current complex density.

    When self.frozen is True, geometry-update machinery (set_geom_, reset,
    _update_basis, _compute_D_dt) is skipped entirely, collapsing the
    Ehrenfest path onto pure electronic propagation.
    '''
    def _record_md(self, t, dm_ao, coords, results):
        '''Populate trajectory + full RT-TDDFT-style electronic fields.
        Consolidated so a single get_veff call serves both the total energy
        (for self.energy_elec / energy_tot) and the component breakdown
        (energy_core / energy_coul / energy_xc) consumed by RTLogger.
        Also computes dipole (per-spin for UKS), field energy, and MO
        populations if self.record_occ is set.'''
        mol = self.mol
        dm_cpu = cupy.asnumpy(dm_ao)

        results['times'].append(t)
        results['coords'].append(coords.copy())
        results['velocities'].append(self.velocities.copy())
        results['forces'].append(self.forces.copy())

        # --- Dipole (atomic units) ---
        charges = mol.atom_charges()
        atom_coords = mol.atom_coords()
        charge_center = np.einsum('z,zr->r', charges, atom_coords) / charges.sum()
        with mol.with_common_orig(charge_center):
            ints = mol.intor('int1e_r')
            dip_nuc = np.einsum('z,zx->x', charges, atom_coords - charge_center)
            if self.is_uks:
                dip_a_el = -np.einsum('xij,ji->x', ints, dm_cpu[0].real)
                dip_b_el = -np.einsum('xij,ji->x', ints, dm_cpu[1].real)
                dip_tot = dip_nuc + dip_a_el + dip_b_el
                dip_a = dip_a_el + 0.5 * dip_nuc
                dip_b = dip_b_el + 0.5 * dip_nuc
                results['dip_alpha'].append(dip_a)
                results['dip_beta'].append(dip_b)
                if self.mu_spin == 'alpha':
                    results['dip'].append(dip_a)
                elif self.mu_spin == 'beta':
                    results['dip'].append(dip_b)
                else:
                    results['dip'].append(dip_tot)
            else:
                dip_el = -np.einsum('xij,ji->x', ints, dm_cpu.real)
                results['dip'].append(dip_nuc + dip_el)

        # --- Energy breakdown from a single get_veff call ---
        dm_re = dm_ao.real
        hcore = cupy.asarray(self.ks.get_hcore())
        if self.is_uks and hcore.ndim == 2:
            hcore = cupy.stack([hcore, hcore])
        veff = self.ks.get_veff(mol, dm_re)
        if self.is_uks:
            e1 = float(cupy.einsum('sij,sji->', hcore, dm_re).real)
        else:
            e1 = float(cupy.einsum('ij,ji->', hcore, dm_re).real)
        ecoul = float(veff.ecoul)
        exc = float(veff.exc)
        enuc = float(mol.energy_nuc())

        # Hybrid correction from imaginary density: +0.5·c_k·Tr(P_im K[P_im])
        ni = self.ks._numint
        e_im = 0.0
        if ni.libxc.is_hybrid_xc(self.ks.xc):
            _, _, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
            if self.is_uks:
                vk_a = self.ks.get_k(mol, dm_ao.imag[0], hermi=0)
                vk_b = self.ks.get_k(mol, dm_ao.imag[1], hermi=0)
                vk_im = cupy.stack([vk_a, vk_b])
                e_im = 0.5 * c_k * float(cupy.einsum('sij,sji->', dm_ao.imag, vk_im).real)
            else:
                vk_im = self.ks.get_k(mol, dm_ao.imag, hermi=0)
                e_im = 0.5 * c_k * float(cupy.einsum('ij,ji->', dm_ao.imag, vk_im).real)

        e_elec = e1 + ecoul + exc + enuc + e_im
        e_kin = 0.5 * float(np.sum(self.masses[:, None] * self.velocities**2))
        self.energy_elec = e_elec

        results['energy_elec'].append(e_elec)
        results['energy_tot'].append(e_elec + e_kin)
        # RTLogger reads 'energy' preferentially when no Ehrenfest keys; we
        # duplicate to 'energy' too for callbacks that only check that key.
        results['energy'].append(e_elec + e_kin)
        results['energy_nuc'].append(enuc)
        results['energy_core'].append(e1)
        results['energy_coul'].append(ecoul)
        results['energy_xc'].append(exc + e_im)

        # --- Field and field energy (in AU, directly) ---
        if self.field_fn is not None:
            efield = np.array(self.field_fn(t))
            dip_au = np.array(results['dip'][-1])
            results['energy_field'].append(float(-np.dot(dip_au, efield)))
            results['field'].append(efield)
        else:
            results['energy_field'].append(0.0)
            results['field'].append(np.zeros(3))

        # --- MO populations (projection onto initial-geometry MOs) ---
        if self.record_occ and hasattr(self, 'mo_proj'):
            if self.is_uks:
                pa = self.mo_proj[0].conj().T @ dm_ao[0] @ self.mo_proj[0]
                pb = self.mo_proj[1].conj().T @ dm_ao[1] @ self.mo_proj[1]
                results['occ_alpha'].append(cupy.asnumpy(cupy.diagonal(pa).real))
                results['occ_beta'].append(cupy.asnumpy(cupy.diagonal(pb).real))
            else:
                pm = self.mo_proj.conj().T @ dm_ao @ self.mo_proj
                results['occ'].append(cupy.asnumpy(cupy.diagonal(pm).real))

    def kernel(self, times, dm0=None, dt=0.02, propagator='magnus_interpol',
               callback=None, n_electronic=1):
        log = logger.new_logger(self, self.verbose)
        mol, mf = self.mol, self.ks
        if dm0 is None: dm0 = mf.make_rdm1()
        dm_ao = cupy.asarray(dm0).astype(cupy.complex128)
        dm_orth = self.to_orth(dm_ao)

        dt_e = dt
        dt_n = dt * n_electronic

        coords = mol.atom_coords()
        if self.velocities is None: self.velocities = np.zeros_like(coords)
        if self.frozen: self.velocities *= 0.0

        log.info(f"Ehrenfest MD: dt_e={dt_e}, n_electronic={n_electronic} "
                 f"(dt_n={dt_n:.4g}){' [FROZEN]' if self.frozen else ''}")

        # Initial forces (frozen runs skip the expensive gradient).
        if self.frozen:
            self.forces = np.zeros_like(coords)
        else:
            self.forces, _ = get_ehrenfest_force(self, dm_ao, t=0.0)

        # MO population projection matrix (uses initial-geometry orbitals for
        # Ehrenfest — exact for frozen nuclei, a stable diagnostic basis for
        # moving nuclei).
        if self.record_occ:
            s_mat = cupy.asarray(self.ks.get_ovlp())
            c_mat = cupy.asarray(self.ks.mo_coeff)
            if self.is_uks:
                self.mo_proj = cupy.einsum('pq,sqr->spr', s_mat, c_mat)
            else:
                self.mo_proj = s_mat @ c_mat

        results = {
            'times': [], 'coords': [], 'velocities': [], 'forces': [],
            'energy_elec': [], 'energy_tot': [],
            'energy': [], 'energy_nuc': [], 'energy_core': [],
            'energy_coul': [], 'energy_xc': [], 'energy_field': [],
            'field': [], 'dip': [],
        }
        if self.is_uks:
            results['dip_alpha'] = []
            results['dip_beta'] = []
            if self.record_occ:
                results['occ_alpha'] = []
                results['occ_beta'] = []
        elif self.record_occ:
            results['occ'] = []

        t_now = 0.0
        self._record_md(t_now, dm_ao, coords, results)
        if callback: callback(t_now, dm_ao, results)

        # Sub-step state persisted across t_target iterations
        total_sub = 0
        f_orth_prev = None
        mol_prev_sub = None
        x_mat_prev_sub = None
        v_mid = None
        r_next = None

        for t_target in times:
            if t_target <= t_now + 1e-6: continue

            # Advance by electronic sub-steps until t_now reaches t_target.
            #
            # Multi-scale scheme (static-then-jump): within each nuclear step,
            # the n_electronic electronic sub-steps see a STATIC geometry R(t).
            # Nuclei jump to R(t+dt_n) only at the nuclear boundary. The basis
            # motion is then applied as a single unitary transport of dm_orth
            # (exp(-D_dt)) followed by the Verlet velocity corrector.
            while t_now < t_target - 1e-6:
                j = total_sub % n_electronic

                if j == 0 and not self.frozen:
                    # Verlet predictor; snapshot old mol/X for later D_dt.
                    accel = self.forces / self.masses[:, None]
                    v_mid = self.velocities + 0.5 * accel * dt_n
                    r_next = coords + v_mid * dt_n
                    mol_prev_sub = mol.copy()
                    x_mat_prev_sub = self.x_mat.copy()

                # Electronic sub-step at the CURRENT (static) geometry.
                if j == 0 and propagator == 'magnus_interpol':
                    f_orth_prev = self._build_f_orth(dm_orth, t_now)
                dm_orth = self._electronic_step(dm_orth, t_now, dt_e, propagator,
                                                f_orth_t=f_orth_prev, D_dt=None)
                t_now += dt_e
                total_sub += 1

                if j == n_electronic - 1:
                    if not self.frozen:
                        # Nuclear jump: R(t) -> R(t+dt_n). One set_geom_ call.
                        mol.set_geom_(r_next, unit='Bohr')
                        mf.reset(mol)
                        self._update_basis()
                        # Basis transport across the jump: D_dt_tot is the full
                        # anti-Hermitian basis-motion matrix over the nuclear
                        # step. Apply unitary rotation dm_orth -> Q dm_orth Q†
                        # with Q = exp(-D_dt_tot).
                        D_dt_tot = self._compute_D_dt(mol_prev_sub, x_mat_prev_sub)
                        Q = self._basis_transport_unitary(D_dt_tot)
                        if self.is_uks:
                            dm_orth = Q @ dm_orth @ Q.conj().swapaxes(-1, -2)
                        else:
                            dm_orth = Q @ dm_orth @ Q.conj().T
                        # Forces at new geometry, Verlet corrector.
                        dm_ao = self.to_ao(dm_orth)
                        self.forces, _ = get_ehrenfest_force(self, dm_ao, t=t_now)
                        new_accel = self.forces / self.masses[:, None]
                        self.velocities = v_mid + 0.5 * new_accel * dt_n
                        coords = r_next
                    self._apply_thermostat(dt_n)
                elif propagator == 'magnus_interpol':
                    # Prep F_orth for the next electronic sub-step (same geom).
                    f_orth_prev = self._build_f_orth(dm_orth, t_now)

            # Reached t_target — record. _record_md recomputes the full
            # energy breakdown from dm_ao, so no mid-step cache is needed.
            dm_ao = self.to_ao(dm_orth)
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
        M = cupy.asarray(intor_cross('int1e_ovlp', mol_old, self.mol))
        XMX = x_mat_old.conj().T @ M @ self.x_mat
        return 0.5 * (XMX - XMX.conj().swapaxes(-1, -2))

    def _basis_transport_unitary(self, D_dt):
        '''Q = exp(-D_dt) for anti-Hermitian D_dt. Since iD_dt is Hermitian,
        diagonalize iD_dt = v·diag(e)·v†, then D_dt = -i·v·diag(e)·v†, so
        Q = exp(-D_dt) = v·diag(exp(i·e))·v†.'''
        H = 1j * D_dt  # Hermitian
        e, v = cupy.linalg.eigh(H)
        # Column-wise scale of v by exp(i·e), then matmul with v†
        scaled = v * cupy.exp(1j * e)  # broadcasts along last axis
        return scaled @ v.conj().swapaxes(-1, -2)

    def _build_f_orth(self, dm_orth, t):
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
        f_couple = -1j * D_dt / dt if D_dt is not None else None

        def _apply(u, p):
            return u @ p @ u.conj().swapaxes(-1, -2) if self.is_uks else u @ p @ u.conj().T
        def _add_couple(f):
            return f if f_couple is None else f + f_couple

        if propagator == 'magnus_interpol':
            if f_orth_t is None:
                dm_ao = self.to_ao(dm_orth)
                f_ao_t = self.get_fock(dm_ao, hcore, ints_mu, t, c_k)
                f_orth_t = self.to_orth_fock(f_ao_t)
            dm_orth_next = dm_orth.copy()
            for _ in range(max_iter):
                dm_prev = dm_orth_next
                dm_ao_next = self.to_ao(dm_orth_next)
                f_ao_next = self.get_fock(dm_ao_next, hcore, ints_mu, t + dt, c_k)
                f_orth_next = self.to_orth_fock(f_ao_next)
                u = self.compute_propagator(_add_couple(0.5 * (f_orth_t + f_orth_next)), dt)
                dm_orth_next = _apply(u, dm_orth)
                if cupy.linalg.norm(dm_orth_next - dm_prev) < threshold: break
            return dm_orth_next
        elif propagator == 'magnus_iter':
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
            dm_ao = self.to_ao(dm_orth)
            f_ao_t = self.get_fock(dm_ao, hcore, ints_mu, t, c_k)
            u = self.compute_propagator(_add_couple(self.to_orth_fock(f_ao_t)), dt)
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
        if self.frozen: self.velocities *= 0.0
        
        self.forces = -mf.Gradients().kernel()
        self.energy_elec = mf.e_tot
        results = {'times': [], 'coords': [], 'velocities': [], 'forces': [], 'energy_elec': [], 'energy_tot': []}
        t_now = 0.0
        self._record_md(t_now, dm_ao, coords, results)
        if callback: callback(t_now, dm_ao, results)
        for t_target in times:
            if t_target <= t_now + 1e-6: continue
            steps = int(np.floor((t_target - t_now) / dt + 1e-6))
            if steps == 0: continue
            for step in range(steps):
                if not self.frozen:
                    accel = self.forces / self.masses[:, None]
                    self.velocities += 0.5 * accel * dt
                    coords += self.velocities * dt
                    mol.set_geom_(coords, unit='Bohr')
                    mf.reset(mol)
                mf.kernel() 
                self.energy_elec = mf.e_tot
                new_forces = -mf.Gradients().kernel()
                if not self.frozen:
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
