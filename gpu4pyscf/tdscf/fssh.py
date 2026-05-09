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
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.tdscf.ehrenfest import BOMD
from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker

def _as_numpy(x):
    if isinstance(x, (list, tuple)):
        return tuple(_as_numpy(i) for i in x)
    if isinstance(x, cupy.ndarray):
        return x.get()
    return np.asarray(x)

class FSSHMD(BOMD):
    '''Fewest-Switches Surface Hopping Molecular Dynamics.
    
    References:
        Tully, J. C. J. Chem. Phys. 1990, 93, 1061.
        Song, H. et al. J. Chem. Theory Comput. 2020, 16, 6418.
        Araujo, J. J. et al. J. Chem. Theory Comput. 2024, 20, 10.1021/acs.jctc.4c00411.
    '''
    def __init__(self, ks, td=None, state=0, scheme='fssh', decoherence='idc', **kwargs):
        # state=0: ground state, state=1..N: excited states
        self.seed = kwargs.pop('seed', None)
        super().__init__(ks, td=td, state=state, **kwargs)
        self.scheme = scheme.lower()
        if self.scheme not in ['fssh', 'fssh2']:
            raise ValueError(f"Unknown scheme {self.scheme}. Choose 'fssh' or 'fssh2'.")
        self.nstates_tot = td.nstates + 1
        self._rng = np.random.default_rng(self.seed)
        
        self.coefficients = np.zeros(self.nstates_tot, dtype=np.complex128)
        self.coefficients[self.state] = 1.0
        
        # Seeding: add small population to all states to help fssh2 growth
        self.coefficients += (self._rng.standard_normal(self.nstates_tot) + 1j * self._rng.standard_normal(self.nstates_tot)) * 1e-6
        self.coefficients /= np.linalg.norm(self.coefficients)
        
        self._prev_coefficients = self.coefficients.copy()
        self.decoherence = decoherence
        
        # Internal state for NACT calculation
        self._prev_td_xy = None
        self._prev_mo_coeff = None
        self._prev_mo_occ = None
        self._prev_mol = None
        self._prev_energies = None

        self._keys.update({'scheme', 'nstates_tot', 'coefficients', '_prev_coefficients', 'decoherence', 'seed'})

    def _propagate_coefficients(self, dt, energies, prev_energies, overlap, nact):
        '''Propagate quantum coefficients.'''
        if self.scheme == 'fssh2':
            # Symmetric Trotter splitting: exp(-i E_new dt/2) @ T @ exp(-i E_old dt/2)
            # T is the Lowdin-orthogonalized time-overlap (polar decomposition unitary).
            u_svd, _, vh_svd = np.linalg.svd(overlap)
            overlap_orth = u_svd @ vh_svd
            phase_new = np.exp(-0.5j * energies * dt)
            phase_old = np.exp(-0.5j * prev_energies * dt)
            u = (phase_new[:, None] * overlap_orth) * phase_old[None, :]
            self.coefficients = u @ self.coefficients
        else:
            h_eff = np.diag(energies) - 1j * nact
            w, v = np.linalg.eigh(h_eff)
            u = v @ np.diag(np.exp(-1j * w * dt)) @ v.conj().T
            self.coefficients = u @ self.coefficients

    def _get_overlap_and_nact(self, td, dt):
        '''Compute time-overlap matrix and numerical/analytical NACT.'''
        if self._prev_td_xy is None:
            return np.eye(self.nstates_tot), np.zeros((self.nstates_tot, self.nstates_tot))

        from pyscf import gto
        s_mix = np.asarray(gto.intor_cross('int1e_ovlp', td.mol, self._prev_mol))
        
        mo_occ = td._scf.mo_occ
        is_uks = (hasattr(mo_occ, 'ndim') and mo_occ.ndim == 2) or \
                 (isinstance(mo_occ, (tuple, list)) and len(mo_occ) == 2)
        
        if is_uks:
            curr_mo_a, curr_mo_b = td._scf.mo_coeff
            prev_mo_a, prev_mo_b = self._prev_mo_coeff
            curr_occ_a, curr_occ_b = td._scf.mo_occ
            prev_occ_a, prev_occ_b = self._prev_mo_occ
            
            nocc_a = int(np.sum(prev_occ_a > 0))
            nocc_b = int(np.sum(prev_occ_b > 0))
            
            u_a = _as_numpy(curr_mo_a).T @ s_mix @ _as_numpy(prev_mo_a)
            u_b = _as_numpy(curr_mo_b).T @ s_mix @ _as_numpy(prev_mo_b)
            
            u_oo_a, u_vv_a = u_a[:nocc_a, :nocc_a], u_a[nocc_a:, nocc_a:]
            u_oo_b, u_vv_b = u_b[:nocc_b, :nocc_b], u_b[nocc_b:, nocc_b:]
            u_ov_a, u_vo_a = u_a[:nocc_a, nocc_a:], u_a[nocc_a:, :nocc_a]
            u_ov_b, u_vo_b = u_b[:nocc_b, nocc_b:], u_b[nocc_b:, :nocc_b]
            
            det_uo_a, det_uo_b = np.linalg.det(u_oo_a), np.linalg.det(u_oo_b)
            det_uo = det_uo_a * det_uo_b
            
            s_mat = np.zeros((self.nstates_tot, self.nstates_tot))
            s_mat[0, 0] = det_uo
            
            for j in range(self.td.nstates):
                (xja, xjb), _ = self._prev_td_xy[j]
                xja, xjb = _as_numpy(xja).reshape(nocc_a, -1), _as_numpy(xjb).reshape(nocc_b, -1)
                
                # S_0j = <Phi_0(t+dt) | Psi_j(t)>
                s_mat[0, j+1] = det_uo_b * np.linalg.det(u_oo_a) * np.sum(u_ov_a * xja) + \
                               det_uo_a * np.linalg.det(u_oo_b) * np.sum(u_ov_b * xjb)
                
                xja_aligned = u_oo_a @ xja @ u_vv_a.T
                xjb_aligned = u_oo_b @ xjb @ u_vv_b.T
                
                for k in range(self.td.nstates):
                    (xka, xkb), _ = td.xy[k]
                    xka, xkb = _as_numpy(xka).reshape(nocc_a, -1), _as_numpy(xkb).reshape(nocc_b, -1)
                    
                    if j == 0:
                        # S_k0 = <Psi_k(t+dt) | Phi_0(t)>
                        s_mat[k+1, 0] = det_uo_b * np.linalg.det(u_oo_a) * np.sum(xka * u_vo_a.T) + \
                                       det_uo_a * np.linalg.det(u_oo_b) * np.sum(xkb * u_vo_b.T)
                    
                    # S_kj = sum over spin channels
                    s_mat[k+1, j+1] = det_uo_b * np.linalg.det(u_oo_a) * np.sum(xka * xja_aligned) + \
                                     det_uo_a * np.linalg.det(u_oo_b) * np.sum(xkb * xjb_aligned)
        else:
            curr_mo = _as_numpy(td._scf.mo_coeff)
            prev_mo = self._prev_mo_coeff
            prev_occ = self._prev_mo_occ
            nocc = int(np.sum(prev_occ > 0))
            u = curr_mo.T @ s_mix @ prev_mo
            u_oo, u_vv = u[:nocc, :nocc], u[nocc:, nocc:]
            u_ov, u_vo = u[:nocc, nocc:], u[nocc:, :nocc]
            
            s_mat = np.zeros((self.nstates_tot, self.nstates_tot))
            det_uo = np.linalg.det(u_oo)
            s_mat[0, 0] = det_uo**2
            
            for j in range(self.td.nstates):
                xj_ref, _ = self._prev_td_xy[j]
                xj_ref = _as_numpy(xj_ref).reshape(nocc, -1)
                s_mat[0, j+1] = np.sqrt(2.0) * det_uo * np.sum(u_ov * xj_ref)
                
                xj_aligned = u_oo @ xj_ref @ u_vv.T
                for k in range(self.td.nstates):
                    xk_disp, _ = td.xy[k]
                    xk_disp = _as_numpy(xk_disp).reshape(nocc, -1)
                    if j == 0:
                        s_mat[k+1, 0] = np.sqrt(2.0) * det_uo * np.sum(xk_disp * u_vo.T)
                    s_mat[k+1, j+1] = det_uo * np.sum(xk_disp * xj_aligned) * 2.0
        
        for i in range(self.nstates_tot):
            if s_mat[i, i] < 0: s_mat[i, :] *= -1.0
                
        # Numerical NACT: A_kj = <psi_k|d/dt|psi_j> = (S_jk - S_kj) / (2*dt)
        nact = (s_mat.T - s_mat) / (2 * dt)
        
        # Standard FSSH requires higher accuracy for NACT scale.
        if self.scheme == 'fssh':
            for m in range(self.nstates_tot):
                for n in range(m + 1, self.nstates_tot):
                    try:
                        nacr = self._get_nacr(m, n)
                        dot_v = np.sum(self.velocities * nacr)
                        nact[m, n] = dot_v
                        nact[n, m] = -dot_v
                    except Exception:
                        pass # keep numerical
        return s_mat, nact

    def _get_nacr(self, old_state, new_state):
        is_uks = hasattr(self.td._scf, 'mo_occ') and isinstance(self.td._scf.mo_occ, (tuple, list))
        if is_uks:
            from gpu4pyscf.nac.tduks import NAC
        else:
            from gpu4pyscf.nac.tdrks import NAC
        
        nac_obj = NAC(self.td)
        if old_state == 0 or new_state == 0:
            idx = max(old_state, new_state) - 1
            de, _, _, _ = nac_obj.get_nacv_ge(self.td.xy[idx], self.td.e[idx], singlet=True)
            return -de if new_state == 0 else de
        
        idx_i, idx_j = old_state - 1, new_state - 1
        de, _, _, _ = nac_obj.get_nacv_ee(self.td.xy[idx_i], self.td.xy[idx_j], 
                                          self.td.e[idx_i], self.td.e[idx_j], singlet=True)
        return de

    def _velocity_rescale(self, old_state, new_state, energies):
        if old_state == new_state: return True
        try:
            nacr = self._get_nacr(old_state, new_state)
        except Exception:
            nacr = self._rng.standard_normal(self.velocities.shape)
        de = energies[new_state] - energies[old_state]
        m_inv = 1.0 / self.masses
        a = 0.5 * np.sum(m_inv[:, None] * nacr**2)
        b = np.sum(self.velocities * nacr)
        det = b**2 - 4 * a * de
        if det < 0: return False
        gamma = (-b + np.sqrt(det)) / (2 * a) if b < 0 else (-b - np.sqrt(det)) / (2 * a)
        self.velocities += gamma * m_inv[:, None] * nacr
        return True

    def _check_hopping(self, dt, energies, nact):
        n = self.state
        probs = np.zeros(self.nstates_tot)
        ann = max(1e-10, np.abs(self.coefficients[n])**2)
        if self.scheme == 'fssh2':
            prev_ann = max(1e-10, np.abs(self._prev_coefficients[n])**2)
            pm_out = max(0.0, (prev_ann - ann) / prev_ann)
            for m in range(self.nstates_tot):
                if m == n: continue
                rho_m_new = np.abs(self.coefficients[m])**2
                rho_m_old = np.abs(self._prev_coefficients[m])**2
                probs[m] = max(0.0, min(pm_out, (rho_m_new - rho_m_old) / prev_ann))
            sum_p = np.sum(probs)
            if sum_p > pm_out and sum_p > 1e-12: probs *= (pm_out / sum_p)
        else:
            for m in range(self.nstates_tot):
                if m == n: continue
                g = dt * 2 * np.real(self.coefficients[n].conj() * self.coefficients[m] * nact[m, n]) / ann
                probs[m] = max(0, g)
        cum_probs = np.cumsum(probs)
        r = self._rng.random()
        for m, cp in enumerate(cum_probs):
            if r < cp: return m
        return self.state

    def _apply_decoherence(self):
        if self.decoherence == 'idc':
            self.coefficients[:] = 0.0
            self.coefficients[self.state] = 1.0

    def kernel(self, times, dt=0.02, callback=None):
        log = logger.new_logger(self, self.verbose)
        mol, mf = self.mol, self.ks
        scanner = self._build_scanner()
        coords = mol.atom_coords()
        if self.velocities is None: self.velocities = np.zeros_like(coords)
        tracker = TransitionDensityTracker(self.td, state_ref=self.state)
        e_tot, de = scanner(mol)
        self.forces = -np.asarray(de)
        self.energy_elec = float(e_tot)
        def _copy_obj(x):
            if isinstance(x, (list, tuple)):
                return tuple(_copy_obj(i) for i in x)
            if hasattr(x, 'copy'):
                return x.copy()
            return x

        self._prev_td_xy = [ (_copy_obj(_as_numpy(x)), _copy_obj(_as_numpy(y))) for x, y in self.td.xy ]
        self._prev_mo_coeff = _copy_obj(_as_numpy(mf.mo_coeff))
        self._prev_mo_occ = _copy_obj(_as_numpy(mf.mo_occ))
        self._prev_mol = mol.copy()
        self._prev_energies = np.concatenate([[float(self.ks.e_tot)], float(self.ks.e_tot) + np.asarray(self.td.e)])
        results = {'times': [], 'coords': [], 'velocities': [], 'forces': [],
                   'energy_elec': [], 'energy_tot': [], 'state_history': [],
                   'coefficients': []}
        t_now = 0.0
        results['times'].append(t_now)
        results['coords'].append(coords.copy())
        results['velocities'].append(self.velocities.copy())
        results['forces'].append(self.forces.copy())
        results['energy_elec'].append(self.energy_elec)
        results['energy_tot'].append(self.energy_elec + 0.5 * np.sum(self.masses[:, None] * self.velocities**2))
        results['state_history'].append(self.state)
        results['coefficients'].append(self.coefficients.copy())
        if callback: callback(t_now, None, results)
        for t_target in times:
            if t_target <= t_now + 1e-6: continue
            steps = int(np.floor((t_target - t_now) / dt + 1e-6))
            for step in range(steps):
                accel = self.forces / self.masses[:, None]
                self.velocities += 0.5 * accel * dt
                coords = coords + self.velocities * dt
                mol.set_geom_(coords, unit='Bohr')
                e_tot, de = scanner(mol)
                self.energy_elec = float(e_tot)
                new_forces = -np.asarray(de)
                self.velocities += 0.5 * (new_forces / self.masses[:, None]) * dt
                self.forces = new_forces
                if hasattr(scanner, 'base'):
                    self.td = getattr(scanner.base, 'base', self.td)
                else:
                    self.td = scanner
                self.ks = getattr(self.td, '_scf', self.ks)
                tracker.assign(self.td, require_converged=False)
                tracker.re_anchor(self.td, state_ref=self.state)
                total_energies = np.concatenate([[float(self.ks.e_tot)], float(self.ks.e_tot) + np.asarray(self.td.e)])
                overlap, nact = self._get_overlap_and_nact(self.td, dt)
                self._prev_coefficients = self.coefficients.copy()
                self._propagate_coefficients(dt, total_energies, self._prev_energies, overlap, nact)
                self._prev_energies = total_energies
                old_state = self.state
                new_state = self._check_hopping(dt, total_energies, nact)
                if new_state != old_state:
                    if self._velocity_rescale(old_state, new_state, total_energies):
                        log.note(f"FSSH Hop: {old_state} -> {new_state} at t={t_now+dt:.4f}")
                        self.state = new_state
                        self._apply_decoherence()
                        self._scanner = None
                        scanner = self._build_scanner()
                        e_tot, de = scanner(mol)
                        self.energy_elec = float(e_tot)
                        self.forces = -np.asarray(de)
                def _copy_obj(x):
                    if isinstance(x, (list, tuple)):
                        return tuple(_copy_obj(i) for i in x)
                    if hasattr(x, 'copy'):
                        return x.copy()
                    return x

                self._prev_td_xy = [ (_copy_obj(_as_numpy(x)), _copy_obj(_as_numpy(y))) for x, y in self.td.xy ]
                self._prev_mo_coeff = _copy_obj(_as_numpy(mf.mo_coeff))
                self._prev_mo_occ = _copy_obj(_as_numpy(mf.mo_occ))
                self._prev_mol = mol.copy()
                t_now += dt
            results['times'].append(t_now)
            results['coords'].append(coords.copy())
            results['velocities'].append(self.velocities.copy())
            results['forces'].append(self.forces.copy())
            results['energy_elec'].append(self.energy_elec)
            results['energy_tot'].append(self.energy_elec + 0.5 * np.sum(self.masses[:, None] * self.velocities**2))
            results['state_history'].append(self.state)
            results['coefficients'].append(self.coefficients.copy())
            if callback: callback(t_now, None, results)
            log.info(f"Time: {t_now:10.4f} au | Energy: {results['energy_tot'][-1]:20.12f} | State: {self.state}")
        return results
