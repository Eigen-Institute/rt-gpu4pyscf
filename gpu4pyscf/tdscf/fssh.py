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
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf.ehrenfest import BOMD
from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker

class FSSHMD(BOMD):
    '''Fewest-Switches Surface Hopping Molecular Dynamics.
    
    References:
        Tully, J. C. J. Chem. Phys. 1990, 93, 1061.
        Song, H. et al. J. Chem. Theory Comput. 2020, 16, 6418.
    '''
    def __init__(self, ks, td=None, state=0, decoherence='idc', **kwargs):
        # state=0: ground state, state=1..N: excited states
        super().__init__(ks, td=td, state=state, **kwargs)
        self.nstates_tot = td.nstates + 1
        self.coefficients = np.zeros(self.nstates_tot, dtype=np.complex128)
        self.coefficients[self.state] = 1.0
        self.decoherence = decoherence
        self.seed = kwargs.get('seed', None)
        self._rng = np.random.default_rng(self.seed)
        
        # Internal state for NACT calculation
        self._prev_td_xy = None
        self._prev_mo_coeff = None
        self._prev_mo_occ = None
        self._prev_mol = None
        
        self._keys.update({'nstates_tot', 'coefficients', 'decoherence', 'seed'})

    def _propagate_coefficients(self, dt, energies, nact):
        '''Propagate quantum coefficients using Eq. (5) of Song2020.
        
        Using a simple unitary propagator for one step:
        c(t+dt) = exp(-i/hbar * (E - i hbar A) * dt) c(t)
        '''
        # Hamiltonian in adiabatic basis: H_nm = E_n delta_nm - i hbar A_nm
        # Here A_nm is the NACT <n|d/dt|m>.
        # hbar = 1 in atomic units.
        h_eff = np.diag(energies) - 1j * nact
        w, v = np.linalg.eigh(h_eff)
        u = v @ np.diag(np.exp(-1j * w * dt)) @ v.conj().T
        self.coefficients = u @ self.coefficients

    def _get_nact(self, td, dt):
        '''Compute numerical NACT via transition density overlap.
        A_nm = <Psi_n(t) | d/dt Psi_m(t)> \approx (S_nm(t, t+dt) - S_mn(t, t+dt)) / (2*dt)
        '''
        if self._prev_td_xy is None:
            return np.zeros((self.nstates_tot, self.nstates_tot))

        # Cross-overlap of AOs
        from pyscf import gto
        s_mix = np.asarray(gto.intor_cross('int1e_ovlp', td.mol, self._prev_mol))
        
        # MO alignment
        curr_mo = np.asarray(td._scf.mo_coeff)
        prev_mo = self._prev_mo_coeff
        curr_occ = np.asarray(td._scf.mo_occ)
        prev_occ = self._prev_mo_occ
        
        # For simplicity, assume RKS/TDA for now.
        nocc = int(np.sum(prev_occ > 0))
        u = curr_mo.T @ s_mix @ prev_mo
        u_o = u[:nocc, :nocc]
        u_v = u[nocc:, nocc:]
        
        # s_mat includes ground state (index 0)
        s_mat = np.zeros((self.nstates_tot, self.nstates_tot))
        
        # Ground-ground overlap
        s_mat[0, 0] = np.trace(u_o.T @ u_o) / nocc # normalized overlap of ground determinant
        
        for j in range(self.td.nstates):
            # Reference J at t (excited states)
            xj_ref, _ = self._prev_td_xy[j]
            xj_ref = np.asarray(xj_ref).reshape(nocc, -1)
            norm_j = np.linalg.norm(xj_ref)
            if norm_j > 1e-10: xj_ref /= norm_j
            
            # Aligned reference in current MO space
            xj_aligned = u_o @ xj_ref @ u_v.T
            
            for k in range(self.td.nstates):
                # Displaced K at t+dt
                xk_disp, _ = td.xy[k]
                xk_disp = np.asarray(xk_disp).reshape(nocc, -1)
                norm_k = np.linalg.norm(xk_disp)
                if norm_k > 1e-10: xk_disp /= norm_k
                
                s_mat[k+1, j+1] = np.sum(xk_disp * xj_aligned)
                
        # Numerical NACT: A = (S - S.T) / (2*dt)
        nact = (s_mat - s_mat.T) / (2 * dt)
        return nact

    def _get_nacr(self, old_state, new_state):
        '''Calculate Non-Adiabatic Coupling Vector (NACR) between two states.'''
        # old_state, new_state are 0-indexed.
        
        # Build NAC object
        from gpu4pyscf.nac.tdrks import NAC
        nac_obj = NAC(self.td)
        
        # Ground state (0) to excited state (I > 0)
        if old_state == 0 or new_state == 0:
            idx = max(old_state, new_state) - 1 # 0-indexed in td.xy
            x_y = self.td.xy[idx]
            energy = self.td.e[idx]
            de, _, _, _ = nac_obj.get_nacv_ge(x_y, energy, singlet=True)
            if new_state == 0: de = -de # sign convention
            return de
            
        # Excited state (I) to excited state (J)
        idx_i = old_state - 1
        idx_j = new_state - 1
        x_y_i = self.td.xy[idx_i]
        x_y_j = self.td.xy[idx_j]
        e_i = self.td.e[idx_i]
        e_j = self.td.e[idx_j]
        
        de, _, _, _ = nac_obj.get_nacv_ee(x_y_i, x_y_j, e_i, e_j, singlet=True)
        return de

    def _velocity_rescale(self, old_state, new_state, energies):
        '''Rescale velocity along NACR to conserve energy.
        Returns True if hop is successful, False if classically forbidden.
        '''
        if old_state == new_state: return True
        
        nacr = self._get_nacr(old_state, new_state) # (natm, 3)
        de = energies[new_state] - energies[old_state]
        
        m_inv = 1.0 / self.masses # (natm,)
        a = 0.5 * np.sum(m_inv[:, None] * nacr**2)
        b = np.sum(self.velocities * nacr)
        c = de
        
        det = b**2 - 4 * a * c
        if det < 0:
            return False # Hop rejected
            
        if b < 0:
            gamma = (-b + np.sqrt(det)) / (2 * a)
        else:
            gamma = (-b - np.sqrt(det)) / (2 * a)
            
        self.velocities += gamma * m_inv[:, None] * nacr
        return True

    def _check_hopping(self, dt, energies, nact):
        '''Fewest-Switches Surface Hopping probability and execution.'''
        n = self.state # current state 0-indexed
        probs = np.zeros(self.nstates_tot)
        
        # a_nn = |c_n|^2
        ann = np.abs(self.coefficients[n])**2
        if ann < 1e-10: ann = 1e-10
        
        for m in range(self.nstates_tot):
            if m == n: continue
            # Eq. (11)-(12) of Song2020:
            # g = dt * 2 * Re(c_n* c_m A_mn) / a_nn
            g = dt * 2 * np.real(self.coefficients[n].conj() * self.coefficients[m] * nact[m, n]) / ann
            probs[m] = max(0, g)
            
        cum_probs = np.cumsum(probs)
        r = self._rng.random()
        
        for m, cp in enumerate(cum_probs):
            if r < cp:
                return m # new state 0-indexed
        return self.state

    def _apply_decoherence(self):
        if self.decoherence == 'idc':
            # IDC: reset coefficients to 1.0 for active state, 0 elsewhere
            self.coefficients[:] = 0.0
            self.coefficients[self.state] = 1.0

    def kernel(self, times, dt=0.02, callback=None):
        log = logger.new_logger(self, self.verbose)
        mol, mf = self.mol, self.ks
        scanner = self._build_scanner()
        
        coords = mol.atom_coords()
        if self.velocities is None: self.velocities = np.zeros_like(coords)
        
        # Initial step
        e_tot, de = scanner(mol)
        self.forces = -np.asarray(de)
        self.energy_elec = float(e_tot)
        
        # Cache for NACT
        self._prev_td_xy = [ (np.asarray(x).copy(), np.asarray(y).copy()) for x, y in self.td.xy ]
        self._prev_mo_coeff = np.asarray(mf.mo_coeff).copy()
        self._prev_mo_occ = np.asarray(mf.mo_occ).copy()
        self._prev_mol = mol.copy()
        
        results = {'times': [], 'coords': [], 'velocities': [], 'forces': [],
                   'energy_elec': [], 'energy_tot': [], 'state_history': [],
                   'coefficients': []}
        
        t_now = 0.0
        
        # Record initial state
        results['times'].append(t_now)
        results['coords'].append(coords.copy())
        results['velocities'].append(self.velocities.copy())
        results['forces'].append(self.forces.copy())
        results['energy_elec'].append(self.energy_elec)
        e_kin = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
        results['energy_tot'].append(self.energy_elec + e_kin)
        results['state_history'].append(self.state)
        results['coefficients'].append(self.coefficients.copy())
        
        if callback: callback(t_now, None, results)
        
        for t_target in times:
            if t_target <= t_now + 1e-6: continue
            steps = int(np.floor((t_target - t_now) / dt + 1e-6))
            for step in range(steps):
                # 1. Nuclear step (Verlet predictor)
                accel = self.forces / self.masses[:, None]
                self.velocities += 0.5 * accel * dt
                coords = coords + self.velocities * dt
                mol.set_geom_(coords, unit='Bohr')
                
                # 2. Electronic step
                e_tot, de = scanner(mol)
                self.energy_elec = float(e_tot)
                new_forces = -np.asarray(de)
                new_accel = new_forces / self.masses[:, None]
                self.velocities += 0.5 * new_accel * dt
                self.forces = new_forces
                
                # 3. Surface Hopping logic
                energies = np.asarray(self.td.e) # excitation energies
                mf_e = float(self.ks.e_tot)
                total_energies = np.concatenate([[mf_e], mf_e + energies])
                
                # FSSH needs nstates+1 because it includes ground state
                nact = self._get_nact(self.td, dt)
                self._propagate_coefficients(dt, total_energies, nact)
                
                old_state = self.state
                new_state = self._check_hopping(dt, total_energies, nact)
                
                if new_state != old_state:
                    # Attempt hop
                    success = self._velocity_rescale(old_state, new_state, total_energies)
                    if success:
                        log.note(f"FSSH Hop: {old_state} -> {new_state} at t={t_now+dt:.4f}")
                        self.state = new_state
                        self._apply_decoherence()
                        # Update scanner and forces for new state
                        self._scanner = None
                        scanner = self._build_scanner()
                        e_tot, de = scanner(mol)
                        self.energy_elec = float(e_tot)
                        self.forces = -np.asarray(de)
                    else:
                        log.note(f"FSSH Forbidden Hop: {old_state} -> {new_state} at t={t_now+dt:.4f}")
                
                # Cache for next NACT
                self._prev_td_xy = [ (np.asarray(x).copy(), np.asarray(y).copy()) for x, y in self.td.xy ]
                self._prev_mo_coeff = np.asarray(mf.mo_coeff).copy()
                self._prev_mo_occ = np.asarray(mf.mo_occ).copy()
                self._prev_mol = mol.copy()
                
                t_now += dt
                
            # Recording...
            results['times'].append(t_now)
            results['coords'].append(coords.copy())
            results['velocities'].append(self.velocities.copy())
            results['forces'].append(self.forces.copy())
            results['energy_elec'].append(self.energy_elec)
            e_kin = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)
            results['energy_tot'].append(self.energy_elec + e_kin)
            results['state_history'].append(self.state)
            results['coefficients'].append(self.coefficients.copy())
            
            if callback: callback(t_now, None, results)
            log.info(f"Time: {t_now:10.4f} au | Energy: {results['energy_tot'][-1]:20.12f} | State: {self.state}")
            
        return results
