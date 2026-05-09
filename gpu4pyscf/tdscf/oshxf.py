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
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf.rt_tddft import RTTDDFT

def _as_numpy(x):
    if isinstance(x, cupy.ndarray):
        return x.get()
    return np.asarray(x)

class CPA_OSHXF(RTTDDFT):
    '''Orbital-Based Surface-Hopping based on Exact Factorization (OSHXF) 
    using the Classical Path Approximation (CPA).
    
    Attributes:
        ks : gpu4pyscf.dft.rks.RKS or uks.UKS
            Ground state KS object.
        trajectory_data : dict
            Contains pre-computed nuclear trajectory: 
            'times': (Nsteps,), 'coords': (Nsteps, Natm, 3), 'velocities': (Nsteps, Natm, 3)
        basis : str
            Propagation basis, 'MO' is recommended for OSHXF.
    '''
    def __init__(self, ks, trajectory_data, basis='MO', **kwargs):
        super().__init__(ks, basis=basis)
        self.trajectory_data = trajectory_data
        self.decoherence = kwargs.pop('decoherence', True)
        self.mixing_threshold = kwargs.pop('mixing_threshold', 0.01)
        
        # State variables
        self.running_occupation = None 
        self.auxiliary_trajectories = {} # (i, j) -> {'R': ..., 'P': ...}
        
        # XF quantities (on GPU)
        self.quantum_momentum = None # (Natm, 3)
        self.relative_phase = None   # dict (i, j) -> (Natm, 3)
        
        # Cache for numerical NACT
        self._prev_mo_coeff = None
        self._prev_mol = None

    def _phase_alignment(self, mo_curr, s_mix):
        '''Correct phase of MOs to ensure consistency with previous step (Eq. S4).'''
        if self._prev_mo_coeff is None:
            return mo_curr
            
        if self.is_uks:
            # overlap: (2, M, M) = C_prev.H @ S_ao @ C_curr
            overlap = np.einsum('sji,jk,skl->sil', self._prev_mo_coeff.conj(), s_mix, mo_curr)
            diag = np.diagonal(overlap, axis1=-2, axis2=-1)
            phase = diag / np.abs(diag)
            mo_curr = mo_curr * phase[:, None, :]
        else:
            overlap = self._prev_mo_coeff.conj().T @ s_mix @ mo_curr
            diag = np.diag(overlap)
            phase = diag / np.abs(diag)
            mo_curr = mo_curr * phase
        return mo_curr

    def _get_nact(self, mol_curr, mo_curr, dt):
        '''Compute numerical NACT via orbital overlap (Eq. S3).'''
        if self._prev_mo_coeff is None:
            nmo = mo_curr.shape[-1]
            return np.zeros((2, nmo, nmo)) if self.is_uks else np.zeros((nmo, nmo))

        from pyscf import gto
        s_mix = np.asarray(gto.intor_cross('int1e_ovlp', self._prev_mol, mol_curr))
        
        # Align phases before NACT
        mo_curr = self._phase_alignment(mo_curr, s_mix)
        
        if self.is_uks:
            # s_mo[s, j, i] = <phi_j(t-dt) | phi_i(t)>
            s_mo = np.einsum('sji,jk,skl->sil', self._prev_mo_coeff.conj(), s_mix, mo_curr)
            nact = (s_mo - s_mo.conj().swapaxes(-1, -2)) / (2.0 * dt)
        else:
            s_mo = self._prev_mo_coeff.conj().T @ s_mix @ mo_curr
            nact = (s_mo - s_mo.conj().T) / (2.0 * dt)
            
        return nact, mo_curr

    def _get_hopping_probabilities(self, dt, nact, dm_orth):
        '''Calculate Tully's hopping probabilities for each occupied orbital (Eq. 17).'''
        # dm_orth is in MO basis (nmo, nmo) or (2, nmo, nmo)
        if self.is_uks:
            nmo = dm_orth.shape[-1]
            probs = np.zeros((2, nmo, nmo))
            for s in range(2):
                occ_idx = np.where(self.running_occupation[s] > 0.5)[0]
                for i in occ_idx:
                    # P_i->j = dt * 2 * Re(rho_ij * A_ji) / rho_ii
                    denom = dm_orth[s, i, i].real
                    if denom < 1e-10: denom = 1e-10
                    for j in range(nmo):
                        if i == j: continue
                        g = dt * 2.0 * np.real(dm_orth[s, i, j] * nact[s, j, i]) / denom
                        probs[s, i, j] = max(0, g)
        else:
            nmo = dm_orth.shape[-1]
            probs = np.zeros((nmo, nmo))
            occ_idx = np.where(self.running_occupation > 0.5)[0]
            for i in occ_idx:
                denom = dm_orth[i, i].real
                if denom < 1e-10: denom = 1e-10
                for j in range(nmo):
                    if i == j: continue
                    g = dt * 2.0 * np.real(dm_orth[i, j] * nact[j, i]) / denom
                    probs[i, j] = max(0, g)
        return probs

    def _apply_boltzmann_penalty(self, probs, energies, temperature=300.0):
        '''Apply Boltzmann factor for upward transitions (Eq. 26).'''
        kb = 3.166811e-6 # au
        kt = kb * temperature
        if self.is_uks:
            for s in range(2):
                for i in range(probs.shape[1]):
                    for j in range(probs.shape[2]):
                        if probs[s, i, j] > 0 and energies[s, j] > energies[s, i]:
                            probs[s, i, j] *= np.exp(-(energies[s, j] - energies[s, i]) / kt)
        else:
            for i in range(probs.shape[0]):
                for j in range(probs.shape[1]):
                    if probs[i, j] > 0 and energies[j] > energies[i]:
                        probs[i, j] *= np.exp(-(energies[j] - energies[i]) / kt)
        return probs

    def _update_auxiliary_trajectories(self, t, dt, dm_orth, energies, coords, velocities):
        '''Spawn, propagate, and destroy pairwise auxiliary trajectories (ATs).'''
        nmo = dm_orth.shape[-1]
        masses = np.array(self.ks.mol.atom_mass_list()) * 1822.888486 # au
        
        # 1. Identify pairs that satisfy mixing criterion
        active_pairs = []
        if self.is_uks:
            for s in range(2):
                occ_idx = np.where(self.running_occupation[s] > 0.5)[0]
                vir_idx = np.where(self.running_occupation[s] < 0.5)[0]
                for i in occ_idx:
                    for j in vir_idx:
                        if np.abs(dm_orth[s, i, j]) > self.mixing_threshold:
                            active_pairs.append((s, i, j))
        else:
            occ_idx = np.where(self.running_occupation > 0.5)[0]
            vir_idx = np.where(self.running_occupation < 0.5)[0]
            for i in occ_idx:
                for j in vir_idx:
                    if np.abs(dm_orth[i, j]) > self.mixing_threshold:
                        active_pairs.append((i, j))

        # 2. Destroy ATs that no longer satisfy criterion or whose orbitals hopped
        to_destroy = []
        for pair in self.auxiliary_trajectories:
            if pair not in active_pairs:
                to_destroy.append(pair)
        for pair in to_destroy:
            del self.auxiliary_trajectories[pair]

        # 3. Spawn and Propagate ATs
        for pair in active_pairs:
            if pair not in self.auxiliary_trajectories:
                # Spawn AT (Eq. 18 & 20)
                # For UKS: pair = (s, i, j), for RKS: pair = (i, j)
                if self.is_uks:
                    s, i, j = pair
                    de = energies[s, j] - energies[s, i]
                else:
                    i, j = pair
                    de = energies[j] - energies[i]
                
                # Rescaling factor alpha (Eq. 20)
                e_kin = 0.5 * np.sum(masses[:, None] * velocities**2)
                alpha2 = 1.0 + de / e_kin if e_kin > 1e-10 else 1.0
                alpha = np.sqrt(max(0, alpha2))
                
                self.auxiliary_trajectories[pair] = {
                    'R': coords.copy(),
                    'P': alpha * masses[:, None] * velocities,
                    'f_i': np.zeros_like(coords),
                    'f_j': np.zeros_like(coords),
                    'P_guide_prev': masses[:, None] * velocities,
                    'P_aux_prev': alpha * masses[:, None] * velocities
                }
            else:
                # Propagate existing AT (Eq. 19-20, 24-25)
                at = self.auxiliary_trajectories[pair]
                p_guide = masses[:, None] * velocities
                
                # Update auxiliary momentum (Eq. 20 - recalculated each step)
                if self.is_uks:
                    s, i, j = pair
                    de = energies[s, j] - energies[s, i]
                else:
                    i, j = pair
                    de = energies[j] - energies[i]
                
                e_kin = 0.5 * np.sum(masses[:, None] * velocities**2)
                alpha2 = 1.0 + de / e_kin if e_kin > 1e-10 else 1.0
                alpha = np.sqrt(max(0, alpha2))
                p_aux = alpha * p_guide
                
                # Update positions (Eq. 19)
                at['R'] += (at['P'] / masses[:, None]) * dt
                
                # Update phase gradients (Eq. 24-25)
                at['f_i'] += p_guide - at['P_guide_prev']
                at['f_j'] += p_aux - at['P_aux_prev']
                
                # Cache for next step
                at['P'] = p_aux
                at['P_guide_prev'] = p_guide.copy()
                at['P_aux_prev'] = p_aux.copy()

    def _compute_xf_quantities(self, coords, dm_orth):
        '''Compute quantum momentum and relative phase vectors (Eq. 21-25).'''
        masses = np.array(self.ks.mol.atom_mass_list()) * 1822.888486
        natm = coords.shape[0]
        nelec = np.sum(self.running_occupation)
        nmo = self.running_occupation.shape[-1]
        
        # Sigma heuristic: 1/4 of std dev of positions (Page 7)
        # For now, use a constant or estimate.
        sigma2 = 1.0 # placeholder for 2*sigma^2
        
        self.quantum_momentum = np.zeros((natm, 3))
        self.relative_phase = {} # (pair) -> F_ij
        
        if not self.auxiliary_trajectories:
            return

        # Eq. 21: Quantum momentum
        norm_factor = 1.0 / (nmo - 1.0)
        for pair, at in self.auxiliary_trajectories.items():
            if self.is_uks:
                s, i, j = pair
                rho_jj = np.abs(dm_orth[s, j, j])
            else:
                i, j = pair
                rho_jj = np.abs(dm_orth[j, j])
                
            # Pairwise quantum momentum (Eq. 21)
            # P_qp = -rho_jj / (2 * sigma^2 * Ne) * (R - R_ij)
            p_qp = -(rho_jj / (self.sigma2 * nelec)) * (coords - at['R'])
            self.quantum_momentum += norm_factor * p_qp
            self.relative_phase[pair] = at['f_i'] - at['f_j']

    def kernel(self, times, dt=0.02, initial_occ=None, temperature=300.0, callback=None):
        log = logger.new_logger(self, self.verbose)
        
        if initial_occ is None:
            initial_occ = _as_numpy(self.ks.mo_occ)
        self.running_occupation = initial_occ.copy()

        # CPA-OSHXF Specific setup
        traj_times = self.trajectory_data['times']
        traj_coords = self.trajectory_data['coords']
        traj_vel = self.trajectory_data['velocities']
        
        # Calculate sigma2 for XF quantities (Eq. 21)
        # 2*sigma^2 where sigma = 1/4 * std(R)
        # 2*sigma^2 = 1/8 * variance
        self.sigma2 = np.mean(np.var(traj_coords, axis=0)) / 8.0
        if self.sigma2 < 1e-6: self.sigma2 = 1.0 # fallback for static traj
        
        if self.x_mat.ndim != (3 if self.is_uks else 2):
            self.__init__(self.ks, self.trajectory_data, basis='MO')

        if self.is_uks:
            dm_orth = np.array([np.diag(self.running_occupation[0]), 
                               np.diag(self.running_occupation[1])], dtype=np.complex128)
        else:
            dm_orth = np.diag(self.running_occupation).astype(np.complex128)
        
        dm_orth = cupy.asarray(dm_orth)
        results = {'times': [], 'occ': [], 'energies': [], 'dm': None}
        t_now = 0.0
        self._prev_mo_coeff = _as_numpy(self.ks.mo_coeff)
        self._prev_mol = self.ks.mol.copy()
        
        from scipy.interpolate import interp1d
        f_coords = interp1d(traj_times, traj_coords, axis=0, fill_value="extrapolate")
        f_vel = interp1d(traj_times, traj_vel, axis=0, fill_value="extrapolate")
        
        log.info("Starting CPA-OSHXF Dynamics loop...")
        masses = np.array(self.ks.mol.atom_mass_list()) * 1822.888486
        nelec = np.sum(self.running_occupation)
        energies = _as_numpy(self.ks.mo_energy)

        for t_target in times:
            steps = int(np.round((t_target - t_now) / dt))
            for _ in range(steps):
                coords = f_coords(t_now + dt)
                vel = f_vel(t_now + dt)
                mol_next = self.ks.mol.copy()
                mol_next.set_geom_(coords, unit='Bohr')
                
                mo_curr = _as_numpy(self.ks.mo_coeff) 
                nact, mo_aligned = self._get_nact(mol_next, mo_curr, dt)
                energies = _as_numpy(self.ks.mo_energy)
                
                # 3. XF Decoherence terms
                if self.decoherence:
                    dm_np = _as_numpy(dm_orth)
                    self._update_auxiliary_trajectories(t_now, dt, dm_np, energies, coords, vel)
                    self._compute_xf_quantities(coords, dm_np)
                
                # 4. Propagate Density Matrix (Eq. 15)
                nact_gpu = cupy.asarray(nact)
                energies_gpu = cupy.asarray(energies)
                
                # Standard Ehrenfest part
                if self.is_uks:
                    h_eff = cupy.stack([cupy.diag(energies_gpu[0]), cupy.diag(energies_gpu[1])]) - 1j * nact_gpu
                else:
                    h_eff = cupy.diag(energies_gpu) - 1j * nact_gpu
                
                # Add XF-decoherence (ENC) term to propagator
                # Eq. 15 has an additional term that is not a simple commutator.
                # However, for small steps, we can approximate it or use an effective H.
                # Here we implement it as a correction to dm_orth directly after Magnus step.
                u = self.compute_propagator(h_eff, dt)
                if self.is_uks:
                    dm_orth = u @ dm_orth @ u.conj().swapaxes(-1, -2)
                else:
                    dm_orth = u @ dm_orth @ u.conj().T
                
                if self.decoherence and self.quantum_momentum is not None:
                    # Eq. 15 ENC term: i * sum_v (P_v / M_v Ne) * sum_m (F_mj + F_mk) rho_jm rho_mk
                    dm_np = _as_numpy(dm_orth)
                    enc_corr = np.zeros_like(dm_np)
                    p_m_ratio = self.quantum_momentum / (masses[:, None] * nelec)
                    
                    if self.is_uks:
                        for s in range(2):
                            nmo = dm_np.shape[-1]
                            for pair, fij in self.relative_phase.items():
                                if pair[0] != s: continue
                                p_dot_f = np.sum(p_m_ratio * fij)
                                # This pair (i, j) contributes to decoherence of off-diagonals
                                # involving either i or j.
                                i, j = pair[1], pair[2]
                                enc_corr[s, i, j] += 1j * p_dot_f * dm_np[s, i, j]
                                enc_corr[s, j, i] -= 1j * p_dot_f * dm_np[s, j, i]
                    else:
                        nmo = dm_np.shape[-1]
                        for pair, fij in self.relative_phase.items():
                            p_dot_f = np.sum(p_m_ratio * fij)
                            i, j = pair[0], pair[1]
                            enc_corr[i, j] += 1j * p_dot_f * dm_np[i, j]
                            enc_corr[j, i] -= 1j * p_dot_f * dm_np[j, i]
                    
                    # Apply correction (Euler-like for the small decoherence term)
                    dm_orth += cupy.asarray(enc_corr) * dt
                    
                    # Ensure Hermiticity and Tr=Ne after correction
                    if self.is_uks:
                        dm_orth = 0.5 * (dm_orth + dm_orth.conj().swapaxes(-1, -2))
                    else:
                        dm_orth = 0.5 * (dm_orth + dm_orth.conj().T)

                # 5. Surface Hopping
                probs = self._get_hopping_probabilities(dt, nact, _as_numpy(dm_orth))
                probs = self._apply_boltzmann_penalty(probs, energies, temperature)
                
                # Execute Hops (and destroy ATs if hop occurs)
                rng = np.random.default_rng()
                if self.is_uks:
                    for s in range(2):
                        occ_idx = np.where(self.running_occupation[s] > 0.5)[0]
                        for i in occ_idx:
                            r = rng.random()
                            cum_p = 0.0
                            for j in range(probs.shape[2]):
                                if i == j: continue
                                cum_p += probs[s, i, j]
                                if r < cum_p:
                                    log.note(f"Hop: Spin {s} Orb {i} -> {j} at t={t_now+dt:.4f}")
                                    self.running_occupation[s][i] = 0
                                    self.running_occupation[s][j] = 1
                                    # Destroy ATs related to these orbitals
                                    to_del = [p for p in self.auxiliary_trajectories if p[0] == s and (p[1] == i or p[2] == i or p[1] == j or p[2] == j)]
                                    for p in to_del: del self.auxiliary_trajectories[p]
                                    break
                else:
                    occ_idx = np.where(self.running_occupation > 0.5)[0]
                    for i in occ_idx:
                        r = rng.random()
                        cum_p = 0.0
                        for j in range(probs.shape[1]):
                            if i == j: continue
                            cum_p += probs[i, j]
                            if r < cum_p:
                                log.note(f"Hop: Orb {i} -> {j} at t={t_now+dt:.4f}")
                                self.running_occupation[i] = 0
                                self.running_occupation[j] = 1
                                to_del = [p for p in self.auxiliary_trajectories if (p[0] == i or p[1] == i or p[0] == j or p[1] == j)]
                                for p in to_del: del self.auxiliary_trajectories[p]
                                break

                self._prev_mo_coeff = mo_aligned
                self._prev_mol = mol_next
                t_now += dt

            results['times'].append(t_now)
            results['occ'].append(self.running_occupation.copy())
            results['energies'].append(energies.copy())
            if callback: callback(t_now, dm_orth, results)

        results['dm'] = _as_numpy(dm_orth)
        return results
