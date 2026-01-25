import numpy as np
import cupy
from pyscf import lib
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.dft import rks, uks
from gpu4pyscf.scf import hf, uhf

class RTTDDFT(lib.StreamObject):
    def __init__(self, ks):
        self.ks = ks
        self.mol = ks.mol
        self.verbose = ks.verbose
        self.stdout = ks.stdout
        self._keys = {'ks', 'mol', 'verbose', 'stdout', 'field_fn'}
        self.field_fn = None # Function of t returning [Ex, Ey, Ez]
        self.is_uks = isinstance(ks, uks.UKS)
        
        # Disable dynamic grid pruning to prevent energy noise during propagation
        self.ks.small_rho_cutoff = 0
        
        # Precompute orthogonalization matrices (Lowdin symmetric orthogonalization)
        # S = U s U.T
        # X = S^(-1/2) = U s^(-1/2) U.T
        # X_inv = S^(1/2) = U s^(1/2) U.T
        s = cupy.asarray(ks.get_ovlp())
        e, v = cupy.linalg.eigh(s)
        
        # Filter small eigenvalues for stability
        mask = e > 1e-15
        e = e[mask]
        v = v[:, mask]
        
        self.x_mat = v @ cupy.diag(e**(-0.5)) @ v.T
        self.x_inv = v @ cupy.diag(e**(0.5)) @ v.T
        
        # Purification parameter
        self.purify_interval = None 
        self.mu_spin = 'total' # 'total', 'alpha', 'beta'
        self.record_occ = False # Track MO populations

    def kernel(self, times, dm0=None, dt=0.02, propagator='magnus_step', callback=None):
        '''
        Propagate the density matrix in time.
        
        Args:
            times (ndarray): Time points to return results for.
            dm0 (ndarray): Initial density matrix. If None, use ks.make_rdm1().
            dt (float): Time step for propagation.
            propagator (str): Propagator method. Currently supported: 'magnus_step'.
            callback (callable): Function called at each step as callback(t, dm, results).
            
        Returns:
            dict: Results containing 'dip' (dipole moments), 'energy', 'dm' (final density).
        '''
        log = logger.new_logger(self, self.verbose)
        if dm0 is None:
            dm0 = self.ks.make_rdm1()
        
        # Ensure dm0 is on GPU and complex
        dm_ao = cupy.asarray(dm0).astype(cupy.complex128)
        
        # Transform DM to Orthogonal Basis: P_orth = X_inv * P_ao * X_inv.T (since X_inv is symmetric S^(1/2))
        # P_orth = S^(1/2) P S^(1/2)
        if self.is_uks:
            # dm_ao is (2, N, N)
            dm_orth = self.x_inv @ dm_ao @ self.x_inv.T
        else:
            dm_orth = self.x_inv @ dm_ao @ self.x_inv.T
            
        # Precompute projection matrices for Population Analysis
        if self.record_occ:
            s_mat = cupy.asarray(self.ks.get_ovlp())
            c_mat = cupy.asarray(self.ks.mo_coeff)
            if self.is_uks:
                # C is (2, N, N). S is (N, N).
                # Proj = S @ C -> (2, N, N)
                # We want S @ C[s]. Einsum broadcasts S over spin dim.
                self.mo_proj = cupy.einsum('pq,sqr->spr', s_mat, c_mat)
            else:
                self.mo_proj = s_mat @ c_mat
        
        # Check if hybrid
        ni = self.ks._numint
        hybrid = ni.libxc.is_hybrid_xc(self.ks.xc)
        if hybrid:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
        else:
            c_k = 0.0

        hcore = cupy.asarray(self.ks.get_hcore())
        if self.is_uks and hcore.ndim == 2:
            hcore = cupy.stack([hcore, hcore])
        
        # Precompute dipole integrals for field interaction
        # mu = - <phi|r|phi>
        with self.mol.with_common_orig((0,0,0)):
            ints_mu = cupy.asarray(self.mol.intor('int1e_r'))
        
        # Time loop
        t_now = 0.0
        results = {
            'times': [],
            'dip': [],
            'energy': []
        }
        if self.is_uks:
            results['dip_alpha'] = []
            results['dip_beta'] = []
            if self.record_occ:
                results['occ_alpha'] = []
                results['occ_beta'] = []
        elif self.record_occ:
            results['occ'] = []
        
        # Initial properties
        self._record(0.0, dm_ao, results)
        if callback is not None:
            callback(0.0, dm_ao, results)
        
        total_steps = 0
        for t_target in times:
            if t_target <= t_now: continue
            
            steps = int(np.round((t_target - t_now) / dt))
            for step in range(steps):
                total_steps += 1
                
                # McWeeny Purification
                if self.purify_interval and total_steps % self.purify_interval == 0:
                    dm_orth = self._apply_mcweeny(dm_orth)

                if propagator == 'magnus_step':
                    # Magnus 2nd Order / MMUT (Gauss-Legendre 2)
                    
                    # 1. Predictor: Estimate F(t+dt/2)
                    # Use current density P_ao(t) to build F_ao(t)
                    dm_ao = self.to_ao(dm_orth)
                    f_ao_t = self.get_fock(dm_ao, hcore, ints_mu, t_now, c_k)
                    
                    # Transform F to Orthogonal Basis: F_orth = X.T * F_ao * X
                    f_orth_t = self.to_orth_fock(f_ao_t)
                    
                    # Compute Propagator U(dt/2)
                    u_half = self.compute_propagator(f_orth_t, dt/2)
                    
                    # Predict P_orth(t+dt/2)
                    if self.is_uks:
                        dm_orth_mid = u_half @ dm_orth @ u_half.conj().swapaxes(-1, -2)
                    else:
                        dm_orth_mid = u_half @ dm_orth @ u_half.conj().T
                    
                    # 2. Corrector: Build F(t+dt/2)
                    dm_ao_mid = self.to_ao(dm_orth_mid)
                    f_ao_mid = self.get_fock(dm_ao_mid, hcore, ints_mu, t_now + dt/2, c_k)
                    
                    # Transform F to Orthogonal Basis
                    f_orth_mid = self.to_orth_fock(f_ao_mid)
                    
                    # 3. Propagate: P_orth(t+dt) = U(dt) P_orth(t) U(dt)+
                    u_full = self.compute_propagator(f_orth_mid, dt)
                    
                    if self.is_uks:
                        dm_orth = u_full @ dm_orth @ u_full.conj().swapaxes(-1, -2)
                    else:
                        dm_orth = u_full @ dm_orth @ u_full.conj().T
                
                else:
                    raise ValueError(f"Unknown propagator: {propagator}")
                
                t_now += dt
            
            # Convert back to AO for recording
            dm_ao = self.to_ao(dm_orth)
            self._record(t_now, dm_ao, results)
            if callback is not None:
                callback(t_now, dm_ao, results)
            log.info(f"Time: {t_now:10.4f} au | Energy: {results['energy'][-1]:20.12f}")

        results['dm'] = cupy.asnumpy(dm_ao)
        return results

    def to_ao(self, dm_orth):
        '''Transform Orthogonal Density -> AO Density: P_ao = X * P_orth * X.T'''
        # P_ao = S^(-1/2) P_orth S^(-1/2)
        if self.is_uks:
            # dm_orth is (2, N, N), x_mat is (N, N)
            # i, j, k, l are spatial AO indices. s is spin.
            return cupy.einsum('ij,sjk,kl->sil', self.x_mat, dm_orth, self.x_mat.T)
        else:
            return self.x_mat @ dm_orth @ self.x_mat.T

    def to_orth_fock(self, f_ao):
        '''Transform AO Fock -> Orthogonal Fock: F_orth = X.T * F_ao * X'''
        # F_orth = S^(-1/2) F_ao S^(-1/2)
        if self.is_uks:
            # f_ao is (2, N, N)
            return cupy.einsum('ij,sjk,kl->sil', self.x_mat.T, f_ao, self.x_mat)
        else:
            return self.x_mat.T @ f_ao @ self.x_mat

    def _apply_mcweeny(self, dm):
        '''McWeeny purification: P = 3P^2 - 2P^3'''
        # P^2
        if self.is_uks:
            dm2 = dm @ dm
            dm3 = dm2 @ dm
        else:
            dm2 = dm @ dm
            dm3 = dm2 @ dm
            
        return 3.0 * dm2 - 2.0 * dm3

    def get_fock(self, dm, hcore, ints_mu, t, c_k=0.0):
        '''
        Construct the complex Fock matrix.
        F = h_core + J[P_re] + V_xc[P_re] - c_k * K[P] - mu . E(t)
        '''
        dm_re = dm.real.copy() 
        
        # Get Veff (J + Vxc - c_k * K_re)
        v_eff_re = self.ks.get_veff(self.mol, dm_re)
        
        fock = (hcore + v_eff_re).astype(cupy.complex128)
        
        if c_k != 0.0:
            if hasattr(v_eff_re, 'vk'):
                vk_re = v_eff_re.vk
                fock += vk_re
            
            # Compute full complex K
            # Split into Real and Imaginary parts because CUDA kernels expect real DMs
            vk_real = self.ks.get_k(self.mol, dm.real)
            vk_imag = self.ks.get_k(self.mol, dm.imag)
            vk_tot = vk_real + 1j * vk_imag
            fock -= c_k * vk_tot
            
        # Field interaction: - mu . E(t) = + r . E(t) (since mu = -r)
        if self.field_fn is not None:
            E = self.field_fn(t)
            # ints_mu shape (3, n, n)
            h_field = cupy.einsum('xij,x->ij', ints_mu, cupy.asarray(E))
            
            if self.is_uks:
                # Broadcast field to both spins
                fock += h_field[None, :, :]
            else:
                fock += h_field

        # Enforce Hermiticity for numerical stability
        # F = 0.5 * (F + F.H)
        if self.is_uks:
            fock = 0.5 * (fock + fock.conj().swapaxes(-1, -2))
        else:
            fock = 0.5 * (fock + fock.conj().T)

        return fock

    def compute_propagator(self, fock, dt):
        '''
        Compute U = exp(-i * F * dt)
        '''
        # Diagonalize F: F = V e V+
        # exp(-i F dt) = V exp(-i e dt) V+
        # Cupy.linalg.eigh supports stacked matrices since recent versions
        
        # F is Hermitian
        e, v = cupy.linalg.eigh(fock)
        
        # exp(-i * e * dt)
        exp_e = cupy.exp(-1j * e * dt)
        
        # Reconstruct
        if fock.ndim == 3:
            # UKS / Batched: (2, N, N)
            # v @ diag(exp_e) @ vH
            # exp_e is (2, N). Expand to (2, 1, N) for broadcasting against vH rows
            u = v @ (exp_e[..., None] * v.conj().swapaxes(-1, -2))
        else:
            # RKS: (N, N)
            u = v @ (exp_e[:, None] * v.conj().T)
        return u

    def _record(self, t, dm, results):
        dm_cpu = cupy.asnumpy(dm)
        results['times'].append(t)
        
        # Dipole
        mol = self.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        charge_center = np.einsum('z,zr->r', charges, coords) / charges.sum()
        
        with mol.with_common_orig(charge_center):
            if self.is_uks:
                # Individual spin electronic dipoles
                # PySCF dip_moment adds nuclear contribution.
                # To get resolved spins, we calculate electronic part manually or call carefully.
                # Total Dipole = Elec_Alpha + Elec_Beta + Nuc
                
                # Full total dipole (incl nuclear)
                dip_tot = self.ks.dip_moment(mol, dm_cpu, verbose=0)
                
                # Electronic part only
                from pyscf import scf
                ints = mol.intor('int1e_r')
                dip_elec_a = -np.einsum('xij,ji->x', ints, dm_cpu[0].real)
                dip_elec_b = -np.einsum('xij,ji->x', ints, dm_cpu[1].real)
                # Nuclear part
                dip_nuc = dip_tot - (dip_elec_a + dip_elec_b)
                
                # Resolved dipoles (partition nuclear part 50/50)
                dip_a = dip_elec_a + 0.5 * dip_nuc
                dip_b = dip_elec_b + 0.5 * dip_nuc
                
                results['dip_alpha'].append(dip_a)
                results['dip_beta'].append(dip_b)
                
                if self.mu_spin == 'alpha':
                    results['dip'].append(dip_a)
                elif self.mu_spin == 'beta':
                    results['dip'].append(dip_b)
                else:
                    results['dip'].append(dip_tot)
            else:
                # RKS
                dip = self.ks.dip_moment(mol, dm_cpu, verbose=0)
                results['dip'].append(dip.real)
        
        # Energy
        # E = Tr(h P) + 1/2 Tr(P (J - c K) P) + Exc
        # This is expensive to compute every step fully consistent.
        # For now, approximate or skip? Let's compute it.
        # Using self.get_fock to help?
        # E = Tr(h P) + E_coul + E_xc - c_k E_ex
        # Re-calling get_veff(dm.real) gives Ecoul and Exc for real part.
        dm_re = dm.real
        veff = self.ks.get_veff(self.mol, dm_re)
        
        hcore = self.ks.get_hcore()
        if self.is_uks:
            if isinstance(hcore, np.ndarray): hcore = cupy.asarray(hcore)
            if hcore.ndim == 2: hcore = cupy.array([hcore, hcore])
            e1 = cupy.einsum('sij,sji->', hcore, dm_re).real
        else:
            if isinstance(hcore, np.ndarray): hcore = cupy.asarray(hcore)
            e1 = cupy.einsum('ij,ji->', hcore, dm_re).real
            
        ecoul = veff.ecoul
        exc = veff.exc
        
        e_tot = e1 + ecoul + exc + self.ks.energy_nuc()
        
        # Hybrid correction for energy
        # Ex_hybrid = - c_k * 0.5 * Tr(P K P)
        # Note: veff.vk is c_k * K_re. Energy contrib was -0.5 * Tr(P_re * veff.vk)
        
        ni = self.ks._numint
        hybrid = ni.libxc.is_hybrid_xc(self.ks.xc)
        if hybrid:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.ks.xc, spin=self.mol.spin)
            c_k = hyb if self.is_uks else 0.5 * hyb
            
            # Remove real exchange from E_tot
            # Energy contrib was -0.5 * Tr(dm_re * veff.vk)
            if self.is_uks:
                e_ex_re = -0.5 * cupy.einsum('sij,sji->', dm_re, veff.vk).real
            else:
                e_ex_re = -0.5 * cupy.einsum('ij,ji->', dm_re, veff.vk).real
            e_tot -= e_ex_re
            
            # Add complex exchange
            # E_ex_tot = -0.5 * c_k * Tr(dm * K_tot)
            vk_real = self.ks.get_k(self.mol, dm.real)
            vk_imag = self.ks.get_k(self.mol, dm.imag)
            k_tot = vk_real + 1j * vk_imag
            
            if self.is_uks:
                e_ex_tot = -0.5 * c_k * cupy.einsum('sij,sji->', dm, k_tot).real
            else:
                e_ex_tot = -0.5 * c_k * cupy.einsum('ij,ji->', dm, k_tot).real
            e_tot += e_ex_tot

        results['energy'].append(float(e_tot.real))
        
        # MO Populations
        if self.record_occ:
            if self.is_uks:
                # mo_proj is (2, N, N) -> S @ C[s]
                # P_mo = C[s].H @ S @ P[s] @ S @ C[s] = mo_proj[s].H @ P[s] @ mo_proj[s]
                # dm is (2, N, N)
                
                # Alpha
                p_mo_a = self.mo_proj[0].conj().T @ dm[0] @ self.mo_proj[0]
                occ_a = cupy.diagonal(p_mo_a).real
                results['occ_alpha'].append(cupy.asnumpy(occ_a))
                
                # Beta
                p_mo_b = self.mo_proj[1].conj().T @ dm[1] @ self.mo_proj[1]
                occ_b = cupy.diagonal(p_mo_b).real
                results['occ_beta'].append(cupy.asnumpy(occ_b))
            else:
                # RKS
                # mo_proj is (N, N)
                p_mo = self.mo_proj.conj().T @ dm @ self.mo_proj
                occ = cupy.diagonal(p_mo).real
                results['occ'].append(cupy.asnumpy(occ))