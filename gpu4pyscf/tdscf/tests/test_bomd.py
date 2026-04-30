# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for excited-state BOMD in gpu4pyscf.tdscf.ehrenfest."""

import unittest
import numpy as np
from pyscf import gto

from gpu4pyscf import dft
from gpu4pyscf.tdscf.ehrenfest import BOMD
from gpu4pyscf.tdscf import rtutils as rtu


def _build_h2():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    ks = dft.RKS(mol); ks.xc = 'b3lyp'
    ks.kernel()
    return mol, ks


class TestMaxwellBoltzmann(unittest.TestCase):
    def test_kinetic_energy_matches_temperature(self):
        n_atoms = 32
        masses = np.full(n_atoms, 1.0 * 1822.888486)  # H-mass
        T = 500.0  # K
        n_samples = 5000
        rng = np.random.default_rng(0)
        v_all = np.array([
            rtu.maxwell_boltzmann_velocities(masses, T, rng=rng)
            for _ in range(n_samples)
        ])
        # Average kinetic energy per DOF: 0.5 * k_B * T
        ke_per_dof = 0.5 * (masses[None, :, None] * v_all**2).mean()
        expected = 0.5 * rtu.KB_HARTREE * T
        # Statistical tolerance: O(1/sqrt(n_samples * 3 * n_atoms))
        rel_err = abs(ke_per_dof - expected) / expected
        self.assertLess(rel_err, 0.02,
                        f"<KE>/dof = {ke_per_dof:.4e}, expected {expected:.4e}")

    def test_remove_com_momentum(self):
        masses = np.array([1.0, 16.0, 1.0]) * 1822.888486
        v = np.array([[1.0, 0, 0], [0.5, 0.2, 0], [-0.3, 0.1, 0.05]])
        rtu.remove_com_momentum(masses, v)
        p_total = (masses[:, None] * v).sum(axis=0)
        np.testing.assert_allclose(p_total, 0.0, atol=1e-12)


class TestBOMDGroundState(unittest.TestCase):
    """Ground-state BOMD: state=0 path must remain available and conserve."""

    def test_gs_energy_conservation(self):
        mol, ks = _build_h2()
        md = BOMD(ks, state=0)
        # Stretch H2 a bit: tiny nuclear velocities along the bond
        md.velocities = np.array([[0, 0, -1e-3], [0, 0, +1e-3]])
        md.thermostat = None
        md.com_step = 0
        # dt=5 au (~0.12 fs) is well-resolved for H2 vibration.
        times = np.arange(0, 50.0 + 0.5, 5.0)
        results = md.kernel(times=times, dt=5.0)
        e = np.array(results['energy_tot'])
        # NVE conservation: Verlet gives O(dt^2) per-step error;
        # 10 steps at dt=5 au should be tight.
        self.assertLess(np.max(np.abs(e - e[0])), 1e-5)


class TestBOMDExcitedState(unittest.TestCase):
    """Excited-state BOMD: state=1 path differs from GS and conserves."""

    def setUp(self):
        self.mol, self.ks = _build_h2()
        # Need a TDDFT object to feed BOMD
        self.td = self.ks.TDDFT(); self.td.nstates = 3
        self.td.kernel()

    def test_excited_state_trajectory_differs_from_gs(self):
        # Run GS and excited-state trajectories from the same kick; compare.
        v0 = np.array([[0, 0, -1e-3], [0, 0, +1e-3]])

        md_gs = BOMD(self.ks, state=0)
        md_gs.velocities = v0.copy()
        md_gs.thermostat = None; md_gs.com_step = 0
        res_gs = md_gs.kernel(times=np.arange(0, 50.0 + 0.5, 5.0), dt=5.0)

        # Rebuild ks for fresh state (kernel mutates mol in place)
        mol, ks = _build_h2()
        td = ks.TDDFT(); td.nstates = 3; td.kernel()
        md_es = BOMD(ks, td=td, state=1)
        md_es.velocities = v0.copy()
        md_es.thermostat = None; md_es.com_step = 0
        res_es = md_es.kernel(times=np.arange(0, 50.0 + 0.5, 5.0), dt=5.0)

        coords_gs = np.asarray(res_gs['coords'][-1])
        coords_es = np.asarray(res_es['coords'][-1])
        max_diff = np.max(np.abs(coords_gs - coords_es))
        self.assertGreater(max_diff, 1e-4,
                           f"GS and S1 trajectories indistinguishable "
                           f"(|diff|={max_diff:.3e} Bohr)")

    def test_excited_state_energy_conservation(self):
        md = BOMD(self.ks, td=self.td, state=1)
        md.velocities = np.array([[0, 0, -1e-3], [0, 0, +1e-3]])
        md.thermostat = None; md.com_step = 0
        results = md.kernel(times=np.arange(0, 50.0 + 0.5, 5.0), dt=5.0)
        e = np.array(results['energy_tot'])
        # Excited-state energy = mf.e_tot + td.e[state-1] + T_nuc.
        # H2/sto-3g S1 is a steep antibonding state, so dt=5 au is on the
        # edge for Verlet; we just check drift is bounded (no divergence).
        self.assertLess(np.max(np.abs(e - e[0])), 2e-3)
        # State should not have re-assigned itself (energy gap is large in H2).
        self.assertEqual(md.state, 1)
        # state_history populated and consistent with self.state
        self.assertGreater(len(results['state_history']), 1)
        self.assertEqual(results['state_history'][-1], 1)


if __name__ == '__main__':
    unittest.main()
