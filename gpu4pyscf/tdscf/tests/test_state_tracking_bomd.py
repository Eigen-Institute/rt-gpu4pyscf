# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

"""Unit tests for BOMD's transition-density state-tracking integration.

The deterministic test simulates a root flip by swapping ``td.xy[0]`` and
``td.xy[1]`` in place between two ``_check_overlap`` calls -- this isolates
the wiring without needing a physical avoided crossing.

The smoke test runs a short trajectory through each tracking mode to
verify the kernel path doesn't error out.
"""

import unittest

import numpy as np
from pyscf import lib as pyscf_lib
from pyscf import gto

from gpu4pyscf import dft
from gpu4pyscf.tdscf.ehrenfest import BOMD
from gpu4pyscf.tdscf import rtutils as rtu


def _h2o_ks():
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='6-31g', verbose=0)
    ks = dft.RKS(mol); ks.xc = 'b3lyp'
    ks.kernel()
    return ks


class TestBOMDOverlapTracking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ks = _h2o_ks()

    def _fresh_td(self, nstates=5):
        td = self.ks.TDA(); td.nstates = nstates
        td.kernel()
        return td

    def _silent_log(self, md):
        return pyscf_lib.logger.new_logger(md, verbose=0)

    def test_simulated_root_swap_overlap(self):
        """Anchor on root 1, swap xy[0]<->xy[1], expect tracker to detect
        the flip and update self.state from 1 to 2."""
        td = self._fresh_td()
        md = BOMD(self.ks, td=td, state=1, tracking_method='overlap')
        log = self._silent_log(md)

        md._check_overlap(log)
        self.assertIsNotNone(md._tracker)
        self.assertEqual(md.state, 1)

        x0, y0 = td.xy[0]
        x1, y1 = td.xy[1]
        td.xy[0] = (x1, y1)
        td.xy[1] = (x0, y0)
        try:
            md._check_overlap(log)
            self.assertEqual(md.state, 2)
        finally:
            td.xy[0] = (x0, y0)
            td.xy[1] = (x1, y1)

    def test_no_change_when_amplitudes_unchanged(self):
        """Two consecutive calls without mutation must not change state."""
        td = self._fresh_td()
        md = BOMD(self.ks, td=td, state=1, tracking_method='overlap')
        log = self._silent_log(md)
        md._check_overlap(log)
        md._check_overlap(log)
        self.assertEqual(md.state, 1)

    def test_energy_tracker_misses_swap(self):
        """The legacy energy-ordering tracker compares E_tot to the previous
        step's energy. After a pure amplitude swap (energies unchanged in the
        td.e array because the eigenvalue ordering doesn't change with a
        manual xy-swap), the energy tracker will *not* detect the flip --
        confirming why character-based tracking is needed."""
        td = self._fresh_td()
        md = BOMD(self.ks, td=td, state=1, tracking_method='energy')
        log = self._silent_log(md)

        # Seed prev_state_energy
        prev = float(self.ks.e_tot) + float(td.e[0])
        x0, y0 = td.xy[0]
        x1, y1 = td.xy[1]
        td.xy[0] = (x1, y1)
        td.xy[1] = (x0, y0)
        try:
            md._check_energy(prev, log)
            # Energy tracker uses td.e (untouched by the xy swap) and
            # therefore matches the same index -- it does not catch the flip.
            self.assertEqual(md.state, 1)
        finally:
            td.xy[0] = (x0, y0)
            td.xy[1] = (x1, y1)

    def test_state_zero_is_skipped(self):
        """Ground-state BOMD (state=0) should bypass tracking entirely
        through the dispatcher."""
        md = BOMD(self.ks, td=None, state=0, tracking_method='overlap')
        log = self._silent_log(md)
        result = md._check_state_following(prev_state_energy=None, log=log)
        self.assertEqual(result, 0)
        self.assertIsNone(md._tracker)

    def test_invalid_tracking_method_raises(self):
        with self.assertRaises(ValueError):
            BOMD(self.ks, td=None, state=0, tracking_method='nonsense')


class TestBOMDSmoke(unittest.TestCase):
    """Run a 3-step trajectory in each tracking mode; assert no crashes and
    that state_history is populated."""

    @classmethod
    def setUpClass(cls):
        cls.ks = _h2o_ks()
        td = cls.ks.TDA(); td.nstates = 5
        td.kernel()
        cls.td = td

    def _run(self, method, n_steps=3, dt=20.0):
        md = BOMD(self.ks, td=self.td, state=1, tracking_method=method)
        md.verbose = 0
        md.thermostat = None
        md.com_step = 100
        md.velocities = rtu.maxwell_boltzmann_velocities(
            md.masses, 300.0, rng=np.random.default_rng(0))
        rtu.remove_com_momentum(md.masses, md.velocities)
        times = np.arange(0, n_steps * dt + dt * 0.5, dt)
        return md.kernel(times=times, dt=dt)

    def test_smoke_overlap(self):
        results = self._run('overlap')
        self.assertEqual(len(results['state_history']), 4)  # t=0 + 3 steps
        self.assertTrue(all(s == 1 for s in results['state_history']))
        # Energy drift over 3 steps should be modest
        e_tot = np.array(results['energy_tot'])
        self.assertLess(float(np.max(np.abs(e_tot - e_tot[0]))), 5e-4)

    def test_smoke_energy(self):
        results = self._run('energy')
        self.assertEqual(len(results['state_history']), 4)
        self.assertTrue(all(s == 1 for s in results['state_history']))

    def test_smoke_auto(self):
        results = self._run('auto')
        self.assertEqual(len(results['state_history']), 4)
        self.assertTrue(all(s == 1 for s in results['state_history']))


class TestGradScannerTracker(unittest.TestCase):
    """Verify the optional `tracker=` kwarg on td.nuc_grad_method().as_scanner.
    Closes the historical 'TODO: Check root flip' in grad/tdrhf.py."""

    @classmethod
    def setUpClass(cls):
        cls.ks = _h2o_ks()

    def _fresh_td(self):
        td = self.ks.TDA(); td.nstates = 5
        td.kernel()
        return td

    def test_scanner_runs_with_tracker(self):
        from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker
        td = self._fresh_td()
        tracker = TransitionDensityTracker(td, state_ref=1)
        scanner = td.nuc_grad_method().as_scanner(state=1, tracker=tracker)
        e_tot, g = scanner(self.ks.mol)
        # No flip at equilibrium; state stays put.
        self.assertEqual(scanner.state, 1)
        self.assertEqual(np.asarray(g).shape, (3, 3))
        # Tracker is still tracking root 1 (re-anchored, but same target).
        self.assertEqual(tracker.state_ref0, 0)

    def test_scanner_no_tracker_unchanged(self):
        """Default tracker=None gives the legacy scanner with no _tracker
        attached -- baseline that the kwarg adds no behavior when omitted."""
        td = self._fresh_td()
        scanner = td.nuc_grad_method().as_scanner(state=1)
        e_tot, g = scanner(self.ks.mol)
        self.assertEqual(np.asarray(g).shape, (3, 3))
        self.assertIsNone(scanner._tracker)


if __name__ == '__main__':
    unittest.main()
