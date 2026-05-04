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

"""Unit tests for MECPScanner's transition-density character tracking.

Each test isolates one piece of the integration:

  * trackers built at init,
  * resolved-pair returns distinct roots when no conflict,
  * a simulated amplitude swap is detected and propagated to ``states``
    while the trackers re-anchor on the right character,
  * a forced same-root conflict is resolved via runner-up,
  * energy ordering of ``states`` is preserved through a swap.

We avoid running the full MECP optimization (slow, system-dependent) by
constructing the scanner directly and invoking only the parts we want to
exercise.
"""

import unittest

import numpy as np
from pyscf import gto

from gpu4pyscf import dft
from gpu4pyscf.nac.mecp import MECPScanner, ConicalIntersectionOptimizer


def _h2o_td(nstates=5):
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='6-31g', verbose=0)
    ks = dft.RKS(mol); ks.xc = 'b3lyp'
    ks.kernel()
    td = ks.TDA(); td.nstates = nstates
    td.kernel()
    return ks, td


class TestMECPTracking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ks, cls.td = _h2o_td()

    def _make_scanner(self, states=(1, 2)):
        opt = ConicalIntersectionOptimizer(self.td, states=states,
                                           crossing_type='n-2')
        return MECPScanner(opt)

    def test_trackers_built_at_init(self):
        scanner = self._make_scanner(states=(1, 2))
        self.assertIsNotNone(scanner._tracker_a)
        self.assertIsNotNone(scanner._tracker_b)
        # Tracker a is anchored on state 1 (root index 0); b on state 2.
        self.assertEqual(scanner._tracker_a.state_ref0, 0)
        self.assertEqual(scanner._tracker_b.state_ref0, 1)

    def test_resolve_pair_no_conflict(self):
        """At the reference geometry, each tracker matches its anchor root."""
        scanner = self._make_scanner(states=(1, 2))
        root_a, root_b, match_a, match_b = scanner._resolve_pair()
        self.assertEqual(root_a, 0)
        self.assertEqual(root_b, 1)
        self.assertGreater(match_a.overlap, 0.99)
        self.assertGreater(match_b.overlap, 0.99)

    def test_resolve_pair_after_amplitude_swap(self):
        """Swap td.xy[0]<->td.xy[1] in place; the trackers should detect that
        character_a is now at root 1 and character_b at root 0."""
        scanner = self._make_scanner(states=(1, 2))
        x0, y0 = self.td.xy[0]
        x1, y1 = self.td.xy[1]
        self.td.xy[0] = (x1, y1)
        self.td.xy[1] = (x0, y0)
        try:
            root_a, root_b, match_a, match_b = scanner._resolve_pair()
            self.assertEqual(root_a, 1)
            self.assertEqual(root_b, 0)
            self.assertGreater(match_a.overlap, 0.99)
            self.assertGreater(match_b.overlap, 0.99)
        finally:
            self.td.xy[0] = (x0, y0)
            self.td.xy[1] = (x1, y1)

    def test_states_reordered_by_energy_after_swap(self):
        """After an amplitude swap, the algorithm-facing ``self.states``
        should still be ordered by current energy (lower first); the
        trackers re-anchor on the swapped roots."""
        scanner = self._make_scanner(states=(1, 2))
        # Simulate the energy-ordering update path manually (without running
        # td_scanner, which would re-solve and undo the swap).
        x0, y0 = self.td.xy[0]
        x1, y1 = self.td.xy[1]
        self.td.xy[0] = (x1, y1)
        self.td.xy[1] = (x0, y0)
        try:
            root_a, root_b, _, _ = scanner._resolve_pair()
            e_states = self.td.e
            if float(e_states[root_a]) <= float(e_states[root_b]):
                new_states = (root_a + 1, root_b + 1)
            else:
                new_states = (root_b + 1, root_a + 1)
            # td.e is sorted ascending, root_b=0 is lower, root_a=1 is higher
            self.assertEqual(new_states, (1, 2))
            # Re-anchor and verify each tracker now follows the swapped root
            scanner._tracker_a.re_anchor(self.td, state_ref=root_a + 1)
            scanner._tracker_b.re_anchor(self.td, state_ref=root_b + 1)
            self.assertEqual(scanner._tracker_a.state_ref0, root_a)
            self.assertEqual(scanner._tracker_b.state_ref0, root_b)
        finally:
            self.td.xy[0] = (x0, y0)
            self.td.xy[1] = (x1, y1)

    def test_conflict_resolution_runner_up(self):
        """Force both trackers to anchor on the same character; resolve_pair
        should give the loser its runner-up rather than a duplicate root."""
        opt = ConicalIntersectionOptimizer(self.td, states=(1, 2),
                                           crossing_type='n-2')
        scanner = MECPScanner(opt)
        # Manually re-anchor tracker_b on character_a (state 1) so both now
        # match root 0 -- the conflict path.
        scanner._tracker_b.re_anchor(self.td, state_ref=1)
        root_a, root_b, _, _ = scanner._resolve_pair()
        # Both can't be root 0; one must fall back.
        self.assertNotEqual(root_a, root_b)
        self.assertIn(0, (root_a, root_b))


if __name__ == '__main__':
    unittest.main()
