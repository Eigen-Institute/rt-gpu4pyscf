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

import unittest

import numpy as np
from pyscf import gto

from gpu4pyscf import dft
from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker


def _h2o(displaced=False):
    if displaced:
        # 0.005 Bohr ~ 0.00265 A on the O along z
        atom = '''
        O  0.000000  0.000000  0.002645
        H  0.757000  0.000000  0.586900
        H -0.757000  0.000000  0.586900
        '''
    else:
        atom = '''
        O  0.000000  0.000000  0.000000
        H  0.757000  0.000000  0.586900
        H -0.757000  0.000000  0.586900
        '''
    return gto.M(atom=atom, basis='6-31g', verbose=0)


class TestTransitionDensityTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mol = _h2o(displaced=False)
        ks = dft.RKS(mol); ks.xc = 'b3lyp'
        ks.kernel()
        td = ks.TDA(); td.nstates = 5
        td.kernel()
        cls.ks_eq = ks
        cls.td_eq = td

    def test_self_overlap_rks_tda(self):
        tracker = TransitionDensityTracker(self.td_eq, state_ref=1)
        result = tracker.assign(self.td_eq)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.99)
        self.assertNotIn('low_overlap', result.flags)
        self.assertNotIn('index_changed', result.flags)

    def test_self_overlap_other_root(self):
        for s in (2, 3):
            tracker = TransitionDensityTracker(self.td_eq, state_ref=s)
            result = tracker.assign(self.td_eq)
            self.assertEqual(result.root, s - 1)
            self.assertGreater(result.overlap, 0.99)

    def test_small_displacement(self):
        mol_disp = _h2o(displaced=True)
        ks_d = dft.RKS(mol_disp); ks_d.xc = self.ks_eq.xc
        ks_d.kernel()
        self.assertTrue(ks_d.converged)
        td_d = ks_d.TDA(); td_d.nstates = 5
        td_d.kernel()

        tracker = TransitionDensityTracker(self.td_eq, state_ref=1)
        result = tracker.assign(td_d)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.95)

    def test_full_tddft(self):
        td_full = self.ks_eq.TDDFT(); td_full.nstates = 5
        td_full.kernel()
        tracker = TransitionDensityTracker(td_full, state_ref=1)
        result = tracker.assign(td_full)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.95)

    def test_uks_tda(self):
        mol = _h2o(displaced=False)
        ks = dft.UKS(mol); ks.xc = 'b3lyp'
        ks.kernel()
        td = ks.TDA(); td.nstates = 5
        td.kernel()
        tracker = TransitionDensityTracker(td, state_ref=1)
        result = tracker.assign(td)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.99)

    def test_method_mismatch_raises(self):
        td_full = self.ks_eq.TDDFT(); td_full.nstates = 5
        td_full.kernel()
        tracker = TransitionDensityTracker(self.td_eq, state_ref=1)
        with self.assertRaises(RuntimeError):
            tracker.assign(td_full)

    def test_state_index_validation(self):
        with self.assertRaises(ValueError):
            TransitionDensityTracker(self.td_eq, state_ref=0)
        with self.assertRaises(IndexError):
            TransitionDensityTracker(self.td_eq, state_ref=99)

    def test_signed_overlap_and_sign(self):
        """signed_overlap matches the cosine; sign property is +/-1.

        Flips the amplitude in place on the same td object so the test does
        not depend on the (random) phase a second TDA solve would pick.
        """
        tracker = TransitionDensityTracker(self.td_eq, state_ref=1)
        result = tracker.assign(self.td_eq)
        self.assertGreater(result.signed_overlap, 0.99)
        self.assertEqual(result.sign, 1)

        x_orig, y_orig = self.td_eq.xy[0]
        self.td_eq.xy[0] = (-x_orig, y_orig)
        try:
            result_flip = tracker.assign(self.td_eq)
            self.assertEqual(result_flip.root, 0)
            self.assertLess(result_flip.signed_overlap, -0.99)
            self.assertEqual(result_flip.sign, -1)
            self.assertGreater(result_flip.overlap, 0.99)
        finally:
            self.td_eq.xy[0] = (x_orig, y_orig)

    def test_from_amplitudes_self_overlap(self):
        """from_amplitudes + assign_amplitudes round-trip on the same data."""
        mf = self.td_eq._scf
        tracker = TransitionDensityTracker.from_amplitudes(
            mf.mol, mf.mo_coeff, mf.mo_occ, self.td_eq.xy,
            state_ref=1, e_ref=self.td_eq.e)
        result = tracker.assign_amplitudes(
            mf.mol, mf.mo_coeff, self.td_eq.xy, e_disp=self.td_eq.e)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.99)
        self.assertNotIn('low_overlap', result.flags)

    def test_from_amplitudes_no_e_ref(self):
        """e_ref=None disables the energy_jump diagnostic but matching still
        works."""
        mf = self.td_eq._scf
        tracker = TransitionDensityTracker.from_amplitudes(
            mf.mol, mf.mo_coeff, mf.mo_occ, self.td_eq.xy,
            state_ref=1, e_ref=None)
        result = tracker.assign_amplitudes(
            mf.mol, mf.mo_coeff, self.td_eq.xy)
        self.assertEqual(result.root, 0)
        self.assertGreater(result.overlap, 0.99)
        # de_target must be NaN when e_ref unknown
        import math
        self.assertTrue(math.isnan(result.de_target))
        self.assertNotIn('energy_jump', result.flags)


if __name__ == '__main__':
    unittest.main()
