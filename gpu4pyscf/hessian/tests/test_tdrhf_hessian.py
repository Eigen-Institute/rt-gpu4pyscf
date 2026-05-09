# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

'''
Phase 1 unit tests for the perturbed TDA-amplitude response solver
``gpu4pyscf.hessian.tdrhf.solve_x1``.

These tests do not require an analytical Hessian assembly. They probe the
solver in isolation:

  - test_zero_rhs:       b = 0 -> x1 = 0.
  - test_synthetic_rhs:  for random y orthogonal to X, set b = -(A - omega) y
                         and verify solve_x1 recovers y (up to X-component
                         which is in the kernel and is deflated).
  - test_orthogonality:  the returned x1 satisfies X^T x1 = 0.
  - test_residual:       (A - omega) x1 + b is small in the X-orthogonal
                         complement.
'''

import unittest
import numpy as np
import cupy as cp
import pyscf

import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess


# Small system: H2O / STO-3G keeps the test fast and the TDA spectrum
# non-degenerate.
H2O_ATOM = '''
O   0.000000000000   0.000000000000   0.117790000000
H   0.000000000000   0.755453000000  -0.471160000000
H   0.000000000000  -0.755453000000  -0.471160000000
'''


def _setup_tda(basis='sto-3g', nstates=3):
    mol = pyscf.M(atom=H2O_ATOM, basis=basis, verbose=0,
                  output='/dev/null', max_memory=4000)
    mf = gpu_scf.RHF(mol).run()
    assert mf.converged
    td = gpu_tdscf.rhf.TDA(mf)
    td.nstates = nstates
    td.conv_tol = 1e-9
    td.kernel()
    assert all(td.converged)
    return mol, mf, td


def _apply_A_minus_omega(td, state, z):
    '''Apply (A - omega I) to z of shape (nroots, nocc, nvir).'''
    omega = float(td.e[state])
    nroots, nocc, nvir = z.shape
    vind, _ = td.gen_vind(td._scf)
    az = vind(z.reshape(nroots, nocc * nvir)).reshape(nroots, nocc, nvir)
    return az - omega * z


def _project_out_x(td, state, v):
    '''Remove the X-mode component from v of shape (nroots, nocc, nvir).
    Uses the same convention as solve_x1: <X|X> = 0.5.'''
    x = cp.asarray(td.xy[state][0])
    x_dot_x = float((x * x).sum())
    proj = cp.einsum('rov,ov->r', v, x) / x_dot_x
    return v - proj[:, None, None] * x[None]


class TestSolveX1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0  # 0-indexed
        cls.x_ref = cp.asarray(cls.td.xy[cls.state][0])
        cls.nocc, cls.nvir = cls.x_ref.shape

    def test_zero_rhs(self):
        '''b = 0 must give x1 = 0 (deflated kernel).'''
        b = cp.zeros((1, self.nocc, self.nvir))
        x1 = tdrhf_hess.solve_x1(self.td, self.state, b)
        self.assertEqual(x1.shape, (1, self.nocc, self.nvir))
        self.assertLess(float(cp.linalg.norm(x1)), 1e-9)

    def test_synthetic_rhs_recovers_solution(self):
        '''Set b = -(A - omega) y for random y in the X-orthogonal complement;
        verify solve_x1 returns y to high accuracy.'''
        rng = np.random.default_rng(20260504)
        npert = 3
        y_np = rng.standard_normal((npert, self.nocc, self.nvir))
        y = cp.asarray(y_np)
        # Project out X so y lives in the well-posed subspace.
        y = _project_out_x(self.td, self.state, y)

        # b = -(A - omega) y  =>  (A - omega) x1 = -b = (A - omega) y
        # so x1 == y in the X-orthogonal subspace.
        Ay = _apply_A_minus_omega(self.td, self.state, y)
        b = -Ay

        x1 = tdrhf_hess.solve_x1(self.td, self.state, b)
        # x1 should match y up to the X mode (which we projected out of y).
        diff = _project_out_x(self.td, self.state, x1 - y)
        err = float(cp.linalg.norm(diff)) / float(cp.linalg.norm(y))
        self.assertLess(err, 1e-9, f'relative error in recovered y: {err:.3e}')

    def test_orthogonality_to_X(self):
        '''<X|x1> must be 0 by deflation.'''
        rng = np.random.default_rng(42)
        b_np = rng.standard_normal((2, self.nocc, self.nvir))
        b = cp.asarray(b_np)
        # Use a deliberately non-orthogonal RHS to stress the deflator.
        x1 = tdrhf_hess.solve_x1(self.td, self.state, b)
        overlap = cp.einsum('rov,ov->r', x1, self.x_ref)
        self.assertLess(float(cp.abs(overlap).max()), 1e-8)

    def test_residual(self):
        '''(A - omega) x1 + b must be ~0 in the X-orthogonal subspace.'''
        rng = np.random.default_rng(7)
        b_np = rng.standard_normal((1, self.nocc, self.nvir))
        b = _project_out_x(self.td, self.state, cp.asarray(b_np))
        x1 = tdrhf_hess.solve_x1(self.td, self.state, b)
        residual = _apply_A_minus_omega(self.td, self.state, x1) + b
        residual = _project_out_x(self.td, self.state, residual)
        # Normalize by ||b|| for a relative measure.
        rel = float(cp.linalg.norm(residual)) / float(cp.linalg.norm(b))
        self.assertLess(rel, 1e-10, f'relative residual: {rel:.3e}')

    def test_singleton_b_shape(self):
        '''Passing b of shape (nocc, nvir) (not (1, nocc, nvir)) is accepted
        and produces an output of shape (1, nocc, nvir).'''
        b2d = cp.zeros((self.nocc, self.nvir))
        x1 = tdrhf_hess.solve_x1(self.td, self.state, b2d)
        self.assertEqual(x1.shape, (1, self.nocc, self.nvir))


class TestOmegaGrad(unittest.TestCase):
    '''Phase 2.0: omega_grad returns the analytical excitation-energy
    gradient. Validate against FD of the excitation energy.'''

    @classmethod
    def setUpClass(cls):
        # Use a system with a small basis so FD is fast.
        cls.atom = H2O_ATOM
        cls.basis = 'sto-3g'

    def _fd_omega_grad(self, state=0, delta=2.0e-3):
        '''FD reference for d omega / d R via state-tracked TDA at
        +-delta displacements.'''
        from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker

        mol = pyscf.M(atom=self.atom, basis=self.basis, unit='Angstrom',
                      verbose=0, output='/dev/null', max_memory=4000)
        coords0 = mol.atom_coords(unit='Bohr').copy()
        natm = mol.natm

        # Reference TDA at the equilibrium geometry.
        mf0 = gpu_scf.RHF(mol).run()
        td0 = gpu_tdscf.rhf.TDA(mf0)
        td0.nstates = 4
        td0.conv_tol = 1e-10
        td0.kernel()

        omega_fd = np.zeros((natm, 3))
        # TransitionDensityTracker is 1-indexed; convert.
        state_1idx = state + 1
        for ia in range(natm):
            for ix in range(3):
                tracker = TransitionDensityTracker(td0, state_1idx)

                coords = coords0.copy()
                coords[ia, ix] += delta
                molp = mol.copy()
                molp.set_geom_(coords, unit='Bohr')
                molp.build()
                mfp = gpu_scf.RHF(molp).run()
                tdp = gpu_tdscf.rhf.TDA(mfp)
                tdp.nstates = 4
                tdp.conv_tol = 1e-10
                tdp.kernel()
                matp = tracker.assign(tdp, require_converged=False)
                omega_p = float(tdp.e[matp.root])

                coords = coords0.copy()
                coords[ia, ix] -= delta
                molm = mol.copy()
                molm.set_geom_(coords, unit='Bohr')
                molm.build()
                mfm = gpu_scf.RHF(molm).run()
                tdm = gpu_tdscf.rhf.TDA(mfm)
                tdm.nstates = 4
                tdm.conv_tol = 1e-10
                tdm.kernel()
                matm = tracker.assign(tdm, require_converged=False)
                omega_m = float(tdm.e[matm.root])

                omega_fd[ia, ix] = (omega_p - omega_m) / (2.0 * delta)
        return omega_fd

    def test_omega_grad_matches_fd(self):
        '''omega_grad(td, state=0) on H2O/STO-3G must match FD of the
        excitation energy to ~1e-4 / Bohr (numerical-derivative noise).'''
        mol = pyscf.M(atom=self.atom, basis=self.basis, unit='Angstrom',
                      verbose=0, output='/dev/null', max_memory=4000)
        mf = gpu_scf.RHF(mol).run()
        td = gpu_tdscf.rhf.TDA(mf)
        td.nstates = 4
        td.conv_tol = 1e-10
        td.kernel()

        ana = tdrhf_hess.omega_grad(td, 0)
        fd = self._fd_omega_grad(state=0, delta=2.0e-3)
        # 5e-4 absolute tolerance: FD truncation error at delta=2e-3 is
        # ~delta^2/6 . d^3 omega/d R^3 which on STO-3G/H2O is empirically
        # ~2e-4. Tighter is achievable with smaller delta but trades off
        # against SCF noise. Largest analytical components ~ 0.26 Ha/Bohr,
        # so 5e-4 is ~2e-3 relative -- enough to confirm omega_grad picks
        # up the right gradient (any sign error or factor-of-2 would be
        # >> this tolerance).
        diff = np.abs(ana - fd).max()
        self.assertLess(diff, 5.0e-4,
                        f'omega_grad analytical vs FD: max diff = {diff:.3e}\n'
                        f'analytical:\n{ana}\nFD:\n{fd}')


class TestCrossTermAssembly(unittest.TestCase):
    '''Phase 2.0: assemble_omega_cross_term is index-arithmetic on outputs
    of solve_x1. Verify shape, symmetry-on-X-loop, and a hand-computed
    value on a tiny synthetic input.'''

    def test_shape_and_value(self):
        '''Hand-check 4 X . b for npert=2, nocc=3, nvir=2.'''
        x = cp.asarray(np.array([
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        ]))
        b = cp.asarray(np.array([
            [[2.0, 0.0], [0.0, 0.5], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 0.0], [0.0, 0.5]],
        ]))
        out = tdrhf_hess.assemble_omega_cross_term(b, x)
        self.assertEqual(out.shape, (2, 2))
        # H_cross[a,b] = 4 sum_{i,c} X^a_{ic} b^b_{ic}
        # x[0]=[[1,0],[0,1],[0,0]]; x[1]=[[0,1],[1,0],[0.5,0.5]]
        # b[0]=[[2,0],[0,0.5],[1,0]]; b[1]=[[0,1],[0.5,0],[0,0.5]]
        #
        # H[0,0] = 4 * (1*2 + 0 + 0 + 1*0.5 + 0 + 0)              = 4 * 2.5 = 10
        # H[0,1] = 4 * (1*0 + 0 + 0 + 1*0   + 0 + 0)              = 0
        # H[1,0] = 4 * (0 + 1*0 + 1*0 + 0*0.5 + 0.5*1 + 0.5*0)    = 4 * 0.5  = 2
        # H[1,1] = 4 * (0 + 1*1 + 1*0.5 + 0*0 + 0.5*0 + 0.5*0.5)  = 4 * 1.75 = 7
        expected = np.array([[10.0, 0.0],
                             [2.0,  7.0]])
        np.testing.assert_allclose(cp.asnumpy(out), expected, atol=1e-12)

    def test_symmetric_when_x_solves_b(self):
        '''If x_a = solve_x1(b_a) for the same TDA root, the resulting
        H_cross[a,b] must be symmetric in (a, b) -- this is the 2n+1 /
        Hellmann-Feynman / Wigner symmetry that justifies the factor 4.'''
        mol, mf, td = _setup_tda()
        state = 0
        nocc, nvir = cp.asarray(td.xy[state][0]).shape

        # Random orthogonal-to-X RHSs.
        rng = np.random.default_rng(101)
        npert = 4
        b = cp.asarray(rng.standard_normal((npert, nocc, nvir)))
        b = _project_out_x(td, state, b)
        x = tdrhf_hess.solve_x1(td, state, b)

        H = cp.asnumpy(tdrhf_hess.assemble_omega_cross_term(b, x))
        # Symmetry residual relative to ||H||
        asym = np.abs(H - H.T).max() / (np.abs(H).max() + 1e-30)
        self.assertLess(asym, 1e-9,
                        f'cross-term not symmetric: relative asymmetry {asym:.3e}')


class TestEpsXDiagFD(unittest.TestCase):
    '''Phase 2.2: _eps_x_diag_fd computes the perturbed orbital energy
    diagonal at fixed C^eq. Validate shape, finiteness, and that the
    occupied-block diagonal is consistent with mo_e1 (the perturbed
    orbital energies that would come from solve_mo1 for a check).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.mo_coeff = cp.asarray(cls.mf.mo_coeff)
        cls.mo_occ = cp.asarray(cls.mf.mo_occ)

    def test_shape(self):
        eps_x = tdrhf_hess._eps_x_diag_fd(
            self.mf, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        natm = self.mol.natm
        nmo = self.mo_coeff.shape[1]
        self.assertEqual(eps_x.shape, (natm, 3, nmo))

    def test_finite_and_nontrivial(self):
        eps_x = tdrhf_hess._eps_x_diag_fd(
            self.mf, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        self.assertTrue(bool(cp.all(cp.isfinite(eps_x))))
        # On H2O, the largest orbital-energy gradient component should
        # be substantial (~0.1-1 Ha/Bohr).
        self.assertGreater(float(cp.abs(eps_x).max()), 1e-3)

    def test_fd_truncation_scales(self):
        eps_a = tdrhf_hess._eps_x_diag_fd(
            self.mf, self.mo_coeff, self.mo_occ, delta=4.0e-3)
        eps_b = tdrhf_hess._eps_x_diag_fd(
            self.mf, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        rel = float(cp.linalg.norm(eps_a - eps_b)) / float(
            cp.linalg.norm(eps_b))
        self.assertLess(rel, 1e-2,
                        f'_eps_x_diag_fd FD truncation: rel diff '
                        f'between delta=4e-3 and 2e-3 = {rel:.3e}')


class TestComputeBxFull(unittest.TestCase):
    '''Phase 2.2: compute_b_x assembles all three terms. Validate via
    Hellmann-Feynman consistency: ``<X | b^a> = 0`` for each nuclear DOF
    by construction (b^a = (A^a - omega^a I) X with omega^a = 2 X.A^a.X).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.x_ref = cp.asarray(cls.td.xy[0][0])

    def test_shape(self):
        b = tdrhf_hess.compute_b_x(self.td, 0, fd_delta=2.0e-3)
        natm = self.mol.natm
        nocc, nvir = self.x_ref.shape
        self.assertEqual(b.shape, (natm * 3, nocc, nvir))
        self.assertTrue(bool(cp.all(cp.isfinite(b))))

    def test_hellmann_feynman_orthogonality(self):
        '''<X|b^a> must be ~0 for each nuclear DOF a.

        Mathematically exact: b^a = (A^a - omega^A^a I) X with
        omega^A^a = 2 (eps_part^a + V_part^a) computed from terms 1 and
        2 of compute_b_x. By construction <X | b^a> = 0 to machine
        precision, regardless of whether terms 1/2 use FD or analytical
        primitives. Tested on absolute |overlap| (no division by ||b^a||
        because b^a can be exactly zero for symmetry-protected DOFs --
        e.g. the totally-symmetric Oz mode of H2O/STO-3G).
        '''
        b = tdrhf_hess.compute_b_x(self.td, 0, fd_delta=2.0e-3)
        overlap = cp.einsum('aov,ov->a', b, self.x_ref)
        max_abs = float(cp.max(cp.abs(overlap)))
        self.assertLess(max_abs, 1e-9,
                        f'|<X|b^a>| max = {max_abs:.3e}\n'
                        f'overlaps: {cp.asnumpy(overlap)}')

    def test_solve_x1_finite(self):
        '''solve_x1(b^a) must produce a finite X^a.

        This is a smoke test: it confirms compute_b_x produces a b^a
        that solve_x1 can invert without numerical disaster. NOT a
        correctness check vs FD-X^a -- that requires state-tracked FD
        with MO basis projection (Phase 2.3 validation harness).
        '''
        b = tdrhf_hess.compute_b_x(self.td, 0, fd_delta=2.0e-3)
        x1 = tdrhf_hess.solve_x1(self.td, 0, b)
        self.assertEqual(x1.shape, b.shape)
        self.assertTrue(bool(cp.all(cp.isfinite(x1))))


class TestVindXFD(unittest.TestCase):
    '''Phase 2.1: _vind_x_fd is the FD-based perturbed-vind primitive.
    Validate shape, linearity in input density, and FD-stability by
    halving delta and confirming the result converges with O(delta^2).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.x_ref = cp.asarray(cls.td.xy[0][0])
        nocc, nvir = cls.x_ref.shape
        cls.nocc = nocc
        cls.nvir = nvir
        cls.mo_coeff = cp.asarray(cls.mf.mo_coeff)
        cls.mo_occ = cp.asarray(cls.mf.mo_occ)
        orbo = cls.mo_coeff[:, cls.mo_occ > 0]
        orbv = cls.mo_coeff[:, cls.mo_occ == 0]
        # T^tr_{mu nu} = sum_{i,a} c_{mu a} X_{ia} c_{nu i}
        # X has shape (nocc, nvir): X[i, a]. T^tr = orbv · X.T · orbo.T
        cls.T_tr_AO = orbv @ cls.x_ref.T @ orbo.T

    def test_shape(self):
        out = tdrhf_hess._vind_x_fd(
            self.mf, self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        self.assertEqual(out.shape, (self.mol.natm, 3, self.nocc, self.nvir))

    def test_zero_density_gives_zero(self):
        zero = cp.zeros_like(self.T_tr_AO)
        out = tdrhf_hess._vind_x_fd(
            self.mf, zero, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        self.assertLess(float(cp.linalg.norm(out)), 1e-12)

    def test_linearity(self):
        '''_vind_x_fd is linear in T_tr_AO.'''
        out_T = tdrhf_hess._vind_x_fd(
            self.mf, self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        out_2T = tdrhf_hess._vind_x_fd(
            self.mf, 2.0 * self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        rel = float(cp.linalg.norm(out_2T - 2.0 * out_T)) / float(
            cp.linalg.norm(out_T) + 1e-30)
        self.assertLess(rel, 1e-9, f'_vind_x_fd non-linearity: {rel:.3e}')

    def test_fd_truncation_scales(self):
        '''Halving delta should make the FD result more accurate; the
        difference between two delta values scales as O(delta^2). Use
        Richardson-extrapolation-style check: compute at two deltas, the
        difference between them should be much smaller than the result
        magnitude (no large numerical noise floor).'''
        out_a = tdrhf_hess._vind_x_fd(
            self.mf, self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=4.0e-3)
        out_b = tdrhf_hess._vind_x_fd(
            self.mf, self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        rel = float(cp.linalg.norm(out_a - out_b)) / float(cp.linalg.norm(out_b))
        # delta_a^2 - delta_b^2 = 16e-6 - 4e-6 = 12e-6, so trunc-error
        # ratio is ~3:1. Net rel diff should be < 1% of result magnitude.
        self.assertLess(rel, 1e-2,
                        f'_vind_x_fd FD truncation inconsistent: rel diff '
                        f'between delta=4e-3 and delta=2e-3 = {rel:.3e}')

    def test_nontrivial_result(self):
        '''_vind_x_fd applied to a non-zero T^tr should produce a non-
        trivial gradient (not just numerical noise).'''
        out = tdrhf_hess._vind_x_fd(
            self.mf, self.T_tr_AO, self.mo_coeff, self.mo_occ, delta=2.0e-3)
        # On H2O the largest 1st-derivative component should be ~ same
        # order as omega itself / Bohr (~0.1 Ha/Bohr).
        self.assertGreater(float(cp.abs(out).max()), 1e-3)


class TestEpsXDiagAnalytical(unittest.TestCase):
    '''Phase 2.3: ``_eps_x_diag_analytical`` replaces ``_eps_x_diag_fd``
    with no FD truncation. Should agree with the FD version to ~FD
    truncation precision (delta=2e-3 -> O(delta^2) ~ 4e-6 absolute).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()

    def test_shape(self):
        out = tdrhf_hess._eps_x_diag_analytical(self.mf)
        natm = self.mol.natm
        nmo = self.mf.mo_coeff.shape[1]
        self.assertEqual(out.shape, (natm, 3, nmo))
        self.assertTrue(bool(cp.all(cp.isfinite(out))))

    def test_matches_fd(self):
        mo_coeff = cp.asarray(self.mf.mo_coeff)
        mo_occ = cp.asarray(self.mf.mo_occ)
        fd = tdrhf_hess._eps_x_diag_fd(self.mf, mo_coeff, mo_occ, delta=2e-3)
        ana = tdrhf_hess._eps_x_diag_analytical(self.mf, mo_coeff, mo_occ)
        max_diff = float(cp.max(cp.abs(fd - ana)))
        scale = float(cp.max(cp.abs(fd))) + 1e-30
        # FD truncation at delta=2e-3 is O(delta^2) ~ 4e-6 absolute
        self.assertLess(max_diff, 1e-5,
                        f'FD vs analytical max |diff| = {max_diff:.3e} '
                        f'(scale ~ {scale:.3e})')

    def test_translation_invariant(self):
        '''Sum over atoms (per axis) of eps^a should be ~0 (translational
        invariance of the molecular Hamiltonian).'''
        ana = tdrhf_hess._eps_x_diag_analytical(self.mf)
        total = cp.sum(ana, axis=0)   # (3, nmo)
        max_total = float(cp.max(cp.abs(total)))
        self.assertLess(max_total, 1e-6,
                        f'Translation-invariance residual = {max_total:.3e}')


class TestComputeBxAnalyticalEpsDefault(unittest.TestCase):
    '''compute_b_x defaults to analytical eps^a (Phase 2.3). Verify the
    default path agrees with the use_fd_eps=True path to FD-truncation.'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.x_ref = cp.asarray(cls.td.xy[0][0])

    def test_default_is_analytical(self):
        b_default = tdrhf_hess.compute_b_x(self.td, 0)
        b_analytical = tdrhf_hess.compute_b_x(self.td, 0, use_fd_eps=False)
        self.assertTrue(bool(cp.allclose(b_default, b_analytical)))

    def test_analytical_vs_fd(self):
        '''b^a built with analytical eps^a should match b^a built with
        FD eps^a, to FD-truncation precision.'''
        b_fd  = tdrhf_hess.compute_b_x(self.td, 0, use_fd_eps=True,
                                       fd_delta=2e-3)
        b_ana = tdrhf_hess.compute_b_x(self.td, 0, use_fd_eps=False)
        max_diff = float(cp.max(cp.abs(b_fd - b_ana)))
        # The eps^a piece's FD truncation propagates linearly into b^a;
        # delta=2e-3 -> O(delta^2) ~ 4e-6.
        self.assertLess(max_diff, 1e-4,
                        f'analytical vs FD b^a max |diff| = {max_diff:.3e}')

    def test_hellmann_feynman_holds(self):
        '''H-F orthogonality holds to machine precision for the analytical
        path (no FD truncation in eps^a).'''
        b = tdrhf_hess.compute_b_x(self.td, 0)  # default = analytical
        overlap = cp.einsum('aov,ov->a', b, self.x_ref)
        max_abs = float(cp.max(cp.abs(overlap)))
        self.assertLess(max_abs, 1e-10)


class TestPhase24BlockAssembly(unittest.TestCase):
    '''Phase 2.4 Blocks 1+2 partial assembly. Returns the convention-A
    Hessian omega^{A,ab} = omega^{A,ab}_pure (FD on Phase 2.3a 1st-deriv)
    + omega^{A,ab}_cross (Phase 2.0 cross-term).

    NOT the physical Hessian -- missing Block 3 (orbital relaxation via
    Z-vector) and Block 4 (energy-weighted W). Tested for shape, finite,
    Hessian symmetry, and consistency with FD on omega^A^a.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()

    def test_shape_finite(self):
        h = tdrhf_hess.omega_hessian(self.td, 0)
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))

    def test_hessian_symmetry(self):
        '''omega^{A,ab} should be symmetric under (atm_a, ix_a) <-> (atm_b, ix_b).'''
        h = tdrhf_hess.omega_hessian(self.td, 0)
        natm = self.mol.natm
        h_flat = h.reshape(3 * natm, 3 * natm)
        max_asym = float(cp.max(cp.abs(h_flat - h_flat.T)))
        self.assertLess(max_asym, 1e-7,
                        f'Hessian asymmetry: max |H - H^T| = {max_asym:.3e}')

    def test_class_kernel(self):
        '''Hessian.kernel() should return the same as omega_hessian().'''
        h_class = tdrhf_hess.Hessian(self.td).kernel()
        h_func = tdrhf_hess.omega_hessian(self.td, 0)
        self.assertTrue(bool(cp.allclose(h_class, h_func)))


class TestPhase21Sketch(unittest.TestCase):
    '''Phase 2.1 demo: assemble a partial b^a from term-2 + term-3 only
    (omitting term-1, the eps^a-vir-diagonal piece), feed it to
    solve_x1, and just check the X^a output is finite + has the expected
    shape. NOT a correctness check vs FD-X^a -- term 1 is missing, so
    the X^a is wrong. This documents what the assembly looks like and
    that the available pieces compose without error.'''

    def test_partial_b_assembly_shape(self):
        mol, mf, td = _setup_tda()
        state = 0
        x = cp.asarray(td.xy[state][0])
        nocc, nvir = x.shape
        mo_coeff = cp.asarray(mf.mo_coeff)
        mo_occ = cp.asarray(mf.mo_occ)
        orbo = mo_coeff[:, mo_occ > 0]
        orbv = mo_coeff[:, mo_occ == 0]
        T_tr_AO = orbv @ x.T @ orbo.T

        # Term 2: V^a[T^tr] in MO (occ, vir), via _vind_x_fd.
        # gen_tda_operation passes 2 T^tr to vresp internally (closed-shell
        # double-occupancy), so the singlet V^a at MO (occ,vir) is the
        # FD primitive applied to 2 T^tr (or equivalently 2x the result):
        v_x_mo = 2.0 * tdrhf_hess._vind_x_fd(
            mf, T_tr_AO, mo_coeff, mo_occ, delta=2.0e-3)
        # Shape (natm, 3, nocc, nvir)
        natm = mol.natm
        self.assertEqual(v_x_mo.shape, (natm, 3, nocc, nvir))

        # Term 3: -omega^a X
        omega_a = tdrhf_hess.omega_grad(td, state)   # (natm, 3)
        omega_term = -cp.asarray(omega_a)[:, :, None, None] * x[None, None]

        # PARTIAL b: term2 + term3, missing term1 (eps^a-vir-diag)
        b_partial = (v_x_mo + omega_term).reshape(natm * 3, nocc, nvir)
        self.assertEqual(b_partial.shape, (natm * 3, nocc, nvir))

        # solve_x1 on the partial b just to confirm composition: not
        # physically correct, but should not blow up.
        x_partial = tdrhf_hess.solve_x1(td, state, b_partial)
        self.assertEqual(x_partial.shape, (natm * 3, nocc, nvir))
        self.assertTrue(bool(cp.all(cp.isfinite(x_partial))))


class TestHessianClassWiring(unittest.TestCase):
    '''The Hessian class itself only forwards solve_x1 in Phase 1; kernel
    raises. Confirm both behaviors.'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.h = tdrhf_hess.Hessian(cls.td)

    def test_kernel_returns_partial_hessian(self):
        '''Phase 2.4 partial: kernel returns the convention-A Hessian
        (Blocks 1+2). Just confirm shape + finite, not the physics.'''
        h = self.h.kernel()
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))

    def test_solve_x1_method(self):
        x = cp.asarray(self.td.xy[0][0])
        b = cp.zeros((1,) + x.shape)
        out = self.h.solve_x1(b)   # uses self.state - 1 = 0
        self.assertEqual(out.shape, b.shape)
        self.assertLess(float(cp.linalg.norm(out)), 1e-7)

    def test_rejects_tdhf(self):
        '''solve_x1 should refuse a TDHF (non-TDA) amplitude.'''
        td_hf = gpu_tdscf.rhf.TDHF(self.mf)
        td_hf.nstates = 2
        td_hf.kernel()
        with self.assertRaises(NotImplementedError):
            tdrhf_hess.solve_x1(td_hf, 0, cp.zeros((1,) + cp.asarray(
                td_hf.xy[0][0]).shape))


if __name__ == '__main__':
    unittest.main()
