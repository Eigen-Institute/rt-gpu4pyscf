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
Unit tests for gpu4pyscf.hessian.tdrhf.

Covers the full pipeline:
  solve_z_vector -> make_cptddft_rhs -> solve_cptddft ->
  make_intermediates -> make_perturbed_intermediates ->
  omega_grad -> omega_hessian (semi-analytical) ->
  analytical_omega_hessian -> Hessian class
'''

import unittest
import numpy as np
import cupy as cp
import pyscf

import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from gpu4pyscf import tdscf as gpu_tdscf
from gpu4pyscf.hessian import tdrhf as tdrhf_hess
from gpu4pyscf.hessian import rhf as rhf_hess_gpu


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


def _build_mo1_ux(mf, mol):
    '''Build ground-state MO responses (mo1, mo_e1) and full Ux matrix.

    mo1  : (natm, 3, nao, nocc)
    mo_e1: perturbed orbital energies from solve_mo1
    Ux   : (natm, 3, nao, nao) full MO rotation matrix
    '''
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nocc = int((mo_occ > 0).sum())
    nao = mol.nao
    natm = mol.natm

    mf_hess = rhf_hess_gpu.Hessian(mf)
    h1mo = mf_hess.make_h1(mo_coeff, mo_occ)
    fx = mf_hess.gen_vind(mo_coeff, mo_occ)
    mo1, mo_e1 = mf_hess.solve_mo1(mo_energy, mo_coeff, mo_occ, h1mo, fx)
    mo1 = cp.asarray(mo1)

    _, _, s1a_basis = rhf_hess_gpu.get_ovlp(mol)
    s1a_basis = cp.asarray(s1a_basis)
    aoslices = mol.aoslice_by_atom()

    s1ao = cp.zeros((natm, 3, nao, nao))
    for atm_id in range(natm):
        p0, p1 = aoslices[atm_id][2:]
        s1ao[atm_id, :, p0:p1] += s1a_basis[:, p0:p1]
        s1ao[atm_id, :, :, p0:p1] += s1a_basis[:, p0:p1].transpose(0, 2, 1)

    s1mo = cp.zeros((natm, 3, nao, nao))
    for i in range(natm):
        for j in range(3):
            s1mo[i, j] = mo_coeff.T @ s1ao[i, j] @ mo_coeff

    Ux = cp.zeros((natm, 3, nao, nao))
    Ux[:, :, :, :nocc] = mo1
    Ux[:, :, :nocc, nocc:] = (-s1mo[:, :, :nocc, nocc:]
                               - mo1[:, :, nocc:, :].transpose(0, 1, 3, 2))
    Ux[:, :, :nocc, :nocc] = -0.5 * s1mo[:, :, :nocc, :nocc]
    Ux[:, :, nocc:, nocc:] = -0.5 * s1mo[:, :, nocc:, nocc:]

    return mo1, mo_e1, Ux, s1mo


# ---------------------------------------------------------------------------
# Z-vector
# ---------------------------------------------------------------------------

class TestSolveZVector(unittest.TestCase):
    '''solve_z_vector returns the Z-vector of shape (nvir, nocc).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0
        mo_occ = cp.asarray(cls.mf.mo_occ)
        cls.nocc = int((mo_occ > 0).sum())
        cls.nvir = mo_occ.shape[0] - cls.nocc
        from gpu4pyscf.grad import tdrhf as tdrhf_grad
        cls.td_grad_obj = tdrhf_grad.Gradients(cls.td)
        cls.x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in cls.td.xy[cls.state]])

    def test_shape(self):
        z1 = tdrhf_hess.solve_z_vector(self.td_grad_obj, self.x_y)
        self.assertEqual(z1.shape, (self.nocc, self.nvir))

    def test_finite_and_nontrivial(self):
        z1 = tdrhf_hess.solve_z_vector(self.td_grad_obj, self.x_y)
        self.assertTrue(bool(cp.all(cp.isfinite(z1))))
        self.assertGreater(float(cp.abs(z1).max()), 1e-6)


# ---------------------------------------------------------------------------
# CP-TDDFT RHS
# ---------------------------------------------------------------------------

class TestMakeCptddftRhs(unittest.TestCase):
    '''make_cptddft_rhs returns Delta, Upsilon of shape (natm, 3, nocc, nvir).'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0
        mo_occ = cp.asarray(cls.mf.mo_occ)
        cls.nocc = int((mo_occ > 0).sum())
        cls.nvir = mo_occ.shape[0] - cls.nocc
        cls.x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in cls.td.xy[cls.state]])
        cls.omega = float(cls.td.e[cls.state])
        cls.h = tdrhf_hess.Hessian(cls.td)
        cls.mo1, cls.mo_e1, cls.Ux, cls.s1mo = _build_mo1_ux(cls.mf, cls.mol)

    def test_shape(self):
        Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1)
        natm = self.mol.natm
        self.assertEqual(Delta.shape, (natm, 3, self.nocc, self.nvir))

        self.assertEqual(Upsilon.shape, (natm, 3, self.nocc, self.nvir))

    def test_finite(self):
        Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1)
        self.assertTrue(bool(cp.all(cp.isfinite(Delta))))
        self.assertTrue(bool(cp.all(cp.isfinite(Upsilon))))

    def test_nontrivial(self):
        Delta, Upsilon = tdrhf_hess.make_cptddft_rhs(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1)
        self.assertGreater(float(cp.abs(Delta).max()), 1e-6)


# ---------------------------------------------------------------------------
# CP-TDDFT solver
# ---------------------------------------------------------------------------

class TestSolveCptddft(unittest.TestCase):
    '''solve_cptddft returns x1, y1 of shape (natm, 3, nocc, nvir).
    For TDA, y1 is zero by construction.'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0
        mo_occ = cp.asarray(cls.mf.mo_occ)
        cls.nocc = int((mo_occ > 0).sum())
        cls.nvir = mo_occ.shape[0] - cls.nocc
        cls.x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in cls.td.xy[cls.state]])
        cls.omega = float(cls.td.e[cls.state])
        cls.h = tdrhf_hess.Hessian(cls.td)
        cls.mo1, cls.mo_e1, cls.Ux, cls.s1mo = _build_mo1_ux(cls.mf, cls.mol)

    def test_shape(self):
        x1, y1 = tdrhf_hess.solve_cptddft(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1, self.s1mo)
        natm = self.mol.natm
        self.assertEqual(x1.shape, (natm, 3, self.nocc, self.nvir))
        self.assertEqual(y1.shape, (natm, 3, self.nocc, self.nvir))

    def test_tda_y1_is_zero(self):
        x1, y1 = tdrhf_hess.solve_cptddft(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1, self.s1mo)
        self.assertLess(float(cp.linalg.norm(y1)), 1e-14)

    def test_x1_finite_and_nontrivial(self):
        x1, y1 = tdrhf_hess.solve_cptddft(
            self.h, self.x_y, self.omega, self.mo1, self.mo_e1, self.s1mo)
        self.assertTrue(bool(cp.all(cp.isfinite(x1))))
        self.assertGreater(float(cp.abs(x1).max()), 1e-8)


# ---------------------------------------------------------------------------
# Unperturbed intermediates
# ---------------------------------------------------------------------------

class TestMakeIntermediates(unittest.TestCase):
    '''make_intermediates returns the correct keys and (nao, nao) shapes.'''

    _EXPECTED_KEYS = ('P_I', 'R_I', 'T_I', 'P_I_prime',
                      'W_I', 'P', 'F_AO')

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0
        cls.x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in cls.td.xy[cls.state]])
        cls.h = tdrhf_hess.Hessian(cls.td)
        from gpu4pyscf.grad import tdrhf as tdrhf_grad
        td_grad_obj = tdrhf_grad.Gradients(cls.td)
        cls.z1 = tdrhf_hess.solve_z_vector(td_grad_obj, cls.x_y)
        cls.d = tdrhf_hess.make_intermediates(cls.h, cls.x_y, cls.z1)

    def test_keys_present(self):
        for k in self._EXPECTED_KEYS:
            self.assertIn(k, self.d)

    def test_shapes(self):
        nao = self.mol.nao
        for k in self._EXPECTED_KEYS:
            self.assertEqual(self.d[k].shape, (nao, nao), msg=f'key {k}')

    def test_all_finite(self):
        for k, v in self.d.items():
            self.assertTrue(bool(cp.all(cp.isfinite(v))),
                            msg=f'key {k} contains non-finite values')

    def test_ground_state_density_symmetric(self):
        P = self.d['P']
        asym = float(cp.max(cp.abs(P - P.T)))
        self.assertLess(asym, 1e-10,
                        f'Ground-state density P is not symmetric: {asym:.3e}')

    def test_P_I_prime_hermitian(self):
        '''P_I_prime = P_I + 0.5*(z1-part) should be symmetric for real MOs.'''
        P_I_prime = self.d['P_I_prime']
        asym = float(cp.max(cp.abs(P_I_prime - P_I_prime.T)))
        self.assertLess(asym, 1e-10,
                        f'P_I_prime not symmetric: {asym:.3e}')


# ---------------------------------------------------------------------------
# Perturbed intermediates
# ---------------------------------------------------------------------------

class TestMakePerturbedIntermediates(unittest.TestCase):
    '''make_perturbed_intermediates returns (natm, 3, nmo, nmo) arrays.'''

    _EXPECTED_KEYS = ('P_y_MO', 'R_I_y_MO', 'T_I_y_MO', 'P_I_prime_y_MO',
                      'W_I_y_MO', 'G_x_PI_AO_integral', 'Gp_x_RI_AO',
                      'Gm_x_TI_AO', 'F_x_AO_integral')

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.state = 0
        cls.x_y = tuple([cp.asarray(v) * cp.sqrt(2) for v in cls.td.xy[cls.state]])
        cls.omega = float(cls.td.e[cls.state])
        cls.h = tdrhf_hess.Hessian(cls.td)
        from gpu4pyscf.grad import tdrhf as tdrhf_grad
        td_grad_obj = tdrhf_grad.Gradients(cls.td)
        cls.z1 = tdrhf_hess.solve_z_vector(td_grad_obj, cls.x_y)
        cls.mo1, cls.mo_e1, cls.Ux, cls.s1mo = _build_mo1_ux(cls.mf, cls.mol)
        cls.x1, cls.y1 = tdrhf_hess.solve_cptddft(
            cls.h, cls.x_y, cls.omega, cls.mo1, cls.mo_e1, cls.s1mo)
        cls.intermediates = tdrhf_hess.make_intermediates(cls.h, cls.x_y, cls.z1)
        cls.d = tdrhf_hess.make_perturbed_intermediates(
            cls.h, cls.intermediates, cls.x_y, cls.x1, cls.y1, cls.Ux, cls.z1, cls.s1mo)

    def test_keys_present(self):
        for k in self._EXPECTED_KEYS:
            self.assertIn(k, self.d)

    def test_shapes(self):
        nao = self.mol.nao
        natm = self.mol.natm
        for k in self._EXPECTED_KEYS:
            self.assertEqual(self.d[k].shape, (natm, 3, nao, nao), msg=f'key {k}')

    def test_all_finite(self):
        for k, v in self.d.items():
            self.assertTrue(bool(cp.all(cp.isfinite(v))),
                            msg=f'key {k} contains non-finite values')


# ---------------------------------------------------------------------------
# Excitation-energy gradient
# ---------------------------------------------------------------------------

class TestOmegaGrad(unittest.TestCase):
    '''omega_grad returns the analytical excitation-energy gradient.
    Validate against FD of the excitation energy.'''

    @classmethod
    def setUpClass(cls):
        cls.atom = H2O_ATOM
        cls.basis = 'sto-3g'

    def _fd_omega_grad(self, state=0, delta=2.0e-3):
        from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker

        mol = pyscf.M(atom=self.atom, basis=self.basis, unit='Angstrom',
                      verbose=0, output='/dev/null', max_memory=4000)
        coords0 = mol.atom_coords(unit='Bohr').copy()
        natm = mol.natm

        mf0 = gpu_scf.RHF(mol).run()
        td0 = gpu_tdscf.rhf.TDA(mf0)
        td0.nstates = 4
        td0.conv_tol = 1e-10
        td0.kernel()

        omega_fd = np.zeros((natm, 3))
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
        mol = pyscf.M(atom=self.atom, basis=self.basis, unit='Angstrom',
                      verbose=0, output='/dev/null', max_memory=4000)
        mf = gpu_scf.RHF(mol).run()
        td = gpu_tdscf.rhf.TDA(mf)
        td.nstates = 4
        td.conv_tol = 1e-10
        td.kernel()

        ana = tdrhf_hess.omega_grad(td, 0)
        fd = self._fd_omega_grad(state=0, delta=2.0e-3)
        # Absolute tolerance ~5e-4: FD truncation at delta=2e-3 is O(delta^2)
        # on H2O/STO-3G. Any sign error or factor-of-2 bug would be >> this.
        diff = np.abs(ana - fd).max()
        self.assertLess(diff, 5.0e-4,
                        f'omega_grad analytical vs FD: max diff = {diff:.3e}\n'
                        f'analytical:\n{ana}\nFD:\n{fd}')


# ---------------------------------------------------------------------------
# Semi-analytical Hessian
# ---------------------------------------------------------------------------

class TestSemiAnalyticalHessian(unittest.TestCase):
    '''omega_hessian (FD on analytical gradient) shape and symmetry.'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()

    def test_shape_finite(self):
        h = tdrhf_hess.omega_hessian(self.td, 0)
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))

    def test_hessian_symmetry(self):
        h = tdrhf_hess.omega_hessian(self.td, 0)
        natm = self.mol.natm
        h_flat = h.reshape(3 * natm, 3 * natm)
        max_asym = float(cp.max(cp.abs(h_flat - h_flat.T)))
        self.assertLess(max_asym, 1e-7,
                        f'Semi-analytical Hessian asymmetry: {max_asym:.3e}')


# ---------------------------------------------------------------------------
# Analytical Hessian
# ---------------------------------------------------------------------------

class TestAnalyticalHessian(unittest.TestCase):
    '''analytical_omega_hessian structural checks.

    Structural test only (shape, finiteness). Numerical agreement vs the
    semi-analytical baseline is covered by TestAnalyticalAgreement.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()

    def test_shape_finite(self):
        h_obj = tdrhf_hess.Hessian(self.td)
        h_obj.method = 'analytical'
        h = h_obj.kernel()
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))


# ---------------------------------------------------------------------------
# Analytical Agreement
# ---------------------------------------------------------------------------

class TestAnalyticalAgreement(unittest.TestCase):
    '''Verify analytical vs semi-analytical agreement on small systems.'''

    def test_h2_sto3g_tda(self):
        import sys
        print(f"DEBUG PATH: {sys.path}")
        import gpu4pyscf.hessian.tdrhf as tdrhf_debug
        print(f"DEBUG FILE: {tdrhf_debug.__file__}")
        mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = gpu_scf.RHF(mol).run()
        td = gpu_tdscf.rhf.TDA(mf)
        td.kernel()

        h_obj = tdrhf_hess.Hessian(td)
        h_obj.method = 'analytical'
        h_ana = h_obj.kernel()

        h_obj.method = 'semi-analytical'
        h_semi = h_obj.kernel()

        diff = float(cp.max(cp.abs(h_ana - h_semi)))
        if diff > 1e-4:
            print(f"DEBUG: h_ana[0,:,0,:]:\n{h_ana[0,:,0,:]}")
            print(f"DEBUG: h_semi[0,:,0,:]:\n{h_semi[0,:,0,:]}")
        self.assertLess(diff, 1e-4, f"H2/STO-3G max diff: {diff:.3e}")

    def test_h2_sto3g_ti(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = gpu_scf.RHF(mol).run()
        td = gpu_tdscf.rhf.TDA(mf)
        td.kernel()

        h_obj = tdrhf_hess.Hessian(td)
        h_obj.method = 'analytical'
        h = h_obj.kernel()

        # h.shape is (natm, 3, natm, 3)
        # Sum over first natm (axis 0) or second natm (axis 2) should be zero.
        ti_err = float(cp.abs(h.sum(axis=0)).max())
        self.assertLess(ti_err, 1e-6, f"H2/STO-3G TI error: {ti_err:.3e}")
# ---------------------------------------------------------------------------
# Hessian class wiring
# ---------------------------------------------------------------------------

class TestHessianClass(unittest.TestCase):
    '''Hessian class: default method, kernel dispatch, helper methods.'''

    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf, cls.td = _setup_tda()
        cls.h = tdrhf_hess.Hessian(cls.td)

    def test_default_method_is_semi_analytical(self):
        self.assertEqual(self.h.method, 'semi-analytical')

    def test_kernel_semi_analytical_returns_hessian(self):
        h = self.h.kernel()
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))

    def test_kernel_analytical_returns_hessian(self):
        h_obj = tdrhf_hess.Hessian(self.td)
        h_obj.method = 'analytical'
        h = h_obj.kernel()
        natm = self.mol.natm
        self.assertEqual(h.shape, (natm, 3, natm, 3))
        self.assertTrue(bool(cp.all(cp.isfinite(h))))

    def test_kernel_semi_matches_omega_hessian(self):
        h_class = self.h.kernel()
        h_func = tdrhf_hess.omega_hessian(self.td, 0)
        self.assertTrue(bool(cp.allclose(h_class, h_func)))

    def test_omega_grad_method(self):
        g = self.h.omega_grad()
        self.assertEqual(g.shape, (self.mol.natm, 3))
        self.assertTrue(np.all(np.isfinite(g)))

    def test_hess_alias(self):
        self.assertEqual(self.h.hess, self.h.kernel)


if __name__ == '__main__':
    unittest.main()
