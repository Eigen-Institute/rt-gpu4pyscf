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
Analytical excited-state Hessian for closed-shell TDA on RHF reference.

Status: PHASE 1 + PHASE 2.0. Phase 1 (perturbed amplitude response solver
``solve_x1``) and Phase 2.0 (analytical ``omega_grad``, cross-term
assembly primitive ``assemble_omega_cross_term``, and the b^a stub with
formula) are shipped. The full ``b^a`` construction and Hessian assembly
remain Phase 2.1+; calling ``Hessian.kernel`` still raises
``NotImplementedError`` because the b^a integral-derivative piece needs
new perturbed-vind infrastructure that isn't yet a primitive.

Phase 1 derivation
==================

Closed-shell singlet TDA on an RHF reference. The TDA eigenvalue equation:

    A(R) X(R) = omega(R) X(R),    with normalization  2 X^T X = 1

where (RHF, singlet kernel)

    A_{ia,jb} = (eps_a - eps_i) delta_{ij} delta_{ab} + 2 (ia|jb) - (ij|ab)

and where MOs and orbital energies depend on R through SCF: C(R), eps(R).

Differentiating the eigenvalue equation with respect to a nuclear DOF R_x
(letting superscript x denote total derivative wrt R_x):

    A^x X + A X^x = omega^x X + omega X^x.

Left-multiplying by X^T (and using A^T = A, X^T X = 1/2 normalization
absorbed -- see below):

    omega^x = (X^T A^x X) / (X^T X) = 2 X^T A^x X    (since 2 X^T X = 1).

Subtracting:

    (A - omega I) X^x = -(A^x - omega^x I) X = -b^x.

The constraint X^T X^x = 0 follows from differentiating 2 X^T X = 1.

Two notes on normalization. (1) The stored amplitudes in gpu4pyscf satisfy
``2 sum(X**2) = 1`` (the factor of 2 is closed-shell double occupancy, see
``tdscf/rhf.py::TDA.kernel``: ``xy = [(xi.reshape(nocc,nvir) * .5**.5, 0)]``).
So X^T X = 1/2, and the projector onto the X mode is

    P_X v = X (X^T v) / (X^T X) = 2 X (X^T v).

(2) (A - omega I) is singular -- X is in its kernel. The system is consistent
because ``b^x`` is automatically orthogonal to X:

    X^T b^x = X^T (A^x - omega^x I) X = (X^T A^x X) - omega^x (X^T X)
            = (omega^x / 2) - (omega^x / 2) = 0.

So the unique solution X^x with X^T X^x = 0 exists. We obtain it by
solving (A - omega I) X^x = -b^x in the orthogonal complement of X
(deflation).

Phase 1 deliverable: this solver. Phase 2: build b^x from CP-SCF U^x +
integral first derivatives, then assemble the Hessian. See
``cpscf_init.md`` in the repo root.

Phase 2 Hessian formula
=======================

Total Hessian of the excited-state energy E_I = E_GS + omega_I:

    H^I_{ab} = H^GS_{ab} + omega^{ab}_I

with H^GS_{ab} the existing GS RHF Hessian and (by the 2n+1 rule applied
to the variational TDA Lagrangian, with X normalized 2 X^T X = 1):

    omega^{ab} = 2 X^T A^{ab} X + 4 X^T_b b^a
                                   ^^^^^^^^^^
                                   cross term, symmetric in (a, b)

where X^x = solve_x1(b^x) is the perturbed amplitude (Phase 1) and:

    b^a := (A^a - omega^a I) X.

The factor 4 = 2 (from singlet/normalization) x 2 (from the
double-derivative symmetrization a<->b) is the Furche-Ahlrichs
convention for closed-shell singlet TDA on RHF. The first term is
the explicit 2nd-derivative-integral contraction with the transition
density. The second is the new cross term using Phase 1's solver.

Phase 2.0 ships: omega_grad (analytical scalar gradient), the cross-term
assembler ``assemble_omega_cross_term``, and the b^a stub with formula.
Phase 2.1 ships: the b^a builder (needs perturbed-vind integral
infrastructure) and the explicit 2nd-derivative-integral term.

References
----------
- van Caillie & Amos, Chem. Phys. Lett. 308, 249 (1999); 317, 159 (2000).
- Furche & Ahlrichs, J. Chem. Phys. 117, 7433 (2002) (gradient -- the
  Lagrangian extends to give the Hessian).
- Send & Furche, J. Chem. Phys. 132, 044107 (2010) (RPA second derivatives).
- Liu, Furche et al., J. Chem. Phys. 154, 074104 (2021) (modern formulation
  including spin-flip).
'''

import cupy as cp
import numpy as np
from pyscf import lib

from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils


# Direct-solve dimension cap. For nov above this, ``solve_x1`` raises so the
# caller knows to wire up an iterative solver. Direct-solve is correct and
# fast for everything below; ~1000 keeps memory under ~10 MB for the explicit
# matrix and is comfortably above any small-molecule test.
_DIRECT_SOLVE_NOV_LIMIT = 4000


def _get_zeroth_order(td, state):
    '''Pull the zeroth-order quantities needed to assemble the X^x equation.

    Returns
    -------
    x : (nocc, nvir) cupy array
        The TDA amplitude as stored in ``td.xy[state]`` (so 2 x.x = 1).
    omega : float
        The TDA excitation energy ``td.e[state]``.
    '''
    x_y = td.xy[state]
    if x_y[1] is not None and not (isinstance(x_y[1], (int, float)) and x_y[1] == 0):
        raise NotImplementedError(
            'hessian/tdrhf: Phase 1 supports TDA only (Y == 0). '
            'Got non-zero Y; use TDA, not TDHF/RPA.')
    x = cp.asarray(x_y[0])
    omega = float(td.e[state])
    return x, omega


def solve_x1(td, state, b, regularization=1.0, verbose=None):
    '''Solve the perturbed TDA-amplitude equation

        (A - omega I) X^x = -b

    with the orthogonality constraint X^T X^x = 0, for closed-shell TDA on
    an RHF reference.

    The system is exactly singular at the converged TDA root (X is in the
    kernel of (A - omega I)) so we solve the regularized non-singular form

        M' X^x = -b,    M' = (A - omega I) + alpha P_X,

    where P_X = X X^T / (X^T X) projects onto the X mode and alpha > 0 is a
    regularization. For b orthogonal to X (which holds exactly for any RHS
    of the form b = (A^x - omega^x I) X) the X-component of the solution is
    zero and M' x = -b returns exactly the correct X-perp solution
    independent of alpha. We default to alpha = 1 (numerically benign).

    Parameters
    ----------
    td : ``gpu4pyscf.tdscf.rhf.TDA`` (must be converged)
        Provides the operator A via ``td.get_ab()`` and the amplitudes via
        ``td.xy[state]``.
    state : int
        0-indexed root.
    b : cupy.ndarray of shape (npert, nocc, nvir) or (nocc, nvir)
        Right-hand side. For Phase 2 use, this is built as
        b^x = (A^x - omega^x I) X. The caller is responsible for ensuring
        X^T b^x = 0 (it is mathematically exact, see the module docstring).
    regularization : float
        alpha in M' = (A - omega) + alpha P_X. Default 1.0.
    verbose : logger or int

    Returns
    -------
    x1 : cupy.ndarray of shape (npert, nocc, nvir)
        Solution X^x with X^T X^x = 0.

    Notes
    -----
    Phase 1 implementation: build (A - omega I) explicitly via
    ``td.get_ab()`` and solve via cupy.linalg.solve. Correct for any size
    that fits in memory; raises if nov > 4000 (caller is responsible for
    wiring up an iterative deflated solver in that regime). The explicit
    matrix is the same one used by the (CPU-fallback) FD validation in the
    gradient tests, so any A-construction issue would already be exposed
    elsewhere.
    '''
    log = logger.new_logger(td, verbose)
    t0 = log.init_timer()

    x_ref, omega = _get_zeroth_order(td, state)
    nocc, nvir = x_ref.shape
    nov = nocc * nvir

    if nov > _DIRECT_SOLVE_NOV_LIMIT:
        raise NotImplementedError(
            f'solve_x1: nov = {nov} exceeds the direct-solve limit '
            f'({_DIRECT_SOLVE_NOV_LIMIT}). An iterative deflated solver is '
            'a Phase 1.5 follow-up. For now, smaller test cases only.')

    # Stored amplitudes satisfy 2 sum(X**2) = 1 (closed-shell), so <X|X> = 0.5.
    x_dot_x = float((x_ref * x_ref).sum())
    if abs(x_dot_x - 0.5) > 1e-6:
        log.warn(
            'solve_x1: stored amplitude has 2 X.X = %.6e (expected 1.0); '
            'check normalization of td.xy[%d].', 2.0 * x_dot_x, state)

    b = cp.asarray(b)
    if b.ndim == 2:                          # accept (nocc, nvir) for a single RHS
        b = b[None]
    assert b.shape[-2:] == (nocc, nvir), \
        f'solve_x1: b must have shape (..., nocc, nvir), got {b.shape}'
    npert = b.shape[0]

    # Build the explicit A matrix via td.get_ab(). This calls into the same
    # singlet-RHF kernel that the TDA eigensolver uses, so consistency with
    # td.e[state] is guaranteed.
    a_mat, _ = td.get_ab()
    A = cp.asarray(a_mat).reshape(nov, nov)

    x_flat = x_ref.ravel()
    P_X = cp.outer(x_flat, x_flat) / x_dot_x       # P_X = X X^T / <X|X>
    M_prime = A - omega * cp.eye(nov) + regularization * P_X

    # Diagnostic: smallest eigenvalue of M' should be > 0 (no kernel).
    if log.verbose >= logger.DEBUG:
        eigvals = cp.linalg.eigvalsh(0.5 * (M_prime + M_prime.T))
        log.debug('solve_x1: M\' eigvals (lo3) = %s', cp.asnumpy(eigvals[:3]))

    rhs = -b.reshape(npert, nov).T                  # (nov, npert)
    x1_flat = cp.linalg.solve(M_prime, rhs).T       # (npert, nov)
    x1 = x1_flat.reshape(npert, nocc, nvir)

    # Final projection: scrub residual X-component due to floating-point
    # noise in b and in the matrix solve. Correct in exact arithmetic; ~1e-12
    # in practice.
    proj = cp.einsum('rov,ov->r', x1, x_ref) / x_dot_x
    x1 = x1 - proj[:, None, None] * x_ref[None]

    log.timer('solve_x1 (perturbed TDA amplitude)', *t0)
    return x1


# -----------------------------------------------------------------------------
# Phase 2.0 primitives
# -----------------------------------------------------------------------------


def omega_grad(td, state, atmlst=None, with_solvent=False, singlet=True):
    '''Analytical gradient of the TDA excitation energy:

        omega^a = (TDA_gradient - GS_gradient)[a, x]

    Uses the existing closed-shell ``grad/tdrhf.py::grad_elec`` as the TDA
    energy gradient (which by Furche-Ahlrichs is omega^a + GS_grad^a),
    then subtracts the GS RHF electronic gradient. The result is the
    pure excitation-energy gradient, i.e. 2 X^T A^a X for closed-shell
    TDA on RHF reference.

    Parameters
    ----------
    td : converged TDA object
    state : 0-indexed root
    atmlst : list of atom indices or None
    with_solvent : forwarded to grad_elec
    singlet : forwarded to grad_elec

    Returns
    -------
    omega_grad : numpy.ndarray of shape (natm, 3)
        The excitation-energy gradient. (CPU array, matching the
        gradient module convention.)
    '''
    # Avoid a circular import: grad/tdrhf imports from tdscf which imports
    # from hessian via Hessian property registration in __init__.
    from gpu4pyscf.grad import tdrhf as tdrhf_grad
    from gpu4pyscf.grad import rhf as rhf_grad

    mf = td._scf
    mol = td.mol

    x_y = td.xy[state]
    if x_y[1] is not None and not (isinstance(x_y[1], (int, float)) and x_y[1] == 0):
        raise NotImplementedError(
            'omega_grad: Phase 2.0 supports TDA only (Y == 0).')

    td_grad_obj = tdrhf_grad.Gradients(td)
    de_tda_elec = td_grad_obj.grad_elec(
        x_y, singlet=singlet, atmlst=atmlst, with_solvent=with_solvent)

    # GS electronic gradient via the same SCF gradient class the TDA
    # gradient uses (so any QM/MM, solvent, ECP modifications track).
    mf_grad = mf.nuc_grad_method()
    de_gs_elec = mf_grad.grad_elec(atmlst=atmlst)

    # Both grad_elec returns are CPU numpy arrays (per gpu4pyscf convention).
    de_tda_elec = np.asarray(de_tda_elec)
    de_gs_elec = np.asarray(de_gs_elec)
    return de_tda_elec - de_gs_elec


def assemble_omega_cross_term(b_a, x_a):
    '''Assemble the ``4 X^T_b b^a`` cross term of the TDA Hessian addendum.

    Given:
      - ``b_a`` : (npert, nocc, nvir) — the perturbed RHS for each nuclear DOF
      - ``x_a`` : (npert, nocc, nvir) — the corresponding solve_x1(b_a) outputs

    returns the matrix ``H_cross`` of shape (npert, npert) where::

        H_cross[a, b] = 4 * sum_{i,c} X^a_{ic} b^b_{ic}.

    The full TDA-addendum Hessian is::

        omega^{ab} = 2 X^T A^{ab} X + H_cross[a, b]

    with the first term being the explicit 2nd-derivative-integral
    contribution (Phase 2.1).

    This factor of 4 is the closed-shell singlet RHF convention: factor 2
    from the 2X^T X = 1 normalization, factor 2 from a<->b symmetrization
    (which is automatic since x_a = solve_x1(b_a) and the cross sum is
    symmetric in (a, b) by the Hellmann-Feynman / Wigner 2n+1 rule).

    Note: in production this will be reshaped to (natm, 3, natm, 3) when
    added to the GS Hessian. The flat (npert, npert) form keeps this
    primitive index-agnostic; reshape at the call site.
    '''
    b_a = cp.asarray(b_a)
    x_a = cp.asarray(x_a)
    assert b_a.shape == x_a.shape, \
        f'assemble_omega_cross_term: shape mismatch b_a={b_a.shape} x_a={x_a.shape}'
    assert b_a.ndim == 3, \
        f'assemble_omega_cross_term: expected (npert, nocc, nvir), got {b_a.shape}'
    # contract over the (occ, vir) inner indices, broadcast over (a, b).
    return 4.0 * cp.einsum('aov,bov->ab', x_a, b_a)


def _vind_x_fd(mf, T_tr_AO, mo_coeff, mo_occ, delta=2.0e-3):
    '''Finite-difference perturbed-vind primitive for closed-shell singlet
    HF/RHF. Computes the AO-1st-derivative action of (J - 0.5 K) on a
    FIXED AO transition density T_tr_AO, transformed to MO (occ, vir):

        out[atm, x, o, v] = (orbo.T @ d/dR_x[J(T) - 0.5 K(T)] @ orbv)[o, v]

    where T = T_tr_AO is held fixed in AO basis (i.e., not re-projected
    at displaced geometries) -- this is the "Convention A explicit"
    derivative needed by ``compute_b_x``.

    Implementation: rebuild J and K at +-delta-displaced geometries
    using the SCF object's get_j/get_k (NOT calling .run() -- only the
    JK builder is needed). Central FD.

    Cost: 6 * natm JK builds (3 directions x 2 signs x natm). For a
    medium molecule ~1-2 orders of magnitude cheaper than full
    FD-on-gradient, since the SCF and TDA solves are skipped at every
    displacement. NOT fast enough for production -- the analytical
    version via 1st-derivative AO ERI kernels is Phase 2.2.

    Parameters
    ----------
    mf : converged closed-shell RHF object
    T_tr_AO : cupy.ndarray of shape (nao, nao)
        AO transition density (rank-1, asymmetric).
        T_tr[mu, nu] = sum_{a,i} c_{mu a} X_{ia} c_{nu i}
    mo_coeff, mo_occ : the MO coefficients and occupations at equilibrium
    delta : float
        FD step in Bohr.

    Returns
    -------
    cupy.ndarray of shape (natm, 3, nocc, nvir)
    '''
    from gpu4pyscf import scf as _gpu_scf
    mol = mf.mol
    natm = mol.natm
    coords0 = mol.atom_coords(unit='Bohr').copy()

    mo_coeff = cp.asarray(mo_coeff)
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]
    nocc = int(orbo.shape[1])
    nvir = int(orbv.shape[1])

    T_tr_AO = cp.asarray(T_tr_AO)

    out = cp.zeros((natm, 3, nocc, nvir))

    for ia in range(natm):
        for ix in range(3):
            v_pm = []
            for sign in (+1, -1):
                coords = coords0.copy()
                coords[ia, ix] += sign * delta
                molp = mol.copy()
                molp.set_geom_(coords, unit='Bohr')
                molp.build()
                # Build a fresh RHF object only for its JK builder; do NOT
                # run SCF -- we want vresp at the perturbed geometry on
                # the FIXED equilibrium T_tr_AO.
                mfp = _gpu_scf.RHF(molp)
                vj = mfp.get_j(molp, T_tr_AO, hermi=0)
                vk = mfp.get_k(molp, T_tr_AO, hermi=0)
                v_pm.append(cp.asarray(vj) - 0.5 * cp.asarray(vk))
            v_x_AO = (v_pm[0] - v_pm[1]) / (2.0 * delta)
            # MO transformation must match gen_tda_operation's convention:
            #   v_mo[o, v] = sum_{mu, nu} c_{mu, v} v_ao[mu, nu] c_{nu, o}
            # which differs from orbo.T @ v_ao @ orbv by a transpose on the
            # AO matrix (matters when v_ao is asymmetric, i.e., K acting on
            # an asymmetric transition density). Equivalent to
            # (orbv.T @ v_ao @ orbo).T:
            v_x_MO = (orbv.T @ v_x_AO @ orbo).T
            out[ia, ix] = v_x_MO

    return out


def _eps_x_diag_fd(mf, mo_coeff, mo_occ, delta=2.0e-3):
    '''Finite-difference perturbed orbital energies at fixed equilibrium MOs.

    Returns the full MO diagonal of the perturbed Fock matrix
    ``F^x_{pp}|_{C^eq}`` for all MOs p (occupied and virtual), of shape
    ``(natm, 3, nmo)``. This is the "term 1" piece of
    ``compute_b_x``: the orbital-energy gradient at fixed C, which
    contributes ``(eps^x_a - eps^x_i) X_{ai}`` to b^a.

    Implementation: rebuild ``h + (J - 0.5 K)[D^GS_eq]`` at +-delta-
    displaced geometries on the FIXED equilibrium GS density, AO-FD
    difference, transform diagonal to MO. Costs 6 * natm
    (h, J, K) builds. Production speedup is Phase 2.3 (analytical via
    AO 1st-derivative kernels).

    Parameters
    ----------
    mf : converged closed-shell RHF object
    mo_coeff, mo_occ : equilibrium MOs and occupations
    delta : float, FD step in Bohr.

    Returns
    -------
    cupy.ndarray of shape (natm, 3, nmo)
    '''
    from gpu4pyscf import scf as _gpu_scf
    mol = mf.mol
    natm = mol.natm
    nmo = mo_coeff.shape[1]
    coords0 = mol.atom_coords(unit='Bohr').copy()

    mo_coeff = cp.asarray(mo_coeff)
    mocc = mo_coeff[:, mo_occ > 0]
    D_GS = 2.0 * mocc @ mocc.T          # GS density at fixed C^eq, symmetric

    eps_x = cp.zeros((natm, 3, nmo))
    for ia in range(natm):
        for ix in range(3):
            F_pm = []
            for sign in (+1, -1):
                coords = coords0.copy()
                coords[ia, ix] += sign * delta
                molp = mol.copy()
                molp.set_geom_(coords, unit='Bohr')
                molp.build()
                mfp = _gpu_scf.RHF(molp)
                h = cp.asarray(mfp.get_hcore(molp))
                vj = cp.asarray(mfp.get_j(molp, D_GS, hermi=1))
                vk = cp.asarray(mfp.get_k(molp, D_GS, hermi=1))
                F_pm.append(h + vj - 0.5 * vk)
            F_x_AO = (F_pm[0] - F_pm[1]) / (2.0 * delta)
            # Diagonal in MO: eps^x_p = c_p^T @ F^x_AO @ c_p
            eps_x[ia, ix] = cp.einsum('mp,mn,np->p', mo_coeff, F_x_AO, mo_coeff)
    return eps_x


def compute_b_x(td, state, h1mo=None, mo1=None, mo_e1=None,
                fd_delta=2.0e-3):
    '''Build the perturbed RHS b^a = (A^a - omega^a I) X for each nuclear DOF.

    PHASE 2.1 STUB (still). The full closed-shell TDA-HF singlet b^a
    formula in the equilibrium MO basis ("Convention B" / explicit
    derivative at fixed C, which IS the physical b^a -- see derivation
    note below):

        b^a_{ai} = (eps^a_a - eps^a_i)|_{C^eq} X_{ai}            (term 1)
                 + V^a[T^tr]_{ai}|_{C^eq}                        (term 2)
                 - omega^a X_{ai}                                (term 3)

    where:
      - T^tr_{mu nu} = sum_{bj} c_{mu b} X_{bj} c_{nu j} is the AO
        transition density (rank-1, asymmetric).
      - eps^a_p|_{C^eq} = F^a_{pp} at fixed C^eq, in equilibrium MO basis;
        equals (h^a + (J - 0.5 K)^a[D^GS_eq])_{pp}, NO U^a contribution
        because we hold C fixed at equilibrium.
      - V^a[T^tr]|_{C^eq} = (2 J - K)^a [T^tr] at fixed C^eq, transformed
        to MO (occ, vir). For closed-shell singlet, the convention is
        V[T^tr] = vresp(2 T^tr) where vresp = J - 0.5 K (singlet hermi=0),
        which equals 2 J(T^tr) - K(T^tr).
      - omega^a = ``omega_grad(td, state)`` -- already implemented.

    Derivation note (why no U^a mixing): differentiating the eigenvalue
    equation in the equilibrium MO basis,

        A_{eq-basis}(R) X_{eq-basis}(R) = omega(R) X_{eq-basis}(R),

    where A_{eq-basis}(R)_{ai,bj} = <c_a^eq c_i^eq | H(R) | c_b^eq c_j^eq>
    has only AO-integral R-dependence (the eq MOs are R-independent).
    Differentiating gives (A_eq - omega I) X_eq^a = -b^a with b^a
    constructed only from explicit (fixed-C) derivatives. The X^a from
    solve_x1 is then in eq basis. The Hessian formula
    ``omega^{ab} = 2 X . A^{ab} . X + 4 X^b . b^a`` is invariant to
    convention so this is sufficient.

    Phase 2.2 status: ALL THREE TERMS implemented. Term 1 via
    ``_eps_x_diag_fd``, term 2 via ``_vind_x_fd``, term 3 via
    ``omega_grad``. Both FD primitives cost ~6*natm AO/JK builds and are
    correct but slow; analytical replacements are Phase 2.3.

    Parameters
    ----------
    td : converged TDA object
    state : 0-indexed root
    h1mo, mo1, mo_e1 : optional precomputed GS Hessian primitives.
        Currently unused (placeholder for the analytical Phase 2.3 path).
    fd_delta : float
        FD step (Bohr) for ``_eps_x_diag_fd`` and ``_vind_x_fd``.

    Returns
    -------
    b : cupy.ndarray of shape (natm*3, nocc, nvir)
    '''
    x_ref, omega = _get_zeroth_order(td, state)
    nocc, nvir = x_ref.shape
    mf = td._scf
    mol = td.mol
    natm = mol.natm

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    occidx = mo_occ > 0
    viridx = mo_occ == 0
    orbo = mo_coeff[:, occidx]
    orbv = mo_coeff[:, viridx]

    # AO transition density T^tr_{mu nu} = sum_{i,a} c_{mu a} X[i, a] c_{nu i}
    # = orbv @ X^T @ orbo^T   (X has shape (nocc, nvir))
    T_tr_AO = orbv @ x_ref.T @ orbo.T

    # Term 2: V^a[T^tr] in MO (occ, vir).
    # gen_tda_operation passes 2 T^tr to vresp = J - 0.5 K (singlet hermi=0)
    # so the singlet 2-electron contribution to (A . X) is 2 J(T^tr) - K(T^tr).
    # _vind_x_fd returns J^a(T^tr) - 0.5 K^a(T^tr) for the input density,
    # so factor of 2 here matches the closed-shell convention.
    v_x_mo = 2.0 * _vind_x_fd(mf, T_tr_AO, mo_coeff, mo_occ, delta=fd_delta)

    # Term 1: (eps^a_a - eps^a_i) X[i, a]
    eps_x = _eps_x_diag_fd(mf, mo_coeff, mo_occ, delta=fd_delta)  # (natm, 3, nmo)
    occidx_np = cp.asnumpy(occidx)
    viridx_np = cp.asnumpy(viridx)
    eps_x_occ = eps_x[..., occidx_np]   # (natm, 3, nocc)
    eps_x_vir = eps_x[..., viridx_np]   # (natm, 3, nvir)
    # Broadcast to (natm, 3, nocc, nvir):
    #   delta_eps[atm, x, i, a] = eps_x_vir[atm, x, a] - eps_x_occ[atm, x, i]
    eps_term = (eps_x_vir[..., None, :] - eps_x_occ[..., :, None]) * x_ref[None, None]

    # Term 3: -omega^a X
    omega_a = cp.asarray(omega_grad(td, state))   # (natm, 3) numpy -> cupy
    omega_term = -omega_a[..., None, None] * x_ref[None, None]

    b = (eps_term + v_x_mo + omega_term).reshape(natm * 3, nocc, nvir)
    return b


class Hessian(rhf_hess.HessianBase):
    '''Analytical Hessian of the total energy E_I = E_GS + omega_I for a
    closed-shell TDA root I built on an RHF reference.

    Phase 1 + Phase 2.0 STATUS: ``solve_x1`` (perturbed amplitude
    response), ``omega_grad`` (analytical excitation-energy gradient),
    and ``assemble_omega_cross_term`` (4 X^T_b b^a primitive) are
    implemented. ``compute_b_x`` is a Phase 2.1 stub and ``kernel``
    raises NotImplementedError until the b^a builder lands.
    '''

    cphf_max_cycle = 50
    cphf_conv_tol = 1e-8

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    _keys = {'cphf_max_cycle', 'cphf_conv_tol', 'mol', 'base', 'state',
             'atmlst', 'de'}

    def __init__(self, td):
        self.verbose = td.verbose
        self.stdout = td.stdout
        self.mol = td.mol
        self.base = td             # the TDA object
        self.max_memory = self.mol.max_memory
        self.state = 1             # 1-indexed by convention (state=0 -> GS)
        self.atmlst = None
        self.de = np.zeros((0, 0, 3, 3))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        log.info('cphf_conv_tol  = %g', self.cphf_conv_tol)
        log.info('cphf_max_cycle = %d', self.cphf_max_cycle)
        log.info('State          = %d', self.state)
        return self

    def solve_x1(self, b, state=None, regularization=1.0):
        '''Solve the perturbed TDA-amplitude equation. See module-level
        ``solve_x1`` for details.'''
        if state is None:
            state = self.state - 1   # convert 1-indexed self.state -> 0-indexed
        return solve_x1(self.base, state, b,
                        regularization=regularization,
                        verbose=self.verbose)

    def omega_grad(self, state=None, atmlst=None, with_solvent=False,
                   singlet=True):
        '''Analytical excitation-energy gradient. See module-level
        ``omega_grad`` for details.'''
        if state is None:
            state = self.state - 1
        return omega_grad(self.base, state, atmlst=atmlst,
                          with_solvent=with_solvent, singlet=singlet)

    def assemble_omega_cross_term(self, b_a, x_a):
        '''4 X^T_b b^a cross term. Stateless wrapper around module-level
        ``assemble_omega_cross_term``.'''
        return assemble_omega_cross_term(b_a, x_a)

    def compute_b_x(self, state=None, fd_delta=2.0e-3):
        '''Build b^a for the current state. See module-level ``compute_b_x``.'''
        if state is None:
            state = self.state - 1
        return compute_b_x(self.base, state, fd_delta=fd_delta)

    def kernel(self, *args, **kwargs):
        raise NotImplementedError(
            'Analytical excited-state Hessian assembly: Phase 2.0 ships '
            'the omega_grad primitive (analytical scalar gradient of the '
            'excitation energy) and the cross-term assembler '
            '(``assemble_omega_cross_term``). The full Hessian needs the '
            'b^a builder (``compute_b_x``) which is Phase 2.1 and depends '
            'on a perturbed-vind primitive (1st-derivative AO ERIs '
            'contracted with the transition density at the (occ, vir) MO '
            'level) that gpu4pyscf does not yet expose. Until then, use '
            'FD on the gradient -- see '
            '``~/research/templates/geom_excited_json.py``. See '
            'cpscf_init.md and the module docstring of '
            '``gpu4pyscf/hessian/tdrhf.py`` for the full Phase 2 plan.')

    hess = kernel
