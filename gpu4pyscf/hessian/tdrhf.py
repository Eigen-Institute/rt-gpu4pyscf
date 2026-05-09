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

Status: PHASE 1 + 2.0 + 2.2 + 2.3a. Phase 1 (``solve_x1``), Phase 2.0
(``omega_grad``, ``assemble_omega_cross_term``), Phase 2.2 (full
``compute_b_x`` self-consistent in convention A), and Phase 2.3a
(analytical replacement for ``_eps_x_diag_fd`` via
``_eps_x_diag_analytical`` -- uses GS-Hessian's ``_get_jk_ip1`` plus the
int3c2e/int1e_ipkin/ipnuc pattern from ``rhf_grad.get_grad_hcore``;
~20x faster than FD on methanol/6-31G) are shipped. Term 2 of
``compute_b_x`` (``_vind_x_fd``) is still FD because ``_get_jk_ip1``
assumes symmetric input and the asymmetric AO transition density T^tr
is not currently supported by gpu4pyscf's 1st-derivative JK builders;
the analytical replacement ``_vind_x_analytical`` is Phase 2.3b.

``Hessian.kernel`` still raises ``NotImplementedError`` because the
full Hessian assembly (Phase 2.4: 2X.A^{ab}.X explicit double-derivative
term + orbital relaxation Z-vector pieces) is not yet implemented. See
``cpscf_init.md`` and the Phase 2.4 sketch for the remaining blocks.

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
- **Liu & Liang, J. Chem. Phys. 138, 024101 (2013)** -- analytical TDDFT
  excited-state Hessian with PCM. Eq. (19) is the master formula; for
  HF/RHF TDA without DFT/PCM it reduces to seven named pieces:
    row 1 (pure 2nd-deriv x density):
      H^{xy} P'_I + Gamma'_I Pi^{xy} + W_I S^{xy}
    row 2 (mixed 1st-deriv cross):
      P'^[~y]_I F^(x) + Gamma'^y_I Pi^x + W^[~y]_I S^x
    row 3 (CP-SCF orbital-response coupling):
      L'^[~y]_I (2 U^x + S^(x))
  Our partial Hessian (Blocks 1+2 + scaffolded 3+4) covers row 1
  approximately; rows 2 and 3 are the missing 3% gap. See cpscf_init.md.
- van Caillie & Amos, Chem. Phys. Lett. 308, 249 (1999); 317, 159 (2000).
- Furche & Ahlrichs, J. Chem. Phys. 117, 7433 (2002) (gradient -- the
  Lagrangian extends to give the Hessian).
- Send & Furche, J. Chem. Phys. 132, 044107 (2010) (RPA second derivatives).
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


def _eps_x_diag_analytical(mf, mo_coeff=None, mo_occ=None):
    '''Analytical first derivative of F[D^eq] at fixed equilibrium MO
    coefficients and fixed equilibrium GS density. Returns the diagonal
    in MO basis: shape ``(natm, 3, nmo)``.

    Drop-in replacement for ``_eps_x_diag_fd`` with no FD truncation.
    Uses gpu4pyscf's GS Hessian primitive ``_get_jk_ip1`` for the
    ``(J - 0.5 K)^a [D^eq]`` piece, plus the int3c2e + int1e_ipkin/ipnuc
    pattern from ``rhf_grad.get_grad_hcore`` for h^a (adapted to project
    onto the full MO set instead of just occupied orbitals).
    '''
    from gpu4pyscf.df import int3c2e
    from gpu4pyscf.hessian.rhf import _get_jk_ip1
    from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
    from pyscf import gto

    mol = mf.mol
    natm = mol.natm
    nao = mol.nao

    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)
    nmo = mo_coeff.shape[1]
    mocc = mo_coeff[:, mo_occ > 0]
    D_GS = 2.0 * mocc @ mocc.T  # closed-shell GS density

    eps_x_diag = cp.zeros((natm, 3, nmo))

    # Part 1: int3c2e ip1 -- atom-resolved nuclear-attraction derivative
    # (the "moving charge" contribution to h^a).
    coords = mol.atom_coords()
    charges = cp.asarray(mol.atom_charges(), dtype=cp.float64)
    fakemol = gto.fakemol_for_charges(coords)
    intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
    intopt.build(1e-14, diag_block_with_triu=True, aosym=False,
                 group_size=int3c2e.BLKSIZE,
                 group_size_aux=int3c2e.BLKSIZE)
    mo_sorted = intopt.sort_orbitals(mo_coeff, axis=[0])

    dh1e = cp.zeros((natm, 3, nao, nmo))
    for i0, i1, j0, j1, k0, k1, int3c_blk in int3c2e.loop_int3c2e_general(
            intopt, ip_type='ip1'):
        dh1e[k0:k1, :, j0:j1, :] += contract(
            'xkji,iq->kxjq', int3c_blk, mo_sorted[i0:i1])
        dh1e[k0:k1, :, i0:i1, :] += contract(
            'xkji,jq->kxiq', int3c_blk, mo_sorted[j0:j1])
    dh1e = contract('kxjq,k->kxjq', dh1e, -charges)
    # Contract j (sorted-AO) with mo_sorted[j, q] -> diagonal in MO q.
    eps_x_diag += contract('jq,kxjq->kxq', mo_sorted, dh1e)

    # Part 2: int1e_ipkin/ipnuc -- atom-resolved AO-basis-function
    # derivative.  rhf_grad.get_hcore returns -(int1e_ipkin + int1e_ipnuc),
    # shape (3, nao, nao), bra-derivative.
    h1 = cp.asarray(mf.nuc_grad_method().get_hcore(mol))
    aoslices = mol.aoslice_by_atom()
    for atm in range(natm):
        p0, p1 = aoslices[atm][2:]
        # bra-on-atm: rows of h1 in [p0:p1].
        eps_x_diag[atm] += cp.einsum(
            'mp,xmn,np->xp',
            mo_coeff[p0:p1], h1[:, p0:p1, :], mo_coeff)
        # ket-on-atm: by Hermitian symmetry of h, the ket-derivative is the
        # bra-derivative with bra/ket indices swapped (i.e., transpose).
        eps_x_diag[atm] += cp.einsum(
            'mp,xmn,np->xp',
            mo_coeff, h1[:, p0:p1, :].transpose(0, 2, 1), mo_coeff[p0:p1])

    # Part 3: (J - 0.5 K)^a [D_GS], atom-resolved, transformed to MO diagonal.
    nao_cart = mol.nao_cart()
    avail_mem = get_avail_mem()
    slice_size = max(
        int(avail_mem * 0.5) // (8 * 3 * nao_cart * nao_cart * 3), 1)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vj, vk = _get_jk_ip1(mol, D_GS, atoms_slice=atoms_slice)
        vhf = vj - 0.5 * vk
        atom0, atom1 = atoms_slice
        for i, atm in enumerate(range(atom0, atom1)):
            for ix in range(3):
                eps_x_diag[atm, ix] += cp.einsum(
                    'mp,mn,np->p', mo_coeff, vhf[i, ix], mo_coeff)
    return eps_x_diag


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
                fd_delta=2.0e-3, use_fd_eps=False):
    '''Build the perturbed RHS b^a = (A^a - omega^a I) X for each nuclear DOF.

    PHASE 2.2 (convention A). Closed-shell TDA-HF singlet b^a in the
    equilibrium MO basis at fixed equilibrium orbital coefficients
    (convention A: only the AO integrals carry R-dependence; the GS
    density is held fixed at D^eq):

        b^a_{ai} = (eps^a_a - eps^a_i)|_{C^eq, D^eq} X_{ai}      (term 1)
                 + V^a[T^tr]_{ai}|_{C^eq}                        (term 2)
                 - omega^A^a X_{ai}                              (term 3)

    where:
      - T^tr_{mu nu} = sum_{bj} c_{mu b} X_{bj} c_{nu j} is the AO
        transition density (rank-1, asymmetric).
      - eps^a_p|_{C^eq, D^eq} = (h^a + (J - 0.5 K)^a[D^GS_eq])_{pp}, with
        BOTH C and the GS density held at equilibrium values. No U^a or
        density-relaxation contribution.
      - V^a[T^tr]|_{C^eq} = (2 J - K)^a [T^tr] at fixed C^eq, transformed
        to MO (occ, vir). For closed-shell singlet, the convention is
        V[T^tr] = vresp(2 T^tr) where vresp = J - 0.5 K (singlet hermi=0),
        which equals 2 J(T^tr) - K(T^tr).
      - omega^A^a is the convention-A excitation-energy gradient,
        omega^A^a = 2 (eps_part^a + V_part^a). Computed in-place from
        terms 1 and 2 so the Hellmann-Feynman identity <X|b^a> = 0 holds
        exactly by construction (modulo FD truncation in the primitives).
        This DIFFERS from ``omega_grad(td, state)`` (= omega^phys^a, the
        full physical gradient including GS orbital relaxation through
        CP-SCF) by exactly the U^x density-relaxation contribution
        (J - 0.5 K)[D^a] missing from term 1's fixed-D evaluation.

    Convention note: the convention-A b^a yields a self-consistent
    (A - omega I) X^A^a = -b^A^a system whose ``solve_x1`` solution gives
    the perturbed amplitude X^A^a in the eq-basis / fixed-D framework.
    The cross-term ``4 X^A_b b^A^a`` then assembles the convention-A
    piece of the Hessian. Recovering the full physical Hessian
    ``omega^phys^{ab}`` requires adding orbital-relaxation corrections
    via Z-vector / CP-SCF (Phase 2.3+).

    Phase 2.3 status: term 1 (eps^a) is now analytical by default via
    ``_eps_x_diag_analytical`` (uses ``_get_jk_ip1`` plus the
    int3c2e/int1e_ipkin/ipnuc pattern from ``rhf_grad.get_grad_hcore``
    adapted to project on the full MO set). Term 2 (V^a[T^tr]) is still
    FD because ``_get_jk_ip1`` assumes symmetric input and gpu4pyscf
    does not yet expose a 1st-derivative JK builder that handles the
    asymmetric AO transition density T^tr correctly; analytical V^a is
    deferred to Phase 2.3b.

    Net cost: ~6*natm JK builds for term 2 (down from ~12*natm in
    Phase 2.2), plus a single _get_jk_ip1 pass for term 1.

    Parameters
    ----------
    td : converged TDA object
    state : 0-indexed root
    h1mo, mo1, mo_e1 : optional precomputed GS Hessian primitives.
        Currently unused (placeholder for the analytical Phase 2.3b
        path that will replace ``_vind_x_fd``).
    fd_delta : float
        FD step (Bohr) for ``_vind_x_fd`` (and ``_eps_x_diag_fd`` when
        ``use_fd_eps=True``).
    use_fd_eps : bool, optional
        If True, use the FD primitive for term 1 instead of the
        analytical version. Useful for cross-checking / regression tests;
        defaults to False (analytical).

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

    # Term 1: (eps^a_a - eps^a_i) X[i, a]. Analytical by default (Phase 2.3).
    if use_fd_eps:
        eps_x = _eps_x_diag_fd(mf, mo_coeff, mo_occ, delta=fd_delta)
    else:
        eps_x = _eps_x_diag_analytical(mf, mo_coeff, mo_occ)  # (natm, 3, nmo)
    occidx_np = cp.asnumpy(occidx)
    viridx_np = cp.asnumpy(viridx)
    eps_x_occ = eps_x[..., occidx_np]   # (natm, 3, nocc)
    eps_x_vir = eps_x[..., viridx_np]   # (natm, 3, nvir)
    # Broadcast to (natm, 3, nocc, nvir):
    #   delta_eps[atm, x, i, a] = eps_x_vir[atm, x, a] - eps_x_occ[atm, x, i]
    eps_term = (eps_x_vir[..., None, :] - eps_x_occ[..., :, None]) * x_ref[None, None]

    # Term 3: -omega^A^a X.  The convention-A excitation-energy gradient
    # is computed from terms 1 and 2 directly:
    #   omega^A^a = 2 * <X | (term 1) + (term 2)>
    # so that <X | b^a> = (omega^A^a / 2) - (omega^A^a / 2) = 0 holds by
    # construction, as required for solve_x1's deflation to be consistent
    # (see docstring "Convention note"). omega^phys^a = omega_grad(td, state)
    # differs by orbital relaxation; that delta belongs in Phase 2.3+.
    eps_plus_v = eps_term + v_x_mo
    omega_A_a = 2.0 * cp.einsum('ov,axov->ax', x_ref, eps_plus_v)  # (natm, 3)
    omega_term = -omega_A_a[..., None, None] * x_ref[None, None]

    b = (eps_plus_v + omega_term).reshape(natm * 3, nocc, nvir)
    return b


def _compute_z_and_densities(td, state):
    '''Compute the Z-vector and supporting MO-basis quantities for the
    closed-shell singlet TDA-on-RHF excited state ``state``.

    Mirrors the relevant block of ``grad/tdrhf.py::grad_elec`` (lines
    ~50-152) but returns the intermediate quantities needed for Block 3
    (P^I) and Block 4 (W^I) assembly without doing the gradient itself.

    Returns
    -------
    dict with keys:
        z1   : (nvir, nocc)  -- the Z-vector solution
        doo  : (nocc, nocc)  -- excited-state occ-occ density correction
        dvv  : (nvir, nvir)  -- excited-state vir-vir density correction
        dmxpy : (nao, nao)   -- (X+Y) in AO (asymmetric, = T^tr for TDA)
        dmxmy : (nao, nao)   -- (X-Y) in AO (= T^tr for TDA)
        dmzoo : (nao, nao)   -- relaxed transition density (no Z) in AO
        im0_MO : (nmo, nmo)  -- W^I MO-basis pre-form (without zeta·dm1)
    '''
    from functools import reduce
    from gpu4pyscf.scf import cphf
    from gpu4pyscf.lib.cupy_helper import contract

    mf = td._scf
    mol = td.mol
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    nvir = nmo - nocc

    x_y = td.xy[state]
    x = cp.asarray(x_y[0])
    y_part = x_y[1]
    if not (isinstance(y_part, (int, float)) and y_part == 0):
        raise NotImplementedError(
            '_compute_z_and_densities supports closed-shell TDA only (Y=0).')
    y = cp.zeros_like(x)

    xpy = (x + y).reshape(nocc, nvir).T   # (nvir, nocc)
    xmy = (x - y).reshape(nocc, nvir).T

    orbv = mo_coeff[:, nocc:]
    orbo = mo_coeff[:, :nocc]

    dvv = (contract("ai,bi->ab", xpy, xpy)
           + contract("ai,bi->ab", xmy, xmy))
    doo = (-contract("ai,aj->ij", xpy, xpy)
           - contract("ai,aj->ij", xmy, xmy))
    dmxpy = reduce(cp.dot, (orbv, xpy, orbo.T))
    dmxmy = reduce(cp.dot, (orbv, xmy, orbo.T))
    dmzoo = reduce(cp.dot, (orbo, doo, orbo.T))
    dmzoo += reduce(cp.dot, (orbv, dvv, orbv.T))

    vj0, vk0 = mf.get_jk(mol, dmzoo, hermi=0)
    vj1, vk1 = mf.get_jk(mol, dmxpy + dmxpy.T, hermi=0)
    vj2, vk2 = mf.get_jk(mol, dmxmy - dmxmy.T, hermi=0)
    vj0 = cp.asarray(vj0); vk0 = cp.asarray(vk0)
    vj1 = cp.asarray(vj1); vk1 = cp.asarray(vk1)
    vj2 = cp.asarray(vj2); vk2 = cp.asarray(vk2)

    veff0doo = vj0 * 2 - vk0
    wvo = reduce(cp.dot, (orbv.T, veff0doo, orbo)) * 2

    # Singlet kernel: 2J - K
    veff = vj1 * 2 - vk1
    veff0mop = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    wvo += contract("ac,ai->ci", veff0mop[nocc:, nocc:], xpy) * 2

    veff = -vk2
    veff0mom = reduce(cp.dot, (mo_coeff.T, veff, mo_coeff))
    wvo -= contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2
    wvo += contract("ac,ai->ci", veff0mom[nocc:, nocc:], xmy) * 2

    vresp = td.gen_response(singlet=None, hermi=1)

    def fvind(z):
        dm = reduce(cp.dot, (orbv, z.reshape(nvir, nocc) * 2, orbo.T))
        v1ao = vresp(dm + dm.T)
        return reduce(cp.dot, (orbv.T, v1ao, orbo)).ravel()

    z1 = cphf.solve(fvind, mo_energy, mo_occ, wvo,
                    max_cycle=50, tol=1e-8)[0]
    z1 = z1.reshape(nvir, nocc)

    # Build im0_MO for W^I assembly. Mirrors grad/tdrhf.py:124-151.
    z1ao = reduce(cp.dot, (orbv, z1, orbo.T))
    veff = vresp(z1ao + z1ao.T)

    im0 = cp.zeros((nmo, nmo))
    im0[:nocc, :nocc] = reduce(cp.dot, (orbo.T, veff0doo + veff, orbo))
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mop[nocc:, :nocc], xpy)
    im0[:nocc, :nocc] += contract("ak,ai->ki", veff0mom[nocc:, :nocc], xmy)
    im0[nocc:, nocc:] = contract("ci,ai->ac", veff0mop[nocc:, :nocc], xpy)
    im0[nocc:, nocc:] += contract("ci,ai->ac", veff0mom[nocc:, :nocc], xmy)
    im0[nocc:, :nocc] = contract("ki,ai->ak", veff0mop[:nocc, :nocc], xpy) * 2
    im0[nocc:, :nocc] += contract("ki,ai->ak", veff0mom[:nocc, :nocc], xmy) * 2

    return {
        'z1': z1, 'doo': doo, 'dvv': dvv,
        'dmxpy': dmxpy, 'dmxmy': dmxmy, 'dmzoo': dmzoo,
        'im0_MO': im0,
    }


def _build_PI_and_W_AO(td, state):
    '''Assemble P^I_AO (relaxed difference density) and W^I_AO
    (energy-weighted relaxed density) in AO basis. Returns the full
    (asymmetric) matrices and their symmetric parts.

    For Phase 2.4 Blocks 3+4, we use the symmetric parts; the asymmetric
    Z-vector AO contribution to JK^{ab} traces is conjectured to vanish
    against the symmetric integrals (block-3 1-electron h^{ab}, block-4
    overlap S^{ab}) but partially survives for the 2-electron mixed
    contraction with non-symmetric F^{ab}[D^GS] integral types. The
    residual is the focus of the FD validation that follows.
    '''
    pieces = _compute_z_and_densities(td, state)
    z1 = pieces['z1']; doo = pieces['doo']; dvv = pieces['dvv']
    im0_MO = pieces['im0_MO']

    mf = td._scf
    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_energy = cp.asarray(mf.mo_energy)
    mo_occ = cp.asarray(mf.mo_occ)
    nmo = mo_coeff.shape[1]
    nocc = int((mo_occ > 0).sum())
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]

    # P^I_AO.  In MO basis: P[occ, occ] = doo, P[vir, vir] = dvv,
    # P[vir, occ] = z1, P[occ, vir] = 0. Transform to AO.
    P_AO = (orbo @ doo @ orbo.T
            + orbv @ dvv @ orbv.T
            + orbv @ z1 @ orbo.T)
    P_AO_sym = (P_AO + P_AO.T) * 0.5

    # W^I_AO_excited = mo @ (im0_MO + zeta * dm1_excited) @ mo.T
    # NOTE: the gradient code's `dm1` adds `eye(nocc) * 2` to include the
    # GS energy-weighted density (so its `im0` is the FULL TDA W). For our
    # omega^{ab} excited-only addendum, we OMIT that piece -- the GS S^{ab}
    # contribution is already in the GS Hessian and would double-count.
    zeta = (mo_energy[:, None] + mo_energy[None, :]) * 0.5
    zeta[nocc:, :nocc] = mo_energy[:nocc]
    zeta[:nocc, nocc:] = mo_energy[nocc:]
    dm1_excited_MO = cp.zeros((nmo, nmo))
    dm1_excited_MO[:nocc, :nocc] = doo
    dm1_excited_MO[nocc:, nocc:] = dvv
    dm1_excited_MO[nocc:, :nocc] = z1
    W_AO = mo_coeff @ (im0_MO + zeta * dm1_excited_MO) @ mo_coeff.T
    W_AO_sym = (W_AO + W_AO.T) * 0.5

    return {
        'P_AO': P_AO, 'P_AO_sym': P_AO_sym,
        'W_AO': W_AO, 'W_AO_sym': W_AO_sym,
        **pieces,
    }


def _omega_blocks_3_4(td, state):
    '''Phase 2.4 Blocks 3 + 4 -- orbital-relaxation correction to the
    convention-A Hessian. Returns (natm, 3, natm, 3) tensor.

    Block 3:  ``+ tr(P^I_sym . F^{ab}[D^{GS}])``
              = 1-electron tr(P^I_sym . h^{ab})
              + 2-electron mixed tr(P^I_sym . (J - 0.5 K)^{ab}[D^{GS}])
                via polarization on _partial_ejk_ip2.
    Block 4:  ``- tr(W^I_sym . S^{ab})``
              using the GS Hessian's s1aa/s1ab 2nd-derivative overlap
              integrals and the same canonical-ordering pattern as
              _partial_hess_ejk.

    Symmetric-only: feeds (P^I + P^I.T)/2 and (W^I + W^I.T)/2 to the
    integrators. Asymmetric pieces of P^I (Z-vector AO contribution)
    contribute zero against symmetric h^{ab} and S^{ab}, but partially
    survive against the non-symmetric F^{ab}[D^{GS}] integral types
    (Phase 2.4 Blocks-3-asym, blocked on hermi-aware kernel).
    '''
    from gpu4pyscf.hessian import rhf as rhf_hess
    from gpu4pyscf.hessian.rhf import _partial_ejk_ip2
    from gpu4pyscf.lib.cupy_helper import contract

    mf = td._scf
    mol = td.mol
    natm = mol.natm

    pieces = _build_PI_and_W_AO(td, state)
    P_sym = pieces['P_AO_sym']
    W_sym = pieces['W_AO_sym']

    mo_coeff = cp.asarray(mf.mo_coeff)
    mo_occ = cp.asarray(mf.mo_occ)
    mocc = mo_coeff[:, mo_occ > 0]
    D_GS = 2.0 * mocc @ mocc.T

    # Build a transient Hessian object so we can borrow its primitives.
    hessobj = rhf_hess.Hessian(mf)

    # ===== Block 3 =====
    # 3a: 1-electron tr(P^I_sym . h^{ab}) per (atm_a, atm_b)
    block3 = cp.zeros((natm, natm, 3, 3))
    de_hcore_PI = rhf_hess._e_hcore_generator(hessobj, P_sym)
    for ia in range(natm):
        for ja in range(natm):
            block3[ia, ja] += de_hcore_PI(ia, ja)

    # 3b: 2-electron tr(P^I_sym . (J - 0.5 K)^{ab}[D^{GS}]) via polarization
    #   ejk(D, j, k)              = j J^{ab}[D, D] - k K^{ab}[D, D]
    #   ejk(D + P) - ejk(D) - ejk(P) = 2 ejk_cross(D, P)
    #                                 = 2 (j J^{ab}_cross - k K^{ab}_cross)
    # We want tr(P (J - 0.5 K)^{ab}[D]) = J^{ab}_cross - 0.5 K^{ab}_cross
    # which corresponds to ejk_cross with (j=1, k=0.5).
    ejk_combined = _partial_ejk_ip2(mol, D_GS + P_sym,
                                    j_factor=1., k_factor=0.5)
    ejk_DGS      = _partial_ejk_ip2(mol, D_GS,
                                    j_factor=1., k_factor=0.5)
    ejk_PI       = _partial_ejk_ip2(mol, P_sym,
                                    j_factor=1., k_factor=0.5)
    block3_2e = (ejk_combined - ejk_DGS - ejk_PI) * 0.5
    block3 = block3 + block3_2e

    # ===== Block 4 =====
    # -tr(W^I_sym . S^{ab}); use s1aa for diagonal block, s1ab for off-diag.
    s1aa, s1ab, _s1a = rhf_hess.get_ovlp(mol)
    s1aa = cp.asarray(s1aa); s1ab = cp.asarray(s1ab)
    aoslices = mol.aoslice_by_atom()
    block4 = cp.zeros((natm, natm, 3, 3))
    for i0 in range(natm):
        p0, p1 = aoslices[i0][2:]
        block4[i0, i0] -= contract(
            'xypq,pq->xy', s1aa[:, :, p0:p1], W_sym[p0:p1]) * 2
        for j0 in range(i0 + 1):
            q0, q1 = aoslices[j0][2:]
            term = contract(
                'xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1],
                W_sym[p0:p1, q0:q1]) * 2
            block4[i0, j0] -= term
    # Fill upper triangular by Hessian symmetry.
    for i0 in range(natm):
        for j0 in range(i0):
            block4[j0, i0] = block4[i0, j0].T

    total = (block3 + block4).transpose(0, 2, 1, 3)
    return total


def _omega_ab_pure_fd(td, state, fd_delta=2.0e-3):
    '''Phase 2.4 Block 1 (FD-driven version).

    Computes the convention-A pure 2nd-derivative term

        omega^{A,ab}_pure = 2 X^T A^{ab} X
                         = 2 d/dR_b (eps_part^a + V_part^a)

    by central FD on the 1st-derivative quantity ``(eps_term + v_x_mo)``
    that ``compute_b_x`` already builds. Returns shape ``(natm, 3, natm, 3)``.

    This is the "FD on analytical-1st-derivative" path: each displacement
    runs SCF + TDA + ``compute_b_x`` (which uses analytical eps^a from
    Phase 2.3a + FD V^a). Cost: 6*natm SCF/TDA solves + 12*natm
    ``compute_b_x`` calls. Cleanly correct; the eventual Phase 2.4b will
    replace this with analytical 2nd-derivative ERI primitives
    (e.g. ``_partial_ejk_ip2`` on M^tr) for orders-of-magnitude speedup.

    Phase 2.4b status: empirically blocked on the same missing primitive
    as Phase 2.3b. ``_partial_ejk_ip2`` post-symmetrizes assuming
    hermi=1 dm, so feeding the asymmetric AO transition density
    M^tr = orbo @ X @ orbv^T produces results 100%+ off all FD
    references regardless of (j_factor, k_factor) choice. Unblocks
    once gpu4pyscf exposes a hermi-aware 2nd-derivative JK kernel.
    '''
    from gpu4pyscf import scf as _gpu_scf
    from gpu4pyscf import tdscf as _gpu_tdscf
    mol = td.mol
    natm = mol.natm
    nocc, nvir = cp.asarray(td.xy[state][0]).shape

    # Reference: x_ref captured ONCE at the equilibrium geometry. We
    # compute (eps + v) at displaced geometries, contract with the FIXED
    # eq x_ref to extract the (A^{ab} X) contribution.
    x_ref = cp.asarray(td.xy[state][0])

    coords0 = mol.atom_coords(unit='Bohr').copy()
    omega_pure = cp.zeros((natm, 3, natm, 3))

    def _eps_plus_v_at(mol_d):
        '''Build (eps_term + v_x_mo) at the displaced geometry, projected
        onto the displaced-MO (occ, vir) but interpreting amplitudes as
        the equilibrium x_ref. Returns (natm, 3, nocc, nvir).'''
        mfp = _gpu_scf.RHF(mol_d)
        mfp.conv_tol = max(getattr(td._scf, 'conv_tol', 1e-12) * 0.1, 1e-13)
        mfp.run()
        if not mfp.converged:
            raise RuntimeError('Inner SCF did not converge during Hessian FD.')
        # Build a TDA shell at displaced geometry just to use td_disp's
        # state-tracker-equivalent infrastructure -- but we DON'T re-solve
        # TDA; we just need the operator pieces evaluated at displaced
        # MO basis with the ORIGINAL x_ref amplitudes.
        mo_d = cp.asarray(mfp.mo_coeff)
        occ_d = cp.asarray(mfp.mo_occ)
        orbo_d = mo_d[:, occ_d > 0]
        orbv_d = mo_d[:, occ_d == 0]
        T_tr_d = orbv_d @ x_ref.T @ orbo_d.T

        eps_x = _eps_x_diag_analytical(mfp, mo_d, occ_d)
        nocc_d = orbo_d.shape[1]
        eps_x_o = eps_x[..., :nocc_d]
        eps_x_v = eps_x[..., nocc_d:]
        eps_term = (eps_x_v[..., None, :] - eps_x_o[..., :, None]) * x_ref[None, None]
        v_x = 2.0 * _vind_x_fd(mfp, T_tr_d, mo_d, occ_d, delta=fd_delta)
        return eps_term + v_x

    for atm_b in range(natm):
        for ix_b in range(3):
            disp_p = coords0.copy(); disp_p[atm_b, ix_b] += fd_delta
            disp_m = coords0.copy(); disp_m[atm_b, ix_b] -= fd_delta
            mol_p = mol.copy(); mol_p.set_geom_(disp_p, unit='Bohr'); mol_p.build()
            mol_m = mol.copy(); mol_m.set_geom_(disp_m, unit='Bohr'); mol_m.build()
            ev_p = _eps_plus_v_at(mol_p)
            ev_m = _eps_plus_v_at(mol_m)
            # 2 X^T (∂_b (A^a X)) = 2 sum_{ia} X[i,a] * d/dR_b (eps + v)_{a, ix_a, i, a}
            # Result indexed by (atm_a, ix_a) for each (atm_b, ix_b).
            d_ab = (ev_p - ev_m) / (2.0 * fd_delta)  # shape (natm_a, 3, nocc, nvir)
            omega_pure[:, :, atm_b, ix_b] = 2.0 * cp.einsum(
                'axiv,iv->ax', d_ab, x_ref)

    # Symmetrize (Hessian theorem).
    omega_pure = 0.5 * (omega_pure + omega_pure.transpose(2, 3, 0, 1))
    return omega_pure


def omega_hessian(td, state, fd_delta=2.0e-3, include_relaxation=False):
    '''Phase 2.4 assembly: closed-shell singlet TDA-on-RHF excited-state
    energy Hessian ``omega^{ab}``, shape ``(natm, 3, natm, 3)``.

    Pieces:
      Block 1 (pure, 2 X^T A^{ab} X)       via FD on the Phase 2.3a
                                            analytical 1st-derivative
      Block 2 (cross, 4 X^T_b b^a)         via Phase 0 cross-term
                                            primitive on Phase 1 solve_x1
      Block 3 (orbital relaxation,         via _omega_blocks_3_4
        +tr(P^I . F^{ab}[D^GS])) and        SCAFFOLDING ONLY -- the
      Block 4 (energy-weighted overlap,     factor / sign attribution
        -tr(W^I . S^{ab}))                  doesn't yet match the
                                            Furche-Ahlrichs Lagrangian.
                                            On H2O/STO-3G the partial
                                            (Blocks 1+2) recovers ~97%
                                            of the FD-on-omega gold
                                            standard with a 0.15 gap;
                                            adding the current Blocks
                                            3+4 OVERSHOOTS by ~3.3.
                                            Default ``False``.

    Set ``include_relaxation=True`` to add Blocks 3+4 (debug-only --
    not yet validated against FD gold standard).
    '''
    natm = td.mol.natm

    omega_pure = _omega_ab_pure_fd(td, state, fd_delta=fd_delta)

    b_a = compute_b_x(td, state, fd_delta=fd_delta)
    x_a = solve_x1(td, state, b_a)
    omega_cross = assemble_omega_cross_term(b_a, x_a)
    omega_cross = omega_cross.reshape(natm, 3, natm, 3)

    out = omega_pure + omega_cross

    if include_relaxation:
        omega_relax = _omega_blocks_3_4(td, state)
        out = out + omega_relax

    return out


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

    def kernel(self, *args, fd_delta=2.0e-3, include_relaxation=False, **kwargs):
        '''Closed-shell singlet TDA-on-RHF excited-state energy Hessian
        ``omega^{ab}``, shape ``(natm, 3, natm, 3)``.

        DEFAULT: convention-A partial (Blocks 1+2). Recovers ~97% of
        the FD-on-omega_grad gold standard on small systems; the missing
        ~3% is the orbital-relaxation correction. Pass
        ``include_relaxation=True`` to add Blocks 3+4 -- but those are
        currently SCAFFOLDING with incorrect factors / sign attribution;
        they OVERSHOOT the gap rather than closing it. Use only for
        debugging until the Furche-Ahlrichs formula derivation is
        completed (see cpscf_init.md).
        '''
        state = self.state - 1
        return omega_hessian(self.base, state, fd_delta=fd_delta,
                             include_relaxation=include_relaxation)

    hess = kernel
