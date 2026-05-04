"""Transition-density-based excited-state tracking.

Given a converged TDDFT calculation at a reference geometry, identify which
root in a calculation at a different geometry corresponds to a chosen
reference root, by computing the wavefunction overlap in the
single-excitation manifold.

Math (RKS / TDA):

  |Psi^J> = sum_{jb} X^J_{jb} a^dag_b a_j |Phi_0>

For reference state J at R_0 and displaced state K at R_0 + dR,

  S_KJ ~= sum_{ia} X^K_{ia}(disp) * [ U_o @ X^J_ref @ U_v.T ]_{ia}

with orbital-correspondence matrices

  U_o = (C_disp_occ).T @ S_mix @ C_ref_occ      # <phi_i^disp | phi_j^ref>
  U_v = (C_disp_vir).T @ S_mix @ C_ref_vir
  S_mix = <chi_mu^disp | chi_nu^ref>            # gto.intor_cross

Each transition tensor is normalized to unit Frobenius norm before the inner
product, so the matching score is cosine similarity in [-1, 1]. Sign is
preserved on the result so callers can do amplitude phase correction.

For full TDDFT (X, Y), the matching tensor is X+Y. For UKS, the metric sums
over spin.

Two entry points:
  * ``__init__`` / ``assign`` -- reference and displaced come from gpu4pyscf
    TDDFT objects (.xy, .e, ._scf).
  * ``from_amplitudes`` / ``assign_amplitudes`` -- raw arrays for callers that
    diagonalize TDDFT manually (e.g. nac/finite_diff.py) or otherwise lack a
    standard td object.

Out of scope: spin-flip TDDFT (raises), norm-preserving Slater-determinant
overlap (linear approximation only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False


def _to_numpy(x):
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


@dataclass
class AssignmentResult:
    """Outcome of TransitionDensityTracker.assign().

    Attributes
    ----------
    root : int
        0-indexed root in the displaced TDDFT solution that best matches the
        reference target.
    overlap : float
        |S_K*,Jref| -- absolute cosine similarity of normalized transition
        tensors, in [0, 1].
    signed_overlap : float
        S_K*,Jref with sign preserved. ``sign`` returns +/-1 based on this
        and is what callers use to phase-correct amplitudes from the
        displaced eigenvector (which has arbitrary sign).
    runner_up : (int, float)
        0-indexed second-best root and its |overlap|.
    energies : np.ndarray
        Copy of td_disp.e (excitation energies, Hartree); NaN-filled if
        unavailable.
    de_target : float
        |E_disp[K*] - E_ref|; NaN if e_ref or e_disp was not provided.
    flags : list[str]
        Diagnostics: 'low_overlap', 'near_degenerate', 'energy_jump',
        'index_changed'.
    """
    root: int
    overlap: float
    signed_overlap: float
    runner_up: Tuple[int, float]
    energies: np.ndarray
    de_target: float
    flags: List[str] = field(default_factory=list)

    def state_1indexed(self) -> int:
        return self.root + 1

    @property
    def sign(self) -> int:
        return 1 if self.signed_overlap >= 0 else -1


class TransitionDensityTracker:
    """Track an excited state across geometry/time changes.

    Two construction paths:

      * ``TransitionDensityTracker(td_ref, state_ref)`` -- reference data
        comes from a gpu4pyscf TDDFT object.
      * ``TransitionDensityTracker.from_amplitudes(mol, mo_coeff, mo_occ,
        xy, ...)`` -- raw arrays for callers without a td object.

    Parameters (init form)
    ----------
    td_ref : TDBase
        Converged TDDFT object at the reference geometry. Must expose
        ``.xy`` (per-state amplitudes), ``.e`` (excitation energies), and
        ``._scf`` (with ``mo_coeff``, ``mo_occ``, ``mol``).
    state_ref : int
        1-indexed reference root to track.
    overlap_threshold : float, optional
        Below this absolute overlap, flag 'low_overlap'. Default 0.7.
    near_degenerate_ratio : float, optional
        |S_top| / |S_runner| below this, flag 'near_degenerate'. Default 1.2.
    energy_jump_ha : float, optional
        |E_match - E_ref| above this (Ha), flag 'energy_jump'. Default 0.05.
    """

    def __init__(self, td_ref, state_ref, *,
                 overlap_threshold: float = 0.7,
                 near_degenerate_ratio: float = 1.2,
                 energy_jump_ha: float = 0.05):
        self._init_thresholds(state_ref, overlap_threshold,
                              near_degenerate_ratio, energy_jump_ha)
        mf = td_ref._scf
        self._capture_from_data(mf.mol, mf.mo_coeff, mf.mo_occ,
                                td_ref.xy, e_ref=td_ref.e)

    @classmethod
    def from_amplitudes(cls, mol_ref, mo_coeff_ref, mo_occ_ref, xy_ref, *,
                        state_ref: int = 1, e_ref=None,
                        overlap_threshold: float = 0.7,
                        near_degenerate_ratio: float = 1.2,
                        energy_jump_ha: float = 0.05):
        """Build a tracker from raw reference arrays instead of a td object.

        Useful for callers that diagonalize the TDDFT/TDA matrix manually
        (e.g. ``nac/finite_diff.py``).

        Parameters
        ----------
        mol_ref : pyscf.gto.Mole
            Reference geometry.
        mo_coeff_ref, mo_occ_ref :
            RKS: arrays of shape ``(nao, nmo)`` and ``(nmo,)``.
            UKS: stacked ``(2, nao, nmo)`` / ``(2, nmo)`` or matching tuples.
        xy_ref : list of (X, Y)
            Same convention as ``td.xy``. For TDA, Y is 0 (RKS) or (0, 0)
            (UKS). UKS X is ``(Xa, Xb)``.
        state_ref : int, optional
            1-indexed root index.
        e_ref : float | array-like | None, optional
            Reference excitation energy or array of energies. If a scalar,
            used directly. If array-like, ``e_ref[state_ref - 1]`` is used.
            If None, the ``de_target`` diagnostic and ``energy_jump`` flag
            are disabled.
        """
        instance = cls.__new__(cls)
        instance._init_thresholds(state_ref, overlap_threshold,
                                  near_degenerate_ratio, energy_jump_ha)
        instance._capture_from_data(mol_ref, mo_coeff_ref, mo_occ_ref,
                                    xy_ref, e_ref=e_ref)
        return instance

    def _init_thresholds(self, state_ref, overlap_threshold,
                         near_degenerate_ratio, energy_jump_ha):
        self.state_ref0 = int(state_ref) - 1
        if self.state_ref0 < 0:
            raise ValueError("state_ref must be >= 1 (1-indexed).")
        self.overlap_threshold = float(overlap_threshold)
        self.near_degenerate_ratio = float(near_degenerate_ratio)
        self.energy_jump_ha = float(energy_jump_ha)

    def _capture_from_data(self, mol_ref, mo_coeff_ref, mo_occ_ref, xy_ref,
                           *, e_ref=None):
        self._is_uks = self._is_uks_layout(mo_coeff_ref)
        self._is_tddft = self._is_tddft_layout(xy_ref[0][1])

        if self.state_ref0 >= len(xy_ref):
            raise IndexError(
                f"state_ref={self.state_ref0 + 1} but xy_ref only has "
                f"{len(xy_ref)} roots.")

        if self._is_uks:
            x_part, _ = xy_ref[0]
            if not (isinstance(x_part, (tuple, list)) and len(x_part) == 2):
                raise NotImplementedError(
                    "Spin-flip TDDFT amplitudes are not supported.")

        # Snapshot a Mole copy so subsequent in-place set_geom_ on the
        # original (e.g. inside BOMD) does not corrupt the reference.
        self.mol_ref = mol_ref.copy()

        if self._is_uks:
            mo_a, mo_b = mo_coeff_ref
            occ_a, occ_b = mo_occ_ref
            self.C_ref_a = _to_numpy(mo_a)
            self.C_ref_b = _to_numpy(mo_b)
            self.occ_ref_a = _to_numpy(occ_a)
            self.occ_ref_b = _to_numpy(occ_b)
            self._nocca_ref = int(np.sum(self.occ_ref_a > 0))
            self._noccb_ref = int(np.sum(self.occ_ref_b > 0))
        else:
            self.C_ref = _to_numpy(mo_coeff_ref)
            self.occ_ref = _to_numpy(mo_occ_ref)
            self._nocc_ref = int(np.sum(self.occ_ref > 0))

        self.T_ref = self._extract_T(xy_ref[self.state_ref0])

        if e_ref is None:
            self.e_ref = float('nan')
        else:
            try:
                e_arr = _to_numpy(e_ref)
                if e_arr.ndim == 0:
                    self.e_ref = float(e_arr)
                else:
                    self.e_ref = float(e_arr[self.state_ref0])
            except (TypeError, AttributeError, IndexError):
                self.e_ref = float(e_ref)

    @staticmethod
    def _is_uks_layout(mo_coeff) -> bool:
        if isinstance(mo_coeff, (tuple, list)):
            return True
        try:
            arr = _to_numpy(mo_coeff)
        except Exception:
            return False
        return arr.ndim == 3

    @staticmethod
    def _is_tddft_layout(y) -> bool:
        if y is None:
            return False
        if isinstance(y, (int, float)) and y == 0:
            return False
        if isinstance(y, (tuple, list)) and all(
                isinstance(yy, (int, float)) and yy == 0 for yy in y):
            return False
        return True

    @staticmethod
    def _detect_uks(td) -> bool:
        return TransitionDensityTracker._is_uks_layout(td.xy[0][0])

    @staticmethod
    def _detect_tddft(td) -> bool:
        return TransitionDensityTracker._is_tddft_layout(td.xy[0][1])

    def _extract_T(self, xy_entry):
        """Build the matching tensor T = X (TDA) or X+Y (TDDFT), normalized
        to unit Frobenius norm so the matching metric is cosine similarity in
        [-1, 1]. UKS variants are normalized jointly across spins."""
        x_part, y_part = xy_entry
        if self._is_uks:
            xa, xb = x_part
            xa = _to_numpy(xa); xb = _to_numpy(xb)
            if self._is_tddft:
                ya, yb = y_part
                xa = xa + _to_numpy(ya)
                xb = xb + _to_numpy(yb)
            norm = np.sqrt(np.sum(xa * xa) + np.sum(xb * xb))
            if norm < 1e-14:
                raise RuntimeError("Zero-norm transition tensor.")
            return (xa / norm, xb / norm)
        x = _to_numpy(x_part)
        if self._is_tddft:
            x = x + _to_numpy(y_part)
        norm = np.linalg.norm(x)
        if norm < 1e-14:
            raise RuntimeError("Zero-norm transition tensor.")
        return x / norm

    def re_anchor(self, td_new, state_ref=None):
        """Replace the reference snapshot with data from ``td_new`` while
        preserving thresholds. Optionally retarget a different reference
        root via ``state_ref`` (1-indexed).

        Used by rolling-reference workflows (BOMD, geom-opt scanner) where
        each step's matched state becomes the next step's reference, so
        smooth character drift is followed without re-allocating a tracker.
        """
        if state_ref is not None:
            new0 = int(state_ref) - 1
            if new0 < 0:
                raise ValueError("state_ref must be >= 1 (1-indexed).")
            self.state_ref0 = new0
        mf = td_new._scf
        self._capture_from_data(mf.mol, mf.mo_coeff, mf.mo_occ,
                                td_new.xy, e_ref=td_new.e)

    def assign(self, td_disp, *, require_converged: bool = True) -> AssignmentResult:
        """Identify which root in `td_disp` best matches the reference target.

        Raises if convergence is required but a solved root failed, or if the
        shell type / method changes.
        """
        conv = getattr(td_disp, 'converged', None)
        if require_converged and conv is not None:
            conv_list = list(conv)
            if not all(conv_list):
                bad = [i for i, c in enumerate(conv_list) if not c]
                raise RuntimeError(
                    f"td_disp roots {bad} did not converge. "
                    "Pass require_converged=False to override.")

        return self.assign_amplitudes(
            td_disp._scf.mol, td_disp._scf.mo_coeff, td_disp.xy,
            e_disp=td_disp.e)

    def assign_amplitudes(self, mol_disp, mo_coeff_disp, xy_disp, *,
                          e_disp=None) -> AssignmentResult:
        """Lower-level assign that takes raw arrays.

        Assumes the displaced ``mo_coeff_disp`` orders occupied MOs in the
        first ``nocc_ref`` columns (true for energy-ordered SCF output and
        for the Hungarian reorderings used in ``nac/finite_diff.py``).
        """
        is_uks_disp = self._is_uks_layout(mo_coeff_disp)
        if is_uks_disp != self._is_uks:
            raise RuntimeError("mo_coeff_disp shell type differs from reference.")
        is_tddft_disp = self._is_tddft_layout(xy_disp[0][1])
        if is_tddft_disp != self._is_tddft:
            raise RuntimeError("xy_disp method (TDA vs TDDFT) differs from reference.")

        S_mix = self._cross_overlap(mol_disp, self.mol_ref)

        if self._is_uks:
            T_aligned = self._align_uks(mo_coeff_disp, S_mix)
            scores = np.array(
                [self._frob_uks(self._extract_T(xy_disp[k]), T_aligned)
                 for k in range(len(xy_disp))])
        else:
            T_aligned = self._align_rks(mo_coeff_disp, S_mix)
            scores = np.array(
                [self._frob_rks(self._extract_T(xy_disp[k]), T_aligned)
                 for k in range(len(xy_disp))])

        abs_scores = np.abs(scores)
        order = np.argsort(abs_scores)[::-1]
        K_star = int(order[0])

        if e_disp is not None:
            e_disp_arr = _to_numpy(e_disp).copy()
        else:
            e_disp_arr = np.full(len(xy_disp), np.nan)

        if not np.isnan(self.e_ref) and e_disp is not None:
            de_target = float(abs(e_disp_arr[K_star] - self.e_ref))
        else:
            de_target = float('nan')

        if len(order) > 1:
            runner = (int(order[1]), float(abs_scores[order[1]]))
        else:
            runner = (-1, 0.0)

        result = AssignmentResult(
            root=K_star,
            overlap=float(abs_scores[K_star]),
            signed_overlap=float(scores[K_star]),
            runner_up=runner,
            energies=e_disp_arr,
            de_target=de_target,
        )

        if result.overlap < self.overlap_threshold:
            result.flags.append('low_overlap')
        if runner[1] > 0 and result.overlap / max(runner[1], 1e-12) < self.near_degenerate_ratio:
            result.flags.append('near_degenerate')
        if not np.isnan(result.de_target) and result.de_target > self.energy_jump_ha:
            result.flags.append('energy_jump')
        if K_star != self.state_ref0:
            result.flags.append('index_changed')

        return result

    @staticmethod
    def _cross_overlap(mol_a, mol_b):
        from pyscf import gto
        return np.asarray(gto.intor_cross('int1e_ovlp', mol_a, mol_b))

    def _align_rks(self, mo_coeff_disp, S_mix):
        C_d = _to_numpy(mo_coeff_disp)
        nocc = self._nocc_ref
        U = C_d.T @ S_mix @ self.C_ref          # (nmo_d, nmo_r)
        U_o = U[:nocc, :nocc]
        U_v = U[nocc:, nocc:]
        if U_o.shape != self.T_ref.shape[:1] + (nocc,) or \
                U_v.shape[1] != self.T_ref.shape[1]:
            raise RuntimeError(
                f"MO partition mismatch: U_o {U_o.shape}, U_v {U_v.shape}, "
                f"T_ref {self.T_ref.shape}.")
        return U_o @ self.T_ref @ U_v.T

    def _align_uks(self, mo_coeff_disp, S_mix):
        C_da, C_db = mo_coeff_disp
        C_da = _to_numpy(C_da); C_db = _to_numpy(C_db)

        Ua = C_da.T @ S_mix @ self.C_ref_a
        Ub = C_db.T @ S_mix @ self.C_ref_b

        na, nb = self._nocca_ref, self._noccb_ref
        Uoa, Uva = Ua[:na, :na], Ua[na:, na:]
        Uob, Uvb = Ub[:nb, :nb], Ub[nb:, nb:]

        Ta_ref, Tb_ref = self.T_ref
        if (Uoa.shape[1], Uva.shape[1]) != Ta_ref.shape:
            raise RuntimeError(
                f"Alpha MO partition mismatch: U_o {Uoa.shape}, U_v {Uva.shape}, "
                f"T_a_ref {Ta_ref.shape}.")
        if (Uob.shape[1], Uvb.shape[1]) != Tb_ref.shape:
            raise RuntimeError(
                f"Beta MO partition mismatch: U_o {Uob.shape}, U_v {Uvb.shape}, "
                f"T_b_ref {Tb_ref.shape}.")

        return (Uoa @ Ta_ref @ Uva.T,
                Uob @ Tb_ref @ Uvb.T)

    @staticmethod
    def _frob_rks(T_disp, T_aligned) -> float:
        return float(np.einsum('ia,ia->', T_disp, T_aligned))

    @staticmethod
    def _frob_uks(T_disp, T_aligned) -> float:
        Ta_d, Tb_d = T_disp
        Ta_a, Tb_a = T_aligned
        return float(np.einsum('ia,ia->', Ta_d, Ta_a) +
                     np.einsum('ia,ia->', Tb_d, Tb_a))
