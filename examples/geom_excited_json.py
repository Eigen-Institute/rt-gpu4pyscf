"""
GPU4PySCF excited-state geometry optimization + numerical Hessian /
vibrational analysis.

Same JSON schema as geom_json.py with an optional "excited" block:

    "excited": {
        "state":   1,        // 1-indexed root from the TDDFT solver
        "nstates": 5,        // number of states to solve for
        "tda":     true,     // TDA (true) or full TDDFT (false)
        "fd_step_bohr": 0.005 // optional, default 0.005 Bohr
    }

If "excited" is absent the script behaves identically to geom_json.py.

The numerical Hessian is built by central finite differences of the analytical
TDDFT/TDA nuclear gradient at the excited-state minimum, then fed to
pyscf.hessian.thermo.harmonic_analysis.  Frequency / normal-mode output is
written in the same vib.json format as the ground-state script, so the
existing project_modes.py works without changes.

Cost: 6 * natoms gradient evaluations.  Each call is ~ one SCF + one TDDFT
solve + one CPHF.  For 4 atoms (formaldehyde) this is 24 gradient calls,
minutes on a GPU.
"""

import argparse
import json
import sys
import time

import numpy as np
from pyscf import gto
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo

from gpu4pyscf import dft
from gpu4pyscf.tdscf.state_tracking import TransitionDensityTracker

sys.path.append("/home/craig/research/templates/eigen")
import footer  # noqa: E402

start_time = time.time()

parser = argparse.ArgumentParser(description="GPU4PySCF excited-state opt + freq")
parser.add_argument("--input", "-i", type=str, required=True)
args = parser.parse_args()

with open(args.input) as f:
    cfg = json.load(f)

calc_name = cfg.get("calcName", "geom_opt").replace(" ", "_").lower()

mol_data = cfg.get("molecule", {})
mol = gto.M(
    atom=mol_data.get("atom", "opt.xyz"),
    basis=mol_data.get("basis", "3-21g"),
    verbose=mol_data.get("verbose", 4),
    charge=mol_data.get("charge", 0),
    spin=mol_data.get("spin", 0),
)

theory = cfg.get("theory", {})
shell = theory.get("shell", "closed").lower()
if shell == "open":
    ks = dft.UKS(mol)
elif shell == "closed":
    ks = dft.RKS(mol)
else:
    raise ValueError(f"Unknown shell {shell!r}")
ks.xc = theory.get("xc", "pbe0")
ks.chkfile = theory.get("initial guess", calc_name + ".chk")
ks.init_guess = "chk"
ks.kernel()

excited = cfg.get("excited")
geomopt = cfg.get("geomopt", {})
maxsteps = geomopt.get("maxsteps", 100)
opt_xyz = geomopt.get("output_file", calc_name + "_opt.xyz")


def _build_td(scf_obj):
    """Return a fresh TDA/TDDFT object on top of scf_obj."""
    tda = bool(excited.get("tda", True))
    nstates = int(excited.get("nstates", max(int(excited.get("state", 1)), 5)))
    td = scf_obj.TDA() if tda else scf_obj.TDDFT()
    td.nstates = nstates
    td.kernel()
    return td


def write_xyz(symbols, coords_ang, fname, comment):
    with open(fname, "w") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords_ang):
            fh.write(f"{sym:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")


# ---------- Geometry optimization ----------
if excited is None:
    print("\n=== Ground-state optimization ===")
    mol_eq = geometric_solver.optimize(ks, maxsteps=maxsteps)
    grad_for_hess = None  # use analytical ks.Hessian()
else:
    state = int(excited.get("state", 1))
    print(f"\n=== Excited-state ({'TDA' if excited.get('tda', True) else 'TDDFT'}) "
          f"optimization on root {state} ===")
    td = _build_td(ks)
    grad_method = td.nuc_grad_method()
    grad_method.state = state
    excited_scanner = grad_method.as_scanner(state=state)
    mol_eq = geometric_solver.optimize(excited_scanner, maxsteps=maxsteps)

symbols = [a[0] for a in mol_eq.atom]
coords_ang = mol_eq.atom_coords(unit="Ang")
write_xyz(
    symbols,
    coords_ang,
    opt_xyz,
    f"Optimized geometry {calc_name} @{mol.basis}/{ks.xc}"
    + (f" state {excited['state']}" if excited else ""),
)
print(f"Optimized geometry written to {opt_xyz}")

# ---------- Properties (Hessian + vib) ----------
props = cfg.get("properties", {})
want_hess = props.get("Hessian", False)
want_vib = props.get("vib", False)

if want_hess or want_vib:
    print("\n" + "=" * 50)
    print("--- Properties calculation ---")
    print("=" * 50)

    # Re-converge SCF (and TDDFT) at the optimized geometry
    ks.reset(mol_eq)
    ks.kernel()

    if excited is None:
        print("\nAnalytical ground-state Hessian...")
        t0 = time.time()
        hessian = ks.Hessian().kernel()
        print(f"Hessian time: {time.time() - t0:.2f} s")
    else:
        # Rebuild TDDFT at the optimized geometry to seed the FD loop and
        # snapshot the transition-density reference for root-following.
        td = _build_td(ks)
        state_ref = int(excited["state"])
        tracker = TransitionDensityTracker(td, state_ref=state_ref)

        # Reference gradient + sanity check on residual force
        e_eq, g_eq = td.nuc_grad_method().as_scanner(state=state_ref)(mol_eq)
        gmax = float(np.max(np.abs(g_eq)))
        print(f"\n|grad|_max at optimized geometry = {gmax:.2e} Ha/Bohr "
              f"(should be < ~1e-3 if opt converged)")

        step = float(excited.get("fd_step_bohr", 0.005))
        allow_bad = bool(excited.get("allow_bad_match", False))
        tda = bool(excited.get("tda", True))
        nstates_d = int(excited.get("nstates", max(state_ref, 5)))

        natoms = mol_eq.natm
        coords_bohr = mol_eq.atom_coords(unit="Bohr").copy()
        hessian = np.zeros((natoms, natoms, 3, 3))

        print(f"\nNumerical excited-state Hessian: "
              f"{2 * 3 * natoms} gradient evaluations, step = {step} Bohr")
        t0 = time.time()
        for i in range(natoms):
            for a in range(3):
                for sign in (+1, -1):
                    disp = coords_bohr.copy()
                    disp[i, a] += sign * step
                    mol_disp = mol_eq.set_geom_(disp, unit="Bohr", inplace=False)

                    # Re-converge SCF + TDDFT at displaced geometry
                    ks_d = ks.copy(); ks_d.reset(mol_disp); ks_d.kernel()
                    if not ks_d.converged:
                        raise RuntimeError(
                            f"SCF did not converge at coord ({i},{a},{sign:+d}).")
                    td_d = ks_d.TDA() if tda else ks_d.TDDFT()
                    td_d.nstates = nstates_d
                    td_d.kernel()

                    match = tracker.assign(td_d)
                    bad = set(match.flags) & {'low_overlap', 'energy_jump'}
                    if bad and not allow_bad:
                        raise RuntimeError(
                            f"State-tracking flags at ({i},{a},{sign:+d}): "
                            f"{match.flags}; |S|={match.overlap:.3f}, "
                            f"runner_up={match.runner_up}, "
                            f"|dE|={match.de_target:.3e} Ha. "
                            f"Set excited.allow_bad_match=true to override.")
                    print(f"  ({i},{a},{sign:+d}): root={match.state_1indexed()} "
                          f"|S|={match.overlap:.3f} runner-up={match.runner_up} "
                          f"flags={match.flags or ['ok']}")

                    g = np.asarray(td_d.nuc_grad_method().kernel(
                        state=match.state_1indexed()))

                    if sign == +1:
                        g_plus = g
                    else:
                        # H_{i a, j b} ~ (g_plus - g_minus) / (2 * step)
                        d_g = (g_plus - g) / (2.0 * step)
                        hessian[i, :, a, :] = d_g
                idx = 3 * i + a + 1
                print(f"  coord {idx}/{3 * natoms} done "
                      f"(t = {time.time() - t0:.1f} s)")
        # Symmetrize  H_{iajb} = H_{jbia}
        H_flat = hessian.transpose(0, 2, 1, 3).reshape(3 * natoms, 3 * natoms)
        H_flat = 0.5 * (H_flat + H_flat.T)
        hessian = H_flat.reshape(natoms, 3, natoms, 3).transpose(0, 2, 1, 3)
        print(f"Numerical Hessian time: {time.time() - t0:.1f} s")

    if want_hess:
        hess_file = f"{calc_name}_hessian.npy"
        np.save(hess_file, hessian)
        print(f"Hessian shape {hessian.shape} -> {hess_file}")

    if want_vib:
        print("\nVibrational analysis...")
        vib = thermo.harmonic_analysis(ks.mol, hessian)
        freqs = vib["freq_wavenumber"]
        if np.iscomplexobj(freqs):
            freqs = freqs.real - np.abs(freqs.imag)
        print(f"\n{'Mode':>5} {'Freq (cm^-1)':>15} {'Red. Mass (AMU)':>15}")
        for i, (f, rm) in enumerate(zip(freqs, vib["reduced_mass"])):
            print(f"{i + 1:5d} {f:15.2f} {rm:15.4f}")
        out = {
            "frequencies_cm1": freqs.tolist(),
            "reduced_mass_amu": vib["reduced_mass"].tolist(),
            "normal_modes": vib["norm_mode"].tolist(),
        }
        if excited is not None:
            out["excited_state"] = {
                "state": int(excited["state"]),
                "tda": bool(excited.get("tda", True)),
                "nstates": int(excited.get("nstates", 5)),
                "fd_step_bohr": float(excited.get("fd_step_bohr", 0.005)),
            }
        with open(f"{calc_name}_vib.json", "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Vibrational results -> {calc_name}_vib.json")

print(f"\n     wall time: {time.time() - start_time:.1f} s")
footer.print_footer()
