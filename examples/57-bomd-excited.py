#!/usr/bin/env python
"""Excited-state BOMD example: H2O on the S1 (B3LYP/6-31g*) surface.

Maxwell-Boltzmann velocity initialization at 300 K, velocity-Verlet
integration of nuclei, gradients from gpu4pyscf TDDFT scanner. Asserts
that the conserved Ehrenfest-style total energy (E_elec + T_nuc, where
E_elec = mf.e_tot + td.e[state-1]) drifts by less than 1e-3 Ha over the
trajectory.
"""
import numpy as np
from pyscf import gto

from gpu4pyscf import dft
from gpu4pyscf.tdscf.ehrenfest import BOMD
from gpu4pyscf.tdscf import rtutils as rtu


def main():
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='6-31g*',
        verbose=0,
    )
    ks = dft.RKS(mol); ks.xc = 'b3lyp'
    ks.kernel()

    # 5-root TDDFT solve seeds the scanner with a good initial guess.
    td = ks.TDDFT(); td.nstates = 5
    td.kernel()
    print(f"SCF E0:        {ks.e_tot:.8f} Ha")
    print(f"S1 excitation: {float(td.e[0]):.6f} Ha "
          f"(={float(td.e[0]) * 27.2114:.2f} eV)")

    md = BOMD(ks, td=td, state=1)
    md.verbose = 4
    md.thermostat = None  # NVE for conservation check
    md.com_step = 100
    md.velocities = rtu.maxwell_boltzmann_velocities(md.masses, 300.0,
                                                     rng=np.random.default_rng(0))
    rtu.remove_com_momentum(md.masses, md.velocities)

    dt = 20.0  # a.u. (~0.48 fs)
    n_steps = 50
    times = np.arange(0, n_steps * dt + dt * 0.5, dt)
    results = md.kernel(times=times, dt=dt)

    e_tot = np.array(results['energy_tot'])
    drift = e_tot - e_tot[0]
    print(f"\nE_tot[0] = {e_tot[0]:.8f} Ha   over {n_steps} steps at dt={dt} au")
    print(f"max |dE| = {np.max(np.abs(drift)):.3e} Ha")
    print(f"final dE = {drift[-1]:+.3e} Ha")

    assert np.max(np.abs(drift)) < 1e-3, "energy drift too large"
    print("PASS: excited-state BOMD energy is conserved within tolerance.")


if __name__ == '__main__':
    main()
