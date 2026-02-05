#!/home/craig/pyvenv/bin/python
"""
pyscf_spectrum.py
Adapted from nw_spectrum.py for PySCF TDDFT output.

Parses PySCF TDDFT output for excitation energies and oscillator strengths,
and optionally generates a Lorentzian broadened spectrum.
"""

import sys
import argparse
import math
import textwrap

VERSION = "1.0"

def ev2au(e_ev):
    return (1.0 / 27.2114) * e_ev

def au2ev(e_au):
    return 27.2114 * e_au

def ev2nm(e_ev):
    # E(eV) = 1239.84193 / lambda(nm)
    return 1239.84193 / e_ev

def parse_pyscf_tddft(lines):
    """
    Parses lines from PySCF output.
    Looks for: Excited State   1:      0.41936 eV   2956.52 nm  f=0.0016
    """
    roots = []
    for line in lines:
        if "Excited State" in line and ":" in line and "eV" in line and "f=" in line:
            parts = line.split()
            try:
                # Find energy in eV
                energy_ev = None
                for i, p in enumerate(parts):
                    if p == "eV":
                        energy_ev = float(parts[i-1])
                        break
                
                # Find oscillator strength f=...
                osc = None
                for p in parts:
                    if p.startswith("f="):
                        osc = float(p.split('=')[1])
                        break
                
                if energy_ev is not None and osc is not None:
                    roots.append([energy_ev, osc])
            except (ValueError, IndexError):
                continue
    return roots

def make_energy_list(roots, npoints, width):
    if not roots:
        return []
    
    # Padding around the roots
    epad = 20.0 * width
    emin = min(r[0] for r in roots) - epad
    emax = max(r[0] for r in roots) + epad
    
    # Avoid negative energies if they don't make sense (optional)
    # if emin < 0: emin = 0
    
    de = (emax - emin) / (npoints - 1)
    
    # Ensure width is at least 2 grid points to avoid aliasing
    if width < 2 * de:
        width = 2 * de
        sys.stderr.write(f"Warning: Forced broadening width to {width:.4f} eV\n")
        
    energies = [emin + i * de for i in range(npoints)]
    return energies, width

def gen_spectrum(roots, energies, width):
    if not energies:
        return
    
    gamma = 0.5 * width
    gamma_sqrd = gamma * gamma
    de = (energies[-1] - energies[0]) / (len(energies) - 1)
    # normalization: integral of (gamma/pi) / (x^2 + gamma^2) is 1.
    # We multiply by de for the discrete sum representation if we want to match intensities?
    # nw_spectrum.py multiplied by de.
    prefac = (gamma / math.pi) * de

    for energy in energies:
        stot = 0.0
        for root_e, osc in roots:
            xx0 = energy - root_e
            # Lorentzian: L(E) = (gamma/pi) / ((E-E0)^2 + gamma^2)
            stot += osc / (xx0 * xx0 + gamma_sqrd)
        yield [energy, stot * prefac]

def dump_data(opts, spectrum_gen, roots):
    c = opts.comment
    if not opts.clean:
        print(f"{c} ================================")
        print(f"{c}  PySCF spectrum parser ver {VERSION}")
        print(f"{c} ================================")
        print(f"{c} Units: {opts.units}")
        if opts.width:
            print(f"{c} Broadening: {opts.width} eV")
        print(f"{c}")
        
        header1 = "Energy [eV]"
        if opts.units == "au": header1 = "Energy [au]"
        if opts.units == "nm": header1 = "Wavelen. [nm]"
        
        print(f"{c}{header1:>15}{opts.delim}{'Abs. [au]':>15}")
        print(f"{c}" + "-"*31)

    for energy, intensity in spectrum_gen:
        out_e = energy
        if opts.units == "au": out_e = ev2au(energy)
        elif opts.units == "nm": out_e = ev2nm(energy)
        
        print(f"{out_e:15.10e}{opts.delim}{intensity:15.10e}")

def main():
    parser = argparse.ArgumentParser(
        description="Parses PySCF TDDFT output and generates absorption spectra.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example:
              pyscf_spectrum.py -b 0.2 -p 2000 < tddft.out > spectrum.dat
        """)
    )
    
    parser.add_argument("-b", "--broaden", type=float, default=0.1, dest="width",
                        help="Lorentzian FWHM in eV (default: 0.1)")
    parser.add_argument("-p", "--points", type=int, default=2000,
                        help="Number of points in spectrum (default: 2000)")
    parser.add_argument("-w", "--units", choices=["ev", "au", "nm"], default="ev",
                        help="Units for energy/wavelength (default: ev)")
    parser.add_argument("-d", "--delim", default="    ",
                        help="Output delimiter (default: 4 spaces)")
    parser.add_argument("-x", "--extract", action="store_true",
                        help="Extract raw roots only (no broadening)")
    parser.add_argument("-C", "--clean", action="store_true",
                        help="Data only output (no headers/comments)")
    parser.add_argument("-c", "--comment", default="#",
                        help="Comment character (default: #)")

    args = parser.parse_args()

    # Read all lines from stdin
    lines = sys.stdin.readlines()
    roots = parse_pyscf_tddft(lines)
    
    if not roots:
        sys.stderr.write("Error: No TDDFT roots found in input.\n")
        sys.exit(1)

    if args.extract:
        # Just output the roots
        if not args.clean:
            print(f"{args.comment} Extracted raw roots (no broadening)")
            header1 = "Energy [eV]"
            if args.units == "au": header1 = "Energy [au]"
            if args.units == "nm": header1 = "Wavelen. [nm]"
            print(f"{args.comment}{header1:>15}{args.delim}{'f':>15}")

        for r_e, r_f in roots:
            out_e = r_e
            if args.units == "au": out_e = ev2au(r_e)
            elif args.units == "nm": out_e = ev2nm(r_e)
            print(f"{out_e:15.10e}{args.delim}{r_f:15.10e}")
    else:
        # Generate spectrum
        energies, final_width = make_energy_list(roots, args.points, args.width)
        args.width = final_width
        spectrum_gen = gen_spectrum(roots, energies, args.width)
        dump_data(args, spectrum_gen, roots)

if __name__ == "__main__":
    main()
