#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zdir.py — Fejér main-lobe share via protocol lock (no x_q; no numerics)

ETHOS NOTES
  • Dependency results are only imported from upstream scripts, not hard-coded in this script.
  • Constants and formulas that use invariable constants are documented and justified by citation.
  • Any derivations are clearly explained and rooted in referenced theory.

PURPOSE
  Under the Fejér power window K(x)=sinc^2 x and the *quarter-FWHM lock* K(x_q)=R
  (R is provided by the pipeline as --lock-ratio, e.g. 8/15), the per-direction
  main-lobe share in the quartic channel is

      ζ_dir = (3/5) · R

  and with N parity sectors the per-sector share is

      ζ_wedge = ζ_dir / N.

  For R = 8/15 and N = 5 this yields ζ_dir = 8/25 and ζ_wedge = 8/125.

CITATION SKETCH
  • Fejér (power) window & FWHM practice: Harris (Proc. IEEE, 1978); Oppenheim & Schafer.
  • Sector/equidistribution on the main-lobe cap at the mid-edge, consistent with the
    pipeline’s parity/sector decomposition (cf. zparity.py and num_orbits.py).
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import mpmath as mp

from utils import (
    parse_rat_or_float,
    default_json_out,
    write_json,
    make_meta,
    ledger_header,
    console_show,
    require,
)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fejér main-lobe share from protocol lock: ζ_dir = (3/5)·(lock_ratio); ζ_wedge = ζ_dir/N."
    )
    ap.add_argument("--lock-ratio", required=True,
                    help="Protocol lock ratio R where K(x_q)=R (e.g. '8/15' from hypothesis.py).")
    ap.add_argument("--num-sectors", required=True, type=int,
                    help="Parity sector count N (e.g. 5 from num_orbits.py).")
    args = ap.parse_args()

    # Parse exact lock ratio (returns rational tag and mpf value)
    lock_rat_str, lock_rat = parse_rat_or_float(args.lock_ratio)
    N = int(args.num_sectors)
    require(N > 0, "num-sectors must be a positive integer.")

    # Compute shares directly from the lock ratio (single-line derivation)
    three_fifths = mp.mpf(3) / mp.mpf(5)
    z_dir   = three_fifths * lock_rat
    z_wedge = z_dir / mp.mpf(N)

    # Build exact rational outputs when lock was given as p/q
    z_dir_rat = None
    z_wedge_rat = None
    if lock_rat_str is not None:
        R = Fraction(lock_rat_str)           # exact R
        z_dir_frac = Fraction(3, 5) * R      # ζ_dir = (3/5) R
        z_dir_rat = f"{z_dir_frac.numerator}/{z_dir_frac.denominator}"
        z_wedge_frac = z_dir_frac / N        # ζ_wedge = ζ_dir / N
        z_wedge_rat = f"{z_wedge_frac.numerator}/{z_wedge_frac.denominator}"

    # Console
    ledger_header("Fejér main-lobe share (protocol lock)")
    console_show("lock_ratio", lock_rat_str, lock_rat)
    console_show("ζ_dir",   z_dir_rat,   z_dir)
    console_show("ζ_wedge", z_wedge_rat, z_wedge)

    # JSON (no Python floats; rational + high-precision decimal)
    out = {
        "meta": make_meta(__file__,
                          description="Main-lobe share from lock ratio: ζ_dir=(3/5)·R; ζ_wedge=ζ_dir/N.",
                          ethos_note="CLI-only; uses protocol lock; no JSON reads; no numerical integration."),
        "inputs": {
            "lock_ratio": {
                **({"rational": lock_rat_str} if lock_rat_str else {}),
                "decimal": mp.nstr(lock_rat, 50)
            },
            "num_sectors": {"int": N}
        },
        "outputs": {
            "z_dir": {
                **({"rational": z_dir_rat} if z_dir_rat else {}),
                "decimal": mp.nstr(z_dir, 50)
            },
            "z_wedge": {
                **({"rational": z_wedge_rat} if z_wedge_rat else {}),
                "decimal": mp.nstr(z_wedge, 50)
            }
        },
        "status": {"ok": True}
    }

    out_path = default_json_out(None, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
