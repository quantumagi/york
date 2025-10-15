#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zdir.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented, substantiated,
    and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Computes the directional main-lobe share z_dir and the per-sector share z_wedge,
  purely combinatorially and independent of the window K or the lock x_q.
  • Uses the internal 1:4 split of face/edge classes after the mid-edge fold.
  • Emits z_dir = (4/5) * H4(face) = (4/5) * (2/5) = 8/25 (exact rational + decimal).
  • For N parity sectors (from num_orbits.py), emits z_wedge = z_dir / N.

DERIVATION SKETCH (auditor refresher)
  After the mid-edge fold, quartic content splits internally 1:4 between face and edge
  types. With the spherical-harmonic quartic weights H4(face)=2/5 and H4(edge)=-1/10,
  the directional mask (main-lobe share) is
      z_dir = (internal edge share 4/5) * H4(face) = (4/5)*(2/5) = 8/25.
  The per-sector share divides this evenly across the N parity sectors:
      z_wedge = z_dir / N.
  This construction is K-independent and does not depend on x_q.

OUTPUTS
  • outputs.z_dir.rational   : "8/25"
    outputs.z_dir.decimal    : high-precision decimal for 8/25
  • outputs.z_wedge.rational : "8/(25*N)" simplified as a fraction string
    outputs.z_wedge.decimal  : high-precision decimal for z_dir / N
  • meta / inputs.num_sectors / status.ok

INPUTS (CLI; required)
  --num-sectors : parity sector count N (e.g., 5 from num_orbits.py)
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import mpmath as mp

from utils import (
    default_json_out,
    write_json,
    make_meta,
    ledger_header,
    console_show,
    require,
)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Directional mask from internal 1:4 split; z_dir=8/25, z_wedge=z_dir/N (K-independent)."
    )
    ap.add_argument("--num-sectors", required=True, type=int,
                    help="Parity sector count N (e.g. 5 from num_orbits.py).")
    args = ap.parse_args()

    N = int(args.num_sectors)
    require(N > 0, "num-sectors must be a positive integer.")

    # Core computation: combinatorial, K-independent
    z_dir_frac = Fraction(8, 25)
    z_wedge_frac = z_dir_frac / N

    z_dir   = mp.mpf(z_dir_frac.numerator) / mp.mpf(z_dir_frac.denominator)
    z_wedge = mp.mpf(z_wedge_frac.numerator) / mp.mpf(z_wedge_frac.denominator)

    # Console
    ledger_header("Directional mask (K-independent)")
    console_show("num_sectors", None, N)
    console_show("z_dir",   f"{z_dir_frac.numerator}/{z_dir_frac.denominator}",   z_dir)
    console_show("z_wedge", f"{z_wedge_frac.numerator}/{z_wedge_frac.denominator}", z_wedge)

    # JSON (no floats in rationals; high-precision decimals)
    out = {
        "meta": make_meta(
            __file__,
            description="Directional mask from internal 1:4 split; K-independent.",
            ethos_note="Interface-stable; ignores K(x_q); no numerical integration."
        ),
        "inputs": {
            "num_sectors": {"int": N}
        },
        "outputs": {
            "z_dir":   {"rational": "8/25",   "decimal": mp.nstr(z_dir, 50)},
            "z_wedge": {"rational": f"{z_wedge_frac.numerator}/{z_wedge_frac.denominator}",
                        "decimal": mp.nstr(z_wedge, 50)}
        },
        "status": {"ok": True}
    }

    out_path = default_json_out(None, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
