#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chitt.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented from pre-existing
    theory, substantiated, and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Computes the transverse–traceless linear-response scale χ_TT from an upstream 
  York constant C_Q, with exact-fraction propagation whenever the upstream provides 
  an exact Fraction via utils.parse_number.

INPUTS
  1) --C-Q <scalar or exact fraction>   (required; produced by cq.py)
  [Optional] --json-out <path>          (output JSON path)

DERIVATION (no (a,b))
  • Cubic symmetry (O_h) ⇒ χ_TT = 3 · C_Q (sum over xy, yz, zx planes).
    (Dresselhaus–Dresselhaus–Jorio, *Group Theory*, Springer 2008;
     York, *J. Math. Phys.* 14, 456–464 (1973))

OUTPUT (JSON → outputs/chitt.json)
  {
    "meta": {...},
    "inputs": {...},
    "intermediates": {...},
    "outputs": {
      "chi_tt":   { "decimal_24": "...", "rational": "p/q" }
    },
    "status": {"ok": true}
  }
"""

from __future__ import annotations
import argparse, platform, sys
from pathlib import Path
from typing import Optional
from fractions import Fraction

import mpmath as mp

# Shared repo utilities (authoritative parsing + JSON I/O)
from utils import (
    default_json_out,   # (user_path: Optional[str], this_file: str) -> Path
    write_json,         # (path: Path, obj: dict) -> None
    parse_number,       # returns an object with .float and .fraction (if input was exact)
)

# ------------------------------- printing -------------------------------
DEC_SIG = 24
def nstr_plain(x: mp.mpf, sig: int = DEC_SIG) -> str:
    return mp.nstr(mp.mpf(x), sig)

# --------------------- theory-fixed, non-tunable constants ---------------------
# χ_TT = 3·C_Q (sum over xy/yz/zx planes under cubic symmetry O_h).
# Refs near use: Dresselhaus–Dresselhaus–Jorio (2008); York (1973).
YORK_ORIENTATION_MULT = mp.mpf('3')

def comb2(d: int) -> int:
    return d * (d - 1) // 2

# ---------------------------------- main ----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive chi_tt from York C_Q (exact-fraction aware; no rational guessing)."
    )
    ap.add_argument("--C-Q", dest="C_Q", type=str, required=True,
                    help="C_Q value (float or exact fraction). Upstream: cq.py")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Optional output path (default: outputs/chitt.json).")
    args = ap.parse_args()

    mp.mp.dps = 80  # robust internal precision

    # Parse via shared utils; if upstream passed a fraction, utils now exposes .fraction
    pn = parse_number(args.C_Q)
    C_Q = pn.float
    C_Q_frac: Optional[Fraction] = pn.fraction if isinstance(pn.fraction, Fraction) else None

    # Sanity (not a default): confirm 3 = C(3,2)
    if int(YORK_ORIENTATION_MULT) != comb2(3):
        raise SystemExit("[chitt] Internal check failed: 3 ≠ C(3,2).")

    # Results
    chi_tt   = YORK_ORIENTATION_MULT * C_Q

    # Exact propagation if C_Q was rational
    chi_frac: Optional[Fraction] = None
    tt_frac: Optional[Fraction] = None
    if C_Q_frac is not None:
        chi_frac = Fraction(3, 1) * C_Q_frac
        tt_frac  = Fraction(1, 6) * chi_frac

    # Console summary (exact prints when available)
    print("\n=== χ_TT and TT drift from York constant ===")
    print(f"C_Q (input)              : {args.C_Q}")
    print(f"χ_TT = 3·C_Q             : {nstr_plain(chi_tt)}")
    if chi_frac is not None:
        print(f"  (exact)                : {chi_frac.numerator}/{chi_frac.denominator}")
    print()

    # JSON — data only; references live in comments near the constants above.
    out: dict = {
        "meta": {
            "schema_version": "1.0",
            "script": Path(__file__).name,
            "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
        },
        "inputs": {
            "C_Q": {
                "raw": args.C_Q,
                "decimal_24": nstr_plain(C_Q),
                **({"rational": f"{C_Q_frac.numerator}/{C_Q_frac.denominator}"} if C_Q_frac else {})
            }
        },
        "intermediates": {
            "constants": {
                "YORK_ORIENTATION_MULT": {"expr": "3", "decimal_24": nstr_plain(YORK_ORIENTATION_MULT)},
                "n_planes_theory": {"int": comb2(3)}  # C(3,2)=3
            }
        },
        "outputs": {
            "chi_tt": {
                "decimal_24": nstr_plain(chi_tt),
                **({"rational": f"{chi_frac.numerator}/{chi_frac.denominator}"} if chi_frac else {})
            }
        },
        "status": {"ok": True}
    }

    out_path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
