#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kappa.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented, substantiated, 
    and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Combines χ_TT (unsigned, from chitt.py) with a χ-free Δ (from m4.py) to produce κ_∞:
      κ_∞ = 0.5 − χ_TT^signed · Δ, with χ_TT^signed := −χ_TT.
  • Emits κ_∞ and (optionally) an absolute bound propagated from the Δ bound.
  • Prints a concise console summary (inputs, κ_base, κ_∞, optional dκ).

DERIVATION SKETCH (auditor refresher)
    Conventions:
      • χ_TT is provided unsigned (from chitt.py). We set χ_TT^signed := −χ_TT.
      • Δ := M4/3 − E[u_x^4] is taken directly from m4.py (χ-free).
      • κ_base := 0.5 (combiner intercept).
      • κ_∞ := κ_base − χ_TT^signed · Δ.

    NOTE ON BOUNDS
      • If a bound for Δ is provided (--dDelta), we propagate: dκ := |χ_TT| · dΔ.
      • --dDelta is not required by the scripts table and may be omitted.

OUTPUTS
  • kappa_infty.decimal     High-precision decimal string for κ_∞.
  • kappa_infty.bounds      Optional {"abs": ..., "method": "prop_dDelta"} when --dDelta is given.
  • JSON shape and path are stable for downstream consumers.

INPUTS (CLI)
  REQUIRED (per scripts table)
    --chi-tt   p/q              Unsigned χ_TT as a rational (e.g., 11/260).
    --Delta    <decimal>        Δ := M4/3 − E[u_x^4] (from m4.json).

  OPTIONAL (not in scripts table)
    --dDelta   <decimal>        Absolute bound for Δ (propagated as dκ = |χ_TT|·dΔ).
    --dps      <int>            mpmath precision in digits (default: 120).
    --json-out <path>           Output path (default: outputs/kappa.json).
"""

from __future__ import annotations
import sys, platform, argparse, json
from fractions import Fraction
from typing import Any
from pathlib import Path
import mpmath as mp
from mpmath import nstr

def parse_fraction(s: str) -> Fraction:
    s2 = str(s).strip()
    if "/" not in s2:
        raise ValueError(f"Expected a fraction like 'p/q', got {s!r}")
    return Fraction(s2)

def frac_to_mpf(fr: Fraction) -> mp.mpf:
    return mp.mpf(fr.numerator) / mp.mpf(fr.denominator)

def main() -> None:
    ap = argparse.ArgumentParser(description="Combine χ_TT with Δ to produce κ_∞.")
    ap.add_argument("--chi-tt", type=str, required=True,
                    help="Unsigned χ_TT as 'p/q' (e.g., '11/260').")
    ap.add_argument("--Delta", type=str, required=True,
                    help="Δ := M4/3 − E[u_x^4] (decimal; from m4.json).")
    ap.add_argument("--dDelta", type=str, required=False,
                    help="Absolute bound for Δ (decimal; optional).")
    ap.add_argument("--dps", type=int, default=120,
                    help="mpmath precision (digits; default 120).")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Output path (default: outputs/kappa.json)")

    args = ap.parse_args()

    mp.mp.dps = int(args.dps)

    try:
        chi_unsigned = parse_fraction(args.chi_tt)
    except Exception as e:
        raise SystemExit(f"[kappa] Failed to parse chi-tt: {e}") from e

    # Prefer Δ if provided; otherwise compute from M4 and E[u^4]
    if args.Delta is not None:
        Delta = mp.mpf(args.Delta)
        dDelta_abs = mp.mpf(args.dDelta) if args.dDelta is not None else None
    else:
        if args.M4 is None or args.E_u4 is None:
            raise SystemExit("[kappa] Provide either --Delta (preferred) or both --M4 and --E-u4.")
        try:
            M4 = mp.mpf(args.M4)
            E_u4 = parse_fraction(args.E_u4)
        except Exception as e:
            raise SystemExit(f"[kappa] Failed to parse M4/E[u_x^4]: {e}") from e
        Delta = M4/3 - frac_to_mpf(E_u4)
        dDelta_abs = None

    chi_signed = -chi_unsigned   # convention
    chi_signed_mpf = frac_to_mpf(chi_signed)

    kappa_base = mp.mpf('0.5')   # invariant intercept lives here (combiner)
    kappa_val  = kappa_base - chi_signed_mpf * Delta

    dkappa_abs = None
    if dDelta_abs is not None:
        dkappa_abs = mp.fabs(chi_signed_mpf) * mp.mpf(dDelta_abs)

    # Console summary
    print("\n=== kappa.py (combiner) ===")
    print(f"χ_TT (unsigned): {chi_unsigned} (= {nstr(frac_to_mpf(chi_unsigned), 40)})")
    print(f"χ_TT (signed)  : {-chi_unsigned} (= {nstr(chi_signed_mpf, 40)})")
    print(f"Delta          : {nstr(Delta, 40)}")
    print(f"κ_base         : 0.5")
    print(f"κ_∞            : {nstr(kappa_val, 40)}")
    if dkappa_abs is not None:
        print(f"dκ (abs)       : {nstr(dkappa_abs, 24)}  [propagated from dΔ]")

    # Build outputs safely (omit bounds if none)
    kappa_out = {
        "decimal": nstr(kappa_val, max(60, args.dps)),
        "desc": "Continuum κ_∞ (TT sector)"
    }
    if dkappa_abs is not None:
        kappa_out["bounds"] = {
            "abs": nstr(dkappa_abs, 60),
            "method": "prop_dDelta"
        }

    out: dict[str, Any] = {
        "meta": {
            "schema_version": "1.0",
            "script": Path(__file__).name,
            "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
        },
        "inputs": {
            "chi_tt_unsigned": {
                "rational": f"{chi_unsigned.numerator}/{chi_unsigned.denominator}",
                "decimal": nstr(frac_to_mpf(chi_unsigned), max(60, args.dps)),
                "desc": "Unsigned χ_TT (from chitt.py)"
            },
            "Delta": {
                "decimal": nstr(Delta, max(60, args.dps)),
                "desc": "Δ := M4/3 − E[u_x^4] (from m4.py or reconstructed)"
            },
        },
        "outputs": {
            "kappa_infty": kappa_out
        }
    }

    out_path = Path(args.json_out) if args.json_out else Path("outputs") / "kappa.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
