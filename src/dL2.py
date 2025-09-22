#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dL2.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented from pre-existing
    theory, substantiated, and contextually appropriate at the point of usage.
  • No rational guessing: fractions are preserved if passed as "p/q"; otherwise
    decimals are used as provided.

WHAT THIS CERTIFIES
  Computes the quadratic anisotropy correction to the Fejér–York lock width,
      ΔL^(2) = (1/2) * f''(x_q) * ⟨H4^2⟩ * (χ_TT * Γ)^2,
  consuming ONLY upstream-provided scalars via CLI.

INPUTS (CLI; all required)
  --fpp      f''(x_q) from xq.py (window curvature at the lock)
  --H4-sq    ⟨H4^2⟩ from h4.py (quartic invariant second moment on S^2)
  --chi-tt   χ_TT from chitt.py (York TT linear-response scale)
  --Gamma    Γ from gamma.py (collector product: protocol/parity/Jacobian/etc.)
  [--json-out optional override for output path]

DERIVATION NOTES (context where used)
  • The quadratic term comes from the second-order expansion of the lock ledger
    in the small parameter p := χ_TT * Γ, with curvature set by f''(x_q).
    See: York (1973) for TT normalization; standard cumulant expansion /
    second-order response arguments; and the Fejér window curvature at the
    quarter–FWHM lock (x_q) as supplied by xq.py.
    References: York 1973 (JMP 14, 456–464); NIST DLMF §18, §24 for standard
    Legendre/Bernoulli background (used upstream).

OUTPUT (JSON → outputs/dL2.json)
  "outputs": { "deltaL2": { "decimal_48": "<…>" } }
"""

from __future__ import annotations
import argparse
from pathlib import Path
import mpmath as mp

# ---- standardized utilities (provided in utils.py) ----
from utils import (
    parse_number,
    default_json_out, write_json, make_meta,
    ledger_header, console_show, ensure_finite
)

def _node(pn) -> dict:
    """Standard scalar node: {raw, (rational), float:str}."""
    out = {"raw": pn.raw, "float": mp.nstr(pn.float, 60)}
    if pn.rational:
        out["rational"] = pn.rational
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ΔL^(2) from upstream values (mpmath arithmetic).")
    ap.add_argument("--fpp",     required=True,  help="f''(x_q) from xq.py")
    ap.add_argument("--H4-sq",   required=True,  help="⟨H4^2⟩ from h4.py")
    ap.add_argument("--chi-tt",  required=True,  help="χ_TT from chitt.py")
    ap.add_argument("--Gamma",   required=True,  help="Γ from gamma.py")
    ap.add_argument("--json-out", required=False, help="Override JSON output path (optional)")
    args = ap.parse_args()

    # Parse all inputs (supports exact 'p/q' or decimal) → mpf, with fraction preserved if given.
    # ETHOS: no rational guessing from decimals -> reconstruct_if_float=False.
    fpp    = parse_number(args.fpp,    reconstruct_if_float=False)
    H4_sq  = parse_number(args.H4_sq,  reconstruct_if_float=False)
    chi_tt = parse_number(args.chi_tt, reconstruct_if_float=False)
    Gamma  = parse_number(args.Gamma,  reconstruct_if_float=False)

    ensure_finite([
        ("fpp", fpp.float),
        ("H4_sq", H4_sq.float),
        ("chi_tt", chi_tt.float),
        ("Gamma", Gamma.float),
    ])

    # Core formula (placed here with references-in-context):
    # ΔL^(2) = (1/2) * f''(x_q) * ⟨H4^2⟩ * (χ_TT * Γ)^2
    #   – York TT scaling (χ_TT): York 1973, JMP 14, 456–464.
    #   – Window curvature f''(x_q): supplied by xq.py at the Fejér lock x_q.
    #   – ⟨H4^2⟩: spherical moment supplied by h4.py (Legendre/Beta background: NIST DLMF §26.5).
    half = mp.mpf('1')/2
    p    = chi_tt.float * Gamma.float
    deltaL2 = half * fpp.float * H4_sq.float * (p ** 2)

    # Console ledger
    ledger_header("ΔL^(2) (quadratic correction)")
    console_show("f''(x_q)",  fpp.rational,    fpp.float)
    console_show("<H4^2>",    H4_sq.rational,  H4_sq.float)
    console_show("chi_TT",    chi_tt.rational, chi_tt.float)
    console_show("Gamma",     Gamma.rational,  Gamma.float)
    print("-" * 40)
    print(f"{'p = χ_TT·Γ':>12} : {mp.nstr(p, 24)}")
    print(f"{'ΔL^(2)':>12} : {mp.nstr(deltaL2, 24)}   [-]")

    # JSON payload (stringified numerics; no Python floats)
    out = {
        "meta": make_meta(
            __file__,
            description="Quadratic lock correction ΔL^(2) = (1/2) f''(x_q) ⟨H4^2⟩ (χ_TT Γ)^2",
            ethos_note="Upstream-only inputs; mpmath arithmetic; no embedded constants; no rational guessing."
        ),
        "inputs": {
            "fpp":    _node(fpp),
            "H4_sq":  _node(H4_sq),
            "chi_tt": _node(chi_tt),
            "Gamma":  _node(Gamma),
        },
        "outputs": {
            "deltaL2": {"decimal_48": mp.nstr(deltaL2, 48)}
        },
        "status": {"ok": True}
    }

    out_path: Path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
