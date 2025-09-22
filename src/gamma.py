#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gamma.py — Unified mid-edge amplitude Γ (product of upstream factors; CLI only)

ETHOS
  • Consumes ONLY upstream scalars via CLI (provided by run.py).
  • No JSON reads, no embedded theory constants, no network I/O.
  • Uses mpmath.mpf exclusively (no Python float / Decimal).
  • Deterministic JSON written to outputs/gamma.json (or --json-out).

WHAT THIS CERTIFIES
  Deterministic assembly of Γ = ζ_parity · J2 · ζ_dir · C_A1g from upstream-only inputs.

INPUTS
  --z-parity, --j2, --z-dir, --c-a1g  (each “p/q” or decimal)

OUTPUT
  outputs/gamma.json with inputs echoed ({raw, rational?, decimal}) and
  outputs.Gamma.{decimal, rational?}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import mpmath as mp

# ---- Standardized utilities (ETHOS-compliant; no JSON reads) ----
from utils import (
    parse_number,            # returns ParsedNumber with .float (mpf) and .rational tag
    mpf_to_rational_str,     # rational reconstruction helper
    default_json_out,
    write_json,
    make_meta,
    ledger_header,
    console_show,
    ensure_finite,
)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute Γ = z_parity * J2 * z_dir * C_A1g (CLI-only, mpf-only)."
    )
    ap.add_argument("--z-parity", required=True, type=str, help='Parity factor ("p/q" or decimal string).')
    ap.add_argument("--j2",        required=True, type=str, help='Quarter-hop Jacobian squared ("p/q" or decimal).')
    ap.add_argument("--z-dir",     required=True, type=str, help='Per-direction main-lobe share ("p/q" or decimal).')
    ap.add_argument("--c-a1g",     required=True, type=str, help='A1g projection coefficient ("p/q" or decimal).')
    ap.add_argument("--json-out",  required=False, type=str, help='Optional output path (default: outputs/gamma.json)')
    args = ap.parse_args()

    # Parse inputs uniformly (mpf only; no Python floats/Decimal)
    pn_zpar = parse_number(args.z_parity, reconstruct_if_float=True)
    pn_j2   = parse_number(args.j2,       reconstruct_if_float=True)
    pn_zdir = parse_number(args.z_dir,    reconstruct_if_float=True)
    pn_a1g  = parse_number(args.c_a1g,    reconstruct_if_float=True)

    # Sanity: ensure all are finite mpf
    ensure_finite([
        ("z_parity", pn_zpar.float),
        ("J2",       pn_j2.float),
        ("z_dir",    pn_zdir.float),
        ("C_A1g",    pn_a1g.float),
    ])

    # Product in mp.mpf
    gamma_mpf = pn_zpar.float * pn_j2.float * pn_zdir.float * pn_a1g.float

    # Try exact rational reconstruction (tight tolerances)
    gamma_rat = mpf_to_rational_str(gamma_mpf, abs_tol='1e-24', rel_tol='1e-24', max_den=10**9)

    # Console ledger (mpf only)
    ledger_header("Γ (mid-edge amplitude; CLI-only product)")
    console_show("z_parity", pn_zpar.rational, pn_zpar.float)
    console_show("J2",       pn_j2.rational,   pn_j2.float)
    console_show("z_dir",    pn_zdir.rational, pn_zdir.float)
    console_show("C_A1g",    pn_a1g.rational,  pn_a1g.float)
    console_show("Gamma",    gamma_rat,        gamma_mpf)
    print()

    # JSON payload (emit strings/rationals; no Python floats)
    out: Dict[str, Any] = {
        "meta": make_meta(__file__,
            description="CLI-only multiplication of upstream scalars (mpf-only).",
            ethos_note="ETHOS: CLI in → JSON out; no embedded constants; no Python floats/Decimal."),
        "inputs": {
            "z_parity": {"raw": pn_zpar.raw, "rational": pn_zpar.rational,
                         "decimal": mp.nstr(pn_zpar.float, 60)},
            "J2":       {"raw": pn_j2.raw,   "rational": pn_j2.rational,
                         "decimal": mp.nstr(pn_j2.float, 60)},
            "z_dir":    {"raw": pn_zdir.raw, "rational": pn_zdir.rational,
                         "decimal": mp.nstr(pn_zdir.float, 60)},
            "C_A1g":    {"raw": pn_a1g.raw,  "rational": pn_a1g.rational,
                         "decimal": mp.nstr(pn_a1g.float, 60)},
        },
        "outputs": {
            "Gamma": {
                **({"rational": gamma_rat} if gamma_rat is not None else {}),
                "decimal": mp.nstr(gamma_mpf, 60),
                "desc": "Mid-edge amplitude Γ = ζ_parity · J2 · ζ_dir · C_A1g (mpf arithmetic).",
            }
        },
        "status": {"ok": True},
    }

    out_path: Path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
