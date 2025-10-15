#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zparity.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented, substantiated, 
    and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  We certify the parity scalar
      ζ_parity := 8 · E[u_x^2 u_y^2 u_z^2] on S^2,
  by evaluating the spherical moment using product separation and mp.mpf-only quadrature:
      E[...] = (1/(4π)) [∫_{-1}^{1} (1−μ^2)^2 μ^2 dμ] · [∫_{0}^{2π} cos^2φ sin^2φ dφ].
  This avoids hard-coding 1/105 or 8/105; we compute them to round-off.

DERIVATION SKETCH (auditor refresher)
  With u = (sinθ cosφ, sinθ sinφ, cosθ) and μ = cosθ,
    E[u_x^2 u_y^2 u_z^2] = (1/4π) ∫_{-1}^{1}(1−μ^2)^2 μ^2 dμ · ∫_{0}^{2π} cos^2φ sin^2φ dφ.
  For reference (not used as a constant here): the exact value is 1/105 ⇒ ζ_parity = 8/105.

INPUTS (record-only; do not change the value)
  --window, --moment-power, --half-shift   Recorded for the runner; ζ_parity is angular-only.

OUTPUTS
  outputs/zparity.json with fields including:
    • outputs.moment.rational  = "1/105"  (when reconstruction succeeds)
    • outputs.z_parity.rational = "8/105" (when reconstruction succeeds)
    • matching high-precision decimal strings
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

import mpmath as mp

# ETHOS-safe helpers (no JSON reads; mpf-friendly I/O)
from utils import (
    default_json_out,
    write_json,
    make_meta,
    ledger_header,
    console_show,
    mpf_to_rational_str,
)

def integrand_mu(mu: mp.mpf) -> mp.mpf:
    """(1 - μ^2)^2 · μ^2, computed in mp.mpf."""
    mu2 = mu * mu
    r2 = mp.mpf(1) - mu2
    return (r2 * r2) * mu2

def integrand_phi(phi: mp.mpf) -> mp.mpf:
    """cos^2 φ · sin^2 φ, computed in mp.mpf."""
    c = mp.cos(phi); s = mp.sin(phi)
    return (c * c) * (s * s)

def compute_moment_and_zeta() -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """
    Compute:
      I_mu  = ∫_{-1}^{1} (1-μ^2)^2 μ^2 dμ
      I_phi = ∫_{0}^{2π} cos^2φ sin^2φ dφ
      moment = (I_mu * I_phi) / (4π)
      zeta   = 8 * moment
    All values are mp.mpf.
    """
    I_mu  = mp.quad(lambda t: integrand_mu(t),  [-1, 1])
    I_phi = mp.quad(lambda t: integrand_phi(t), [0, 2*mp.pi])
    moment = (I_mu * I_phi) / (4 * mp.pi)
    zeta   = 8 * moment
    return I_mu, I_phi, moment, zeta

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ζ_parity via separated mp.mpf quadratures (no floats, no hard-coded constants)."
    )
    # Recorded-only protocol flags (do not affect the computation)
    ap.add_argument("--window",        type=str, default="sinc2")
    ap.add_argument("--moment-power",  type=int, default=4)
    ap.add_argument("--half-shift",    type=int, default=1)
    ap.add_argument("--json-out",      type=str, default=None)
    
    mp.mp.dps = 200  # set precision early
    args = ap.parse_args()

    # Compute with current mp.mp.dps (parse_number in other scripts sets mp.mp.dps early in runs)
    I_mu, I_phi, moment, zeta = compute_moment_and_zeta()

    # Attempt exact rational reconstruction (tight tolerances OK with mpf integration)
    rat_moment = mpf_to_rational_str(moment, abs_tol='1e-20', rel_tol='1e-20', max_den=10**9)
    rat_zeta   = mpf_to_rational_str(zeta,   abs_tol='1e-20', rel_tol='1e-20', max_den=10**9)

    # Console summary (mpf-only)
    ledger_header("ζ_parity via product quadrature (deterministic)")
    print(f"Protocol flags     : window='{args.window}', moment_power={args.moment_power}, half_shift={args.half_shift}")
    console_show("I_mu",    None, I_mu)
    console_show("I_phi",   None, I_phi)
    console_show("moment",  rat_moment, moment)
    console_show("zeta_parity", rat_zeta, zeta)
    print()

    # JSON out (emit rationals when certified; always include high-precision decimal)
    out_path = default_json_out(args.json_out, __file__)
    payload: Dict[str, Any] = {
        "meta": make_meta(
            __file__,
            description="Parity factor via separated mp.mpf quadratures on S^2 (no floats).",
            ethos_note="No hard-coded ζ; computes I_mu and I_phi and forms ζ_parity = 8·moment.",
        ),
        "inputs": {
            "window":       {"value": args.window,            "desc": "Recorded protocol tag; angular-only computation."},
            "moment_power": {"value": int(args.moment_power), "desc": "Recorded even power used in parity derivation."},
            "half_shift":   {"value": int(args.half_shift),   "desc": "Recorded parity half-shift flag."},
        },
        "intermediates": {
            "I_mu":  mp.nstr(I_mu, 60),
            "I_phi": mp.nstr(I_phi, 60),
        },
        "outputs": {
            "moment": {
                **({"rational": rat_moment} if rat_moment else {}),
                "decimal": mp.nstr(moment, 60),
                "desc": "E[u_x^2 u_y^2 u_z^2] on S^2, computed from mpf quadratures.",
            },
            "z_parity": {
                **({"rational": rat_zeta} if rat_zeta else {}),
                "decimal": mp.nstr(zeta, 60),
                "desc": "ζ_parity = 8·E[u_x^2 u_y^2 u_z^2] on S^2, computed (no hard-coded constants).",
            },
        },
        "status": {"ok": True},
    }
    write_json(out_path, payload)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
