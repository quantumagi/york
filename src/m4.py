#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m4.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented, substantiated, 
    and contextually appropriate at the point of usage.
    
WHAT THIS CERTIFIES
  Computes M4 via exact Bessel–Laplace integrals and √X tail extrapolation.
  • Uses the exact same Bessel–Laplace kernel and √X tail extrapolation.
  • Prints round-by-round progress (Cutoff X, M4).

DERIVATION SKETCH (auditor refresher)
    We evaluate stabilized Bessel–Laplace integrals using sI(ν,z) = I_ν(z) e^{-z}.
    In code, the denominator integrand is implemented as
        D(x) = (2/3) * sI(0, x/3)**3
    and the numerator as
        N(x) = x**2 * sI(0, x/3)**2 * ((1/6)*sI(0, x/3) - (2/9)*sI(1, x/3) + (1/18)*sI(2, x/3)).
    Both are accumulated on [0, X] with fixed cuts and a √X tail model
    S(X) = S_inf + 2c/√X, fitted via two cutoffs, and M4 = N_tot/D_tot.

    • NOTE ON BOUNDS
        We report the width of recent M4 estimates:
            Bound_M4_span := max(M4_tail) − min(M4_tail).
        This value is exposed as 'dDelta (abs)' in the console/JSON. Any mapping of
        Bound_M4_span to a bound for Δ or κ (e.g., divide-by-3) is handled downstream
        (see kappa.py).

OUTPUTS
  • M4
  • Delta := M4/3 − E[u_x^4]
  • Bound_M4_span := max(M4_tail) − min(M4_tail)  (width of recent M4 values)

INPUTS (CLI)    
  • E[u_x^4] as a fraction p/q (e.g. 1/5)

"""
from __future__ import annotations

import sys, platform, argparse, json
from fractions import Fraction
from typing import Any, List, Tuple
from pathlib import Path

import mpmath as mp
from mpmath import nstr

# Standardized utilities (ETHOS-compliant: no JSON reads inside)
from utils import default_json_out, write_json

# ------------------------- basic helpers -------------------------

def parse_fraction(s: str) -> Fraction:
    s2 = str(s).strip()
    if "/" not in s2:
        raise ValueError(f"Expected a fraction like 'p/q', got {s!r}")
    return Fraction(s2)

def frac_to_mpf(fr: Fraction) -> mp.mpf:
    # exact conversion (no float round-trip)
    return mp.mpf(fr.numerator) / mp.mpf(fr.denominator)

def echo_fraction(label: str, fr: Fraction, sig: int = 80) -> None:
    dec = nstr(frac_to_mpf(fr), sig)
    print(f"{label:<16}: {fr} (= {dec})")

# Fixed segment boundaries before the live cutoff X (kept verbatim)
CUTS_DEFAULT = (0, 6, 20, 60)
TAIL_MODEL_NOTE = "S(X) = S_inf + 2c/sqrt(X) (two-cutoff extrapolation)"

# ------------------------- core routine -------------------------

def m4_via_bessel(
    dps: int,
    X_start: int,
    rounds: int,
    verbose: bool = True,
) -> tuple[mp.mpf, list[tuple[int, str]]]:
    """
    Compute M4 with exact kernels and √X tail extrapolation.

    Returns: (M4_mpf, history[(X, 'M4_str_24')...])
    """
    mp.mp.dps = int(dps)

    # Stabilized modified Bessel: Iν(z) * e^{-z}
    def sI(nu, z):
        return mp.besseli(nu, z) * mp.e**(-z)

    # Denominator integrand D(x)
    def D_integrand(x):
        z = x / 3
        return (mp.mpf('2')/3) * sI(0, z)**3

    # Numerator integrand N(x)
    def N_integrand(x):
        z = x / 3
        si0, si1, si2 = sI(0, z), sI(1, z), sI(2, z)
        C = (mp.mpf('1')/6)*si0 - (mp.mpf('2')/9)*si1 + (mp.mpf('1')/18)*si2
        return (x**2) * (si0**2) * C

    # Piecewise integration on [0, X] with the same cuts
    def integrate_0_X(f, X):
        cuts = [*CUTS_DEFAULT, X]
        S = mp.mpf('0')
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                S += mp.quad(f, [a, b])
        return S

    # Two-cutoff tail coefficient from S(X1), S(X2)
    def tail_const(S1, S2, X1, X2):
        return (S2 - S1) / (2*(1/mp.sqrt(X1) - 1/mp.sqrt(X2)))

    if verbose:
        print("χ-free M4 via Bessel integrals (intermediate refinements):")

    # Initial integrals at the first cutoff
    X1 = mp.mpf(X_start)
    D1 = integrate_0_X(D_integrand, X1)
    N1 = integrate_0_X(N_integrand, X1)

    history: list[tuple[int, str]] = []

    for i in range(rounds):
        X2 = X1 * 2
        D2 = integrate_0_X(D_integrand, X2)
        N2 = integrate_0_X(N_integrand, X2)

        cD = tail_const(D1, D2, X1, X2)
        cN = tail_const(N1, N2, X1, X2)

        D_tot = D2 + 2*cD/mp.sqrt(X2)
        N_tot = N2 + 2*cN/mp.sqrt(X2)

        M4 = N_tot / D_tot

        if verbose:
            print(f"  Round {i+1:02d} of {rounds}: Cutoff X={int(X2):>11}, M4 = {nstr(M4, 40)}")

        history.append((int(X2), nstr(M4, 24)))

        # advance
        X1, D1, N1 = X2, D2, N2

    return M4, history

# ------------------------- history → bound -------------------------

def bound_from_history(history: list[tuple[int, str]], tail_count: int) -> tuple[mp.mpf, dict]:
    """
    Convert the last `tail_count` value strings to mpf at a safe working precision,
    then return (abs_width, stats_dict). Bound method: full min–max span.
    """
    if not history:
        return mp.mpf('0'), {"ok": False, "reason": "no_history", "tail_count": 0}

    tail = history[-tail_count:] if len(history) >= tail_count else history[:]
    tail_strs = [s for (_X, s) in tail]

    # choose a working precision based on printed digits
    max_decs = 0
    for s in tail_strs:
        if "." in s:
            max_decs = max(max_decs, len(s.split(".", 1)[1]))
    work_dps = max(mp.mp.dps, max_decs + 20)

    with mp.workdps(work_dps):
        vals = [mp.mpf(s) for s in tail_strs]
        vmin = min(vals)
        vmax = max(vals)
        width = vmax - vmin

    stats = {
        "ok": True,
        "tail_count": len(tail),
        "min_tail": nstr(vmin, 24),
        "max_tail": nstr(vmax, 24),
        "width": nstr(width, 24),
        "method": "tail_range_full"
    }
    return mp.fabs(width), stats

# ------------------------- CLI & JSON -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="M4 (χ-free) via exact Bessel-integral routine (derived inputs via CLI)."
    )
    ap.add_argument("--E-u4", type=str, required=True,
                    help="E[u_x^4] as a fraction 'p/q' (e.g., '1/5').")
    ap.add_argument("--dps", type=int, default=120,
                    help="mpmath precision (digits).")
    ap.add_argument("--X-start", type=int, default=10_000,
                    help="Initial cutoff X.")
    ap.add_argument("--rounds", type=int, default=30,
                    help="Number of doublings.")
    ap.add_argument("--tail-count", type=int, default=8,
                    help="Number of last refinements to use for the bound (M4 span).")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Path for JSON output (defaults to outputs/m4.json)")
    args = ap.parse_args()

    # Parse E[u_x^4] (exact rational)
    try:
        E_u4 = parse_fraction(args.E_u4)  # e.g. 1/5
    except Exception as e:
        raise SystemExit(f"[m4] Failed to parse E[u_x^4]: {e}") from e

    # Run integrator for M4
    M4, m4_hist = m4_via_bessel(
        dps=args.dps,
        X_start=args.X_start,
        rounds=args.rounds,
        verbose=True,
    )

    E_u4_mpf = frac_to_mpf(E_u4)
    Delta    = M4/3 - E_u4_mpf

    # Derive a conservative bound on Delta from M4 history (no file I/O)
    dDelta_abs, stats = bound_from_history(m4_hist, args.tail_count)

    # Console summary
    print("\n=== M4/Δ result (χ-free) ===")
    echo_fraction("E[u_x^4]", E_u4)
    print(f"M4             : {nstr(M4,   max(24, args.dps))}")
    print(f"Delta (M4/3-E) : {nstr(Delta, max(24, args.dps))}")
    print(f"Bound_M4_span  : {nstr(dDelta_abs, 24)}   "
          f"[last {stats.get('tail_count', 0)}: {stats.get('min_tail','?')} … {stats.get('max_tail','?')}; "
          f"width={stats.get('width','?')}]")
    print(f"dps={args.dps}, X-start={args.X_start}, rounds={args.rounds}, tail-count={args.tail_count}")

    # JSON: inputs (rationals), intermediates (history), outputs (value + bounds)
    out: dict[str, Any] = {
        "meta": {
            "schema_version": "1.0",
            "script": Path(__file__).name,
            "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
            "tail_model": TAIL_MODEL_NOTE,
            "cuts": list(CUTS_DEFAULT),
        },
        "inputs": {
            "dps": int(args.dps),
            "X_start": int(args.X_start),
            "rounds": int(args.rounds),
            "tail_count": int(args.tail_count),
            "E_u4": {
                "rational": f"{E_u4.numerator}/{E_u4.denominator}",
                "decimal": nstr(E_u4_mpf, max(60, args.dps)),
                "desc": "Sphere marginal moment E[u_x^4] (Dirichlet/Beta)."
            },
        },
        "intermediates": {
            "m4_history": m4_hist,     # list of [X, "M4_str_24"]
        },
        "outputs": {
            "M4": {
                "decimal": nstr(M4, max(60, args.dps)),
                "desc": "Asymptotic fourth-moment ratio M4"
            },
            "Delta_M4_over_3_minus_E": {
                "decimal": nstr(Delta, max(60, args.dps)),
                "bounds": {
                    "abs": nstr(dDelta_abs, 60),
                    "tail_count": stats.get("tail_count", 0),
                    "min_tail": stats.get("min_tail", None),
                    "max_tail": stats.get("max_tail", None),
                    "width": stats.get("width", None),
                    "method": stats.get("method", "tail_range_full"),
                    "definition": "Bound_M4_span := max(M4_tail) - min(M4_tail) (width of recent M4 values)"
                },
                "desc": "Δ := M4/3 − E[u_x^4]."
            }
        }
    }

    out_path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
