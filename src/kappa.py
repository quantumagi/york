#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kappa.py
 
ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented from pre-existing
    theory, substantiated, and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Compute κ_∞ in the TT sector via exact Bessel–Laplace kernels with a √X tail
  extrapolation model, using ONLY upstream rationals for χ_TT (unsigned) and
  E[u_x^4]. No file I/O for inputs; exact rationals on input/echo; mpf only at
  the final combination. The script ALSO derives an absolute uncertainty bound
  dkappa from the run’s own refinement history (tail range over the last T
  entries).

INPUTS
  --chi-tt     : χ_TT as a fraction 'p/q' from chitt.py (unsigned, e.g. '11/260')
  --E-u4       : E[u_x^4] as a fraction 'p/q' from h4.py (e.g., '1/5')
  --dps        : (optional) mpmath precision (digits), default 120
  --X-start    : (optional) initial cutoff X, default 10000
  --rounds     : (optional) number of doublings, default 30
  --tail-count : (optional) last N history entries to use for bound, default 8
  --json-out   : (optional) output path (defaults to outputs/kappa.json)

DERIVATION
  Denominator D(X) = ∫_0^X (2/3) sI_0(x/3)^3 dx,
  Numerator   N(X) = ∫_0^X x^2 sI_0(x/3)^2 [ (1/6)sI_0 - (2/9)sI_1 + (1/18)sI_2 ] dx,
  with sI_ν(z) = I_ν(z)e^{-z}. Two-cutoff √X tails:
     c(S; X1→X2) = (S(X2) - S(X1)) / ( 2(1/√X1 − 1/√X2) ),
     S_∞(X2) ≈ S(X2) + 2c/√X2.
  Then M4 = N_∞ / D_∞ and
     κ_∞ = 1/2 − χ_TT^signed · ( M4/3 − E[u_x^4] ),
  where χ_TT^signed = −χ_TT (unsigned input). The bound dkappa is the FULL RANGE
  (max−min) across the last T printed κ_∞ refinements in this run.

OUTPUT (JSON → outputs/kappa.json)
  {
    "meta": {...},
    "inputs": { "chi_tt_signed": {...}, "E_u4": {...}, ... },
    "intermediates": { "history": [[X1, "κ1"], [X2, "κ2"], ...] },
    "outputs": {
      "kappa_infty": {
        "decimal": "...",
        "bounds": {
          "abs": "...",
          "tail_count": T,
          "min_tail": "...",
          "max_tail": "...",
          "width": "...",
          "method": "tail_range_full"
        },
        "desc": "Continuum κ_∞ (TT sector)"
      },
      "M4": { "decimal": "...", "desc": "Asymptotic fourth-moment ratio M4" }
    }
  }
"""
from __future__ import annotations

import sys, platform, argparse, json
from fractions import Fraction
from typing import Any, List, Tuple
from pathlib import Path

import mpmath as mp
from mpmath import nstr

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

def kappa_tt_via_bessel(
    dps: int,
    X_start: int,
    rounds: int,
    chi_unsigned: Fraction,  # e.g. +11/260 from chitt.py
    E_u4: Fraction,          # e.g. 1/5 from h4.py
    verbose: bool = True,
) -> tuple[mp.mpf, mp.mpf, list[tuple[int, str]]]:
    """
    Compute κ_∞ with exact kernels and √X tail extrapolation.

    Returns: (kappa_mpf, M4_mpf, history[(X, 'kappa_str_24')...])
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
        print("TT-sector κ_∞ via Bessel integrals (intermediate refinements):")

    # Initial integrals at the first cutoff
    X1 = mp.mpf(X_start)
    D1 = integrate_0_X(D_integrand, X1)
    N1 = integrate_0_X(N_integrand, X1)

    # Prepare signed constants once (exact → mpf) for the affine map
    chi_signed = -chi_unsigned  # original κ-map uses the signed factor
    rat_offset_mpf = frac_to_mpf(Fraction(1, 2) + chi_signed*E_u4)  # (1/2 + χ_signed*E_u4)
    chi_over_3_mpf = (mp.mpf(chi_signed.numerator) /
                      mp.mpf(3 * chi_signed.denominator))          # χ_signed/3

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

        # κ_∞ = (1/2 + χ_signed*E_u4) − (χ_signed/3) * M4
        kappa = rat_offset_mpf - chi_over_3_mpf * M4

        if verbose:
            print(f"  Round {i+1:02d} of {rounds}: Cutoff X={int(X2):>11}, κ = {nstr(kappa, 40)}")

        history.append((int(X2), nstr(kappa, 24)))

        # advance
        X1, D1, N1 = X2, D2, N2

    return kappa, M4, history

# ------------------------- history → bound -------------------------

def bound_from_history(history: list[tuple[int, str]], tail_count: int) -> tuple[mp.mpf, dict]:
    """
    Convert the last `tail_count` κ strings to mpf at a safe working precision,
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
        description="κ_∞ (TT sector) via exact Bessel-integral routine (derived inputs via CLI)."
    )
    ap.add_argument("--chi-tt", type=str, required=True,
                    help="χ_TT as a fraction 'p/q' from chitt.py (positive, e.g. '11/260').")
    ap.add_argument("--E-u4", type=str, required=True,
                    help="E[u_x^4] as a fraction 'p/q' from h4.py (e.g., '1/5').")
    ap.add_argument("--dps", type=int, default=120,
                    help="mpmath precision (digits).")
    ap.add_argument("--X-start", type=int, default=10_000,
                    help="Initial cutoff X.")
    ap.add_argument("--rounds", type=int, default=30,
                    help="Number of doublings.")
    ap.add_argument("--tail-count", type=int, default=8,
                    help="Number of last refinements to use for dkappa bound (default 8).")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Path for JSON output (defaults to outputs/<script>.json)")
    args = ap.parse_args()

    # Parse Fractions (exact)
    try:
        chi_unsigned = parse_fraction(args.chi_tt)  # +11/260
        E_u4         = parse_fraction(args.E_u4)    # 1/5
    except Exception as e:
        raise SystemExit(f"[kappa] Failed to parse fractions: {e}") from e

    # Run integrator
    kappa, M4, diag = kappa_tt_via_bessel(
        dps=args.dps,
        X_start=args.X_start,
        rounds=args.rounds,
        chi_unsigned=chi_unsigned,
        E_u4=E_u4,
        verbose=True,
    )

    # Derive dkappa from this run's history (no files)
    dkappa_abs, stats = bound_from_history(diag, args.tail_count)

    # Console summary
    print("\n=== κ_∞ result (exact kernels; derived inputs via CLI) ===")
    echo_fraction("χ_TT (signed)", -chi_unsigned)  # show the signed one actually used
    echo_fraction("E[u_x^4]",       E_u4)
    print(f"κ_∞            : {nstr(kappa, max(24, args.dps))}")
    print(f"M4             : {nstr(M4,   max(24, args.dps))}")
    print(f"dkappa (abs)   : {nstr(dkappa_abs, 24)}   "
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
            "chi_tt_signed": {
                "rational": f"{(-chi_unsigned).numerator}/{(-chi_unsigned).denominator}",
                "decimal": nstr(frac_to_mpf(-chi_unsigned), max(60, args.dps)),
                "desc": "Signed χ_TT used in κ map (original sign convention)."
            },
            "E_u4": {
                "rational": f"{E_u4.numerator}/{E_u4.denominator}",
                "decimal": nstr(frac_to_mpf(E_u4), max(60, args.dps)),
                "desc": "Sphere marginal moment E[u_x^4] (Dirichlet/Beta)."
            },
        },
        "intermediates": {
            "history": diag,  # list of [X, "kappa_str_24"]
        },
        "outputs": {
            "kappa_infty": {
                "decimal": nstr(kappa, max(60, args.dps)),
                "bounds": {
                    "abs": nstr(dkappa_abs, 60),
                    "tail_count": stats.get("tail_count", 0),
                    "min_tail": stats.get("min_tail", None),
                    "max_tail": stats.get("max_tail", None),
                    "width": stats.get("width", None),
                    "method": stats.get("method", "tail_range_full"),
                },
                "desc": "Continuum κ_∞ (TT sector)"
            },
            "M4": {
                "decimal": nstr(M4, max(60, args.dps)),
                "desc": "Asymptotic fourth-moment ratio M4"
            },
        },
    }

    out_path = Path(args.json_out) if args.json_out else Path("outputs") / (Path(__file__).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
