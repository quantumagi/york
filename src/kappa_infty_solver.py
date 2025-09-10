#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal κ∞ (TT sector) runner using the exact routine provided by the user.

- Keeps the integrand, cut windows, and two-cutoff tail model exactly as given
- Adds a tiny CLI and JSON output (no extra “strategies”)
"""

import argparse
import json
from typing import Any, List, Tuple
from pathlib import Path

import mpmath as mp

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

# -------------------------------
# EXACT routine (unchanged)
# -------------------------------
def kappa_tt_via_bessel(dps=40, X_start=10000, rounds=22):
    mp.mp.dps = dps

    def sI(nu, z):
        return mp.besseli(nu, z) * mp.exp(-z)

    def D_integrand(x):
        z = x/3
        return mp.mpf('2')/mp.mpf('3') * sI(0, z)**3

    def N_integrand(x):
        z = x/3
        si0, si1, si2 = sI(0, z), sI(1, z), sI(2, z)
        C = (mp.mpf('1')/6)*si0 - (mp.mpf('2')/9)*si1 + (mp.mpf('1')/18)*si2
        return (x**2) * (si0**2) * C

    def integrate_0_X(f, X):
        cuts = [0, 6, 20, 60, X]
        S = mp.mpf('0')
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b > a:
                S += mp.quad(f, [a, b])
        return S

    def tail_const(S1, S2, X1, X2):
        return (S2 - S1) / (2*(1/mp.sqrt(X1) - 1/mp.sqrt(X2)))

    print("TT-sector κ∞ via Bessel integrals (intermediate refinements):")
    
    X1 = mp.mpf(X_start)
    D1 = integrate_0_X(D_integrand, X1)
    N1 = integrate_0_X(N_integrand, X1)

    diag = []
    chi = -mp.mpf(11)/mp.mpf(260)
    
    for i in range(rounds):
        X2 = X1 * 2
        D2 = integrate_0_X(D_integrand, X2)
        N2 = integrate_0_X(N_integrand, X2)
        cD = tail_const(D1, D2, X1, X2)
        cN = tail_const(N1, N2, X1, X2)
        D_tot = D2 + 2*cD/mp.sqrt(X2)
        N_tot = N2 + 2*cN/mp.sqrt(X2)
        M4 = N_tot / D_tot
        kappa = mp.mpf('0.5') - chi*(M4/3 - mp.mpf('0.2'))
        
        # Display intermediate result immediately
        print(f"  Round {i+1} of {rounds}: Cutoff X={int(X2)}, κ = {float(kappa):.21f}")
        diag.append((int(X2), mp.nstr(kappa, 20)))
        
        X1, D1, N1 = X2, D2, N2

    return float(kappa), float(M4), diag

# -------------------------------
# Thin CLI wrapper
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="Minimal κ∞ runner (fast).")
    ap.add_argument("--dps", type=int, default=120, help="mpmath precision (digits).")
    ap.add_argument("--X-start", type=int, default=10000, help="initial cutoff X.")
    ap.add_argument("--rounds", type=int, default=22, help="number of doublings.")
    ap.add_argument("--json-out", type=str, default=None, help="JSON output path.")
    args = ap.parse_args()

    kappa, M4, diag = kappa_tt_via_bessel(dps=args.dps, X_start=args.X_start, rounds=args.rounds)

    print("\n=== κ∞ result (minimal) ===")
    print(f"κ∞        : {kappa:.18f}")
    print(f"M4        : {M4:.18f}")
    print(f"dps       : {args.dps}")
    print(f"X-start   : {args.X_start}")
    print(f"rounds    : {args.rounds}")

    # diag is already JSON-serializable (list of [int, str])
    res: dict[str, Any] = {
        "kappa_infty": kappa,
        "M4": M4,
        "dps": int(args.dps),
        "X_start": int(args.X_start),
        "rounds": int(args.rounds),
        "history": diag,
        "notes": [
            "Function body is exactly the user-supplied routine.",
            "Two-cutoff tail model: S(X) = S_inf + 2c/sqrt(X).",
            "Windows: [0,6], [6,20], [20,60], [60,X].",
        ],
    }
    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
