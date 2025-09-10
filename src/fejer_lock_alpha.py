#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 — Fejér–York lock → alpha and m_r (correct Bernoulli-series implementation)

This script evaluates the effective lock exponent L_eff using:
  L_eff = L0 + (11/1560) + S_odd + S_even + ΔL^(2)

where
  - L0 = -log( sinc(x_q)^2 ), with x_q > 0 s.t. sinc(x_q)^2 = 8/15
  - S_odd = sum_{k>=3} 1/(2k+1)!
  - S_even = sum_{n>=4} [ 2 B_{2n} / (n (2n)!) ] * s^{2n}, with s = 2/sqrt(3)
  - ΔL^(2) = (1/2) f''(x_q) * <H4^2> * (χ_TT)^2 * Γ^2
       with f''(x) = 2 (csc^2 x - x^{-2}),
            <H4^2> = 16/525,
            χ_TT   = 11/260,
            Γ      = 4/375.

Then:
  m_r     = exp(L_eff / m)
  alpha^-1 = kappa * m / (exp(2 L_eff / m) - 1)

Inputs:
  --tol           : absolute tolerance for series tail cutoffs / solver
  --m             : multiplicity (default 19)
  --kappa         : κ_∞ continuum constant
  --alpha-target  : (optional) target alpha^{-1} for reporting Δ
  --mr-target     : (optional) target m_r for reporting Δ
  --json-out      : (optional) path to write a JSON results dictionary

Requirements:
  Python 3.9+ and mpmath (no mixing with decimal/float beyond final casting).
"""

import argparse
import json
import math
import os
from typing import Dict, Any
from pathlib import Path

import mpmath as mp

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

# -------------------------------
# Utilities and core math pieces
# -------------------------------

def sinc(x: mp.mpf) -> mp.mpf:
    """sinc(x) = sin(x)/x with sinc(0)=1."""
    return mp.one if x == 0 else mp.sin(x) / x


def K_power_window(x: mp.mpf) -> mp.mpf:
    """Fejér main-lobe power: K(x) = sinc(x)^2."""
    s = sinc(x)
    return s * s


def fpp_fejer(x: mp.mpf) -> mp.mpf:
    """f''(x) for f(x) = -log(sinc(x)^2) = 2*(csc^2 x - x^{-2})."""
    # use mpmath csc as 1/sin
    return 2 * (mp.csc(x) ** 2 - (1 / (x * x)))


def solve_x_q(tol: mp.mpf) -> mp.mpf:
    """
    Solve for x_q > 0 such that K(x_q)=8/15 using mp.findroot.
    Provides two initial guesses for robustness.
    """
    target = mp.mpf(8) / mp.mpf(15)
    f = lambda x: K_power_window(x) - target
    # Use two bracketing guesses around ~1.33 (quarter-FWHM)
    xq = mp.findroot(f, (mp.mpf('1.2'), mp.mpf('1.4')))
    # refine to tol using one more step if needed (mp.findroot returns high-precision)
    return xq


def sum_S_odd(tol: mp.mpf, kmax: int = 1000) -> mp.mpf:
    """
    S_odd = sum_{k>=3} 1/(2k+1)! = 1/7! + 1/9! + ...
    Stops when term < tol or k reaches kmax.
    """
    total = mp.mpf('0')
    for k in range(3, kmax + 1):
        n = 2 * k + 1
        term = 1 / mp.factorial(n)
        total += term
        if term < tol:
            break
    return total


def S_even_bern_series(s: mp.mpf, tol: mp.mpf, nmax: int = 400) -> mp.mpf:
    """
    S_even(s) = sum_{n>=4} [ 2*B_{2n} / ( n * (2n)! ) ] * s^{2n}
    Uses mpmath.bernoulli for B_{2n}.
    Convergence is rapid since (s/(2π))^2 ≈ 0.0338 for s=2/√3.
    """
    total = mp.mpf('0')
    s2 = s * s
    for n in range(4, nmax + 1):
        B2n = mp.bernoulli(2 * n)
        term = (2 * B2n) / (n * mp.factorial(2 * n)) * (s2 ** n)
        total += term
        if abs(term) < tol:
            break
    return total


def to_float(x) -> float:
    """Cast mpmath mpf to Python float safely (for JSON)."""
    return float(x)


def ensure_dir(path: str) -> None:
    """Ensure the directory for path exists."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------------
# Main computation
# -------------------------------

def compute_results(tol: float,
                    m: int,
                    kappa: float,
                    alpha_target: float = None,
                    mr_target: float = None) -> Dict[str, Any]:
    # Precision: make it generous relative to tol
    # Aim for ~ -log10(tol) + margin
    if tol <= 0:
        raise ValueError("--tol must be positive.")
    digits = max(50, int(-math.log10(tol)) + 40)
    mp.mp.dps = digits

    # Inputs as mpf
    tol_mpf = mp.mpf(tol)
    m_mpf = mp.mpf(m)
    kappa_mpf = mp.mpf(kappa)

    # Constants from the paper
    york_linear = mp.mpf(11) / mp.mpf(1560)              # 11/1560
    H4_sq = mp.mpf(16) / mp.mpf(525)                     # ⟨H4^2⟩
    chi_TT = mp.mpf(11) / mp.mpf(260)                    # 11/260
    Gamma = mp.mpf(4) / mp.mpf(375)                      # 4/375
    s = mp.mpf(2) / mp.sqrt(3)                           # s = 2/√3
    target_8_over_15 = mp.mpf(8) / mp.mpf(15)

    # Solve for x_q
    x_q = solve_x_q(tol_mpf * mp.mpf('1e-3'))  # slightly looser for root-finding
    Kx = K_power_window(x_q)
    L0 = -mp.log(Kx)

    # f''(x_q)
    fpp = fpp_fejer(x_q)

    # Series pieces
    S_odd = sum_S_odd(tol_mpf)
    S_even = S_even_bern_series(s, tol_mpf, nmax=800)

    # Sanity checks
    residual = Kx - target_8_over_15
    if not (abs(residual) < mp.mpf('1e-12')):
        raise RuntimeError(f"x_q residual too large: {residual}")

    # The Bernoulli tail at s=2/√3 should be small and negative
    if not (S_even < 0):
        raise RuntimeError(f"S_even must be negative at s=2/sqrt(3), got {S_even}")

    # Universal quadratic correction ΔL^(2)
    delta_L2 = mp.mpf('0.5') * fpp * H4_sq * (chi_TT ** 2) * (Gamma ** 2)

    # Effective widths
    L_eff_no_quad = L0 + york_linear + S_odd + S_even
    L_eff_final = L_eff_no_quad + delta_L2

    # Outputs
    m_r_no_quad = mp.e ** (L_eff_no_quad / m_mpf)
    m_r_final = mp.e ** (L_eff_final / m_mpf)

    def alpha_inv_from(L: mp.mpf) -> mp.mpf:
        denom = mp.e ** (2 * L / m_mpf) - 1
        return kappa_mpf * m_mpf / denom

    alpha_inv_no_quad = alpha_inv_from(L_eff_no_quad)
    alpha_inv_final = alpha_inv_from(L_eff_final)

    # Prepare report
    result = {
        "x_q": to_float(x_q),
        "K(x_q)": to_float(Kx),
        "target_8_over_15": to_float(target_8_over_15),
        "residual": to_float(residual),
        "L0": to_float(L0),
        "york_linear_11_over_1560": to_float(york_linear),
        "S_odd": to_float(S_odd),
        "S_even": to_float(S_even),
        "fpp": to_float(fpp),
        "<H4^2>": to_float(H4_sq),
        "chi_TT": to_float(chi_TT),
        "Gamma": to_float(Gamma),
        "delta_L2": to_float(delta_L2),
        "L_eff_no_quad": to_float(L_eff_no_quad),
        "L_eff_final": to_float(L_eff_final),
        "m": m,
        "kappa": to_float(kappa_mpf),
        "m_r_no_quad": to_float(m_r_no_quad),
        "m_r_final": to_float(m_r_final),
        "alpha_inv_no_quad": to_float(alpha_inv_no_quad),
        "alpha_inv_final": to_float(alpha_inv_final),
    }

    if alpha_target is not None:
        d_alpha = to_float(alpha_inv_final - mp.mpf(alpha_target))
        result["alpha_inv_target"] = float(alpha_target)
        result["alpha_inv_delta"] = d_alpha

    if mr_target is not None:
        d_mr = to_float(m_r_final - mp.mpf(mr_target))
        result["m_r_target"] = float(mr_target)
        result["m_r_delta"] = d_mr

    return result


def print_report(res: Dict[str, Any]) -> None:
    # Make the printout match the earlier style
    print("\n=== S5: Fejér–York lock → α and m_r (Bernoulli series) ===")
    print(f"x_q                    : {res['x_q']:.16f}  (K(x_q)={res['K(x_q)']:.16f}, residual={res['residual']:.3e})")
    print(f"L0 = -log K(x_q)       : {res['L0']:.16f}  (≈ log(15/8))")
    print(f"York linear (11/1560)  : {res['york_linear_11_over_1560']:.16f}")
    print(f"S_odd (k≥3)            : {res['S_odd']:.16e}")
    print(f"S_even(s=2/√3)         : {res['S_even']:.16e}  (note=Bernoulli tail n≥4)")
    print(f"f''(x_q)               : {res['fpp']:.16f}")
    print(f"<H4^2>                 : {res['<H4^2>']:.16f}")
    print(f"χ_TT                   : {res['chi_TT']:.16f}")
    print(f"Γ                      : {res['Gamma']:.16f}")
    print(f"ΔL^(2)                 : {res['delta_L2']:.16e}")
    print(f"L_eff (no quad)        : {res['L_eff_no_quad']:.16f}")
    print(f"L_eff (final)          : {res['L_eff_final']:.16f}")
    print(f"m                      : {res['m']}")
    print(f"κ_∞                    : {res['kappa']:.17f}")
    print(f"m_r (no quad)          : {res['m_r_no_quad']:.16f}")
    print(f"m_r (final)            : {res['m_r_final']:.16f}")
    print(f"α^-1 (no quad)       : {res['alpha_inv_no_quad']:.20f}")
    print(f"α^-1 (final)         : {res['alpha_inv_final']:.20f}")

    if "alpha_inv_target" in res:
        print(f"Δ vs target α^-1     : {res['alpha_inv_final'] - res['alpha_inv_target']:+.3e}")
    if "m_r_target" in res:
        print(f"Δ vs target m_r        : {res['m_r_final'] - res['m_r_target']:+.3e}")


def main():
    ap = argparse.ArgumentParser(description="S5 — Fejér–York lock → alpha and m_r (Bernoulli series)")
    ap.add_argument("--tol", type=float, default=1e-30, help="absolute tolerance for series/root (default: 1e-30)")
    ap.add_argument("--m", type=int, default=19, help="multiplicity m (default: 19)")
    ap.add_argument("--kappa", type=float, default=0.49926851200152955, help="κ_∞ (default: 0.49926851200152955)")
    ap.add_argument("--alpha-target", type=float, default=None, help="optional target α^{-1} to report delta")
    ap.add_argument("--mr-target", type=float, default=None, help="optional target m_r to report delta")
    ap.add_argument("--json-out", type=str, default=None, help="optional path to write JSON results")
    args = ap.parse_args()

    res = compute_results(
        tol=args.tol,
        m=args.m,
        kappa=args.kappa,
        alpha_target=args.alpha_target,
        mr_target=args.mr_target,
    )

    print_report(res)

    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
