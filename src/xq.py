#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xq.py  — Fejér quarter-FWHM lock and curvature (high-precision, rigorous bisection)

This version uses mpmath everywhere (configurable dps), keeps the lock ratio as an
exact rational ("8/15"), derives L0 exactly as log(15/8), and emits high-precision
decimal strings in JSON (alongside legacy floats for backward compatibility).
"""

from __future__ import annotations
from fractions import Fraction
import sys, platform, argparse, json
from pathlib import Path
from typing import Any, Dict, Tuple
import mpmath as mp

# Standardized utilities (ETHOS-compliant: no JSON reads inside)
from utils import default_json_out, write_json

# ------------------------ Defaults ------------------------
DEFAULT_BRACKET = (mp.mpf("1.2"), mp.mpf("1.5"))  # straddle quarter-FWHM root
ABS_TOL_DEFAULT = mp.mpf("1e-30")                 # tighter default now that we use mp
REL_TOL_DEFAULT = mp.mpf("1e-30")
AUTOBRK_EPS     = mp.mpf("1e-9")
AUTOBRK_STEPS   = 20000
DEFAULT_DPS     = 80                              # working precision (digits)

# Pretty decimal string (no scientific notation surprise from repr)
def dstr(x: mp.mpf, nd: int = 48) -> str:
    return mp.nstr(mp.mpf(x), nd)

# ------------------------ Fejér window & derivatives (mpmath) ------------------------
def K(x: mp.mpf) -> mp.mpf:
    """K(x) := (sin x / x)^2 with the standard x→0 extension."""
    if x == 0:
        return mp.mpf(1)
    s = mp.sin(x)
    return (s / x) * (s / x)

def fpp(x: mp.mpf) -> mp.mpf:
    """f″(x) = 2(csc^2 x − x^{−2})."""
    s = mp.sin(x)
    return 2 * (1 / (s * s) - 1 / (x * x))

def f3(x: mp.mpf) -> mp.mpf:
    """f‴(x) = −4 csc^2 x·cot x + 4 x^{−3}."""
    s, c = mp.sin(x), mp.cos(x)
    return -4 * (1 / (s * s)) * (c / s) + 4 / (x ** 3)

def f4(x: mp.mpf) -> mp.mpf:
    """f⁽⁴⁾(x) = 8 csc^2 x·cot^2 x + 4 csc^4 x − 12 x^{−4}."""
    s, c = mp.sin(x), mp.cos(x)
    csc2, cot = 1 / (s * s), c / s
    return 8 * csc2 * (cot * cot) + 4 * (csc2 * csc2) - 12 / (x ** 4)

# ------------------------ Root finding with rigorous bracket ------------------------
def g(x: mp.mpf, target: mp.mpf) -> mp.mpf:
    """Residual g(x) := K(x) − target."""
    return K(x) - target

def try_bisection(a: mp.mpf, b: mp.mpf, target: mp.mpf,
                  abs_tol: mp.mpf, rel_tol: mp.mpf,
                  max_iter: int = 512) -> Tuple[bool, Dict[str, Any]]:
    """Bisection on [a,b] for g(x)=K(x)−target. Returns (converged, payload)."""
    fa, fb = g(a, target), g(b, target)
    if fa * fb > 0:
        return False, {"reason": "Bracket does not straddle root", "fa": dstr(fa), "fb": dstr(fb)}
    left, right, f_left, f_right = a, b, fa, fb
    iters = 0
    while iters < max_iter:
        iters += 1
        mid = (left + right) / 2
        f_mid = g(mid, target)
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
        width = right - left
        xhat = (left + right) / 2
        if width < abs_tol + rel_tol * mp.fabs(xhat):
            return True, {
                "a": left, "b": right, "xhat": xhat,
                "K_mid": K(xhat), "residual": K(xhat) - target, "iters": iters
            }
    xhat = (left + right) / 2
    return False, {
        "a": left, "b": right, "xhat": xhat,
        "K_mid": K(xhat), "residual": K(xhat) - target,
        "iters": iters, "reason": "Max iterations reached"
    }

def auto_bracket(target: mp.mpf,
                 x_min: mp.mpf = AUTOBRK_EPS,
                 x_max: mp.mpf = mp.pi - AUTOBRK_EPS,
                 steps: int = AUTOBRK_STEPS) -> Tuple[mp.mpf, mp.mpf]:
    """Deterministic sign-scan to find [x_i, x_{i+1}] with g(x_i)·g(x_{i+1})<0."""
    dx = (x_max - x_min) / steps
    prev_x, prev_g = x_min, g(x_min, target)
    for i in range(1, steps + 1):
        x = x_min + i * dx
        gx = g(x, target)
        if prev_g * gx <= 0:
            return (prev_x, x)
        prev_x, prev_g = x, gx
    raise RuntimeError("auto_bracket failed to find a sign change on the scan domain.")

# ------------------------ Main ------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Quarter-FWHM solver & Fejér curvature (mpmath, rigorous bounds).")
    ap.add_argument("--a", type=str, default=str(DEFAULT_BRACKET[0]), help="Left bracket for x_q (mpf).")
    ap.add_argument("--b", type=str, default=str(DEFAULT_BRACKET[1]), help="Right bracket for x_q (mpf).")
    ap.add_argument("--abs-tol", type=str, default=str(ABS_TOL_DEFAULT), help="Absolute tolerance on x (mpf).")
    ap.add_argument("--rel-tol", type=str, default=str(REL_TOL_DEFAULT), help="Relative tolerance on x (mpf).")
    ap.add_argument("--dps", type=int, default=DEFAULT_DPS, help="mpmath working precision (digits).")
    ap.add_argument("--auto-bracket", action="store_true", help="Scan for a valid bracket if the provided one fails.")
    ap.add_argument("--lock-ratio", type=str, default="8/15",
        help="Lock ratio R with K(x_q)=R, as a fraction string like '8/15'.")
    args = ap.parse_args()

    # Set precision
    mp.mp.dps = max(32, int(args.dps))

    # Exact lock ratio (no float intermediates)
    frac = Fraction(args.lock_ratio.strip())   # exact
    target = mp.mpf(frac.numerator) / mp.mpf(frac.denominator)

    # Parse bracket and tolerances as mpf
    a0 = mp.mpf(args.a)
    b0 = mp.mpf(args.b)
    abs_tol = mp.mpf(args.abs_tol)
    rel_tol = mp.mpf(args.rel_tol)

    # Bisection
    converged, payload = try_bisection(a0, b0, target, abs_tol, rel_tol)
    if not converged and args.auto_bracket:
        ab = auto_bracket(target)
        converged, payload = try_bisection(ab[0], ab[1], target, abs_tol, rel_tol)

    if not converged:
        print("\n[warning] Bisection did not meet tolerance; reporting last bracket and midpoint.\n")

    left, right = payload["a"], payload["b"]
    xhat, iters = payload["xhat"], int(payload["iters"])
    width, half = right - left, (right - left) / 2

    # Derivatives at midpoint and conservative bounds via endpoints
    fpp_x, f3_x, f4_x = fpp(xhat), f3(xhat), f4(xhat)

    # compute endpoints once and bound by min/max (mp.mpf-safe)
    fpp_l, fpp_r = fpp(left), fpp(right)
    f3_l,  f3_r  = f3(left),  f3(right)
    f4_l,  f4_r  = f4(left),  f4(right)

    fpp_min, fpp_max = (fpp_l, fpp_r) if fpp_l <= fpp_r else (fpp_r, fpp_l)
    f3_min,  f3_max  = (f3_l,  f3_r)  if f3_l  <= f3_r  else (f3_r,  f3_l)
    f4_min,  f4_max  = (f4_l,  f4_r)  if f4_l  <= f4_r  else (f4_r,  f4_l)

    # Protocol-exact L0
    L0_exact = mp.log(mp.mpf(frac.denominator) / mp.mpf(frac.numerator))

    # Console summary (audit)
    print("\n=== Quarter-FWHM & Fejér curvature (rigorous bisection) ===")
    print(f"Bracket final       : [{dstr(left, 18)}, {dstr(right, 18)}]  (width={dstr(width, 3)}, Δx={dstr(half, 3)})")
    print(f"x_q (midpoint)      : {dstr(xhat, 18)}")
    print(f"K(x_hat)            : {dstr(payload['K_mid'], 18)}")
    print(f"K(x̂) residual      : {dstr(payload['residual'], 3)}   (target {frac.numerator}/{frac.denominator})")
    print(f"f''(x_q)            : {dstr(fpp_x, 18)}   interval ⊂ [{dstr(fpp_min, 18)}, {dstr(fpp_max, 18)}]")
    print(f"f'''(x_q)           : {dstr(f3_x, 18)}    interval ⊂ [{dstr(f3_min, 18)},  {dstr(f3_max, 18)}]")
    print(f"f''''(x_q)          : {dstr(f4_x, 18)}    interval ⊂ [{dstr(f4_min, 18)},  {dstr(f4_max, 18)}]")
    print(f"Converged           : {converged}   iters={iters}\n")

    # JSON result (emit high-precision decimal strings; no Python floats)
    out = {
        "meta": {
            "schema_version": "1.1",
            "script": Path(__file__).name,
            "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
            "mpmath": {"dps": mp.mp.dps},
        },
        "inputs": {
            "target": {
                "rational": f"{frac.numerator}/{frac.denominator}",
                "decimal_48": dstr(target, 48),
                "desc": "Protocol lock: quarter-FWHM K(x_q) = 8/15."
            },
            "bracket": {"a": dstr(a0, 32), "b": dstr(b0, 32)},
            "abs_tol": {"decimal_48": dstr(abs_tol, 48)},
            "rel_tol": {"decimal_48": dstr(rel_tol, 48)},
            "auto_bracket": {"value": bool(args.auto_bracket)},
        },
        "intermediates": {
            "bracket_final": {"a": dstr(left, 48), "b": dstr(right, 48)},
            "halfwidth": {"decimal_48": dstr(half, 48)},
            "iters": iters,
            "K_mid": {"decimal_48": dstr(payload["K_mid"], 48)},
            "residual": {"decimal_48": dstr(payload["residual"], 48)},
        },
        "outputs": {
            "x_q": {
                "decimal_48": dstr(xhat, 48),
                "precision": {
                    "interval": [dstr(left, 48), dstr(right, 48)],
                    "abs_err_bound": dstr(half, 48),
                    "method": "Bisection midpoint ± half-width (rigorous).",
                },
            },
            "L0": {
                "exact_log_15_over_8": True,
                "decimal_48": dstr(L0_exact, 48),
                "desc": "Protocol-exact L0 = log(15/8).",
            },
            "fpp": {
                "decimal_48": dstr(fpp_x, 48),
                "bound_from_bracket": {"min": dstr(fpp_min, 48), "max": dstr(fpp_max, 48)},
            },
            "f3": {
                "decimal_48": dstr(f3_x, 48),
                "bound_from_bracket": {"min": dstr(f3_min, 48), "max": dstr(f3_max, 48)},
            },
            "f4": {
                "decimal_48": dstr(f4_x, 48),
                "bound_from_bracket": {"min": dstr(f4_min, 48), "max": dstr(f4_max, 48)},
            },
        },
        "status": {"converged": bool(converged), **({"reason": payload.get("reason")} if not converged else {})},
    }

    out_path = default_json_out(None, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
