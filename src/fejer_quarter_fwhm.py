#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fejer_quarter_fwhm.py

S3: Quarter-FWHM solver and Fejér curvature at the lock point.

Definitions:
  K(x) = (sin x / x)^2  (Fejér main-lobe power window)
  x_q  solves K(x_q) = 8/15  (quarter-FWHM condition)
  f''(x) = 2(csc^2 x - x^{-2})
  L0 = -log K(x_q) = log(15/8)

Outputs:
  - x_q (high-precision via bisection)
  - K(x_q) and residual to 8/15
  - fpp = f''(x_q)
  - L0 (numeric) and difference to log(15/8)
  - Optional JSON

Usage:
  python src/fejer_quarter_fwhm.py --abs-tol 1e-15 --rel-tol 1e-15 \
    --json-out outputs/fejer_qfwhm.json
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, Any
from pathlib import Path

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

def K(x: float) -> float:
    """K(x) = (sin x / x)^2 with a safe x→0 limit (not needed here, but robust)."""
    if x == 0.0:
        return 1.0
    s = math.sin(x)
    return (s / x) * (s / x)


def fpp(x: float) -> float:
    """f''(x) = 2(csc^2 x - x^{-2})."""
    s = math.sin(x)
    return 2.0 * ((1.0 / (s * s)) - (1.0 / (x * x)))


def solve_quarter_fwhm(a: float = 1.2, b: float = 1.5,
                       abs_tol: float = 1e-15, rel_tol: float = 1e-15,
                       max_iter: int = 200) -> Dict[str, float]:
    """
    Solve K(x) = 8/15 on [a,b] via bisection with tight tolerances.
    Returns x_q and residual.
    """
    target = 8.0 / 15.0

    fa = K(a) - target
    fb = K(b) - target
    if fa * fb > 0:
        raise ValueError(f"Bracket [{a},{b}] does not straddle the root: "
                         f"K(a)-tgt={fa}, K(b)-tgt={fb}")

    left, right = a, b
    f_left, f_right = fa, fb

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = K(mid) - target

        # Bisection step
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid

        # Convergence checks
        width = right - left
        x = 0.5 * (left + right)
        if width < abs_tol + rel_tol * abs(x):
            return {"x_q": float(x), "Kx": float(K(x)), "residual": float(K(x) - target)}

    # If not converged, return best midpoint anyway
    x = 0.5 * (left + right)
    return {"x_q": float(x), "Kx": float(K(x)), "residual": float(K(x) - target)}


def main() -> None:
    ap = argparse.ArgumentParser(description="S3: Quarter-FWHM and Fejér curvature.")
    ap.add_argument("--a", type=float, default=1.2, help="Left bracket for x_q.")
    ap.add_argument("--b", type=float, default=1.5, help="Right bracket for x_q.")
    ap.add_argument("--abs-tol", type=float, default=1e-15, help="Absolute tolerance on x.")
    ap.add_argument("--rel-tol", type=float, default=1e-15, help="Relative tolerance on x.")
    ap.add_argument("--json-out", type=str, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    sol = solve_quarter_fwhm(a=args.a, b=args.b,
                             abs_tol=args.abs_tol, rel_tol=args.rel_tol)

    x_q = float(sol["x_q"])
    Kx = float(sol["Kx"])
    res = float(sol["residual"])

    target = 8.0 / 15.0
    L0 = -math.log(Kx)
    L0_expected = math.log(15.0 / 8.0)
    dL0 = L0 - L0_expected

    fpp_xq = fpp(x_q)

    # Pretty print
    print("\n=== S3: Quarter-FWHM & Fejér curvature ===")
    print(f"x_q                  : {x_q:.16f}")
    print(f"K(x_q)               : {Kx:.16f}")
    print(f"Target 8/15          : {target:.16f}")
    print(f"Residual             : {res:.3e}")
    print(f"L0 = -log K(x_q)     : {L0:.16f}")
    print(f"log(15/8)            : {L0_expected:.16f}")
    print(f"L0 difference        : {dL0:.3e}")
    print(f"f''(x_q)             : {fpp_xq:.16f}\n")

    out = {
        "x_q": x_q,
        "Kx": Kx,
        "target": target,
        "residual": res,
        "L0": L0,
        "log_15_over_8": L0_expected,
        "L0_diff": dL0,
        "fpp_xq": fpp_xq,
        "abs_tol": float(args.abs_tol),
        "rel_tol": float(args.rel_tol),
        "bracket": [float(args.a), float(args.b)],
    }

    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")


if __name__ == "__main__":
    main()

