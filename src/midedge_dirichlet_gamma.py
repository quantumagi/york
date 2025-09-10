#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
midedge_dirichlet_gamma.py

S4: Mid-edge / Dirichlet sampling amplitude Γ and ΔL^(2) check.

We assemble the exact algebraic factors (all dimensionless):
  - A1g projection coefficient:        C_proj   = 21/64
  - Sampling parity factor:            ζ_parity = 8/105
  - Quarter-hop Jacobian (squared):    ζ_J2     = 4/3
  - Dirichlet/Fejér main-lobe factor:  ζ_dir    = 8/25
  => ζ_samp = ζ_parity * ζ_J2 * ζ_dir = 256/7875
  => Γ = ζ_samp * C_proj = 4/375

Then verify the quadratic width term:
  ΔL^(2) = 0.5 * f''(x_q) * ⟨H4^2⟩ * (χ_TT)^2 * Γ^2
with:
  - x_q solves K(x) = (sin x / x)^2 = 8/15  (quarter-FWHM),
  - f''(x) = 2(csc^2 x - x^{-2}),
  - ⟨H4^2⟩ = 16/525 (exact),
  - χ_TT = 11/260.

Outputs:
  - All exact fractions and their floats,
  - Numerically evaluated f''(x_q), Γ, and ΔL^(2),
  - Optional JSON.

Usage:
  python src/midedge_dirichlet_gamma.py --abs-tol 1e-15 --rel-tol 1e-15 \
    --json-out outputs/gamma_and_dl2.json
"""

from __future__ import annotations
import argparse
import json
import math
import os
from fractions import Fraction
from typing import Dict, Any
from pathlib import Path

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

# --- Fejér window and curvature ---
def K(x: float) -> float:
    """K(x) = (sin x / x)^2."""
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
    """Solve K(x) = 8/15 on [a,b] via bisection."""
    target = 8.0 / 15.0
    fa = K(a) - target
    fb = K(b) - target
    if fa * fb > 0:
        raise ValueError(f"Bracket [{a},{b}] does not straddle the root.")

    left, right = a, b
    f_left, f_right = fa, fb

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = K(mid) - target
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid

        width = right - left
        x = 0.5 * (left + right)
        if width < abs_tol + rel_tol * abs(x):
            return {"x_q": float(x), "Kx": float(K(x)), "residual": float(K(x) - target)}

    x = 0.5 * (left + right)
    return {"x_q": float(x), "Kx": float(K(x)), "residual": float(K(x) - target)}


def main() -> None:
    ap = argparse.ArgumentParser(description="S4: Mid-edge/Dirichlet Γ and ΔL^(2) verification.")
    ap.add_argument("--a", type=float, default=1.2, help="Left bracket for x_q.")
    ap.add_argument("--b", type=float, default=1.5, help="Right bracket for x_q.")
    ap.add_argument("--abs-tol", type=float, default=1e-15, help="Absolute tolerance on x.")
    ap.add_argument("--rel-tol", type=float, default=1e-15, help="Relative tolerance on x.")
    ap.add_argument("--json-out", type=str, default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    # Exact fractions for the algebraic factors:
    C_proj = Fraction(21, 64)
    z_parity = Fraction(8, 105)
    z_J2 = Fraction(4, 3)
    z_dir = Fraction(8, 25)
    z_samp = z_parity * z_J2 * z_dir           # 256/7875
    Gamma = z_samp * C_proj                    # 4/375

    # Exact inputs for ΔL^(2) besides f''(x_q):
    H4_sq = Fraction(16, 525)
    chi_TT = Fraction(11, 260)

    # Quarter-FWHM and curvature
    sol = solve_quarter_fwhm(a=args.a, b=args.b, abs_tol=args.abs_tol, rel_tol=args.rel_tol)
    x_q = float(sol["x_q"])
    Kx = float(sol["Kx"])
    residual = float(sol["residual"])
    fpp_xq = fpp(x_q)

    # ΔL^(2) numeric evaluation
    # ΔL^(2) = 0.5 * f''(x_q) * ⟨H4^2⟩ * χ_TT^2 * Γ^2
    dl2 = 0.5 * fpp_xq * float(H4_sq) * float(chi_TT * chi_TT) * float(Gamma * Gamma)

    # Expected reference (for display): ~3.071538094072598e-9
    print("\n=== S4: Mid-edge / Dirichlet Γ and ΔL^(2) ===")
    print(f"C_proj (exact)      : {C_proj} (~ {float(C_proj):.12f})")
    print(f"ζ_parity (exact)    : {z_parity} (~ {float(z_parity):.12f})")
    print(f"ζ_J2 (exact)        : {z_J2} (~ {float(z_J2):.12f})")
    print(f"ζ_dir (exact)       : {z_dir} (~ {float(z_dir):.12f})")
    print(f"ζ_samp (exact)      : {z_samp} (~ {float(z_samp):.12f})")
    print(f"Γ (exact)           : {Gamma} (~ {float(Gamma):.12f})")
    print(f"x_q                 : {x_q:.16f}")
    print(f"K(x_q)              : {Kx:.16f}  (target 8/15 ≈ {8/15:.16f}, residual={residual:.3e})")
    print(f"f''(x_q)            : {fpp_xq:.16f}")
    print(f"⟨H4^2⟩ (exact)       : {H4_sq} (~ {float(H4_sq):.12f})")
    print(f"χ_TT (exact)        : {chi_TT} (~ {float(chi_TT):.12f})")
    print(f"ΔL^(2) (numeric)    : {dl2:.18e}\n")

    res = {
        "C_proj_fraction": f"{C_proj.numerator}/{C_proj.denominator}",
        "z_parity_fraction": f"{z_parity.numerator}/{z_parity.denominator}",
        "z_J2_fraction": f"{z_J2.numerator}/{z_J2.denominator}",
        "z_dir_fraction": f"{z_dir.numerator}/{z_dir.denominator}",
        "z_samp_fraction": f"{z_samp.numerator}/{z_samp.denominator}",
        "Gamma_fraction": f"{Gamma.numerator}/{Gamma.denominator}",
        "C_proj": float(C_proj),
        "z_parity": float(z_parity),
        "z_J2": float(z_J2),
        "z_dir": float(z_dir),
        "z_samp": float(z_samp),
        "Gamma": float(Gamma),
        "x_q": x_q,
        "Kx": Kx,
        "residual": residual,
        "fpp_xq": fpp_xq,
        "H4_sq": float(H4_sq),
        "chi_TT": float(chi_TT),
        "delta_L2": dl2,
        "abs_tol": float(args.abs_tol),
        "rel_tol": float(args.rel_tol),
        "bracket": [float(args.a), float(args.b)],
    }

    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
