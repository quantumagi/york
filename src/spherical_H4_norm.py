#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spherical_H4_norm.py

S2: Closed-form derivation and numerical validation of
    ⟨ H4(u)^2 ⟩_{u∈S^2} = 16/525 exactly,
where H4(u) = Σ_i u_i^4 − 3/5 and u is uniform on the unit sphere S^2.

Derivation (documented in code comments):
- For u ∈ S^{d-1}, the vector of squares x_i = u_i^2 is Dirichlet(1/2,...,1/2) in d dims.
- With d=3, α_i = 1/2 and α0 = 3/2.
- Using Dirichlet moments:
    E[Π x_i^{m_i}] = Γ(α0)/Γ(α0+Σ m_i) Π Γ(α_i + m_i)/Γ(α_i).
- Map moments:
    E[u_i^4]        = E[x_i^2]           = 1/5
    E[u_i^8]        = E[x_i^4]           = 1/9
    E[u_i^4 u_j^4]  = E[x_i^2 x_j^2]     = 1/105  (i≠j)
- Expand H4^2:
    H4^2 = Σ u_i^8 + 2 Σ_{i<j} u_i^4 u_j^4 − (6/5) Σ u_i^4 + 9/25
- Take expectations:
    ⟨H4^2⟩ = 3*(1/9) + 2*3*(1/105) − (6/5)*3*(1/5) + 9/25 = 16/525.

This script:
- Emits the exact fraction and float,
- Runs a Monte Carlo check (default 2e6 samples fast-path configurable),
- Writes a JSON for audit if requested.

Usage:
  python src/spherical_H4_norm.py --n 2000000 --seed 0 --eps 3e-3 --json-out outputs/spherical_H4_norm.json
"""

from __future__ import annotations
import argparse
import json
import math
import os
from fractions import Fraction
from typing import Dict, Any
from pathlib import Path

import numpy as np

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

def H4(u: np.ndarray) -> float:
    """H4(u) = sum(u_i^4) - 3/5 for u ∈ R^3, ||u||=1."""
    s4 = float(np.sum(u ** 4))
    return s4 - 3.0 / 5.0


def sample_uniform_s2(n: int, seed: int = 0) -> np.ndarray:
    """
    Sample n points uniformly on S^2 by normalizing i.i.d. N(0,1)^3 vectors.
    Deterministic RNG with given seed.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 3))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    # Avoid division by zero (probability 0, but keep code robust)
    norms[norms == 0.0] = 1.0
    return z / norms


def analytic_H4_sq() -> Fraction:
    """
    Return the exact rational value of ⟨H4^2⟩ on S^2: 16/525.
    Also provides a step-by-step reconstruction using fractions.
    """
    # Moments as exact rationals:
    E_ui4 = Fraction(1, 5)     # E[u_i^4]
    E_ui8 = Fraction(1, 9)     # E[u_i^8]
    E_ui4uj4 = Fraction(1, 105)  # E[u_i^4 u_j^4], i≠j

    term1 = 3 * E_ui8                  # Σ_i E[u_i^8] = 3 * 1/9
    term2 = 2 * (3 * E_ui4uj4)         # 2 * Σ_{i<j} E[u_i^4 u_j^4] = 2 * 3 * 1/105
    term3 = Fraction(-6, 5) * (3 * E_ui4)  # −(6/5) * Σ_i E[u_i^4] = −(6/5) * 3 * 1/5
    term4 = Fraction(9, 25)            # + 9/25

    total = term1 + term2 + term3 + term4
    return total  # Fraction(16, 525)


def monte_carlo_H4_sq(n: int, seed: int = 0) -> Dict[str, float]:
    """
    Monte Carlo estimate of ⟨H4^2⟩ with n samples.
    Returns mean and (unbiased) standard error.
    """
    U = sample_uniform_s2(n, seed)
    H = np.sum(U ** 4, axis=1) - 3.0 / 5.0
    vals = H * H
    mean = float(np.mean(vals))
    # Unbiased std error of the mean:
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 0 else float("nan")
    return {"mean": mean, "std": std, "sem": sem}


def main() -> None:
    ap = argparse.ArgumentParser(description="S2: ⟨H4^2⟩ on S^2 (closed form + MC check).")
    ap.add_argument("--n", type=int, default=500000, help="Monte Carlo sample size.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--eps", type=float, default=3e-3,
                    help="Acceptance tolerance for |MC - exact| (absolute).")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Optional path to write a JSON result.")
    args = ap.parse_args()

    exact_frac = analytic_H4_sq()
    exact_float = float(exact_frac)

    mc = monte_carlo_H4_sq(args.n, args.seed)
    err = abs(mc["mean"] - exact_float)
    passed = bool(err <= args.eps)

    print("\n=== S2: ⟨H4^2⟩ on S^2 ===")
    print(f"Exact (fraction)   : {exact_frac}  (~ {exact_float:.12f})")
    print(f"Monte Carlo n      : {args.n}  seed={args.seed}")
    print(f"MC mean ± SEM      : {mc['mean']:.12f} ± {mc['sem']:.2e}")
    print(f"|MC - exact|       : {err:.3e}  (tol={args.eps:.2e})  -> PASS={passed}\n")

    res = {
        "exact_fraction": f"{exact_frac.numerator}/{exact_frac.denominator}",
        "exact_float": exact_float,
        "mc_mean": mc["mean"],
        "mc_sem": mc["sem"],
        "n_samples": int(args.n),
        "seed": int(args.seed),
        "abs_error": err,
        "tolerance": float(args.eps),
        "passed": passed,
    }

    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
