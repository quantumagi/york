#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
york_constant_axis.py

S1: York–quartic contraction constant, verified two ways:
  (i) axis evaluation (u = e1),
  (ii) random unit vectors (irrep-agnostic spot checks).
"""

from __future__ import annotations
import argparse
import json
import math
import os
from typing import Dict, Any
from pathlib import Path

import numpy as np

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

def projector_pi(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(3)
    I = np.eye(3)
    return I - np.outer(u, u)

def projector_tt(u: np.ndarray) -> np.ndarray:
    Pi = projector_pi(u)
    P = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    P[i, j, k, l] = 0.5 * (Pi[i, k] * Pi[j, l] + Pi[i, l] * Pi[j, k]) - 0.5 * Pi[i, j] * Pi[k, l]
    return P


def qhat_tensor() -> np.ndarray:
    Q = np.zeros((3, 3, 3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    s = 0.0
                    for a in range(3):
                        s += 1.0 if (i == a and j == a and k == a and l == a) else 0.0
                    s -= (1.0 / 5.0) * (
                        (1.0 if i == j else 0.0) * (1.0 if k == l else 0.0)
                        + (1.0 if i == k else 0.0) * (1.0 if j == l else 0.0)
                        + (1.0 if i == l else 0.0) * (1.0 if j == k else 0.0)
                    )
                    Q[i, j, k, l] = s
    return Q


def contract4(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.tensordot(A, B, axes=([0, 1, 2, 3], [0, 1, 2, 3])))


def H4(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    return float(np.sum(u ** 4) - 3.0 / 5.0)


def random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n == 0:
        return random_unit_vec(rng)
    return v / n


def check_constant(n_random: int = 1000, seed: int = 0, tol: float = 1e-12) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Qhat = qhat_tensor()

    # Axis evaluation (u = e1)
    u_axis = np.array([1.0, 0.0, 0.0])
    P_axis = projector_tt(u_axis)
    h4_axis = H4(u_axis)
    ratio_axis = contract4(P_axis, Qhat) / h4_axis  # should be 0.5 exactly
    err_axis = abs(ratio_axis - 0.5)

    # Random checks
    ratios = []
    max_err = 0.0
    for _ in range(n_random):
        u = random_unit_vec(rng)
        P = projector_tt(u)
        h4 = H4(u)
        ratio = contract4(P, Qhat) / h4  # should be 0.5 for all u
        ratios.append(ratio)
        max_err = max(max_err, abs(ratio - 0.5))

    mean_ratio = float(np.mean(ratios))
    std_ratio = float(np.std(ratios))

    # Scaled constant for Q = (11/390) Q̂ ⇒ (11/780)
    const_Q = (11.0 / 390.0) * 0.5
    target_Q = 11.0 / 780.0
    err_Q = abs(const_Q - target_Q)

    # --- Cast to built-in bools to keep json.dump happy ---
    status_axis = bool(err_axis <= tol)
    status_rand = bool(max_err <= tol)
    status_scaled = bool(err_Q <= float(np.finfo(float).eps))  # <- cast here

    return {
        "seed": int(seed),
        "n_random": int(n_random),
        "tol": float(tol),
        "ratio_axis": float(ratio_axis),
        "axis_error": float(err_axis),
        "ratio_mean": float(mean_ratio),
        "ratio_std": float(std_ratio),
        "ratio_max_error": float(max_err),
        "passed_axis": bool(status_axis),
        "passed_random": bool(status_rand),
        "constant_for_Q": float(const_Q),
        "target_constant_for_Q": float(target_Q),
        "passed_scaled": bool(status_scaled),
        "all_passed": bool(status_axis and status_rand and status_scaled),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="S1: York–quartic constant via axis & random checks.")
    ap.add_argument("--n-random", type=int, default=1000, help="Number of random unit vectors.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    ap.add_argument("--tol", type=float, default=1e-12, help="Absolute tolerance for checks.")
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to write a JSON result.")
    args = ap.parse_args()

    res = check_constant(n_random=args.n_random, seed=args.seed, tol=args.tol)

    print("\n=== York–Quartic Constant Check (S1) ===")
    print(f"Seed               : {res['seed']}")
    print(f"Random samples     : {res['n_random']}")
    print(f"Tolerance          : {res['tol']:.2e}")
    print(f"Axis ratio         : {res['ratio_axis']:.16f}  (err={res['axis_error']:.2e})  -> PASS={res['passed_axis']}")
    print(f"Random mean ratio  : {res['ratio_mean']:.16f}  (std={res['ratio_std']:.2e}, max_err={res['ratio_max_error']:.2e})  -> PASS={res['passed_random']}")
    print(f"Constant for Q     : {res['constant_for_Q']:.16f}  (target 11/780 ≈ {res['target_constant_for_Q']:.16f})  -> PASS={res['passed_scaled']}")
    print(f"ALL PASSED         : {res['all_passed']}")
    print("Note: ratios are for (P^{TT}:Q̂)/H4(u); expected constant is 0.5 exactly.")
    print("      Scaling by (11/390) gives (P^{TT}:Q)/H4(u) = 11/780 as used in the paper.\n")

    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")


if __name__ == "__main__":
    main()
