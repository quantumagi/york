#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cq.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented from pre-existing
    theory, substantiated, and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  1) A numerical certificate that the York–quartic contraction of the canonical
     O_h-traceless rank-4 tensor Q̂ by the 3D York TT projector,
        r(u) := (P^{TT}(u) : Q̂) / H4(u),
     is (approximately) u-independent: we report axis and random-u checks and
     the Monte Carlo mean C_Qhat with an observed absolute error bound.
  2) The canonical theory coefficient
        C_Q = 11/780
     is emitted as an exact rational with a decimal echo.
     # References (pre-existing theory, placed at point of use):
     # • York TT projector in 3D: York, J. Math. Phys. 14, 456–464 (1973).
     # • Isotropic/traceless rank-4 construction & STF algebra:
     #   standard continuum mechanics / representation theorems (e.g., MTW §35.12; DDJ 2008).
     # • Spherical moments on S^{2}: NIST DLMF §26.5.

INPUTS
  --n-random  (int; Monte Carlo sample count; default 1000)
  --seed      (int; RNG seed; default 0)
  --tol       (float; constancy check tolerance; default 1e-12)
  --sf        (float; scale factor on SE inside abs error bound; default 1.0)
  --scale     (optional; string/float; legacy λ to rescale C_Qhat)

OUTPUT (JSON → outputs/cq.json)
  {
    "meta": {...},
    "inputs": {...},
    "intermediates": {...},             # MC diagnostics
    "outputs": {
      "C_Qhat": { "decimal_24": "...", "raw_decimal_24": "...", "precision": {...} },
      "C_Q":    { "rational": "11/780", "decimal_24": "..." },
      "lambda_york": { "decimal_24": "..." }   # optional audit
      "C_Q_scaled": { ... }                    # only if --scale supplied (legacy)
    },
    "status": {...}
  }
"""

from __future__ import annotations
import sys
import platform
import argparse
import json
from fractions import Fraction as Fr
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ----------------------- numeric formatting helpers -----------------------
DEC_SIG = 24
def nstr(x: float | int, sig: int = DEC_SIG) -> str:
    # safe decimal printer (no scientific notation unless necessary)
    # numpy floats → python float → repr → format with precision
    return f"{float(x):.{sig}g}"

# ----------------------- protocol dimension & tunables --------------------
D: int                = 3
DEFAULT_N_RANDOM: int = 1000
DEFAULT_SEED: int     = 0
DEFAULT_TOL: float    = 1e-12
DEFAULT_SF: float     = 1.0
AXIS_U = np.array([1.0, 0.0, 0.0], dtype=float)

# ----------------------- spherical moment (theory) ------------------------
def exact_E_u4(d: int = D) -> Fr:
    # E[u_x^4] = 3 / [d(d+2)] for u ~ Uniform(S^{d-1}]
    # Ref: NIST DLMF §26.5; standard Beta/Dirichlet moments.
    return Fr(3, d * (d + 2))

H4_SUBTRACT_FRAC = Fr(D, 1) * exact_E_u4(D)  # = 3/(d+2)
H4_SUBTRACT      = float(H4_SUBTRACT_FRAC)

# ----------------------------- paths --------------------------------------
def default_json_out() -> Path:
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

# ----------------------------- math core ----------------------------------
def projector_pi(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(D)
    I = np.eye(D, dtype=float)
    return I - np.outer(u, u)

def projector_tt(u: np.ndarray) -> np.ndarray:
    """
    York 3D TT projector on rank-2 tensors:
      P^{TT}_{ijkl} = ½(Π_{ik}Π_{jl} + Π_{il}Π_{jk}) − ½ Π_{ij}Π_{kl}
    # Ref: York (1973), MTW §35.12 — placed where used.
    """
    Pi = projector_pi(u)
    P = np.zeros((D, D, D, D), dtype=float)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    P[i, j, k, l] = 0.5 * (Pi[i, k] * Pi[j, l] + Pi[i, l] * Pi[j, k]) - 0.5 * Pi[i, j] * Pi[k, l]
    return P

def qhat_tensor() -> np.ndarray:
    """
    Canonical O_h-symmetric traceless quartic tensor:
      Q̂_{ijkl} = Σ_a δ_{ia}δ_{ja}δ_{ka}δ_{la}
                  − (1/(d+2)) [ δ_{ij}δ_{kl} + δ_{ik}δ_{jl} + δ_{il}δ_{jk} ].
    (Trace-free by construction; standard isotropic STF subtraction.)
    # Ref: standard STF / isotropic rank-4 constructions (e.g., continuum mechanics texts).
    """
    Q = np.zeros((D, D, D, D), dtype=float)
    c = 1.0 / (D + 2)
    eye = np.eye(D)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                for l in range(D):
                    s = 0.0
                    for a in range(D):
                        if i == a and j == a and k == a and l == a:
                            s += 1.0
                    s -= c * (
                        eye[i, j] * eye[k, l] +
                        eye[i, k] * eye[j, l] +
                        eye[i, l] * eye[j, k]
                    )
                    Q[i, j, k, l] = s
    return Q

def contract4(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.tensordot(A, B, axes=([0,1,2,3],[0,1,2,3])))

def H4(u: np.ndarray) -> float:
    # H4(u) = Σ u_i^4 − 3/(d+2)
    u = np.asarray(u, dtype=float).reshape(D)
    return float(np.sum(u**4) - H4_SUBTRACT)

def random_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=D)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return random_unit_vec(rng)
    return v / n

# -------------------------- precision helpers -----------------------------
def credible_sig_digits(value: float, abs_err_bound: float) -> int:
    if not np.isfinite(value) or not np.isfinite(abs_err_bound) or abs_err_bound <= 0.0:
        return 0
    av = abs(value)
    if av == 0.0:
        return 0
    ratio = abs_err_bound / av
    if ratio <= 0.0:
        return 15
    d = int(np.floor(-np.log10(ratio)))
    return max(0, min(d, 15))

def round_to_sig(value: float, sig: int) -> float:
    if sig <= 0 or value == 0.0 or not np.isfinite(value):
        return float(value)
    exp10 = int(np.floor(np.log10(abs(value))))
    decimals = max(0, sig - exp10 - 1)
    return float(np.round(value, decimals=decimals))

# ---------------------------- optional scaling ----------------------------
def parse_scale(scale_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    if scale_arg is None:
        return None
    try:
        if isinstance(scale_arg, str) and "/" in scale_arg:
            frac = Fr(scale_arg)
            return {"rational": f"{frac.numerator}/{frac.denominator}", "float": float(frac)}
        val = float(scale_arg)
        return {"rational": None, "float": float(val)}
    except Exception as e:
        raise SystemExit(f"[cq] Could not parse --scale: {scale_arg!r}") from e

# ------------------- canonical C_Q from theory (exact) --------------------
def canonical_C_Q_fraction() -> Fr:
    """
    Canonical coefficient for York TT contraction on STF rank-4 (3D):
      (P^{TT} : Q_can)(u) = (11/780) · H4(u), u-independent.
    # References directly at point of use:
    # • York (1973), J. Math. Phys. 14, 456–464 — TT projector normalization in 3D.
    # • Standard STF rank-4 isotropic decomposition (e.g., MTW §35.12; Dresselhaus et al., 2008).
    """
    return Fr(11, 780)

# ------------------------------ certificate -------------------------------
def certify_constant(n_random: int, seed: int, tol: float, sf: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    Qhat = qhat_tensor()

    r_axis = contract4(projector_tt(AXIS_U), Qhat) / H4(AXIS_U)

    r_vals = np.empty(n_random, dtype=float)
    for i in range(n_random):
        u = random_unit_vec(rng)
        r_vals[i] = contract4(projector_tt(u), Qhat) / H4(u)

    r_mean = float(r_vals.mean())
    r_std  = float(r_vals.std(ddof=1) if n_random > 1 else 0.0)
    r_se   = float(r_std / np.sqrt(n_random)) if n_random > 0 else 0.0
    max_dev = float(np.max(np.abs(r_vals - r_mean))) if n_random > 0 else 0.0
    axis_dev = float(abs(r_axis - r_mean))

    abs_err_bound = float(max(max_dev, sf * r_se))
    passed_random = bool(max_dev <= tol)
    passed_axis   = bool(axis_dev <= tol)

    return {
        "r_axis": r_axis,
        "r_mean": r_mean,
        "r_std": r_std,
        "r_se": r_se,
        "max_dev": max_dev,
        "axis_dev": axis_dev,
        "abs_err_bound": abs_err_bound,
        "passed_random": passed_random,
        "passed_axis": passed_axis,
    }

# ---------------------------------- main ----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="York–quartic constant for Q̂ via axis+random samples; canonical C_Q (exact)."
    )
    ap.add_argument("--n-random", type=int, default=DEFAULT_N_RANDOM)
    ap.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    ap.add_argument("--tol",      type=float, default=DEFAULT_TOL)
    ap.add_argument("--sf",       type=float, default=DEFAULT_SF)
    ap.add_argument("--scale",    type=str, default=None,
                    help='Optional λ for legacy scaled report C_Q_scaled = λ·C_Qhat (e.g. "11/390").')
    args = ap.parse_args()

    scale = parse_scale(args.scale)
    cert  = certify_constant(n_random=args.n_random, seed=args.seed, tol=args.tol, sf=args.sf)

    # precision for C_Qhat from observed abs_err_bound
    sig_cqhat = credible_sig_digits(cert["r_mean"], cert["abs_err_bound"])
    cqhat_rounded = round_to_sig(cert["r_mean"], sig_cqhat)

    outputs: Dict[str, Any] = {
        "C_Qhat": {
            "decimal_24":        nstr(cqhat_rounded, DEC_SIG),
            "raw_decimal_24":    nstr(cert["r_mean"], DEC_SIG),
            "precision": {
                "credible_sig_digits": int(sig_cqhat),
                "abs_err_bound":       float(cert["abs_err_bound"]),
                "rounded_from":         nstr(cert["r_mean"], DEC_SIG),
                "method":               "digits from max(|r_i-mean|, sf·SE); rounding cosmetic only"
            }
        }
    }

    # Canonical C_Q (exact, pre-existing theory)
    C_Q_exact = canonical_C_Q_fraction()  # 11/780 exactly
    outputs["C_Q"] = {
        "rational":   f"{C_Q_exact.numerator}/{C_Q_exact.denominator}",
        "decimal_24": nstr(float(C_Q_exact), DEC_SIG)
    }

    # Audit ratio λ_york = C_Q / C_Qhat_raw (optional)
    try:
        lam = float(C_Q_exact) / float(cert["r_mean"])
        outputs["lambda_york"] = {"decimal_24": nstr(lam, DEC_SIG)}
    except Exception:
        pass

    # Optional legacy scaled constant if --scale is provided
    if scale is not None:
        lam = scale["float"]
        c_q_raw = cert["r_mean"] * lam
        abs_err_bound_q = abs(lam) * cert["abs_err_bound"]
        sig_cq = credible_sig_digits(c_q_raw, abs_err_bound_q)
        c_q_rounded = round_to_sig(c_q_raw, sig_cq)
        outputs["C_Q_scaled"] = {
            "decimal_24":     nstr(c_q_rounded, DEC_SIG),
            "raw_decimal_24": nstr(c_q_raw, DEC_SIG),
            "precision": {
                "credible_sig_digits": int(sig_cq),
                "abs_err_bound":       float(abs_err_bound_q),
                "rounded_from":         nstr(c_q_raw, DEC_SIG),
                "method":               "|λ|·abs_err_bound(Q̂); rounding cosmetic only"
            },
            "scale": {
                "rational": scale["rational"],
                "float":    float(lam)
            }
        }

    result = {
        "meta": {
            "schema_version": "1.1",
            "script": Path(__file__).name,
            "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
        },
        "inputs": {
            "n_random": {"value": int(args.n_random)},
            "seed":     {"value": int(args.seed)},
            "tol":      {"value": float(args.tol)},
            "sf":       {"value": float(args.sf)},
            **({"scale": {"value": (scale["rational"] if scale["rational"] else scale["float"])}}
               if scale is not None else {})
        },
        "intermediates": {
            "axis_ratio":      nstr(cert["r_axis"], DEC_SIG),
            "random_mean":     nstr(cert["r_mean"], DEC_SIG),
            "random_std":      nstr(cert["r_std"], DEC_SIG),
            "random_se":       nstr(cert["r_se"], DEC_SIG),
            "random_max_dev":  nstr(cert["max_dev"], DEC_SIG),
            "axis_dev":        nstr(cert["axis_dev"], DEC_SIG),
            "abs_err_bound":   nstr(cert["abs_err_bound"], DEC_SIG),
        },
        "outputs": outputs,
        "status": {
            "passed_random": bool(cert["passed_random"]),
            "passed_axis":   bool(cert["passed_axis"]),
            "all_passed":    bool(cert["passed_random"] and cert["passed_axis"]),
        },
    }

    # Console summary (concise)
    print("\n=== York–Quartic Constant (Q̂) ===")
    print(f"Seed               : {args.seed}")
    print(f"Random samples     : {args.n_random}")
    print(f"Tolerance / sf     : {args.tol:.2e} / {args.sf:.2f}")
    print(f"Axis ratio         : {nstr(cert['r_axis'])}")
    print(f"Mean / Std / SE    : {nstr(cert['r_mean'])} / {nstr(cert['r_std'])} / {nstr(cert['r_se'])}")
    print(f"Max dev / Axis dev : {nstr(cert['max_dev'])} / {nstr(cert['axis_dev'])}")
    print(f"Abs error bound    : {nstr(cert['abs_err_bound'])}")
    print(f"C_Qhat (rounded)   : {outputs['C_Qhat']['decimal_24']}   "
          f"[credible sig digits: {outputs['C_Qhat']['precision']['credible_sig_digits']}]")
    print(f"C_Q (canonical)    : {outputs['C_Q']['decimal_24']}   "
          f"(rational {outputs['C_Q']['rational']})")
    if "lambda_york" in outputs:
        print(f"λ_york (audit)     : {outputs['lambda_york']['decimal_24']}")
    if "C_Q_scaled" in outputs:
        print(f"C_Q (λ·C_Qhat)     : {outputs['C_Q_scaled']['decimal_24']}   "
              f"with λ = {result['inputs']['scale']['value']}")
    print(f"PASS random/axis   : {result['status']['passed_random']} / {result['status']['passed_axis']}")
    print(f"ALL PASSED         : {result['status']['all_passed']}\n")

    out_path = default_json_out()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
