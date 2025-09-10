#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6: Sensitivity verifier for the Fejér–York lock.
- Verifies d/dL alpha^{-1} via central finite differences against the closed form.
- Verifies d/dL m_r via central finite differences against the closed form.
- Checks that derivative * ΔL^(2) matches the actual change between L_noquad and L_final.

Usage examples:
  python src/fejer_lock_sensitivity.py --from-json outputs/lock_alpha.json --json-out outputs/fejer_lock_sensitivity.json
  python src/fejer_lock_sensitivity.py --L-noquad 0.6358598629243455 --L-final 0.6358598659958836 --m 19 --kappa 0.49926851200152955 --json-out outputs/fejer_lock_sensitivity..json
"""

import argparse
import json
import math
from typing import Any, Dict, Optional
from pathlib import Path

try:
    import mpmath as mp
except ImportError:
    raise SystemExit("This script requires mpmath. Run: pip install mpmath")

def default_json_out(arg):
    if arg:
        return Path(arg)
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(__file__).stem + ".json")

def to_float(x: Any) -> float:
    # Convert mpmath/Decimal/NumPy scalars to plain float for JSON
    try:
        return float(x)
    except Exception:
        return float(mp.mpf(str(x)))


def to_bool(x: Any) -> bool:
    return bool(x)


def alpha_inv(L, m, kappa):
    # α^{-1}(L) = κ * m / (exp(2L/m) - 1)
    L = mp.mpf(L); m = mp.mpf(m); kappa = mp.mpf(kappa)
    E = mp.e**(2*L/m)
    return kappa * m / (E - 1)


def d_alpha_inv_dL(L, m, kappa):
    # d/dL α^{-1}(L) = -2 κ * e^{2L/m} / (e^{2L/m} - 1)^2
    L = mp.mpf(L); m = mp.mpf(m); kappa = mp.mpf(kappa)
    E = mp.e**(2*L/m)
    return -2 * kappa * E / (E - 1)**2


def mr(L, m):
    # m_r(L) = exp(L/m)
    L = mp.mpf(L); m = mp.mpf(m)
    return mp.e**(L/m)


def d_mr_dL(L, m):
    # d/dL m_r(L) = m_r / m
    L = mp.mpf(L); m = mp.mpf(m)
    return mr(L, m) / m


def central_diff(f, x, h):
    x = mp.mpf(x); h = mp.mpf(h)
    return (f(x + h) - f(x - h)) / (2*h)


def load_s5_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_value(d: Dict[str, Any], *candidates: str, required: bool = True) -> Optional[Any]:
    for k in candidates:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"None of the keys {candidates} found in JSON.")
    return None


def main():
    ap = argparse.ArgumentParser(description="S6: Sensitivity verifier for the Fejér–York lock.")
    ap.add_argument("--from-json", type=str, default=None,
                    help="Path to S5 JSON (lock_alpha.json). If provided, pulls L_eff, m, kappa, ΔL^(2).")
    ap.add_argument("--L-noquad", type=str, default=None, help="L_eff without quadratic term.")
    ap.add_argument("--L-final", type=str, default=None, help="L_eff with quadratic term.")
    ap.add_argument("--m", type=int, default=None, help="Multiplicity m.")
    ap.add_argument("--kappa", type=str, default=None, help="κ_∞.")
    ap.add_argument("--h", type=str, default="1e-9", help="Central-difference step in L (default 1e-9).")
    ap.add_argument("--deriv-tol", type=str, default="1e-12",
                    help="Absolute tolerance for derivative checks (default 1e-12).")
    ap.add_argument("--rel-tol", type=str, default="1e-12",
                    help="Relative tolerance for derivative checks (default 1e-12).")
    ap.add_argument("--dps", type=int, default=120, help="mpmath precision (default 120 dps).")
    ap.add_argument("--json-out", type=str, default=None, help="Where to write JSON results.")
    args = ap.parse_args()

    mp.mp.dps = int(args.dps)

    # Pull inputs from JSON if provided
    if args.from_json is not None:
        j = load_s5_json(args.from_json)
        L_noquad = extract_value(j, "L_eff (no quad)", "L_eff_no_quad")
        L_final  = extract_value(j, "L_eff (final)", "L_eff_final")
        m_val    = extract_value(j, "m")
        # κ may be stored as "κ_∞" or "kappa" depending on S5 version
        kappa_val = extract_value(j, "κ_∞", "kappa", "kappa_infty")
        # ΔL^(2) is nice-to-have (if absent we compute from L's directly)
        deltaL2 = extract_value(j, "ΔL^(2)", "deltaL2", required=False)
        if deltaL2 is None:
            deltaL2 = mp.mpf(L_final) - mp.mpf(L_noquad)
    else:
        # Require explicit inputs
        if args.L_noquad is None or args.L_final is None or args.m is None or args.kappa is None:
            raise SystemExit("Provide --from-json OR all of --L-noquad, --L-final, --m, --kappa.")
        L_noquad = mp.mpf(args.L_noquad)
        L_final  = mp.mpf(args.L_final)
        m_val    = int(args.m)
        kappa_val = mp.mpf(args.kappa)
        deltaL2 = mp.mpf(L_final) - mp.mpf(L_noquad)

    # Prepare numbers
    Lc = mp.mpf(L_final)  # check derivatives at the final L (can change if desired)
    h  = mp.mpf(args.h)
    atol = mp.mpf(args.deriv_tol)
    rtol = mp.mpf(args.rel_tol)
    m_val = mp.mpf(m_val)
    kappa_val = mp.mpf(kappa_val)

    # Analytic derivatives
    dAlpha_dL_analytic = d_alpha_inv_dL(Lc, m_val, kappa_val)
    dmr_dL_analytic    = d_mr_dL(Lc, m_val)

    # Finite-difference derivatives
    dAlpha_dL_fd = central_diff(lambda L: alpha_inv(L, m_val, kappa_val), Lc, h)
    dmr_dL_fd    = central_diff(lambda L: mr(L, m_val), Lc, h)

    # Errors
    def abs_rel_err(fd, an):
        fd = mp.mpf(fd); an = mp.mpf(an)
        abs_err = mp.fabs(fd - an)
        rel_err = abs_err / (mp.mpf(1) if mp.fabs(an) == 0 else mp.fabs(an))
        return abs_err, rel_err

    a_abs_err, a_rel_err = abs_rel_err(dAlpha_dL_fd, dAlpha_dL_analytic)
    r_abs_err, r_rel_err = abs_rel_err(dmr_dL_fd, dmr_dL_analytic)

    pass_alpha = (a_abs_err <= atol) or (a_rel_err <= rtol)
    pass_mr    = (r_abs_err <= atol) or (r_rel_err <= rtol)

    # ΔL^(2) propagation check on α^{-1}
    alpha_noquad = alpha_inv(L_noquad, m_val, kappa_val)
    alpha_final  = alpha_inv(L_final,  m_val, kappa_val)
    delta_alpha_actual = alpha_final - alpha_noquad
    delta_alpha_pred   = d_alpha_inv_dL(L_noquad, m_val, kappa_val) * (mp.mpf(deltaL2))

    shift_abs_err = mp.fabs(delta_alpha_pred - delta_alpha_actual)
    shift_rel_err = shift_abs_err / (mp.mpf(1) if mp.fabs(delta_alpha_actual) == 0 else mp.fabs(delta_alpha_actual))
    pass_shift = (shift_abs_err <= atol) or (shift_rel_err <= rtol)

    # Print summary
    print("\n=== S6: Sensitivity verifier ===")
    print(f"L_eff (no quad)       : {to_float(L_noquad)}")
    print(f"L_eff (final)         : {to_float(L_final)}")
    print(f"ΔL^(2)                : {to_float(deltaL2)}")
    print(f"m                     : {to_float(m_val)}")
    print(f"κ_∞                   : {to_float(kappa_val)}")
    print(f"h (central diff step) : {to_float(h)}")
    print(f"dps (mpmath)          : {mp.mp.dps}")

    print("\n— Derivative checks at L = L_final —")
    print(f"d(α^-1)/dL  (analytic): {to_float(dAlpha_dL_analytic)}")
    print(f"d(α^-1)/dL  (FD)      : {to_float(dAlpha_dL_fd)}")
    print(f"abs err / rel err     : {to_float(a_abs_err)}  /  {to_float(a_rel_err)}  -> PASS={pass_alpha}")

    print(f"\nd(m_r)/dL   (analytic): {to_float(dmr_dL_analytic)}")
    print(f"d(m_r)/dL   (FD)      : {to_float(dmr_dL_fd)}")
    print(f"abs err / rel err     : {to_float(r_abs_err)}  /  {to_float(r_rel_err)}  -> PASS={pass_mr}")

    print("\n— ΔL^(2) propagation on α^{-1} (at L_noquad) —")
    print(f"α^-1(no quad)         : {to_float(alpha_noquad)}")
    print(f"α^-1(final)           : {to_float(alpha_final)}")
    print(f"Δα^-1 (actual)        : {to_float(delta_alpha_actual)}")
    print(f"Δα^-1 (predicted)     : {to_float(delta_alpha_pred)}")
    print(f"abs err / rel err     : {to_float(shift_abs_err)}  /  {to_float(shift_rel_err)}  -> PASS={pass_shift}")

    all_pass = pass_alpha and pass_mr and pass_shift
    print(f"\nALL PASSED             : {all_pass}")

    # JSON output
    res = {
        "L_eff (no quad)": to_float(L_noquad),
        "L_eff (final)": to_float(L_final),
        "ΔL^(2)": to_float(deltaL2),
        "m": int(m_val),
        "kappa_infty": to_float(kappa_val),
        "dps": mp.mp.dps,
        "h": to_float(h),
        "derivatives": {
            "d_alpha_inv_dL": {
                "analytic": to_float(dAlpha_dL_analytic),
                "fd": to_float(dAlpha_dL_fd),
                "abs_err": to_float(a_abs_err),
                "rel_err": to_float(a_rel_err),
                "pass": to_bool(pass_alpha),
            },
            "d_mr_dL": {
                "analytic": to_float(dmr_dL_analytic),
                "fd": to_float(dmr_dL_fd),
                "abs_err": to_float(r_abs_err),
                "rel_err": to_float(r_rel_err),
                "pass": to_bool(pass_mr),
            },
        },
        "deltaL2_propagation_on_alpha": {
            "alpha_noquad": to_float(alpha_noquad),
            "alpha_final": to_float(alpha_final),
            "delta_alpha_actual": to_float(delta_alpha_actual),
            "delta_alpha_predicted": to_float(delta_alpha_pred),
            "abs_err": to_float(shift_abs_err),
            "rel_err": to_float(shift_rel_err),
            "pass": to_bool(pass_shift),
        },
        "ALL_PASSED": to_bool(all_pass),
        "notes": [
            "Analytic: d/dL α^{-1} = -2 κ_∞ e^{2L/m} / (e^{2L/m}-1)^2;  d/dL m_r = m_r / m.",
            "Propagation check evaluates derivative at L_noquad times ΔL^(2) and compares to actual α^{-1} shift.",
        ],
    }
    out_path = default_json_out(getattr(args, "json_out", None))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(f"Wrote JSON results to: {out_path}")


if __name__ == "__main__":
    main()
