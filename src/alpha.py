#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpha.py
 
ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented from pre-existing
    theory, substantiated, and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Deterministically assemble the Fejér–York lock width L_eff from upstream
  scalars and map it to the observables
      m_r = exp(L_eff / m), 
      α^{-1} = κ_∞ m / (exp(2 L_eff / m) − 1),
  while propagating uncertainties from the Fejér tails and from an upstream
  absolute κ uncertainty (no file reads, no defaults for true inputs).

INPUTS
  (all required unless marked audit-only; *optional* where noted)
  --tol         : absolute tolerance guiding Fejér series termination (optional; defaults to 1e-30)
  --m           : multiplicity m (protocol)
  --kappa       : κ_∞ (numeric value from upstream kappa script)
  --dkappa      : absolute κ_∞ uncertainty (from upstream; no file reads)
  --x-q         : x_q from xq.py
  --L0          : (audit-only) optional L0; NEVER overrides L0(x_q)
  --chi-tt | --tt-drift : one is required (χ_TT or its per-m drift tt=χ_TT/6)
  --deltaL2     : ΔL^(2) from dL2.py
  --Gamma       : Γ from gamma.py
  --f3, --f4    : f'''(x_q), f''''(x_q) from xq.py
  --H4-m3       : ⟨H4^3⟩ from h4.py
  --H4-m4       : ⟨H4^4⟩ from h4.py
  --J2 | --s    : quarter-hop J^2 from j2.py OR s = sqrt(J2) (one required)

DERIVATION (no (a,b))
  1) L0 comes from x_q only: L0 = −log(sinc^2 x_q).
  2) TT drift: tt := χ_TT/6 if χ_TT given; else tt provided directly.
  3) Fejér tails:
       S_odd = Σ_{k≥3} 1/(2k+1)!    (with remainder bound),
       S_even(s) = Σ_{n≥4} [2 B_{2n}/(n(2n)!)] s^{2n}  (with remainder bound),
     where s = √J2.
  4) Protocol corrections:
       ΔL = ΔL^(2) + ΔL^(3) + ΔL^(4),
       ΔL^(3) = (f3/6) ⟨H4^3⟩ p^3,  ΔL^(4) = (f4/24) ⟨H4^4⟩ p^4,  p = χ_TT Γ.
     Higher terms (≥5) are conservatively bounded from |ΔL⁴/ΔL³|.
  5) Observables:
       L_eff = L0 + tt + S_odd + S_even + ΔL,
       m_r = exp(L_eff/m),   α^{-1} = κ_∞ m / (exp(2L_eff/m) − 1).

OUTPUT (JSON → outputs/alpha.json)
  {
    "inputs": {...}, "intermediates": {...}, "outputs": {...}, "status": {"ok": true}
  }
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mpmath as mp

# utils expected in repo
from utils import (
    parse_number,
    fixed_places,
    default_json_out,
    write_json,
)

# ------------------------------ precision ------------------------------
mp.mp.dps = 200  # house style: set high precision before any parsing/printing

# ------------------------------ config ------------------------------
DEC_SIG      = 24
MAX_DECIMALS = 36
MIN_DECIMALS = 6

# ------------------------------ small helpers ------------------------------
def nstr_plain(x: mp.mpf, sig: int) -> str:
    return mp.nstr(mp.mpf(x), sig)

def decimals_from_abs_error(err: mp.mpf, *, cap: int = MAX_DECIMALS) -> int:
    if not mp.isfinite(err) or err <= 0:
        return MIN_DECIMALS
    places = int(mp.ceil(-mp.log10(err)))
    soft_cap = max(MIN_DECIMALS, int(mp.mp.dps - 8))
    return int(max(MIN_DECIMALS, min(places, min(cap, soft_cap))))

# ------------------------------ Fejér tails ------------------------------
def S_odd_tail(tol: mp.mpf, kmax: int = 2000) -> tuple[mp.mpf, mp.mpf]:
    total = mp.mpf('0')
    last_term = None
    last_k = None
    for k in range(3, kmax + 1):
        n = 2 * k + 1
        term = 1 / mp.factorial(n)
        total += term
        if term < tol:
            last_term = term
            last_k = k
            break
    if last_term is None:
        return total, mp.mpf('inf')
    r = 1 / (mp.mpf(2 * last_k + 2) * mp.mpf(2 * last_k + 3))
    rem = last_term / (1 - r) if r < 1 else last_term
    return total, mp.fabs(rem)

def S_even_bern(s: mp.mpf, tol: mp.mpf, nmax: int = 1000) -> tuple[mp.mpf, mp.mpf]:
    total = mp.mpf('0')
    s2 = s * s
    last_term_mag = None
    last_n = None
    for n in range(4, nmax + 1):
        B2n = mp.bernoulli(2 * n)
        term = (2 * B2n) / (n * mp.factorial(2 * n)) * (s2 ** n)
        total += term
        tmag = mp.fabs(term)
        if tmag < tol:
            last_term_mag = tmag
            last_n = n
            break
    if last_term_mag is None:
        return total, mp.mpf('inf')
    n_next = last_n + 1
    B2n1 = mp.bernoulli(2 * n_next)
    next_term_mag = mp.fabs((2 * B2n1) / (n_next * mp.factorial(2 * n_next)) * (s2 ** n_next))
    r = next_term_mag / last_term_mag if last_term_mag > 0 else mp.mpf('1')
    r = min(mp.mpf('0.9'), mp.fabs(r))
    rem = next_term_mag / (1 - r) if r < 1 else next_term_mag
    return total, mp.fabs(rem)

# ------------------------------ ΔL higher-tail bound ------------------------------
def higher_dL_tail_bound(dL3: mp.mpf, dL4: mp.mpf) -> mp.mpf:
    """
    Conservative bound for omitted ΔL terms (n >= 5) assuming approximate geometric decline
    suggested by the last two magnitudes:
        r_est = min(0.95, 1.25 * |ΔL⁴/ΔL³|)   (fallback r_est=0.5 if ΔL³≈0)
        tail  ≤ |ΔL⁴| * r_est / (1 - r_est)
    """
    T3 = mp.fabs(dL3)
    T4 = mp.fabs(dL4)
    if not mp.isfinite(T3) or not mp.isfinite(T4):
        return mp.mpf('inf')
    if T4 == 0:
        return mp.mpf('0')
    if T3 <= 0:
        r_est = mp.mpf('0.5')
    else:
        r_est = 1.25 * (T4 / T3)
        r_est = min(mp.mpf('0.95'), mp.fabs(r_est) if r_est > 0 else mp.mpf('0.1'))
    return T4 * r_est / (1 - r_est)

# ------------------------------ assembly core ------------------------------
def assemble_and_observe(
    *,
    m_s: str,
    kappa_s: str,
    dkappa_s: str,
    xq_s: str,
    L0_s_opt: Optional[str],
    chi_tt_s: Optional[str],
    tt_drift_s: Optional[str],
    deltaL2_s: str,
    Gamma_s: str,
    f3_s: str,
    f4_s: str,
    J2_s_opt: Optional[str],
    s_s_opt: Optional[str],
    tol: mp.mpf
) -> Dict[str, Any]:

    if tol <= 0:
        raise ValueError("--tol must be positive")

    # precision guided by tol
    mp.mp.dps = max(80, int(-mp.log10(tol)) + 60)

    m       = parse_number(m_s).float
    kappa   = parse_number(kappa_s).float
    dkappa  = parse_number(dkappa_s).float
    x_q     = parse_number(xq_s).float
    Gamma   = parse_number(Gamma_s).float
    dL2     = parse_number(deltaL2_s).float
    f3      = parse_number(f3_s).float
    f4      = parse_number(f4_s).float

    # s or J2
    J2_val: Optional[mp.mpf] = None
    s_val: Optional[mp.mpf]  = None
    if J2_s_opt is not None:
        J2_val = parse_number(J2_s_opt).float
        s_val  = mp.sqrt(J2_val)
    if s_s_opt is not None and s_val is None:
        s_val = parse_number(s_s_opt).float
    if s_val is None:
        raise SystemExit("[alpha] Must provide --J2 or --s (Bernoulli tail parameter).")

    # L0 from x_q (audit-only if L0 provided)
    L0_from_xq = -mp.log((mp.sin(x_q) / x_q) ** 2)
    if L0_s_opt is not None:
        L0_in = parse_number(L0_s_opt).float
        if mp.fabs(L0_in - L0_from_xq) > mp.mpf('1e-18'):
            print(f"[alpha] WARNING: supplied --L0 differs from -log(sinc^2 x_q). Using x_q-derived.")
    L0_use = L0_from_xq

    # χ_TT / tt_drift reconciliation
    chi_tt = parse_number(chi_tt_s).float if chi_tt_s is not None else None
    tt_in  = parse_number(tt_drift_s).float if tt_drift_s is not None else None
    if chi_tt is None and tt_in is None:
        raise SystemExit("[alpha] Need one of --chi-tt or --tt-drift.")
    if chi_tt is None:
        chi_tt = 6 * tt_in
    tt = chi_tt / 6
    if tt_in is not None and mp.fabs(tt - tt_in) > mp.mpf('1e-18'):
        print(f"[alpha] WARNING: tt_drift differs from χ_TT/6 (audit-only).")

    # Fejér tails with remainder bounds
    Sodd,  Sodd_err  = S_odd_tail(tol)
    Seven, Seven_err = S_even_bern(s_val, tol)

    # ΔL^3, ΔL^4 (protocol)
    def compute_dL_tail(H4_m3_val: mp.mpf, H4_m4_val: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        p = chi_tt * Gamma
        dL3 = (f3 / 6)  * H4_m3_val * (p ** 3)
        dL4 = (f4 / 24) * H4_m4_val * (p ** 4)
        return p, dL3, dL4

    return {
        "internal": {
            "m": m, "kappa": kappa, "dkappa": dkappa,
            "x_q": x_q, "L0_use": L0_use,
            "chi_tt": chi_tt, "tt": tt, "Gamma": Gamma,
            "f3": f3, "f4": f4, "J2": J2_val, "s": s_val,
            "Sodd": Sodd, "Seven": Seven, "Sodd_err": Sodd_err, "Seven_err": Seven_err,
            "deltaL2": dL2,
            "compute_dL_tail": compute_dL_tail
        }
    }

def propagate_errors(L: mp.mpf, m: mp.mpf, kappa: mp.mpf,
                     dL_abs: mp.mpf, dkappa_abs: mp.mpf) -> dict:
    """
    Worst-case (L1) combination of independent sources:
      δα_total = |∂α/∂L| δL + |∂α/∂κ| δκ,    ∂α/∂κ = α/κ
      δm_r     = |∂m_r/∂L| δL = (m_r/m) δL
    """
    mr = mp.e ** (L / m)
    ex = mp.e ** (2 * L / m)
    alpha_inv = kappa * m / (ex - 1)

    dalpha_dL = mp.fabs(-2 * kappa * ex / ((ex - 1) ** 2))
    dalpha_dk = mp.fabs(alpha_inv / kappa) if kappa != 0 else mp.mpf('inf')
    dmrdL     = mp.fabs(mr / m)

    err_L_eff       = mp.fabs(dL_abs)
    err_mr          = dmrdL * err_L_eff
    err_alpha_fromL = dalpha_dL * err_L_eff
    err_alpha_fromk = dalpha_dk * mp.fabs(dkappa_abs)
    err_alpha_total = err_alpha_fromL + err_alpha_fromk

    return {
        "mr": mr, "alpha": alpha_inv,
        "err": {
            "L_eff": err_L_eff,
            "m_r": err_mr,
            "alpha_from_L": err_alpha_fromL,
            "from_kappa": err_alpha_fromk,
            "alpha_total": err_alpha_total
        }
    }

# ------------------------------ reporting ------------------------------
def print_report(bundle: Dict[str, Any], m_int: int, kappa_str: str, dkappa_str: str,
                 H4_m3: mp.mpf, H4_m4: mp.mpf) -> Dict[str, Any]:
    B = bundle["internal"]
    p, dL3, dL4 = B["compute_dL_tail"](H4_m3, H4_m4)
    dL_sum = B["deltaL2"] + dL3 + dL4

    # Higher-order ΔL (≥5) tail bound
    dL_higher = higher_dL_tail_bound(dL3, dL4)

    # Final L_eff
    L_eff = B["L0_use"] + B["tt"] + B["Sodd"] + B["Seven"] + dL_sum

    # total L uncertainty = Fejér series tails + ΔL higher tail
    dL_err_total = mp.fabs(B["Sodd_err"]) + mp.fabs(B["Seven_err"]) + mp.fabs(dL_higher)

    # propagate
    prop = propagate_errors(L_eff, B["m"], B["kappa"], dL_err_total, B["dkappa"])
    mr = prop["mr"]
    alpha_inv = prop["alpha"]
    err = prop["err"]

    # choose decimals from the errors
    L_places  = decimals_from_abs_error(err["L_eff"])
    mr_places = decimals_from_abs_error(err["m_r"])
    a_places  = decimals_from_abs_error(err["alpha_total"])

    # decimal_24 for machine use
    L_eff_dec = nstr_plain(L_eff, DEC_SIG)
    m_r_dec   = nstr_plain(mr,   DEC_SIG)
    alpha_dec = nstr_plain(alpha_inv, DEC_SIG)

    # human-facing auto prints
    L_eff_auto  = fixed_places(L_eff,  L_places)
    m_r_auto    = fixed_places(mr,     mr_places)
    alpha_auto  = fixed_places(alpha_inv, a_places)

    print("\n=== S5: Fejér–York lock → α and m_r (deterministic assembly) ===")
    print(f"x_q                    : {nstr_plain(B['x_q'], DEC_SIG)}")
    print(f"L0 (from upstream)     : {nstr_plain(B['L0_use'], DEC_SIG)}")
    print(f"TT drift               : {nstr_plain(B['tt'], DEC_SIG)}   (supplied/derived)")
    print(f"S_odd (k≥3)            : {nstr_plain(B['Sodd'], DEC_SIG)}  [±{nstr_plain(B['Sodd_err'], 8)}]")
    print(f"S_even(s=√J2)          : {nstr_plain(B['Seven'], DEC_SIG)}   [±{nstr_plain(B['Seven_err'], 8)}]  [s={nstr_plain(B['s'], 18)}]")
    print(f"ΔL² (given)            : {nstr_plain(B['deltaL2'], DEC_SIG)}")
    print(f"ΔL³ (protocol)         : {nstr_plain(dL3, DEC_SIG)}")
    print(f"ΔL⁴ (protocol)         : {nstr_plain(dL4, DEC_SIG)}")
    print(f"ΔL (ledger sum)        : {nstr_plain(dL_sum, DEC_SIG)}")
    print(f"ΔL≥5 tail (bound)      : ≤{nstr_plain(dL_higher, 8)}   (from |ΔL⁴/ΔL³|)")
    print(f"κ_∞                    : {kappa_str}   [±{dkappa_str}]")
    print(f"L_eff (final)          : {L_eff_auto}  [±{nstr_plain(err['L_eff'], 8)}] (auto {L_places} dp)")
    print(f"m                      : {m_int}")
    print(f"m_r                    : {m_r_auto}    [±{nstr_plain(err['m_r'], 8)}] (auto {mr_places} dp)")
    print(f"α^-1                   : {alpha_auto}  [±{nstr_plain(err['alpha_total'], 8)}] "
          f"(= from L {nstr_plain(err['alpha_from_L'],8)} + from κ {nstr_plain(err['from_kappa'],8)}; auto {a_places} dp)")

    out_bundle = {
        "pieces": {
            "x_q":          {"decimal_24": nstr_plain(B["x_q"], DEC_SIG)},
            "L0":           {"decimal_24": nstr_plain(B["L0_use"], DEC_SIG)},
            "tt_drift":     {"decimal_24": nstr_plain(B["tt"], DEC_SIG)},
            "S_odd":        {"decimal_24": nstr_plain(B["Sodd"],  DEC_SIG), "err_abs": nstr_plain(B["Sodd_err"], 24)},
            "S_even":       {"decimal_24": nstr_plain(B["Seven"], DEC_SIG), "err_abs": nstr_plain(B["Seven_err"], 24)},
            "deltaL2":      {"decimal_24": nstr_plain(B["deltaL2"], DEC_SIG)},
            "deltaL3":      {"decimal_24": nstr_plain(dL3, DEC_SIG)},
            "deltaL4":      {"decimal_24": nstr_plain(dL4, DEC_SIG)},
            "deltaL_sum":   {"decimal_24": nstr_plain(dL_sum, DEC_SIG)},
            "deltaL_ge5_bound": {"abs_bound": nstr_plain(dL_higher, 24)},
            "chi_tt":       {"decimal_24": nstr_plain(B["chi_tt"], DEC_SIG)},
            "Gamma":        {"decimal_24": nstr_plain(B["Gamma"], DEC_SIG)},
            "f3":           {"decimal_24": nstr_plain(B["f3"], DEC_SIG)},
            "f4":           {"decimal_24": nstr_plain(B["f4"], DEC_SIG)},
            "J2":           ({"decimal_24": nstr_plain(B["J2"], DEC_SIG)} if B["J2"] is not None else {"decimal_24": None}),
            "s":            {"decimal_24": nstr_plain(B["s"], DEC_SIG)},
            "kappa_uncertainty_abs": nstr_plain(B["dkappa"], 24),
        },
        "observables": {
            "L_eff":     {"decimal_24": L_eff_dec, "auto": L_eff_auto, "auto_places": L_places,
                          "err_abs": nstr_plain(err["L_eff"], 24)},
            "m_r":       {"decimal_24": m_r_dec,   "auto": m_r_auto,   "auto_places": mr_places,
                          "err_abs": nstr_plain(err["m_r"], 24)},
            "alpha_inv": {"decimal_24": alpha_dec, "auto": alpha_auto, "auto_places": a_places,
                          "err_abs": nstr_plain(err["alpha_total"], 24),
                          "err_breakdown": {
                              "from_L": nstr_plain(err["alpha_from_L"], 24),
                              "from_kappa": nstr_plain(err["from_kappa"], 24)
                          }},
        }
    }
    return out_bundle

# ------------------------------ CLI ------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="S5 — Fejér–York lock → α and m_r (deterministic assembly; rigorous auto-precision)."
    )
    ap.add_argument("--tol",     type=str, default="1e-30",
                    help="abs tol guiding Fejér series termination (default: 1e-30)")
    ap.add_argument("--m",       type=int, required=True, help="multiplicity m (protocol)")
    ap.add_argument("--kappa",   type=str, required=True, help="κ_∞ (numeric value from upstream)")
    ap.add_argument("--dkappa",  type=str, required=True, help="absolute κ_∞ uncertainty (from upstream)")
    ap.add_argument("--x-q",     type=str, required=True, help="x_q from xq.py")
    ap.add_argument("--L0",      type=str, required=False, help="optional L0 audit (never overrides L0(x_q))")
    ap.add_argument("--chi-tt",  type=str, required=False, help="χ_TT from chitt.py (or provide tt-drift)")
    ap.add_argument("--tt-drift",type=str, required=False, help="tt drift directly (or provide chi-tt)")
    ap.add_argument("--deltaL2", type=str, required=True, help="ΔL^(2) from dL2.py")
    ap.add_argument("--Gamma",   type=str, required=True, help="Γ from gamma.py")
    ap.add_argument("--f3",      type=str, required=True, help="f'''(x_q) from xq.py")
    ap.add_argument("--f4",      type=str, required=True, help="f''''(x_q) from xq.py")
    ap.add_argument("--H4-m3",   type=str, required=True, help="⟨H4^3⟩ from h4.py")
    ap.add_argument("--H4-m4",   type=str, required=True, help="⟨H4^4⟩ from h4.py")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--J2", type=str, required=False, help="quarter-hop J^2 from j2.py (preferred)")
    g.add_argument("--s",  type=str, required=False, help="Bernoulli-tail parameter s (if J2 not provided)")

    ap.add_argument("--json-out", type=str, default=None, help="optional path to write JSON results")
    args = ap.parse_args()

    tol_mpf = parse_number(args.tol).float

    stage1 = assemble_and_observe(
        m_s=str(args.m),
        kappa_s=args.kappa,
        dkappa_s=args.dkappa,
        xq_s=args.x_q,
        L0_s_opt=args.L0,
        chi_tt_s=args.chi_tt,
        tt_drift_s=args.tt_drift,
        deltaL2_s=args.deltaL2,
        Gamma_s=args.Gamma,
        f3_s=args.f3,
        f4_s=args.f4,
        J2_s_opt=args.J2,
        s_s_opt=args.s,
        tol=tol_mpf
    )

    H4_m3 = parse_number(args.H4_m3).float
    H4_m4 = parse_number(args.H4_m4).float

    out_obs = print_report(stage1, args.m, args.kappa, args.dkappa, H4_m3, H4_m4)

    out = {
        "inputs": {
            "x_q":      {"decimal": args.x_q},
            "L0":       ({"decimal": args.L0} if args.L0 is not None else {"decimal": None}),
            "chi_tt":   ({"decimal": args.chi_tt} if args.chi_tt is not None else {"decimal": None}),
            "tt_drift": ({"decimal": args.tt_drift} if args.tt_drift is not None else {"decimal": None}),
            "deltaL2":  {"decimal": args.deltaL2},
            "Gamma":    {"decimal": args.Gamma},
            "f3":       {"decimal": args.f3},
            "f4":       {"decimal": args.f4},
            "H4_m3":    {"decimal": args.H4_m3},
            "H4_m4":    {"decimal": args.H4_m4},
            "J2":       ({"decimal": args.J2} if args.J2 is not None else {"decimal": None}),
            "s":        ({"decimal": args.s}  if args.s  is not None else {"decimal": None}),
            "kappa":    {"decimal": args.kappa},
            "dkappa":   {"decimal": args.dkappa},
        },
        "intermediates": {
            "pieces": out_obs["pieces"]
        },
        "outputs": out_obs["observables"],
        "status": {"ok": True}
    }

    out_path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
