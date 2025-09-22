#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m.py — EL-lock isolation + dyadic gate (CLI-only; uses utils.py)

WHAT THIS DOES
  • Sweeps odd m in [--m-min, --m-max] and checks:
      (i)   odd m (quarter-hop),
      (ii)  dyadic richness proxy: ord_m(2) ≥ 18,
      (iii) Fejér isolation bound evaluated at the EL lock x_q (provided via --xq).
  • Consumes ONLY CLI flags; NO JSON reads.

INPUTS (all required except bounds)
  --xq         : EL lock abscissa x_q (decimal or 'p/q')
  --m-min      : minimum m (default: 1; coerced to odd)
  --m-max      : maximum m (default: 41)
  --json-out   : optional output path (default: outputs/m.json via utils.default_json_out)

OUTPUT
  • JSON with meta, intermediates, per-m table, and outputs.m_first
"""

from __future__ import annotations
import argparse, math, platform, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import mpmath as mp  # mpf arithmetic only

# --- shared repo utilities (authoritative CLI/IO helpers; no JSON reads here) ---
from utils import (
    parse_number,      # parse_number("1.234" or "8/15") -> object with .raw, .rational, .float (mpf)
    default_json_out,  # default_json_out(user_path_or_None, __file__) -> Path
    write_json,        # write_json(Path, dict) -> None
)

# ---------------- Fejér window & calculus ----------------

def sinc(x: float) -> float:
    """Normalized sinc."""
    return 1.0 if x == 0.0 else math.sin(x) / x

def K_fejer(x: float) -> float:
    """Fejér (power) window K(x) = sinc^2 x."""
    v = sinc(x)
    return v * v

def f_prime(x: float) -> float:
    r"""f'(x) for f(x) = -log(sinc^2 x) = -2(log sin x - log x) ⇒ f'(x) = -2 cot x + 2/x."""
    return -2.0 * (math.cos(x) / math.sin(x)) + 2.0 / x

def f_double_prime(x: float) -> float:
    r"""f''(x) = 2(csc^2 x - x^{-2})."""
    s = math.sin(x)
    return 2.0 * (1.0 / (s * s) - 1.0 / (x * x))

# ---------------- Isolation check (EL lock x_q) ----------------

def isolation_at_lock(m: int, x_lock: float) -> Tuple[bool, float, float]:
    r"""Fejér main-lobe isolation at the EL lock x_q.
        Δ = π/(2m), t = π - Δ.
        Check: L_m := K(t)/K(x_q) ≤ ε_lin := [ f''(x_q) / (2 f'(x_q)) ] Δ^2 .
    """
    Δ = math.pi / (2 * m)
    t_side = math.pi - Δ
    Lm = K_fejer(t_side) / K_fejer(x_lock)
    fp = f_prime(x_lock)
    fpp = f_double_prime(x_lock)
    eps_lin = (fpp / (2.0 * fp)) * (Δ * Δ)
    return (Lm <= eps_lin, Lm, eps_lin)

# ---------------- Dyadic microtests ----------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def multiplicative_order_mod(a: int, m: int) -> Optional[int]:
    """Smallest k>0 with a^k ≡ 1 (mod m), if gcd(a,m)=1; else None."""
    if gcd(a, m) != 1:
        return None
    val = 1 % m
    for k in range(1, 10 * m):  # ample cap
        val = (val * a) % m
        if val == 1 % m:
            return k
    return None

def quartic_orbit_count() -> int:
    """ℓ=4: faces (6) + edges (12) = 18."""
    return 6 + 12

def dyadic_proxy_pass(m: int) -> Tuple[bool, Optional[int], int]:
    """PASS ⇔ m odd, gcd(2,m)=1, and ord_m(2) ≥ 18."""
    N_orbits = quartic_orbit_count()
    if m % 2 == 0 or gcd(2, m) != 1:
        return (False, None, N_orbits)
    ord2 = multiplicative_order_mod(2, m)
    ok = (ord2 is not None and ord2 >= N_orbits)
    return (ok, ord2, N_orbits)

# ---------------- Sweep & emit ----------------

@dataclass
class Row:
    m: int
    odd: bool
    dyadic_pass: bool
    ord2: Optional[int]
    N_orbits: int
    iso_metric_1: float  # L_m
    iso_metric_2: float  # ε_lin
    isolation_pass: bool
    all_pass: bool

def main() -> None:
    ap = argparse.ArgumentParser(description="Quarter-hop microtests (EL-lock isolation; dyadic proxy).")
    ap.add_argument("--xq", required=True, type=str, help="EL lock x_q (decimal or 'p/q').")
    ap.add_argument("--m-min", type=int, default=1, help="Min odd m (inclusive; coerced to odd).")
    ap.add_argument("--m-max", type=int, default=41, help="Max m (inclusive).")
    ap.add_argument("--json-out", type=str, default=None, help="Output JSON path (default: outputs/m.json).")
    args = ap.parse_args()

    # Parse x_q via shared utils (mpf; no Python floats)
    pn_xq = parse_number(args.xq)        # -> object with .float (mpf)
    x_q = float(pn_xq.float)             # local numeric for math.* calls (internal only)

    m_min = args.m_min if args.m_min % 2 == 1 else args.m_min + 1
    m_max = args.m_max

    rows: List[Row] = []
    m_first: Optional[int] = None

    for m in range(m_min, m_max + 1, 2):
        odd_ok = True
        dy_ok, ord2, N_orb = dyadic_proxy_pass(m)
        ok_iso, Lm, eps_lin = isolation_at_lock(m, x_q)

        all_ok = odd_ok and dy_ok and ok_iso
        rows.append(Row(
            m=m, odd=odd_ok, dyadic_pass=dy_ok, ord2=ord2, N_orbits=N_orb,
            iso_metric_1=float(Lm), iso_metric_2=float(eps_lin),
            isolation_pass=ok_iso, all_pass=all_ok
        ))
        if all_ok and m_first is None:
            m_first = m

    if m_first is None:
        print("No odd m in range passed all filters.", file=sys.stderr)
        sys.exit(1)

    # Build JSON (decimals as strings; include raw/rational tag for x_q)
    out = {
        "meta": {
            "script": Path(__file__).name,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "notes": "Isolation bound evaluated at EL lock x_q provided via --xq.",
        },
        "intermediates": {
            "N_orbits": quartic_orbit_count(),
            "x_q": {
                "raw": pn_xq.raw,
                **({"rational": pn_xq.rational} if pn_xq.rational else {}),
                "decimal": mp.nstr(pn_xq.float, 60),
            },
            "isolation_metrics_note": "iso_metric_1, iso_metric_2 are (L_m, ε_lin) at the EL lock.",
        },
        "table": [asdict(r) for r in rows],
        "outputs": {"m_first": m_first},
        "status": {"ok": True},
    }

    out_path: Path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")
    print(f"First odd m with all filters satisfied: m = {m_first}")

if __name__ == "__main__":
    main()
