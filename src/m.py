#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
m.py

ETHOS
  • Dependency results are only imported from upstream scripts, to avoid 
    circularity, and not hard-coded in this script.
  • Explicit constants, ratios and formulas are documented, substantiated,
    and contextually appropriate at the point of usage.

WHAT THIS CERTIFIES
  Dyadic richness gate sweep over odd m (CLI-only; no JSON reads).
  • Scans odd m in [m_min, m_max] and requires ord_m(2) ≥ 18
    (covering the 18 post-fold quartic classes: 6 faces + 12 edges).
  • Emits the first m that satisfies the gate (outputs.m_first).

OUTPUTS
  • JSON with:
      - meta (script, python, platform),
      - table: per-m diagnostics (odd, ord2, dyadic_pass, all_pass),
      - outputs.m_first: first odd m passing ord_m(2) ≥ 18.

INPUTS (CLI)
  OPTIONAL (not in scripts table)
    --m-min    <int>   Minimum m (inclusive; coerced to odd). Default: 1
    --m-max    <int>   Maximum m (inclusive). Default: 41
    --json-out <path>  Output path (default via utils.default_json_out).
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# --- shared repo utilities (authoritative CLI/IO helpers; no JSON reads here) ---
from utils import (
    default_json_out,  # default_json_out(user_path_or_None, __file__) -> Path
    write_json,        # write_json(Path, dict) -> None
)

# ---------------- Basic number-theory helpers ----------------

def gcd(a: int, b: int) -> int:
    """Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def multiplicative_order_mod(a: int, m: int) -> Optional[int]:
    """
    Smallest k>0 with a^k ≡ 1 (mod m), if gcd(a,m)=1; else None.
    A simple O(m) loop is ample for our small scan ranges.
    """
    if gcd(a, m) != 1:
        return None
    val = 1 % m
    for k in range(1, 10 * m):  # ample cap; returns early in practice
        val = (val * a) % m
        if val == 1 % m:
            return k
    return None

# ---------------- Row schema ----------------

@dataclass
class Row:
    m: int
    odd: bool
    ord2: Optional[int]
    dyadic_pass: bool
    all_pass: bool

# ---------------- Main ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Dyadic richness gate over odd m.")
    ap.add_argument("--m-min", type=int, default=1,
                    help="Min m (inclusive; coerced to odd). Default: 1")
    ap.add_argument("--m-max", type=int, default=41,
                    help="Max m (inclusive). Default: 41")
    ap.add_argument("--json-out", type=str, default=None,
                    help="Output JSON path (default via utils.default_json_out).")
    args = ap.parse_args()

    m_min = args.m_min if args.m_min % 2 == 1 else args.m_min + 1
    m_max = args.m_max

    rows: List[Row] = []
    m_first: Optional[int] = None

    for m in range(m_min, m_max + 1, 2):
        is_odd = (m % 2 == 1)
        k = multiplicative_order_mod(2, m)
        dy_pass = (k is not None and k >= 18)
        all_ok = is_odd and dy_pass

        rows.append(Row(
            m=m,
            odd=is_odd,
            ord2=k,
            dyadic_pass=dy_pass,
            all_pass=all_ok
        ))

        if all_ok and m_first is None:
            m_first = m

    if m_first is None:
        print("No odd m in range passed ord_m(2) ≥ 18.", file=sys.stderr)
        sys.exit(1)

    # Build JSON
    out = {
        "meta": {
            "script": Path(__file__).name,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "notes": "Single-gate (dyadic richness) scan; ord_m(2) ≥ 18.",
        },
        "table": [asdict(r) for r in rows],
        "outputs": {"m_first": int(m_first)},
        "status": {"ok": True},
    }

    out_path: Path = default_json_out(args.json_out, __file__)
    write_json(out_path, out)
    print(f"Wrote JSON results to: {out_path}")
    print(f"First odd m with ord_m(2) ≥ 18: m = {m_first}")

if __name__ == "__main__":
    main()
