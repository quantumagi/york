#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypothesis.py — protocol choices only (no derived constants)

Emits:
  • window = "sinc2"
  • lock_ratio = 8/15   (exact; used downstream to compute L0 = log(15/8))
"""

from __future__ import annotations
import json, sys, platform
from pathlib import Path
from fractions import Fraction
from utils import default_json_out, make_meta  # no JSON reads; helpers only

def main() -> None:
    window_value = "sinc2"   # Fejér power window K(x) = sinc^2 x

    # Exact quarter-FWHM lock ratio: sinc^2(x_q) = 8/15
    lock_num = 8
    lock_den = 15

    out = {
        "meta": make_meta(__file__, description="Protocol choices (primitive hypothesis only)"),
        "outputs": {
            "window": {
                "value": window_value,
                "desc": "Protocol window flag (Fejér power window K=sinc^2)."
            },
            "lock_ratio": {
                "num": lock_num,
                "den": lock_den,
                "fraction": f"{lock_num}/{lock_den}",
                "desc": "Exact quarter-FWHM lock: sinc^2(x_q)=8/15."
            }
        },
        "status": {"ok": True}
    }

    p = default_json_out(None, __file__)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote JSON results to: {p}")

if __name__ == "__main__":
    main()
