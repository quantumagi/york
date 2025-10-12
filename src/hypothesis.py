#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hypothesis.py — protocol choices only (no derived constants)

Emits:
  • window = "sinc2"
"""

from __future__ import annotations
import json, sys, platform
from pathlib import Path
from fractions import Fraction
from utils import default_json_out, make_meta  # no JSON reads; helpers only

def main() -> None:
    window_value = "sinc2"   # Fejér power window K(x) = sinc^2 x

    out = {
        "meta": make_meta(__file__, description="Protocol choices (primitive hypothesis only)"),
        "outputs": {
            "window": {
                "value": window_value,
                "desc": "Protocol window flag (Fejér power window K=sinc^2)."
            }
        },
        "status": {"ok": True}
    }

    p = default_json_out(None, __file__)
    p.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote JSON results to: {p}")

if __name__ == "__main__":
    main()
