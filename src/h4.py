##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h4.py — Spherical quartic moments for the A1g harmonic (exact, upstream derivation)

ETHOS
  • CLI-only script (no JSON reads, no external I/O beyond writing outputs/…).
  • No hard-coded “mystery” constants: every value is derived in-script from standard identities.
  • References are placed right next to the identities where they are used.

PURPOSE
  For a unit vector u = (u_x, u_y, u_z) uniformly distributed on S^2, define the cubic (A1g)
  quartic harmonic
      H4(u) := u_x^4 + u_y^4 + u_z^4 − 3/5 .
  This script computes the exact moments
      ⟨H4^2⟩, ⟨H4^3⟩, ⟨H4^4⟩
  by reducing them to Dirichlet moments of w := (u_x^2, u_y^2, u_z^2).

DERIVATION SKETCH (standard, cited)
  (1) Sphere → Dirichlet mapping.
      If U is uniform on S^{p-1}, then W := (U_1^2,…,U_p^2) ∼ Dirichlet(½,…,½).
      Construction via U = Z/||Z|| with i.i.d. Z_i ∼ N(0,1), then W_i = Z_i^2 / ∑ Z_j^2.
      [Marsaglia 1972; Johnson–Kotz–Balakrishnan 2000, Vol.1 §49]

  (2) Dirichlet mixed moments.
      For X ∼ Dirichlet(α), with r_i ∈ ℕ_0,
         E[∏ X_i^{r_i}] = ∏ (α_i)_{r_i} / (α_0)_{∑ r_i},  where (a)_n is the rising factorial
         and α_0 = ∑ α_i.  Here p=3 and α_i = 1/2 ⇒ α_0 = 3/2. We only need small exponents.
      [Johnson–Kotz–Balakrishnan 2000, Vol.1 §49; NIST DLMF §26.5]

  (3) A1g quartic polynomial.
      H4(u) = ∑ u_i^4 − 3/5 is the unique degree-4, mean-zero, fully symmetric (A1g) cubic
      harmonic built from quartic axis terms (subtracting 3/5 centers it, since
      E[∑ u_i^4] = 3·E[u_x^4] = 3·(1/5)).
      [Dresselhaus–Dresselhaus–Jorio 2008, Ch.4–5]
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict, Tuple
import mpmath as mp

# Standardized utilities (ETHOS-compliant: no JSON reads inside)
from utils import (
    ledger_header,
    console_show,
    default_json_out,
    write_json,
    make_meta,
)

# --------------------------- exact arithmetic helpers ---------------------------

def rising(a: Fraction, n: int) -> Fraction:
    """Rising factorial (a)_n with exact rationals. [Dirichlet moments: JKB 2000, Vol.1 §49]"""
    out = Fraction(1, 1)
    for k in range(n):
        out *= (a + k)
    return out

def dirichlet_moment(r: Tuple[int, int, int], alpha: Tuple[Fraction, Fraction, Fraction]) -> Fraction:
    """
    Exact Dirichlet mixed moment:
        E[ w1^{r1} w2^{r2} w3^{r3} ] = ∏ (α_i)_{r_i} / (α0)_{r1+r2+r3},  α0 = ∑ α_i.
    [JKB 2000, Vol.1 §49; NIST DLMF §26.5]
    """
    a1, a2, a3 = alpha
    r1, r2, r3 = r
    a0 = a1 + a2 + a3
    num = rising(a1, r1) * rising(a2, r2) * rising(a3, r3)
    den = rising(a0, r1 + r2 + r3)
    return num / den

# --------------------------- exact derivation for S^2 ---------------------------

def compute_all() -> Dict[str, Fraction]:
    """
    Compute ⟨H4^k⟩ (k=2,3,4) exactly from Dirichlet moments with α=(1/2,1/2,1/2).
    H4(u) := S1 − 3/5, where S1 := u_x^4 + u_y^4 + u_z^4 = w1^2 + w2^2 + w3^2.
    """
    # Sphere→Dirichlet map (S^2 ⇒ Dir(½,½,½))  [Marsaglia '72; JKB 2000 §49]
    a = (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2))

    # Base Dirichlet moments (all exact Fractions)
    # E[u_x^4]   = E[w1^2]               = (1/2)_2 / (3/2)_2 = 1/5
    # E[u_x^8]   = E[w1^4]               = (1/2)_4 / (3/2)_4 = 1/9
    # E[u_x^4 u_y^4] = E[w1^2 w2^2]      = (1/2)_2(1/2)_2 / (3/2)_4 = 1/105
    E_w1_2      = dirichlet_moment((2, 0, 0), a)
    E_w1_4      = dirichlet_moment((4, 0, 0), a)
    E_w1_2w2_2  = dirichlet_moment((2, 2, 0), a)
    assert E_w1_2 == Fraction(1, 5)
    assert E_w1_4 == Fraction(1, 9)
    assert E_w1_2w2_2 == Fraction(1, 105)

    # S1 := sum of axis quartics
    E_S1   = 3 * E_w1_2
    E_S1_2 = 3 * E_w1_4 + 6 * E_w1_2w2_2

    # Centering constant μ = E[S1] = 3/5 (A1g mean-zero constraint)  [DDJ 2008]
    mu = Fraction(3, 5)

    # Second central moment: ⟨H4²⟩ = E[(S1-μ)²] = E[S1²] − 2 μ E[S1] + μ²  → 16/525
    H4_sq = E_S1_2 - 2 * mu * E_S1 + mu * mu

    # Third raw moment of S1 via (a+b+c)^3 expansion; symmetry groups terms
    #   sum a^3 → 3*E[w1^6];  sum a_i^2 a_j (i≠j) → 18*E[w1^4 w2^2];  abc → 6*E[w1^2 w2^2 w3^2]
    E_w1_6             = dirichlet_moment((6, 0, 0), a)
    E_w1_4w2_2         = dirichlet_moment((4, 2, 0), a)
    E_w1_2w2_2w3_2     = dirichlet_moment((2, 2, 2), a)
    E_S1_3 = 3*E_w1_6 + 18*E_w1_4w2_2 + 6*E_w1_2w2_2w3_2

    # Third central moment: ⟨H4³⟩ = E[S1³] − 3 μ E[S1²] + 3 μ² E[S1] − μ³  → 384/125125
    H4_m3 = E_S1_3 - 3*mu*E_S1_2 + 3*(mu**2)*E_S1 - (mu**3)

    # Fourth raw moment of S1 via partitions of 4 into ≤3 parts:
    #   3*E[w1^8] + 24*E[w1^6 w2^2] + 18*E[w1^4 w2^4] + 36*E[w1^4 w2^2 w3^2]
    E_w1_8             = dirichlet_moment((8, 0, 0), a)
    E_w1_6w2_2         = dirichlet_moment((6, 2, 0), a)
    E_w1_4w2_4         = dirichlet_moment((4, 4, 0), a)
    E_w1_4w2_2w3_2     = dirichlet_moment((4, 2, 2), a)
    E_S1_4 = (
        3*E_w1_8
        + 24*E_w1_6w2_2
        + 18*E_w1_4w2_4
        + 36*E_w1_4w2_2w3_2
    )

    # Fourth central moment: ⟨H4⁴⟩ = E[(S1-μ)⁴]  → 22784/10635625
    H4_m4 = (
        E_S1_4
        - 4*mu*E_S1_3
        + 6*(mu**2)*E_S1_2
        - 4*(mu**3)*E_S1
        + (mu**4)
    )

    return {
        "E_u4":  E_w1_2,   # 1/5
        "H4_sq": H4_sq,    # 16/525
        "H4_m3": H4_m3,    # 384/125125
        "H4_m4": H4_m4,    # 22784/10635625
    }

# --------------------------- formatting helpers ---------------------------

def _frac_str(fr: Fraction) -> str:
    return f"{fr.numerator}/{fr.denominator}"

def _mpf(fr: Fraction) -> mp.mpf:
    """Exact mpf from Fraction (no Python-float roundtrip)."""
    return mp.mpf(fr.numerator) / mp.mpf(fr.denominator)

# --------------------------- main / JSON emission ---------------------------

def main() -> None:
    mp.mp.dps = 80
    res = compute_all()

    # Console ledger (exact + high-precision decimal echoes)
    ledger_header("Spherical ⟨H4^k⟩ (exact via Dirichlet; no hard-coded constants)")
    console_show("E[u_x^4]", _frac_str(res["E_u4"]),  _mpf(res["E_u4"]))
    console_show("⟨H4²⟩",    _frac_str(res["H4_sq"]), _mpf(res["H4_sq"]))
    console_show("⟨H4³⟩",    _frac_str(res["H4_m3"]), _mpf(res["H4_m3"]))
    console_show("⟨H4⁴⟩",    _frac_str(res["H4_m4"]), _mpf(res["H4_m4"]))

    # JSON payload (runner-friendly; strings only for numerics — no Python floats)
    out = {
        "meta": make_meta(
            __file__,
            description="Exact Dirichlet-based moments for the cubic A1g quartic harmonic H4 on S^2.",
            ethos_note="CLI-only; no JSON reads; values derived from mainstream identities."
        ),
        "intermediates": {
            "moments": {
                # Exposed for upstream users (e.g., kappa.py wants E[u_x^4])
                "E_u4": _frac_str(res["E_u4"])
            }
        },
        "outputs": {
            "H4_sq": {
                "rational": _frac_str(res["H4_sq"]),
                "decimal_48": mp.nstr(_mpf(res["H4_sq"]), 48)
            },
            "H4_m3": {
                "rational": _frac_str(res["H4_m3"]),
                "decimal_48": mp.nstr(_mpf(res["H4_m3"]), 48)
            },
            "H4_m4": {
                "rational": _frac_str(res["H4_m4"]),
                "decimal_48": mp.nstr(_mpf(res["H4_m4"]), 48)
            },
        },
        "status": {"ok": True}
    }

    out_path = default_json_out(None, __file__)
    write_json(out_path, out)
    print(f"\nWrote JSON results to: {out_path}")

if __name__ == "__main__":
    main()
