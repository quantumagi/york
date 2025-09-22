#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py — shared CLI/number/JSON helpers (ETHOS-compliant, no JSON reads)

ETHOS
  • Scripts consume ONLY CLI flags. No JSON file reads inside helpers.
  • No embedded theory constants. Pure plumbing/formatting utilities.
  • No Python floats in the public API: use mp.mpf / Fraction / str / int.
  • Back-compat function names retained (e.g., fixed_places, decimal_from_cli),
    but implemented with mpmath instead of Decimal.

WHAT THIS PROVIDES
  Parsing & Numbers
    - parse_number("8/105")  -> ParsedNumber(raw="8/105", rational="8/105",
                                             float=mpf)
    - parse_number("0.32")   -> ParsedNumber(raw="0.32", rational=None,
                                             float=mpf)
    - decimal_from_cli("21/64") → mp.mpf("0.328125")  (compat shim)
    - pretty_fraction_or_none(mpf_value) → "1/3" (cosmetic)
    - fixed_places(x, places)  → "0.273079700750000000" (mpf rounding)

  JSON I/O (write only) & Meta
    - default_json_out(args.json_out, __file__)
    - write_json(path, payload)
    - make_meta(__file__, description="...", ethos_note="...")

  Console Ledger
    - ledger_header("Title")
    - console_show("J2", rat_tag, value)  # aligned name/value/[p/q] tag
"""

from __future__ import annotations

import json
import sys
import platform
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Iterable, Tuple, Dict

import mpmath as mp

# --------------------------- precision & cosmetics ---------------------------

RAT_RECON_MAX_DEN = 1_000_000  # cosmetic rational reconstruction bound

# ------------------------------- data classes --------------------------------

@dataclass(frozen=True)
class ParsedNumber:
    """Uniform representation of a CLI-provided scalar (no Python floats)."""
    raw: str                 # original CLI string as passed on CLI
    rational: Optional[str]  # "p/q" if provided or reconstructed, else None
    float: mp.mpf            # mpmath high-precision value
    fraction: Optional[Fraction]  # Fraction if rational is not None, else None

# ------------------------------- num parsing ---------------------------------

def parse_rat_or_float(s: str) -> Tuple[Optional[str], mp.mpf]:
    """
    Accept 'p/q' or a decimal-like string; return (rational_str_or_None, mpf_value).
    NOTE: returns mp.mpf (exact via string/ratio). Python floats are NOT used.
    """
    mp.mp.dps = 200  # high precision for intermediate parsing
    t = str(s).strip()
    if "/" in t:
        fr = Fraction(t)  # exact rational
        val = mp.mpf(fr.numerator) / mp.mpf(fr.denominator)  # no float round-trip
        return f"{fr.numerator}/{fr.denominator}", val
    return None, mp.mpf(t)

def parse_number(s: str, *, reconstruct_if_float: bool = False) -> ParsedNumber:
    """
    Parse a CLI scalar into (raw, rational tag, mpf).
    If reconstruct_if_float=True (name kept for compatibility), attempt a
    cosmetic rational reconstruction when input was not "p/q".
    """
    rat, mpv = parse_rat_or_float(s)
    if not mp.isfinite(mpv):
        raise ValueError(f"Non-finite value after parsing {s!r}")
    if rat is None and reconstruct_if_float:
        rat = pretty_fraction_or_none(mpv)
    if rat:  # ensure canonical form
        fr = Fraction(rat)
    else:
        fr = None
    return ParsedNumber(raw=s, rational=rat, float=mpv, fraction=fr)

# ----------------------------- formatting helpers ----------------------------

def _format_mpf_fixed(x: mp.mpf, places: int) -> str:
    """
    Fixed-point decimal string with exactly `places` digits after the dot.
    Rounds to nearest (ties handled by mp.nint's policy).
    """
    if not isinstance(x, mp.mpf):
        x = mp.mpf(x)
    scale = mp.mpf(10) ** places
    n = mp.nint(x * scale)  # nearest integer
    n_int = int(n)          # safe: n is an integer mpf
    sign = "-" if n_int < 0 else ""
    n_abs = abs(n_int)
    s = str(n_abs)
    if places == 0:
        return f"{sign}{s}"
    # left-pad with zeros if needed
    if len(s) <= places:
        s = "0" * (places - len(s) + 1) + s
    head, tail = s[:-places], s[-places:]
    return f"{sign}{head}.{tail}"

def fixed_places(x: mp.mpf | str | int | Fraction, places: int) -> str:
    """
    Convert x to mpf and return a fixed-point string with `places` digits
    after the decimal point using mpmath rounding.
    """
    return _format_mpf_fixed(mp.mpf(x), places)

# ---------------------------- cosmetic rational tags -------------------------

def _best_rational_approx(v: mp.mpf, max_den: int) -> Optional[Fraction]:
    """
    Continued-fraction convergents for the best rational approximation to v
    with denominator <= max_den. Returns None if v is not finite.
    """
    if not mp.isfinite(v):
        return None
    x = mp.mpf(v)
    # Handle sign
    sgn = -1 if x < 0 else 1
    x = mp.fabs(x)
    # Initialize convergents
    a0 = mp.floor(x)
    p0, q0 = int(a0), 1
    p1, q1 = 1, 0
    # If already integer
    if x == a0:
        return Fraction(sgn * p0, q0)
    # Continued fraction loop
    while True:
        a = int(mp.floor(x))
        pn, qn = a * p0 + p1, a * q0 + q1
        if qn > max_den:
            break
        p1, q1 = p0, q0
        p0, q0 = pn, qn
        # next term
        frac = x - a
        if frac == 0:
            break
        x = 1 / frac
    return Fraction(sgn * p0, q0)

def pretty_fraction_or_none(x: mp.mpf | str | Fraction | int,
                            max_den: int = RAT_RECON_MAX_DEN) -> Optional[str]:
    """
    Try to represent x as 'p/q' exactly (or within tiny tolerance).
    Accepts mpf/str/Fraction/int. Python floats are rejected.
    Cosmetic only — does not affect computation.
    """
    if isinstance(x, float):
        raise TypeError("Python float is not allowed. Use mp.mpf instead.")
    try:
        v = mp.mpf(x)
    except Exception:
        return None
    if not mp.isfinite(v):
        return None
    fr = _best_rational_approx(v, max_den)
    if fr is None:
        return None
    # accept if very close
    approx = mp.mpf(fr.numerator) / mp.mpf(fr.denominator)
    tol = mp.mpf('1e-60') * (1 + mp.fabs(v))
    if mp.fabs(approx - v) <= tol:
        return f"{fr.numerator}/{fr.denominator}"
    return None

# ------------------------------ JSON utilities -------------------------------

def default_json_out(arg: Optional[str], script_file: str) -> Path:
    """
    Compute the output JSON path following the project convention:
      - if arg is provided, use it (create parent dirs)
      - else write to outputs/<script_basename>.json
    """
    if arg:
        p = Path(arg)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / (Path(script_file).with_suffix(".json").name)

def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    """
    Deterministically write JSON (sorted keys, 2-space indent).
    (Note: callers should ensure payload numbers are JSON-serializable —
     typically strings for high-precision decimals.)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def make_meta(script_file: str, *, description: Optional[str] = None,
              ethos_note: Optional[str] = None) -> Dict[str, Any]:
    """
    Standard meta block with script name, runtime, and optional description/ethos note.
    """
    return {
        "schema_version": "1.0",
        "script": Path(script_file).name,
        "run_env": {"python": sys.version.split()[0], "platform": platform.platform()},
        **({"description": description} if description else {}),
        **({"ethos": ethos_note} if ethos_note else {}),
    }

# ----------------------------- console formatting ----------------------------

def ledger_header(title: str) -> None:
    print(f"\n=== {title} ===")

def console_show(name: str, rational_tag: Optional[str], value: mp.mpf, width: int = 12) -> None:
    """
    Pretty console line for a factor: right-aligned name, value, and a [p/q] tag if present.
    Formats via mpmath to avoid binary-float artifacts. Python floats are not accepted.
    """
    if isinstance(value, float):
        raise TypeError("Python float is not allowed. Use mp.mpf instead.")
    v = mp.mpf(value)
    v_str = mp.nstr(v, 24)  # 24 sig figs default for console summaries
    tag = rational_tag or pretty_fraction_or_none(v) or "-"
    print(f"{name:>{width}} : {v_str}   [{tag}]")

# ------------------------------ validations ----------------------------------

def ensure_finite(name_value_pairs: Iterable[Tuple[str, mp.mpf]]) -> None:
    """
    Assert all values are finite (mpf); raise ValueError otherwise.
    """
    for name, v in name_value_pairs:
        if isinstance(v, float):
            raise TypeError(f"Python float is not allowed for {name}. Use mp.mpf instead.")
        try:
            vv = mp.mpf(v)
        except Exception as e:
            raise ValueError(f"Non-numeric value for {name}: {v!r}") from e
        if not mp.isfinite(vv):
            raise ValueError(f"Non-finite value for {name}: {v!r}")

def require(condition: bool, message: str) -> None:
    """
    Fail loudly with a clear message if a required condition is not met.
    """
    if not condition:
        raise ValueError(message)

# ---------------------------- back-compat shims -------------------------------

def decimal_from_cli(s: str) -> mp.mpf:
    """
    Compatibility shim: returns mp.mpf parsed from 'p/q' or decimal string.
    """
    t = str(s).strip()
    if "/" in t:
        p, q = t.split("/", 1)
        return mp.mpf(p) / mp.mpf(q)
    return mp.mpf(t)

def decimal_precision(_prec: int):
    """
    No-op context manager for callers that used decimal_precision().
    Kept to avoid churn; all math is done with mpmath's own precision.
    """
    class _Noop:
        def __enter__(self): return None
        def __exit__(self, *exc): return False
    return _Noop()

def to_decimal(x) -> mp.mpf:
    """
    Compatibility shim: returns mp.mpf. (Avoid Python floats.)
    """
    if isinstance(x, float):
        raise TypeError("Python float is not allowed. Use mp.mpf instead.")
    return mp.mpf(x)

def mpf_to_rational_str(x, *, abs_tol='1e-24', rel_tol='1e-24',
                        max_den=10**9, max_terms=1000):
    """
    Try to reconstruct an exact rational p/q for mp.mpf x using continued fractions.
    Returns "p/q" (with integers) if |p/q - x| <= max(abs_tol, rel_tol*|x|) and q <= max_den,
    else returns None. Only mpf + Python ints are used (no Decimal/float).
    """
    x = mp.mpf(x)
    if not mp.isfinite(x):
        return None

    # tolerances (mpf)
    t_abs = mp.mpf(abs_tol)
    t_rel = mp.mpf(rel_tol)
    ax = mp.fabs(x)
    if ax == mp.mpf('0'):
        return "0/1"

    # near-integer fast path
    n = int(mp.nint(x))
    if mp.fabs(x - n) <= max(t_abs, t_rel*ax):
        return f"{n}/1"

    # work with positive value; track sign
    sgn = -1 if x < 0 else 1
    y = ax

    # continued fraction initial term
    a0 = int(mp.floor(y))
    # convergents:
    # p[-2]=1, p[-1]=a0 ; q[-2]=0, q[-1]=1
    p_nm2, p_nm1 = 1, a0
    q_nm2, q_nm1 = 0, 1

    # best-so-far (to allow q>max_den early exit)
    p_best, q_best = p_nm1, q_nm1
    err_best = mp.fabs(mp.mpf(p_best)/q_best - ax)

    # fractional part
    frac = y - a0

    for _ in range(max_terms):
        if frac == 0:
            # y is rational; current convergent is exact
            p, q = p_nm1, q_nm1
            p *= sgn
            return f"{p}/{q}"

        y = 1/frac
        a = int(mp.floor(y))

        # next convergent
        p = a * p_nm1 + p_nm2
        q = a * q_nm1 + q_nm2

        # update error / best
        err = mp.fabs(mp.mpf(p)/q - ax)
        if err < err_best:
            err_best, p_best, q_best = err, p, q

        # success?
        if err <= max(t_abs, t_rel*ax) and q <= max_den:
            p *= sgn
            return f"{p}/{q}"

        # stop if denominator too large (keep best)
        if q > max_den:
            break

        # shift window
        p_nm2, p_nm1 = p_nm1, p
        q_nm2, q_nm1 = q_nm1, q
        frac = y - a

    # fall back to best if it meets tolerance
    if err_best <= max(t_abs, t_rel*ax) and q_best <= max_den:
        p_best *= sgn
        return f"{p_best}/{q_best}"
    return None
