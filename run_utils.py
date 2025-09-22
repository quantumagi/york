# run_utils.py
from __future__ import annotations
from typing import Any, Tuple, Optional
import re
import mpmath as mp
from fractions import Fraction

# generous working precision for parsing
mp.mp.dps = 120

# ----------------------------- coercion & precision -----------------------------

# Recognized CLI coercion suffixes for var paths
COERCION_SUFFIXES = (".float", ".int", ".decimal", ".rational", ".str")

# Optional precision-hinted float keys we recognize (highest digits win) for JSON leaves
_FLOAT_KEY_PATTERNS = [
    re.compile(r"^decimal(?:[_:](\d+))?$", re.I),  # decimal or decimal_24 / decimal:24
    re.compile(r"^float(?:[_:](\d+))?$", re.I),    # float or float_24 / float:24
    re.compile(r"^fixed[_:](\d+)$", re.I),         # fixed_18 / fixed:18
]

def strip_precision_hint(s: str) -> tuple[str, Optional[int]]:
    """Remove trailing ':NN' or '_NN' precision hint from a token, return (base, digits|None)."""
    if ":" in s:
        base, maybe = s.rsplit(":", 1)
        if maybe.isdigit():
            return base, int(maybe)
    if "_" in s:
        base, maybe = s.rsplit("_", 1)
        if maybe.isdigit():
            return base, int(maybe)
    return s, None

def normalize_varpath(v: str) -> str:
    """
    Remove trailing precision and coercion suffixes from a var path.
      'kappa_infty.bounds.abs.float:24' -> 'kappa_infty.bounds.abs'
      'kappa_infty.float_24'            -> 'kappa_infty'
    """
    v2, _ = strip_precision_hint(v)
    for suf in COERCION_SUFFIXES:
        if v2.endswith(suf):
            v2 = v2[: -len(suf)]
            break
    # strip a second time in case of suffix forms like '.float_24'
    v2, _ = strip_precision_hint(v2)
    return v2

def split_root_sub_comp(v: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Split a var spec into (root, subpath, component), where component is a coercion tag:
      'kappa_infty.bounds.abs.float:24' -> ('kappa_infty', 'bounds.abs', 'float:24')
      'kappa_infty.float:24'            -> ('kappa_infty', None, 'float:24')
      'kappa_infty'                     -> ('kappa_infty', None, None)
    """
    comp: Optional[str] = None
    base = v
    if "." in v:
        pre, last = v.rsplit(".", 1)
        if last in ("int", "float", "rational") or last.startswith("float:"):
            comp = last
            base = pre
    if "." in base:
        root, sub = base.split(".", 1)
    else:
        root, sub = base, None
    return root, sub, comp

# Back-compat: original signature used in older run.py paths (base+comp only)
def split_var_component(v: str) -> Tuple[str, Optional[str]]:
    """
    Legacy helper: return (base, component) where base includes any subpath.
      'kappa_infty.bounds.abs.float:24' -> ('kappa_infty.bounds.abs', 'float:24')
      'kappa_infty'                     -> ('kappa_infty', None)
    """
    root, sub, comp = split_root_sub_comp(v)
    base = root if sub is None else f"{root}.{sub}"
    return base, comp

# ----------------------------- JSON value parsing (unchanged) -----------------------------

def _unwrap_singleton(obj: Any) -> Any:
    """
    Allow parse_input({'x': node}) -> node for single-key wrappers.
    If dict has exactly one key, unwrap it; otherwise return as-is.
    """
    if isinstance(obj, dict) and len(obj) == 1:
        (k, v), = obj.items()
        return v
    return obj

def _looks_like_int_string(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return s.isdigit()

def _as_int_token(x: Any) -> int:
    """
    Interpret an exact integer token (int or int-looking string).
    Does NOT accept floats or decimal strings.
    """
    if isinstance(x, int) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, str) and _looks_like_int_string(x):
        return int(x.strip())
    raise ValueError("array rational must contain integer tokens")

def _parse_rational_from_leaf(leaf: Any) -> Tuple[int, int] | None:
    """
    Exact rational intent ONLY when explicitly provided as:
      - plain STRING containing '/', e.g. "8/15"
      - dict['rational'] or dict['fraction'] STRING containing '/'
      - 2-element ARRAY [n, d] (direct leaf or under 'rational'/'fraction'),
        where n and d are integers (or int-looking strings)

    NOTE:
      * Decimal strings (e.g. "0.5") are NOT rationals.
      * Plain integers (e.g. 7) are NOT rationals.
      * We keep (n, d) even if d == 1 (intent is rational).
    """
    # 1) plain "n/d" string
    if isinstance(leaf, str):
        s = leaf.strip()
        if "/" in s:
            fr = Fraction(s)
            return fr.numerator, fr.denominator
        return None

    # 2) direct 2-element array [n, d]
    if isinstance(leaf, list) and len(leaf) == 2:
        n = _as_int_token(leaf[0])
        d = _as_int_token(leaf[1])
        if d == 0:
            raise ValueError("rational denominator cannot be zero")
        fr = Fraction(n, d)
        return fr.numerator, fr.denominator

    # 3) dict forms: 'rational' / 'fraction'
    if isinstance(leaf, dict):
        for key in ("rational", "fraction"):
            val = leaf.get(key)
            if isinstance(val, str):
                s = val.strip()
                if "/" in s:
                    fr = Fraction(s)
                    return fr.numerator, fr.denominator
            if isinstance(val, list) and len(val) == 2:
                n = _as_int_token(val[0])
                d = _as_int_token(val[1])
                if d == 0:
                    raise ValueError("rational denominator cannot be zero")
                fr = Fraction(n, d)
                return fr.numerator, fr.denominator

    return None

def _parse_int_from_leaf(leaf: Any) -> int | None:
    """
    Integer intent ONLY when explicitly tagged with an 'int' key.
    Accepts integer or integer-looking string. Plain '123' (without 'int') is NOT int intent.
    """
    if isinstance(leaf, dict) and leaf.get("int") is not None:
        v = leaf["int"]
        if isinstance(v, int) and not isinstance(v, bool):
            return int(v)
        if isinstance(v, str) and _looks_like_int_string(v):
            return int(v.strip())
    return None

def _best_float_candidate(leaf: Any) -> str | float | int | None:
    """
    Choose the best float-like representation for mpf parsing.
    Preference:
      1) precision-hinted keys (decimal_30 > decimal_24; float:24 > float:16; fixed_18, …)
      2) simple float-ish keys: 'decimal', 'float', 'value', 'raw'
      3) explicit 'int' key (int or integer-looking string) as fallback numeric
      4) direct scalar leaf (int/float or decimal string)
    Returns the raw candidate; caller converts to mpf.
    """
    if isinstance(leaf, dict):
        # 1) precision-hinted candidates
        scored: list[tuple[int, Any]] = []
        for k, v in leaf.items():
            for pat in _FLOAT_KEY_PATTERNS:
                m = pat.match(k)
                if m and v is not None:
                    digits = int(m.group(1)) if m.group(1) else 0
                    scored.append((digits, v))
                    break
        if scored:
            scored.sort(key=lambda t: t[0], reverse=True)
            return scored[0][1]

        # 2) simple float-ish keys
        for k in ("decimal", "float", "value", "raw"):
            if k in leaf and leaf[k] is not None:
                return leaf[k]

        # 3) explicit int key as numeric fallback
        if leaf.get("int") is not None:
            v = leaf["int"]
            if isinstance(v, int) and not isinstance(v, bool):
                return v
            if isinstance(v, str) and _looks_like_int_string(v):
                return v

    # 4) direct scalar leaf
    if isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
        return leaf
    if isinstance(leaf, str) and "/" not in leaf.strip():
        return leaf.strip()

    return None

def parse_input(obj: Any) -> mp.mpf | Tuple[int, int] | int | str:
    """
    Parse a node (or single-key wrapper) into:
      • (n:int, d:int)  if explicit rational intent is provided (string 'n/d' or [n, d])
      • int             if explicit integer intent is provided via {'int': ...}
      • mp.mpf          otherwise (from best float/decimal candidate)
      • str             pass-through if numeric parsing fails (symbolic tokens)
    """
    leaf = _unwrap_singleton(obj)

    # 1) Explicit rational intent
    rat = _parse_rational_from_leaf(leaf)
    if rat is not None:
        fr = Fraction(rat[0], rat[1])  # normalize
        return fr.numerator, fr.denominator

    # 2) Explicit integer intent
    iv = _parse_int_from_leaf(leaf)
    if iv is not None:
        return iv

    # 3) Float/decimal path → mpf OR pass-through string
    cand = _best_float_candidate(leaf)
    if cand is None:
        raise ValueError(f"parse_input: could not parse value from {obj!r}")

    # robust mpf conversion or pass-through symbolic string
    if isinstance(cand, (int, float)) and not isinstance(cand, bool):
        return mp.mpf(repr(cand))
    if isinstance(cand, str):
        try:
            return mp.mpf(cand)  # numeric string → mpf
        except Exception:
            return cand          # non-numeric string (e.g., "sinc2") → pass through
    return mp.mpf(str(cand))

# Optional alias for convenience
def parse_inputs(obj: Any):
    return parse_input(obj)
