#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — dependency-aware runner (single dict; deps inferred from inputs/outputs)

Usage:
  python run.py <script_name.py> [--force]
"""

from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from typing import Any, Dict, Tuple
import mpmath as mp

# shared argument-passing primitives & value parser
from run_utils import parse_input, split_root_sub_comp

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

# ------------------------------- CONFIG -------------------------------

CONFIG = {
  "scripts": {
    # ---- Protocol flags (window) ----
    "hypothesis.py": {
      "outputs": {
          "window": "outputs.window"
      },
      "inputs": []
    },

    # ---- Microtests → m ----
    "m.py": {
      "outputs": {"m_first": "outputs.m_first"},
      "inputs": [
      ]
    },

    "xq.py": {
      "outputs": {
        "x_q": "outputs.x_q", "L0": "outputs.L0",
        "fpp": "outputs.fpp", "f3": "outputs.f3", "f4": "outputs.f4"
      },
      "inputs": [
        {"flag": "--lock-ratio", "var": "lock_ratio"}   # pure χ from chitt.py
      ]
    },

    # ---- Spherical moments ----
    "h4.py": {
      "outputs": {
        "H4_sq":     "outputs.H4_sq",
        "E_u4_frac": "intermediates.moments.E_u4",
        "H4_m3":     "outputs.H4_m3",
        "H4_m4":     "outputs.H4_m4"
      },
      "inputs": []
    },

    # ---- Parity factor ----
    "zparity.py": {
      "outputs": {"z_parity": "outputs.z_parity"},
      "inputs": [
        {"flag": "--window", "var": "window"},
        {"flag": "--moment-power", "literal": "4"},
        {"flag": "--half-shift", "literal": "1"}
      ]
    },

    # ---- Quarter-hop Jacobian ----
    "j2.py": {
      "outputs": {"J2": "outputs.J2"},
      "inputs": []
    },

    # ---- Fejér main-lobe share (no JSON pass) ----
    "zdir.py": {
      "outputs": {"z_dir": "outputs.z_dir", "z_wedge": "outputs.z_wedge"},
      "inputs": [
        {"flag": "--num-sectors", "var": "num_orbits"}
      ]
    },

    # ---- Optional orbit counter ----
    "num_orbits.py": {
      "outputs": {"num_orbits": "outputs.num_orbits"},
      "inputs": []
    },

    # ---- York constant & χ_TT ----
    "cq.py": {
      "outputs": {"C_Qhat": "outputs.C_Qhat", "C_Q": "outputs.C_Q"},
      "inputs": []
    },

    # chitt now CONSUMES xi_TT (for visibility), gamma_TT, gamma_edge, C_A1g, C_Q
    "chitt.py": {
      "outputs": {"chi_tt": "outputs.chi_tt"},
      "inputs": [
        {"flag": "--C-Q",   "var": "C_Q.rational"}
      ]
    },
    
    # ---- Mid-edge parity quotient (derived, producer) ----
    "edge_parity.py": {
      "outputs": {"gamma_edge": "outputs.gamma_edge", "lock_ratio": "outputs.lock_condition.k_xq.rational"},
      "inputs": []
    },
    
    "a1g_xi_tt.py": {
      "outputs": {
        "C_A1g":      "outputs.C_A1g",
        "xi_TT":      "outputs.xi_TT",
        "gamma_TT":   "intermediates.gains.gamma_TT",
        "ratio_inner":"intermediates.legendre.ratio_axis_vs_P2sq"
      },
      "inputs": [
        {"flag": "--gamma-edge", "var": "gamma_edge.rational"}  # from edge_parity.py
      ]
    },

    # ---- Γ (product of factors; no JSON pass) ----
    "gamma.py": {
      "outputs": {"Gamma": "outputs.Gamma"},
      "inputs": [
        {"flag": "--z-parity", "var": "z_parity"},
        {"flag": "--j2",       "var": "J2"},
        {"flag": "--z-dir",    "var": "z_dir"},
        {"flag": "--c-a1g",    "var": "C_A1g"}
      ]
    },

    # ---- ΔL^(2) from first principles (no JSON pass) ----
    "dL2.py": {
      "outputs": {"deltaL2": "outputs.deltaL2"},
      "inputs": [
        {"flag": "--fpp",    "var": "fpp"},
        {"flag": "--H4-sq",  "var": "H4_sq"},
        {"flag": "--chi-tt", "var": "chi_tt"},
        {"flag": "--Gamma",  "var": "Gamma"}
      ]
    },

    "m4.py": {
      "outputs": {
        "M4": "outputs.M4",
        "Delta_M4_over_3_minus_E": "outputs.Delta_M4_over_3_minus_E",
        "slope_S": "outputs.slope_S"
      },
      "inputs": [
        {"flag": "--E-u4", "var": "E_u4_frac"},  # from h4.py
        # optionally pass dps/X-start/rounds/tail-count from your config
      ]
    },

    "kappa.py": {
      "outputs": {"kappa_infty": "outputs.kappa_infty"},
      "inputs": [
        {"flag": "--chi-tt", "var": "chi_tt"},  # from chitt.py (unsigned)
        {"flag": "--Delta",  "var": "Delta_M4_over_3_minus_E.decimal"},
        {"flag": "--dDelta", "var": "Delta_M4_over_3_minus_E.bounds.abs", "optional": True}
      ]
    },

    "alpha.py": {
      "outputs": {"alpha_inv": "outputs.alpha_inv"},
      "inputs": [
        {"flag": "--m",        "var": "m_first.int"},
        {"flag": "--kappa",    "var": "kappa_infty.float:24"},
        {"flag": "--dkappa",   "var": "kappa_infty.bounds.abs.float"},
        {"flag": "--x-q",      "var": "x_q.float:48"},
        {"flag": "--L0",       "var": "L0", "optional": True},
        {"flag": "--chi-tt",   "var": "chi_tt"},
        {"flag": "--deltaL2",  "var": "deltaL2"},
        {"flag": "--Gamma",    "var": "Gamma"},
        {"flag": "--f3",       "var": "f3.float:48"},
        {"flag": "--f4",       "var": "f4.float:48"},
        {"flag": "--H4-m3",    "var": "H4_m3.float:48"},
        {"flag": "--H4-m4",    "var": "H4_m4.float:48"},
        {"flag": "--J2",       "var": "J2"}
      ]
    }
  }
}

# ----------------------------- helpers (generic) -----------------------------

def jpath(script: str) -> Path:
    return OUT / (Path(script).with_suffix(".json").name)

def exists(p: Path) -> bool:
    try:
        return p.is_file() and p.stat().st_size > 0
    except Exception:
        return False

def load_json(p: Path) -> dict:
    return json.loads(p.read_text())

def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur[part]
    return cur

# ----------------------------- coercion for CLI -----------------------------

def _format_for_cli_from_node(node: Any, comp: str | None) -> str:
    """
    Single source of truth:
      - Always read with parse_input (intent-aware).
      - If no typed suffix: prefer exact rational "n/d"; else int; else decimal; else pass-through string.
      - If typed: enforce that type; symbolic strings are rejected for numeric types.
    """
    parsed = parse_input({"x": node})  # -> (n,d) | int | mpf | str

    def mpf_to_str(val: mp.mpf, digits: int = 24) -> str:
        return mp.nstr(val, digits)

    # Untyped: rational > int > float > string
    if comp is None:
        if isinstance(parsed, tuple):
            n, d = parsed
            return f"{int(n)}/{int(d)}"
        if isinstance(parsed, int):
            return str(parsed)
        if isinstance(parsed, str):
            return parsed
        return mpf_to_str(parsed, 24)

    # Typed: rational
    if comp == "rational":
        if isinstance(parsed, tuple):
            n, d = parsed
            return f"{int(n)}/{int(d)}"
        if isinstance(parsed, int):
            return f"{parsed}/1"
        raise SystemExit("[run.py] Requested '.rational' but source is not rational or int.")

    # Typed: int
    if comp == "int":
        if isinstance(parsed, int):
            return str(parsed)
        if isinstance(parsed, tuple):
            n, d = parsed
            if int(d) != 1:
                raise SystemExit("[run.py] Requested '.int' but rational has denominator != 1.")
            return str(int(n))
        if isinstance(parsed, str):
            raise SystemExit("[run.py] Requested '.int' but source is a non-numeric string.")
        if mp.floor(parsed) != parsed:
            raise SystemExit("[run.py] Requested '.int' but value is not an exact integer.")
        return str(int(parsed))

    # Typed: float (optionally with digits)
    if comp == "float" or comp.startswith("float:"):
        digits = 24
        if comp.startswith("float:"):
            try:
                digits = max(1, int(comp.split(":", 1)[1]))
            except Exception:
                digits = 24
        if isinstance(parsed, tuple):
            n, d = parsed
            return mpf_to_str(mp.mpf(n) / mp.mpf(d), digits)
        if isinstance(parsed, int):
            return mpf_to_str(mp.mpf(parsed), digits)
        if isinstance(parsed, str):
            raise SystemExit("[run.py] Requested '.float' but source is a non-numeric string.")
        return mpf_to_str(parsed, digits)

    raise SystemExit(f"[run.py] Unsupported component suffix: {comp!r}")

# ----------------------------- dependency engine -----------------------------

_JSON_CACHE: Dict[str, dict] = {}
_VAR_PRODUCER: Dict[str, str] = {}  # output root -> script
_VAR_PATH: Dict[str, str] = {}      # output root -> base json path
_VAR_CACHE: Dict[str, Any] = {}
_RUNNING: set[str] = set()

# index declared output ROOTS
for sname, spec in CONFIG["scripts"].items():
    for var_root, path in spec.get("outputs", {}).items():
        if var_root in _VAR_PRODUCER:
            raise SystemExit(
                f"[run.py] Variable root '{var_root}' produced by multiple scripts: "
                f"{_VAR_PRODUCER[var_root]} and {sname}"
            )
        _VAR_PRODUCER[var_root] = sname
        _VAR_PATH[var_root] = path

def _load_script_json(script: str) -> dict:
    if script not in _JSON_CACHE:
        ensure_ran(script, force=False)
        _JSON_CACHE[script] = load_json(jpath(script))
    return _JSON_CACHE[script]

def inferred_deps(script: str) -> list[str]:
    deps: set[str] = set()
    for item in CONFIG["scripts"][script].get("inputs", []):
        if "var" in item:
            v = item["var"]
            root, sub, comp = split_root_sub_comp(v)
            prod = _VAR_PRODUCER.get(root)
            if prod:
                deps.add(prod)
    return list(deps)

def resolve_var(var: str) -> Any:
    if var in _VAR_CACHE:
        return _VAR_CACHE[var]

    # split into root, subpath, and type component
    root, sub, comp = split_root_sub_comp(var)
    prod = _VAR_PRODUCER.get(root)
    if not prod:
        raise SystemExit(f"[run.py] Missing required value '{root}'. No producer.")

    data = _load_script_json(prod)
    base_path = _VAR_PATH[root]  # e.g. 'outputs.kappa_infty'
    full_path = base_path if sub is None else f"{base_path}.{sub}"

    node = get_by_path(data, full_path)
    if node is None:
        raise SystemExit(f"[run.py] Missing '{root}' in {jpath(prod)} at '{full_path}'.")

    val = _format_for_cli_from_node(node, comp)
    _VAR_CACHE[var] = val
    return val

def arg_value(item: dict) -> Any:
    if "literal" in item:
        return item["literal"]
    if "var" in item:
        return resolve_var(item["var"])
    return None

def build_args(script: str) -> list[str]:
    argv: list[str] = []
    for item in CONFIG["scripts"][script].get("inputs", []):
        flag = item["flag"]
        val  = arg_value(item)
        if (val is None or val == "") and item.get("optional", False):
            continue
        if val is None or val == "":
            src = ("var" in item and f"var:{item['var']}") or "literal"
            raise SystemExit(f"[run.py] Missing required CLI value for {flag} in {script} (source: {src}).")
        argv += [flag, str(val)]
    return argv

def explain_args(script: str, argv: list[str]) -> list[tuple[str,str]]:
    items = CONFIG["scripts"][script].get("inputs", [])
    pairs = []
    for i in range(0, len(argv), 2):
        flag = argv[i]
        item = next((it for it in items if it["flag"] == flag), None)
        if not item:
            pairs.append((flag, "unmapped"))
            continue
        if "literal" in item:
            src = f"literal:{item['literal']}"
        elif "var" in item:
            v = item["var"]
            root, sub, comp = split_root_sub_comp(v)
            prod = _VAR_PRODUCER.get(root, "<missing producer>")
            src = f"var:{v} from {prod}"
        else:
            src = "unknown"
        pairs.append((flag, src))
    return pairs

def ensure_ran(script: str, force: bool) -> None:
    pj = jpath(script)
    if exists(pj) and not force:
        print(f"[run.py] {script}: using cached {pj} (use --force to re-run)")
        return
    if script in _RUNNING:
        raise SystemExit(f"[run.py] Cycle detected while running {script}")
    _RUNNING.add(script)

    for dep in inferred_deps(script):
        ensure_ran(dep, force=False)

    argv = build_args(script)
    print(f"[run.py] {script} args:")
    for a, src in explain_args(script, argv):
        print(f"         {a:>16} <- {src}")
    cmd = ["python", str(SRC / script), *argv]
    run_cmd(cmd)
    if not exists(pj):
        raise SystemExit(f"[run.py] '{script}' ran but did not write its default JSON: {pj}")
    print(f"[run.py] wrote {pj}")
    _RUNNING.remove(script)

# --------------------------------- main ---------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python run.py <script_name.py> [--force]")
        sys.exit(2)
    target = sys.argv[1]
    force = (len(sys.argv) == 3 and sys.argv[2] == "--force")
    if target not in CONFIG["scripts"]:
        print(f"[run.py] Unknown script '{target}'. Add it under CONFIG['scripts'].")
        sys.exit(2)
    ensure_ran(target, force=force)
