# The Scalar–Singlet Saddle (S³): Minimal Repro

Deterministic “theoretical experiment” scripts that produce the key
certificates/constants referenced in the paper.

- No network access required  
- CPU only, fast, fully reproducible  
- Outputs written to `outputs/` as JSON (strings for numerics; no Python floats)

## What this is

This repo contains a **multi-script, reproducible pipeline** that ties together:

- The **TT–sector normalization \( \kappa_\infty \)** computed *ab initio* via
  stabilized Schlömilch/Bessel integrals.
- The **Fejér–York lock** that produces the effective length \(L_{\mathrm{eff}}\),
  reduced mass \(m_r=e^{L_{\mathrm{eff}}/m}\), and a prediction for
  \( \alpha^{-1} = \kappa_\infty m/(e^{2L_{\mathrm{eff}}/m}-1)\).
- **Explicit error control:** absolutely convergent series/pieces with printed
  remainder bounds, and bracketed/bisected lock solution.
- **Ablations/diagnostics:** dyadic gate & isolation at the EL lock, parity and
  A1g factors assembled from small, auditable scripts.

> Scope note: this minimal repro implements the **baseline Bessel route** for
> \( \kappa_\infty \). There is **no separate “Route A” continuum cross-check** in
> this repo, and no commutator \(z_\chi\) experiments.

## Components (src/)

- `xq.py` — **Fejér quarter-FWHM lock** \(K(x_q)=8/15\) via rigorous bisection;
  emits `x_q`, `L0=log(15/8)`, and derivatives `fpp`, `f3`, `f4`.  
  *JSON carries high-precision decimal strings only (no float echoes).*
- `m.py` — **EL-lock isolation + dyadic gate** over odd `m`; reports first `m`
  that passes: oddness, dyadic richness proxy `ord_m(2) ≥ 18`, and Fejér isolation
  at `x_q`.
- `kappa.py` — **TT baseline certificate** for \( \kappa_\infty \) using exact
  stabilized Bessel kernels with √X tail extrapolation and an internal tail-span
  bound.
- `h4.py` — **Exact S² moments** for the cubic \(A_{1g}\) quartic harmonic
  \(H_4\): ⟨H4²⟩, ⟨H4³⟩, ⟨H4⁴⟩ (Dirichlet moments; rationals + high-precision echoes).
- `cq.py` — **York–quartic constant**:
  Monte-Carlo certificate for \(C_{\hat Q}\) constancy and the exact canonical
  \( C_Q = 11/780 \).
- `chitt.py` — **Linear-response scale** \( \chi_{TT} = 3\,C_Q \) (exact-fraction aware).
- `j2.py` — **Quarter-hop Jacobian²** certificate at mid-edge: emits `J2 = 4/3`
  (with small surgical checks; JSON carries the rational).
- `zparity.py` — **Parity factor** \( \zeta_{\text{parity}} = 8\,\mathbb{E}[u_x^2 u_y^2 u_z^2] \)
  via separated mp.mpf quadratures (reconstructs `1/105` → `8/105`).
- `edge_parity.py` — **Mid-edge parity/Jacobian quotient** \( \gamma_{\text{edge}}=15/16 \),
  also records the Fejér lock condition \(K(x_q)=8/15\).
- `num_orbits.py` — **“Five equal mid-edge sectors”**: enumerate 12 edge directions,
  collapse to 6 antipodal classes, and group into 5 parity sectors (canonical paper
  partition).
- `zdir.py` — **Main-lobe share from lock**:
  \( \zeta_{\text{dir}} = (3/5)\,R \), \( \zeta_{\text{wedge}} = \zeta_{\text{dir}}/N \)
  for lock ratio `R` (e.g. `8/15`) and sector count `N` (e.g. `5`).
- `a1g_xi_tt.py` — A1g/normalization helper used in the amplitude chain
  (emits the A1g projection coefficient consumed by `gamma.py`).
- `midedge_blend_ratio_mc.py` — Mid-edge blend ratio Monte Carlo (auxiliary).
- `gamma.py` — **Unified mid-edge amplitude**  
  \( \Gamma = \zeta_{\text{parity}} \cdot J_2 \cdot \zeta_{\text{dir}} \cdot C_{A1g} \)  
  (CLI-only product; exact reconstruction when possible).
- `dL2.py` — **Quadratic correction**  
  \( \Delta L^{(2)} = \tfrac12 f''(x_q)\,\langle H_4^2\rangle\,(\chi_{TT}\Gamma)^2 \).
- `alpha.py` — **Final assembly** of \(L_{\mathrm{eff}}\), \(m_r=e^{L_{\mathrm{eff}}/m}\),
  and \( \alpha^{-1} \); propagates Fejér-series remainders and a supplied absolute
  \( \delta\kappa \) into errors.
- `hypothesis.py` — Protocol/window helper.
- `utils.py` — Shared helpers: CLI parsing (`parse_number`), JSON writers, ledger
  printing, rational reconstruction, deterministic output filenames, etc.

## Runner (dependency-aware)

A lightweight orchestrator lives at the repo top-level:

- **`run.py`** — infers dependencies from a single **target** and from the
  inputs/outputs mapping; pulls required values from prior JSON artifacts; builds
  the exact CLI for each dependency; and runs only the missing pieces.

**Usage**

```bash
python run.py <script_name.py> [--force]
```

- `<script_name.py>` must be one of the scripts declared in `CONFIG["scripts"]`
  inside `run.py` (see the list above).
- `--force` re-runs the target even if its default JSON already exists.

**Typical invocations**

```bash
# End-to-end observables. Will run everything needed for alpha.json in order.
python run.py alpha.py

# Baseline κ∞ certificate (pulls χ_TT and E[u^4] from deps)
python run.py kappa.py

# Amplitude Γ (pulls z_parity, J2, z_dir, C_A1g)
python run.py gamma.py

# Freshen just the lock & curvature (and anything it depends on)
python run.py xq.py --force
```

**What the runner actually does**

- **Infers deps** by scanning the target’s declared `inputs` for variables whose
  **producer** is another script (e.g., `alpha.py` needs `m_first` → produced by `m.py`).
- **Extracts values** from the producers’ JSON using **dot paths** and optional
  **type components** (e.g., `kappa_infty.float:24`, `m_first.int`,
  `C_Q.rational`, `kappa_infty.bounds.abs.float`).
- **Builds flags** from the mapping and executes `python src/<script>.py ...`.
- **Caches** per-script JSON at `outputs/<script>.json`. If present, deps are
  skipped unless `--force` is supplied for the **target** (deps are still reused
  from cache).

## Requirements

- Python **3.11** recommended  
- `mpmath` (and any other packages listed in `requirements.txt`)

Install once:

```bash
python -m pip install -r requirements.txt
```

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# One command to generate final observables (and all prerequisites)
python run.py alpha.py
```

### Manual pipeline (advanced / explicit)

If you prefer to drive each step yourself:

```bash
# 1) Lock & curvature
python src/xq.py --auto-bracket --dps 80

# 2) Dyadic/isolation gate (pass the x_q decimal from outputs/xq.json)
python src/m.py --xq "<x_q_decimal>"

# 3) Core scalars
python src/h4.py
python src/cq.py
python src/chitt.py
python src/j2.py
python src/zparity.py
python src/edge_parity.py
python src/num_orbits.py
python src/zdir.py --lock-ratio 8/15 --num-sectors 5
python src/gamma.py --z-parity 8/105 --j2 4/3 --z-dir 8/25 --c-a1g "<C_A1g>"

# 4) κ∞ and corrections
python src/kappa.py --chi-tt 11/260 --E-u4 1/5
python src/dL2.py --fpp "<fpp>" --H4-sq 16/525 --chi-tt 11/260 --Gamma "<Gamma>"

# 5) Final observables (supply J2 or s=√J2, plus H4^3,H4^4 from h4.py)
python src/alpha.py --m 19 --kappa "<kappa>" --dkappa "<dkappa_abs>"   --x-q "<x_q>" --L0 "<L0>" --deltaL2 "<ΔL2>" --Gamma "<Gamma>"   --f3 "<f3>" --f4 "<f4>" --H4-m3 384/125125 --H4-m4 22784/10635625 --J2 4/3
```

> **Windows (PowerShell)**  
> Activate with `. .\.venv\Scripts\Activate.ps1` then:
> ```powershell
> python run.py alpha.py
> ```
> A convenience wrapper is also provided: `.
un_all.cmd`.

## Expected checks (sanity)

After a successful pass with defaults you should typically see:

- `outputs/m.json` ⇒ `"m_first": 19` (row for `m=19` shows `ord2: 18`,
  `dyadic_pass: true`, `isolation_pass: true`).
- `outputs/xq.json` ⇒ `x_q ≈ 1.3297024653…`, `L0 = log(15/8)`,
  and high-precision decimals for `fpp`, `f3`, `f4` (no float fields).
- `outputs/kappa.json` ⇒ a stable `κ_∞` value with a printed tail-span bound.
- `outputs/alpha.json` ⇒ \(m_r\), \( \alpha^{-1} \), and an error breakdown
  consistent with Fejér tails + \( \delta\kappa \).

## How things plug together

- `xq.py` → `x_q`, `L0`, `fpp`, `f3`, `f4`  
- `m.py`  → validates odd-`m` dyadic/isolation at `x_q`  
- `h4.py`, `cq.py`, `chitt.py`, `j2.py`, `zparity.py`, `edge_parity.py`,
  `num_orbits.py`, `zdir.py`, `a1g_xi_tt.py`, `midedge_blend_ratio_mc.py` → small audited factors  
- `gamma.py` → combines parity/Jacobian/direction/A1g into \( \Gamma \)  
- `kappa.py` → computes \( \kappa_\infty \) (baseline)  
- `dL2.py` → quadratic lock correction \( \Delta L^{(2)} \)  
- `alpha.py` → assembles \(L_{\mathrm{eff}}\), propagates errors, outputs \(m_r\), \( \alpha^{-1} \)  
- **`run.py`** → give it a **target** (e.g., `alpha.py`) and it will **infer and run** all prerequisites.

All scripts follow the same CLI contract and the shared helpers in `utils.py`.

## Re-running & cache

- Each script writes `outputs/<script>.json`.  
- `run.py` **reuses** those artifacts; to recompute a target from scratch, pass `--force`:
  ```bash
  python run.py xq.py --force
  ```

## Troubleshooting

- **Module not found**: activate the venv and (re)install requirements.  
- **Precision**: raise `--dps` (for `xq.py`) or tighten tolerances if you need finer brackets/series tails.  
- **Runner says “Missing required value”**: ensure the **producer** script for that value has written its JSON (or just invoke the higher-level target so `run.py` runs deps automatically).  
- **Help/usage**: `python run.py` with wrong args prints the usage banner.

## License

See `LICENSE` for code licensing. If a separate data license applies to files in
`outputs/`, see the repository notes.
