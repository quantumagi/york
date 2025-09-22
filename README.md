# The Scalar–Singlet Saddle (S³): Minimal Repro

Deterministic “theoretical experiment” scripts that produce the key
certificates/constants referenced in the paper.

- No network access required
- CPU only, fast, fully reproducible
- Outputs written to `outputs/`

## What this is
This repo contains a *single-file, reproducible pipeline* that ties together:
- **TT sector normalization \(\kappa_\infty\)** computed *ab initio* (Bessel-factorized baseline) with an independent **continuum “Route A” cross-check** (TT-complete via symmetry ×6, polar patch around the diagonal, and \(r_0\!\to\!0\) extrapolation).
- The **Fejér–York lock** that produces the effective length \(L_{\mathrm{eff}}\), the small parameter \(\varepsilon\), reduced mass \(m_r=1/\varepsilon\), and a prediction for \(\alpha^{-1}\) from the same rigidity pipeline.
- **Rigorous error control:** absolutely convergent expressions, analytic tail/remainder bounds, and guarded quadrature near the \(t_1\!\approx\!0\) singular manifold.
- **Ablations & diagnostics:** York toggle, parity “what-if”, sensitivity \(d(\alpha^{-1})/dL\), cached vs. computed \(\kappa_\infty\), and convergence/extrapolation scans.
- **Commutator experiments \(z_\chi\):** projector-based and true low-\(q\) commutator probes that *do not* alter the kinematic York coefficient (used only for diagnostics; CODATA is shown for context rather than as an input).

The goal is to blunt “numerology” critiques by (i) deriving every ingredient from the same \(L_{\mathrm{eff}}\)/rigidity scheme with explicit bounds, and (ii) providing an out-of-sector cross-check (Route A) that shares no discretization artifacts with the baseline.

## Highlights
- **One-file run:** No figures, just scalars suitable for the paper’s “Lock Universality and Ablations” section.
- **Two independent \(\kappa_\infty\) routes:**  
  - *Baseline:* Bessel-factorized TT integrals.  
  - *Route A:* Continuum integral with TT completion (×6), polar patch around \(t_2=t_3=0\), and linear \(r_0\)-extrapolation to remove the regulated core.
- **Fejér–York lock (series-exact):** \(L_0=\log(15/8)\), York increment \(11/1560\), odd/even tails evaluated with absolute convergence and rigorous bounds.
- **Precision on rails:** Configurable decimal precision for the lock and integrators; all tolerances are finite and guarded (no infinite loops).
- **Remainder bounds everywhere:** Poisson-summed/series tails and polar-patch contributions carry explicit, printed bounds.
- **Ablations you can trust:** York toggle, optional parity knob, and first-order \(\Delta L\) needed to match an external target (diagnostic only).
- **Sensitivity reporting:** Exact \(d(\alpha^{-1})/dL\) at the evaluated \(L_{\mathrm{eff}}\) to translate tiny theory shifts into ppb predictions.
- **Convergence dashboards:** GL node sweeps, Richardson-style extrapolation for Route A, and slope diagnostics.
- **Safe near singularities:** Open-interval and polar decompositions plus small-\(r\) analytic limits avoid division-by-zero traps.
- **Cache or compute:** Use the cached \(\kappa_\infty\) for quick runs or recompute both routes when you want a clean *ab initio* pass.
- **Transparent \(z_\chi\) probes:** Implements projector and true commutator measurements to *quantify* (not force) any mismatch; by design, York’s kinematic coefficient is unchanged by these probes.

## Requirements
- Python **3.11** (recommended)
- Packages pinned in `requirements.txt` (e.g. `mpmath`)

Install once:

```bash
python -m pip install -r requirements.txt
```

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run_all.cmd
```

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Manual sequence (same as run_all.cmd)
python run.py xq.py --force
python run.py m.py  --force    # run.py passes --xq from xq.py into m.py
python run.py kappa.py --force
python run.py a1g.py  --force
python run.py alpha.py --force
```

> Tip: Omit `--force` to re-use cached results in `outputs/`.

## Expected checks (sanity)

After a successful run you should see:

- `outputs/m.json`
  - `"m_first": 19`
  - Row for `m=19` shows `"ord2": 18`, `"dyadic_pass": true`, `"isolation_pass": true`
- `outputs/xq.json`
  - `x_q ≈ 1.3297024653297962…`
  - `L0 = log(15/8)`
- `outputs/alpha.json`
  - `alpha^{-1} ≈ 137.035999176279489…`
  - Internal uncertainty split consistent with the paper

## How things plug together

- `xq.py` computes the EL-lock abscissa `x_q` (Fejér lock).
- `m.py` **reads `--xq` via utils** and certifies the dyadic gate & isolation at the EL lock.
- `kappa.py` produces the Schlömilch/Bessel integral certificate `κ∞`.
- `a1g.py` and companions derive group-theoretic factors used in the master relation.
- `alpha.py` assembles everything to reproduce the predicted `α^{-1}`.

All scripts follow the same CLI contract and shared helpers in `utils.py`.

## Re-running & cache

Results are cached under `outputs/`. Use `--force` on any `run.py <script> --force`
invocation to recompute that step from scratch.

## Troubleshooting

- **Module not found**: activate the virtual environment (`. .\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate` on Unix) and reinstall requirements.
- **Different Python version**: prefer Python 3.11 to match expected numerics and JSON formatting.

## License

See `LICENSE` for code licensing. If a separate data license applies to files in `outputs/`, see the repository notes.