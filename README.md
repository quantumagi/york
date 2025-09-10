# York–Fejér Quadratic Lock (ΔL²) — Minimal Repro

This repo contains the handful of deterministic “theoretical experiment” scripts that
cross-check each constant used in the paper. No makefiles, no certificates.

## Quick start
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run_all.ps1
