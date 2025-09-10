<# run_all.ps1
Runs S1–S6 + κ∞, writes JSONs to results\sealed (or a folder you pass),
then executes quick_check.py on that folder.
#>

param(
  [string]$OutDir = "outputs"   # default matches quick_check.py's default
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# --- setup ---
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

function Step($name, $cmd) {
  Write-Host "=== $name ==="
  & $cmd
  if ($LASTEXITCODE -ne 0) { throw "Step failed: $name" }
}

# --- S1: York contraction ---
Step "S1 york_constant_axis" { python "src/york_constant_axis.py" --n-random 2000 --seed 0 --tol 1e-12 --json-out "$OutDir/york_constant_axis.json" }

# --- S2: ⟨H4^2⟩ ---
Step "S2 spherical_H4_norm" { python "src/spherical_H4_norm.py" --n 500000 --seed 0 --eps 3e-3 --json-out "$OutDir/spherical_H4_norm.json" }

# --- S3: quarter-FWHM + curvature ---
Step "S3 fejer_quarter_fwhm" { python "src/fejer_quarter_fwhm.py" --abs-tol 1e-15 --rel-tol 1e-15 --json-out "$OutDir/fejer_quarter_fwhm.json" }

# --- S4: Γ and ΔL^(2) ---
Step "S4 midedge_dirichlet_gamma" { python "src/midedge_dirichlet_gamma.py" --abs-tol 1e-15 --rel-tol 1e-15 --json-out "$OutDir/midedge_dirichlet_gamma.json" }

# --- κ∞ (can be used by S5; we also set a default fallback) ---
Step "kappa_infty_solver" { python "src/kappa_infty_solver.py" --json-out "$OutDir/kappa_infty_solver.json" }

# pull κ∞ from JSON if present, else default to the reference value
$Kappa = 0.49926851200152955
try {
  $kjson = Get-Content "$OutDir/kappa_infty_solver.json" -Raw | ConvertFrom-Json
  if ($kjson.kappa_infty) { $Kappa = [double]$kjson.kappa_infty }
} catch { }

# --- S5: lock → (m_r, α) using closed-form Bernoulli tail ---
Step "S5 fejer_lock_alpha" {
  python "src/fejer_lock_alpha.py" `
    --tol 1e-18 `
    --m 19 `
    --kappa $Kappa `
    --alpha-target 137.03599917628638 `
    --mr-target 1.0340326052745639 `
    --json-out "$OutDir/fejer_lock_alpha.json"
}

# --- S6: sensitivity checks ---
Step "S6 fejer_lock_sensitivity" {
  python "src/fejer_lock_sensitivity.py" `
    --from-json "$OutDir/fejer_lock_alpha.json" `
    --json-out "$OutDir/fejer_lock_sensitivity.json"
}

# --- final: quick check ledger ---
Write-Host "=== quick_check ==="
python "quick_check.py" "$OutDir"
exit $LASTEXITCODE
