# quick_check.py
import json, sys, pathlib as p
root = p.Path("outputs")
expect = {
  "fejer_lock_alpha.json": ("alpha_inv_final", None, 137.0359991762865, 5e-10),
  "fejer_lock_sensitivity.json": ("ALL_PASSED", None, True, None),  
  "fejer_quarter_fwhm.json": ("fpp_xq", None, 0.9897581823707557, 1e-15),  
  "kappa_infty_solver.json": ("kappa_infty", None, 0.499268512001529552, 5e-18),  
  "midedge_dirichlet_gamma.json": ("delta_L2", None, 3.071538094072599e-09, 1e-15),
  "spherical_H4_norm.json": ("passed", None, True, None),
  "york_constant_axis.json": ("all_passed", None, True, None),
}
ok = True
for fname,(k,sub,ref,tol) in expect.items():
    f = root/fname
    if not f.exists():
        print("MISSING", f); ok=False; continue
    data = json.loads(f.read_text())
    val = data[k] if sub is None else data[k][sub]
    if tol is None:
        if val is not True: print("FAIL", f, k, val); ok=False
    else:
        if not (abs(val-ref) <= tol): print("FAIL", f, k, val, "≠", ref); ok=False
print("ALL PASS" if ok else "SOME FAIL"); sys.exit(0 if ok else 1)
