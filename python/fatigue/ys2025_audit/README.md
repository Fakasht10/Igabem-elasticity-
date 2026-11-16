# Independent audit of the Yaren–Susmel crack-rule fits (2026-11-16)

A subagent that was given the code and data but not the conclusions audited the review's `rank.py` and the extensions
`gamma_chi.py`, `rminus.py`, `kchi.py` and `lfree.py`, which live in `/mnt/project-files/yaren-susmel-2025/`. It refit
every variant from scratch with its own loader, fields, Gauss weight-function quadrature, likelihood and a multi-start
L-BFGS-B / Powell optimiser. Files: `fit.py`, `fields.py`, `run_*.py`, `their_obj.py`, `restat.py`, `xcheck.py`,
`summarize.py`. Results are in `res_*.json` and `refit.json`.

## Findings

1. The field is mirrored periodically and the crack size a is not capped (`rank.py`), so a > ligament is non-physical.
   This opens a second likelihood mode at a ≈ 23 mm, with AIC about 20 lower for k(χ) and β. That mode is inadmissible.
2. The 24-point midpoint rule for the weight-function mean is 2–3 % low for V at a = 3–6 mm, which shifts AIC by about 1.
3. With plain + V as the only fitted geometries, the γ(χ) and β drivers cannot be told apart by the fitted likelihood.
4. The stored statistics reproduce exactly from the stored parameters.

The suffixes `_std`, `_wide` and `_con` mean the declared bounds, widened bounds, and a ≤ 6 mm respectively.
