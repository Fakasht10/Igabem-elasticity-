Step 2: every method calibrated the paper's way (plain P50 + sharp V Fig. 8a curve), U r1 and U r3 held out.

    python -m es2020.compare [out_dir]   -> compare_methods.md, fig_compare_methods.png/pdf

Stresses: sigma_n,max of each specimen as plotted in Fig. 8 (the paper's chain). Residual = log10(sigma_eq / plain P50
at the specimen's life), failures only (8 run-outs left out). Band = inside the plain P10-P90 band.

| method | constants (fit on plain + sharp V) | fitted sharp V: bias / rms / slope / in band | held-out U r1 | held-out U r3 | held-out both: rms / in band |
|---|---|---|---|---|---|
| Net stress (no method) | none | -0.008 / 0.125 / -0.080 / 89% | -0.002 / 0.119 / -0.017 / 94% | +0.124 / 0.158 / +0.051 / 74% | 0.141 / 83% |
| TCD Line Method (Eq. 3) | L = 14.7 N^-0.238 mm | +0.049 / 0.133 / -0.030 / 89% | +0.041 / 0.137 / +0.049 / 85% | +0.146 / 0.190 / +0.085 / 70% | 0.167 / 77% |
| Weibull weakest link (volume) | m 3.44 (plain band), V0 130 mm^3 | +0.035 / 0.130 / -0.080 / 89% | +0.039 / 0.125 / -0.017 / 96% | +0.175 / 0.201 / +0.051 / 59% | 0.170 / 76% |
| TCD Area Method (Eq. 4) | L = 25.8 N^-0.265 mm | +0.006 / 0.102 / +0.029 / 96% | +0.045 / 0.151 / +0.089 / 83% | +0.152 / 0.211 / +0.120 / 65% | 0.185 / 74% |
| TCD Point Method (Eq. 2) | L_M = 15.8 N^-0.237 mm | +0.005 / 0.102 / +0.032 / 96% | +0.070 / 0.158 / +0.085 / 77% | +0.172 / 0.219 / +0.109 / 61% | 0.193 / 69% |
| Modified HSV (Zhang et al. 2023) | C_S 1.6 mm^3, L_CD 0.504 mm | +0.035 / 0.130 / -0.080 / 89% | +0.256 / 0.282 / -0.017 / 46% | +0.454 / 0.464 / +0.051 / 0% | 0.389 / 22% |
