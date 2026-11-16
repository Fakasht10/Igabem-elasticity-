Step 2: every method calibrated the paper's way (plain P50 + sharp V Fig. 8a curve), U r1 and U r3 held out.

    python -m es2020.compare [out_dir]   -> compare_methods.md, fig_compare_methods.png/pdf

Stresses: sigma_n,max of each specimen as plotted in Fig. 8 (the paper's chain). Residual = log10(sigma_eq / plain P50
at the specimen's life), failures only (8 run-outs left out). Band = inside the plain P10-P90 band.

| method | constants (fit on plain + sharp V) | fitted sharp V: bias / rms / slope / in band | held-out U r1 | held-out U r3 | held-out both: rms / in band |
|---|---|---|---|---|---|
| Net stress, no method | none | -0.008 / 0.125 / -0.080 / 89% | -0.002 / 0.119 / -0.017 / 94% | +0.124 / 0.158 / +0.051 / 74% | 0.141 / 83% |
| Stress gradient (Siebel-Stieler support) | K_t / (1 + sqrt(c chi)), c = 38.2 N^-0.282 mm | +0.003 / 0.105 / +0.037 / 96% | -0.064 / 0.150 / +0.075 / 75% | -0.010 / 0.150 / +0.126 / 81% | 0.150 / 78% |
| TCD Point Method in SWT variable | SWT variable (not in paper); L_M = 8.25 N^-0.168 mm | +0.001 / 0.050 / +0.008 / 100% | +0.066 / 0.099 / +0.046 / 85% | +0.183 / 0.205 / +0.081 / 31% | 0.164 / 57% |
| Net stress x one constant fitted on sharp V | sigma_n,max x 1.1 | +0.035 / 0.130 / -0.080 / 89% | +0.040 / 0.126 / -0.017 / 96% | +0.166 / 0.193 / +0.051 / 59% | 0.165 / 76% |
| TCD Line Method (Eq. 3) | L = 14.7 N^-0.238 mm | +0.036 / 0.107 / -0.006 / 96% | +0.041 / 0.137 / +0.049 / 85% | +0.144 / 0.189 / +0.089 / 70% | 0.167 / 77% |
| Averaged SED (Lazzarin-Berto), constant R0 | R0 = 1.07 mm (constant) | +0.037 / 0.131 / -0.080 / 91% | +0.033 / 0.124 / -0.017 / 96% | +0.175 / 0.201 / +0.051 / 59% | 0.169 / 76% |
| Weibull weakest link (whole specimen volume) | whole volume; m 3.44 FIXED (plain band), V0 130 mm^3 | +0.035 / 0.130 / -0.080 / 89% | +0.039 / 0.125 / -0.017 / 96% | +0.175 / 0.201 / +0.051 / 59% | 0.170 / 76% |
| Averaged SED (Lazzarin-Berto), R0 life-dependent | R0 = 17.5 N^-0.260 mm (life-dependent, as L_M) | +0.003 / 0.104 / +0.035 / 96% | +0.023 / 0.154 / +0.103 / 81% | +0.133 / 0.208 / +0.137 / 67% | 0.184 / 74% |
| TCD Point Method (Eq. 2) | L_M = 15.8 N^-0.237 mm | +0.005 / 0.102 / +0.032 / 96% | +0.070 / 0.158 / +0.085 / 77% | +0.172 / 0.219 / +0.109 / 61% | 0.193 / 69% |
| TCD Area Method (Eq. 4) | L = point L_M = 15.8 N^-0.237 mm (not refitted, as the paper) | +0.044 / 0.110 / +0.026 / 89% | +0.079 / 0.161 / +0.082 / 75% | +0.178 / 0.225 / +0.111 / 57% | 0.197 / 66% |
| Modified HSV (Zhang et al. 2023) | C_S 1.6 mm^3; FIXED: L_CD 0.504 mm (point L_M at 2e6), k_N 0.675 (plain band) | +0.035 / 0.130 / -0.080 / 89% | +0.256 / 0.282 / -0.017 / 46% | +0.454 / 0.464 / +0.051 / 0% | 0.389 / 22% |

SWT row is a departure from the paper and not like-for-like: its plain curve is our refit of the Fig. 1 points in sigma_SWT (T 1.96, narrower than 2.451), so its rms and in-band share refer to a different band.
