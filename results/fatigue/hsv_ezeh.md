# Modified HSV (Zhang et al. 2023) on every Ezeh-Susmel notched PLA specimen

L_CD 0.490 mm (authors' L_M at 2e6, fitted on Ezeh-Susmel), L_FCI 0.4 mm (ASSUMED), plain gauge 30 mm (ASSUMED): S_S 540 mm^2, V90_S 540 mm^3.

Life Weibull shape k from each plain series (2019 Table 1): p0 R-1 2.36, p0 R-0.5 1.47, p0 R0 0.90, p0 R0.3 2.25, p30 R-1 1.11, p30 R-0.5 1.06, p30 R0 0.84, p30 R0.3 0.68, p45 R-1 1.56, p45 R-0.5 1.21, p45 R0 1.56, p45 R0.3 0.73

| notch | peak sigma_1 / sigma_n (surface) | S90 (mm^2) | V90 (mm^3) | V_N = S90 L_CD (mm^3) | sigma_1(L_CD/2) / sigma_n |
|---|---|---|---|---|---|
| U r3 (blunt) | 1.622 | 14.814 | 2.4180 | 7.256 | 1.392 |
| U r1 | 2.628 | 3.951 | 0.1926 | 1.935 | 1.735 |
| V r0.15 (sharp) | 6.561 | 0.653 | 0.0051 | 0.320 | 1.789 |

Every failed specimen (149). Cells: bias / rms log10(N_pred/N); within x2; within x3; inside the predicted P_S 10-90 % band. Per notch: bias.

| method | all: bias / rms | within x2 | within x3 | in P10-P90 band | blunt | U r1 | V sharp |
|---|---|---|---|---|---|---|---|
| **plain S-N = 2019 Table 1 series at the same p and R** | | | | | | | |
| TCD point, authors' L_M(N) (their own method) | -0.42 / 0.65 | 40 % | 58 % | 53 % | -0.77 | -0.27 | -0.16 |
| TCD point, L_CD 0.49 (no size term) | -1.67 / 1.81 | 2 % | 3 % | 2 % | -1.73 | -1.55 | -1.73 |
| traditional HSV (V90, peak stress) | -0.56 / 1.49 | 12 % | 23 % | 35 % | -0.30 | +0.01 | -1.44 |
| modified HSV (L_FCI 0.4) | +0.01 / 0.91 | 30 % | 42 % | 44 % | -0.55 | +0.09 | +0.57 |
| variant, NOT in the paper: modified HSV with L_M(N) (L_FCI 0.4) | +0.47 / 0.86 | 23 % | 38 % | 44 % | -0.22 | +0.61 | +1.14 |
| **plain S-N = 2020 Fig. 1 unifying curve (P50 0.16 UTS at 2e6, k 5.1, T 2.451)** | | | | | | | |
| TCD point, authors' L_M(N) (their own method) | -0.37 / 0.66 | 42 % | 64 % | 78 % | -0.68 | -0.30 | -0.09 |
| TCD point, L_CD 0.49 (no size term) | -1.35 / 1.45 | 1 % | 5 % | 15 % | -1.37 | -1.30 | -1.36 |
| traditional HSV (V90, peak stress) | +2.59 / 2.72 | 0 % | 1 % | 7 % | +1.77 | +2.89 | +3.21 |
| modified HSV (L_FCI 0.4) | +1.75 / 2.01 | 5 % | 9 % | 34 % | +0.81 | +1.74 | +2.83 |
| variant, NOT in the paper: modified HSV with L_M(N) (L_FCI 0.4) | +1.65 / 1.91 | 9 % | 13 % | 38 % | +0.65 | +1.81 | +2.62 |

Sensitivity of the modified HSV to the two assumed inputs, plain S-N = 2019 Table 1 series at the same p and R (bias / rms log10 life, in P10-P90 band):

| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |
|---|---|---|---|
| 0.05 | -0.86 / 1.19, 27 % | -0.72 / 1.09, 30 % | -0.54 / 1.00, 36 % |
| 0.1 | -0.62 / 1.04, 33 % | -0.48 / 0.97, 37 % | -0.30 / 0.91, 44 % |
| 0.2 | -0.37 / 0.93, 43 % | -0.23 / 0.90, 45 % | -0.05 / 0.90, 46 % |
| 0.4 | -0.13 / 0.89, 46 % | +0.01 / 0.91, 44 % | +0.19 / 0.95, 43 % |
| 0.49 | -0.06 / 0.90, 45 % | +0.08 / 0.92, 43 % | +0.26 / 0.98, 40 % |
| 1 | +0.19 / 0.95, 43 % | +0.33 / 1.01, 40 % | +0.51 / 1.12, 33 % |

In-sample: L_FCI FITTED on these notched specimens (gauge 30): 0.278 mm, bias -0.12, rms 0.89, per notch blunt -0.68, intermediate -0.03, sharp +0.44

Sensitivity of the modified HSV to the two assumed inputs, plain S-N = 2020 Fig. 1 unifying curve (P50 0.16 UTS at 2e6, k 5.1, T 2.451) (bias / rms log10 life, in P10-P90 band):

| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |
|---|---|---|---|
| 0.05 | +0.15 / 1.01, 68 % | +0.41 / 1.08, 66 % | +0.74 / 1.24, 56 % |
| 0.1 | +0.59 / 1.16, 64 % | +0.85 / 1.31, 50 % | +1.18 / 1.55, 50 % |
| 0.2 | +1.04 / 1.44, 50 % | +1.30 / 1.64, 50 % | +1.63 / 1.91, 43 % |
| 0.4 | +1.49 / 1.79, 50 % | +1.75 / 2.01, 34 % | +2.08 / 2.30, 16 % |
| 0.49 | +1.62 / 1.90, 44 % | +1.88 / 2.13, 28 % | +2.21 / 2.42, 15 % |
| 1 | +2.08 / 2.30, 16 % | +2.34 / 2.54, 14 % | +2.67 / 2.85, 7 % |

In-sample: L_FCI FITTED on these notched specimens (gauge 30): 0.0264 mm, bias -0.00, rms 1.00, per notch blunt -0.94, intermediate -0.01, sharp +1.08
