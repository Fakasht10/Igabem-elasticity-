# Modified HSV (Zhang et al. 2023) on every Ezeh-Susmel notched PLA specimen

L_CD 0.490 mm (authors' L_M at 2e6, fitted on Ezeh-Susmel), L_FCI 0.4 mm (ASSUMED), plain gauge 30 mm (ASSUMED): S_S 540 mm^2, V90_S 540 mm^3.

Life Weibull shape k from each plain series (2019 Table 1): p0 R-1 2.36, p0 R-0.5 1.47, p0 R0 0.90, p0 R0.3 2.25, p30 R-1 1.11, p30 R-0.5 1.06, p30 R0 0.84, p30 R0.3 0.68, p45 R-1 1.56, p45 R-0.5 1.21, p45 R0 1.56, p45 R0.3 0.73

| notch | peak sigma_1 / sigma_n (surface) | S90 (mm^2) | V90 (mm^3) | V_N = S90 L_CD (mm^3) | sigma_1(L_CD/2) / sigma_n |
|---|---|---|---|---|---|
| U r3 (blunt) | 1.622 | 14.039 | 2.4180 | 6.876 | 1.392 |
| U r1 | 2.628 | 3.813 | 0.1926 | 1.868 | 1.735 |
| V r0.15 (sharp) | 6.561 | 0.636 | 0.0051 | 0.312 | 1.789 |

Every failed specimen (149). Cells: bias / rms log10(N_pred/N); within x2; within x3; inside the predicted P_S 10-90 % band. Per notch: bias.

| method | all: bias / rms | within x2 | within x3 | in P10-P90 band | blunt | U r1 | V sharp |
|---|---|---|---|---|---|---|---|
| **plain S-N = 2019 Table 1 series at the same p and R** | | | | | | | |
| blind Weibull (m from plain scatter, gauge volume) | -0.90 / 1.45 | 31 % | 40 % | 44 % | -0.56 | -0.37 | -1.84 |
| TCD point, authors' L_M(N) (their own method) | -0.42 / 0.65 | 40 % | 58 % | 53 % | -0.77 | -0.27 | -0.16 |
| TCD point, L_CD 0.49 (no size term) | -1.67 / 1.81 | 2 % | 3 % | 2 % | -1.73 | -1.55 | -1.73 |
| traditional HSV (V90, peak stress) | -0.56 / 1.49 | 12 % | 23 % | 35 % | -0.30 | +0.01 | -1.44 |
| modified HSV (L_FCI 0.4) | +0.02 / 0.91 | 30 % | 42 % | 44 % | -0.53 | +0.11 | +0.58 |
| variant, NOT in the paper: modified HSV with L_M(N) (L_FCI 0.4) | +0.48 / 0.86 | 24 % | 37 % | 44 % | -0.21 | +0.62 | +1.14 |
| **plain S-N = 2020 Fig. 1 unifying curve (P50 0.16 UTS at 2e6, k 5.1, T 2.451)** | | | | | | | |
| blind Weibull (m from plain scatter, gauge volume) | +0.40 / 0.74 | 40 % | 52 % | 87 % | +0.03 | +0.63 | +0.60 |
| TCD point, authors' L_M(N) (their own method) | -0.37 / 0.66 | 42 % | 64 % | 78 % | -0.68 | -0.30 | -0.09 |
| TCD point, L_CD 0.49 (no size term) | -1.35 / 1.45 | 1 % | 5 % | 15 % | -1.37 | -1.30 | -1.36 |
| traditional HSV (V90, peak stress) | +2.59 / 2.72 | 0 % | 1 % | 7 % | +1.77 | +2.89 | +3.21 |
| modified HSV (L_FCI 0.4) | +1.77 / 2.03 | 5 % | 9 % | 34 % | +0.85 | +1.76 | +2.85 |
| variant, NOT in the paper: modified HSV with L_M(N) (L_FCI 0.4) | +1.67 / 1.93 | 7 % | 14 % | 37 % | +0.69 | +1.83 | +2.64 |

Sensitivity of the modified HSV to the two assumed inputs, plain S-N = 2019 Table 1 series at the same p and R (bias / rms log10 life, in P10-P90 band):

| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |
|---|---|---|---|
| 0.05 | -0.85 / 1.17, 27 % | -0.70 / 1.08, 31 % | -0.53 / 0.99, 36 % |
| 0.1 | -0.60 / 1.03, 34 % | -0.46 / 0.96, 39 % | -0.28 / 0.91, 45 % |
| 0.2 | -0.36 / 0.93, 43 % | -0.22 / 0.90, 45 % | -0.04 / 0.90, 46 % |
| 0.4 | -0.12 / 0.89, 46 % | +0.02 / 0.91, 44 % | +0.20 / 0.96, 43 % |
| 0.49 | -0.05 / 0.90, 46 % | +0.09 / 0.92, 42 % | +0.27 / 0.98, 40 % |
| 1 | +0.20 / 0.96, 43 % | +0.34 / 1.02, 38 % | +0.52 / 1.12, 33 % |

In-sample: L_FCI FITTED on these notched specimens (gauge 30): 0.268 mm, bias -0.12, rms 0.89, per notch blunt -0.67, intermediate -0.03, sharp +0.43

Sensitivity of the modified HSV to the two assumed inputs, plain S-N = 2020 Fig. 1 unifying curve (P50 0.16 UTS at 2e6, k 5.1, T 2.451) (bias / rms log10 life, in P10-P90 band):

| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |
|---|---|---|---|
| 0.05 | +0.17 / 1.00, 68 % | +0.43 / 1.08, 68 % | +0.76 / 1.25, 56 % |
| 0.1 | +0.62 / 1.17, 64 % | +0.88 / 1.32, 50 % | +1.21 / 1.56, 50 % |
| 0.2 | +1.07 / 1.45, 50 % | +1.33 / 1.65, 50 % | +1.65 / 1.93, 41 % |
| 0.4 | +1.51 / 1.81, 49 % | +1.77 / 2.03, 34 % | +2.10 / 2.32, 16 % |
| 0.49 | +1.64 / 1.92, 41 % | +1.90 / 2.14, 27 % | +2.23 / 2.44, 15 % |
| 1 | +2.10 / 2.32, 16 % | +2.36 / 2.56, 13 % | +2.69 / 2.87, 7 % |

In-sample: L_FCI FITTED on these notched specimens (gauge 30): 0.0255 mm, bias -0.00, rms 0.99, per notch blunt -0.93, intermediate -0.01, sharp +1.07
