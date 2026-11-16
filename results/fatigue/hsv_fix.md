# Modified HSV (Zhang et al. 2023) on Ezeh-Susmel PLA: diagnosis and best honest version

Protocol of fatigue.calib: fitted on plain (Fig. 1 P50) + 47 failed sharp V; held out 102 failed U r1 + U r3. Residual log10(S_eff / P50(N)); band = P10-P90 of T 2.451.

S90 (trapezoid weights, hsv.S90_surface): V r0.15 0.636, U r1 3.813, U r3 14.039 mm^2; sigma_1(L_CD/2)/sigma_n: V r0.15 1.789, U r1 1.735, U r3 1.392.

Life Weibull shape k (plain data unless marked) and the stress exponent e = 1/(5.1 k) of the size term:

- band T 2.451 (Fig. 1, as in hsv.py): k 0.675, e 0.291
- Fig. 1 plain points, Weibull MLE (paper's procedure): k 0.594, e 0.330
- 2019 series pooled: k 1.111, e 0.176
- sharp-V life scatter (NOT paper's source): k 3.351, e 0.059
- 2019 series at own p, R: k 0.68-2.36, e 0.083-0.286

## Root cause: decomposition relative to the sharp V (log10 of the stress factor)

| notch | point-method part | size part, band k (e 0.290) | size part, series-pooled k | what the data need (held-out, diagnosis only) |
|---|---|---|---|---|
| U r1 | -0.013 | +0.226 | +0.137 | -0.000 |
| U r3 | -0.109 | +0.391 | +0.237 | -0.111 |

## Every variant tried (all fitted on plain + sharp V only)

| k source | L_FCI / L_CD | fitted values | sharp V (fitted): bias / rms | U r1 held out: bias / rms | U r3 held out: bias / rms | HELD OUT: rms, in band |
|---|---|---|---|---|---|---|
| band T 2.451 (Fig. 1, as in hsv.py) | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | -0.558 / 0.568 | -0.345 / 0.364 | -0.165 / 0.196 | 0.288, 27 % |
| band T 2.451 (Fig. 1, as in hsv.py) | L_FCI fitted, L_CD 0.489 | L_FCI 0.0048 mm, L_CD 0.49 mm | +0.000 / 0.102 | +0.213 / 0.242 | +0.393 / 0.407 | 0.340, 25 % |
| band T 2.451 (Fig. 1, as in hsv.py) | L_FCI and L_CD fitted | L_FCI 0.0116 mm, L_CD 0.0236 mm | -0.000 / 0.102 | -0.126 / 0.170 | -0.046 / 0.115 | 0.144, 75 % |
| Fig. 1 plain points, Weibull MLE (paper's procedure) | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | -0.671 / 0.679 | -0.427 / 0.442 | -0.225 / 0.248 | 0.353, 23 % |
| Fig. 1 plain points, Weibull MLE (paper's procedure) | L_FCI fitted, L_CD 0.489 | L_FCI 0.00372 mm, L_CD 0.49 mm | +0.000 / 0.102 | +0.244 / 0.270 | +0.446 / 0.459 | 0.382, 23 % |
| Fig. 1 plain points, Weibull MLE (paper's procedure) | L_FCI and L_CD fitted | L_FCI 0.00165 mm, L_CD 2.37 mm | +0.000 / 0.102 | +0.304 / 0.325 | +0.639 / 0.648 | 0.521, 11 % |
| 2019 series at own p, R | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | -0.184 / 0.221 | -0.071 / 0.127 | +0.035 / 0.103 | 0.115, 91 % |
| 2019 series at own p, R **(chosen)** | L_FCI fitted, L_CD 0.489 | L_FCI 0.0245 mm, L_CD 0.49 mm | +0.009 / 0.087 | +0.120 / 0.154 | +0.225 / 0.246 | 0.208, 56 % |
| 2019 series at own p, R | L_FCI and L_CD fitted | L_FCI 0.0113 mm, L_CD 0.705 mm | +0.000 / 0.083 | +0.145 / 0.179 | +0.280 / 0.302 | 0.252, 39 % |
| 2019 series at own p, R, stress modulus | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | -0.095 / 0.191 | -0.007 / 0.148 | +0.083 / 0.149 | 0.149, 81 % |
| 2019 series at own p, R, stress modulus | L_FCI fitted, L_CD 0.489 | L_FCI 0.0398 mm, L_CD 0.49 mm | +0.033 / 0.131 | +0.120 / 0.170 | +0.208 / 0.233 | 0.206, 52 % |
| 2019 series at own p, R, stress modulus | L_FCI and L_CD fitted | L_FCI 0.0036 mm, L_CD 1.24 mm | -0.000 / 0.100 | +0.147 / 0.190 | +0.316 / 0.339 | 0.279, 34 % |
| 2019 series pooled | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | -0.234 / 0.255 | -0.110 / 0.159 | +0.006 / 0.106 | 0.134, 82 % |
| 2019 series pooled | L_FCI fitted, L_CD 0.489 | L_FCI 0.0189 mm, L_CD 0.49 mm | +0.000 / 0.102 | +0.124 / 0.169 | +0.240 / 0.262 | 0.223, 49 % |
| 2019 series pooled | L_FCI and L_CD fitted | L_FCI 0.383 mm, L_CD 0.0899 mm | -0.000 / 0.102 | -0.106 / 0.157 | -0.074 / 0.129 | 0.143, 81 % |
| sharp-V life scatter (NOT paper's source) | L_FCI 0.4 ASSUMED, L_CD 0.489 (no fit) | L_FCI 0.4 mm, L_CD 0.49 mm | +0.101 / 0.144 | +0.134 / 0.176 | +0.182 / 0.211 | 0.195, 57 % |
| sharp-V life scatter (NOT paper's source) | L_FCI fitted, L_CD 0.489 | L_FCI 21.4 mm, L_CD 0.49 mm | +0.000 / 0.102 | +0.032 / 0.120 | +0.081 / 0.133 | 0.127, 89 % |
| sharp-V life scatter (NOT paper's source) | L_FCI and L_CD fitted | L_FCI 2.23 mm, L_CD 0.639 mm | +0.000 / 0.102 | +0.059 / 0.129 | +0.129 / 0.167 | 0.150, 80 % |
| limit e -> 0 (NOT the paper: no size term) | global scale fitted | scale 0.540 | +0.000 / 0.102 | -0.013 / 0.116 | +0.002 / 0.106 | 0.111, 96 % |

Sensitivity of the no-fit variant (k from 2019 series at own p, R; L_CD 0.489) to the two ASSUMED inputs (held-out rms, in band; V = bias on the fitted sharp V):

| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |
|---|---|---|---|
| 0.1 | 0.152, 81 %; V -0.060 | 0.134, 86 %; V -0.088 | 0.118, 93 %; V -0.124 |
| 0.2 | 0.124, 92 %; V -0.108 | 0.114, 93 %; V -0.136 | 0.113, 93 %; V -0.171 |
| 0.4 | 0.112, 92 %; V -0.156 | 0.115, 91 %; V -0.184 | 0.129, 87 %; V -0.219 |
| 0.8 | 0.121, 89 %; V -0.204 | 0.136, 85 %; V -0.232 | 0.159, 74 %; V -0.267 |

References (fatigue.calib, same protocol): net stress, no notch factor 0.144, TCD line, constant L 0.139, Weibull, m fitted on sharp V 0.123.
