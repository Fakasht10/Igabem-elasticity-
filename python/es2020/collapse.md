Dimensionless collapse of the notched data (Buckingham-Pi study, 2026-11-16).

    python -m es2020.collapse [out_dir]  -> collapse.md, fig_collapse_groups.png/pdf, fig_collapse_closures.png/pdf

Response group Pi_S = sigma_n,max / sigma_0(N), sigma_0 = plain P50 (paper: 6.7 MPa at 2e6, k 5.1, capped at UTS).
Per specimen K_f,i = sigma_0(N_i) / sigma_n,max,i (Fig. 8 stresses, the paper's chain). Groups:
    rho / L(N)      notch number          L(N) = paper Eq. 6, 16.4 N^-0.242 mm
    Lambda = b / L  ligament number       b = 3 mm half net width
    sigma_n,max/UTS static group          UTS 42.9 MPa
Closures K_f(N[, sigma]) are calibrated the paper's way (plain P50 + Fig. 8a sharp curve on 1e3..2e6) and U r1,
U r3 are never used in a fit. Post-processing only: K_t, chi and the profiles are the stored IgBEM fields.

| closure | fit rms (curve) | fitted sharp V: bias / rms / slope / in band | held-out U r1 | held-out U r3 | held-out both: rms / in band | held-out bias R -1 / R 0 |
|---|---|---|---|---|---|---|
| C6 Neuber x constraint, rho' = 6.73e+04 N^-0.659 mm, NSR 1.57 | 0.004 | +0.005 / 0.090 / +0.029 / 96% | -0.002 / 0.116 / +0.064 / 90% | +0.063 / 0.141 / +0.108 / 81% | 0.130 / 85% | -0.041 / +0.115 |
| F4 Weibull, m = 2.41 N^0.090 (one extra parameter), V0 1.01e+03 mm^3 | 0.002 | +0.004 / 0.104 / +0.035 / 96% | -0.095 / 0.157 / +0.045 / 69% | -0.020 / 0.121 / +0.085 / 93% | 0.139 / 81% | -0.143 / +0.044 |
| C6'' net x constraint only, c 1.6, NSR 2.69 | 0.033 | +0.007 / 0.086 / +0.025 / 100% | +0.036 / 0.106 / +0.065 / 92% | +0.101 / 0.165 / +0.123 / 74% | 0.140 / 82% | +0.016 / +0.131 |
| C0 net stress, K_f = 1 | 0.121 | -0.008 / 0.125 / -0.080 / 89% | -0.002 / 0.119 / -0.017 / 94% | +0.124 / 0.158 / +0.051 / 74% | 0.141 / 83% | -0.019 / +0.158 |
| G1 peak sigma_1 / (1 + sqrt(c chi L_M(N))), paper L_M, c 1.53 (one material constant) | 0.016 | +0.007 / 0.101 / +0.021 / 96% | -0.063 / 0.145 / +0.062 / 81% | -0.007 / 0.141 / +0.115 / 81% | 0.143 / 81% | -0.122 / +0.067 |
| L2 local strain, Neuber on sigma_1 at paper L_M(N)/2, K' 3.2e+04 MPa, n' 1.42 (fitted) | 0.002 | +0.004 / 0.112 / +0.038 / 94% | +0.036 / 0.141 / +0.058 / 85% | +0.089 / 0.152 / +0.084 / 78% | 0.147 / 81% | -0.029 / +0.169 |
| C4 Siebel-Stieler, s = 38.2 N^-0.282 mm | 0.003 | +0.003 / 0.105 / +0.037 / 96% | -0.064 / 0.150 / +0.075 / 75% | -0.010 / 0.150 / +0.126 / 81% | 0.150 / 78% | -0.125 / +0.066 |
| T3 Neuber x constraint x triaxiality, rho' = 4.42e+03 N^-0.688 mm, NSR 1.17, exp(beta) 0.491 | 0.000 | +0.004 / 0.099 / +0.033 / 96% | +0.008 / 0.140 / +0.092 / 83% | +0.104 / 0.181 / +0.127 / 72% | 0.163 / 77% | -0.026 / +0.155 |
| C3 Neuber, rho' = 1.3e+09 N^-1.357 mm | 0.061 | +0.044 / 0.106 / -0.021 / 96% | +0.040 / 0.132 / +0.028 / 88% | +0.146 / 0.186 / +0.077 / 69% | 0.163 / 77% | +0.009 / +0.195 |
| C6' El Haddad x constraint, L = 94.7 N^-0.359 mm, NSR 1.56 | 0.004 | +0.005 / 0.090 / +0.029 / 96% | +0.032 / 0.133 / +0.086 / 83% | +0.116 / 0.192 / +0.139 / 70% | 0.167 / 76% | -0.000 / +0.163 |
| S2c W_dev + alpha W_vol, constant R0 = 1.05 mm, alpha 0.902 | 0.113 | +0.035 / 0.130 / -0.080 / 89% | +0.032 / 0.123 / -0.017 / 96% | +0.174 / 0.200 / +0.051 / 59% | 0.168 / 76% | +0.024 / +0.201 |
| S1c averaged SED, constant R0 = 1.05 mm | 0.113 | +0.037 / 0.131 / -0.080 / 91% | +0.033 / 0.124 / -0.017 / 96% | +0.175 / 0.201 / +0.051 / 59% | 0.169 / 76% | +0.025 / +0.202 |
| S2 W_dev + alpha W_vol, R0 = 17.2 N^-0.258 mm, alpha 1.02 | 0.003 | +0.003 / 0.104 / +0.035 / 96% | +0.023 / 0.153 / +0.103 / 81% | +0.133 / 0.207 / +0.136 / 67% | 0.184 / 74% | -0.010 / +0.184 |
| S1 averaged SED (alpha = 1, Lazzarin-Berto), R0 = 17.6 N^-0.261 mm | 0.003 | +0.003 / 0.104 / +0.035 / 96% | +0.023 / 0.154 / +0.103 / 81% | +0.134 / 0.208 / +0.137 / 67% | 0.184 / 74% | -0.010 / +0.185 |
| C5 El Haddad + ligament, L = 1.54e+04 N^-0.719 mm | 0.061 | +0.044 / 0.107 / -0.020 / 96% | +0.059 / 0.147 / +0.049 / 83% | +0.173 / 0.221 / +0.107 / 65% | 0.190 / 74% | +0.029 / +0.221 |
| C2 Peterson, a = 1.4e+04 N^-0.678 mm | 0.061 | +0.044 / 0.106 / -0.021 / 96% | +0.080 / 0.159 / +0.061 / 83% | +0.180 / 0.222 / +0.100 / 61% | 0.195 / 72% | +0.042 / +0.235 |
| F1 point x (3 eta_line[0,2L])^beta, L = 6.42 N^-0.241 mm, exp(beta) 0.555 | 0.001 | +0.004 / 0.104 / +0.036 / 96% | +0.110 / 0.175 / +0.077 / 73% | +0.193 / 0.235 / +0.104 / 54% | 0.209 / 63% | +0.066 / +0.253 |
| F3 point x (3 eta_Weibull(R=L))^beta, L = 2.67 N^-0.228 mm, exp(beta) 0.186 | 0.001 | +0.004 / 0.104 / +0.036 / 96% | +0.160 / 0.203 / +0.049 / 56% | +0.239 / 0.266 / +0.084 / 44% | 0.239 / 50% | +0.116 / +0.299 |
| C5 El Haddad + ligament, paper's L(N), no fit | 0.125 | +0.131 / 0.164 / -0.024 / 64% | +0.147 / 0.193 / +0.041 / 62% | +0.257 / 0.287 / +0.097 / 43% | 0.247 / 52% | +0.117 / +0.305 |
| T2 Neuber x (3 eta(rho'/2))^beta, rho' = 1.03 N^-0.194 mm, exp(beta) 0.217 | 0.006 | +0.005 / 0.102 / +0.033 / 96% | +0.197 / 0.230 / +0.023 / 50% | +0.258 / 0.278 / +0.063 / 33% | 0.257 / 41% | +0.144 / +0.325 |
| mHSV corrected (k_N 1.111, 2019 series), C_S 4.52 mm^3 | 0.113 | +0.035 / 0.130 / -0.080 / 89% | +0.167 / 0.205 / -0.017 / 54% | +0.300 / 0.316 / +0.051 / 13% | 0.270 / 32% | +0.154 / +0.331 |
| F2 line x (3 eta_line[0,2L])^beta, L = 0.342 N^-0.165 mm, exp(beta) 0.183 | 0.002 | +0.004 / 0.104 / +0.036 / 96% | +0.218 / 0.248 / +0.020 / 50% | +0.269 / 0.288 / +0.063 / 30% | 0.270 / 39% | +0.160 / +0.341 |
| L1 local strain, Neuber on the field peak, K' 35 MPa, n' 0.449 (fitted) | 0.001 | +0.002 / 0.066 / +0.023 / 100% | -0.233 / 0.251 / +0.061 / 33% | -0.295 / 0.317 / +0.104 / 22% | 0.288 / 27% | -0.326 / -0.199 |
| T1 net x c (3 eta)^beta at paper L_M/2, c 0.159, exp(beta) 11.4 | 0.015 | +0.006 / 0.103 / +0.028 / 96% | -0.266 / 0.304 / -0.094 / 31% | -0.381 / 0.393 / -0.046 / 0% | 0.354 / 15% | -0.405 / -0.239 |
| C1 K_f = K_t (elastic) | 0.772 | +0.799 / 0.809 / -0.080 / 0% | +0.402 / 0.419 / -0.017 / 4% | +0.326 / 0.341 / +0.051 / 6% | 0.380 / 5% | +0.277 / +0.457 |

Net-stress residual by R and raster angle (all notches, failures): value / n / mean log10(sigma_n,max/P50)
- R: -1.0: n 78, -0.045, 0.0: n 71, +0.137
- p: 0: n 50, +0.060, 30: n 48, +0.038, 45: n 51, +0.027
- K_t (net, IgBEM): sharp 6.41, intermediate 2.54, blunt 1.59; DENT g = K/sigma_net = 1.929 sqrt(mm)

Field-only ranking (any 3D part; net stress kept only as a coupon reference line):
| field measure | fit rms (curve) | fitted sharp V: bias / rms / slope | U r1 bias / rms / slope | U r3 bias / rms / slope | held-out both: rms / in band / slope |
|---|---|---|---|---|---|
| F4 Weibull, m = 2.41 N^0.090 (one extra parameter), V0 1.01e+03 mm^3 | 0.002 | +0.004 / 0.104 / +0.035 | -0.095 / 0.157 / +0.045 | -0.020 / 0.121 / +0.085 | 0.139 / 81% / +0.063 |
| REFERENCE ONLY (coupon, not defined on a general part): net stress | 0.121 | -0.008 / 0.125 / -0.080 | -0.002 / 0.119 / -0.017 | +0.124 / 0.158 / +0.051 | 0.141 / 83% / +0.014 |
| G1 peak sigma_1 / (1 + sqrt(c chi L_M(N))), paper L_M, c 1.53 (one material constant) | 0.016 | +0.007 / 0.101 / +0.021 | -0.063 / 0.145 / +0.062 | -0.007 / 0.141 / +0.115 | 0.143 / 81% / +0.088 |
| L2 local strain, Neuber on sigma_1 at paper L_M(N)/2, K' 3.2e+04 MPa, n' 1.42 (fitted) | 0.002 | +0.004 / 0.112 / +0.038 | +0.036 / 0.141 / +0.058 | +0.089 / 0.152 / +0.084 | 0.147 / 81% / +0.069 |
| peak sigma_1 / (1 + sqrt(c chi L)) (FKM-type, peak and chi from field): K_t / (1 + sqrt(c chi)), c = 38.2 N^-0.282 mm | 0.003 | +0.003 / 0.105 / +0.037 | -0.064 / 0.150 / +0.075 | -0.010 / 0.150 / +0.126 | 0.150 / 78% / +0.099 |
| TCD line (mean sigma_1 over 2L on the steepest-descent path): L = 14.7 N^-0.238 mm | 0.037 | +0.036 / 0.107 / -0.006 | +0.041 / 0.137 / +0.049 | +0.144 / 0.189 / +0.089 | 0.167 / 77% / +0.065 |
| S2c W_dev + alpha W_vol, constant R0 = 1.05 mm, alpha 0.902 | 0.113 | +0.035 / 0.130 / -0.080 | +0.032 / 0.123 / -0.017 | +0.174 / 0.200 / +0.051 | 0.168 / 76% / +0.013 |
| S1c averaged SED, constant R0 = 1.05 mm | 0.113 | +0.037 / 0.131 / -0.080 | +0.033 / 0.124 / -0.017 | +0.175 / 0.201 / +0.051 | 0.169 / 76% / +0.013 |
| Weibull, whole volume, m 3.44 fixed: whole volume; m 3.44 FIXED (plain band), V0 130 mm^3 | 0.113 | +0.035 / 0.130 / -0.080 | +0.039 / 0.125 / -0.017 | +0.175 / 0.201 / +0.051 | 0.170 / 76% / +0.014 |
| S2 W_dev + alpha W_vol, R0 = 17.2 N^-0.258 mm, alpha 1.02 | 0.003 | +0.003 / 0.104 / +0.035 | +0.023 / 0.153 / +0.103 | +0.133 / 0.207 / +0.136 | 0.184 / 74% / +0.114 |
| averaged SED, R0(N): R0 = 17.5 N^-0.260 mm | 0.003 | +0.003 / 0.104 / +0.035 | +0.023 / 0.154 / +0.103 | +0.133 / 0.208 / +0.137 | 0.184 / 74% / +0.114 |
| TCD point (sigma_1 at L/2 on the steepest-descent path = bisector here): L_M = 15.8 N^-0.237 mm | 0.007 | +0.005 / 0.102 / +0.032 | +0.070 / 0.158 / +0.085 | +0.172 / 0.219 / +0.109 | 0.193 / 69% / +0.092 |
| TCD area (point L_M): L = point L_M = 15.8 N^-0.237 mm (not refitted, as the paper) | 0.039 | +0.044 / 0.110 / +0.026 | +0.079 / 0.161 / +0.082 | +0.178 / 0.225 / +0.111 | 0.197 / 66% / +0.092 |
| F1 point x (3 eta_line[0,2L])^beta, L = 6.42 N^-0.241 mm, exp(beta) 0.555 | 0.001 | +0.004 / 0.104 / +0.036 | +0.110 / 0.175 / +0.077 | +0.193 / 0.235 / +0.104 | 0.209 / 63% / +0.087 |
| F3 point x (3 eta_Weibull(R=L))^beta, L = 2.67 N^-0.228 mm, exp(beta) 0.186 | 0.001 | +0.004 / 0.104 / +0.036 | +0.160 / 0.203 / +0.049 | +0.239 / 0.266 / +0.084 | 0.239 / 50% / +0.063 |
| mHSV corrected (k_N 1.111, 2019 series), C_S 4.52 mm^3 | 0.113 | +0.035 / 0.130 / -0.080 | +0.167 / 0.205 / -0.017 | +0.300 / 0.316 / +0.051 | 0.270 / 32% / +0.014 |
| F2 line x (3 eta_line[0,2L])^beta, L = 0.342 N^-0.165 mm, exp(beta) 0.183 | 0.002 | +0.004 / 0.104 / +0.036 | +0.218 / 0.248 / +0.020 | +0.269 / 0.288 / +0.063 | 0.270 / 39% / +0.041 |
| L1 local strain, Neuber on the field peak, K' 35 MPa, n' 0.449 (fitted) | 0.001 | +0.002 / 0.066 / +0.023 | -0.233 / 0.251 / +0.061 | -0.295 / 0.317 / +0.104 | 0.288 / 27% / +0.088 |
| modified HSV: C_S 1.6 mm^3; FIXED: L_CD 0.504 mm (point L_M at 2e6), k_N 0.675 (plain band) | 0.113 | +0.035 / 0.130 / -0.080 | +0.256 / 0.282 / -0.017 | +0.454 / 0.464 / +0.051 | 0.389 / 22% / +0.010 |

Model-free collapse (per-specimen K_f, all three notches):
| coordinate x (one quadratic in log x, all notches) | rms of log10 K_f | rms of q |
|---|---|---|
| ligament b/L(N) (= one curve in N, no notch group) | 0.129 | 0.282 |
| rho_L | 0.109 | 0.211 |
| chi_L | 0.109 | 0.209 |
| static | 0.119 | 0.270 |
| wvol_wdev | 0.109 | 0.196 |
| v90_L3 | 0.113 | 0.228 |
| floor: separate curve in N per notch | 0.099 | 0.184 |
