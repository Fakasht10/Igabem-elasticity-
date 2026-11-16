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
| C6'' net x constraint only, c 1.6, NSR 2.69 | 0.033 | +0.007 / 0.086 / +0.025 / 100% | +0.036 / 0.106 / +0.065 / 92% | +0.101 / 0.165 / +0.123 / 74% | 0.140 / 82% | +0.016 / +0.131 |
| C0 net stress, K_f = 1 | 0.121 | -0.008 / 0.125 / -0.080 / 89% | -0.002 / 0.119 / -0.017 / 94% | +0.124 / 0.158 / +0.051 / 74% | 0.141 / 83% | -0.019 / +0.158 |
| C4 Siebel-Stieler, s = 38.2 N^-0.282 mm | 0.003 | +0.003 / 0.105 / +0.037 / 96% | -0.064 / 0.150 / +0.075 / 75% | -0.010 / 0.150 / +0.126 / 81% | 0.150 / 78% | -0.125 / +0.066 |
| C3 Neuber, rho' = 1.3e+09 N^-1.357 mm | 0.061 | +0.044 / 0.106 / -0.021 / 96% | +0.040 / 0.132 / +0.028 / 88% | +0.146 / 0.186 / +0.077 / 69% | 0.163 / 77% | +0.009 / +0.195 |
| C6' El Haddad x constraint, L = 94.7 N^-0.359 mm, NSR 1.56 | 0.004 | +0.005 / 0.090 / +0.029 / 96% | +0.032 / 0.133 / +0.086 / 83% | +0.116 / 0.192 / +0.139 / 70% | 0.167 / 76% | -0.000 / +0.163 |
| C5 El Haddad + ligament, L = 1.54e+04 N^-0.719 mm | 0.061 | +0.044 / 0.107 / -0.020 / 96% | +0.059 / 0.147 / +0.049 / 83% | +0.173 / 0.221 / +0.107 / 65% | 0.190 / 74% | +0.029 / +0.221 |
| C2 Peterson, a = 1.4e+04 N^-0.678 mm | 0.061 | +0.044 / 0.106 / -0.021 / 96% | +0.080 / 0.159 / +0.061 / 83% | +0.180 / 0.222 / +0.100 / 61% | 0.195 / 72% | +0.042 / +0.235 |
| C5 El Haddad + ligament, paper's L(N), no fit | 0.125 | +0.131 / 0.164 / -0.024 / 64% | +0.147 / 0.193 / +0.041 / 62% | +0.257 / 0.287 / +0.097 / 43% | 0.247 / 52% | +0.117 / +0.305 |
| C1 K_f = K_t (elastic) | 0.772 | +0.799 / 0.809 / -0.080 / 0% | +0.402 / 0.419 / -0.017 / 4% | +0.326 / 0.341 / +0.051 / 6% | 0.380 / 5% | +0.277 / +0.457 |

Net-stress residual by R and raster angle (all notches, failures): value / n / mean log10(sigma_n,max/P50)
- R: -1.0: n 78, -0.045, 0.0: n 71, +0.137
- p: 0: n 50, +0.060, 30: n 48, +0.038, 45: n 51, +0.027
- K_t (net, IgBEM): sharp 6.41, intermediate 2.54, blunt 1.59; DENT g = K/sigma_net = 1.929 sqrt(mm)

Model-free collapse (per-specimen K_f, all three notches):
| coordinate x (one quadratic in log x, all notches) | rms of log10 K_f | rms of q |
|---|---|---|
| ligament b/L(N) (= one curve in N, no notch group) | 0.129 | 0.282 |
| rho_L | 0.109 | 0.211 |
| chi_L | 0.109 | 0.209 |
| static | 0.119 | 0.270 |
| v90_L3 | 0.113 | 0.228 |
| floor: separate curve in N per notch | 0.099 | 0.184 |
