# Every criterion calibrated the paper's way (plain P50 + pooled sharp-V curve), sigma_max/UTS, UTS 42.9

Sharp-V net sigma_max S-N: k 3.56, S50(2e6) 4.78 MPa. Plain: 6.7 MPa at 2e6, k 5.1.

| method | p(N) = A N^B | solved on N grid | trend k (all notches) | trend at 2e6 | U r1 median / in band | U r3 median / in band | V median / in band |
|---|---|---|---|---|---|---|---|
| authors' TCD point (Fig. 10 points, their L = 16.4 N^-0.242) | 16.4 N^-0.242 (published) | - | 8.17 | 0.261 | +0.074 / 77 % | +0.159 / 59 % | -0.024 / 96 % |
| TCD point, L(N) | 10.4 N^-0.183 | 1000-2000000 (40/40) | 8.17 | 0.256 | +0.046 / 77 % | +0.153 / 59 % | -0.011 / 96 % |
| TCD line, L(N) | 13.7 N^-0.208 | 27473-2000000 (23/40) | 6.60 | 0.220 | +0.024 / 85 % | +0.121 / 74 % | +0.045 / 96 % |
| Weibull, m(N) | 2.32 N^+0.078 | 1000-2000000 (40/40) | 6.72 | 0.183 | -0.071 / 83 % | +0.005 / 89 % | +0.003 / 96 % |
| HSV V80, m(N) | 3.47 N^+0.046 | 1000-2000000 (40/40) | 7.36 | 0.187 | -0.111 / 73 % | -0.013 / 87 % | -0.028 / 96 % |
| supported TCD line, l(N) (L = our L_M(N)): no solution on the V curve | - | none | - | - | - | - | - |
