# es2020: Ezeh & Susmel 2020 (notched FDM PLA fatigue), reproduced on IgBEM

Step 1 of a clean restart: reproduce the paper exactly before comparing any other method.

| file | what |
| --- | --- |
| `paper.py` | every constant printed in the paper (UTS 42.9, plain P50 6.7 MPa k 5.1 T 2.451, design 0.1 UTS k 5.5, Fig. 8a, Eq. 6, Table 1) and data loaders |
| `point.py` | Point Method (Eq. 2) and the Fig. 9e calibration of L_M(N) on an IgBEM bisector profile |
| `fig10.py` | Fig. 10 recomputed and compared marker by marker; writes `fig10_point_reproduction.png/pdf` |
| `extract.py` | vector extraction of Figs 8 and 10 from the accepted manuscript (no pixel digitising) |
| `data/` | Figs 5–7 specimens, Fig. 8 and Fig. 10 markers, IgBEM bisector profiles (3D, mesh-converged) |

Results (`python -m pytest python/es2020 -q`, < 1 s):
- Figs 5–7 refit reproduces all 18 Table 1 rows; Fig. 8 refit reproduces the printed Fig. 8a–c curves.
- Fig. 9e calibration on our sharp-V profile with the printed Fig. 8a curve: L_M = 16.7 N^-0.242 (paper 16.4 N^-0.242).
- Fig. 10: all 157 markers within 2.6 % (rms 1.0 %) using the Fig. 8 stresses and Eq. 6.

Data caveat found in the paper: Fig. 8 (and therefore Fig. 10) plots sigma_n,max values that differ from
Figs 5–7 converted with 2/(1 - R) for 73 of 157 specimens by more than 3 %, mostly at long life (worst -32 %).
Regressing the Figs 5–7 stresses instead gives a sharp curve k 3.56, 4.78 MPa and L_M = 10.4 N^-0.183.
The one departure from the paper's procedure: K(d) is the thickness-mean sigma_1 of a 3D IgBEM solve, the paper used 2D FE.
