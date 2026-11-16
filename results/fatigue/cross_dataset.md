# Cross-dataset prediction: criterion fitted on one data set, every test of the other predicted

Cells: bias / rms log10(N_pred/N), count within x2 in life.

| criterion | fitted on | parameters | on the fit set (in-sample) | on the other set (predicted) |
|---|---|---|---|---|
| kt | Ezeh-Susmel | - | -3.44 / 3.79, 0/149 | -1.83 / 1.83, 0/14 |
| kt | Seifollahi | - | -1.83 / 1.83, 0/14 | -3.44 / 3.79, 0/149 |
| hotspot | Ezeh-Susmel | l 4.78 | -0.38 / 0.70, 46/149 | +0.36 / 0.37, 3/14 |
| hotspot | Seifollahi | l 0.832 | +0.00 / 0.09, 14/14 | -0.88 / 1.07, 22/149 |
| tcd_point | Ezeh-Susmel | L 2.42 | -0.05 / 0.73, 35/149 | +0.22 / 0.23, 13/14 |
| tcd_point | Seifollahi | L 1.91 | +0.00 / 0.05, 14/14 | -0.31 / 0.77, 46/149 |
| tcd_line | Ezeh-Susmel | L 2.17 | -0.15 / 0.64, 39/149 | +0.32 / 0.33, 5/14 |
| tcd_line | Seifollahi | L 1.39 | +0.00 / 0.05, 14/14 | -0.45 / 0.74, 40/149 |
| weibull | Ezeh-Susmel | m 5.44 | +0.06 / 0.55, 66/149 | +1.00 / 1.00, 0/14 |
| weibull | Seifollahi | m 12.4 | +0.00 / 0.09, 14/14 | -1.50 / 1.80, 8/149 |
| sweibull | Ezeh-Susmel | m 18.1, l 0.41 | +0.03 / 0.52, 68/149 | +0.88 / 0.89, 0/14 |
| sweibull | Seifollahi | m 1e+03, l 0.776 | +0.00 / 0.08, 14/14 | -0.84 / 1.04, 21/149 |

In-sample floor, a free constant notch factor per notch geometry (no criterion can beat it):
- Ezeh-Susmel: +0.01 / 0.46, 81/149; Kf blunt 0.742, intermediate 0.972, sharp 0.949
- Seifollahi: +0.00 / 0.05, 14/14; Kf edge 1.126, hole 1.164
