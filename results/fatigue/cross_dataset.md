# Cross-dataset prediction: criterion fitted on one data set, every test of the other predicted

Cells: bias / rms log10(N_pred/N), count within x2 in life.

| criterion | fitted on | parameters | on the fit set (in-sample) | on the other set (predicted) |
|---|---|---|---|---|
| kt | Ezeh-Susmel | - | -2.15 / 2.29, 0/18 | -1.83 / 1.83, 0/14 |
| kt | Seifollahi | - | -1.83 / 1.83, 0/14 | -2.15 / 2.29, 0/18 |
| hotspot | Ezeh-Susmel | l 1.21 | +0.04 / 0.43, 8/18 | +0.26 / 0.28, 8/14 |
| hotspot | Seifollahi | l 0.832 | +0.00 / 0.09, 14/14 | -0.20 / 0.50, 10/18 |
| tcd_point | Ezeh-Susmel | L 1.68 | +0.00 / 0.82, 4/18 | -0.13 / 0.15, 14/14 |
| tcd_point | Seifollahi | L 1.91 | +0.00 / 0.05, 14/14 | +0.11 / 0.83, 2/18 |
| tcd_line | Ezeh-Susmel | L 1.35 | +0.01 / 0.69, 4/18 | -0.02 / 0.06, 14/14 |
| tcd_line | Seifollahi | L 1.39 | +0.00 / 0.05, 14/14 | +0.03 / 0.69, 4/18 |
| weibull | Ezeh-Susmel | m 7.27 | +0.02 / 0.44, 7/18 | +0.78 / 0.78, 0/14 |
| weibull | Seifollahi | m 12.4 | +0.00 / 0.09, 14/14 | -0.69 / 0.87, 6/18 |
| sweibull | Ezeh-Susmel | m 55.5, l 0.624 | +0.03 / 0.41, 7/18 | +0.33 / 0.34, 4/14 |
| sweibull | Seifollahi | m 1e+03, l 0.776 | +0.00 / 0.08, 14/14 | -0.22 / 0.49, 10/18 |

In-sample floor, a free constant notch factor per notch geometry (no criterion can beat it):
- Ezeh-Susmel: +0.03 / 0.25, 14/18; Kf blunt 0.742, intermediate 1.243, sharp 1.357
- Seifollahi: +0.00 / 0.05, 14/14; Kf edge 1.126, hole 1.164
