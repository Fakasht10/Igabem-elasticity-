# Print-resolved Weibull, ezeh-susmel-pla: contour 0.4 mm + infill (45, -45)

Blind: S_F / S_T = 1.994 (static PLA card), gauge 30 mm.

| R | m (pooled plain) | S_T (MPa) | S_F (MPa) | shear ratio r | plain fit rms log |
|---|---|---|---|---|---|
| -1 | 7.4 | 11.74 | 23.40 | 0.000 | 0.176 |
| 0 | 7.7 | 9.87 | 19.69 | 0.960 | 0.015 |

| notch | p | R | measured | blind | LOO (S_F/S_T from the other two notches) | blind hazard shares |
|---|---|---|---|---|---|---|
| blunt | 0 | -1 | 8.2 | 8.91 (+9%) | 7.77 (-5%) | infill-T 89%, contour-F 10% |
| blunt | 0 | 0 | 7.3 | 7.03 (-4%) | 5.99 (-18%) | infill-T 94%, contour-F 5% |
| blunt | 30 | -1 | 6.7 | 7.40 (+10%) | 7.34 (+10%) | infill-T 97%, contour-F 3% |
| blunt | 30 | 0 | 6.9 | 6.22 (-10%) | 5.70 (-17%) | infill-T 97%, contour-F 2% |
| blunt | 45 | -1 | 9.2 | 7.15 (-22%) | 7.22 (-22%) | infill-T 97%, contour-F 2% |
| blunt | 45 | 0 | 7.5 | 6.07 (-19%) | 5.62 (-25%) | infill-T 98%, contour-F 2% |
| intermediate | 0 | -1 | 4.9 | 8.46 (+73%) | 7.18 (+46%) | infill-T 65%, contour-F 34% |
| intermediate | 0 | 0 | 4.4 | 6.89 (+57%) | 5.78 (+31%) | infill-T 75%, contour-F 25% |
| intermediate | 30 | -1 | 5.1 | 7.43 (+46%) | 6.89 (+35%) | infill-T 86%, contour-F 13% |
| intermediate | 30 | 0 | 4.7 | 6.26 (+33%) | 5.60 (+19%) | infill-T 88%, contour-F 12% |
| intermediate | 45 | -1 | 4.4 | 7.22 (+64%) | 6.81 (+55%) | infill-T 89%, contour-F 11% |
| intermediate | 45 | 0 | 4.3 | 6.13 (+43%) | 5.54 (+29%) | infill-T 89%, contour-F 10% |
| sharp | 0 | -1 | 4.3 | 6.30 (+46%) | 4.02 (-6%) | contour-F 83%, infill-T 10%, contour-T 7% |
| sharp | 0 | 0 | 4.5 | 5.21 (+16%) | 3.00 (-33%) | contour-F 78%, infill-T 11%, contour-T 10% |
| sharp | 30 | -1 | 4.8 | 6.17 (+29%) | 4.02 (-16%) | contour-F 71%, infill-T 23%, contour-T 6% |
| sharp | 30 | 0 | 3.9 | 5.14 (+32%) | 3.00 (-23%) | contour-F 70%, infill-T 21%, contour-T 9% |
| sharp | 45 | -1 | 4.1 | 6.13 (+49%) | 4.02 (-2%) | contour-F 68%, infill-T 27%, contour-T 5% |
| sharp | 45 | 0 | 3.7 | 5.12 (+38%) | 3.00 (-19%) | contour-F 68%, infill-T 23%, contour-T 9% |

- blind: mean +27.2%, rms log 0.317, within 20% 6/18
- leave one notch out: mean +2.1%, rms log 0.252, within 20% 9/18
- fitted on all 18 (S_F/S_T 1.21): mean +1.8%, rms log 0.192, within 20% 12/18
- LOO ratios S_F/S_T: held out blunt: 1.15, held out intermediate: 1.42, held out sharp: 0.98
