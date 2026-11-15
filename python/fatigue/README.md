# Fatigue transfer harness

Predict the fatigue strength of notched or holed printed specimens from the plain specimen's fatigue strength. Every series is matched on print angle and R, so no mean-stress model enters. The prediction runs from one cached IgBEM solve per geometry.

```
specimens.py   quarter models: edge U/V notch plate, central-hole plate, plain bar (STEP via OCC, bc_case body)
fields.py      unit-net-stress field, cached: bisector sigma_1 at 5 thickness stations, Kt, TET10 Gauss cloud + |grad sigma_1|
criteria.py    kt | tcd_point | tcd_line | weibull (m from scatter) | weibull_support (Maede et al. support length)
transfer.py    dataset -> fields -> every criterion -> table (md, json) + paper figure (png, pdf) in results/fatigue
data/*.json    datasets, values transcribed from the papers' tables, with provenance
```

Run it from `python/` with `python3 -m fatigue.transfer`. The first run takes about 10 minutes, mostly the TET10 Gauss clouds. Later runs take seconds.

Each criterion is allowed to know different things. `kt` and `weibull` are blind: Weibull takes m from the plain series' own scatter ratio T, and the plain gauge volume is W·T·L_g. `tcd_*` uses the dataset's own fitted critical distance, so it is a reference, not a blind prediction.

## Ezeh & Susmel PLA (2019 plain, 2020 notched)

The data is New Verbatim PLA on an Ultimaker 2E+ with the same settings in both papers. Specimens are 6 mm net, 3 mm thick, ±45 raster, at p = 0/30/45 and R = −1/0. The notches are printed: U r3, U r1, and V 35° with r 0.15. All 18 notched series have a matching plain series.

Two caveats limit what this dataset can show:

- The plain gauge length is not given in either paper, so Weibull is run at L_g = 15, 30 and 60 mm. One factor of 2 in L_g moves the prediction by 2^(1/m).
- The plain series scatter is large. For example, p0 R−1 is 10.4 MPa against 4.5 MPa for p30 R−1. Several notched series beat their plain series: blunt Kf = 0.68–0.73 at p = 30/45.

A clean single-campaign test of plain plus open hole is Seifollahi & Kabir 2026 (Polymers 18(1):1). Its tables are images on the MDPI page and are not yet transcribed.

Gate: with the authors' unified plain curve (σ_max 6.7 MPa) and their L_M = 0.49 mm at 2e6 cycles, the point method on our 3D field gives 3.80 MPa for the sharp notch. The authors report 3.9 MPa, a −2.6% difference. Their L was calibrated on that notch, so this verifies the fields against their 2D FE.

Mesh: refining h from 1.5 to 1.0 roughly doubles the unknowns (13–19k to 26–36k; the refined run uses the H-matrix solver).

| Notch | Change in Kt | Change in the TCD point stress |
| --- | --- | --- |
| U r3 | +0.21% | +0.08% |
| U r1 | +0.04% | +0.06% |
| V r0.15 | +1.9% | −1.8% |

All of these are small next to the ±20–50% transfer errors, so h 1.5 is the default.

Mean stress: `--walker R` calibrates on the plain series at R only. It maps the plain strength to every other R through a Walker exponent fitted per angle on the full plain table, σ_ar = σ_max^(1−γ) σ_a^γ. The fitted γ is 0.25, 0.68 and 0.33 at p = 0, 30 and 45°. On the 9 R = 0 notched series, calibrating from R = −1 moves the Weibull (L_g 30) mean error from −11% to −31%.

Walker collapses the plain data per angle. The scatter of the four R levels about one curve is a log-sd of 0.09, 0.15 and 0.09, inside each series' own scatter of 0.10, 0.23 and 0.15. The notched data show almost no mean-stress effect: a γ fitted to the notched R = −1/0 pairs comes out at about 0.9–1.1, meaning amplitude only. So the plain γ is not applied by default. The likely cause is local mean-stress relaxation at the root, which a linear-elastic transfer cannot represent.
