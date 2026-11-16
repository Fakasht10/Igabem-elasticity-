# Fatigue transfer harness

Predict the fatigue strength of notched or holed printed specimens from the plain specimen's fatigue strength. Every series is matched on print angle and R, so no mean-stress model enters. The prediction runs from one cached IgBEM solve per geometry.

```
specimens.py   quarter models: edge U/V notch plate, central-hole plate, plain bar (outline, STEP via OCC, bc_case body)
blocks.py      exact multipatch of the edge-notch quarter from a designed layout: O-grid fan at the root, graded by
               knot insertion only (radial, along the flank, through the thickness); < 0.3 s, no CAD import
figures.py     specimen + BC + mesh figure; --convergence H... (mesh study of Kt and the TCD stress)
fields.py      unit-net-stress field, cached: bisector sigma_1 at 5 thickness stations, Kt, TET10 Gauss cloud + |grad sigma_1|
criteria.py    kt | tcd_point | tcd_line | weibull (m from scatter) | weibull_support (Maede et al. support length)
transfer.py    dataset -> fields -> every criterion -> table (md, json) + paper figure (png, pdf) in results/fatigue
data/*.json    datasets, values transcribed from the papers' tables, with provenance
```

Run it from `python/` with `python3 -m fatigue.transfer`. The first run takes about 6 minutes, almost all of it the TET10 Gauss clouds for the Weibull volume integral (38, 95 and 235 s). Each IgBEM solve takes 8–16 s. Later runs take seconds.

## Geometry and mesh

Edge notches are built by `blocks.py` rather than imported. A CAD import (`step_auto`) of these quarters took 62–305 s per mesh, and its general quad layout put a 0.15 mm sliver patch with skewed neighbours at the V root. Grading through its focus option failed the element-Jacobian check (min 0.009–0.03). The designed layout is exact: arcs are rational, top-face area matches the outline to within 1e-6 (tested), and patches are conforming. It builds in under 0.3 s with a min span Jacobian of about 0.5–0.7.

The defaults at h 1.5 are:
- first radial element ρ/8, growth 1.4;
- arc spans such that the root elements are square;
- 12 spans through the thickness, graded to T/60 at both free faces.

`field(h=...)` scales all of these together. Uniform thickness spans (6 × 0.5 mm) gave a false root stress near the free faces of the V notch: 7.39 at z = t/8 against 6.16 at 7t/8. With graded spans both read 6.28, and Kt at mid-thickness is unchanged (6.555 → 6.558 at 21k unknowns).

Each criterion is allowed to know different things. `kt` and `weibull` are blind: Weibull takes m from the plain series' own scatter ratio T, and the plain gauge volume is W·T·L_g. `tcd_*` uses the dataset's own fitted critical distance, so it is a reference, not a blind prediction.

## Ezeh & Susmel PLA (2019 plain, 2020 notched)

The data is New Verbatim PLA on an Ultimaker 2E+ with the same settings in both papers. Specimens are 6 mm net, 3 mm thick, ±45 raster, at p = 0/30/45 and R = −1/0. The notches are printed: U r3, U r1, and V 35° with r 0.15. All 18 notched series have a matching plain series.

Two caveats limit what this dataset can show:

- The plain gauge length is not given in either paper, so Weibull is run at L_g = 15, 30 and 60 mm. One factor of 2 in L_g moves the prediction by 2^(1/m).
- The plain series scatter is large. For example, p0 R−1 is 10.4 MPa against 4.5 MPa for p30 R−1. Several notched series beat their plain series: blunt Kf = 0.68–0.73 at p = 30/45.

A clean single-campaign test of plain plus open hole is Seifollahi & Kabir 2026 (Polymers 18(1):1). Its tables are images on the MDPI page and are not yet transcribed.

Gate: with the authors' unified plain curve (σ_max 6.7 MPa) and their L_M = 0.49 mm at 2e6 cycles, the point method on our 3D field gives 3.75 MPa for the sharp notch. The authors report 3.9 MPa, a −4.0% difference (−2.6% on the old CAD-import mesh, which had a lower root peak). Their L was calibrated on that notch, so this verifies the fields against their 2D FE.

Convergence: `python3 -m fatigue.figures --convergence` runs nested levels. The graded default is the finest level and the reference, and coarser levels keep every 2nd, 4th and 8th knot of each class. Going from every 2nd knot (3.5–5.3k unknowns) to the default (10.6–17.3k), the changes are:

| Notch | Change in root Kt (thickness mean) | Change in the TCD point stress |
| --- | --- | --- |
| U r3 | −0.5% | +0.2% |
| U r1 | −0.6% | −0.2% |
| V r0.15 | −0.6% | −0.5% |

Kt at mid-thickness is 1.621, 2.626 and 6.555. A further refinement of the V (h0/2, 22 thickness spans) changes it by +0.05%. The old CAD-import mesh under-read the V root by 9–14%, while its bisector stress beyond about 0.03 mm was within 1%, so the transfer table did not move: Weibull L_g 30 gives −8.5% mean error, with 11/18 series within 20%.

Mean stress: `--walker R` calibrates on the plain series at R only. It maps the plain strength to every other R through a Walker exponent fitted per angle on the full plain table, σ_ar = σ_max^(1−γ) σ_a^γ. The fitted γ is 0.25, 0.68 and 0.33 at p = 0, 30 and 45°. On the 9 R = 0 notched series, calibrating from R = −1 moves the Weibull (L_g 30) mean error from −11% to −31%.

Walker collapses the plain data per angle. The scatter of the four R levels about one curve is a log-sd of 0.09, 0.15 and 0.09, inside each series' own scatter of 0.10, 0.23 and 0.15. The notched data show almost no mean-stress effect: a γ fitted to the notched R = −1/0 pairs comes out at about 0.9–1.1, meaning amplitude only. So the plain γ is not applied by default. The likely cause is local mean-stress relaxation at the root, which a linear-elastic transfer cannot represent.
