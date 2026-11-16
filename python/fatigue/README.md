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

Convergence: `python3 -m fatigue.figures --convergence` runs nested levels 4…0, where level 0 is the graded default and the reference. Each coarser level removes knots from the one below. Near the root it drops one ring per level: the c knots nearest the root on each graded class (radial, flank, ligament, the root half of the arc) and the c layers nearest each free face. Classes away from the root keep every 2^c-th knot. The grip rows (y > Y1) grow from h to h_far = 4h at every level. Unknowns fall about 1.4× per level (sharp: 3.3k, 4.2k, 5.9k, 8.1k, 12.5k).

| Root Kt (thickness mean) vs level 0 | L4 | L3 | L2 | L1 |
| --- | --- | --- | --- | --- |
| U r3 | −1.7% | −1.2% | −1.2% | −0.9% |
| U r1 | −10.8% | −6.9% | −3.4% | −1.0% |
| V r0.15 | −8.0% | −8.0% | −2.9% | −1.0% |

The TCD point stress moves by less than 2.5% at every level and every notch. A conforming tensor-product multipatch still carries the root's flank columns and thickness layers to the grip, about 35% of the V's unknowns. Only a non-conforming seam would stop that.

Kt at mid-thickness is 1.621, 2.626 and 6.555. A further refinement of the V (h0/2, 22 thickness spans) changes it by +0.05%. The old CAD-import mesh under-read the V root by 9–14%, while its bisector stress beyond about 0.03 mm was within 1%, so the transfer table did not move: Weibull L_g 30 gives −8.5% mean error, with 11/18 series within 20%.

Mean stress: `--walker R` calibrates on the plain series at R only. It maps the plain strength to every other R through a Walker exponent fitted per angle on the full plain table, σ_ar = σ_max^(1−γ) σ_a^γ. The fitted γ is 0.25, 0.68 and 0.33 at p = 0, 30 and 45°. On the 9 R = 0 notched series, calibrating from R = −1 moves the Weibull (L_g 30) mean error from −11% to −31%.

Walker collapses the plain data per angle. The scatter of the four R levels about one curve is a log-sd of 0.09, 0.15 and 0.09, inside each series' own scatter of 0.10, 0.23 and 0.15. The notched data show almost no mean-stress effect: a γ fitted to the notched R = −1/0 pairs comes out at about 0.9–1.1, meaning amplitude only. So the plain γ is not applied by default. The likely cause is local mean-stress relaxation at the root, which a linear-elastic transfer cannot represent.

## Print angle and print-resolved Weibull (`fatigue.printed`)

On Ezeh–Susmel the notched strengths hardly change with the manufacturing angle p, while the plain strengths do (R −1: 10.4 / 4.5 / 6.3 MPa). The notches are printed and have a 0.4 mm perimeter bead at the root. Pooling the plain series over p at the same R (`weibullangle`, Lg 30) is the best blind transfer: +1%, rms log 0.198, 13/18 within 20%. Matching on p gives 11/18 at rms 0.275.

`python3 -m fatigue.printed` sums the hazards over two regions and two failure modes. The regions are the contour bead, with its frame along the edge, and the infill at ±45 − p. The modes are F (filament) and T (filament bond). S_T and the shear ratio are calibrated on the plain series at each R, and S_F comes from the static PLA card.

| run | mean | rms log | within ±20% |
|---|---|---|---|
| blind | +27% | 0.317 | 6/18 |
| S_F/S_T fitted, leave one notch out | +2% | 0.252 | 9/18 |
| S_F/S_T fitted on all 18 (1.21) | +2% | 0.192 | 12/18 |

The result is negative and stable for t_c from 0.2 to 0.8 mm:
- The model learns the plain angle effect as an infill-T effect and carries it into the notches, which don't show it.
- The fitted S_F/S_T of about 1.2 tends toward isotropic.

Bead-level localisation tensors would only rescale S0 per mode, so they were not built. The field cache now stores the full stress tensor (`sig`) at the TET10 points.

## Seifollahi & Kabir 2026 (`data/seifollahi_kabir.json`, `study_seifollahi.py`)

The specimens are YZ (on-edge) PLA cut from concentric-infill plates, with the filament along the load. The double edge notches R2 and the central hole R2 are laser-cut, so no perimeter wraps them. Both leave a 9 mm net section of the 13 × 3.2 D638-I. Loading is R 0.05, at 60–85 % of UTS, with lives of 6·10³ to 1.1·10⁵. Strengths are taken at N = 2·10⁴ from a Basquin refit of Table 6, which matches the paper's Table 7. Kt (thickness mean) is 2.15 for the edge notch and 2.39 for the hole, while the measured Kf is only 1.12 and 1.15.

| criterion | edge notch | hole |
|---|---|---|
| Kf = Kt | −48 % | −52 % |
| Weibull, m 119 from the reported scatter (2 "representative" lives per level) | −42 % | −46 % |
| Weibull, m 7.5 (Ezeh–Susmel PLA pooled) | +33 % | +29 % |
| m that reproduces the notch | 13.0 | 11.6 |
| cross-calibrated Weibull (m from the other notch) | +6 % | −6 % |
| cross-calibrated TCD PM (L from the other notch, 1.9–2.0 mm) | +1 % | −1 % |
| **TCD with L(N) = 16.4 N^−0.242 = 1.49 mm fitted on Ezeh–Susmel (another printer, PLA and lab): it transfers**, PM / LM | −8 % / +2 % | −11 % / +1 % |

The two notches have nearly the same field (same net section, Kt 2.15 and 2.39), so the cross-calibration is a weak test. Key finding: the TCD length fitted on one PLA data set transfers to another printer, PLA and lab (all 14 lives within ×2). It is "fitted on another data set", not blind.

Full S–N transfer (`python3 -m fatigue.sn_transfer`): each of the 14 notched specimens is predicted from the smooth Basquin curve. Errors are log10 of predicted over measured life.

| criterion | mean log10 error | rms log10 error | within ×2 | within ×3 |
|---|---|---|---|---|
| Kt (blind) | −1.83 | 1.83 | 0/14 | 0/14 |
| Weibull, m 7.5 fitted on Ezeh–Susmel | +0.74 | 0.75 | 0/14 | 0/14 |
| supported Weibull, l 0.713 fitted on Ezeh–Susmel, **m 119 from Seifollahi's own smooth scatter** | +0.16 | 0.18 | 14/14 | 14/14 |
| TCD point, L(N) fitted on Ezeh–Susmel | −0.10 | 0.15 | 12/14 | 14/14 |
| TCD line, L(N) fitted on Ezeh–Susmel | +0.08 | 0.13 | 14/14 | 14/14 |
| *in-sample references, fitted on these notches (not predictions)* | | | | |
| Weibull, m 12 | +0.05 | 0.10 | 14/14 | 14/14 |
| measured Kf per notch (1.124 edge, 1.153 hole), constant in N | +0.01 | 0.06 | 14/14 | 14/14 |

How to read it (independent review, 2026-11-16):
- **×2 does not separate criteria here.** Any constant Kf from 1.10 to 1.20 puts 14/14 within ×2. The paper kept two "representative" lives per level (largest pair ratio 1.34), and there are only 7 stress levels. Compare rms against the in-sample floor of 0.06 instead.
- **The supported Weibull is not a pure transfer.** Its m 119 is m_from_T(1.028), the Seifollahi smooth scatter. `python3 -m fatigue.support_fit` refits l on Ezeh–Susmel for each m. That fit assumes an Ezeh–Susmel plain gauge of 30 mm, which the paper does not give. Ezeh–Susmel cannot pick m: its rms ln error is 0.169–0.170 for every m ≥ 20.

  | m | source | l (mm) | Seifollahi bias / rms log10 | within ×2 |
  |---|---|---|---|---|
  | 7.5 | Ezeh–Susmel scatter | 0.001 | +0.80 / 0.80 | 0/14 |
  | 20 | chosen | 0.159 | +0.35 / 0.36 | 4/14 |
  | 40 | chosen | 0.393 | +0.23 / 0.25 | 12/14 |
  | 119 | Seifollahi smooth scatter | 0.713 | +0.16 / 0.18 | 14/14 |

  At m 119, about 10 Gauss points carry 90 % of the integral. The criterion then acts as a gradient-reduced hot-spot factor, so it depends on how well the volume cloud resolves the peak. The Gauss-point maximum is 2.12, against a surface Kt of 2.15.
- **TCD carries a caveat too.** Ezeh–Susmel's L(N) is a point-method calibration on R = −1 and R = 0 data reaching 2·10⁶ cycles. Here it is applied at R = 0.05 and 10⁴ cycles. The line method, which Ezeh–Susmel did not use, scores best.
- **Cross-calibration is weak here.** Fitting on one notch and predicting the other gives rms 0.07–0.08, but the fitted (A_L, b_L) pairs differ widely. The two notches share a 9 mm net section and nearly the same field, so this test cannot identify L(N).

Figures: `fig_sn_transfer_*_overview` and `fig_sn_transfer_*`. Solid lines are the Basquin fits of the tests (equal to the paper's Table 7 to 0.04 %). Broken lines are transferred predictions. Kt, failed criteria and in-sample references appear only in the table.
