# Failure layer on IgBEM

nurb-fea-tcd's failure layer (material cards, hot spots, notch apex and bisector, MATS path, CP-series regression,
TCD / Energy TCD / ASED, Weibull) running on IgBEM solutions. The criteria code is vendored unchanged where possible;
`igabem_backend.py` gives it the same result contract as nurb-fea-tcd's `feakit/bem_backend.py` (bemkit).

| file | role |
|---|---|
| `fracture.py`, `weibull.py`, `materials.py` | vendored from nurb-fea-tcd at 553dd2a (MIT, `LICENSE-nurb-fea-tcd`) |
| `data/` | the material cards and sha-gated calibration artifacts the cards load (`data/calibration/PROVENANCE.md`) |
| `fields.py` | helpers vendored from `feakit/bem_backend.py`: TET10 shape functions and 4-point rule, harmonic extension, sharp-edge rings |
| `igabem_backend.py` | `fea_mesh` (TET10 + tri6, Gmsh face tags), `igabem_geometry` (STEP to multipatch, cached), `Solve` (`bc_case --serve`, one LU kept alive; `stress_all` returns every load case from one query), `fields` (the bem_backend dict) |
| `assess.py` | the Assess chain for one feature: snap, MATS path or re-entrant edge, root radius, notch apex, exact line, PM / Energy PM failure loads, CP regression |
| `validate_cicero.py` | Cicero et al. 2020 ABS SENB, four radii, against measured loads |
| `weibull_parity.py` | log J(m) of the PLA UB / US quarters against the NGSolve reductions the PLA cards were fitted on |
| `test_failure.py` | fast unit checks (ctest `failure_layer_unit`) |
| `aniso.py` | print orientation: ply frames (build direction + layup), modes F / T / L (Hashin form, tension only) and P (isotropic), competing-risk weakest link with optional gradient support, build-direction sweep, per-mode TCD point method, orientation cards. Pure numpy, vendorable |
| `prob.py` | failure probability of a predicted load: physical scatter m_s from replicates, model error as a family random effect, P_f(F), load at P |
| `pla_orient.py` | G3/G4: one card for all rasters on the 54 notched plates of Ahmed & Susmel (PLA) or Ng & Susmel (ABS, `--material abs`), three models, campaign split and leave-one-family-out, band coverage |
| `cicero_orient.py` | G6: one per-mode TCD card for the three Cicero ABS layups, in sample and leave-one-layup-out |
| `data/orientation_cards.json` | the fitted cards, their scatter blocks and domains (`*-demo-L` is a demonstration, not data) |
| `test_aniso.py` | G1 (mode P bit-identical to `weibull.py`), G2 (off-axis bar closed form), G5 (PETG +-45), G8, kernels (ctest `failure_orientation_unit`) |

## Local changes to the vendored files

- `fracture.py`: the `fea_gmsh` import (JAX-FEM) is lazy (`_fea()`), so the criteria import without JAX. Only the
  JAX-FEM helpers (`sed_per_tet`, `principal_stresses_per_tet`, `quad_point_samples`, `recovered_nodal_fields`)
  need it, and the IgBEM path never calls them.
- `materials.py`: `_ROOT` is this package; calibration paths point to `data/calibration/`.
- Headers note the source commit and licence. No numerical code was changed.

## Conventions

- TET10 cells are in feakit's (Abaqus) order; `fea_mesh` swaps Gmsh nodes 8 and 9.
- Face ids are Gmsh surface tags. Cases written for nurb-fea-tcd use build123d indices; `feakit_face_map`
  translates them by face centre.
- Stress at TET10 nodes and probes is the exact boundary-integral value. Point sets of 1000 or more go through the
  treecode (`src/treecode.hpp`, ~1e-7 of max |sigma|), nearer than 0.05 h to the surface through the direct
  near-singular evaluator. Displacement inside the body is a harmonic extension for display only.

## Print orientation and failure probability

The isotropic IgBEM stress is kept; anisotropy is in the strength. A print is a build direction d and plies
[theta, theta - 90] about d. Modes per ply (frame e1 filament, e2, e3 = d): F = sqrt(<s11>^2 + (X/S12)^2 (s12^2 +
s13^2)), T = sqrt(<s22>^2 + (Y/S12)^2 s12^2), L = sqrt(<s33>^2 + (Z/S13)^2 (s13^2 + s23^2)). Their hazards add
(one m), so the characteristic load stays closed form. The gradient support n = 1 + sqrt(l |grad s| / s) (Maede,
Schmitz, Gottschalk & Beck 2017) lets m be the physical scatter modulus while the notch effect comes from l.

Results (2026-10-31, `results/fig_orient_validation.png`). The PLA quarters have the drawn circular notch roots
(nurb-fea-tcd #74 found the port's quadratic Bezier / interpolating spline gave 2 rho and 0.61 rho; the 2026-10-30
PLA numbers here were on those roots and are superseded):

| gate | result |
|---|---|
| G1 mode P vs `weibull.py` | bit-identical log J |
| TCD parity with nurb-fea-tcd | isotropic sigma0 at L 2.37 mm on the same 54 plates: point 58.72 MPa (nurb 58.53), line 54.81 (54.97) |
| G3 PLA, one card for 0/30/45, leave-one-family-out RMSE | TCD line F+T 9.1 %, TCD line isotropic 9.4 %, Weibull F+T with support 9.4 %, per-angle isotropic Weibull 9.8 %, TCD point F+T 12.5 % |
| ABS, Ng & Susmel 54 plates, one card for 0/30/45 (`--material abs`, same fields) | TCD parity with nurb-fea-tcd: point sigma0 22.67 MPa (22.5), line 22.25 (22.2) at L 4.07 mm. LOFO: TCD line isotropic 15.0 %, Weibull F+T with support 17.0 %, per-angle Weibull 17.0 %, TCD line F+T 17.3 %. The modes do not pay: replicate scatter is 11.5 % (m_s 11.1) and the raster effect (about 8 % from 0 to 45 deg) is within it. The Cicero card over-predicts these plates by 42 % |
| G4 PLA 90 % band (TCD line F+T) | covers 87 % with scatter + model error (scatter alone 41 %) |
| G6 Cicero ABS, one TCD card for 3 layups | -8.9 .. +7.5 % (rms 4.2 %), leave-one-layup-out rms 7.5 % (per-layup isotropic, recalibrated: 7.7 %) |
| G5 out-of-plane | no build-tilt data found; in-plane PETG +-45 (Popa 2025): -3.2 % |

The PLA orientation card (`pla-as2018-orient`) carries both: a `tcd` block (line method, sigma0_F 113, sigma0_T 57.2
MPa, Y/S12 2.3 at the held L) and the `weibull` block the model tree's weibull node uses (m = m_s, support l 6.5 mm,
S0_T 39.5 MPa, Y/S12 1.9). The models differ by about one point: on these cross-ply prints the transverse ply always
governs, so the filament strength is not identified by the Weibull data (its S0_F is set from the TCD ratio and says
so) and only weakly by TCD. The support length is close to the half-ligament (7.5 mm): it absorbs net-section
yielding, not a material length.

Physical scatter: within-group log-std 0.034 (36 dof), m_s 37.5 (nurb-fea-tcd's value too). The volume-law m (about
4) is not the scatter: with it the 1 % load is 0.36 of the mean, with m_s and the model error about 0.8. Shear
sensitivity Y/S12 is 1.8-2.3 across PLA, ABS and PETG. The model tree's weibull node takes `orientation_card`,
`build_dir`, `layup`.

## Runs

```
cd python
python3 -m failure.validate_cicero      # ~5 min, results/failure/cicero.json (paper's K_mat, L)
python3 -m failure.validate_cicero --card ABS-M30-iso --sigma-uts 65   # cicero_card.json
python3 -m failure.weibull_parity       # ~2 min, card pla-raster-0-weibull, results/failure/weibull_parity.json
python3 plot_failure.py                 # results/fig_failure.png
python3 -m unittest failure.test_failure
python3 -m failure.pla_orient --hmat    # fields 2-4 min per family once (17 min), fit ~12 min with LOFO
python3 -m failure.pla_orient --material abs   # same cached fields, fit ~17 min with LOFO
python3 -m failure.cicero_orient        # 4 solves ~1 min each once, fit seconds
python3 -m igamodel.gate_orient         # weibull node with an orientation card = pla_orient (bitwise)
python3 plot_orient.py                  # figures + VTU; the 1500-direction sweep ~5 min, cached
python3 -m unittest failure.test_aniso
```
