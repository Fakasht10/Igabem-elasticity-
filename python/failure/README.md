# Failure layer on IgBEM

nurb-fea-tcd's failure layer (material cards, hot spots, notch apex and bisector, MATS path, CP-series regression,
TCD / Energy TCD / ASED, Weibull) running on IgBEM solutions. The criteria code is vendored unchanged where possible;
`igabem_backend.py` gives it the same result contract as nurb-fea-tcd's `feakit/bem_backend.py` (bemkit).

| file | role |
|---|---|
| `fracture.py`, `weibull.py`, `materials.py` | vendored from nurb-fea-tcd at 553dd2a (MIT, `LICENSE-nurb-fea-tcd`) |
| `data/` | the material cards and sha-gated calibration artifacts the cards load (`data/calibration/PROVENANCE.md`) |
| `fields.py` | helpers vendored from `feakit/bem_backend.py`: TET10 shape functions and 4-point rule, harmonic extension, sharp-edge rings |
| `igabem_backend.py` | `fea_mesh` (TET10 + tri6, Gmsh face tags), `igabem_geometry` (STEP to multipatch, cached), `Solve` (`bc_case --serve`, one LU kept alive), `fields` (the bem_backend dict) |
| `assess.py` | the Assess chain for one feature: snap, MATS path or re-entrant edge, root radius, notch apex, exact line, PM / Energy PM failure loads, CP regression |
| `validate_cicero.py` | Cicero et al. 2020 ABS SENB, four radii, against measured loads |
| `weibull_parity.py` | log J(m) of the PLA UB / US quarters against the NGSolve reductions the PLA cards were fitted on |
| `test_failure.py` | fast unit checks (ctest `failure_layer_unit`) |

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

## Runs

```
cd python
python3 -m failure.validate_cicero      # ~5 min, results/failure/cicero.json
python3 -m failure.weibull_parity       # ~2 min, results/failure/weibull_parity.json
python3 plot_failure.py                 # results/fig_failure.png
python3 -m unittest failure.test_failure
```
