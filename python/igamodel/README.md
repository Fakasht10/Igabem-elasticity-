# igamodel: the model tree

The core of the workbench redesign (ElmerStudio's design used as a blueprint, written from scratch): a typed tree
whose file format is the case file `app/bc_case` already runs (`src/case_io.hpp`). The tree adds no second format.
The tree is standard library only; `backends` and `run` import numpy and the
failure layer when they are imported.

| File | Lines | What |
|---|---|---|
| `model.py` | 423 | `Prop` and `NodeType` schemas (JSON schema per type, for forms and agent tools); regions in case-file keys (`face_id`, `face_ids`, `patches`, `select`); `Node` and `Model` with plain-dict operations (`add`, `set`, `select`, `remove`), each checked before anything changes; snapshot undo and redo; versioned model file |
| `nodes.py` | 142 | case, material, solver, grade, support (fixed, symmetry, spring, rbe2), loadcase, load (force and moment, pressure, traction, bearing, rbe2 master), output; cad and assessment, which live in the model file only |
| `casefile.py` | 193 | case file to tree and back (`merge_case` into an existing tree); the comment block at the top is kept; unknown keys are refused, never dropped |
| `backends.py` | 155 | capability words; `features(model)` says what a model asks for, so a backend refuses a model before any work; `Run` keeps one operator alive and answers each point query for every load case at once; the IgBEM backend (`bc_case --serve`) |
| `run.py` | 163 | `assess(model, base, workdir)`: geometry from the cad node, one solve, the failure layer's Assess chain at each assessment point for each load case |
| `submodel.py` | 166 | a box of the part (STEP ∩ box), meshed at its own h; its cut faces driven exactly by the global solve; ill-posed boxes refused |
| `gate_cicero.py` | 103 | phase-1 gate: Cicero 2020 failure loads through the tree against `failure/validate_cicero.py` |

```python
import igamodel as im
m = im.read_case("cases/bracket_bolted.json")
h = im.History(m)
lc = h.do({"op": "add", "parent": "root", "kind": "loadcase", "props": {"name": "side"}})
h.do({"op": "add", "parent": lc, "kind": "load", "region": {"face_id": 7}, "props": {"force_N": 5.0, "direction": [0, 1, 0]}})
im.write_case(m, "cases/bracket_bolted_side.json")
```

## Checks

- `python3 -m unittest igamodel.test_igamodel` (from `python/`, ctest `model_tree_unit`, < 1 s). Covers:
    - the round trip of every file in `cases/` and of every form documented in `case_io.hpp`
    - `bracket_bolted.json` written back byte for byte
    - 18 refusals of cases bc_case would misread
    - atomic operations, undo and redo, and journal replay
    - no numpy, gmsh or CAD imports
- `python/igamodel/solver_gate.sh <build>/bc_case cases/*.json` runs bc_case on each hand-written case and on the
  tree-written copy, and requires byte-identical CSV and VTU. All six cases pass. Each run takes 0.9 to 84 s;
  `bracket_clamped_graded` is the slowest.

## Assessments (phase 1)

```python
from igamodel import Model, merge_case, run
m = Model()
m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": "geo/step/senb_rho1.step"}})
merge_case(m, case)                                   # supports and load cases, as in a case file
m.apply({"op": "add", "parent": "root", "kind": "material", "props": {"E": 2300.0, "nu": 0.35}})
m.apply({"op": "add", "parent": "root", "kind": "assessment",
         "props": {"point": [22, 5, 0], "card": "ABS-M30-iso", "sigma_uts": 65.0, "rho": 1.0}})
rows = run.assess(m, base=".", workdir="cache/failure")  # one row per assessment and load case
```

- `factor_pm` and `factor_energy_pm` are the multiples of the load case at which the point method (s1 at L/2 =
  sigma_0) and the energy point method (W at R0 = W_c) are met. A load case of one `force_N` load also gets
  `failure_load_N`.
- The geometry is generated from the cad node only when the case has none (`step_auto.py`, cached by the STEP's bytes
  and the settings).
- A card must agree with the material node's E and nu. The solve uses the material node, so a mismatch is refused
  rather than mixing two materials.
- Every load case shares one operator. bc_case answers each point query for all load cases at once, and the per-case
  views share that answer, so a second load case adds no stress evaluation.
- Not yet an assessment method: Weibull (`failure/weibull_parity.py` is still a driver).

## Submodels

```python
m.apply({"op": "add", "parent": "root", "kind": "submodel",
         "props": {"label": "notch", "lo": [20, 3, -3], "hi": [24, 7, 3], "h": 1.0}})
m.apply({"op": "set", "tag": assessment_tag, "key": "submodel", "value": "notch"})
```

How it works:

- The box is cut out of the STEP and meshed at its own h. The global solve stays alive, so one global solve drives every
  submodel and every load case.
- The cut faces take the global displacement. bc_case gives it exactly, by the Somigliana identity, at the points the cut
  faces are interpolated at:
    - `bc_case --greville` writes those points (0.4 s, no assembly).
    - The serve command `displacement` evaluates them.
    - A support's `"field"`, or a case's `"fields"`, reads them back.
  Nothing is interpolated between the two models, and a point missing from the file is an error.
- These boxes are refused:
    - a box face lying in a face of the part, since the cut faces would then be ambiguous
    - a box that reaches a loaded or supported face
    - a box that misses the part or the feature point
    - a box that the probe line leaves
- Supports and loads must be on CAD faces when a model has a submodel, so the box can be checked against them.

Checks:

- `test_field.py` (ctest `field_patch`): uniform tension prescribed as a field on every face of a cube gives the uniform
  stress inside to 1e-5.
- `test_backends.py`: the box refusals and the cut-face detection.
- `gate_cicero.py --submodel` gives the same digits as the hand-run study below.

SENB notch root, point method (paper props). Coarse global = the whole part at h 3:

| rho | model | unknowns | solve (s) | failure load (N) |
|---|---|---|---|---|
| 1 | coarse global | 8,958 | 27 to 55 | 138.552 |
| 1 | global, refinement sphere h 0.5 | 12,328 | 83 | 138.660 |
| 1 | global, refinement sphere h 0.25 | | | step_auto: no valid layout |
| 1 | submodel ±2 mm, h 1 / 0.7 / 0.5 | 6,084 / 6,780 / 10,134 | 15 / 25 / 70 | 138.572 / 138.576 / 138.577 |
| 1 | submodel ±3 mm, h 1 | 6,060 | 14 | 138.599 |
| 0.25 | coarse global | 9,828 | 35 to 48 | 133.167 |
| 0.25 | global, refinement sphere h 0.25 | over 20k | | out of memory (11 GB, dense LU) |
| 0.25 | submodel x ±1, y −1/+2.5 mm, h 0.5 | 12,036 | 76 | 133.236 |

What the table shows:

- The submodel converges to 1e-5 under h refinement, and every model agrees to within 0.1 %.
- The coarse global model is already converged at the notch root, to within 0.05 % (rho 0.25) and 0.1 % (rho 1).
- On this 9k-unknown part the submodel is a refinement that always fits in memory, not a speed-up: its solve costs
  about what the global one does.
- It pays off where a refinement sphere fails:
    - step_auto layouts that collapse
    - a fine notch on a large part beyond the 20k-unknown dense LU
    - re-solving only the box while the notch geometry changes
- Keep the box solve under about 12k unknowns while the global solve is alive (15 GB machine). Use curvature 6, not 12:
  at rho 0.25, curvature 12 gave 7.8k control points and ran out of memory.

## Phase-1 gate

`gate_cicero.py` builds each Cicero radius as a tree by checked operations. It saves the tree as a model file, loads
it back, and runs `run.assess`. Its rows are compared with `failure/validate_cicero.py` run on the same machine.

| run | rho (mm) | point method (N) | energy PM (N) | relative difference |
|---|---|---|---|---|
| paper K_mat, L | 0.25 / 0.5 / 1 / 2 | 133.1670 / 131.1414 / 138.5518 / 155.2359 | 144.8941 / 140.2489 / 143.7362 / 157.0518 | 0 (bit-identical) |
| card ABS-M30-iso, sigma_uts 65 | 1 / 2 | 93.3017 / 104.9780 | 96.8468 / 106.3183 | 0 (bit-identical) |
| two load cases (1 N, 3 N) on one solve, paper props | 2 | 155.2359 both | | factor ratio 3 to 4e-13 |

- For every radius the geometry cache was hit, so the tree called step_auto with the same arguments.
- The direct run here matches the committed `results/failure/cicero.json` to within 2e-9. That file was made on
  another machine.
- The focus sphere has to be the L that validate_cicero is given. L recomputed from K and sigma_0 comes out as
  0.44000000000000006, and Gmsh then makes a mesh with 273 more TET10 nodes at rho 0.25. That moved the failure load
  by 1.5e-8, which is the mesh sensitivity of the result.
- Run times on 4 cores: solve 23 to 48 s, then the Assess chain 7 to 202 s. rho 0.25 is the slowest at 130k TET10
  nodes.

```
python3 -m failure.validate_cicero --work W --out D.json         # ~9 min, 4 radii
python3 -m igamodel.gate_cicero D.json --work W --load-cases     # ~9 min; geometry from W's cache
```
