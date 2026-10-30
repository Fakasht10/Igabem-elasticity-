# igamodel: the model tree

The core of the workbench redesign (ElmerStudio's design used as a blueprint, written from scratch): a typed tree
whose file format is the case file `app/bc_case` already runs (`src/case_io.hpp`). The tree adds no second format.
Standard library only.

| File | Lines | What |
|---|---|---|
| `model.py` | 423 | `Prop` and `NodeType` schemas (JSON schema per type, for forms and agent tools); regions in case-file keys (`face_id`, `face_ids`, `patches`, `select`); `Node` and `Model` with plain-dict operations (`add`, `set`, `select`, `remove`), each checked before anything changes; snapshot undo and redo; versioned model file |
| `nodes.py` | 111 | case, material, solver, grade, support (fixed, symmetry, spring, rbe2), loadcase, load (force and moment, pressure, traction, bearing, rbe2 master), output |
| `casefile.py` | 185 | case file to tree and back; the comment block at the top is kept; unknown keys are refused, never dropped |

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
