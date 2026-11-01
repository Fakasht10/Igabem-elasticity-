# stl_repair_core

Conservative STL intake, diagnosis, repair and verification for
structural-mechanics preprocessing. It is extracted from
[Sutura 0.9.8](https://github.com/Krateian/Sutura/commit/158d9ab688fd43098cdd7fe8ad58f8f0049c34f9).
The package is Python with a small Rust kernel for exact predicates and is licensed
**PolyForm Noncommercial 1.0.0**; see `LICENSE`, `NOTICE` and `docs/LICENSES.md`.

The package repairs only what can be repaired without inventing a solid:
duplicate and degenerate faces, near-coincident vertices (explicit tolerance),
orientation, fin faces on non-manifold edges, small holes bounded by an explicit
diameter, and debris below an explicit area. Every operation is a candidate that must pass
the adoption gates. If the result is still not a valid surface, the **original file is
returned** with a FAIL report.

## Build and test (one command)

```sh
./run_all.sh     # fresh venv -> pip install ".[test]" (builds the Rust kernel) -> pytest
```

Requirements: Python ≥ 3.10, a Rust toolchain ≥ 1.80, and `libOpenGL.so.0` for the
PyMeshLab test oracle (`apt install libopengl0 libglu1-mesa`).

## API

```python
from stl_repair_core import repair_stl, validate_stl

result = repair_stl(
    "part.stl", "part_fixed.stl",
    max_hole_diameter=0.5,        # model units; larger loops are intentional openings (0 = no filling)
    weld_tolerance=1e-6,          # fraction of bbox diagonal (0 = bit-exact merge only)
    debris_area_threshold=0.0,    # model units^2; 0 = never remove components
    max_surface_deviation=None,   # model units; None -> 1e-5 x bbox diagonal
    max_volume_change=1e-4,       # fraction of |V|
    protected_faces={"contact": [...], "bolt_hole_1": [...]},   # input facet ids
    json_path="report.json",
)
result["status"]        # "PASS" | "FAIL"
```

## CLI

```sh
stl-repair-core validate input.stl --json report.json
stl-repair-core repair input.stl -o repaired.stl --json report.json \
    [--max-hole-diameter D] [--weld-tolerance T] [--debris-area-threshold A] \
    [--max-surface-deviation E] [--max-volume-change V] \
    [--protected regions.json] [--protect-tags] [--require-watertight]
```

The exit code is 0 for PASS and 1 for FAIL. `--protect-tags` protects every non-zero binary
attribute word (geomio writes CAD face ids there) and every named ASCII `solid` block.

## geomio

`python/geomio/stlgate.py` uses this package as an optional gate in front of geomio's STL reader. It keeps CAD face ids, and patch faces take the id of the best-aligned tagged neighbour:

```python
m, info = geomio.read_stl_gated("part.stl", max_hole_diameter=0.5, report="r.json")   # raises StlGateError on FAIL
```

## Report (JSON)

`status`, `problems`, `before`/`after` diagnosis, `adopted_operations` and
`rejected_operations` (each with its defect vector, deviation statistics,
local dV and reasons), `thresholds`, `policy_hash`, input and output `sha256` and
order-invariant `geometry_hash`, `provenance` (output-to-input facet map
statistics and per-region recovery), `reload` (re-read and re-check of the written
file), `runtime_s` with a breakdown, and `report_hash` (deterministic; excludes
wall-clock fields). The full output-to-input facet map is written next to the
output as `<output>.provenance.npz`.

See `docs/REPORT.md` for verified capability, measured results, limitations
and recommended tolerances. See `docs/MAPPING.md` for the mapping from each Sutura
function and every behavioural difference.
