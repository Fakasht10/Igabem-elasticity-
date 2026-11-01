# stl_repair_core: engineering report

Extracted from Sutura 0.9.8 (`158d9ab`). Date: 2026-11-01.

## Executive summary

- `stl_repair_core` is a **gatekeeper**. Every STL it emits is finite, closed or opened only where the input was, consistently oriented, free of proper self-intersections and reloadable. Otherwise it returns the original file byte for byte with a FAIL report that names every defect.
- It **repairs only defects with a single correct answer**: duplicates, degenerate caps, explicit-tolerance welds, orientation, fin faces, holes below an explicit diameter, and debris below an explicit area. Each repair is adopted only if no defect class gets worse and the measured surface deviation and volume change stay within tolerance.
- **It does not resolve self-intersections, overlapping shells, or coplanar overlaps.** On the 40-mesh Sutura corpus, 21 of the 25 FAILs come from proper self-intersections in the input. The 15 PASSes were already clean. Sutura's full pipeline makes 38 of 40 watertight, but 15 of those still carry proper self-intersections, and 12 deviate by more than 1e-3 × diagonal. That is the trade this package refuses to make for FEA/BEM input.
- One command builds and tests from a clean venv: `./run_all.sh` takes 55 s (86 tests, 19 s of pytest).
- **License:** PolyForm Noncommercial 1.0.0, inherited. It is not linked into geomio. The options are in `docs/LICENSES.md`.

## Verified capability

| Check | Result | Evidence |
|---|---|---|
| Clean build + tests | 86/86 pass, 55 s wall | `run_all.sh`, `benchmarks/pytest_junit.xml` |
| Exact classifier, Rust vs. Fraction reference | identical on all 66 168 pairs of 6 random lattice soups (heavy coplanarity) + 3 full repairs | `tests/test_classifier.py` |
| Synthetic defect matrix | cube, missing triangle, reversed face, duplicates, zero-area / needle, NM fin, pinched vertex, proper + coplanar SI, multi-component, debris kept and removed, thin wall: expected PASS/FAIL and counts | `tests/test_repair_cases.py` |
| Bracket with bolt holes and protected face IDs | 0.559 load hole closed, 0.6 bolt holes kept, every protected region recovered 100 %, attribute words preserved | `tests/test_bracket.py` |
| Order invariance | 4 random vertex/face permutations: identical status, counts and geometry hash | `tests/test_permutation.py` |
| Independent oracles on every written file | manifold3d (status, genus, volume), trimesh (watertight, winding, bodies), PyMeshLab (SI, NM) | `tests/oracles.py` |
| Intake hardening | truncated, bad count, NaN/Inf, oversize, mixed ASCII grammar, empty: rejected with a clear error | `tests/test_intake.py` |
| Sutura 0.9.8 test suites (26 relevant) | 25 pass, 1 fail (`test_engine_no_pymeshlab` needs PySide6, the GUI, out of scope) | `benchmarks/sutura_tests_0.9.8.json` |
| SI cross-check vs. PyMeshLab | zero / non-zero agreement on 38/40 corpus meshes | `benchmarks/si_crosscheck.json` |

## Corpus comparison (40 meshes, same yardstick for every tool)

| | stl_repair_core | Sutura Stage 1 | Sutura full |
|---|---|---|---|
| Valid output (PASS / watertight) | 15 | 35 | 38 (1 timeout) |
| Proper SI left in output | 0 (FAIL instead) | 22 meshes | 15 meshes |
| Healthy components lost or changed | 0 | 5 | 10 |
| Deviation > 1e-3 × diagonal | 0 (all outputs = input) | 4 | 12 |
| Runtime median / max | 4.5 s / 58 s (now 24 s) | – | – |

Reasons for our 25 FAILs: 21 have proper self-intersections in the input. The other four are coplanar shell overlaps (2), 226 degenerate faces on overlapping coplanar shells (1), and 32 NM edges with complex loops that are not unambiguous fins (1). Full table: `benchmarks/corpus_comparison.md`.

## Limitations

- No self-intersection repair, no remeshing, no simplification, no shell merging. Meshes exported from CAD with overlapping bodies FAIL. The fix belongs upstream: export with a union, or use the STEP path.
- Hole filling uses Liepa minimum-dihedral triangulation, capped at 100 loop vertices, without refinement. A filled hole is flat-ish, so the fill is suitable only for small holes. The surface-deviation gate cannot measure invented area, so the only gate on a fill is `max_hole_diameter`.
- Pinched vertices cannot be represented in STL and are reported as FAIL.
- Deviation is measured only on changed faces, at 10 barycentric samples per face, using the exact point-to-triangle distance. It is not a Hausdorff bound.
- Runtime is dominated by the SI scan, about 1–15 s per 10⁵ faces. The worst corpus mesh took 24 s.

## Recommended safe tolerances (FEA / BEM preprocessing)

| Parameter | Default | Printed ABS/PLA part (mm) | Note |
|---|---|---|---|
| `weld_tolerance` | 1e-6 × diag | 1e-6 × diag | above ~1e-5 risks collapsing thin features |
| `max_hole_diameter` | 0 (off) | 0.5 mm, or below the smallest intentional hole | everything larger is treated as an opening |
| `debris_area_threshold` | 0 (off) | 1 mm² | only components that are not healthy |
| `max_surface_deviation` | 1e-5 × diag | 0.01 mm | about 1/20 of a 0.2 mm layer |
| `max_volume_change` | 1e-4 | 1e-4 | |
| `--require-watertight` | off | on | for solid FEA/BEM input |
| `--protect-tags` | off | on with geomio STL | keeps CAD face ids on the attribute word |

## Reproduce

```sh
./run_all.sh                                                     # build + 86 tests
python tools/run_sutura_tests.py <sutura-checkout> benchmarks/sutura_tests_0.9.8.json
python tools/compare_corpus.py <sutura-checkout> benchmarks/corpus_comparison.json --jobs 4
python tools/si_crosscheck.py  <sutura-checkout> benchmarks/si_crosscheck.json
```
