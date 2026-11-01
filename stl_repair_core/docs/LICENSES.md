# Dependency and license inventory

## This package

| Component | License | Notes |
|---|---|---|
| `stl_repair_core` (Python + Rust kernel) | PolyForm Noncommercial 1.0.0 | Derivative of Sutura 0.9.8. Carries Sutura's LICENSE verbatim and the `Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)` line (see `NOTICE`). |

### Licensing caveat (needs the owner's decision)

- The Sutura repository LICENSE says v0.2.0 and later are **PolyForm Noncommercial 1.0.0**, and v0.1.0–v0.1.9 are **Apache-2.0**.
- At the pinned commit, the 90 Python source files carry `SPDX-License-Identifier: Apache-2.0` headers.
- The `sutura-geom` Rust crate declares `MIT OR Apache-2.0` in its Cargo.toml, its pyproject and Sutura's own `docs/ATTRIBUTION.md`, which says it is "separate from Sutura's PolyForm Noncommercial application license".
- This package follows the most restrictive reading, PolyForm NC for everything, until the licensor clarifies. The Rust kernel here was written against the contract of `triangle_intersection.rs`, which lives in the MIT/Apache crate. The Python modules follow the PolyForm-licensed application code.
- **Owner's decision (2026-11-01):** use is for noncommercial research, which PolyForm NC permits. `python/geomio/stlgate.py` calls the package as an **optional runtime import** (`read_stl_gated`, `load(..., gate=...)`, `python -m geomio --gate`). No code is copied into geomio, and geomio works without it.
- **Consequence:** the repository is GPLv3, and PolyForm NC is not GPL-compatible. So the two stay separately licensed works: this directory keeps its own LICENSE and NOTICE, and anyone redistributing geomio for commercial use must leave out `stl_repair_core` and the `--gate` path.
- **Options:** (a) ask the licensor for a permissive grant on this extraction; (b) re-derive from Sutura **v0.1.9** (Apache-2.0), which lacks the native exact core but whose STL intake and Stage 1 chain cover most of this scope; (c) a clean-room rewrite from public algorithms only (Shewchuk predicates, Liepa 2003, Ericson closest point). The current code is already close to (c), but provenance was taken from 0.9.8, so (c) would need an independent author.

## Runtime dependencies (not vendored)

| Dependency | Version tested | License | Used for |
|---|---|---|---|
| Python | 3.13.18 | PSF | runtime |
| NumPy | 2.5.4 | BSD-3-Clause | arrays |
| SciPy | latest pip at build time | BSD-3-Clause | `cKDTree` (weld, distances), `csgraph.connected_components` |

## Rust extension dependencies (`native/Cargo.toml`, statically linked)

| Crate | Version | License |
|---|---|---|
| pyo3 | 0.29.2 | MIT OR Apache-2.0 |
| numpy (rust-numpy) | 0.29.0 | BSD-2-Clause |
| robust | 1.2.0 | MIT OR Apache-2.0 (port of Shewchuk's public-domain predicates) |
| num-bigint / num-rational / num-traits | 0.4 / 0.4 / 0.2 | MIT OR Apache-2.0 |
| ndarray (via rust-numpy) | 0.16 | MIT OR Apache-2.0 |

Build tool: maturin (MIT OR Apache-2.0), Rust toolchain ≥ 1.80.

## Test-only dependencies (independent oracles)

| Dependency | License | Role |
|---|---|---|
| pytest | MIT | runner |
| manifold3d | Apache-2.0 | manifold status, genus, volume, area of reloaded outputs |
| trimesh (+ networkx) | MIT (+ BSD-3) | independent STL reader/merger: watertight, winding, volume, bodies |
| PyMeshLab | GPL-3.0 | VCG self-intersection and topology oracle. Used only in tests as a separate process library, never imported by the package. Needs system `libOpenGL.so.0`. |

## Benchmark data

The 40-mesh corpus (`tests/real-world-samples/` in the pinned Sutura checkout) is **read in place and never copied** into this repository. The 8 Artec scans are CC BY 4.0. The 32 Thingi10K models are CC BY 4.0 or CC0, with a few older CC BY-SA / CC BY-ND files; see Sutura's `tests/real-world-samples/ATTRIBUTION.md`. All synthetic fixtures are generated in code (`tests/fixtures.py`).
