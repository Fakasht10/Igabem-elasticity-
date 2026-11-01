"""Exact triangle-pair classifier: Rust kernel vs. pure-Python Fraction
reference, analytic cases, and the PyMeshLab self-intersection oracle."""
import numpy as np
import pytest

import fixtures as fx
from stl_repair_core import intersect as X

needs_native = pytest.mark.skipif(not X.native_available(), reason="native kernel not built")


def _cls(P, Q, shared=0):
    V = np.asarray(list(P) + list(Q), float)
    F = np.array([[0, 1, 2], [3, 4, 5]])
    return X.classify_pairs(V, F, [[0, 1]], engine="python")[0]


T0 = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]


@pytest.mark.parametrize("Q,expect", [
    ([(0.5, 0.5, -1), (1, 0.5, 1), (0.5, 1, 1)], X.PROPER),          # Sutura test_si_definition fixture
    ([(1.4, 0.3, 0), (1.6, 0.3, 0), (1.4, 0.5, 0)], X.COPLANAR),     # coplanar, inside
    ([(3, 0, 0), (4, 0, 0), (3, 1, 0)], X.DISJOINT),                  # coplanar, apart
    ([(2, 0, 0), (3, 0, 0), (3, 1, 0)], X.TOUCH),                     # coplanar vertex contact
    ([(0.5, 0.5, 0), (1, 1, 1), (0, 1, 1)], X.TOUCH),                 # vertex on interior
    ([(0.5, 0.5, 1), (1, 1, 1), (0, 1, 1)], X.DISJOINT),
    ([(0.5, -1, 0), (0.5, 1, 0), (0.5, 0, 1)], X.TOUCH),              # edge lies in plane, one side
])
def test_analytic(Q, expect):
    assert _cls(T0, Q) == expect


def test_adjacent_cases():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0.5, 0.5, 0], [0.25, 0, 0]], float)
    F = np.array([[0, 1, 2], [0, 3, 1], [1, 4, 2], [0, 1, 5], [0, 2, 3]])
    pr = [[0, 1], [0, 2], [0, 3], [0, 4]]
    c = X.classify_pairs(V, F, pr, engine="python")
    assert c.tolist() == [X.ADJ_OK, X.ADJ_OK, X.COPLANAR, X.ADJ_OK]
    if X.native_available():
        assert X.classify_pairs(V, F, pr).tolist() == c.tolist()


@needs_native
@pytest.mark.parametrize("seed", range(6))
def test_native_matches_fraction_reference(seed):
    """Random triangles on a coarse integer lattice: frequent coplanarity,
    shared vertices, collinear edges and touching contacts."""
    rng = np.random.default_rng(seed)
    V = rng.integers(-2, 3, size=(40, 3)).astype(float) * 0.5
    F = rng.integers(0, 40, size=(160, 3))
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    pr = np.array([(i, j) for i in range(len(F)) for j in range(i + 1, len(F))])
    a = X.classify_pairs(V, F, pr, engine="native")
    b = X.classify_pairs(V, F, pr, engine="python")
    assert np.array_equal(a, b), np.flatnonzero(a != b)[:10]
    assert set(np.unique(a)) >= {X.DISJOINT, X.ADJ_OK}


@needs_native
def test_broad_phase_complete():
    rng = np.random.default_rng(3)
    V = rng.random((300, 3))
    F = rng.integers(0, 300, size=(400, 3))
    P = V[F]
    lo, hi = P.min(1), P.max(1)
    brute = {(i, j) for i in range(len(F)) for j in range(i + 1, len(F))
             if (lo[i] <= hi[j]).all() and (hi[i] >= lo[j]).all()}
    got = {tuple(x) for x in X.candidate_pairs(V, F).tolist()}
    assert got == brute
    n = X._native
    try:
        X._native = None
        got2 = {tuple(x) for x in X.candidate_pairs(V, F).tolist()}
    finally:
        X._native = n
    assert got2 == brute


@pytest.mark.parametrize("name", ["overlapping_cubes", "poking_triangle", "box"])
def test_against_pymeshlab(tmp_path, name):
    """Faces in proper crossings must be flagged by VCG too; clean -> 0 both."""
    import oracles
    V, F = getattr(fx, name)()
    p = fx.soup_write(tmp_path / "m.stl", V, F)
    si = X.self_intersections(V, F)
    ours = set(np.unique(si["proper"]).tolist())
    ml = oracles.meshlab_check(p)["self_intersecting_faces"]
    if ours:
        assert ml >= len(ours) - 2 and ml > 0
    else:
        assert ml == 0


@needs_native
@pytest.mark.parametrize("name", ["cube_missing_triangle", "overlapping_cubes", "cube_zero_area_cap"])
def test_full_repair_python_engine_matches_native(tmp_path, name):
    """The pure-Python exact reference gives the same repair as the Rust kernel."""
    from stl_repair_core import repair_stl
    V, F = getattr(fx, name)()
    p = fx.soup_write(tmp_path / "m.stl", V, F)
    a = repair_stl(p, tmp_path / "a.stl", max_hole_diameter=1.5, engine="native")
    b = repair_stl(p, tmp_path / "b.stl", max_hole_diameter=1.5, engine="python")
    assert a["status"] == b["status"]
    assert a["output"]["geometry_hash"] == b["output"]["geometry_hash"]
    assert a["before"]["self_intersections"] == b["before"]["self_intersections"]
