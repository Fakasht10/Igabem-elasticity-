"""grade_dat.py: knot insertion keeps the surface exact and conforming, and grades symmetrically.
python3 -m pytest python/test_grade_dat.py -q  (~2 s)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grade_dat as G  # noqa: E402
from make_geometries import read_dat  # noqa: E402
from spline_checks import conformity_gap, evaluate  # noqa: E402

GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo")


def _same_surface(A, B):
    rng = np.random.default_rng(0)
    err = 0.0
    for P, Q in zip(A, B):
        for u, v in rng.random((20, 2)):
            uu, vv = P.U[0] + u * (P.U[-1] - P.U[0]), P.V[0] + v * (P.V[-1] - P.V[0])
            err = max(err, np.linalg.norm(evaluate(P, uu, vv)[0] - evaluate(Q, uu, vv)[0]))
    return err


def test_plate_graded_towards_clamp():
    A = read_dat(os.path.join(GEO, "plate_t0.2_uniform.dat"))
    B, added = G.grade(read_dat(os.path.join(GEO, "plate_t0.2_uniform.dat")), [0], 0.025, 1.4)
    assert added > 0
    assert conformity_gap(B)[0] < 1e-12 and conformity_gap(B)[2] == 0
    assert _same_surface(A, B) < 1e-12
    # the face y = -1/2 (patch 2) has new knots near x = 0 only, first element h0 = 0.025
    u = np.unique(B[2].U) * 3.0
    assert abs(u[1] - 0.025) < 1e-9 and np.all(u[u > 0.5] % 0.125 < 1e-9)


def test_bracket_graded_both_ways():
    path = os.path.join(GEO, "step", "bracket_exact_p3_h0.7_c3.dat")
    ids = G.read_faces(path[:-4] + ".faces")
    A = read_dat(path)
    T = [k for k, f in enumerate(ids) if f == 6]
    inner = [k for k, f in enumerate(ids) if f in (1, 13)]
    B, _ = G.grade(read_dat(path), T, 0.025, 1.4, 0, inner)
    gap, shared, lonely = conformity_gap(B)
    assert gap < 1e-12 and lonely == 0
    assert _same_surface(A, B) < 1e-12
    # through the clamp's thickness the knots mirror about the mid-plane
    for k in T:
        for K in (B[k].U, B[k].V):
            s = (np.unique(K) - K[0]) / (K[-1] - K[0])
            if len(s) > 3:
                assert np.allclose(np.sort(1 - s), s, atol=1e-9)
