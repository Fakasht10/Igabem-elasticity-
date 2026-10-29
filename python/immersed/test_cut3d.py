"""Regression tests of the cut-cell quadrature (python/immersed): exact cut volumes and polynomial moments of
polyhedral solids on a Cartesian grid, the integration engine for immersed IGA-FEM unit cells.

    python3 -m pytest python/immersed -q      (needs numba and scipy; ~60 s on the first run for the JIT, then ~5 s)

Every reference is independent of cut3d: closed-form box and prism integrals, or a signed tetrahedron fan of the
STL with a conical Gauss rule (exact for degree <= 7)."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("numba")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cut_quadrature as cq  # noqa: E402

EXPS = cq._exponents(3)


# ------------------------------------------------------------------ shapes (closed, outward triangle meshes)
def box(lo, hi, inward=False):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    T = []
    for a, b, c, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        T += [[a, b, c], [a, c, d]]
    T = np.array(T)
    return V, (T[:, ::-1] if inward else T)


def icosphere(r, c, sub):
    t = (1 + 5 ** 0.5) / 2
    V = [np.array(v, float) / np.linalg.norm(v) for v in
         [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
          [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]]
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    for _ in range(sub):
        mid = {}

        def m(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                v = V[a] + V[b]; V.append(v / np.linalg.norm(v)); mid[k] = len(V) - 1
            return mid[k]
        F = [f for a, b, d in F for f in ([a, m(a, b), m(d, a)], [b, m(b, d), m(a, b)], [d, m(d, a), m(b, d)],
                                          [m(a, b), m(b, d), m(d, a)])]
    return np.array(V) * r + np.asarray(c, float), np.array(F)


def prism(r, c, z0, z1, k):
    """k-gon cylinder of circumradius r, axis z through c"""
    th = 2 * np.pi * np.arange(k) / k
    ring = np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)]
    V = np.r_[np.c_[ring, np.full(k, z0)], np.c_[ring, np.full(k, z1)], [[c[0], c[1], z0], [c[0], c[1], z1]]]
    T = []
    for i in range(k):
        j = (i + 1) % k
        T += [[i, j, k + j], [i, k + j, k + i], [2 * k, j, i], [2 * k + 1, k + i, k + j]]
    return V, np.array(T)


# ------------------------------------------------------------------ independent references
def poly_integrals(V, T, exps):
    """int x^a y^b z^c over the polyhedron: signed tets (p0, v0, v1, v2), conical Gauss 4^3 points (degree 7)"""
    g, w = np.polynomial.legendre.leggauss(4); g = 0.5 * (g + 1); w = 0.5 * w
    U, Vv, Wv = np.meshgrid(g, g, g, indexing='ij'); wu = np.einsum('i,j,k->ijk', w, w, w)
    a1 = U.ravel(); a2 = (Vv * (1 - U)).ravel(); a3 = (Wv * (1 - U) * (1 - Vv)).ravel()
    jw = (wu * (1 - U) ** 2 * (1 - Vv)).ravel()                       # Duffy Jacobian on the unit tet
    p0 = V.mean(0); A = V[T[:, 0]] - p0; B = V[T[:, 1]] - p0; C = V[T[:, 2]] - p0
    det = np.einsum('ij,ij->i', A, np.cross(B, C))                       # 6 * signed volume
    X = p0 + a1[None, :, None] * A[:, None] + a2[None, :, None] * B[:, None] + a3[None, :, None] * C[:, None]
    return np.array([np.sum(det[:, None] * jw[None] * X[..., 0] ** a * X[..., 1] ** b * X[..., 2] ** c)
                     for a, b, c in exps])


def box_integrals(lo, hi, exps):
    f = lambda e, l, u: (u ** (e + 1) - l ** (e + 1)) / (e + 1)
    return np.array([f(a, lo[0], hi[0]) * f(b, lo[1], hi[1]) * f(c, lo[2], hi[2]) for a, b, c in exps])


def rule_integrals(q, exps):
    X, W = q['X'], q['W']
    return np.array([np.sum(W * X[:, 0] ** a * X[:, 1] ** b * X[:, 2] ** c) for a, b, c in exps])


def checks(g, tol=1e-12):
    c = g['check']
    assert c['flag_ties'] == 0
    assert abs(c['vol_rel']) < tol
    assert c['closure_max'] < tol


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


def positive_exact(g, q, tol=1e-12):
    """every weight >= 0, no failed fit, and every cell's degree <= 3 moments reproduced (reference-cell units)"""
    assert q['failed'] == 0
    assert q['W'].min() >= 0.0
    h = g['h']; ncell = int(np.prod(g['n']))
    xi = (q['X'] - cq._centres(g, q['cell'])) / h
    Mq = np.stack([np.bincount(q['cell'], q['W'] * xi[:, 0] ** a * xi[:, 1] ** b * xi[:, 2] ** c, ncell)
                   for a, b, c in EXPS], 1) / h ** 3
    assert np.abs(Mq - cq.volume_moments(g, 3) / h ** 3).max() < tol


# ------------------------------------------------------------------ gates
def test_box_off_grid():
    lo, hi = np.array([0.13, 0.21, 0.17]), np.array([0.71, 0.83, 0.90])
    V, T = box(lo, hi); g = cq.cut_cells(V, T, (-0.1,) * 3, 0.1, 12); checks(g)
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    assert rel(rule_integrals(q, EXPS), box_integrals(lo, hi, EXPS)) < 1e-12


def test_box_on_grid_planes():
    """the unit-cell case: every face of the solid on a grid plane, so no cell is really cut"""
    h = 0.25; V, T = box((0, 0, 0), (1, 1, 1)); g = cq.cut_cells(V, T, (-h,) * 3, h, 6); checks(g)
    act = g['active'].reshape(6, 6, 6)
    assert act[1:5, 1:5, 1:5].all() and act.sum() == 64
    assert np.allclose(g['vol'][g['active']], h ** 3, rtol=0, atol=1e-14)
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    assert rel(rule_integrals(q, EXPS), box_integrals((0, 0, 0), (1, 1, 1), EXPS)) < 1e-12
    s = cq.surface_rule(g)
    assert s['orphan'] == 0
    assert abs(s['W'].sum() - 6.0) < 1e-12 and np.abs(s['W'] @ s['N']).max() < 1e-12


def test_sphere_polyhedral_moments():
    V, T = icosphere(0.4, (0.503, 0.497, 0.511), 3)                       # 1280 facets, off the grid
    g = cq.cut_cells(V, T, (0.0,) * 3, 1 / 16, 16); checks(g)
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    assert rel(rule_integrals(q, EXPS), poly_integrals(V, T, EXPS)) < 1e-11
    assert q['points_per_cut_cell'] <= len(EXPS)          # NNLS is sparse: <= nmom points per cell
    # facet sag only: the 1280-facet polyhedron is 0.86% under the true ball volume
    assert abs(g['check']['vol_sum'] / (4 / 3 * np.pi * 0.4 ** 3) - 1) < 1e-2


def test_cylinder_volume_and_moments():
    k, r, c, z0, z1 = 64, 0.3, (0.52, 0.47), 0.11, 0.87
    V, T = prism(r, c, z0, z1, k); g = cq.cut_cells(V, T, (0.0,) * 3, 1 / 20, 20); checks(g)
    vol = 0.5 * k * r ** 2 * np.sin(2 * np.pi / k) * (z1 - z0)
    assert abs(g['check']['vol_sum'] / vol - 1) < 1e-12
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    assert rel(rule_integrals(q, EXPS), poly_integrals(V, T, EXPS)) < 1e-11


def test_hollow_box_volume_fraction():
    """a thin-walled shell around a void, the infill-like case: wall 0.3 h thick, so cells hold two surfaces"""
    h = 1 / 10; t = 0.3 * h
    Vo, To = box((0.12, 0.12, 0.12), (0.88, 0.88, 0.88)); Vi, Ti = box((0.12 + t,) * 3, (0.88 - t,) * 3, inward=True)
    V = np.r_[Vo, Vi]; T = np.r_[To, Ti + len(Vo)]
    g = cq.cut_cells(V, T, (0.0,) * 3, h, 10); checks(g)
    vol = 0.76 ** 3 - (0.76 - 2 * t) ** 3
    assert abs(g['check']['vol_sum'] / vol - 1) < 1e-12
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    ref = box_integrals((0.12,) * 3, (0.88,) * 3, EXPS) - box_integrals((0.12 + t,) * 3, (0.88 - t,) * 3, EXPS)
    assert rel(rule_integrals(q, EXPS), ref) < 1e-11


def test_degree_guard_and_margin():
    V, T = box((0.2,) * 3, (0.8,) * 3); g = cq.cut_cells(V, T, (0.0,) * 3, 0.1, 10)
    with pytest.raises(ValueError):
        cq.volume_rule(g, 4)
    with pytest.raises(ValueError):
        cq.cut_cells(V, T, (0.3,) * 3, 0.1, 10)


def test_thin_slivers_positive():
    """faces 1e-3 h past a grid plane: sliver cells with almost no Gauss point inside; the boundary candidates carry
    the fit"""
    h = 0.1; e = 1e-3 * h
    lo, hi = np.array([0.2 - e, 0.3 - e, 0.25]), np.array([0.7 + e, 0.6 + e, 0.65 + e])
    V, T = box(lo, hi); g = cq.cut_cells(V, T, (0.0,) * 3, h, 10); checks(g)
    q = cq.volume_rule(g, 3); positive_exact(g, q)
    assert rel(rule_integrals(q, EXPS), box_integrals(lo, hi, EXPS)) < 1e-12


def test_minnorm_negative_nnls_not():
    """the reason for the NNLS rule: the minimum-norm fit on the Gauss grid has negative weights on the same cells"""
    V, T = icosphere(0.4, (0.503, 0.497, 0.511), 2); g = cq.cut_cells(V, T, (0.0,) * 3, 1 / 8, 8)
    old = cq.volume_rule(g, 3, method='minnorm'); new = cq.volume_rule(g, 3)
    assert old['W'].min() < 0 and new['W'].min() >= 0
    assert rel(rule_integrals(new, EXPS), rule_integrals(old, EXPS)) < 1e-12


def test_inside_parity():
    lo, hi = np.array([0.13, 0.21, 0.17]), np.array([0.71, 0.83, 0.90])
    V, T = box(lo, hi); g = cq.cut_cells(V, T, (-0.1,) * 3, 0.1, 12)
    X = np.random.default_rng(1).uniform(0.0, 1.0, (20000, 3))
    assert (cq.inside(g, X) == ((X > lo) & (X < hi)).all(1)).all()
