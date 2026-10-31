"""Conforming knot refinement of a multipatch layout of single-span (Bezier) patches, graded toward marked points.

Every patch direction (patch k, u or v) is linked to the directions of the neighbours that share its two edges in
that direction (chords of the layout), with the relative orientation. Each chord class gets one knot set: the union
of what its members ask for, uniform spans plus geometric grading toward a parametric end whose edge point is
marked (graded(x) True: e.g. a bond tip, where the stress is singular). Inserting each class's knots into all its
members keeps every shared edge identical, so the mesh stays conforming (and the + faces of a periodic cell stay
translated copies of the - faces when the marking is periodic).
"""
import numpy as np

from make_geometries import insert_knot_curve
from spline_checks import evaluate


def _edges(P):
    """(direction, side) -> Euclidean control points of the edge running in that direction."""
    X = P.Pw[..., :3] / P.Pw[..., 3:4]
    return {(0, 0): X[0], (0, 1): X[-1], (1, 0): X[:, 0], (1, 1): X[:, -1]}


def _key(x, tol):
    return tuple(np.round(np.asarray(x) / tol).astype(np.int64))


def chord_classes(patches, tol=1e-7):
    """Union-find over (patch, direction) with orientation flips; returns root and flip-to-root per member."""
    n = 2 * len(patches)
    parent, flip = list(range(n)), [0] * n

    def find(i):
        if parent[i] == i:
            return i, 0
        r, f = find(parent[i])
        parent[i], flip[i] = r, flip[i] ^ f
        return r, flip[i]

    def union(i, j, f):  # param_i = f ? 1 - param_j : param_j
        (ri, fi), (rj, fj) = find(i), find(j)
        if ri != rj:
            parent[rj], flip[rj] = ri, fi ^ fj ^ f
        elif fi ^ fj ^ f:
            raise ValueError("inconsistent chord orientation")

    buckets = {}
    for k, P in enumerate(patches):
        for (d, s), E in _edges(P).items():
            buckets.setdefault(frozenset((_key(E[0], tol), _key(E[-1], tol))), []).append((k, d, E))
    for group in buckets.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                (k, d, E), (l, e, F) = group[a], group[b]
                if np.allclose(E, F, atol=tol):
                    union(2 * k + d, 2 * l + e, 0)
                elif np.allclose(E, F[::-1], atol=tol):
                    union(2 * k + d, 2 * l + e, 1)
    return [find(i) for i in range(n)]


def graded_knots(level, near0, near1, grade, q=0.5):
    """Knots in (0, 1): 2^level uniform spans, plus `grade` geometric knots toward each marked end."""
    t = set(np.linspace(0, 1, 2 ** level + 1)[1:-1].round(12))
    h = 1.0 / 2 ** level
    for j in range(1, grade + 1):
        if near0:
            t.add(round(h * q ** j, 12))
        if near1:
            t.add(round(1 - h * q ** j, 12))
    return t


def refine(patches, level=1, graded=None, grade=0, q=0.5, tol=1e-7):
    """Conforming refinement: uniform `level` plus `grade` geometric knots toward ends marked by graded(x)."""
    cls = chord_classes(patches, tol)
    want = {}
    for k, P in enumerate(patches):
        for d in (0, 1):
            ends = [False, False]
            if graded is not None:
                # end s of direction d is graded when one of its corners is marked and the patch leaves that
                # corner across the marked set (the edge along d from it is not itself marked, as the axial
                # direction of a corner surface along a tip line is)
                at = (lambda s, o: evaluate(P, s, o)[0]) if d == 0 else (lambda s, o: evaluate(P, o, s)[0])
                for s in (0.0, 1.0):
                    for o in (0.0, 1.0):
                        if graded(at(s, o)) and not graded(at(0.5, o)):
                            ends[int(s)] = True
            r, f = cls[2 * k + d]
            if f:
                ends = ends[::-1]
            want.setdefault(r, set()).update(graded_knots(level, ends[0], ends[1], grade, q))
    out = []
    for k, P in enumerate(patches):
        Pw, U, V = P.Pw.copy(), np.asarray(P.U, float), np.asarray(P.V, float)
        for d in (0, 1):
            r, f = cls[2 * k + d]
            ts = sorted((1 - t) if f else t for t in want[r])
            for t in ts:
                if d == 0:
                    rows = [insert_knot_curve(P.p, U, Pw[j], t) for j in range(Pw.shape[0])]
                    U, Pw = rows[0][0], np.stack([x[1] for x in rows], axis=0)
                else:
                    cols = [insert_knot_curve(P.q, V, Pw[:, i], t) for i in range(Pw.shape[1])]
                    V, Pw = cols[0][0], np.stack([x[1] for x in cols], axis=1)
        out.append(type(P)(P.p, P.q, U, V, Pw))
    return out
