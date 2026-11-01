# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Areas, volumes, bounding box and bidirectional surface distances.

Distances are exact point-to-triangle distances (Ericson, *Real-Time
Collision Detection* 5.1.5) evaluated at deterministic barycentric samples;
the sampled maximum is therefore a lower bound of the continuous Hausdorff
distance, tight to within the sample spacing reported with it.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def tri_areas(V, F):
    if len(F) == 0:
        return np.zeros(0)
    P = V[F]
    return 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)


def signed_volume(V, F, ref=None):
    """Sum of signed tetra volumes w.r.t. ``ref`` (bbox centre by default);
    reference-independent only for closed surfaces."""
    if len(F) == 0:
        return 0.0
    if ref is None:
        ref = 0.5 * (V.min(0) + V.max(0))
    P = V[F] - ref
    return float(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6.0)


def bbox_diagonal(V):
    return float(np.linalg.norm(V.max(0) - V.min(0))) if len(V) else 0.0


def needle_mask(V, F, aspect=1.0e3):
    """Non-degenerate slivers: longest edge^2 / (2 area) above ``aspect``."""
    if len(F) == 0:
        return np.zeros(0, bool)
    P = V[F]
    L2 = np.max([((P[:, i] - P[:, (i + 1) % 3]) ** 2).sum(1) for i in range(3)], axis=0)
    A2 = 2.0 * tri_areas(V, F)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = L2 / A2
    return (A2 > 0) & (r > aspect)


# ------------------------------------------------------------------ distances
_BARY = None


def _bary(level=3):
    """Barycentric lattice with ``level`` subdivisions (vertices, edge and
    interior points); level 3 -> 10 points per triangle."""
    pts = [(i / level, j / level, 1 - (i + j) / level)
           for i in range(level + 1) for j in range(level + 1 - i)]
    return np.asarray(pts)


def sample_faces(V, F, level=3):
    B = _bary(level)
    P = V[F]                                     # (m,3,3)
    S = np.einsum("kj,mjd->mkd", B, P)           # (m,K,3)
    return S.reshape(-1, 3), len(B)


def closest_point_dist(p, a, b, c):
    """Vectorised exact distance from points p to triangles (a,b,c)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    out = np.empty_like(p)
    done = np.zeros(len(p), bool)

    def put(mask, q):
        nonlocal done
        m = mask & ~done
        out[m] = q[m]
        done |= m
    put((d1 <= 0) & (d2 <= 0), a)
    put((d3 >= 0) & (d4 <= d3), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[:, None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        w = d2 / (d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[:, None] * ac)
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w2[:, None] * (c - b))
        den = 1.0 / (va + vb + vc)
        v3, w3 = vb * den, vc * den
        put(np.ones(len(p), bool), a + ab * v3[:, None] + ac * w3[:, None])
    # degenerate triangles: fall back to the nearest of the three edges
    bad = ~np.isfinite(out).all(1)
    if bad.any():
        dd = np.full(bad.sum(), np.inf)
        for u, w_ in ((a, b), (b, c), (c, a)):
            uu, ww, pp = u[bad], w_[bad], p[bad]
            e = ww - uu
            ee = (e * e).sum(1)
            t = np.clip(np.divide(((pp - uu) * e).sum(1), ee, out=np.zeros_like(ee), where=ee > 0), 0, 1)
            dd = np.minimum(dd, np.linalg.norm(pp - (uu + t[:, None] * e), axis=1))
        res = np.linalg.norm(p - out, axis=1)
        res[bad] = dd
        return res
    return np.linalg.norm(p - out, axis=1)


class SurfaceIndex:
    """Exact point-to-surface distance query.

    Candidate triangles for a query point p are those whose centroid lies
    within ``d_c + R`` of p, where ``d_c`` is the distance to the nearest
    centroid (an upper bound of the true distance) and ``R`` the largest
    centroid-to-vertex radius of the indexed (small) triangles.  Triangles
    with radius above ``4 x median`` are tested exhaustively, so the query is
    exact for any size mix.
    """

    def __init__(self, V, F):
        self.P = V[F] if len(F) else np.zeros((0, 3, 3))
        self.m = len(F)
        if self.m == 0:
            return
        cen = self.P.mean(1)
        rad = np.linalg.norm(self.P - cen[:, None, :], axis=2).max(1)
        cut = 4.0 * float(np.median(rad)) if self.m else 0.0
        self.big = np.flatnonzero(rad > cut)
        self.small = np.flatnonzero(rad <= cut)
        self.R = float(rad[self.small].max()) if len(self.small) else 0.0
        self.tree = cKDTree(cen[self.small]) if len(self.small) else None

    def distance(self, Q, chunk=20000):
        Q = np.asarray(Q, np.float64)
        if self.m == 0:
            return np.full(len(Q), np.inf)
        out = np.full(len(Q), np.inf)
        for s in range(0, len(Q), chunk):
            q = Q[s:s + chunk]
            best = np.full(len(q), np.inf)
            if len(self.big):
                for b in self.big:
                    a, bb, c = (np.broadcast_to(self.P[b, k], q.shape) for k in range(3))
                    best = np.minimum(best, closest_point_dist(q, a, bb, c))
            if self.tree is not None:
                dc, _ = self.tree.query(q)
                rad = np.minimum(dc, best) + self.R
                lists = self.tree.query_ball_point(q, rad)
                lens = np.fromiter((len(x) for x in lists), np.int64, len(q))
                qi = np.repeat(np.arange(len(q)), lens)
                ti = self.small[np.fromiter((i for x in lists for i in x), np.int64, int(lens.sum()))]
                if len(ti):
                    d = closest_point_dist(q[qi], self.P[ti, 0], self.P[ti, 1], self.P[ti, 2])
                    np.minimum.at(best, qi, d)
            out[s:s + chunk] = best
        return out


def weighted_stats(d, w, total_weight):
    """Area-weighted statistics; area not represented in ``d`` counts as 0."""
    out = {"max": 0.0, "mean": 0.0, "rms": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    if total_weight <= 0:
        return out
    if len(d):
        out["max"] = float(d.max())
        out["mean"] = float((d * w).sum() / total_weight)
        out["rms"] = float(np.sqrt((d * d * w).sum() / total_weight))
        zero_w = max(total_weight - float(w.sum()), 0.0)
        o = np.argsort(d, kind="stable")
        cw = np.cumsum(w[o]) + zero_w
        for q in (50, 95, 99):
            k = np.searchsorted(cw, q / 100.0 * total_weight)
            out["p%d" % q] = float(d[o][min(k, len(o) - 1)]) if q / 100.0 * total_weight > zero_w else 0.0
    return out


def one_sided(src_V, src_F, dst_index: SurfaceIndex, faces=None, level=3):
    """Distances from samples on ``src_F[faces]`` to the indexed surface.
    Returns (distances, area weights per sample, sample spacing)."""
    sel = np.arange(len(src_F)) if faces is None else np.asarray(faces, np.int64)
    if len(sel) == 0:
        return np.zeros(0), np.zeros(0), 0.0
    S, K = sample_faces(src_V, src_F[sel], level)
    A = tri_areas(src_V, src_F[sel])
    w = np.repeat(A / K, K)
    d = dst_index.distance(S)
    P = src_V[src_F[sel]]
    hmax = float(np.max([np.linalg.norm(P[:, i] - P[:, (i + 1) % 3], axis=1).max() for i in range(3)])) / level
    return d, w, hmax
