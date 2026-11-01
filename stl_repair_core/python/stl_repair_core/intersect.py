# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Exact triangle-triangle self-intersection classification.

Broad phase: a uniform grid over triangle bounding boxes (large triangles are
tested against everything by box overlap), producing every pair whose closed
boxes overlap.  Narrow phase: the Rust kernel ``_native.classify_pairs``
(adaptive-exact orient3d/orient2d + exact rationals); a pure-Python
``fractions.Fraction`` implementation of the same contract is kept as the
reference and as the fallback when the extension is not built.

Classes (Sutura 0.9.8 ``triangle_intersection.rs`` contract):
  ADJ_OK      adjacent faces meeting only in their shared vertex / edge
  TOUCH       boundary contact without crossing or area overlap
  COPLANAR    coplanar faces whose interiors overlap
  PROPER      interiors cross transversally
  DUPLICATE   same three vertex indices (handled as duplicates, not SI)
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

try:
    from . import _native
except ImportError:  # pragma: no cover - exercised only without the extension
    _native = None

DISJOINT, ADJ_OK, TOUCH, COPLANAR, PROPER, DUPLICATE, DEGENERATE = range(7)
NAMES = {DISJOINT: "disjoint", ADJ_OK: "adjacent_contact", TOUCH: "touching",
         COPLANAR: "coplanar_overlap", PROPER: "proper", DUPLICATE: "duplicate",
         DEGENERATE: "degenerate"}


def native_available():
    return _native is not None


# ---------------------------------------------------------------- broad phase
def candidate_pairs(V, F, subset=None, max_cells=64):
    """All face pairs (i<j) with overlapping closed bounding boxes.

    ``subset``: restrict to pairs with at least one face in ``subset``.
    """
    m = len(F)
    if m < 2:
        return np.zeros((0, 2), np.int64)
    P = V[F]
    lo, hi = P.min(1), P.max(1)
    if subset is not None and len(subset) <= 2000:
        # few query faces (local recheck after an add-only operation): direct box test
        out = []
        for f in np.unique(np.asarray(subset, np.int64)).tolist():
            ov = np.flatnonzero((lo <= hi[f]).all(1) & (hi >= lo[f]).all(1))
            ov = ov[ov != f]
            out.append(np.stack([np.minimum(ov, f), np.maximum(ov, f)], 1))
        if not out:
            return np.zeros((0, 2), np.int64)
        pr = np.concatenate(out)
        key = np.unique(pr[:, 0] * m + pr[:, 1])
        return np.stack([key // m, key % m], 1)
    if _native is not None and hasattr(_native, "box_pairs"):
        pr = np.asarray(_native.box_pairs(np.ascontiguousarray(lo), np.ascontiguousarray(hi)), np.int64)
        if len(pr):
            key = pr[:, 0] * m + pr[:, 1]
            o = np.argsort(key, kind="stable")
            pr = pr[o]
        if subset is not None:
            s = np.zeros(m, bool)
            s[np.asarray(subset, np.int64)] = True
            pr = pr[s[pr[:, 0]] | s[pr[:, 1]]]
        return pr.reshape(-1, 2)
    ext = (hi - lo).max(1)
    h = float(np.median(ext)) * 2.0
    span = float((hi.max(0) - lo.min(0)).max())
    if not h > 0:
        h = span / max(1.0, m ** (1 / 3)) if span > 0 else 1.0
    origin = lo.min(0)
    c0 = np.floor((lo - origin) / h).astype(np.int64)
    c1 = np.floor((hi - origin) / h).astype(np.int64)
    ncell = np.prod(c1 - c0 + 1, axis=1)
    big = ncell > max_cells
    small = np.flatnonzero(~big)
    pairs = []
    if len(small):
        # expand each small triangle into the cells it covers
        rep = ncell[small]
        tri = np.repeat(small, rep)
        off = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep)
        d = (c1 - c0 + 1)[tri]
        ox = off % d[:, 0]
        oy = (off // d[:, 0]) % d[:, 1]
        oz = off // (d[:, 0] * d[:, 1])
        cx, cy, cz = c0[tri, 0] + ox, c0[tri, 1] + oy, c0[tri, 2] + oz
        dims = c1.max(0) + 2
        key = (cx * dims[1] + cy) * dims[2] + cz
        o = np.argsort(key, kind="stable")
        key, tri = key[o], tri[o]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        cnt = np.diff(np.r_[starts, len(key)])
        for c in np.unique(cnt):
            if c < 2:
                continue
            s = starts[cnt == c]
            I, J = np.triu_indices(c, 1)
            pairs.append(np.stack([tri[(s[:, None] + I).ravel()], tri[(s[:, None] + J).ravel()]], 1))
    bigi = np.flatnonzero(big)
    for b in bigi:
        ov = np.flatnonzero((lo <= hi[b]).all(1) & (hi >= lo[b]).all(1))
        ov = ov[ov != b]
        if len(ov):
            pairs.append(np.stack([np.full(len(ov), b), ov], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    pr = np.concatenate(pairs)
    key = np.minimum(pr[:, 0], pr[:, 1]) * m + np.maximum(pr[:, 0], pr[:, 1])
    key.sort(kind="stable")
    key = key[np.r_[True, key[1:] != key[:-1]]]
    pr = np.stack([key // m, key % m], 1)
    pr = pr[pr[:, 0] != pr[:, 1]]
    ok = (lo[pr[:, 0]] <= hi[pr[:, 1]]).all(1) & (hi[pr[:, 0]] >= lo[pr[:, 1]]).all(1)
    pr = pr[ok]
    if subset is not None:
        s = np.zeros(m, bool)
        s[np.asarray(subset, np.int64)] = True
        pr = pr[s[pr[:, 0]] | s[pr[:, 1]]]
    return pr


# ---------------------------------------------------------------- narrow phase
def classify_pairs(V, F, pairs, engine="auto"):
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros(0, np.uint8)
    if engine == "python" or (engine == "auto" and _native is None):
        return np.array([_classify_py(V, F, int(f), int(g)) for f, g in pairs], np.uint8)
    return np.asarray(_native.classify_pairs(np.ascontiguousarray(V, np.float64),
                                             np.ascontiguousarray(F, np.int64),
                                             np.ascontiguousarray(pairs)), np.uint8)


def degenerate_mask(V, F, engine="auto"):
    if len(F) == 0:
        return np.zeros(0, bool)
    if engine == "python" or (engine == "auto" and _native is None):
        return np.array([_degenerate_py([tuple(map(Fraction, V[i])) for i in f]) for f in F], bool)
    return np.asarray(_native.degenerate_mask(np.ascontiguousarray(V, np.float64),
                                              np.ascontiguousarray(F, np.int64)), bool)


def self_intersections(V, F, subset=None, engine="auto"):
    """Classify every candidate pair; return dict of counts + per-class pairs."""
    pr = candidate_pairs(V, F, subset=subset)
    cls = classify_pairs(V, F, pr, engine=engine)
    out = {"candidate_pairs": int(len(pr))}
    for c in (ADJ_OK, TOUCH, COPLANAR, PROPER, DUPLICATE, DEGENERATE):
        out[NAMES[c]] = pr[cls == c]
    return out


def si_counts(si):
    keys = ("adjacent_contact", "touching", "coplanar_overlap", "proper")
    out = {k: int(len(si[k])) for k in keys}
    for k in ("touching", "coplanar_overlap", "proper"):
        out[k + "_faces"] = int(len(np.unique(si[k]))) if len(si[k]) else 0
    out["candidate_pairs"] = si["candidate_pairs"]
    return out


# ------------------------------------------------- pure-Python exact reference
def _o3(a, b, c, d):
    ax, ay, az = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bx, by, bz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cx, cy, cz = c[0] - d[0], c[1] - d[1], c[2] - d[2]
    det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return (det > 0) - (det < 0)


def _proj(p, ax):
    return (p[1], p[2]) if ax == 0 else ((p[2], p[0]) if ax == 1 else (p[0], p[1]))


def _o2(a, b, c, ax):
    a, b, c = _proj(a, ax), _proj(b, ax), _proj(c, ax)
    det = (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
    return (det > 0) - (det < 0)


def _degenerate_py(t):
    return all(_o2(t[0], t[1], t[2], ax) == 0 for ax in range(3))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _interval(t, n, q0, d):
    sd = [_dot(n, _sub(p, q0)) for p in t]
    ts = []
    for i in range(3):
        if sd[i] == 0:
            ts.append(_dot(d, t[i]))
        j = (i + 1) % 3
        if sd[i] * sd[j] < 0:
            lam = sd[i] / (sd[i] - sd[j])
            di, dj = _dot(d, t[i]), _dot(d, t[j])
            ts.append(di + (dj - di) * lam)
    cross = any(x > 0 for x in sd) and any(x < 0 for x in sd)
    return ((min(ts), max(ts)) if ts else None), cross


def _same_dir(a, b, d):
    for k in range(3):
        if b[k] != a[k]:
            return (b[k] > a[k]) == (d[k] > a[k]) and d[k] != a[k]
    return False


def _classify_py(V, F, f, g):
    fi, gi = [int(x) for x in F[f]], [int(x) for x in F[g]]
    sh = [(i, j) for i in range(3) for j in range(3) if fi[i] == gi[j]]
    if len(sh) >= 3:
        return DUPLICATE
    p = [tuple(Fraction(float(x)) for x in V[i]) for i in fi]
    q = [tuple(Fraction(float(x)) for x in V[i]) for i in gi]
    if _degenerate_py(p) or _degenerate_py(q):
        return DEGENERATE
    s = [_o3(q[0], q[1], q[2], x) for x in p]
    t = [_o3(p[0], p[1], p[2], x) for x in q]

    def same(v):
        return v[0] != 0 and v[0] == v[1] == v[2]
    if same(s) or same(t):
        return DISJOINT
    if all(x == 0 for x in s):
        n = _cross(_sub(p[1], p[0]), _sub(p[2], p[0]))
        axes = sorted(range(3), key=lambda a: -abs(n[a]))
        ax = next(a for a in axes if _o2(p[0], p[1], p[2], a) != 0)
        op, oq = _o2(p[0], p[1], p[2], ax), _o2(q[0], q[1], q[2], ax)
        area = touch = True
        for tri, other, o in ((p, q, op), (q, p, oq)):
            for k in range(3):
                a, b = tri[k], tri[(k + 1) % 3]
                sides = [_o2(a, b, x, ax) * o for x in other]
                if all(v <= 0 for v in sides):
                    area = False
                if all(v < 0 for v in sides):
                    touch = False
        if area:
            return COPLANAR
        if not touch:
            return DISJOINT
        if len(sh) == 2:
            return ADJ_OK
        if len(sh) == 1:
            i, j = sh[0]
            a = p[i]
            for b in (p[(i + 1) % 3], p[(i + 2) % 3]):
                for d in (q[(j + 1) % 3], q[(j + 2) % 3]):
                    if _o2(a, b, d, ax) == 0 and _same_dir(a, b, d):
                        return TOUCH
            return ADJ_OK
        return TOUCH
    if len(sh) == 2:
        return ADJ_OK
    if len(sh) == 1:
        i, j = sh[0]
        tn = [t[k] for k in range(3) if k != j]
        sn = [s[k] for k in range(3) if k != i]
        if (tn[0] != 0 and tn[0] == tn[1]) or (sn[0] != 0 and sn[0] == sn[1]):
            return ADJ_OK
    n1 = _cross(_sub(p[1], p[0]), _sub(p[2], p[0]))
    n2 = _cross(_sub(q[1], q[0]), _sub(q[2], q[0]))
    d = _cross(n1, n2)
    i1, x1 = _interval(p, n2, q[0], d)
    i2, x2 = _interval(q, n1, p[0], d)
    if i1 is None or i2 is None:
        return DISJOINT
    lo, hi = max(i1[0], i2[0]), min(i1[1], i2[1])
    if lo > hi:
        return DISJOINT
    if lo == hi:
        return ADJ_OK if len(sh) == 1 else TOUCH
    return PROPER if (x1 and x2) else TOUCH
