# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Deterministic synthetic STL fixtures (no external geometry).

Every generator returns (V, F) with float32-representable coordinates and,
where useful, a dict of named facet-id regions.  Injected-defect variants are
built from the clean meshes so the ground truth is known exactly.
"""
from __future__ import annotations

import numpy as np

from stl_repair_core.stlio import write_stl


def f32(V):
    return np.asarray(V, np.float64).astype(np.float32).astype(np.float64)


def box(lo=(0, 0, 0), hi=(1, 1, 1), n=1):
    """Axis-aligned box, each face an n x n grid of quads (2 tris each), outward."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    verts, faces, index = [], [], {}

    def vid(p):
        k = tuple(np.round(p, 12))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]
    for ax in range(3):
        for side in (0, 1):
            u, w = (ax + 1) % 3, (ax + 2) % 3
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        p = np.empty(3)
                        p[ax] = hi[ax] if side else lo[ax]
                        p[u] = lo[u] + (hi[u] - lo[u]) * a / n
                        p[w] = lo[w] + (hi[w] - lo[w]) * b / n
                        return vid(p)
                    q = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)]
                    if side == 0:
                        q = q[::-1]
                    faces += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return f32(verts), np.asarray(faces, np.int64)


def merge(*meshes):
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def soup_write(path, V, F, fmt="binary", attr=None, solid=None, names=None):
    write_stl(path, V, F, fmt=fmt, attr=attr, solid_of_face=solid, solid_names=names)
    return path


# ------------------------------------------------------------------ defects
def cube_missing_triangle(n=2):
    V, F = box(n=n)
    return V, np.delete(F, 3, axis=0)


def cube_reversed_face(n=2):
    V, F = box(n=n)
    F = F.copy()
    F[5] = F[5][[0, 2, 1]]
    return V, F


def cube_duplicates(n=2):
    V, F = box(n=n)
    return V, np.concatenate([F, F[[2]], F[[7]][:, [0, 2, 1]]])


def cube_zero_area_cap():
    """Closed cube with a collinear (zero-area) cap at a T-junction."""
    V, F = box(n=1)
    V = list(V)
    f = 0
    a, b, c = F[f]
    m = len(V)
    V.append((np.asarray(V[a]) + np.asarray(V[b])) / 2)
    F = np.concatenate([np.delete(F, f, axis=0), [[a, m, c], [m, b, c], [a, b, m]]])
    return f32(V), F


def cube_needle():
    """Closed cube where one face carries a needle triangle (valid geometry)."""
    V, F = box(n=1)
    V = list(V)
    a, b, c = F[0]
    mid = (np.asarray(V[a]) + np.asarray(V[b])) / 2
    p = mid + 1e-4 * (np.asarray(V[c]) - mid)
    m = len(V)
    V.append(p)
    F = np.concatenate([np.delete(F, 0, axis=0), [[a, m, c], [m, b, c], [a, b, m]]])
    return f32(V), F


def two_cubes_sharing_edge():
    return merge(box((0, 0, 0), (1, 1, 1)), box((1, 1, 0), (2, 2, 1)))


def two_cubes_sharing_vertex():
    return merge(box((0, 0, 0), (1, 1, 1)), box((1, 1, 1), (2, 2, 2)))


def cube_with_fin():
    """Closed cube plus one triangle hanging off an edge (non-manifold edge)."""
    V, F = box(n=1)
    e = None
    for t in F:
        if V[t[0]][2] == 1 and V[t[1]][2] == 1:
            e = (t[0], t[1])
            break
    tip = (V[e[0]] + V[e[1]]) / 2 + np.array([0, 0, 0.02])
    V2 = np.vstack([V, tip])
    return f32(V2), np.concatenate([F, [[e[0], e[1], len(V)]]])


def overlapping_cubes():
    return merge(box((0, 0, 0), (1, 1, 1)), box((0.5, 0.25, 0.25), (1.5, 0.75, 0.75)))


def poking_triangle():
    V, F = box(n=1)
    V2 = np.vstack([V, [[0.5, 0.5, 0.5], [0.5, 0.5, 1.5], [0.6, 0.5, 1.5]]])
    return f32(V2), np.concatenate([F, [[8, 9, 10]]])


def coplanar_cubes():
    return merge(box((0, 0, 0), (1, 1, 1)), box((1, 0.25, 0.25), (2, 0.75, 0.75)))


def multi_components():
    """Two separate cubes and a nested cavity (inner box inward-facing)."""
    outer = box((3, 0, 0), (5, 2, 2), n=2)
    inner_V, inner_F = box((3.5, 0.5, 0.5), (4.5, 1.5, 1.5))
    return merge(box((0, 0, 0), (1, 1, 1), n=2), outer, (inner_V, inner_F[:, [0, 2, 1]]))


def cube_with_debris():
    V, F = box(n=2)
    dV = f32([[5, 5, 5], [5.01, 5, 5], [5, 5.01, 5], [5.02, 5.01, 5]])
    return merge((V, F), (dV, np.array([[0, 1, 2], [1, 3, 2]])))


def thin_wall(t=1e-3, n=4):
    V, F = box((0, 0, 0), (1, 1, t), n=n)
    top = np.flatnonzero((V[F][:, :, 2] == V[:, 2].max()).all(1))
    return V, np.delete(F, top[1], axis=0)


# ------------------------------------------------------------------ bracket
def bracket(k=8, t=0.25, r=0.3):
    """3 x 1 plate of thickness t with two through bolt holes of radius r
    (cells 0 and 2).  Returns V, F, regions (input facet ids):
    bolt_hole_1, bolt_hole_2 (hole walls), contact (bottom), load (top of cell 1)."""
    N = 4 * k
    verts, index = [], {}

    def vid(x, y, z):
        key = (round(x, 12), round(y, 12), round(z, 12))
        if key not in index:
            index[key] = len(verts)
            verts.append((x, y, z))
        return index[key]

    def square_ring(cx):
        pts = []
        for s in range(4):
            for i in range(k):
                a = i / k
                if s == 0:
                    pts.append((cx - 0.5 + a, -0.5))
                elif s == 1:
                    pts.append((cx + 0.5, -0.5 + a))
                elif s == 2:
                    pts.append((cx + 0.5 - a, 0.5))
                else:
                    pts.append((cx - 0.5, 0.5 - a))
        return pts
    faces, tags = [], []
    for cell in range(3):
        cx = cell + 0.5
        sq = square_ring(cx)
        hole = cell != 1
        if hole:
            ang = np.arctan2(np.array([p[1] for p in sq]), np.array([p[0] - cx for p in sq]))
            circ = [(cx + r * np.cos(a), r * np.sin(a)) for a in ang]
        for z, top in ((t, True), (0.0, False)):
            tag = ("load" if cell == 1 else "top") if top else "contact"
            if hole:
                for i in range(N):
                    j = (i + 1) % N
                    a, b = vid(*sq[i], z), vid(*sq[j], z)
                    c, d = vid(*circ[j], z), vid(*circ[i], z)
                    q = [[a, b, c], [a, c, d]]
                    if not top:
                        q = [x[::-1] for x in q]
                    faces += q
                    tags += [tag, tag]
            else:
                cen = vid(cx, 0.0, z)
                for i in range(N):
                    j = (i + 1) % N
                    tri = [cen, vid(*sq[i], z), vid(*sq[j], z)]
                    faces.append(tri if top else tri[::-1])
                    tags.append(tag)
        if hole:
            name = "bolt_hole_%d" % (1 if cell == 0 else 2)
            for i in range(N):
                j = (i + 1) % N
                a, b = vid(*circ[i], 0.0), vid(*circ[j], 0.0)
                c, d = vid(*circ[j], t), vid(*circ[i], t)
                faces += [[a, d, c], [a, c, b]]   # normals point into the hole (out of material)
                tags += [name, name]
    # outer side walls
    ring = []
    for i in range(3 * k):
        ring.append((i / k, -0.5))
    for i in range(k):
        ring.append((3.0, -0.5 + i / k))
    for i in range(3 * k):
        ring.append((3.0 - i / k, 0.5))
    for i in range(k):
        ring.append((0.0, 0.5 - i / k))
    M = len(ring)
    for i in range(M):
        j = (i + 1) % M
        a, b = vid(*ring[i], 0.0), vid(*ring[j], 0.0)
        c, d = vid(*ring[j], t), vid(*ring[i], t)
        faces += [[a, b, c], [a, c, d]]
        tags += ["side", "side"]
    V = f32(verts)
    F = np.asarray(faces, np.int64)
    tags = np.asarray(tags)
    regions = {nm: np.flatnonzero(tags == nm).tolist() for nm in ("bolt_hole_1", "bolt_hole_2", "contact", "load")}
    return V, F, regions


def random_permutation(V, F, seed=0):
    """Shuffle facet order and rotate each facet's corners (orientation kept);
    also shuffle the vertex array."""
    rng = np.random.default_rng(seed)
    pv = rng.permutation(len(V))
    inv = np.empty_like(pv)
    inv[pv] = np.arange(len(pv))
    V2 = V[pv]
    F2 = inv[F]
    pf = rng.permutation(len(F))
    F2 = F2[pf]
    rot = rng.integers(0, 3, len(F2))
    idx = (np.arange(3)[None, :] + rot[:, None]) % 3
    return V2, np.take_along_axis(F2, idx, axis=1), pf
