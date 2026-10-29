"""Automatic STEP -> conforming multipatch B-spline surfaces (Bembel .dat), any B-rep solid.

Unlike python/step2dat.py (exact Coons patches of a hand-made quad layout), the layout here
is automatic and the geometry is fitted:

  1. quad mesh: Gmsh (OpenCASCADE kernel) meshes every CAD face, trimmed or not, with
     quasi-structured quads (few irregular vertices); every CAD edge is a chain of mesh edges;
  2. quad layout: separatrices are traced from every irregular vertex (interior valence != 4,
     curve vertex with != 2 quads, corner with != 1 quad in a face) straight through the
     regular vertices until they hit a CAD edge or another irregular vertex. A separatrix that
     ends on a CAD edge also splits the faces on the other side, so blocks meet side to side.
     CAD edges and separatrices cut the mesh into structured n x m blocks (a block that is
     not a rectangle falls back to one patch per quad);
  3. fit: each block becomes one B-spline patch of degree p with the block's n x m mesh quads
     as knot spans (C^{p-1} inside). Each block side is fitted once, by least squares to
     points projected onto its CAD curve (or face, for a separatrix), and shared by both
     neighbouring blocks, so the patches are conforming and watertight; the interior control
     points are fitted to points projected onto the CAD face with the boundary held fixed;
  4. orientation: x_u x x_v points out of the solid (Gmsh inside test);
  5. checks: distance to the CAD faces, interface gap, volume vs. OpenCASCADE.

usage: python3 python/step_auto.py in.step out.dat [--h 0.4] [--degree 3] [--refine k]
"""

import argparse
import os
import sys
from collections import defaultdict

import gmsh
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import Patch, refine_uniform, write_dat  # noqa: E402
from spline_checks import _basis, conformity_gap, evaluate, spline_volume  # noqa: E402


# ---------------------------------------------------------------- 1. quad mesh
class QuadMesh:
    """All-quad surface mesh of a STEP solid with its CAD classification (gmsh session open)."""

    def __init__(self, step, h, curvature=8):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step)
        gmsh.model.occ.synchronize()
        for k, v in {"Mesh.Algorithm": 11, "Mesh.RecombineAll": 1, "Mesh.MeshSizeMax": h,
                     "Mesh.MeshSizeMin": h / 8, "Mesh.MeshSizeFromCurvature": curvature,
                     "Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeExtendFromBoundary": 0,
                     "Mesh.Smoothing": 20}.items():
            gmsh.option.setNumber(k, v)
        # structured grids on four-sided faces (compatible counts chosen by Gmsh); the rest
        # gets quasi-structured quads
        gmsh.model.mesh.setTransfiniteAutomatic(recombine=True)
        _match_translated_faces()
        gmsh.model.mesh.generate(2)
        self.volumes = [t for _, t in gmsh.model.getEntities(3)]
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        self.x = dict(zip(tags.astype(int), xyz.reshape(-1, 3)))
        self.ndim = {}  # node -> dimension of the CAD entity it lies on
        for d in (2, 1, 0):
            for _, t in gmsh.model.getEntities(d):
                for n in gmsh.model.mesh.getNodes(d, t)[0].astype(int):
                    self.ndim[n] = d
        self.curve_of = {}  # sorted mesh edge on a CAD curve -> curve tag
        for _, c in gmsh.model.getEntities(1):
            for et, _, nodes in zip(*gmsh.model.mesh.getElements(1, c)):
                for a, b in nodes.astype(int).reshape(-1, 2 if et == 1 else 3)[:, :2]:
                    self.curve_of[(min(a, b), max(a, b))] = c
        self.quads, self.surf = [], []  # quad -> 4 nodes (Gmsh orientation), CAD surface tag
        for _, s in gmsh.model.getEntities(2):
            for et, _, nodes in zip(*gmsh.model.mesh.getElements(2, s)):
                if et != 3:
                    raise RuntimeError("surface %d has non-quad elements (type %d)" % (s, et))
                for q in nodes.astype(int).reshape(-1, 4):
                    self.quads.append(list(q))
                    self.surf.append(s)
        tags = np.concatenate([np.concatenate(gmsh.model.mesh.getElements(2, s)[1])
                               for _, s in gmsh.model.getEntities(2)])
        self.min_quality = float(np.min(gmsh.model.mesh.getElementQualities(tags, "minSICN")))
        if self.min_quality < 0.1:
            raise RuntimeError("quad mesh has a degenerate or inverted quad (min SICN %.3f)" % self.min_quality)
        self.edge_quads = defaultdict(list)
        for i, q in enumerate(self.quads):
            for k in range(4):
                self.edge_quads[_key(q[k], q[(k + 1) % 4])].append(i)
        self.node_quads = defaultdict(list)
        for i, q in enumerate(self.quads):
            for n in q:
                self.node_quads[n].append(i)

    def hard(self, e):
        return e in self.curve_of


def _match_translated_faces(tol=1e-7):
    """Copy the mesh of a face onto every face that is its exact translate (the two caps of a
    prismatic part): their irregular vertices then line up, so their separatrices coincide
    instead of cutting each other's blocks."""
    faces = [t for _, t in gmsh.model.getEntities(2)]
    info = {f: (np.array(gmsh.model.getBoundingBox(2, f)), gmsh.model.occ.getMass(2, f),
                np.array(gmsh.model.occ.getCenterOfMass(2, f))) for f in faces}
    slaves = set()
    for i, a in enumerate(faces):
        for b in faces[i + 1:]:
            if a in slaves or b in slaves:
                continue
            (ba, ma, ca), (bb, mb, cb) = info[a], info[b]
            d = cb - ca
            if abs(ma - mb) > tol * max(ma, 1.0) or np.linalg.norm(d) < tol \
                    or np.abs((bb - ba) - np.tile(d, 2)).max() > tol:
                continue
            if len(gmsh.model.getBoundary([(2, a)])) != len(gmsh.model.getBoundary([(2, b)])):
                continue
            T = [1, 0, 0, d[0], 0, 1, 0, d[1], 0, 0, 1, d[2], 0, 0, 0, 1]
            try:
                gmsh.model.mesh.setPeriodic(2, [b], [a], T)
                slaves.add(b)
            except Exception:
                pass


def _key(a, b):
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------- 2. quad layout
def _irregular(M, n, s):
    k = sum(M.surf[q] == s for q in M.node_quads[n])
    return k != {2: 4, 1: 2, 0: 1}[M.ndim[n]]


def _edges_at(M, n, s):
    """Mesh edges of surface s at node n."""
    out = set()
    for q in M.node_quads[n]:
        if M.surf[q] != s:
            continue
        Q = M.quads[q]
        i = Q.index(n)
        out |= {_key(n, Q[(i + 1) % 4]), _key(n, Q[(i - 1) % 4])}
    return out


def trace_layout(M):
    """Separatrix edges (a set of sorted node pairs)."""
    sep = set()
    todo = []
    for n in M.x:
        for s in {M.surf[q] for q in M.node_quads[n]}:
            if _irregular(M, n, s):
                todo += [(n, e, s) for e in _edges_at(M, n, s) if not M.hard(e)]
    while todo:
        n, e, s = todo.pop()
        while e not in sep:
            sep.add(e)
            w = e[0] if e[1] == n else e[1]
            if M.ndim[w] < 2:  # reached a CAD edge: split the faces on the other side too
                for s2 in {M.surf[q] for q in M.node_quads[w]} - {s}:
                    todo += [(w, f, s2) for f in _edges_at(M, w, s2) if not M.hard(f)]
                break
            if _irregular(M, w, s):
                break
            # continue straight: the edge at w that shares no quad with e
            qs = set(M.edge_quads[e])
            nxt = [f for f in _edges_at(M, w, s) if f != e and not qs & set(M.edge_quads[f])]
            if len(nxt) != 1:
                break
            n, e = w, nxt[0]
    return sep


def _grid(M, block):
    """(i, j) grid of the quads of a block, or None if the block is not a structured rectangle.
    Returns the node grid N[j][i] of shape (m + 1, n + 1)."""
    q0 = block[0]
    orient = {q0: list(M.quads[q0])}
    pos = {q0: (0, 0)}
    inblock = set(block)
    stack = [q0]
    step = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # across edge k of the oriented quad
    while stack:
        q = stack.pop()
        Q = orient[q]
        for k in range(4):
            a, b = Q[k], Q[(k + 1) % 4]
            for r in M.edge_quads[_key(a, b)]:
                if r == q or r not in inblock:
                    continue
                R = list(M.quads[r])
                if r in orient:
                    continue
                kk = next((t for t in range(4) if R[t] == b and R[(t + 1) % 4] == a), None)
                if kk is None:  # inconsistent orientation: reverse the neighbour
                    R = R[::-1]
                    kk = next(t for t in range(4) if R[t] == b and R[(t + 1) % 4] == a)
                rot = (kk - (k + 2) % 4) % 4
                orient[r] = R[rot:] + R[:rot]
                pos[r] = (pos[q][0] + step[k][0], pos[q][1] + step[k][1])
                stack.append(r)
    if len(orient) != len(block):
        return None
    I = np.array([p[0] for p in pos.values()])
    J = np.array([p[1] for p in pos.values()])
    i0, j0, n, m = I.min(), J.min(), I.max() - I.min() + 1, J.max() - J.min() + 1
    if n * m != len(block) or len(set(pos.values())) != len(block):
        return None
    N = -np.ones((m + 1, n + 1), dtype=np.int64)
    for q, (i, j) in pos.items():
        a, b, c, d = orient[q]
        for (ii, jj), node in (((0, 0), a), ((1, 0), b), ((1, 1), c), ((0, 1), d)):
            jj, ii = j - j0 + jj, i - i0 + ii
            if N[jj, ii] not in (-1, node):
                return None
            N[jj, ii] = node
    return N


def blocks(M, sep):
    """Structured blocks: (surface tag, node grid N[j][i])."""
    parent = list(range(len(M.quads)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e, qs in M.edge_quads.items():
        if len(qs) == 2 and e not in sep and not M.hard(e) and M.surf[qs[0]] == M.surf[qs[1]]:
            parent[find(qs[0])] = find(qs[1])
    groups = defaultdict(list)
    for q in range(len(M.quads)):
        groups[find(q)].append(q)
    out, fallback = [], 0
    for g in groups.values():
        N = _grid(M, g)
        if N is None:
            fallback += 1
            out += [(M.surf[q], _grid(M, [q])) for q in g]
        else:
            out.append((M.surf[g[0]], N))
    return out, fallback


# ---------------------------------------------------------------- 3. fit
def _bspline_basis(p, n_span, t):
    """Clamped uniform B-spline basis (len(t), n_span + p) and its knot vector."""
    U = np.concatenate([np.zeros(p), np.linspace(0, 1, n_span + 1), np.ones(p)])
    return np.array([_basis(p, U, min(x, 1.0))[0] for x in t]), U


def _project(dim, tag, X):
    return np.asarray(gmsh.model.getClosestPoint(dim, tag, np.asarray(X).ravel())[0]).reshape(-1, 3)


class SideTable:
    """One fitted curve per block side (a chain of mesh nodes), shared by its two blocks."""

    def __init__(self, M, p, k=4):
        self.M, self.p, self.k, self.curves = M, p, k, {}

    def get(self, chain, surf):
        key = tuple(chain) if (chain[0], chain[1]) < (chain[-1], chain[-2]) else tuple(chain[::-1])
        if key not in self.curves:
            self.curves[key] = self._fit(list(key), surf)
        C = self.curves[key]
        return C if tuple(chain) == key else C[::-1]

    def _fit(self, chain, surf):
        M, p, n = self.M, self.p, len(chain) - 1
        s = np.linspace(0, 1, self.k + 2)[1:-1]
        T, X = [0.0], [M.x[chain[0]]]
        for i in range(n):
            a, b = M.x[chain[i]], M.x[chain[i + 1]]
            e = _key(chain[i], chain[i + 1])
            dim, tag = (1, M.curve_of[e]) if M.hard(e) else (2, surf)
            P = _project(dim, tag, (1 - s)[:, None] * a + s[:, None] * b)
            T += list((i + s) / n) + [(i + 1) / n]
            X += list(P) + [M.x[chain[i + 1]]]
        B, _ = _bspline_basis(p, n, np.array(T))
        X = np.array(X)
        C = np.zeros((n + p, 3))
        C[0], C[-1] = X[0], X[-1]
        rhs = X - np.outer(B[:, 0], C[0]) - np.outer(B[:, -1], C[-1])
        C[1:-1] = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]
        return C


def fit_block(M, sides, surf, N, k=3):
    """B-spline patch (degree p, one span per mesh quad) of a block with node grid N."""
    p = sides.p
    m, n = N.shape[0] - 1, N.shape[1] - 1
    C = np.full((m + p, n + p, 3), np.nan)
    C[0] = sides.get(list(N[0]), surf)
    C[-1] = sides.get(list(N[-1]), surf)
    C[:, 0] = sides.get(list(N[:, 0]), surf)
    C[:, -1] = sides.get(list(N[:, -1]), surf)
    s = (np.arange(k) + 0.5) / k
    U, V, X = [], [], []
    for j in range(m):
        for i in range(n):
            a, b, c, d = (M.x[N[j, i]], M.x[N[j, i + 1]], M.x[N[j + 1, i + 1]], M.x[N[j + 1, i]])
            S, T = np.meshgrid(s, s)
            S, T = S.ravel(), T.ravel()
            Y = ((1 - S) * (1 - T))[:, None] * a + (S * (1 - T))[:, None] * b + (S * T)[:, None] * c \
                + ((1 - S) * T)[:, None] * d
            X += list(_project(2, surf, Y))
            U += list((i + S) / n)
            V += list((j + T) / m)
            if i > 0 and j > 0:
                U.append(i / n), V.append(j / m), X.append(M.x[N[j, i]])
    Bu, KU = _bspline_basis(p, n, np.array(U))
    Bv, KV = _bspline_basis(p, m, np.array(V))
    X = np.array(X)
    free = np.zeros((m + p, n + p), bool)
    free[1:-1, 1:-1] = True
    A = np.einsum("sj,si->sji", Bv, Bu).reshape(len(X), -1)
    Cf = C.reshape(-1, 3)
    fixed = ~free.ravel()
    rhs = X - A[:, fixed] @ Cf[fixed]
    if free.any():
        Cf[free.ravel()] = np.linalg.lstsq(A[:, free.ravel()], rhs, rcond=None)[0]
    Pw = np.concatenate([Cf.reshape(m + p, n + p, 3), np.ones((m + p, n + p, 1))], axis=-1)
    return Patch(p, p, KU, KV, Pw)


def _outward(M, P, eps):
    x, xu, xv = evaluate(P, 0.5, 0.5)
    nrm = np.cross(xu, xv)
    y = x + eps * nrm / np.linalg.norm(nrm)
    inside = any(gmsh.model.isInside(3, v, list(y)) for v in M.volumes)
    return not inside


def jacobian_ratio(P, m=5):
    """min / max of |x_u x x_v| over an m x m grid: near 0 flags a degenerate patch."""
    t = np.linspace(0, 1, m)
    J = [np.linalg.norm(np.cross(*evaluate(P, u, v)[1:])) for u in t for v in t]
    return min(J) / max(J)


def _deviation(P, surf, m=4):
    t = (np.arange(m) + 0.5) / m
    X = np.array([evaluate(P, u, v)[0] for u in t for v in t])
    return np.linalg.norm(_project(2, surf, X) - X, axis=1).max()


def convert(step, out, h=0.4, degree=3, refine=0, curvature=8, name=None):
    M = QuadMesh(step, h, curvature)
    try:
        sep = trace_layout(M)
        B, fallback = blocks(M, sep)
        sides = SideTable(M, degree)
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        eps = 1e-4 * np.linalg.norm(bb[3:] - bb[:3])
        patches, dev = [], 0.0
        for surf, N in B:
            P = fit_block(M, sides, surf, N)
            if not _outward(M, P, eps):
                P = Patch(P.q, P.p, P.V, P.U, np.ascontiguousarray(P.Pw.transpose(1, 0, 2)))
            dev = max(dev, _deviation(P, surf))
            patches.append(P)
        v_cad = sum(gmsh.model.occ.getMass(3, v) for v in M.volumes)
        nquads, qmin = len(M.quads), M.min_quality
    finally:
        gmsh.finalize()
    for P in patches:
        for _ in range(refine):
            refine_uniform(P)
    gap, nshared, lonely = conformity_gap(patches)
    v_spl = spline_volume(patches, 4)  # exact for these polynomial patches up to p = 3
    write_dat(out, patches, name or os.path.basename(out))
    return dict(quads=nquads, patches=len(patches), fallback_blocks=fallback, min_quad_quality=qmin,
                min_jacobian_ratio=min(jacobian_ratio(P) for P in patches), max_dev=dev,
                conformity_gap=gap, shared_edges=nshared, open_edges=lonely, vol_cad=v_cad,
                vol_spline=v_spl, vol_rel_err=abs(v_spl - v_cad) / abs(v_cad))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("out")
    ap.add_argument("--h", type=float, default=0.4, help="target mesh size (one knot span)")
    ap.add_argument("--curvature", type=int, default=8, help="mesh spans per 2 pi of curvature")
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--refine", type=int, default=0)
    a = ap.parse_args(argv)
    r = convert(a.step, a.out, a.h, a.degree, a.refine, a.curvature)
    for k, v in r.items():
        print("%-15s %s" % (k, v))
    return r


if __name__ == "__main__":
    main()
