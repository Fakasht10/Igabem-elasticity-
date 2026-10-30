"""Automatic STEP -> conforming multipatch NURBS surfaces (Bembel .dat), any B-rep solid.

Unlike python/step2dat.py (exact Coons patches of a hand-made quad layout), the layout here
is automatic; patches are exact where the CAD allows it and fitted elsewhere:

  0. cut (--split): closed (periodic) faces are cut along their own parameter lines into three
     four-sided faces (OpenCASCADE), so their meshes become structured and parameter-aligned;
  1. quad mesh: Gmsh (OpenCASCADE kernel) meshes every CAD face, trimmed or not, with
     quasi-structured quads (few irregular vertices); every CAD edge is a chain of mesh edges;
  2. quad layout: separatrices are traced from every irregular vertex (interior valence != 4,
     curve vertex with != 2 quads, corner with != 1 quad in a face) straight through the
     regular vertices until they hit a CAD edge or another irregular vertex. A separatrix that
     ends on a CAD edge also splits the faces on the other side, so blocks meet side to side.
     CAD edges and separatrices cut the mesh into structured n x m blocks (a block that is
     not a rectangle falls back to one patch per quad);
  3a. exact: a block that is a parameter rectangle of an analytic face (plane, cylinder,
     cone, sphere, torus) becomes the exact rational patch (circle arcs as rational
     quadratics, degree-elevated and knot-inserted to the common degree p and knot vectors);
     its sides are registered first;
  3b. fit: every other block becomes a B-spline patch of degree p with the block's n x m
     mesh quads as knot spans (C^{p-1} inside). Each side not taken by an exact block is
     fitted once, by least squares to points projected onto its CAD curve (or face, for a
     separatrix); every side is shared by both neighbouring blocks, so the patches are
     conforming and watertight; the interior control points (weight 1) are fitted to points
     projected onto the CAD face with the boundary held fixed;
  4. orientation: x_u x x_v points out of the solid (Gmsh inside test);
  5. checks: distance to the CAD faces, interface gap, volume vs. OpenCASCADE.

usage: python3 python/step_auto.py in.step out.dat [--h 0.4] [--degree 3] [--refine k]
                                     [--fit-all] [--split] [--focus x,y,z,r,hf]...
"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict

import gmsh
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import Patch, refine_uniform, write_dat  # noqa: E402
from spline_checks import _basis, conformity_gap, evaluate, spline_volume  # noqa: E402


# ---------------------------------------------------------------- 1. quad mesh
def _focus_field(focus, h, slope=0.3):
    """Background size: (x, y, z, r, hf) focus spheres, size hf inside radius r and growing by `slope`
    per unit distance outside it (the same "core" grading as nurb-fea-tcd's Refinement), capped at h.
    A MathEval field, which Gmsh honours on every face (its Ball field is not always honoured)."""
    terms = ["%.17g + %.17g * Max(0, Sqrt((x-(%.17g))^2 + (y-(%.17g))^2 + (z-(%.17g))^2) - %.17g)"
             % (hf, slope, x, y, z, r) for x, y, z, r, hf in focus]
    expr = terms[0]
    for t in terms[1:]:
        expr = "Min(%s, %s)" % (expr, t)
    f = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(f, "F", "Min(%.17g, %s)" % (h, expr))
    gmsh.model.mesh.field.setAsBackgroundMesh(f)
    gmsh.option.setNumber("Mesh.MeshSizeMin", min(gmsh.option.getNumber("Mesh.MeshSizeMin"),
                                                  min(hf for *_, hf in focus) / 2))


class QuadMesh:
    """All-quad surface mesh of a STEP solid with its CAD classification (gmsh session open)."""

    def __init__(self, step, h, curvature=8, split_closed=False, focus=()):
        tmp = None
        if split_closed:
            tmp = tempfile.NamedTemporaryFile(suffix=".brep", delete=False).name
            _split_closed(step, tmp)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(tmp or step)
        if tmp:
            os.unlink(tmp)
        gmsh.model.occ.synchronize()
        for k, v in {"Mesh.Algorithm": 11, "Mesh.RecombineAll": 1, "Mesh.MeshSizeMax": h,
                     "Mesh.MeshSizeMin": h / 8, "Mesh.MeshSizeFromCurvature": curvature,
                     "Mesh.MeshSizeFromPoints": 0, "Mesh.MeshSizeExtendFromBoundary": 0,
                     "Mesh.Smoothing": 20}.items():
            gmsh.option.setNumber(k, v)
        # structured grids on four-sided faces (compatible counts chosen by Gmsh); the rest
        # gets quasi-structured quads
        if focus:
            _focus_field(focus, h)
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


def _split_closed(step, out, pieces=3):
    """Cut every closed (periodic) face along its own parameter lines into `pieces` faces
    (OpenCASCADE ShapeUpgrade_ShapeDivideClosed), so a hole wall becomes four-sided faces with
    a structured, parameter-aligned mesh, i.e. exact patches (each piece opens 2 pi / 3)."""
    from OCP.BRepTools import BRepTools
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader()
    if r.ReadFile(step) != 1 or not r.TransferRoots():
        raise RuntimeError("cannot read " + step)
    d = ShapeUpgrade_ShapeDivideClosed(r.OneShape())
    d.SetNbSplitPoints(pieces - 1)
    d.Perform()
    BRepTools.Write_s(d.Result(), out)


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
    """One curve per block side (a chain of mesh nodes), shared by its two blocks, as
    homogeneous control points (x w, w). Sides of exact blocks are registered first; the
    rest are fitted (w = 1) between corner points that exact blocks may have fixed."""

    def __init__(self, M, p, k=4):
        self.M, self.p, self.k, self.curves, self.corner = M, p, k, {}, {}

    @staticmethod
    def _orient(chain):
        key = tuple(chain) if (chain[0], chain[1]) < (chain[-1], chain[-2]) else tuple(chain[::-1])
        return key, tuple(chain) == key

    def lookup(self, chain):
        key, fwd = self._orient(chain)
        C = self.curves.get(key)
        return None if C is None else (C if fwd else C[::-1])

    def register(self, chain, C):
        """Store an exact side (end weights 1); its end points snap to corners already fixed."""
        for n, e in ((chain[0], 0), (chain[-1], -1)):
            if n in self.corner:
                C[e] = np.append(self.corner[n], 1.0)
            else:
                self.corner[n] = C[e, :3].copy()
        key, fwd = self._orient(chain)
        self.curves[key] = C if fwd else C[::-1]

    def get(self, chain, surf):
        C = self.lookup(chain)
        if C is None:
            key, _ = self._orient(chain)
            self.curves[key] = self._fit(list(key), surf)
            C = self.lookup(chain)
        return C

    def _x(self, n):
        return self.corner.get(n, self.M.x[n])

    def _fit(self, chain, surf):
        M, p, n = self.M, self.p, len(chain) - 1
        s = np.linspace(0, 1, self.k + 2)[1:-1]
        T, X = [0.0], [self._x(chain[0])]
        for i in range(n):
            a, b = M.x[chain[i]], M.x[chain[i + 1]]
            e = _key(chain[i], chain[i + 1])
            dim, tag = (1, M.curve_of[e]) if M.hard(e) else (2, surf)
            P = _project(dim, tag, (1 - s)[:, None] * a + s[:, None] * b)
            T += list((i + s) / n) + [(i + 1) / n]
            X += list(P) + [self._x(chain[i + 1]) if i == n - 1 else M.x[chain[i + 1]]]
        B, _ = _bspline_basis(p, n, np.array(T))
        X = np.array(X)
        C = np.zeros((n + p, 3))
        C[0], C[-1] = X[0], X[-1]
        rhs = X - np.outer(B[:, 0], C[0]) - np.outer(B[:, -1], C[-1])
        C[1:-1] = np.linalg.lstsq(B[:, 1:-1], rhs, rcond=None)[0]
        return np.hstack([C, np.ones((n + p, 1))])


# ---------------------------------------------------------------- 3a. exact blocks
# CAD parameter kinds of the analytic surfaces (OpenCASCADE parametrisations, which Gmsh
# keeps): an angle about an axis ("ang", a circle: exact rational quadratic) or a length
# along a straight line ("lin", exact linear).
_KINDS = {"Plane": ("lin", "lin"), "Cylinder": ("ang", "lin"), "Cone": ("ang", "lin"),
          "Sphere": ("ang", "ang"), "Torus": ("ang", "ang")}


def _bezier_ctrl(X, kind, w):
    """Control points (homogeneous-free) and weights along one direction from the points
    X[0], X[mid], X[1] of a circle arc of opening w (ang) or of a segment (lin)."""
    if kind == "lin":
        return [X[0], X[2]], [1.0, 1.0]
    c = np.cos(w / 2)
    M = 0.5 * (X[0] + X[2])
    C = (M - c * X[1]) / (1 - c)  # circle centre: the chord midpoint lies at cos(w/2) of the way
    return [X[0], C + (X[1] - C) / c, X[2]], [1.0, c, 1.0]


def _bernstein(q, t):
    from math import comb
    t = np.asarray(t)[:, None]
    k = np.arange(q + 1)[None, :]
    return np.array([comb(q, i) for i in range(q + 1)])[None, :] * t ** k * (1 - t) ** (q - k)


def exact_block(M, surf, N, p, max_angle=2 * np.pi / 3):
    """Exact rational patch (degree p, one knot span per mesh quad, same knot vectors as the
    fitted patches) of a block that is a parameter rectangle of an analytic CAD face, or None.
    The block's sides must be iso-parameter lines; each circular direction must open by at
    most max_angle (one rational quadratic Bezier piece), which is then degree-elevated and
    knot-inserted exactly (interpolation at the Greville points of the target space)."""
    t = gmsh.model.getType(2, surf)
    if t not in _KINDS:  # e.g. a face cut along its parameter lines (trimmed surface): try
        # each construction; the guard below keeps only one that reproduces the CAD face
        for kinds in (("ang", "lin"), ("lin", "ang"), ("ang", "ang"), ("lin", "lin")):
            P = _exact_block(M, surf, N, p, max_angle, kinds)
            if P is not None:
                return P
        return None
    return _exact_block(M, surf, N, p, max_angle, _KINDS[t])


def _exact_block(M, surf, N, p, max_angle, kinds):
    m, n = N.shape[0] - 1, N.shape[1] - 1
    X = np.array([M.x[k] for k in N.ravel()])
    UV = np.asarray(gmsh.model.getParametrization(2, surf, X.ravel())).reshape(m + 1, n + 1, 2)
    for d in (0, 1):  # unwrap periodic angles about the block centre
        if kinds[d] == "ang":
            c = UV[m // 2, n // 2, d]
            UV[..., d] = c + (UV[..., d] - c + np.pi) % (2 * np.pi) - np.pi
    # Only the sides must be iso-parameter lines: exactly where a side is a CAD edge; a side
    # inside the face (separatrix) only needs its end points on the line and its nodes near
    # it, since the exact side replaces it for the neighbour too. Interior nodes are unused.
    span = np.ptp(UV.reshape(-1, 2), axis=0)

    def iso(chain, uv, d):
        hard = all(M.hard(_key(a, b)) for a, b in zip(chain[:-1], chain[1:]))
        dev = np.abs(uv[:, d] - uv[0, d])
        return dev[-1] < 1e-8 * (1 + span[d]) and dev.max() < (1e-8 if hard else 0.1 * span[d] / len(uv)) * (1 + span[d])

    rows_const = [iso(N[0], UV[0], d) and iso(N[-1], UV[-1], d) for d in (0, 1)]  # rows j = 0, m
    cols_const = [iso(N[:, 0], UV[:, 0], d) and iso(N[:, -1], UV[:, -1], d) for d in (0, 1)]  # columns
    if cols_const[0] and rows_const[1]:  # CAD u runs along i
        di, dj = 0, 1
    elif cols_const[1] and rows_const[0]:
        di, dj = 1, 0
    else:
        return None
    a = (UV[0, 0, di], UV[0, n, di])  # CAD parameter along i (patch u) and along j (patch v)
    b = (UV[0, 0, dj], UV[m, 0, dj])
    for (lo, hi), kd in ((a, kinds[di]), (b, kinds[dj])):
        if abs(hi - lo) < 1e-12 or (kd == "ang" and abs(hi - lo) > max_angle):
            return None

    def S(ai, bj):  # CAD surface at parameter ai along i, bj along j
        ai, bj = np.broadcast_arrays(np.asarray(ai, float), np.asarray(bj, float))
        uv = np.stack([ai, bj] if di == 0 else [bj, ai], -1)
        return np.asarray(gmsh.model.getValue(2, surf, uv.ravel())).reshape(-1, 3)

    ta, tb = np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])
    A = a[0] + ta * (a[1] - a[0])
    B = b[0] + tb * (b[1] - b[0])
    G = S(A[None, :].repeat(3, 0), B[:, None].repeat(3, 1)).reshape(3, 3, 3)  # G[j, i]
    # j-direction control points along each of the three i-stations, then i-direction
    cols = [_bezier_ctrl(G[:, i], kinds[dj], b[1] - b[0]) for i in range(3)]
    qj = len(cols[0][0]) - 1
    Pij, Wij = [], []
    for k in range(qj + 1):
        P, w = _bezier_ctrl(np.array([cols[i][0][k] for i in range(3)]), kinds[di], a[1] - a[0])
        Pij.append(P)
        Wij.append(np.array(w) * cols[0][1][k])
    Pij, Wij = np.array(Pij), np.array(Wij)  # (qj + 1, qi + 1, 3), weights
    qi = Pij.shape[1] - 1
    Hw = np.concatenate([Pij * Wij[..., None], Wij[..., None]], -1)
    # exact change of basis: Bezier (qi, qj) -> clamped uniform B-splines (p, n x m spans)
    gi = _greville(p, n)
    gj = _greville(p, m)
    Ri = _bernstein(qi, gi)
    Rj = _bernstein(qj, gj)
    Fi, _ = _bspline_basis(p, n, gi)
    Fj, _ = _bspline_basis(p, m, gj)
    vals = np.einsum("sj,ti,jik->stk", Rj, Ri, Hw)  # = Fj Pw Fi^T at the Greville points
    Pw = np.einsum("ab,bck,dc->adk", np.linalg.inv(Fj), vals, np.linalg.inv(Fi))
    KU = np.concatenate([np.zeros(p), np.linspace(0, 1, n + 1), np.ones(p)])
    KV = np.concatenate([np.zeros(p), np.linspace(0, 1, m + 1), np.ones(p)])
    P = Patch(p, p, KU, KV, Pw)
    # guard: corners on the mesh corners, samples on the CAD face
    scale = 1.0 + np.abs(X).max()
    for (jj, ii), (u, v) in (((0, 0), (0, 0)), ((0, n), (1, 0)), ((m, 0), (0, 1)), ((m, n), (1, 1))):
        if np.linalg.norm(evaluate(P, u, v)[0] - M.x[N[jj, ii]]) > 1e-8 * scale:
            return None
    t = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    Y = np.array([evaluate(P, u, v)[0] for u in t for v in t])
    if np.linalg.norm(_project(2, surf, Y) - Y, axis=1).max() > 1e-10 * scale:
        return None
    for jj, ii in ((0, 0), (0, -1), (-1, 0), (-1, -1)):  # corners: weight 1, as fitted sides
        Pw[jj, ii] /= Pw[jj, ii, 3]
    return P


def _greville(p, n_span):
    U = np.concatenate([np.zeros(p), np.linspace(0, 1, n_span + 1), np.ones(p)])
    return np.array([U[i + 1:i + p + 1].mean() for i in range(n_span + p)])


def fit_block(M, sides, surf, N, k=3):
    """B-spline patch (degree p, one span per mesh quad) of a block with node grid N. The
    boundary rows come from the side table (possibly rational, from an exact neighbour); the
    interior control points (weight 1) are fitted in homogeneous form, which is linear."""
    p = sides.p
    m, n = N.shape[0] - 1, N.shape[1] - 1
    C = np.full((m + p, n + p, 4), np.nan)
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
    C[1:-1, 1:-1, 3] = 1.0
    A = np.einsum("sj,si->sji", Bv, Bu).reshape(len(X), -1)
    Cf = C.reshape(-1, 4)
    fixed = ~free.ravel()
    w = A @ Cf[:, 3]  # known weight function at the samples
    rhs = X * w[:, None] - A[:, fixed] @ Cf[fixed, :3]
    if free.any():
        Cf[free.ravel(), :3] = np.linalg.lstsq(A[:, free.ravel()], rhs, rcond=None)[0]
    return Patch(p, p, KU, KV, Cf.reshape(m + p, n + p, 4))


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


def span_jacobian(P):
    """min over the knot spans of min / max |x_u x x_v| on the span's 3 x 3 corner/mid grid: near 0 flags an
    element whose parametrisation collapses at a point (a fitted side with a near-zero speed), which the
    collocation resolves badly (a local stress spike)."""
    U, V = np.unique(P.U), np.unique(P.V)
    worst = 1.0
    for i in range(len(U) - 1):
        for j in range(len(V) - 1):
            J = [np.linalg.norm(np.cross(*evaluate(P, U[i] + a * (U[i + 1] - U[i]), V[j] + b * (V[j + 1] - V[j]))[1:]))
                 for a in (0.0, 0.5, 1.0 - 1e-12) for b in (0.0, 0.5, 1.0 - 1e-12)]
            worst = min(worst, min(J) / max(J))
    return worst


def span_aspect(P):
    """max over the knot spans (elements) of long side / short side, measured along the mid iso-lines:
    a sliver (a thin strip of quads left by the layout) is collocated badly and shows as a stress spike."""
    U, V = np.unique(P.U), np.unique(P.V)
    worst = 1.0
    for i in range(len(U) - 1):
        for j in range(len(V) - 1):
            um, vm = 0.5 * (U[i] + U[i + 1]), 0.5 * (V[j] + V[j + 1])
            _, xu, xv = evaluate(P, um, vm)
            a, b = np.linalg.norm(xu) * (U[i + 1] - U[i]), np.linalg.norm(xv) * (V[j + 1] - V[j])
            worst = max(worst, max(a, b) / max(min(a, b), 1e-300))
    return worst


def _deviation(P, surf, m=4):
    t = (np.arange(m) + 0.5) / m
    X = np.array([evaluate(P, u, v)[0] for u in t for v in t])
    return np.linalg.norm(_project(2, surf, X) - X, axis=1).max()


def convert(step, out, h=0.4, degree=3, refine=0, curvature=8, name=None, exact_faces=True, split_closed=False,
            focus=()):
    M = QuadMesh(step, h, curvature, split_closed and exact_faces, focus)
    try:
        sep = trace_layout(M)
        B, fallback = blocks(M, sep)
        sides = SideTable(M, degree)
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        eps = 1e-4 * np.linalg.norm(bb[3:] - bb[:3])
        patches, dev, nexact, scale = [], 0.0, 0, 1.0 + np.abs(bb).max()
        exact = {}
        if exact_faces:
            for b, (surf, N) in enumerate(B):
                P = exact_block(M, surf, N, degree)
                if P is None:
                    continue
                rows = [(list(N[0]), P.Pw[0]), (list(N[-1]), P.Pw[-1]),
                        (list(N[:, 0]), P.Pw[:, 0]), (list(N[:, -1]), P.Pw[:, -1])]
                old = [sides.lookup(c) for c, _ in rows]
                if any(o is not None and np.abs(o - r).max() > 1e-10 * scale for o, (_, r) in zip(old, rows)):
                    continue  # a side already taken by another exact block, differently parametrised
                for (c, r), o in zip(rows, old):
                    if o is None:
                        sides.register(c, r.copy())
                P.Pw[0], P.Pw[-1] = sides.lookup(rows[0][0]), sides.lookup(rows[1][0])
                P.Pw[:, 0], P.Pw[:, -1] = sides.lookup(rows[2][0]), sides.lookup(rows[3][0])
                exact[b] = P
        for b, (surf, N) in enumerate(B):
            P = exact.get(b)
            if P is None:
                P = fit_block(M, sides, surf, N)
            if not _outward(M, P, eps):
                P = Patch(P.q, P.p, P.V, P.U, np.ascontiguousarray(P.Pw.transpose(1, 0, 2)))
            dev = max(dev, _deviation(P, surf))
            patches.append(P)
        nexact = len(exact)
        v_cad = sum(gmsh.model.occ.getMass(3, v) for v in M.volumes)
        nquads, qmin = len(M.quads), M.min_quality
    finally:
        gmsh.finalize()
    for P in patches:
        for _ in range(refine):
            refine_uniform(P)
    gap, nshared, lonely = conformity_gap(patches)
    v_spl = spline_volume(patches, 8)  # rational patches: Gauss rule no longer exact
    write_dat(out, patches, name or os.path.basename(out))
    return dict(quads=nquads, patches=len(patches), exact_patches=nexact, fallback_blocks=fallback, min_quad_quality=qmin,
                min_jacobian_ratio=min(jacobian_ratio(P) for P in patches),
                min_span_jacobian=min(span_jacobian(P) for P in patches),
                max_span_aspect=max(span_aspect(P) for P in patches), max_dev=dev,
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
    ap.add_argument("--fit-all", action="store_true", help="fit every block (no exact CAD patches)")
    ap.add_argument("--split", action="store_true",
                    help="cut closed faces along their parameter lines first (exact hole walls on prismatic parts;"
                         " on revolved parts such as a bolt it only adds patches)")
    ap.add_argument("--focus", action="append", default=[], metavar="x,y,z,r,hf",
                    help="refine to size hf within radius r of (x, y, z), graded out to h (repeatable)")
    a = ap.parse_args(argv)
    r = convert(a.step, a.out, a.h, a.degree, a.refine, a.curvature, exact_faces=not a.fit_all,
                split_closed=a.split, focus=[tuple(map(float, f.split(","))) for f in a.focus])
    for k, v in r.items():
        print("%-15s %s" % (k, v))
    return r


if __name__ == "__main__":
    main()
