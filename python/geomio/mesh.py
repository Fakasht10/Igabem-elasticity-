"""Face-tagged triangle meshes: weld, orient, check and segment a closed surface.

The repair steps follow materializr (github.com/materializr-cad/materializr, GPLv3, src/io/StlIO.cpp and
src/io/StlExport.cpp at commit c5f9aa5): weld coincident vertices on an exact integer cell key, undo any weld that
leaves an edge shared by more than two triangles (a fold, not a crack), flood-fill the winding over edge adjacency so
every interior edge is traversed once each way, call the mesh watertight only when every edge has exactly two
triangles, and flip an inside-out mesh by the sign of its volume.  Re-implemented here in numpy; two changes:
  - the volume sign is decided per connected component from its nesting depth (winding number of one of its points
    with respect to the other components), so a hollow part keeps its cavity shell pointing into the cavity;
  - segment_faces() regrows CAD-like faces on an untagged mesh across edges with a dihedral angle below a threshold
    (materializr merges near-coplanar facets with ShapeUpgrade_UnifySameDomain; here curved smooth regions merge too).

Mesh convention: V (nv, 3) float64, T (nt, 3) int64 counter-clockwise seen from outside (outward normal
(V1 - V0) x (V2 - V0)), fid (nt,) int64 CAD face id per triangle (0 = untagged)."""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components


@dataclass
class TriMesh:
    V: np.ndarray
    T: np.ndarray
    fid: np.ndarray = None
    names: dict = field(default_factory=dict)   # fid -> CAD face name, when the source had one

    def __post_init__(self):
        self.V = np.ascontiguousarray(self.V, float).reshape(-1, 3)
        self.T = np.ascontiguousarray(self.T, np.int64).reshape(-1, 3)
        self.fid = np.zeros(len(self.T), np.int64) if self.fid is None else np.asarray(self.fid, np.int64).ravel()
        if len(self.fid) != len(self.T):
            raise ValueError("fid has %d entries for %d triangles" % (len(self.fid), len(self.T)))

    def normals(self, unit=True):
        A, B, C = (self.V[self.T[:, k]] for k in range(3))
        n = np.cross(B - A, C - A)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300) if unit else n

    def areas(self):
        return 0.5 * np.linalg.norm(self.normals(False), axis=1)

    def volume(self):
        A, B, C = (self.V[self.T[:, k]] for k in range(3))
        return float(np.einsum("ij,ij->", A, np.cross(B, C)) / 6.0)

    def diag(self):
        return float(np.linalg.norm(self.V.max(0) - self.V.min(0))) if len(self.V) else 0.0

    def faces(self):
        return np.unique(self.fid)


# ---------------------------------------------------------------- edges
def _edges(T):
    """directed edges (3 nt, 2), their triangle, and undirected edge id per directed edge."""
    E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2)
    tri = np.repeat(np.arange(len(T)), 3)
    key, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
    return E, tri, inv.ravel(), cnt


def edge_report(m):
    """counts of boundary (1 triangle), non-manifold (> 2) and misoriented (2 triangles, same direction) edges."""
    E, _, inv, cnt = _edges(m.T)
    forward = E[:, 0] < E[:, 1]
    fw = np.bincount(inv, forward, len(cnt))
    two = cnt == 2
    return dict(edges=int(len(cnt)), boundary=int((cnt == 1).sum()), nonmanifold=int((cnt > 2).sum()),
                misoriented=int((two & (fw != 1)).sum()))


def is_watertight(m):
    r = edge_report(m)
    return r["boundary"] == 0 and r["nonmanifold"] == 0 and r["misoriented"] == 0


# ---------------------------------------------------------------- weld
def _cell_ids(V, tol):
    """vertex -> representative, merging vertices with the same integer cell (exact key, no hash collisions)."""
    K = np.round(V / tol).astype(np.int64) if tol > 0 else V
    _, first, inv = np.unique(K, axis=0, return_index=True, return_inverse=True)
    return first[inv.ravel()]


def weld(m, tol=None):
    """merge coincident vertices (exact first, then within tol: default 1e-7 of the largest coordinate, i.e. float32
    round-off); drop triangles that collapse; undo tolerance welds that fold the mesh (edge with > 2 triangles that
    the exact weld did not have).  Returns a new TriMesh with unused vertices removed."""
    V, T = m.V, m.T
    if tol is None:
        tol = 1e-7 * max(float(np.abs(V).max()) if len(V) else 1.0, 1e-300)
    exact = _cell_ids(V, 0)
    rep = exact[_cell_ids(V[exact], tol)] if tol > 0 else exact

    def tris(r):
        S = r[T]
        keep = (S[:, 0] != S[:, 1]) & (S[:, 1] != S[:, 2]) & (S[:, 2] != S[:, 0])
        return S, keep

    S, keep = tris(rep)
    if tol > 0:
        _, _, inv, cnt = _edges(S[keep])
        bad = cnt[inv] > 2
        if bad.any():
            Se, ke = tris(exact)
            _, _, inv_e, cnt_e = _edges(Se[ke])
            nm_exact = int((cnt_e > 2).sum())
            if int((cnt > 2).sum()) > nm_exact:   # the tolerance weld made new folds: revert those vertices
                Ek = np.stack([S[keep][:, [0, 1]], S[keep][:, [1, 2]], S[keep][:, [2, 0]]], 1).reshape(-1, 2)
                poisoned = np.zeros(len(V), bool)
                poisoned[Ek[bad].ravel()] = True
                rep = np.where(poisoned[rep], exact, rep)
                S, keep = tris(rep)
    S, fid = S[keep], m.fid[keep]
    used, Tn = np.unique(S, return_inverse=True)
    return TriMesh(V[used], Tn.reshape(-1, 3), fid, dict(m.names))


def drop_degenerate(m, rel=1e-14):
    """remove triangles of area below rel * diag^2 (zero-area facets, as materializr's StlIO skips them)."""
    keep = m.areas() > rel * m.diag() ** 2
    return TriMesh(m.V, m.T[keep], m.fid[keep], dict(m.names))


# ---------------------------------------------------------------- orientation
def _components(m):
    E, tri, inv, cnt = _edges(m.T)
    order = np.argsort(inv, kind="stable")
    first = np.r_[0, np.cumsum(cnt)[:-1]]
    two = np.flatnonzero(cnt == 2)
    a, b = order[first[two]], order[first[two] + 1]          # the two directed edges of each manifold edge
    ta, tb = tri[a], tri[b]
    same = E[a, 0] == E[b, 0]                                  # both traverse it the same way: one must flip
    nt = len(m.T)
    G = coo_matrix((np.ones(len(ta)), (ta, tb)), shape=(nt, nt)).tocsr()
    ncomp, lab = connected_components(G, directed=False)
    return ncomp, lab, G, ta, tb, same


def orient(m):
    """consistent winding by a breadth-first flood fill over manifold edges, then outward per component: a component
    nested in an even number of others has positive volume, in an odd number negative (a cavity wall).
    Returns (TriMesh, info)."""
    T = m.T.copy()
    ncomp, lab, G, ta, tb, same = _components(m)
    nt = len(T)
    pk = np.minimum(ta, tb) * nt + np.maximum(ta, tb)          # triangle pair key -> needs a relative flip
    srt = np.argsort(pk)
    pk, same_s = pk[srt], same[srt]
    f = [False] * nt
    for c in range(ncomp):
        seed = int(np.flatnonzero(lab == c)[0])
        order, pred = breadth_first_order(G, seed, directed=False, return_predecessors=True)
        t, p = order[1:], pred[order[1:]]
        rel = same_s[np.searchsorted(pk, np.minimum(t, p) * nt + np.maximum(t, p))]
        for ti, pi, ri in zip(t.tolist(), p.tolist(), rel.tolist()):
            f[ti] = f[pi] ^ ri
    flip = np.array(f, bool)
    T[flip] = T[flip][:, [0, 2, 1]]
    out = TriMesh(m.V, T, m.fid, dict(m.names))

    # outward: signed volume per component against its nesting depth
    A, B, C = (out.V[out.T[:, k]] for k in range(3))
    vol = np.bincount(lab, np.einsum("ij,ij->i", A, np.cross(B, C)) / 6.0, ncomp)
    if ncomp > 1:
        depth = np.zeros(ncomp, int)
        probe = np.array([out.V[out.T[np.flatnonzero(lab == c)[0], 0]] for c in range(ncomp)])
        for d in range(ncomp):
            sel = lab == d
            w = winding_number(out.V, out.T[sel], probe)
            inside = np.abs(w) > 0.5
            inside[d] = False
            depth += inside
        want = np.where(depth % 2 == 0, 1.0, -1.0)
    else:
        want = np.ones(1)
    bad = np.sign(vol) != want
    fl = bad[lab]
    out.T[fl] = out.T[fl][:, [0, 2, 1]]
    rep = edge_report(out)
    return out, dict(components=int(ncomp), flipped=int(flip.sum()), inverted_components=int(bad.sum()), **rep)


def winding_number(V, T, X):
    """generalized winding number of the closed triangle set (V, T) at points X (Van Oosterom-Strackee solid angle)."""
    X = np.atleast_2d(X)
    w = np.zeros(len(X))
    for i0 in range(0, len(T), 20000):
        Tc = T[i0:i0 + 20000]
        a = V[Tc[:, 0]][None] - X[:, None]
        b = V[Tc[:, 1]][None] - X[:, None]
        c = V[Tc[:, 2]][None] - X[:, None]
        la, lb, lc = (np.linalg.norm(v, axis=2) for v in (a, b, c))
        num = np.einsum("pij,pij->pi", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("pij,pij->pi", a, b) * lc + np.einsum("pij,pij->pi", b, c) * la \
            + np.einsum("pij,pij->pi", c, a) * lb
        w += 2 * np.arctan2(num, den).sum(1)
    return w / (4 * np.pi)


def repair(m, weld_tol=None):
    """weld, drop degenerate facets, orient outward; info carries the edge report after repair."""
    m = drop_degenerate(weld(m, weld_tol))
    m, info = orient(m)
    info["watertight"] = info["boundary"] == 0 and info["nonmanifold"] == 0 and info["misoriented"] == 0
    return m, info


# ---------------------------------------------------------------- faces
def segment_faces(m, angle_deg=30.0, start=1):
    """face ids for an untagged mesh: connected regions across manifold edges whose dihedral angle (between the two
    triangle normals) is below angle_deg.  A plane, a cylinder wall or a fillet each become one face when the feature
    edges around them are sharper than the threshold.  Ids start at `start`, largest area first."""
    ncomp, lab, G, ta, tb, same = _components(m)
    N = m.normals()
    smooth = np.einsum("ij,ij->i", N[ta], N[tb]) * np.where(same, -1, 1) > np.cos(np.radians(angle_deg))
    nt = len(m.T)
    H = coo_matrix((np.ones(int(smooth.sum())), (ta[smooth], tb[smooth])), shape=(nt, nt))
    nf, f = connected_components(H, directed=False)
    area = np.bincount(f, m.areas(), nf)
    rank = np.empty(nf, np.int64)
    rank[np.argsort(-area, kind="stable")] = np.arange(nf)
    return TriMesh(m.V, m.T, rank[f] + start, {})
