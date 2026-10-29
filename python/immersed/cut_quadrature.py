"""Volume and surface quadrature on Cartesian cut cells, for immersed IGA-FEM (finite cell) on a B-spline grid.

    g = cut_cells(V, T, lo, h, n)            # cut3d geometry: exact cut volumes, wet faces, surface pieces
    q = volume_rule(g, p=3)                  # X (m,3), W (m,) >= 0, cell (m,): exact for degree <= p on every cell
    s = surface_rule(g)                      # X, W, N (outward), face id, cell: the STL surface split by cell

Theory.  For a cell C cut by the closed surface, the moments m_a = int_{C cap solid} xi^a dV (xi = (x - c_C)/h, |a| <= p)
follow from the divergence theorem with F = e_x h xi_x^(a_x+1)/(a_x+1) xi_y^a_y xi_z^a_z:
    m_a = sum over the cell's wet faces and surface pieces of int F.n dS,
which cut3d integrates exactly for polynomial flux of degree <= 4 on a triangulated (STL) surface: Dunavant degree 4 on
fans of the wet face polygons and of the clipped triangles, 2x2 Gauss on full faces.  So moments are exact for p <= 3.

Rule (method='nnls', default).  Non-negative moment fitting: min ||A w - m|| subject to w >= 0, by Lawson-Hanson NNLS,
over candidate points in the closure of the cut region: the q^3 Gauss points of the cell that lie inside the solid
(ray parity, q = p+2 first), a strided subset of the cell's wet-face and surface quadrature points and, for cells
still not fitted, points stepped inward from those along the normal (slivers down to 1e-6 h are fitted).  Tchakaloff's
theorem guarantees a positive rule with at most nmom nodes in the region; NNLS finds one when the candidate set is rich
enough, and its solution is sparse (<= nmom = 20 points for p = 3, against 64 for the full Gauss grid).  A cell whose fit
is not exact (FIT_REL, FIT_ABS) is retried on a twice finer grid, then once more with the inward-stepped points; cells
that still fail are counted in `failed` (tests require 0).
Boundary points lie on the cell faces or the part surface: evaluate the span's polynomial there by the cell index, not
by a point location that could fall into the neighbouring span.
method='minnorm' is the earlier rule (all (p+1)^3 Gauss points, minimum-norm weights): exact, but weights go negative.

Either rule is exact for total degree <= p only.  A stiffness integrand has higher degree (already y^2 z^2 for trilinear
splines), so on cut cells it is an approximation; refine or subdivide where it matters.  Uncut cells inside the solid
get the plain tensor Gauss rule.

Grid.  Cells [lo + i h, lo + (i+1) h], n per axis; the solid must lie strictly inside the grid (cut3d needs a margin of
at least one cell; solid faces may lie exactly on grid planes, see test_cut3d.py::test_box_on_grid_planes)."""
import numba as nb
import numpy as np
from scipy.optimize import nnls

import cut3d

PMAX = 3
FIT_REL, FIT_ABS = 1e-11, 1e-15   # a fit passes when max |A w - m| <= FIT_REL * m_0 + FIT_ABS (reference cell, full
#                                   volume 1); FIT_ABS is the round-off floor of the flux-sum moments themselves
NI_MAX = 200                      # cap on interior (Gauss and inward-stepped) candidates per cell and round


def cut_cells(V, T, lo, h, n, fid=None):
    """cut3d geometry of the closed, outward triangle mesh (V, T) on the grid lo + h * [0, n]^3.
    Raw Dunavant fans are kept (qdeg=0): the compressed per-square rule of cut3d is exact only to degree 2."""
    V = np.ascontiguousarray(V, float); T = np.ascontiguousarray(T, np.int64); lo = np.asarray(lo, float)
    n = np.broadcast_to(np.asarray(n, np.int64), (3,)).copy()
    if not ((V.min(0) > lo).all() and (V.max(0) < lo + h * n).all()):
        raise ValueError('the solid must lie strictly inside the grid lo + h*[0, n]^3 (cut3d needs a margin)')
    fid = np.zeros(len(T), np.int64) if fid is None else np.asarray(fid, np.int64)
    g = cut3d.build(V, T, fid, lo, h, n, qdeg=0)
    g['V'] = V; g['T'] = T
    return g


def _exponents(p):
    return [(a, b, d - a - b) for d in range(p + 1) for a in range(d + 1) for b in range(d - a + 1)]


def _centres(g, C):
    return g['o'] + g['h'] * (np.stack(np.unravel_index(C, tuple(g['n'])), 1) + 0.5)


def _gauss(q):
    gp, gw = np.polynomial.legendre.leggauss(q); gp = 0.5 * gp; gw = 0.5 * gw
    P = np.stack(np.meshgrid(gp, gp, gp, indexing='ij'), -1).reshape(-1, 3)
    return P, np.einsum('i,j,k->ijk', gw, gw, gw).ravel()


def _vander(xi, E):
    return np.stack([xi[:, 0] ** a * xi[:, 1] ** b * xi[:, 2] ** c for a, b, c in E], 0)


def volume_moments(g, p=PMAX):
    """(ncell, nmom) moments int xi^a dV over the solid part of every cell, xi = (x - centre)/h; exponents _exponents(p)"""
    if p > PMAX: raise ValueError('moments are exact only for p <= %d with cut3d\'s degree-4 surface rules' % PMAX)
    h = g['h']; ncell = int(np.prod(g['n'])); E = _exponents(p)
    M = np.zeros((ncell, len(E)))
    for C, X, W, N, sg in ((g['fown'], g['fX'], g['fW'], g['fN'], 1.0), (g['fnb'], g['fX'], g['fW'], g['fN'], -1.0),
                           (g['bC'], g['bX'], g['bW'], g['bN'], 1.0)):
        xi = (X - _centres(g, C)) / h; w = sg * W * N[:, 0] * h
        for k, (a, b, c) in enumerate(E):
            M[:, k] += np.bincount(C, w * xi[:, 0] ** (a + 1) / (a + 1) * xi[:, 1] ** b * xi[:, 2] ** c, ncell)
    return M


# ------------------------------------------------------------------ inside test (ray parity along +x)
def _columns(V, T, g):
    """triangles bucketed by the (iy, iz) grid column their yz bounding box touches: CSR (start, tri)"""
    o, h, n = g['o'], g['h'], g['n']
    P = V[T]; lo = np.floor((P[:, :, 1:].min(1) - o[1:]) / h).astype(np.int64).clip(0, n[1:] - 1)
    hi = np.floor((P[:, :, 1:].max(1) - o[1:]) / h).astype(np.int64).clip(0, n[1:] - 1)
    ny = hi[:, 0] - lo[:, 0] + 1; nz = hi[:, 1] - lo[:, 1] + 1; cnt = ny * nz
    tri = np.repeat(np.arange(len(T)), cnt); r = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    col = (lo[tri, 0] + r // nz[tri]) * n[2] + lo[tri, 1] + r % nz[tri]; od = np.argsort(col, kind='stable')
    start = np.searchsorted(col[od], np.arange(n[1] * n[2] + 1))
    return start.astype(np.int64), tri[od].astype(np.int64)


@nb.njit(cache=True)
def _edge_in(e, dy, dz):
    """half-open (top-left) edge rule for a counter-clockwise (y, z) triangle: a point on an edge shared by two
    triangles is counted by exactly one of them"""
    return e > 0.0 or (e == 0.0 and (dz < 0.0 or (dz == 0.0 and dy < 0.0)))


@nb.njit(parallel=True, cache=True)
def _parity(X, V, T, start, tri, o, h, n):
    out = np.zeros(X.shape[0], np.bool_)
    for i in nb.prange(X.shape[0]):
        y = X[i, 1]; z = X[i, 2]
        j = min(max(int(np.floor((y - o[1]) / h)), 0), n[1] - 1); k = min(max(int(np.floor((z - o[2]) / h)), 0), n[2] - 1)
        c = j * n[2] + k; cnt = 0
        for s in range(start[c], start[c + 1]):
            t = tri[s]; a = V[T[t, 0]]; b = V[T[t, 1]]; d = V[T[t, 2]]
            e0 = (b[1] - a[1]) * (z - a[2]) - (b[2] - a[2]) * (y - a[1])
            e1 = (d[1] - b[1]) * (z - b[2]) - (d[2] - b[2]) * (y - b[1])
            e2 = (a[1] - d[1]) * (z - d[2]) - (a[2] - d[2]) * (y - d[1])
            ar = e0 + e1 + e2
            if ar == 0.0: continue                                  # triangle edge-on to the ray
            if ar < 0.0:                                            # make it counter-clockwise in (y, z)
                b, d = d, b; e0, e1, e2, ar = -e2, -e1, -e0, -ar
            if not (_edge_in(e0, b[1] - a[1], b[2] - a[2]) and _edge_in(e1, d[1] - b[1], d[2] - b[2])
                    and _edge_in(e2, a[1] - d[1], a[2] - d[2])): continue
            xs = (e1 * a[0] + e2 * b[0] + e0 * d[0]) / ar           # barycentric x of the ray hit
            if xs > X[i, 0]: cnt += 1
        out[i] = (cnt % 2) == 1
    return out


def inside(g, X):
    """True where X lies inside the solid: parity of the +x ray, half-open edge rule so shared edges and vertices count
    once.  A point exactly on the surface is inside or outside, either way."""
    if 'V' not in g: raise ValueError('inside() needs the mesh: build the geometry with cut_cells, not cut3d.build')
    if '_cols' not in g: g['_cols'] = _columns(g['V'], g['T'], g)
    start, tri = g['_cols']
    return _parity(np.ascontiguousarray(X, float), g['V'], g['T'], start, tri, g['o'], g['h'], g['n'])


# ------------------------------------------------------------------ rules
def _boundary_candidates(g):
    """(cell, point, inward unit normal) on the closure of each cell's solid part: wet-face points (both sides) and
    surface points"""
    s = surface_rule(g); wet = g['fW'] > 0
    C = np.r_[g['fown'][wet], g['fnb'][wet], s['cell']]
    X = np.r_[g['fX'][wet], g['fX'][wet], s['X']]
    N = np.r_[-g['fN'][wet], g['fN'][wet], -s['N']]           # the face is the owner's upper side
    ok = (C >= 0) & (C < int(np.prod(g['n'])))
    return C[ok], X[ok], N[ok]


OFFSETS = np.geomspace(1e-5, 0.5, 14)       # round 3: candidates at x_b + t h n_in, for slivers thinner than h/q


def _strided(k, cap):
    """at most cap evenly spread entries of the index array k"""
    return k if len(k) <= cap else k[np.linspace(0, len(k) - 1, cap).astype(np.int64)]


def _nnls_cells(g, ic, M, E, q0, nb_max=24):
    """per cut cell: NNLS weights over candidates in the closure of the cell's solid part, in up to three rounds for
    the cells not yet fitted: (1) inside Gauss points at q0, (2) at 2 q0, (3) at 4 q0 plus points stepped inward from
    the boundary points (OFFSETS: slivers thinner than the Gauss spacing).  Each round adds up to nb_max * 2^round
    boundary points; interior candidates are capped at NI_MAX per kind.  Returns lists of (xi, w), errors, failures"""
    h = g['h']
    bC, bX, bN = _boundary_candidates(g); od = np.argsort(bC, kind='stable'); bC, bX, bN = bC[od], bX[od], bN[od]
    bs = np.searchsorted(bC, ic, 'left'); be = np.searchsorted(bC, ic, 'right')
    cen = _centres(g, ic); tol = FIT_REL * np.abs(M[:, 0]) + FIT_ABS
    out_xi = [np.zeros((0, 3))] * len(ic); out_w = [np.zeros(0)] * len(ic)
    err = np.full(len(ic), np.inf); todo = np.arange(len(ic))
    for rnd in range(3):
        if not len(todo): break
        q = q0 * 2 ** rnd; P, _ = _gauss(q)
        ins = inside(g, (cen[todo][:, None, :] + h * P[None]).reshape(-1, 3)).reshape(len(todo), -1)
        sel = [_strided(np.arange(bs[u], be[u]), nb_max * 2 ** rnd) for u in todo]
        xis = [np.r_[P[_strided(np.nonzero(ins[r])[0], NI_MAX)], (bX[sel[r]] - cen[u]) / h] for r, u in enumerate(todo)]
        if rnd == 2:                                                                 # inward-stepped candidates
            k = np.concatenate(sel).astype(np.int64); cell = np.repeat(np.arange(len(todo)), [len(x) for x in sel])
            Y = (bX[k][:, None, :] + h * OFFSETS[None, :, None] * bN[k][:, None, :]).reshape(-1, 3)
            cy = np.repeat(cell, len(OFFSETS)); xy = (Y - cen[todo][cy]) / h
            ok = (np.abs(xy) < 0.5).all(1); ok[ok] = inside(g, Y[ok]); xy, cy = xy[ok], cy[ok]   # cy is sorted
            lo_, hi_ = np.searchsorted(cy, np.arange(len(todo)), 'left'), np.searchsorted(cy, np.arange(len(todo)), 'right')
            xis = [np.r_[x, xy[_strided(np.arange(lo_[r], hi_[r]), NI_MAX)]] for r, x in enumerate(xis)]
        e = np.full(len(todo), np.inf)
        for r, u in enumerate(todo):
            if r % 1024 == 0:                                       # batched Vandermonde, 1024 cells at a time
                ch = xis[r:r + 1024]; ptr = np.r_[0, np.cumsum([len(x) for x in ch])]
                Ach = _vander(np.concatenate(ch), E) if ptr[-1] else np.zeros((len(E), 0))
            if not len(xis[r]): continue
            A = np.ascontiguousarray(Ach[:, ptr[r % 1024]:ptr[r % 1024 + 1]])
            try:
                w, _ = nnls(A, M[u], maxiter=50 * A.shape[1])
            except RuntimeError:                                                     # no convergence: next round
                continue
            e[r] = np.abs(A @ w - M[u]).max()
            if e[r] < err[u]: keep = w > 0; err[u] = e[r]; out_xi[u] = xis[r][keep]; out_w[u] = w[keep]
        todo = todo[e > tol[todo]]
    return out_xi, out_w, err, len(todo)


def volume_rule(g, p=PMAX, method='nnls'):
    """quadrature on the solid: a moment-fitted rule on every cut cell + tensor Gauss on full cells.
    Returns dict X, W, cell, and diagnostics (moment_error on the reference cell, min weight, failed fits)."""
    h = g['h']; E = _exponents(p); act = g['active']; cut = act & g['cut']; full = act & ~g['cut']
    P, Pw = _gauss(p + 1)
    ic = np.nonzero(cut)[0]; M = volume_moments(g, p)[ic] / h ** 3                     # reference-cell moments
    ifl = np.nonzero(full)[0]
    Xf = (_centres(g, ifl)[:, None, :] + h * P[None]).reshape(-1, 3); Wf = np.tile(Pw, len(ifl)) * h ** 3
    Cf = np.repeat(ifl, len(P))
    if method == 'minnorm':
        Vm = _vander(P, E); Wc = M @ np.linalg.pinv(Vm).T                               # (ncut, npt)
        err = np.abs(Wc @ Vm.T - M).max(1) if len(ic) else np.zeros(0)
        Xc = (_centres(g, ic)[:, None, :] + h * P[None]).reshape(-1, 3); Wc = Wc.ravel() * h ** 3
        Cc = np.repeat(ic, len(P)); failed = 0
    elif method == 'nnls':
        xi, w, err, failed = _nnls_cells(g, ic, M, E, q0=p + 2)
        cnt = np.array([len(a) for a in w], np.int64)
        Cc = np.repeat(ic, cnt); Xc = _centres(g, Cc) + h * (np.concatenate(xi) if len(ic) else np.zeros((0, 3)))
        Wc = (np.concatenate(w) if len(ic) else np.zeros(0)) * h ** 3
    else:
        raise ValueError("method is 'nnls' or 'minnorm'")
    W = np.r_[Wc, Wf]
    return dict(X=np.r_[Xc, Xf], W=W, cell=np.r_[Cc, Cf], p=p, method=method,
                moment_error=float(err.max()) if len(ic) else 0.0, failed=int(failed),
                min_weight_rel=float(Wc.min() / (Pw.min() * h ** 3)) if len(Wc) else 1.0,
                points_per_cut_cell=float(len(Wc) / max(len(ic), 1)),
                cut_cells=int(len(ic)), full_cells=int(len(ifl)))


def surface_rule(g):
    """surface quadrature with its cell re-assigned to the active cell the point lies in.  cut3d gives a triangle lying
    exactly on a grid plane to the cell above the plane, which is outside the solid for an upper face."""
    X, h, o, n = g['bX'], g['h'], g['o'], g['n']
    C = g['bC'].copy(); act = g['active']
    bad = ~act[C]
    if bad.any():
        ijk = np.stack(np.unravel_index(C[bad], tuple(n)), 1)
        best = C[bad].copy(); found = np.zeros(bad.sum(), bool)
        for a in range(3):                       # the neighbour below across the plane the point lies on
            on = np.isclose(X[bad, a], o[a] + ijk[:, a] * h, rtol=0, atol=1e-12 * h) & ~found
            j = ijk.copy(); j[:, a] -= 1
            cand = np.ravel_multi_index(j.T.clip(0), tuple(n))
            ok = on & (j[:, a] >= 0) & act[cand]; best[ok] = cand[ok]; found |= ok
        C[np.nonzero(bad)[0]] = best
    return dict(X=X, W=g['bW'], N=g['bN'], face=g['bF'], cell=C, orphan=int((~act[C]).sum()))
