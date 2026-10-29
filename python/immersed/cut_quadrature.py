"""Volume and surface quadrature on Cartesian cut cells, for immersed IGA-FEM (finite cell) on a B-spline grid.

    g = cut_cells(V, T, lo, h, n)            # cut3d geometry: exact cut volumes, wet faces, surface pieces
    q = volume_rule(g, p=3)                  # X (m,3), W (m,), cell (m,): exact for polynomials of degree <= p per cell
    s = surface_rule(g)                      # X, W, N (outward), face id, cell: the STL surface split by cell

Theory.  For a cell C cut by the closed surface, the moments m_a = int_{C cap solid} xi^a dV (xi = (x - c_C)/h, |a| <= p)
follow from the divergence theorem with F = e_x h xi_x^(a_x+1)/(a_x+1) xi_y^a_y xi_z^a_z:
    m_a = sum over the cell's wet faces and surface pieces of int F.n dS,
which cut3d integrates exactly for polynomial flux of degree <= 4 on a triangulated (STL) surface: Dunavant degree 4 on
fans of the wet face polygons and of the clipped triangles, 2x2 Gauss on full faces.  So moments are exact for p <= 3.
The cell rule is moment fitting: the (p+1)^3 Gauss points of the whole cell, with the minimum-norm weights that
reproduce every moment.  Points can lie outside the solid; that is fine because a B-spline integrand is one
polynomial on each span (cell = knot span).  The rule is exact for total degree <= p only.  A stiffness integrand has
higher degree (already y^2 z^2 for trilinear splines), so on cut cells it is an approximation of order p+1 in h;
refine or subdivide where it matters.  Uncut cells inside the solid get the plain tensor Gauss rule.

Grid.  Cells [lo + i h, lo + (i+1) h], n per axis; the solid must lie strictly inside the grid (cut3d needs a margin of
at least one cell; solid faces may lie exactly on grid planes, see test_cut3d.py::test_box_on_grid_planes)."""
import numpy as np

import cut3d

PMAX = 3


def cut_cells(V, T, lo, h, n, fid=None):
    """cut3d geometry of the closed, outward triangle mesh (V, T) on the grid lo + h * [0, n]^3.
    Raw Dunavant fans are kept (qdeg=0): the compressed per-square rule of cut3d is exact only to degree 2."""
    V = np.asarray(V, float); T = np.asarray(T, np.int64); lo = np.asarray(lo, float)
    n = np.broadcast_to(np.asarray(n, np.int64), (3,)).copy()
    if not ((V.min(0) > lo).all() and (V.max(0) < lo + h * n).all()):
        raise ValueError('the solid must lie strictly inside the grid lo + h*[0, n]^3 (cut3d needs a margin)')
    fid = np.zeros(len(T), np.int64) if fid is None else np.asarray(fid, np.int64)
    return cut3d.build(V, T, fid, lo, h, n, qdeg=0)


def _exponents(p):
    return [(a, b, d - a - b) for d in range(p + 1) for a in range(d + 1) for b in range(d - a + 1)]


def _centres(g, C):
    return g['o'] + g['h'] * (np.stack(np.unravel_index(C, tuple(g['n'])), 1) + 0.5)


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


def volume_rule(g, p=PMAX):
    """moment-fitted rule on cut cells + tensor Gauss on full cells. Returns dict X, W, cell, and diagnostics."""
    h = g['h']; E = _exponents(p); act = g['active']; cut = act & g['cut']; full = act & ~g['cut']
    gp, gw = np.polynomial.legendre.leggauss(p + 1); gp = 0.5 * gp; gw = 0.5 * gw
    P = np.stack(np.meshgrid(gp, gp, gp, indexing='ij'), -1).reshape(-1, 3)
    Pw = np.einsum('i,j,k->ijk', gw, gw, gw).ravel()
    Vm = np.stack([P[:, 0] ** a * P[:, 1] ** b * P[:, 2] ** c for a, b, c in E], 0)       # (nmom, npt)
    ic = np.nonzero(cut)[0]; M = volume_moments(g, p)[ic] / h ** 3                     # reference-cell moments
    Wc = M @ np.linalg.pinv(Vm).T                                                        # (ncut, npt), min-norm
    err = np.abs(Wc @ Vm.T - M).max() if len(ic) else 0.0
    ifl = np.nonzero(full)[0]; cells = np.r_[ic, ifl]
    Wt = np.r_[Wc, np.tile(Pw, (len(ifl), 1))] * h ** 3
    X = (_centres(g, cells)[:, None, :] + h * P[None]).reshape(-1, 3)
    return dict(X=X, W=Wt.ravel(), cell=np.repeat(cells, len(P)), p=p,
                moment_error=float(err), min_weight_rel=float(Wc.min() / Pw.min()) if len(ic) else 1.0,
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
