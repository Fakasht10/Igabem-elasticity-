# Ported from kfbim-octree (github.com/Fakasht10/kfbim-octree, fv/cut3d.py at commit c3f1253), by Aleksey Kashtelyan,
# who approved the copy into this repository on 2026-10-29.  The algorithm is unchanged; the only edits are this
# header.  Wrapper and tests: python/immersed/cut_quadrature.py, python/immersed/test_cut3d.py.
"""3D Cartesian cut cells of a closed, face-tagged triangle mesh (from stl_intake), for the cell-centred FV.

Grid: cells [o + i h, o + (i+1) h] per axis, n cells per axis; the solid lies strictly inside (margin >= 1 cell).

Construction (all compiled, no Python loops over cells, faces or triangles):
  1. Slice: for every grid plane x_a = o_a + i h, the triangles crossing it give oriented segments in the plane's
     (b, c) coordinates (a, b, c cyclic), solid on the left.  A vertex on the plane counts as the + side; the cut point
     of an edge is computed from its lower-index vertex, so neighbouring triangles share it bit for bit and the
     segments form closed curves.
  2. Grid lines in each plane: crossings of the segments with the lines c = c_k and b = b_j (half-open rule).  Node
     inside flags come from the parity along the b-lines.
  3. Wet part of every cut square (plane, j, k) = its boundary: the slice segments clipped to the square plus the
     inside intervals of its 4 edges, oriented counter-clockwise.  Quadrature: signed fan of 6-point Dunavant
     triangles (degree 4) from the wet centroid; exact for polynomials whatever the shape.  Full squares: 2x2 Gauss.
  4. Surface pieces: every triangle clipped to every cell box it overlaps (Sutherland-Hodgman), fanned, Dunavant.
  5. Cell volume and centroid by the divergence theorem over its wet faces and surface pieces (exact: degree 2).
Checks returned: total volume against the mesh volume, max |closed-surface integral of n| per cell, flag parity
mismatches along the c-lines (ties)."""
import numpy as np, numba as nb

DUN_B = np.array([[0.108103018168070, 0.445948490915965, 0.445948490915965], [0.445948490915965, 0.108103018168070, 0.445948490915965],
                  [0.445948490915965, 0.445948490915965, 0.108103018168070], [0.816847572980459, 0.091576213509771, 0.091576213509771],
                  [0.091576213509771, 0.816847572980459, 0.091576213509771], [0.091576213509771, 0.091576213509771, 0.816847572980459]])
DUN_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
G2 = np.array([-1, 1]) / np.sqrt(3.0)


# ------------------------------------------------------------------ 1. slicing
@nb.njit(cache=True)
def _slice(V, T, o, h, n, fill, sa, si, sp, sq):
    ns = 0
    for t in range(T.shape[0]):
        v0, v1, v2 = T[t, 0], T[t, 1], T[t, 2]
        nx = (V[v1, 1] - V[v0, 1]) * (V[v2, 2] - V[v0, 2]) - (V[v1, 2] - V[v0, 2]) * (V[v2, 1] - V[v0, 1])
        ny = (V[v1, 2] - V[v0, 2]) * (V[v2, 0] - V[v0, 0]) - (V[v1, 0] - V[v0, 0]) * (V[v2, 2] - V[v0, 2])
        nz = (V[v1, 0] - V[v0, 0]) * (V[v2, 1] - V[v0, 1]) - (V[v1, 1] - V[v0, 1]) * (V[v2, 0] - V[v0, 0])
        nrm = (nx, ny, nz)
        for a in range(3):
            b = (a + 1) % 3; c = (a + 2) % 3
            lo = min(V[v0, a], V[v1, a], V[v2, a]); hi = max(V[v0, a], V[v1, a], V[v2, a])
            i0 = max(0, int(np.floor((lo - o[a]) / h)) - 1); i1 = min(n[a], int(np.floor((hi - o[a]) / h)) + 1)
            for i in range(i0, i1 + 1):
                P = o[a] + i * h
                s0 = V[v0, a] >= P; s1 = V[v1, a] >= P; s2 = V[v2, a] >= P
                if s0 == s1 and s1 == s2: continue
                pts = np.empty((2, 2)); m = 0
                for (ea, eb, sa_, sb_) in ((v0, v1, s0, s1), (v1, v2, s1, s2), (v2, v0, s2, s0)):
                    if sa_ == sb_: continue
                    l, u = (ea, eb) if ea < eb else (eb, ea)
                    f = (P - V[l, a]) / (V[u, a] - V[l, a])
                    pts[m, 0] = V[l, b] + f * (V[u, b] - V[l, b]); pts[m, 1] = V[l, c] + f * (V[u, c] - V[l, c]); m += 1
                tb = -nrm[c]; tc = nrm[b]
                if (pts[1, 0] - pts[0, 0]) * tb + (pts[1, 1] - pts[0, 1]) * tc < 0:
                    k0, k1 = 1, 0
                else:
                    k0, k1 = 0, 1
                if fill:
                    sa[ns] = a; si[ns] = i
                    sp[ns, 0] = pts[k0, 0]; sp[ns, 1] = pts[k0, 1]; sq[ns, 0] = pts[k1, 0]; sq[ns, 1] = pts[k1, 1]
                ns += 1
    return ns


def slice_mesh(V, T, o, h, n):
    z = np.zeros(1, np.int64); z2 = np.zeros((1, 2))
    ns = _slice(V, T, o, h, n, False, z, z, z2, z2)
    sa = np.empty(ns, np.int64); si = np.empty(ns, np.int64); sp = np.empty((ns, 2)); sq = np.empty((ns, 2))
    _slice(V, T, o, h, n, True, sa, si, sp, sq)
    return sa, si, sp, sq


# ------------------------------------------------------------------ 2. grid-line crossings
@nb.njit(cache=True)
def _cross(sa, si, sp, sq, o, h, n, nm, fill, key, crd):
    m = 0
    for s in range(sa.shape[0]):
        a = sa[s]; i = si[s]; b = (a + 1) % 3; c = (a + 2) % 3; p = sp[s]; q = sq[s]
        for dr in range(2):              # dr 0: lines c = c_k (coordinate b); dr 1: lines b = b_j (coordinate c)
            ax = c if dr == 0 else b; cx = 1 if dr == 0 else 0; ox = 0 if dr == 0 else 1
            lo = min(p[cx], q[cx]); hi = max(p[cx], q[cx])
            k0 = max(0, int(np.floor((lo - o[ax]) / h)) - 1); k1 = min(n[ax], int(np.floor((hi - o[ax]) / h)) + 1)
            for k in range(k0, k1 + 1):
                L = o[ax] + k * h
                if (p[cx] >= L) == (q[cx] >= L): continue
                if fill:
                    f = (L - p[cx]) / (q[cx] - p[cx])
                    key[m] = ((a * (nm + 1) + i) * 2 + dr) * (nm + 1) + k; crd[m] = p[ox] + f * (q[ox] - p[ox])
                m += 1
    return m


def crossings(sa, si, sp, sq, o, h, n):
    nm = int(n.max()); z = np.zeros(1, np.int64); z1 = np.zeros(1)
    m = _cross(sa, si, sp, sq, o, h, n, nm, False, z, z1)
    key = np.empty(m, np.int64); crd = np.empty(m)
    _cross(sa, si, sp, sq, o, h, n, nm, True, key, crd)
    od = np.lexsort((crd, key)); return key[od], crd[od], nm


@nb.njit(cache=True)
def _lineptr(key, kq):
    return np.searchsorted(key, kq, 'left'), np.searchsorted(key, kq, 'right')


@nb.njit(parallel=True, cache=True)
def _flags(key, crd, o, h, n, nm, off, flags):
    """node flags along b-lines; flags[off[a] + (i*(n_b+1) + j)*(n_c+1) + k]"""
    for a in range(3):
        b = (a + 1) % 3; c = (a + 2) % 3; NL = (n[a] + 1) * (n[c] + 1)
        for L in nb.prange(NL):
            i = L // (n[c] + 1); k = L % (n[c] + 1)
            kq = ((a * (nm + 1) + i) * 2 + 0) * (nm + 1) + k
            s = np.searchsorted(key, kq, 'left'); e = np.searchsorted(key, kq, 'right'); cnt = 0; t = s
            for j in range(n[b] + 1):
                x = o[b] + j * h
                while t < e and crd[t] < x: t += 1; cnt += 1
                flags[off[a] + (i * (n[b] + 1) + j) * (n[c] + 1) + k] = (cnt % 2) == 1


@nb.njit(parallel=True, cache=True)
def _flag_check(key, crd, o, h, n, nm, off, flags):
    """walk the c-lines with the b-line flags: number of nodes whose c-line parity disagrees (ties)"""
    bad = 0
    for a in range(3):
        b = (a + 1) % 3; c = (a + 2) % 3; NL = (n[a] + 1) * (n[b] + 1)
        for L in nb.prange(NL):
            i = L // (n[b] + 1); j = L % (n[b] + 1)
            kq = ((a * (nm + 1) + i) * 2 + 1) * (nm + 1) + j
            s = np.searchsorted(key, kq, 'left'); e = np.searchsorted(key, kq, 'right'); cnt = 0; t = s
            for k in range(n[c] + 1):
                x = o[c] + k * h
                while t < e and crd[t] < x: t += 1; cnt += 1
                if ((cnt % 2) == 1) != flags[off[a] + (i * (n[b] + 1) + j) * (n[c] + 1) + k]: bad += 1
    return bad


# ------------------------------------------------------------------ 3. cut squares
@nb.njit(cache=True)
def _clip_sq(sa, si, sp, sq, o, h, n, soff, fill, skey, cp, cq):
    m = 0
    for s in range(sa.shape[0]):
        a = sa[s]; b = (a + 1) % 3; c = (a + 2) % 3; p = sp[s]; q = sq[s]
        j0 = max(0, int(np.floor((min(p[0], q[0]) - o[b]) / h))); j1 = min(n[b] - 1, int(np.floor((max(p[0], q[0]) - o[b]) / h)))
        k0 = max(0, int(np.floor((min(p[1], q[1]) - o[c]) / h))); k1 = min(n[c] - 1, int(np.floor((max(p[1], q[1]) - o[c]) / h)))
        d0 = q[0] - p[0]; d1 = q[1] - p[1]
        for j in range(j0, j1 + 1):
            for k in range(k0, k1 + 1):
                bl = o[b] + j * h; bh = o[b] + (j + 1) * h; cl = o[c] + k * h; ch = o[c] + (k + 1) * h
                t0 = 0.0; t1 = 1.0; ok = True
                for (pp, dd, lo_, hi_) in ((p[0], d0, bl, bh), (p[1], d1, cl, ch)):
                    if dd == 0.0:
                        if pp < lo_ or pp > hi_: ok = False
                    else:
                        ta = (lo_ - pp) / dd; tb = (hi_ - pp) / dd
                        if ta > tb: ta, tb = tb, ta
                        t0 = max(t0, ta); t1 = min(t1, tb)
                if not ok or t1 - t0 <= 1e-14: continue
                if fill:
                    skey[m] = soff[a] + (si[s] * n[b] + j) * n[c] + k
                    if t0 == 0.0: cp[m, 0] = p[0]; cp[m, 1] = p[1]
                    else: cp[m, 0] = p[0] + t0 * d0; cp[m, 1] = p[1] + t0 * d1
                    if t1 == 1.0: cq[m, 0] = q[0]; cq[m, 1] = q[1]
                    else: cq[m, 0] = p[0] + t1 * d0; cq[m, 1] = p[1] + t1 * d1
                m += 1
    return m


@nb.njit(cache=True)
def _edge(key, crd, kq, x0, x1, f0, out, m, fixed, fa, rev):
    """inside intervals of the grid-line segment [x0, x1) starting with flag f0; writes (start, end) along the line
    as 2D points (fixed coordinate at index fa); rev reverses each piece and the order"""
    s = np.searchsorted(key, kq, 'left'); e = np.searchsorted(key, kq, 'right')
    t = s
    while t < e and crd[t] < x0: t += 1
    f = f0; pos = x0; m0 = m
    while t < e and crd[t] < x1:
        if f:
            out[m, fa] = fixed; out[m, 1 - fa] = pos; out[m, 2 + fa] = fixed; out[m, 3 - fa] = crd[t]; m += 1
        f = not f; pos = crd[t]; t += 1
    if f:
        out[m, fa] = fixed; out[m, 1 - fa] = pos; out[m, 2 + fa] = fixed; out[m, 3 - fa] = x1; m += 1
    if rev:
        for r in range(m0, m):
            for z in range(2):
                tmp = out[r, z]; out[r, z] = out[r, 2 + z]; out[r, 2 + z] = tmp
    return m


@nb.njit(parallel=True, cache=True)
def _square_quad(usq, ptr, cp, cq, key, crd, o, h, n, nm, off, soff, flags, fill, qoff, X, W, A, I):
    """per cut square: boundary = clipped segments + edge intervals; signed Dunavant fan from the wet centroid."""
    cnt = np.zeros(usq.shape[0], np.int64)
    for u in nb.prange(usq.shape[0]):
        sk = usq[u]; a = 0
        while a < 2 and sk >= soff[a + 1]: a += 1
        b = (a + 1) % 3; c = (a + 2) % 3; r = sk - soff[a]
        i = r // (n[b] * n[c]); j = (r // n[c]) % n[b]; k = r % n[c]
        ns = ptr[u + 1] - ptr[u]
        seg = np.empty((ns + 64, 4)); m = 0
        for s in range(ptr[u], ptr[u + 1]):
            seg[m, 0] = cp[s, 0]; seg[m, 1] = cp[s, 1]; seg[m, 2] = cq[s, 0]; seg[m, 3] = cq[s, 1]; m += 1
        bl = o[b] + j * h; bh = bl + h; cl = o[c] + k * h; ch = cl + h
        # bottom (c = cl, +b), top (c = ch, -b), right (b = bh, +c), left (b = bl, -c)
        m = _edge(key, crd, ((a * (nm + 1) + i) * 2 + 0) * (nm + 1) + k, bl, bh, flags[off[a] + (i * (n[b] + 1) + j) * (n[c] + 1) + k], seg, m, cl, 1, False)
        m = _edge(key, crd, ((a * (nm + 1) + i) * 2 + 0) * (nm + 1) + k + 1, bl, bh, flags[off[a] + (i * (n[b] + 1) + j) * (n[c] + 1) + k + 1], seg, m, ch, 1, True)
        m = _edge(key, crd, ((a * (nm + 1) + i) * 2 + 1) * (nm + 1) + j + 1, cl, ch, flags[off[a] + (i * (n[b] + 1) + j + 1) * (n[c] + 1) + k], seg, m, bh, 0, False)
        m = _edge(key, crd, ((a * (nm + 1) + i) * 2 + 1) * (nm + 1) + j, cl, ch, flags[off[a] + (i * (n[b] + 1) + j) * (n[c] + 1) + k], seg, m, bl, 0, True)
        # wet area and centroid (fan from the square centre)
        x0 = bl + 0.5 * h; y0 = cl + 0.5 * h; ar = 0.0; gx = 0.0; gy = 0.0
        for t in range(m):
            ax = seg[t, 0] - x0; ay = seg[t, 1] - y0; bx = seg[t, 2] - x0; by = seg[t, 3] - y0
            w = 0.5 * (ax * by - ay * bx); ar += w; gx += w * (ax + bx) / 3; gy += w * (ay + by) / 3
        if ar <= 1e-13 * h * h: continue
        gx = x0 + gx / ar; gy = y0 + gy / ar
        nq = 0
        for t in range(m):
            ax = seg[t, 0] - gx; ay = seg[t, 1] - gy; bx = seg[t, 2] - gx; by = seg[t, 3] - gy
            w = 0.5 * (ax * by - ay * bx)
            if abs(w) <= 1e-15 * h * h: continue
            if fill:
                for g in range(6):
                    q = qoff[u] + nq + g
                    X[q, a] = o[a] + i * h
                    X[q, b] = gx + DUN_B[g, 1] * ax + DUN_B[g, 2] * bx; X[q, c] = gy + DUN_B[g, 1] * ay + DUN_B[g, 2] * by
                    W[q] = DUN_W[g] * w; A[q] = a; I[q, 0] = i; I[q, 1] = j; I[q, 2] = k
            nq += 6
        cnt[u] = nq
    return cnt


# ------------------------------------------------------------------ 4. surface pieces
@nb.njit(cache=True)
def _clip_poly(P, m, a, val, keep_ge):
    out = np.empty((m + 2, 3)); r = 0
    for e in range(m):
        A_ = P[e]; B_ = P[(e + 1) % m]
        ia = (A_[a] >= val) if keep_ge else (A_[a] <= val); ib = (B_[a] >= val) if keep_ge else (B_[a] <= val)
        if ia: out[r] = A_; r += 1
        if ia != ib:
            f = (val - A_[a]) / (B_[a] - A_[a]); out[r] = A_ + f * (B_ - A_); out[r, a] = val; r += 1
    return out, r


@nb.njit(parallel=True, cache=True)
def _surf(V, T, o, h, n, fill, toff, X, W, NR, C, F, fid):
    cnt = np.zeros(T.shape[0], np.int64)
    for t in nb.prange(T.shape[0]):
        p0 = V[T[t, 0]]; p1 = V[T[t, 1]]; p2 = V[T[t, 2]]
        nr = np.cross(p1 - p0, p2 - p0); ln = np.sqrt((nr ** 2).sum())
        if ln == 0.0: continue
        nr = nr / ln
        lo = np.empty(3, np.int64); hi = np.empty(3, np.int64)
        for a in range(3):
            lo[a] = max(0, int(np.floor((min(p0[a], p1[a], p2[a]) - o[a]) / h)))
            hi[a] = min(n[a] - 1, int(np.floor((max(p0[a], p1[a], p2[a]) - o[a]) / h)))
        nq = 0
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    P = np.empty((3, 3)); P[0] = p0; P[1] = p1; P[2] = p2; m = 3
                    cc = (ix, iy, iz)
                    for a in range(3):
                        if m == 0: break
                        P, m = _clip_poly(P, m, a, o[a] + cc[a] * h, True)
                        if m == 0: break
                        P, m = _clip_poly(P, m, a, o[a] + (cc[a] + 1) * h, False)
                    if m < 3: continue
                    cell = (ix * n[1] + iy) * n[2] + iz
                    for e in range(1, m - 1):
                        ar = 0.5 * np.sqrt((np.cross(P[e] - P[0], P[e + 1] - P[0]) ** 2).sum())
                        if ar <= 1e-15 * h * h: continue
                        if fill:
                            for g in range(6):
                                q = toff[t] + nq + g
                                X[q] = DUN_B[g, 0] * P[0] + DUN_B[g, 1] * P[e] + DUN_B[g, 2] * P[e + 1]
                                W[q] = DUN_W[g] * ar; NR[q] = nr; C[q] = cell; F[q] = fid[t]
                        nq += 6
        cnt[t] = nq
    return cnt


def _compress(Xc, Wc, Ac, Ic, sq, o, h, q):
    """moment-fitted rule per cut square: exact for all monomials of total degree <= q over the wet part (whose moments
    the signed Dunavant fan gives exactly); points = (q+1)^2 Gauss points of the whole square."""
    ns = sq.max() + 1 if len(sq) else 0
    if not len(Xc): return Xc, Wc, Ac, Ic                    # no cut square (a tile fully inside the solid, LESSONS L32)
    first = np.r_[0, np.nonzero(np.diff(sq))[0] + 1]; A = Ac[first]; I = Ic[first]
    b = (A + 1) % 3; c = (A + 2) % 3; r = np.arange(len(Xc)); a_ = Ac
    cb = o[(a_ + 1) % 3] + (Ic[:, 1] + 0.5) * h; cc = o[(a_ + 2) % 3] + (Ic[:, 2] + 0.5) * h
    xi = (Xc[r, (a_ + 1) % 3] - cb) / h; et = (Xc[r, (a_ + 2) % 3] - cc) / h
    E = [(i, d - i) for d in range(q + 1) for i in range(d + 1)]
    M = np.stack([np.bincount(sq, Wc * xi ** e1 * et ** e2, ns) for e1, e2 in E], 1)          # (ns, nmom)
    g, _ = np.polynomial.legendre.leggauss(q + 1); g = 0.5 * g
    PX, PY = np.meshgrid(g, g, indexing='ij'); PX, PY = PX.ravel(), PY.ravel()
    Vm = np.stack([PX ** e1 * PY ** e2 for e1, e2 in E], 0)                                   # (nmom, npt)
    Wn = M @ np.linalg.pinv(Vm).T                                                              # (ns, npt)
    npt = len(PX); X = np.empty((ns * npt, 3)); s_ = np.repeat(np.arange(ns), npt)
    As = np.repeat(A, npt); Is = np.repeat(I, npt, 0)
    X[np.arange(len(X)), As] = o[As] + Is[:, 0] * h
    X[np.arange(len(X)), (As + 1) % 3] = o[(As + 1) % 3] + (Is[:, 1] + 0.5 + np.tile(PX, ns)) * h
    X[np.arange(len(X)), (As + 2) % 3] = o[(As + 2) % 3] + (Is[:, 2] + 0.5 + np.tile(PY, ns)) * h
    return X, Wn.ravel(), As, Is


# ------------------------------------------------------------------ driver
def build(V, T, fid, o, h, n, qdeg=2):
    """returns dict of face points (interior), boundary points, cells; plus checks.  Cell id = (ix*ny + iy)*nz + iz."""
    o = np.asarray(o, float); n = np.asarray(n, np.int64); nm = int(n.max())
    V = np.ascontiguousarray(V, float); T = np.ascontiguousarray(T, np.int64)
    sa, si, sp, sq = slice_mesh(V, T, o, h, n)
    key, crd, nm = crossings(sa, si, sp, sq, o, h, n)
    off = np.zeros(4, np.int64); soff = np.zeros(4, np.int64)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        off[a + 1] = off[a] + (n[a] + 1) * (n[b] + 1) * (n[c] + 1); soff[a + 1] = soff[a] + (n[a] + 1) * n[b] * n[c]
    flags = np.zeros(off[3], np.bool_); _flags(key, crd, o, h, n, nm, off, flags)
    bad = _flag_check(key, crd, o, h, n, nm, off, flags)
    z1 = np.zeros(1, np.int64); z2 = np.zeros((1, 2))
    m = _clip_sq(sa, si, sp, sq, o, h, n, soff, False, z1, z2, z2)
    skey = np.empty(m, np.int64); cp = np.empty((m, 2)); cq = np.empty((m, 2)); _clip_sq(sa, si, sp, sq, o, h, n, soff, True, skey, cp, cq)
    od = np.argsort(skey, kind='stable'); skey, cp, cq = skey[od], cp[od], cq[od]
    usq, st = np.unique(skey, return_index=True); ptr = np.r_[st, len(skey)].astype(np.int64)
    zX = np.zeros((1, 3)); zW = np.zeros(1); zA = np.zeros(1, np.int64); zI = np.zeros((1, 3), np.int64); q0 = np.zeros(len(usq) + 1, np.int64)
    cnt = _square_quad(usq, ptr, cp, cq, key, crd, o, h, n, nm, off, soff, flags, False, q0, zX, zW, zA, zI)
    qoff = np.r_[0, np.cumsum(cnt)].astype(np.int64); Mq = int(qoff[-1])
    Xc = np.empty((Mq, 3)); Wc = np.empty(Mq); Ac = np.empty(Mq, np.int64); Ic = np.empty((Mq, 3), np.int64)
    _square_quad(usq, ptr, cp, cq, key, crd, o, h, n, nm, off, soff, flags, True, qoff, Xc, Wc, Ac, Ic)
    if qdeg:      # compress each cut square to a fixed (qdeg+1)^2 Gauss tensor rule on the square, weights = minimum-norm
        Xc, Wc, Ac, Ic = _compress(Xc, Wc, Ac, Ic, np.repeat(np.arange(len(usq)), np.diff(qoff)), o, h, qdeg)
    # full squares: every square of an interior plane with no clipped segment and its (j,k) corner inside
    Xf, Wf, Af, If = [], [], [], []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        fl = flags[off[a]:off[a + 1]].reshape(n[a] + 1, n[b] + 1, n[c] + 1)
        full = fl[:, :-1, :-1] & fl[:, 1:, :-1] & fl[:, :-1, 1:] & fl[:, 1:, 1:]
        sid = soff[a] + np.arange((n[a] + 1) * n[b] * n[c]).reshape(n[a] + 1, n[b], n[c])
        full &= ~np.isin(sid, usq); full[0] = False; full[-1] = False
        ii, jj, kk = np.nonzero(full)
        for g1 in G2:
            for g2 in G2:
                x = np.empty((len(ii), 3)); x[:, a] = o[a] + ii * h; x[:, b] = o[b] + (jj + 0.5 + 0.5 * g1) * h; x[:, c] = o[c] + (kk + 0.5 + 0.5 * g2) * h
                Xf.append(x); Wf.append(np.full(len(ii), h * h / 4)); Af.append(np.full(len(ii), a)); If.append(np.c_[ii, jj, kk])
    Xq = np.concatenate([Xc] + Xf); Wq = np.concatenate([Wc] + Wf); Aq = np.concatenate([Ac] + Af); Iq = np.concatenate([Ic] + If)
    full_face = np.r_[np.zeros(len(Xc), bool), np.ones(len(Xq) - len(Xc), bool)]
    # owner (lower side, normal +e_a) and neighbour cells
    ci = np.empty((len(Xq), 3), np.int64)
    for a in range(3):
        s = Aq == a; b, c = (a + 1) % 3, (a + 2) % 3
        ci[s, a] = Iq[s, 0]; ci[s, b] = Iq[s, 1]; ci[s, c] = Iq[s, 2]
    lin = lambda C: (C[:, 0] * n[1] + C[:, 1]) * n[2] + C[:, 2]
    nbr = lin(ci); cl = ci.copy(); cl[np.arange(len(ci)), Aq] -= 1; own = lin(cl)
    Nq = np.zeros((len(Xq), 3)); Nq[np.arange(len(Xq)), Aq] = 1.0
    # surface pieces
    fid = np.ascontiguousarray(fid, np.int64); zF = np.zeros(1, np.int64)
    cnt = _surf(V, T, o, h, n, False, np.zeros(len(T) + 1, np.int64), zX, zW, zX, zF, zF, fid)
    toff = np.r_[0, np.cumsum(cnt)].astype(np.int64); Mb = int(toff[-1])
    Xb = np.empty((Mb, 3)); Wb = np.empty(Mb); Nb = np.empty((Mb, 3)); Cb = np.empty(Mb, np.int64); Fb = np.empty(Mb, np.int64)
    _surf(V, T, o, h, n, True, toff, Xb, Wb, Nb, Cb, Fb, fid)
    # cell moments by the divergence theorem, about the cell centre
    ncell = int(np.prod(n)); cc = lambda C: o + h * (np.stack(np.unravel_index(C, tuple(n)), 1) + 0.5)
    def mom(C, X, W, N, sg):
        r = X - cc(C); v = sg * W * np.einsum('ij,ij->i', r, N) / 3
        m1 = sg * W[:, None] * 0.5 * r ** 2 * N; nn = sg * W[:, None] * N
        return v, m1, nn
    vol = np.zeros(ncell); m1 = np.zeros((ncell, 3)); cl_n = np.zeros((ncell, 3))
    for C, X, W, N, sg in ((own, Xq, Wq, Nq, 1.0), (nbr, Xq, Wq, Nq, -1.0), (Cb, Xb, Wb, Nb, 1.0)):
        v, mm, nn = mom(C, X, W, N, sg); vol += np.bincount(C, v, ncell)
        for a in range(3): m1[:, a] += np.bincount(C, mm[:, a], ncell); cl_n[:, a] += np.bincount(C, nn[:, a], ncell)
    cut = np.zeros(ncell, bool); cut[Cb] = True; cut[own[~full_face]] = True; cut[nbr[~full_face]] = True
    active = vol > 1e-12 * h ** 3
    xc = cc(np.arange(ncell)); xc[active] += m1[active] / vol[active, None]
    reg = active & ~cut; vol[reg] = h ** 3; xc[reg] = cc(np.nonzero(reg)[0])
    closed, vmesh = None, np.einsum('ij,ij->i', V[T[:, 0]], np.cross(V[T[:, 1]], V[T[:, 2]])).sum() / 6
    chk = dict(flag_ties=int(bad), vol_sum=float(vol[active].sum()), vol_mesh=float(vmesh),
               vol_rel=float(vol[active].sum() / vmesh - 1), closure_max=float(np.abs(cl_n[active]).max() / h ** 2),
               cells=int(active.sum()), cut_cells=int((active & cut).sum()), face_points=int(len(Xq)),
               cut_face_points=int(len(Xc)), surf_points=int(Mb), segments=int(len(sa)))
    return dict(o=o, h=h, n=n, vol=vol, xc=xc, active=active, cut=cut,
                fX=Xq, fW=Wq, fN=Nq, fown=own, fnb=nbr, fA=Aq,
                bX=Xb, bW=Wb, bN=Nb, bC=Cb, bF=Fb, check=chk)
