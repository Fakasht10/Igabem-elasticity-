"""Machined-notch bead window (M3): the 0/90 bead lattice of rve.window with a U-notch machined through it.

A machined notch (Cicero et al. 2020: notches machined into printed SENB specimens) cuts straight through the
raster: the material is the bead lattice minus the notch N = {disk (xc, yc), radius rho} U {slot |x - xc| < rho,
y < yc}, a prism along the build direction z. Every bead is z-extruded in plan view, so a bead piece is
{(a, s, z): (a, s) in its footprint minus N, |z - zc| <= zsec(s)} (a along the filament, s across it, zsec the
section half height: the interlayer flat h/2 for |s| <= b_int/2, the corner arcs out to the lateral tip). The cut is
therefore a plan-view problem:

  footprint grid   per bead: lines along the filament at the section breaks (axis, flat ends, arc splits, lateral
                   tips; arcs refined in their own parameter) and across it at the piece breaks (cell ends, weld
                   square edges), plus the lines x = xc and y = yc through the notch apex and the slot-wall tangency
                   points, so each grid rectangle sees one monotone arc or one straight wall
  cut templates    a rectangle cut by N becomes 1-4 quads (one template per corner pattern, in the frame where the
                   disk is to the lower left); new vertices only inside the rectangle or at the midpoint of a
                   partially kept edge, so neighbouring rectangles (also across beads: lateral welds, weld squares)
                   stay conforming without changing uncut rectangles
  lift             a plan-view edge becomes the 3D edge on the bead surface z = zc +- zsec(s): exact lines along the
                   filament, exact rational sub-arcs of the section across it, exact on the flats; the arc and spoke
                   edges on the curved corners are fitted quadratics, the patch interiors are Coons (planar faces)
                   or least squares (corners)
  faces            top/bottom surfaces of the footprint quads, lateral welds (s = +-w/2), caps (piece ends: filament
                   joints or window cut faces, exact under-graph patches) and the notch curtains (ruled from the
                   axis plane to the lifted cut edge); all single-span biquadratic, conforming

Uncut pieces away from the extra grid lines are the canonical A piece (B = A turned 90 deg, as bead_090): one cached
operator for all of them; the rest are written as their own .dat (pieces()).

usage (from python/): python3 -m rve.notch [--rho 1] [--cells 6:5] [--out dir]   writes the pieces and a figure
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_geometries import Patch, write_dat  # noqa: E402
from rve.beads import K2, corner_arc  # noqa: E402

CICERO = dict(w=0.4, h=0.3, b_int=0.24, b_lat=0.067, alpha=15.0, wc=math.sqrt(0.5))   # line 0.4, layer 0.3 mm


# ------------------------------------------------------------------ rational quadratics (homogeneous, 3 x (d+1))
def _norm(C):
    """Same curve with end weights 1."""
    C = np.asarray(C, float)
    w0, w2 = C[0, -1], C[2, -1]
    return np.array([C[0] / w0, C[1] / math.sqrt(w0 * w2), C[2] / w2])


def _sub(C, t0, t1):
    """Exact piece [t0, t1] of a rational quadratic (blossoming), end weights 1."""
    b = lambda s, t: (1 - s) * (1 - t) * C[0] + ((1 - s) * t + s * (1 - t)) * C[1] + s * t * C[2]  # noqa: E731
    return _norm([b(t0, t0), b(t0, t1), b(t1, t1)])


def _pt(C, t):
    h = (1 - t) ** 2 * C[0] + 2 * t * (1 - t) * C[1] + t ** 2 * C[2]
    return h[:-1] / h[-1]


def _line(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.array([np.r_[p, 1.0], np.r_[(p + q) / 2, 1.0], np.r_[q, 1.0]])


# ------------------------------------------------------------------ bead section
class Section:
    """Half height zsec(s) of the bead section (|s| <= w/2) and its exact pieces; bands between breaks are either the
    interlayer flat (|s| <= b) or one half of a corner arc (rational quadratic, beads.corner_arc)."""

    def __init__(self, g, refine=1):
        self.g = g
        w, h, b_int, b_lat = g["w"], g["h"], g["b_int"], g["b_lat"]
        self.b = b_int / 2
        P0, P1, P2 = corner_arc((0.0, 0.0), 1, 1, w, h, b_int, b_lat, g["alpha"], g["wc"])   # lateral -> interlayer tip
        self.arc = np.array([[*P0, 1.0], [*(np.asarray(P1) * g["wc"]), g["wc"]], [*P2, 1.0]])
        n = 2 ** refine
        ts = np.linspace(0, 1, 2 * n + 1)                                   # both arc halves, n pieces each
        s_arc = sorted(self._s(t) for t in ts)
        s_flat = list(np.linspace(0, self.b, n + 1))
        pos = sorted(set(np.round(s_flat + s_arc, 14)))
        self.breaks = np.array(sorted(set([-s for s in pos] + pos)))      # all section breaks, -w/2 .. w/2

    def _s(self, t):
        return _pt(self.arc, t)[0]

    def tau(self, s):
        """Arc parameter of |s| in [b, w/2] (s decreases from w/2 at t = 0 to b at t = 1)."""
        s = abs(s)
        c = self.arc
        a0, a1, a2 = (c[k, 0] - s * c[k, 2] for k in range(3))
        A, B, C = a0 - 2 * a1 + a2, 2 * (a1 - a0), a0
        if abs(A) < 1e-14:
            return -C / B
        r = np.roots([A, B, C]).real
        r = r[(r > -1e-9) & (r < 1 + 1e-9)]
        return float(np.clip(r[0], 0, 1))

    def flat(self, s):
        return abs(s) <= self.b + 1e-12

    def z(self, s):
        return self.g["h"] / 2 if self.flat(s) else _pt(self.arc, self.tau(s))[1]

    def curve(self, s0, s1):
        """Section piece between s0 and s1 (same sign, same band) as homogeneous (s, z, w), z >= 0."""
        if self.flat(0.5 * (s0 + s1)):
            return _line([s0, self.g["h"] / 2], [s1, self.g["h"] / 2])
        sg = 1.0 if s0 + s1 > 0 else -1.0
        C = _sub(self.arc, self.tau(s0), self.tau(s1))
        C[:, 0] *= sg
        return C


# ------------------------------------------------------------------ notch (plan view)
class Notch:
    """U-notch: disk (xc, yc, rho) plus the slot |x - xc| < rho below yc. rho = 0 or None: no notch."""

    def __init__(self, xc=0.0, yc=0.0, rho=None):
        self.xc, self.yc, self.rho = xc, yc, rho

    def inside(self, p):
        if not self.rho:
            return False
        x, y = p[0] - self.xc, p[1] - self.yc
        return (x * x + y * y < self.rho ** 2) if y >= 0 else abs(x) < self.rho

    def arc(self, p, q):
        """Exact circle arc p -> q (homogeneous, 2D)."""
        c = np.array([self.xc, self.yc])
        a0, a1 = math.atan2(p[1] - c[1], p[0] - c[0]), math.atan2(q[1] - c[1], q[0] - c[0])
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        am = a0 + d / 2
        wm = math.cos(d / 2)
        m = c + self.rho / wm * np.array([math.cos(am), math.sin(am)])
        return np.array([np.r_[p, 1.0], np.r_[m * wm, wm], np.r_[q, 1.0]])

    def arc_mid(self, p, q):
        c = np.array([self.xc, self.yc])
        a0, a1 = math.atan2(p[1] - c[1], p[0] - c[0]), math.atan2(q[1] - c[1], q[0] - c[0])
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        return c + self.rho * np.array([math.cos(a0 + d / 2), math.sin(a0 + d / 2)])

    def _cross(self, p, q):
        """Crossing point of the axis-aligned segment p q (one end inside) with the notch boundary."""
        if abs(p[1] - q[1]) < 1e-14:                                       # horizontal
            y = p[1]
            r = math.sqrt(max(self.rho ** 2 - (y - self.yc) ** 2, 0.0)) if y >= self.yc else self.rho
            xs = [self.xc - r, self.xc + r]
            x = min(xs, key=lambda v: abs(v - 0.5 * (p[0] + q[0])) if min(p[0], q[0]) - 1e-12 <= v <= max(p[0], q[0]) + 1e-12 else np.inf)
            return np.array([x, y])
        x = p[0]                                                            # vertical (arc region only)
        return np.array([x, self.yc + math.sqrt(max(self.rho ** 2 - (x - self.xc) ** 2, 0.0))])

    def pieces(self, p, q):
        """Kept parts of the grid segment p q: [] (removed), [(p, q)] (whole), or the kept part split at its
        midpoint (a partially kept edge; both rectangles sharing it make the same split)."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        ip, iq = self.inside(p), self.inside(q)
        if not ip and not iq:
            return [(p, q)]
        if ip and iq:
            return []
        c = self._cross(p, q)
        a, b = (c, q) if ip else (p, c)
        m = 0.5 * (a + b)
        return [(a, m), (m, b)]


# ------------------------------------------------------------------ cut templates (plan view)
def decompose(N, x0, x1, y0, y1):
    """Quads of the rectangle minus N: [(points (4, 2), kinds (4,))], the edge k running from point k to k + 1,
    kinds 'g' (grid line), 's' (spoke inside the rectangle), 'n' (notch boundary: arc or slot wall). The rectangle
    lies on one side of x = xc and of y = yc."""
    LL, LR, UR, UL = np.array([x0, y0]), np.array([x1, y0]), np.array([x1, y1]), np.array([x0, y1])
    corners = [N.inside(c) for c in (LL, LR, UR, UL)]
    if not any(corners) and not N.inside(0.5 * (LL + UR)):
        return [(np.array([LL, LR, UR, UL]), "gggg")]
    if all(corners):
        return []
    mid = lambda a, b: 0.5 * (a + b)  # noqa: E731
    if y1 <= N.yc + 1e-12:                                                  # slot: one wall crosses the rectangle
        xw = N.xc + N.rho if x1 > N.xc + N.rho else N.xc - N.rho
        if x0 < N.xc:                                                       # left wall: material on the left
            Wb, Wt = np.array([xw, y0]), np.array([xw, y1])
            mB, mT = mid(LL, Wb), mid(UL, Wt)
            return [(np.array([LL, mB, mT, UL]), "gsgg"), (np.array([mB, Wb, Wt, mT]), "gngs")]
        Wb, Wt = np.array([xw, y0]), np.array([xw, y1])
        mB, mT = mid(Wb, LR), mid(Wt, UR)
        return [(np.array([Wb, mB, mT, Wt]), "gsgn"), (np.array([mB, LR, UR, mT]), "gggs")]
    # arc region: frame with the disk centre at the origin and the rectangle in the first quadrant
    fx = 1.0 if x0 >= N.xc - 1e-12 else -1.0
    to = lambda P: np.array([N.xc + fx * P[0], N.yc + P[1]])  # noqa: E731
    X0, X1 = sorted((fx * (x0 - N.xc), fx * (x1 - N.xc)))
    Y0, Y1 = y0 - N.yc, y1 - N.yc
    r2 = N.rho ** 2
    ins = lambda X, Y: X * X + Y * Y < r2  # noqa: E731
    cLL, cLR, cUL = ins(X0, Y0), ins(X1, Y0), ins(X0, Y1)
    L, R = np.array([X0, Y0]), np.array([X1, Y0])
    T_, U = np.array([X1, Y1]), np.array([X0, Y1])                         # frame LL, LR, UR, UL
    cx = lambda Y: math.sqrt(max(r2 - Y * Y, 0.0))  # noqa: E731
    out = []
    if cLL and not cLR and not cUL:                                         # one corner in: pentagon -> 4 quads
        P, Q = np.array([cx(Y0), Y0]), np.array([X0, cx(X0)])
        mB, mL = mid(P, R), mid(Q, U)
        a = N.arc_mid(to(Q), to(P))
        a = np.array([fx * (a[0] - N.xc), a[1] - N.yc])
        c = 0.25 * (mB + T_ + mL + a)
        out = [([P, mB, c, a], "gssn"), ([mB, R, T_, c], "ggss"), ([c, T_, U, mL], "sggs"), ([a, c, mL, Q], "ssgn")]
    elif cLL and cLR and not cUL:                                           # bottom edge in: 2 quads
        Lp, Rp = np.array([X0, cx(X0)]), np.array([X1, cx(X1)])
        mL, mR = mid(Lp, U), mid(Rp, T_)
        out = [([Lp, Rp, mR, mL], "ngsg"), ([mL, mR, T_, U], "sggg")]
    elif cLL and cUL and not cLR:                                           # left edge in: 2 quads
        Bp, Tp = np.array([cx(Y0), Y0]), np.array([cx(Y1), Y1])
        mB, mT = mid(Bp, R), mid(Tp, T_)
        out = [([Bp, mB, mT, Tp], "gsgn"), ([mB, R, T_, mT], "gggs")]
    elif cLL and cLR and cUL:                                               # three corners in: 3 quads
        Rp, Tp = np.array([X1, cx(X1)]), np.array([cx(Y1), Y1])
        mR, mT = mid(Rp, T_), mid(T_, Tp)
        a = N.arc_mid(to(Tp), to(Rp))
        a = np.array([fx * (a[0] - N.xc), a[1] - N.yc])
        c = 0.25 * (mR + T_ + mT + a)
        out = [([Rp, mR, c, a], "gssn"), ([mR, T_, mT, c], "ggss"), ([c, mT, Tp, a], "sgns")]
    else:
        raise ValueError("notch cut: unexpected corner pattern in [%g, %g] x [%g, %g]" % (x0, x1, y0, y1))
    res = []
    for P, k in out:
        G = np.array([to(p) for p in P])
        if fx < 0:                                                          # the reflection reverses the loop
            G, k = G[::-1], "".join(reversed(k[3:] + k[:3]))
        res.append((G, k))
    return res


# ------------------------------------------------------------------ pieces
class Bead:
    """Plan-view frame of one bead: layer 'A' (along x at y = c, zc = 0) or 'B' (along y at x = c, zc = h);
    local (a, s): A (x, y - c), B (y, c - x) (B = A turned 90 deg)."""

    def __init__(self, layer, c, h):
        self.layer, self.c = layer, c
        self.zc = 0.0 if layer == "A" else h

    def local(self, p):
        return (p[0], p[1] - self.c) if self.layer == "A" else (p[1], self.c - p[0])

    def plan(self, a, s):
        return np.array([a, self.c + s]) if self.layer == "A" else np.array([self.c - s, a])


class PieceBuilder:
    """Patches of bead pieces of one window (lattice pitch w, cell centres at multiples of w)."""

    def __init__(self, g, notch, refine=1, extra_x=(), extra_y=()):
        self.g, self.N, self.r = g, notch, refine
        self.sec = Section(g, refine)
        self.extra_x, self.extra_y = list(extra_x), list(extra_y)
        w, b = g["w"], g["b_int"] / 2
        n = 2 ** refine
        base = [-w / 2, -b, 0.0, b, w / 2]
        self.abreaks = np.array(sorted(set(np.round(np.concatenate(
            [np.linspace(base[k], base[k + 1], n + 1) for k in range(4)]), 14))))
        self.edges = {}

    # breaks of one piece in global plan coordinates
    def _breaks(self, bead, a_c):
        a = list(a_c + self.abreaks)
        s = list(self.sec.breaks)
        ex_a, ex_s = (self.extra_x, self.extra_y) if bead.layer == "A" else (self.extra_y, self.extra_x)
        for v in ex_a:
            if a[0] + 1e-9 < v < a[-1] - 1e-9:
                a.append(v)
        for v in ex_s:
            sv = v - bead.c if bead.layer == "A" else bead.c - v
            if s[0] + 1e-9 < sv < s[-1] - 1e-9:
                s.append(sv)
        return np.array(sorted(set(np.round(a, 13)))), np.array(sorted(set(np.round(s, 13))))

    def _key(self, *pts):
        return tuple(tuple(np.round(np.asarray(p, float), 9)) for p in pts)

    def lift(self, bead, sz, p, q, kind):
        """3D edge (3 homogeneous control points) of the plan edge p -> q on the surface z = zc + sz zsec(s)."""
        k = (bead.layer, round(bead.c, 9), sz) + self._key(p, q)
        if k in self.edges:
            return self.edges[k]
        kr = (bead.layer, round(bead.c, 9), sz) + self._key(q, p)
        if kr in self.edges:
            return self.edges[kr][::-1].copy()
        (ap, sp), (aq, sq) = bead.local(p), bead.local(q)
        zc, sec = bead.zc, self.sec
        to3 = lambda P2, z: np.r_[P2, z]  # noqa: E731
        sm = 0.5 * (sp + sq)
        if abs(sp - sq) < 1e-12:                                            # along the filament: line
            z = zc + sz * sec.z(sp)
            C = _line(to3(p, z), to3(q, z))
        elif abs(ap - aq) < 1e-12:                                          # across: exact section piece
            S = sec.curve(sp, sq)
            C = np.array([np.r_[bead.plan(ap, Si[0] / Si[2]) * Si[2], (zc + sz * Si[1] / Si[2]) * Si[2], Si[2]] for Si in S])
        else:
            C2 = self.N.arc(p, q) if kind == "n" and self._on_arc(p, q) else _line(p, q)
            if sec.flat(sm):
                z = zc + sz * self.g["h"] / 2
                C = np.array([np.r_[c[:2], z * c[2], c[2]] for c in C2])
            else:                                                           # curved corner: fitted quadratic
                ts = np.linspace(0, 1, 21)[1:-1]
                X = []
                for t in ts:
                    P2 = _pt(C2, t)
                    X.append(np.r_[P2, zc + sz * sec.z(bead.local(P2)[1])])
                X = np.array(X)
                P0 = np.r_[p, zc + sz * sec.z(sp)]
                P2_ = np.r_[q, zc + sz * sec.z(sq)]
                B1 = 2 * ts * (1 - ts)
                M = ((B1[:, None] * (X - ((1 - ts) ** 2)[:, None] * P0 - (ts ** 2)[:, None] * P2_)).sum(0)
                     / (B1 ** 2).sum())
                C = np.array([np.r_[P0, 1.0], np.r_[M, 1.0], np.r_[P2_, 1.0]])
        self.edges[k] = C
        return C

    def _on_arc(self, p, q):
        N = self.N
        d = lambda P: math.hypot(P[0] - N.xc, P[1] - N.yc)  # noqa: E731
        return bool(N.rho) and abs(d(p) - N.rho) < 1e-9 and abs(d(q) - N.rho) < 1e-9 \
            and min(p[1], q[1]) >= N.yc - 1e-12 and abs(p[0] - q[0]) > 1e-12

    def flat_edge(self, p, q, z):
        """Plan edge p -> q at height z (exact: arc or line)."""
        C2 = self.N.arc(p, q) if self._on_arc(p, q) else _line(p, q)
        return np.array([np.r_[c[:2], z * c[2], c[2]] for c in C2])

    def piece(self, bead, a_c):
        """Patches of the piece of `bead` whose cell is centred at a = a_c (empty if the notch removes it)."""
        A_, S_ = self._breaks(bead, a_c)
        sec, N, g = self.sec, self.N, self.g
        nv = 2 ** self.r
        out = []
        for sz in (1.0, -1.0):
            for i in range(len(A_) - 1):
                for j in range(len(S_) - 1):
                    P = [bead.plan(A_[i], S_[j]), bead.plan(A_[i + 1], S_[j]), bead.plan(A_[i + 1], S_[j + 1]), bead.plan(A_[i], S_[j + 1])]
                    xs, ys = sorted({P[0][0], P[2][0]}), sorted({P[0][1], P[2][1]})
                    for Q, kinds in decompose(N, xs[0], xs[1], ys[0], ys[1]):
                        out.append(self._surface(bead, sz, Q, kinds))
                        for e in range(4):
                            if kinds[e] == "n":
                                out += self._curtain(bead, sz, Q[e], Q[(e + 1) % 4], nv)
            # lateral welds s = +-w/2
            for s_l in (S_[0], S_[-1]):
                for i in range(len(A_) - 1):
                    for p, q in N.pieces(bead.plan(A_[i], s_l), bead.plan(A_[i + 1], s_l)):
                        zt = bead.zc + sz * sec.z(s_l)
                        for k in range(nv):
                            z0 = bead.zc + (zt - bead.zc) * k / nv
                            z1 = bead.zc + (zt - bead.zc) * (k + 1) / nv
                            out.append(_quad_flat([np.r_[p, z0], np.r_[q, z0], np.r_[q, z1], np.r_[p, z1]]))
            # caps at the piece ends
            for a_e in (A_[0], A_[-1]):
                for j in range(len(S_) - 1):
                    for p, q in N.pieces(bead.plan(a_e, S_[j]), bead.plan(a_e, S_[j + 1])):
                        top = self.lift(bead, sz, p, q, "g")
                        bot = _line(np.r_[p, bead.zc], np.r_[q, bead.zc])
                        out += _ruled_split(bot, top, nv)
        return _orient_all(out)

    def _surface(self, bead, sz, Q, kinds):
        E = [self.lift(bead, sz, Q[e], Q[(e + 1) % 4], kinds[e]) for e in range(4)]
        Pw = np.zeros((3, 3, 4))
        Pw[0, :] = E[0]
        Pw[:, 2] = E[1]
        Pw[2, :] = E[2][::-1]
        Pw[:, 0] = E[3][::-1]
        Pw[1, 1] = 0.5 * (Pw[1, 0] + Pw[1, 2] + Pw[0, 1] + Pw[2, 1]) - 0.25 * (Pw[0, 0] + Pw[0, 2] + Pw[2, 0] + Pw[2, 2])
        sm = 0.5 * (min(bead.local(q)[1] for q in Q) + max(bead.local(q)[1] for q in Q))
        if not self.sec.flat(sm):                                           # curved corner: least squares centre
            F = [self.flat_edge(Q[e], Q[(e + 1) % 4], 0.0) if kinds[e] == "n" else _line(np.r_[Q[e], 0.0], np.r_[Q[(e + 1) % 4], 0.0])
                 for e in range(4)]
            G = np.zeros((3, 3, 4))
            G[0, :], G[:, 2], G[2, :], G[:, 0] = F[0], F[1], F[2][::-1], F[3][::-1]
            G[1, 1] = 0.5 * (G[1, 0] + G[1, 2] + G[0, 1] + G[2, 1]) - 0.25 * (G[0, 0] + G[0, 2] + G[2, 0] + G[2, 2])
            us = np.linspace(0.05, 0.95, 7)
            Bq = lambda t: np.array([(1 - t) ** 2, 2 * t * (1 - t), t ** 2])  # noqa: E731
            num, den = np.zeros(3), 0.0
            for u in us:
                for v in us:
                    bu, bv = Bq(u), Bq(v)
                    h2 = np.einsum("j,i,jik->k", bv, bu, G)
                    x2 = h2[:2] / h2[3]
                    T = np.r_[x2, bead.zc + sz * self.sec.z(bead.local(x2)[1])]
                    beta = bv[1] * bu[1]
                    rest = np.einsum("j,i,jik->k", bv, bu, Pw) - beta * Pw[1, 1]
                    num += beta * ((rest[3] + beta) * T - rest[:3])
                    den += beta ** 2
            Pw[1, 1] = np.r_[num / den, 1.0]
        return Patch(2, 2, K2, K2, Pw)

    def _curtain(self, bead, sz, p, q, nv):
        top = self.lift(bead, sz, p, q, "n")
        bot = self.flat_edge(p, q, bead.zc)
        return _ruled_split(bot, top, nv)


def _orient_all(P):
    """Consistent outward orientation of a closed patch set: flood the edge-adjacency graph so every shared edge is
    traversed in opposite directions, then flip all if the enclosed volume is negative (topological, no inside test)."""
    from collections import defaultdict, deque
    from make_geometries import reversed_patch
    key = lambda x: tuple(np.round(x, 9))  # noqa: E731

    def edges(Q):
        X = Q.euclid().reshape(3, 3, 3)
        loop = [X[0, 0], X[0, 1], X[0, 2], X[1, 2], X[2, 2], X[2, 1], X[2, 0], X[1, 0]]
        return [(key(loop[2 * e]), key(loop[2 * e + 1]), key(loop[(2 * e + 2) % 8])) for e in range(4)]

    E = [edges(Q) for Q in P]
    by = defaultdict(list)  # undirected edge -> [(patch, direction)]
    for k, es in enumerate(E):
        for a, m, b in es:
            if a != b:
                by[(min(a, b), m, max(a, b))].append((k, a < b))
    flip = [None] * len(P)
    for s0 in range(len(P)):
        if flip[s0] is not None:
            continue
        flip[s0] = False
        Qd = deque([s0])
        while Qd:
            k = Qd.popleft()
            for a, m, b in E[k]:
                if a == b:
                    continue
                d_k = (a < b) != flip[k]
                for j, d in by[(min(a, b), m, max(a, b))]:
                    if j == k or flip[j] is not None:
                        continue
                    flip[j] = (d == d_k)  # same direction -> flip j
                    Qd.append(j)
    out = [reversed_patch(Q) if f else Q for Q, f in zip(P, flip)]
    return [reversed_patch(Q) for Q in out] if _volume(out) < 0 else out


def _volume(P):
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = 0.5 * (g + 1), 0.5 * w
    B = lambda t: np.array([(1 - t) ** 2, 2 * t * (1 - t), t ** 2])  # noqa: E731
    dB = lambda t: np.array([-2 * (1 - t), 2 - 4 * t, 2 * t])  # noqa: E731
    from make_geometries import bezier_point
    V = 0.0
    e = 1e-6
    for Q in P:
        for i, u in enumerate(g):
            for j, v in enumerate(g):
                x = bezier_point(Q, u, v)
                xu = (bezier_point(Q, u + e, v) - bezier_point(Q, u - e, v)) / (2 * e)
                xv = (bezier_point(Q, u, v + e) - bezier_point(Q, u, v - e)) / (2 * e)
                V += w[i] * w[j] * x @ np.cross(xu, xv) / 3
    return V


def _quad_flat(X):
    """Bilinear (as biquadratic) patch of 4 corners X0 X1 X2 X3 (loop)."""
    X = [np.asarray(x, float) for x in X]
    Pw = np.zeros((3, 3, 4))
    for jv, v in enumerate((0, 0.5, 1)):
        for iu, u in enumerate((0, 0.5, 1)):
            Pw[jv, iu] = np.r_[(1 - u) * (1 - v) * X[0] + u * (1 - v) * X[1] + u * v * X[2] + (1 - u) * v * X[3], 1.0]
    return Patch(2, 2, K2, K2, Pw)


def _ruled_split(bot, top, n):
    """Ruled patches from curve bot (v = 0) to top (v = 1), split in n equal parts in v."""
    out = []
    for k in range(n):
        v0, v1 = k / n, (k + 1) / n
        rows = [(1 - v) * bot + v * top for v in (v0, 0.5 * (v0 + v1), v1)]
        out.append(Patch(2, 2, K2, K2, np.array(rows)))
    return out


# ------------------------------------------------------------------ window
def _square_hits(N, x0, x1, y0, y1):
    """The cell square meets the notch (open sets)."""
    dx = max(x0 - N.xc, 0.0, N.xc - x1)
    dy = max(y0 - N.yc, 0.0, N.yc - y1)
    return dx * dx + dy * dy < N.rho ** 2 or (x0 < N.xc + N.rho and x1 > N.xc - N.rho and y0 < N.yc)


def _grid_lines(B, x_c, y_c):
    """Vertical and horizontal plan lines of the cell (x_c, y_c), both layers: (A a, B s) and (A s, B a)."""
    sb = B.sec.breaks
    return np.r_[x_c + B.abreaks, x_c - sb], np.r_[y_c + sb, y_c + B.abreaks]


def admissible(g, rho, phase, refine=0, span=1):
    """Margin of the lattice phase: the slot walls off the vertical grid lines, the apex off the horizontal ones and
    every grid vertex off the circle, as a fraction of the local grid spacing (a small margin puts cut points next to
    corners: slivers). The lines x = xc, y = yc are part of the grid."""
    N = Notch(0.0, 0.0, rho)
    B = PieceBuilder(g, N, refine, extra_x=[0.0], extra_y=[0.0])
    w = g["w"]
    worst = np.inf
    n = int(np.ceil(rho / w)) + span
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            xc, yc = phase[0] + i * w, phase[1] + j * w
            if not _square_hits(N, xc - w / 2, xc + w / 2, yc - w / 2, yc + w / 2):
                continue
            for layer, bead, a_c in (("A", Bead("A", yc, g["h"]), xc), ("B", Bead("B", xc, g["h"]), yc)):
                A_, S_ = B._breaks(bead, a_c)
                X, Y = (A_, S_ + yc) if layer == "A" else (np.sort(xc - S_), A_)
                hx, hy = np.diff(X).min(), np.diff(Y).min()
                if Y[0] < N.yc:
                    worst = min(worst, min(np.abs(X - xw).min() for xw in (-rho, rho)) / hx)
                if X[0] <= 0.0 <= X[-1]:
                    worst = min(worst, np.abs(Y - rho).min() / hy)
                Xg, Yg = np.meshgrid(X, Y)
                m = Yg >= -1e-12
                if m.any():
                    worst = min(worst, (np.abs(np.hypot(Xg[m], Yg[m]) - rho) / min(hx, hy)).min())
    return worst


def best_phase(g, rho, refine=0, n=24):
    """Lattice phase with the largest admissible() margin on an n x n sweep of one cell."""
    w = g["w"]
    cand = [((i + 0.5) * w / n, (j + 0.5) * w / n) for i in range(n) for j in range(n)]
    return max(cand, key=lambda p: admissible(g, rho, p, refine))


def window(g, rho, cells=(6, 5), below=1, phase=None, refine=0, work="/tmp/rve_notch", log=print):
    """Pieces of a notch window: cells (nx, ny) of the 0/90 lattice, `below` rows under the notch centre (the slot
    runs out through the bottom cut face), notch centre at the origin, root at (0, rho). Writes the canonical A and
    B pieces, the cut pieces and the manifest for bead_window --pieces. Returns a dict (files, manifest, box, ...)."""
    os.makedirs(work, exist_ok=True)
    w, h = g["w"], g["h"]
    if phase is None:
        phase = best_phase(g, rho, refine)
    worst = admissible(g, rho, phase, refine)
    if worst < 0.05:
        raise ValueError("lattice phase %s too close to the notch features (margin %.3f)" % (phase, worst))
    N = Notch(0.0, 0.0, rho)
    B = PieceBuilder(g, N, refine, extra_x=[0.0], extra_y=[0.0])
    nx, ny = cells
    xs = phase[0] + (np.arange(nx) - (nx - 1) // 2) * w
    ys = phase[1] + (np.arange(ny) - below) * w
    tag = "r%d" % refine
    # canonical pieces (local frame): A built, B = A turned 90 deg and lifted by h (same patch order: one operator)
    canon = PieceBuilder(g, Notch(), refine).piece(Bead("A", 0.0, h), 0.0)
    R = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    canonB = []
    for P in canon:
        Pw = P.Pw @ R.T
        Pw[..., 2] += h * Pw[..., 3]
        canonB.append(Patch(P.p, P.q, P.U, P.V, Pw))
    fA, fB = os.path.join(work, "canon_%s_A.dat" % tag), os.path.join(work, "canon_%s_B.dat" % tag)
    _write_if_changed(fA, canon, "notch window canonical A")
    _write_if_changed(fB, canonB, "notch window canonical B")
    pieces = []                                       # (file, type, offset, cell centre, patches, canonical, group)
    grp = 0
    for i in range(nx):
        for j in range(ny):
            x_c, y_c = xs[i], ys[j]
            cut = _square_hits(N, x_c - w / 2, x_c + w / 2, y_c - w / 2, y_c + w / 2) or abs(x_c) < w / 2 or abs(y_c) < w / 2
            kept = False
            for t, bead, a_c in ((0, Bead("A", y_c, h), x_c), (1, Bead("B", x_c, h), y_c)):
                if not cut:
                    pieces.append(((fA, fB)[t], t, (x_c, y_c, 0.0), (x_c, y_c), (canon, canonB)[t], True, grp))
                    kept = True
                    continue
                P = B.piece(bead, a_c)
                if not P:
                    continue
                f = os.path.join(work, "cut_%s_rho%g_%s_%d_%d.dat" % (tag, rho, "AB"[t], i, j))
                _write_if_changed(f, P, "notch window cut piece")
                pieces.append((f, t, (0.0, 0.0, 0.0), (x_c, y_c), P, False, grp))
                kept = True
            grp += kept
    # canonical pieces share an operator when their faces are connected alike (free, weld, periodic, cut face:
    # the condensation keeps different dofs otherwise)
    sig = connections([[p.euclid().reshape(-1, 3) + np.asarray(o) for p in P] for _, _, o, _, P, _, _ in pieces])
    rows, src = [], {}
    nfull = ncut = 0
    for k, (f, t, o, c, P, can, gr) in enumerate(pieces):
        s_ = -1
        if can:
            key = (t, sig[k])
            s_ = src.setdefault(key, k)
            s_ = -1 if s_ == k else s_
            nfull += 1
        else:
            ncut += 1
        rows.append((f, t, o[0], o[1], o[2], c[0], c[1], s_, gr))
    nop = sum(1 for r in rows if r[7] < 0)
    man = os.path.join(work, "pieces_%s_rho%g_%dx%d.txt" % (tag, rho, nx, ny))
    with open(man, "w") as fh:
        for r in rows:
            fh.write("%s %d %.17g %.17g %.17g %.17g %.17g %d %d\n" % r)
    log("notch window rho %g: %d x %d cells, phase (%.3f, %.3f) (margin %.2f), %d canonical + %d cut pieces, %d operators" % (
        rho, nx, ny, phase[0], phase[1], worst, nfull, ncut, nop))
    return dict(manifest=man, canon=(fA, fB), phase=phase, xs=xs, ys=ys, ncut=ncut, nfull=nfull, notch=N,
                box=(xs[0] - w / 2, xs[-1] + w / 2, ys[0] - w / 2, ys[-1] + w / 2))


def connections(R, tol=1e-7):
    """Per piece, the connection of every patch (0 free, 1 shared with another piece, 2 z-periodic, 3 cut face on the
    window box), from the control points (R: per piece, per patch, (n, 3) global)."""
    allX = np.vstack([x for P in R for x in P])
    lo, hi = allX.min(0), allX.max(0)
    key = lambda X: frozenset(map(tuple, np.round(X / tol).astype(np.int64)))  # noqa: E731
    idx = {}
    for i, P in enumerate(R):
        for k, X in enumerate(P):
            idx.setdefault(key(X), []).append(i)
    Lz = hi[2] - lo[2]
    out = []
    for i, P in enumerate(R):
        c = []
        for X in P:
            a = [ax for ax in range(3) if np.ptp(X[:, ax]) < tol]
            if not a:
                c.append(0)
                continue
            a = a[0]
            v = X[0, a]
            if abs(v - lo[a]) < 10 * tol or abs(v - hi[a]) < 10 * tol:
                if a == 2:
                    c.append(2 if key(X + [0, 0, Lz if v < 0.5 * (lo[2] + hi[2]) else -Lz]) in idx else 0)
                else:
                    c.append(3)
                continue
            c.append(1 if len(idx[key(X)]) > 1 else 0)
        out.append(tuple(c))
    return out


def _write_if_changed(path, P, title):
    """Write the .dat unless the same content is there (cached operators hash the file)."""
    import io
    buf = io.StringIO()
    tmp = path + ".tmp"
    write_dat(tmp, P, title)
    with open(tmp) as f:
        new = f.read()
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == new:
                os.unlink(tmp)
                return
    os.replace(tmp, path)
    del buf
