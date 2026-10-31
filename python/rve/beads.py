"""Periodic bead cells of FDM prints as exact multipatch NURBS (.dat) for src/rve.hpp / app/rve_cell.

Bead cross-section (layer height h, filament pitch w): a rectangle w x h whose four corners are rational
quadratic arcs. Each arc runs from a lateral-bond tip (bond to the neighbour in the same layer: a flat contact of
height b_lat on the plane half-way between filaments) to an interlayer-bond tip (a flat contact of width b_int
on the plane half-way between layers). The arc leaves both tips at the notch half-angle alpha from the bond
plane, so two neighbouring beads close the pore with a notch of opening 2 alpha (alpha = 0: the crack-like cusp
of tangent quarter-ellipses); its middle control point has weight wc (sqrt(2)/2: an elliptic arc at alpha = 0).

  cell_00   all layers along x (0/0). Cell [0, L] x [0, w] x [0, h], bead axes on the four x-parallel cell
            edges, the pore channel along x at the cell centre. The y and z faces cut the beads through their
            axes (solid); the x faces carry the pore opening. 32 Bezier patches: per bead quarter two ruled
            patches on each x face and two extruded corner patches; two rectangles per y and z face.

  cell_090  cross-ply 0/90 (also +-45, 30/-60, ...: the same cell rotated). Cell [0, w]^2 x [0, 2h]: layer A
            (filaments along x, axes at y = 0, w) cut through its axes by the z faces, layer B (along y, axes at
            x = 0, w, z = h) in the middle. The interlayer bonds are the b_int x b_int squares where the flat top
            strip of an A bead crosses the flat bottom strip of a B bead; the rest of the strips is free surface
            facing the other layer's pore. 124 Bezier patches: corner surfaces split where the strips cross, bead
            quarters on the x and y faces, rectangles elsewhere.

Every shared edge is the same Bezier curve in both patches, so uniform refinement keeps the mesh conforming,
and the + faces are exact translated copies of the - faces (periodic boundary conditions in src/rve.hpp).

usage: python3 -m rve.beads 00|090 out.dat --w 0.6 --h 0.3 --bint 0.36 --blat 0.10 [--alpha 15] [--wc 0.7071]
                                       [--L 0.3] [--refine 2] [--grade 0]          (run from python/)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_geometries import Patch, bezier_point, reversed_patch, write_dat  # noqa: E402
from rve.refine import refine as conforming_refine  # noqa: E402

K2 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


# ------------------------------------------------------------------ Bezier building blocks (homogeneous rows)
def _h(x, w=1.0):
    return np.r_[np.asarray(x, float) * w, w]


def line(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([_h(a), _h((a + b) / 2), _h(b)])


def arc(P0, P1, P2, w1):
    return np.array([_h(P0), _h(P1, w1), _h(P2)])


def split(C):
    """de Casteljau halves of a rational quadratic (homogeneous, exact)."""
    C01, C12 = (C[0] + C[1]) / 2, (C[1] + C[2]) / 2
    M = (C01 + C12) / 2
    return np.array([C[0], C01, M]), np.array([M, C12, C[2]])


def translate(C, t):
    D = C.copy()
    D[..., :3] += np.asarray(t, float) * D[..., 3:4]
    return D


def ruled(C1, C2):
    """Quadratic-quadratic patch: u along the curves, v from C1 (v = 0) to C2 (v = 1), linear in v."""
    return Patch(2, 2, K2, K2, np.stack([C1, (C1 + C2) / 2, C2]))


def extrude(C, d):
    d = np.asarray(d, float)
    return Patch(2, 2, K2, K2, np.stack([C, translate(C, d / 2), translate(C, d)]))


def rect(P00, P10, P01):
    """Parallelogram P00 + u (P10 - P00) + v (P01 - P00), biquadratic, weights 1."""
    P00, P10, P01 = map(lambda a: np.asarray(a, float), (P00, P10, P01))
    return ruled(line(P00, P10), line(P01, P01 + P10 - P00))


def orient(P, outward):
    """Transpose P if its normal at the parametric centre points against outward(centre)."""
    e = 1e-6
    x = bezier_point(P, 0.5, 0.5)
    xu = bezier_point(P, 0.5 + e, 0.5) - bezier_point(P, 0.5 - e, 0.5)
    xv = bezier_point(P, 0.5, 0.5 + e) - bezier_point(P, 0.5, 0.5 - e)
    o = outward(x) if callable(outward) else np.asarray(outward, float)
    return P if np.dot(np.cross(xu, xv), o) > 0 else reversed_patch(P)


# ------------------------------------------------------------------ bead section
def corner_arc(c, sy, sz, w, h, b_int, b_lat, alpha_deg, wc):
    """Corner arc (2D, in the section plane) of the bead with axis c = (cy, cz), quadrant (sy, sz):
    lateral tip -> interlayer tip."""
    a = math.radians(alpha_deg)
    c = np.asarray(c, float)
    P0 = c + [sy * w / 2, sz * b_lat / 2]
    P2 = c + [sy * b_int / 2, sz * h / 2]
    d0 = np.array([-sy * math.sin(a), sz * math.cos(a)])
    d2 = np.array([sy * math.cos(a), -sz * math.sin(a)])
    st = np.linalg.solve(np.column_stack([d0, -d2]), P2 - P0)
    P1 = P0 + st[0] * d0
    cr = lambda a, b: a[0] * b[1] - a[1] * b[0]  # noqa: E731
    inside = (0 <= sy * (P1[0] - c[0]) <= w / 2) and (0 <= sz * (P1[1] - c[1]) <= h / 2)
    if st[0] <= 0 or st[1] <= 0 or not inside or cr(P2 - P0, P1 - P0) * cr(P2 - P0, c - P0) >= 0:
        raise ValueError("bead corner: notch angle too large for these bond widths (arc not convex)")
    return P0, P1, P2


def section_porosity(w, h, b_int, b_lat, alpha_deg, wc, n=4000):
    """Pore area fraction of the 0/0 section (four corner arcs of one pore), by Green's theorem."""
    area = 0.0
    for cy, cz in ((0, 0), (w, 0), (w, h), (0, h)):
        sy, sz = (1 if cy == 0 else -1), (1 if cz == 0 else -1)
        P0, P1, P2 = corner_arc((cy, cz), sy, sz, w, h, b_int, b_lat, alpha_deg, wc)
        t = np.linspace(0, 1, n)[:, None]
        B = np.hstack([(1 - t) ** 2, 2 * t * (1 - t) * wc, t ** 2])
        X = (B[:, 0:1] * P0 + B[:, 1:2] * P1 + B[:, 2:3] * P2) / B.sum(1, keepdims=True)
        # tips are shared: (0,0) fwd, (0,h) back, (w,h) fwd, (w,0) back closes the pore, i.e. sign sy sz
        area += sy * sz * 0.5 * np.sum(X[:-1, 0] * X[1:, 1] - X[1:, 0] * X[:-1, 1])
    return abs(area) / (w * h)


def tip_marker(lines, tol=1e-9):
    """graded(x): x on one of the tip lines [(point, axis unit vector)] (bond tips: singular notch roots)."""
    lines = [(np.asarray(p, float), np.asarray(a, float)) for p, a in lines]

    def f(x):
        for p, a in lines:
            d = np.asarray(x) - p
            if np.linalg.norm(d - np.dot(d, a) * a) < tol * (1 + np.linalg.norm(p)):
                return True
        return False
    return f


# ------------------------------------------------------------------ 0/0 cell
def cell_00(w, h, b_int, b_lat, alpha=15.0, wc=math.sqrt(0.5), L=None, refine=2, grade=0):
    L = h if L is None else L
    out = []
    E3 = lambda x, yz: np.array([x, yz[0], yz[1]])  # noqa: E731  section (y, z) at x
    for cy, cz in ((0, 0), (w, 0), (0, h), (w, h)):
        sy, sz = (1 if cy == 0 else -1), (1 if cz == 0 else -1)
        P0, P1, P2 = corner_arc((cy, cz), sy, sz, w, h, b_int, b_lat, alpha, wc)
        C, Qlat, Qtop = np.array([cy, cz], float), np.array([cy + sy * w / 2, cz], float), np.array([cy, cz + sz * h / 2], float)
        A, B = split(arc(E3(0, P0), E3(0, P1), E3(0, P2), wc))
        for x, s in ((0.0, -1.0), (L, 1.0)):
            t = [x, 0, 0]
            out.append(orient(ruled(translate(A, t), line(E3(x, Qlat), E3(x, C))), [s, 0, 0]))
            out.append(orient(ruled(translate(B, t), line(E3(x, C), E3(x, Qtop))), [s, 0, 0]))
        axis = np.array([0, cy, cz], float)
        away = lambda x, axis=axis: (x - axis) * [0, 1, 1]  # noqa: E731  corner faces away from the bead axis
        out.append(orient(extrude(A, [L, 0, 0]), away))
        out.append(orient(extrude(B, [L, 0, 0]), away))
    for z, s in ((0.0, -1.0), (h, 1.0)):
        for y0, y1 in ((0, w / 2), (w / 2, w)):
            out.append(orient(rect([0, y0, z], [0, y1, z], [L, y0, z]), [0, 0, s]))
    for y, s in ((0.0, -1.0), (w, 1.0)):
        for z0, z1 in ((0, h / 2), (h / 2, h)):
            out.append(orient(rect([0, y, z0], [0, y, z1], [L, y, z0]), [0, s, 0]))
    ex = [1.0, 0, 0]
    tips = [((0, w / 2, z), ex) for z in (b_lat / 2, h - b_lat / 2)] + [((0, y, h / 2), ex) for y in (b_int / 2, w - b_int / 2)]
    return conforming_refine(out, refine, tip_marker(tips), grade)


# ------------------------------------------------------------------ 0/90 cell
class Bead:
    """A bead along `axis` ('x' or 'y') with its axis at section coordinates c = (s, z); s is the other
    horizontal coordinate. pos(s, z, a) maps section coordinates and the axial coordinate a to 3D."""

    def __init__(self, axis, c, geo):
        self.axis, self.c, self.g = axis, np.asarray(c, float), geo

    def pos(self, s, z, a):
        return np.array([a, s, z]) if self.axis == "x" else np.array([s, a, z])

    def adir(self):
        return np.array([1.0, 0, 0]) if self.axis == "x" else np.array([0, 1.0, 0])

    def arc(self, ss, sz, a=0.0):
        w, h, b_int, b_lat, alpha, wc = self.g
        P0, P1, P2 = corner_arc(self.c, ss, sz, w, h, b_int, b_lat, alpha, wc)
        return split(arc(self.pos(*P0, a), self.pos(*P1, a), self.pos(*P2, a), wc))

    def quarter(self, ss, sz, a, outward):
        """The bead quarter (ss, sz) of the section plane at axial coordinate a: two ruled patches."""
        w, h = self.g[0], self.g[1]
        A, B = self.arc(ss, sz, a)
        C, Qlat, Qtop = self.c, self.c + [ss * w / 2, 0], self.c + [0, sz * h / 2]
        return [orient(ruled(A, line(self.pos(*Qlat, a), self.pos(*C, a))), outward),
                orient(ruled(B, line(self.pos(*C, a), self.pos(*Qtop, a))), outward)]

    def corner(self, ss, sz, pieces):
        """Corner surface (ss, sz) extruded along the axis over the pieces [(a0, a1), ...]."""
        def away(x):
            d = x - self.pos(self.c[0], self.c[1], 0.0)
            return d - np.dot(d, self.adir()) * self.adir()
        out = []
        for a0, a1 in pieces:
            for half in self.arc(ss, sz, a0):
                out.append(orient(extrude(half, (a1 - a0) * self.adir()), away))
        return out


def cell_090(w, h, b_int, b_lat, alpha=15.0, wc=math.sqrt(0.5), refine=2, grade=0, b_lat_b=None):
    """b_lat_b: lateral bond of the B layer (default b_lat), to debond one in-plane family at a time."""
    g = (w, h, b_int, b_lat, alpha, wc)
    gb = (w, h, b_int, b_lat if b_lat_b is None else b_lat_b, alpha, wc)
    b = b_int / 2
    cuts = [(0.0, b), (b, w - b), (w - b, w)]  # axial pieces of a corner surface (strip crossings)
    out = []
    A0 = [Bead("x", (0.0, 0.0), g), Bead("x", (w, 0.0), g)]          # layer A, cut by z = 0 (upper halves)
    A2 = [Bead("x", (0.0, 2 * h), g), Bead("x", (w, 2 * h), g)]      # layer A, cut by z = 2h (lower halves)
    B1 = [Bead("y", (0.0, h), gb), Bead("y", (w, h), gb)]              # layer B, axes on x = 0, w
    sgn = lambda c: 1 if c == 0 else -1  # noqa: E731  quadrant pointing into the cell
    for beads, sz in ((A0, 1), (A2, -1)):
        for bd in beads:
            out += bd.corner(sgn(bd.c[0]), sz, cuts)
            for xf, so in ((0.0, -1.0), (w, 1.0)):
                out += bd.quarter(sgn(bd.c[0]), sz, xf, [so, 0, 0])
    for bd in B1:
        for sz in (-1, 1):
            out += bd.corner(sgn(bd.c[0]), sz, cuts)
            for yf, so in ((0.0, -1.0), (w, 1.0)):
                out += bd.quarter(sgn(bd.c[0]), sz, yf, [0, so, 0])
    # free parts of the strips on the two interface planes
    for zi, zA in ((h / 2, 1.0), (3 * h / 2, -1.0)):   # zA: normal of the A strips there (B strips: opposite)
        for y0, y1 in ((0, b), (w - b, w)):
            out.append(orient(rect([b, y0, zi], [w - b, y0, zi], [b, y1, zi]), [0, 0, zA]))
        for x0, x1 in ((0, b), (w - b, w)):
            out.append(orient(rect([x0, b, zi], [x1, b, zi], [x0, w - b, zi]), [0, 0, -zA]))
    # x faces: the B section (rectangle through the B axes), split where the strips end
    for xf, so in ((0.0, -1.0), (w, 1.0)):
        for y0, y1 in cuts:
            for z0, z1 in ((h / 2, h), (h, 3 * h / 2)):   # split at the B axes (meets the y-face quarters)
                out.append(orient(rect([xf, y0, z0], [xf, y1, z0], [xf, y0, z1]), [so, 0, 0]))
    # y faces: the A sections (half rectangles through the A axes), split where the strips end
    for yf, so in ((0.0, -1.0), (w, 1.0)):
        for z0, z1 in ((0.0, h / 2), (3 * h / 2, 2 * h)):
            for x0, x1 in cuts:
                out.append(orient(rect([x0, yf, z0], [x1, yf, z0], [x0, yf, z1]), [0, so, 0]))
    # z faces: through the A axes, split to meet the y-face pieces (x) and the bead quarters (y = w/2)
    for zf, so in ((0.0, -1.0), (2 * h, 1.0)):
        for x0, x1 in cuts:
            for y0, y1 in ((0, w / 2), (w / 2, w)):
                out.append(orient(rect([x0, y0, zf], [x1, y0, zf], [x0, y1, zf]), [0, 0, so]))
    tips = []
    for bd, szs in [(x, (1,)) for x in A0] + [(x, (-1,)) for x in A2] + [(x, (-1, 1)) for x in B1]:
        for sz in szs:
            P0, _, P2 = corner_arc(bd.c, sgn(bd.c[0]), sz, *bd.g)
            tips += [(bd.pos(*P0, 0.0), bd.adir()), (bd.pos(*P2, 0.0), bd.adir())]
    return conforming_refine(out, refine, tip_marker(tips), grade)


# ------------------------------------------------------------------ single bead (bead-boundary cells)
def bead_00(w, h, b_int, b_lat, alpha=15.0, wc=math.sqrt(0.5), L=None, refine=2, grade=0, debonds=()):
    """One closed bead segment, axis along x through the origin, x in [0, L]: the 0/0 cell bounded by the welds
    instead of by planes through the bead axes. Its bounding box [0, L] x [-w/2, w/2] x [-h/2, h/2] carries
    exactly the connection faces: the in-layer weld strips (y = +-w/2, height b_lat), the interlayer weld strips
    (z = +-h/2, width b_int) and the two end caps (x = 0, L); everything else is free surface. The + faces are
    translated copies of the - faces, so src/rve.hpp's periodic conditions apply unchanged (perfect welds), and
    a weld law t = K [[u]] couples exactly these pairs. 32 Bezier patches before refinement."""
    L = h if L is None else L
    g = (w, h, b_int, b_lat, alpha, wc)
    bd = Bead("x", (0.0, 0.0), g)
    out = []
    for ss in (-1, 1):
        for sz in (-1, 1):
            out += bd.corner(ss, sz, [(0.0, L)])
            for a, so in ((0.0, -1.0), (L, 1.0)):
                out += bd.quarter(ss, sz, a, [so, 0, 0])
            out.append(orient(rect([0, ss * w / 2, 0], [L, ss * w / 2, 0], [0, ss * w / 2, sz * b_lat / 2]), [0, ss, 0]))
            out.append(orient(rect([0, 0, sz * h / 2], [L, 0, sz * h / 2], [0, ss * b_int / 2, sz * h / 2]), [0, 0, sz]))
    ex = [1.0, 0, 0]
    tips = [((0, ss * w / 2, sz * b_lat / 2), ex) for ss in (-1, 1) for sz in (-1, 1)]
    tips += [((0, ss * b_int / 2, sz * h / 2), ex) for ss in (-1, 1) for sz in (-1, 1)]
    fronts = [(1, 2, s * (1 - f) * b_lat / 2) for f in debonds for s in (-1, 1)]
    fronts += [(2, 1, s * (1 - f) * b_int / 2) for f in debonds for s in (-1, 1)]
    return _with_fronts(out, refine, tip_marker(tips), grade, fronts)


def _with_fronts(P, refine, graded, grade, fronts):
    """Conforming refinement, plus C0 cuts at the debond fronts (separate weld patches per debond depth).

    debonds= of bead_00 / bead_090 are depths as fractions of each weld's half width, the same for every weld: a
    front's knot travels through the corner surfaces into the neighbouring weld at the same parameter, so equal
    fractions give one set of cuts (and keep the 0/90 square symmetric under its 90 deg turn)."""
    from rve.refine import split_c0
    R = conforming_refine(P, refine, graded, grade, fronts=fronts)
    return split_c0(R) if fronts else R


def bead_090(w, h, b_int, b_lat, alpha=15.0, wc=math.sqrt(0.5), refine=2, grade=0, debonds=()):
    """The two beads of a 0/90 bead-boundary cell, (A, B). A: axis x through the origin, x in [-w/2, w/2] (centred
    on the B filament it crosses); B = A turned 90 deg about z and lifted by h (axis y at z = h). The cell box is
    [-w/2, w/2]^2 x [-h/2, 3h/2]. A's top strip crosses B's bottom strip in the interlayer weld square
    [-b/2, b/2]^2 at z = h/2 (an internal interface between the two regions), and A's bottom square meets the
    next B's top square through the z faces (periodic). The strips and corners are split at the square's edges so
    the squares are separate patches (in quarters, split at x = 0 and y = 0, so that A's quarters are B's rotated) and
    graded toward all four square edges (A's and B's bond tips); the rest of the strips is free surface. Both lists are conforming and B's
    patches are A's, rotated, in the same order (one cached operator serves both)."""
    g = (w, h, b_int, b_lat, alpha, wc)
    b = b_int / 2
    pieces = [(-w / 2, -b), (-b, 0.0), (0.0, b), (b, w / 2)]   # x = 0 too: A's square quarters map onto B's
    bd = Bead("x", (0.0, 0.0), g)
    A = []
    for ss in (-1, 1):
        for sz in (-1, 1):
            A += bd.corner(ss, sz, pieces)
            for a, so in ((-w / 2, -1.0), (w / 2, 1.0)):
                A += bd.quarter(ss, sz, a, [so, 0, 0])
            for x0, x1 in pieces:
                A.append(orient(rect([x0, ss * w / 2, 0], [x1, ss * w / 2, 0], [x0, ss * w / 2, sz * b_lat / 2]), [0, ss, 0]))
                A.append(orient(rect([x0, 0, sz * h / 2], [x1, 0, sz * h / 2], [x0, ss * b, sz * h / 2]), [0, 0, sz]))
    ex = [1.0, 0, 0]
    tips = [((0, ss * w / 2, sz * b_lat / 2), ex) for ss in (-1, 1) for sz in (-1, 1)]
    tips += [((0, ss * b, sz * h / 2), ex) for ss in (-1, 1) for sz in (-1, 1)]
    tips += [((ss * b, 0, sz * h / 2), [0, 1.0, 0]) for ss in (-1, 1) for sz in (-1, 1)]  # B's bond edges on the square
    b = b_int / 2
    fronts = [(1, 2, s * (1 - f) * b_lat / 2) for f in debonds for s in (-1, 1)]   # A's lateral welds (B's: turned)
    fronts += [(2, k, s * (1 - f) * b) for f in debonds for s in (-1, 1) for k in (0, 1)]   # the weld squares
    A = _with_fronts(A, refine, tip_marker(tips), grade, fronts)
    R = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    B = []
    for P in A:
        Pw = P.Pw @ R.T
        Pw[..., 2] += h * Pw[..., 3]
        B.append(type(P)(P.p, P.q, P.U, P.V, Pw))
    return A, B


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["00", "090", "bead00", "bead090"])
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--bint", type=float, default=0.36)
    ap.add_argument("--blat", type=float, default=0.10)
    ap.add_argument("--blat-b", type=float, default=None, help="0/90: lateral bond of the B layer (default --blat)")
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--wc", type=float, default=math.sqrt(0.5))
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--grade", type=int, default=0, help="geometric knots toward the bond tips (ratio 1/2)")
    a = ap.parse_args(argv)
    if a.kind == "00":
        P = cell_00(a.w, a.h, a.bint, a.blat, a.alpha, a.wc, a.L, a.refine, a.grade)
    elif a.kind == "bead090":   # writes <out>_A.dat and <out>_B.dat
        PA, PB = bead_090(a.w, a.h, a.bint, a.blat, a.alpha, a.wc, a.refine, a.grade)
        base = a.out[:-4] if a.out.endswith(".dat") else a.out
        for tag, P in (("A", PA), ("B", PB)):
            write_dat("%s_%s.dat" % (base, tag), P, "0/90 bead %s w %g h %g bint %g blat %g alpha %g wc %g" %
                      (tag, a.w, a.h, a.bint, a.blat, a.alpha, a.wc))
        print("wrote %s_A.dat, %s_B.dat: %d patches each" % (base, base, len(PA)))
        return
    elif a.kind == "bead00":
        P = bead_00(a.w, a.h, a.bint, a.blat, a.alpha, a.wc, a.L, a.refine, a.grade)
    else:
        P = cell_090(a.w, a.h, a.bint, a.blat, a.alpha, a.wc, a.refine, a.grade, a.blat_b)
    write_dat(a.out, P, "bead cell %s w %g h %g bint %g blat %g alpha %g wc %g" %
              (a.kind, a.w, a.h, a.bint, a.blat, a.alpha, a.wc))
    print("wrote %s: %d patches, section porosity %.4f%%" %
          (a.out, len(P), 100 * section_porosity(a.w, a.h, a.bint, a.blat, a.alpha, a.wc)))


if __name__ == "__main__":
    main()
