"""Exact multipatch quarter of an edge-notched plate, built from a designed block layout (no CAD import, no fitting).

Layout (top view, root at b = (WN/2, 0), notch centre C = (WN/2 + rho, 0)):

        T1 | T2 | T3          rows above y = Y1: one block per column, up to the grip y = L/2
        M1 | M2 | S2          columns split at x = K_x (left of the fan) and x = C'_x (radial line through c)
        L1 | F2 | S1          S1: the shoulder over the flank c -> d
           F1 /               F1, F2: O-grid fan around the root arc, ruled from the arc (rational, exact) to the
                              straight polyline B' -> K -> C' at radius ~R; rays are radial from C

Every block is a ruled (or bilinear) biquadratic patch, so the surface is exact: arcs rational, lines straight. The
surface (top z = T, bottom z = 0 and the walls extruded from the outline) is then refined by knot insertion only:
    h      element size away from the root
    n_arc  spans along each half of the root arc
    h0, g  first size and growth of the radial grading away from the root and along the flank
    h_far  largest element at the grip (default 4 h): the rows above y = Y1 grow from h to it
    hz0    first size of the through-thickness grading towards both free faces (at most T / nz further in)
All of it takes well under a second; the geometry never changes with the mesh.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grade_dat import insert, knot_range, link, mid_line  # noqa: E402
from make_geometries import Patch, oriented, write_dat  # noqa: E402

K3 = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def _arc(C, r, t0, t1):
    """Rational quadratic arc (homogeneous rows x w, y w, w) of radius r about C from angle t0 to t1 (|t1-t0| < pi)."""
    w = math.cos(0.5 * (t1 - t0))
    tm = 0.5 * (t0 + t1)
    P = np.array([[C[0] + r * math.cos(t0), C[1] + r * math.sin(t0)],
                  [C[0] + r / w * math.cos(tm), C[1] + r / w * math.sin(tm)],
                  [C[0] + r * math.cos(t1), C[1] + r * math.sin(t1)]])
    return P, np.array([1.0, w, 1.0])


def _seg(P0, P2, w=1.0, lam=0.5):
    """Straight segment P0 -> P2 with weights (1, w, 1) whose u = 1/2 point is P0 + lam (P2 - P0). Two blocks that
    share a straight side must use the same (w, lam), or equal knots land on different points (non-conforming)."""
    P0, P2 = np.asarray(P0, float), np.asarray(P2, float)
    M = P0 + lam * (P2 - P0)
    P1 = ((2 + 2 * w) * M - P0 - P2) / (2 * w)
    return np.array([P0, P1, P2]), np.array([1.0, w, 1.0])


def _ruled(a, b):
    """2D block ruled between curves a and b (each (points (3, 2), weights (3,))): Pw (3, 3, 3) as (v, u, [xw, yw, w])."""
    (A, wa), (B, wb) = a, b
    ha = np.c_[A * wa[:, None], wa]
    hb = np.c_[B * wb[:, None], wb]
    return np.stack([ha, 0.5 * (ha + hb), hb], axis=0)


def _quad(p00, p10, p11, p01, wl=(1.0, 0.5)):
    """Block between the bottom p00 -> p10 and the top p01 -> p11, both parametrised with (w, lam) = wl."""
    return _ruled(_seg(p00, p10, *wl), _seg(p01, p11, *wl))


def layout(s, R=None, Y1=None):
    """The 2D blocks {name: Pw (3, 3, 3)} and the key points of the edge-notch quarter of specimen s."""
    rho, al = s.rho, math.radians(s.alpha_deg)
    xc = s.WN / 2 + rho
    C = (xc, 0.0)
    t0, t1 = math.pi, 0.5 * math.pi + 0.5 * al
    tm = 0.5 * (t0 + t1)
    if R is None:
        R = min(rho + max(2 * rho, 0.3 * s.WN / 2), xc - 0.25 * s.WN / 2)
    Kx = xc - R
    K = (Kx, -R * math.tan(tm))                                # on the ray at tm (tan < 0 in the 2nd quadrant)
    Cp = (xc + R * math.cos(t1), R * math.sin(t1))             # radial through c, the end of the root arc
    c = (xc + rho * math.cos(t1), rho * math.sin(t1))
    d = (s.WG / 2, c[1] + (s.WG / 2 - c[0]) * math.tan(0.5 * al))
    dp = (s.WG / 2, d[1] + (Cp[1] - c[1]))
    if Y1 is None:
        Y1 = max(dp[1], Cp[1]) + s.WN / 2
    H = s.L / 2
    edge = abs(c[0] - s.WG / 2) < 1e-9 and abs(Cp[0] - s.WG / 2) < 1e-9   # the arc ends on the side edge (semicircle)
    if not (0 < Kx < Cp[0] and (edge or Cp[0] < s.WG / 2) and Y1 < H):
        raise ValueError(f"layout does not fit: K_x {Kx:.3g}, C'_x {Cp[0]:.3g}, Y1 {Y1:.3g}")
    a1, w1 = _arc(C, rho, t0, tm)
    a2, w2 = _arc(C, rho, tm, t1)
    m1 = (Kx, -R * math.tan(0.5 * (t0 + tm)))                  # where the mid rays meet the outer polyline
    l1 = m1[1] / K[1]
    l2 = _ray_t(C, 0.5 * (tm + t1), K, Cp)
    p1, p2 = (w1[1], l1), (w2[1], l2)                          # parametrisation of the fan's outer sides, carried on
    tr = lambda a, b: np.transpose(a, (1, 0, 2)) if b else a  # noqa: E731
    B = {
        "F1": _ruled((a1, w1), _seg((Kx, 0.0), K, *p1)),
        "F2": _ruled((a2, w2), _seg(K, Cp, *p2)),
        "L1": tr(_ruled(_seg((0, 0), (0, K[1]), *p1), _seg((Kx, 0), K, *p1)), True),
        "M1": _quad((0, K[1]), K, (Kx, Y1), (0, Y1)),
        "M2": _quad(K, Cp, (Cp[0], Y1), (Kx, Y1), p2),
        "S1": _quad(c, d, dp, Cp),
        "S2": _quad(Cp, dp, (s.WG / 2, Y1), (Cp[0], Y1)),
        "T1": _quad((0, Y1), (Kx, Y1), (Kx, H), (0, H)),
        "T2": _quad((Kx, Y1), (Cp[0], Y1), (Cp[0], H), (Kx, H), p2),
        "T3": _quad((Cp[0], Y1), (s.WG / 2, Y1), (s.WG / 2, H), (Cp[0], H)),
    }
    if edge:                                                   # no shoulder column: F2, M2, T2 reach the side edge
        for n in ("S1", "S2", "T3"):
            del B[n]
    return B, dict(R=R, K=K, Cp=Cp, c=c, d=d, dp=dp, Y1=Y1, C=C, b=(s.WN / 2, 0.0), sweep=t0 - t1,
                   radial=("F2.top", np.subtract(Cp, c)),
                   lig=("L1.top", (-1.0, 0.0), (Kx, 0.0), Kx),
                   flank=None if edge else ("S1.top", np.subtract(d, c), c, float(np.hypot(d[0] - c[0], d[1] - c[1]))))


def layout_hole(s, R=None, Y1=None):
    """The 2D blocks of the central-hole quarter: O-grid fan around the hole quarter (root at (rho, 0), centre at the
    origin), the ligament block out to the free edge x = W/2, and rows up to the grip.

        T1 | T2
        M1 | M2
        F2 /
         F1 | L1
    """
    rho, W2, H = s.rho, s.W / 2, s.L / 2
    lig = W2 - rho
    if R is None:
        R = rho + min(max(2 * rho, 0.3 * lig), 0.75 * lig)
    if Y1 is None:
        Y1 = R + lig
    if not (rho < R < W2 and Y1 < H):
        raise ValueError(f"layout does not fit: R {R:.3g}, Y1 {Y1:.3g}")
    C, K, Cp = (0.0, 0.0), (R, R), (0.0, R)
    t0, tm, t1 = 0.0, 0.25 * math.pi, 0.5 * math.pi
    a1, w1 = _arc(C, rho, t0, tm)
    a2, w2 = _arc(C, rho, tm, t1)
    l1 = math.tan(0.125 * math.pi)                             # the mid rays (pi/8, 3 pi/8) on the outer square
    l2 = 1.0 - 1.0 / math.tan(0.375 * math.pi)
    p1, p2 = (w1[1], l1), (w2[1], l2)
    B = {
        "F1": _ruled((a1, w1), _seg((R, 0.0), K, *p1)),
        "F2": _ruled((a2, w2), _seg(K, Cp, *p2)),
        "L1": _ruled(_seg((R, 0.0), K, *p1), _seg((W2, 0.0), (W2, R), *p1)),
        "M1": _ruled(_seg(K, Cp, *p2), _seg((R, Y1), (0.0, Y1), *p2)),
        "M2": _quad(K, (W2, R), (W2, Y1), (R, Y1)),
        "T1": _quad((0.0, Y1), (R, Y1), (R, H), (0.0, H), (w2[1], 1.0 - l2)),
        "T2": _quad((R, Y1), (W2, Y1), (W2, H), (R, H)),
    }
    return B, dict(R=R, K=K, Cp=Cp, Y1=Y1, C=C, b=(rho, 0.0), sweep=t1 - t0,
                   radial=("F2.top", (0.0, 1.0)),
                   lig=("L1.top", (1.0, 0.0), (R, 0.0), W2 - R), flank=None)


def _ray_t(C, t, P, Q):
    """Parameter along P -> Q where the ray from C at angle t crosses it."""
    P, Q, C = map(np.asarray, (P, Q, C))
    dvec = np.array([math.cos(t), math.sin(t)])
    A = np.c_[Q - P, -dvec]
    return float(np.linalg.solve(A, C - P)[0])


def _lift(Pw2, z):
    """2D homogeneous block -> 3D homogeneous patch at height z."""
    return np.concatenate([Pw2[..., :2], (z * Pw2[..., 2])[..., None], Pw2[..., 2:3]], axis=-1)


def surface(s, **kw):
    """Closed boundary of the quarter as single-span patches (top, bottom, walls), outward oriented."""
    B, info = (layout_hole if s.shape == "hole" else layout)(s, **kw)
    out, names = [], []
    for n, Pw in B.items():
        for z, nz, tag in ((s.T, 1.0, "top"), (0.0, -1.0, "bot")):
            out.append(oriented(Patch(2, 2, K3, K3, _lift(Pw, z)), np.array([0.0, 0.0, nz])))
            names.append(f"{n}.{tag}")
    # walls: block sides not shared with another block
    sides = []
    for n, Pw in B.items():
        X = Pw[..., :2] / Pw[..., 2:3]
        for k, row in enumerate((Pw[0], Pw[-1], Pw[:, 0], Pw[:, -1])):
            sides.append((n, k, row, X))
    key = lambda row: tuple(sorted([tuple(np.round(row[0, :2] / row[0, 2], 9)), tuple(np.round(row[-1, :2] / row[-1, 2], 9))]))  # noqa: E731
    count = {}
    for n, k, row, X in sides:
        count[key(row)] = count.get(key(row), 0) + 1
    for n, k, row, X in sides:
        if count[key(row)] > 1:
            continue
        Pw = np.stack([_lift(row[None], z)[0] for z in (0.0, 0.5 * s.T, s.T)], axis=0)
        e = row[:, :2] / row[:, 2:3]
        mid, cen = e[1] if row[1, 2] == 1 else 0.5 * (e[0] + e[2]), X.reshape(-1, 2).mean(0)
        t = e[2] - e[0]
        nrm = np.array([t[1], -t[0], 0.0])
        if np.dot(nrm[:2], mid - cen) < 0:
            nrm = -nrm
        out.append(oriented(Patch(2, 2, K3, K3, Pw), nrm))
        names.append(f"{n}.wall{k}")
    return out, names, info


def _class_insert(P, cls, want):
    for k, Q in enumerate(P):
        for d in (0, 1):
            root, f = cls.find((k, d))
            if root not in want:
                continue
            a, b, _ = knot_range(Q, d)
            S = np.unique(np.round(np.array(want[root]), 12))
            insert(Q, d, [a + (b - a) * (1 - x if f else x) for x in S])


def graded(L, h0, g, h):
    """Knots in (0, 1) of a 1D mesh of length L: sizes h0, h0 g, ... from s = 0 while below h, then uniform <= h."""
    x, size, out = 0.0, h0, []
    while size < h and x + size < L - 0.5 * size:
        x += size
        out.append(x)
        size *= g
    n = max(1, math.ceil((L - x) / min(h, max(size, h0)) - 1e-9))
    out += [x + (L - x) * i / n for i in range(1, n)]
    return [v / L for v in out]


def _direction(P, k, along, root_pt):
    """(class key, flip) of the direction of patch k whose iso-lines run along the vector `along`, and whether its
    parameter starts at the end nearer root_pt."""
    X = P[k].euclid()
    du, dv = X[0, -1] - X[0, 0], X[-1, 0] - X[0, 0]
    d = 0 if abs(np.dot(du, along)) * np.linalg.norm(dv) > abs(np.dot(dv, along)) * np.linalg.norm(du) else 1
    start = X[0, 0]
    end = X[0, -1] if d == 0 else X[-1, 0]
    return d, np.linalg.norm(start[:2] - root_pt) < np.linalg.norm(end[:2] - root_pt)


def _coarsen(knots, near, c):
    """Level c of a class's knots, a subset of level c - 1. Away from the root (classes with no end at it) every
    2^c-th knot is kept, so the far field coarsens fast; a class graded from the root loses its c knots nearest the
    root end(s), so the root elements grow one ring per level (sizes h0 -> h0 (1 + g) -> ...) and the rest stays."""
    k = sorted(knots)
    if near is None:
        return k[2 ** c - 1::2 ** c]
    if 0 in near:
        k = k[c:]
    if 1 in near:
        k = k[:len(k) - c] if c < len(k) else []
    return k


def build(s, h=1.5, n_arc=None, h0=None, g=1.4, nz=12, hz0=None, coarsen=0, h_far=None, **kw):
    """Refined exact quarter: (patches, names, info). Sizes in mm; h0 defaults to rho / 8, n_arc (spans on each half
    of the root arc) to the half-arc length / h0, so the root elements are about square in the plane. Geometry never changes:
    every refinement is a knot insertion, the same knots across every shared edge (conforming). coarsen = c gives the
    nested coarser levels of a convergence study (the default is the finest): see _coarsen."""
    h0 = h0 or s.rho / 8
    hz0 = s.T / 60 if hz0 is None else hz0             # 0 = uniform nz spans through the thickness
    if n_arc is None:
        n_arc = None
    P, names, info = surface(s, **kw)
    if n_arc is None:
        n_arc = max(2, math.ceil(0.5 * s.rho * abs(info["sweep"]) / h0 - 1e-9))
    cls, _ = link(P)
    want = {}
    for k, Q in enumerate(P):                                        # 1. uniform, about h
        for d in (0, 1):
            root, _ = cls.find((k, d))
            L = mid_line(Q, d, n=60)[1][-1]
            n = max(1, math.ceil(L / h - 1e-9))
            if len(want.get(root, [])) < n - 1:
                want[root] = [i / n for i in range(1, n)]
    C, b = np.asarray(info["C"]), np.asarray(info["b"])
    near = {}                                # graded classes: which end(s) of the class lie at the root

    def set_dir(name, along, root_pt, knots):
        k = names.index(name)
        d, at_start = _direction(P, k, np.r_[along, 0.0], root_pt)
        root, f = cls.find((k, d))
        want[root] = [x if at_start != f else 1 - x for x in knots]
        near[root] = (0,) if at_start != f else (1,)          # class end(s) at the point of interest

    for half in ("F1.wall0", "F2.wall0"):        # each half of the root arc is its own knot class (F1 links to L1,
        k = names.index(half)                     # F2 to M2), so both need the arc spans
        X = P[k].euclid()
        zd = 0 if abs(X[0, -1, 2] - X[0, 0, 2]) > 0.5 * s.T else 1
        if hz0:                                   # graded towards both free faces, sizes hz0 g^i, at most T / nz
            half = graded(0.5 * s.T, hz0, g, s.T / nz)
            zk = sorted([0.5 * x for x in half] + [0.5] + [1 - 0.5 * x for x in half])
        else:
            zk = [i / nz for i in range(1, nz)]
        want[cls.find((k, zd))[0]] = zk                                                     # 2. through the thickness
        near[cls.find((k, zd))[0]] = (0, 1)
        ra, fa = cls.find((k, 1 - zd))
        want[ra] = [i / n_arc for i in range(1, n_arc)]                                      # 3. along the root arc
        e0 = X[0, 0, :2]                                  # does this half start at the root b?
        e1 = X[0, -1, :2] if zd == 1 else X[-1, 0, :2]
        if half == "F1.wall0":
            near[ra] = (0,) if (np.linalg.norm(e0 - b) < np.linalg.norm(e1 - b)) != fa else (1,)
    radial = graded(info["R"] - s.rho, h0, g, h)
    set_dir(info["radial"][0], np.asarray(info["radial"][1]), C, radial)                     # 4. radial, from the root
    last = (info["R"] - s.rho) * (1 - (radial[-1] if radial else 0.0))     # the fan's outermost radial size: the
    name, along, start, L = info["lig"]                                    # ligament block continues from it, so
    set_dir(name, np.asarray(along), np.asarray(start), graded(L, min(last * g, h), g, h))  # no size jump at the seam
    if info["flank"]:
        name, along, start, L = info["flank"]
        set_dir(name, np.asarray(along), np.asarray(start), graded(L, h0, g, h))           # 5. along the flank
    Y1, H = info["Y1"], s.L / 2                                                            # 6. grip rows: from h
    set_dir("T1.top", np.array([0.0, 1.0]), np.array([0.0, Y1]),                         # at y = Y1, growing to
            graded(H - Y1, h, g, h_far or 4 * h))                                          # h_far at the grip
    near[cls.find((names.index("T1.top"), _direction(P, names.index("T1.top"), np.array([0.0, 1.0, 0.0]),
                                                         np.array([0.0, Y1]))[0]))[0]] = None
    if coarsen:
        want = {r: _coarsen(v, near.get(r), coarsen) for r, v in want.items()}
    _class_insert(P, cls, want)
    info.update(coarsen=coarsen, h=h, n_arc=n_arc, h0=h0, g=g, nz=nz, hz0=hz0, patches=len(P),
                ctrl=int(sum(Q.Pw.shape[0] * Q.Pw.shape[1] for Q in P)))
    return P, names, info


def write(s, path, **kw):
    P, names, info = build(s, **kw)
    write_dat(path, P, os.path.basename(path))
    return info
