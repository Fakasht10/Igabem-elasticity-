"""Cracked-body geometries for the XIGABEM reproduction (Rocha, Trevelyan & Leonel, CMAME 423 (2024) 116872).

A crack is two coincident open surfaces: the upper face (outward normal -m, solid on the +m side) and the lower
face (outward normal +m). Each face has its own patches and its own displacement dofs (src/crack/crack_mesh.hpp
never merges them). The geometry is one .dat (outer patches, then the upper face, then the lower face, patch k of
the lower face mirroring patch k of the upper face) plus a JSON sidecar <name>.crack.json with the roles:

  outer [0, n0)    plus [n0, n1)    minus [n1, n2)
  crown            upper-face patches that carry the crack-front enrichment; the front is their v = 1 edge
  plane            point and unit normal m of the (flat) crack plane
  R, a, b          crack size (penny: R; ellipse: semi-axes a >= b, in the in-plane axes e1, e3)

penny (paper Sec. 4.1, Fig. 8): radius R, crack plane through the origin with normal m = Rz(theta_i) e2, inside the
cube [-100 R, 100 R]^3; each face is 5 biquadratic patches: a central disk of radius 0.8 R (single rational
patch, its four edges quarter arcs) and four quarter-annulus crown patches 0.8 R..R (u: angle, v: radius). The
outer cube is 6 biquadratic single-element patches (flat, the uniform far field is linear, so exact). Level n
inserts knots so every crack patch has n control points per direction (paper: n = 3..7 for 378..1626 dof).

prism (paper Sec. 4.3, Fig. 18): edge crack in [-1, 1]^2 x [0, 2], pure-mode Williams data on the boundary; --n is
the number of elements per unit length k.

usage: make_crack.py penny [--n 3,4,5,6,7] [--incline DEG] [--out geo/crack]
       make_crack.py ellipse [--aspect 0.5] [--n ...] [--incline DEG] [--layout disk|ogrid]   (paper Sec. 4.2)
       make_crack.py prism --n 1,2,3,4
       make_crack.py senb [--a-w 0.5] [--width 10] [--thickness 4] --n 1,2,3,4   (3-point bend coupon, S = 4W)
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_geometries as mg  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo", "crack")
S2 = math.sqrt(0.5)


def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _patch(X, W, p=2, q=2):
    """Single-span patch from Euclidean control points X (nv, nu, 3) and weights W (nv, nu)."""
    Pw = np.concatenate([X * W[..., None], W[..., None]], axis=-1)
    K = [0.0] * 3 + [1.0] * 3
    return mg.Patch(p, q, K, K, Pw)


def _normal(P):
    e = 1e-6
    xu = mg.bezier_point(P, 0.5 + e, 0.5) - mg.bezier_point(P, 0.5 - e, 0.5)
    xv = mg.bezier_point(P, 0.5, 0.5 + e) - mg.bezier_point(P, 0.5, 0.5 - e)
    return np.cross(xu, xv)


def _flip_u(P):
    """Reverse u (same surface, opposite normal; v and the v = 1 front edge are kept)."""
    U = 1.0 - np.asarray(P.U)[::-1]
    return mg.Patch(P.p, P.q, U, P.V, np.ascontiguousarray(P.Pw[:, ::-1]))


def _facing(P, n):
    return P if np.dot(_normal(P), n) > 0 else _flip_u(P)


def _insert(P, n):
    """Knot insertion to n control points per direction (uniform interior knots)."""
    for direction in (0, 1):
        deg = P.p if direction == 0 else P.q
        have = (P.Pw.shape[1] if direction == 0 else P.Pw.shape[0])
        new = [i / (n - deg) for i in range(1, n - deg)] if n > deg + 1 else []
        assert have == deg + 1, "single-span patches only"
        for u in new:
            K = P.U if direction == 0 else P.V
            if direction == 0:
                rows = [mg.insert_knot_curve(deg, K, P.Pw[j], u) for j in range(P.Pw.shape[0])]
                P.U = rows[0][0]
                P.Pw = np.stack([r[1] for r in rows], axis=0)
            else:
                cols = [mg.insert_knot_curve(deg, K, P.Pw[:, i], u) for i in range(P.Pw.shape[1])]
                P.V = cols[0][0]
                P.Pw = np.stack([c[1] for c in cols], axis=1)
    return P


def cube(L):
    out = []
    for P in mg.cube_p2():
        Q = mg.Patch(P.p, P.q, P.U, P.V, P.Pw.copy())
        Q.set_euclid(L * (Q.euclid() - 0.5))
        out.append(Q)
    return out


def penny_face(R, ri, e1, e3, outward):
    """Five biquadratic patches of a disk of radius R in span(e1, e3): central disk ri, crown ri..R."""
    def at(c1, c3):
        return c1 * e1 + c3 * e3
    # central disk: corners on the circle at 45 deg, edge midpoints at ri sqrt2 (weight sqrt2/2)
    g = [-1.0, 0.0, 1.0]
    X = np.zeros((3, 3, 3))
    W = np.ones((3, 3))
    for j, b in enumerate(g):
        for i, a in enumerate(g):
            if a != 0 and b != 0:
                X[j, i] = at(a * ri * S2, b * ri * S2)
            elif a == 0 and b == 0:
                X[j, i] = 0.0
            else:
                X[j, i] = at(a * ri * math.sqrt(2.0), b * ri * math.sqrt(2.0))
                W[j, i] = S2
    out = [_facing(_patch(X, W), outward)]
    # crown k: angles th_k -+ 45 deg, u along the angle, v radial (v = 1 on the front)
    for k in range(4):
        th = math.radians(-90.0 + 90.0 * k)
        X = np.zeros((3, 3, 3))
        W = np.ones((3, 3))
        for i, (dth, rad, w) in enumerate([(-math.pi / 4, 1.0, 1.0), (0.0, math.sqrt(2.0), S2), (math.pi / 4, 1.0, 1.0)]):
            d = np.cos(th + dth) * e1 + np.sin(th + dth) * e3
            for j, r in enumerate([ri, 0.5 * (ri + R), R]):
                X[j, i] = r * rad * d
                W[j, i] = w
        out.append(_facing(_patch(X, W), outward))
    return out


def penny_face_ogrid(R, ri, e1, e3, outward):
    """O-grid: central square with corners on the circle of radius ri, four crown patches from its sides to the
    front arcs (no degenerate patch; the crown is ri..R wide at the corners, R - ri/sqrt2 at the middles)."""
    a = ri * S2
    g = [-1.0, 0.0, 1.0]
    X = np.array([[a * u * e1 + a * v * e3 for u in g] for v in g])
    out = [_facing(_patch(X, np.ones((3, 3))), outward)]
    for k in range(4):
        th = math.radians(-90.0 + 90.0 * k)
        c, s_ = np.cos(th) * e1 + np.sin(th) * e3, -np.sin(th) * e1 + np.cos(th) * e3
        X = np.zeros((3, 3, 3))
        W = np.ones((3, 3))
        for i, (t, w) in enumerate([(-1.0, 1.0), (0.0, S2), (1.0, 1.0)]):
            inner = a * c + a * t * s_
            outer = R * (math.sqrt(2.0) * c if t == 0 else (c + t * s_) * S2)
            for j, f in enumerate([0.0, 0.5, 1.0]):
                X[j, i] = (1 - f) * inner + f * outer
                W[j, i] = w
        out.append(_facing(_patch(X, W), outward))
    return out


def penny(n, R=1.0, L=200.0, incline=0.0, ri_frac=0.8, layout="disk", aspect=1.0):
    """aspect = b / a < 1: the elliptical crack of Sec. 4.2 (semi-axes a = R along e1, b = aspect R along e3), the
    affine image of the penny's patches (exact: NURBS are affine invariant), inner ellipse at ri_frac."""
    Rz = _rot_z(incline)
    m = Rz @ np.array([0.0, 1.0, 0.0])
    e1, e3 = Rz @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    ri = ri_frac * R
    face = penny_face if layout == "disk" else penny_face_ogrid
    plus = [_insert(P, n) for P in face(R, ri, e1, aspect * e3, -m)]
    minus = [_insert(P, n) for P in face(R, ri, e1, aspect * e3, m)]
    outer = cube(L)
    n0, n1 = len(outer), len(outer) + len(plus)
    meta = {
        "kind": "penny", "outer": [0, n0], "plus": [n0, n1], "minus": [n1, n1 + len(minus)],
        "crown": [n0 + k for k in range(1, 5)], "front_edge": "v1",
        "plane": {"point": [0.0, 0.0, 0.0], "normal": m.tolist(), "e1": e1.tolist(), "e3": e3.tolist()},
        "R": R, "a": R, "b": aspect * R, "L": L, "incline_deg": incline, "n": n, "layout": layout,
    }
    return outer + plus + minus, meta


def rect_patch(o, eu, ev, nu, nv, outward, p=2):
    """Flat degree-p patch o + s eu + t ev, s, t in [0, 1], nu x nv uniform elements; u reversed if needed so the
    normal xu x xv points along `outward` (v and its v = 1 edge are kept)."""
    o, eu, ev = map(np.asarray, (o, eu, ev))
    def knots(ne):
        return [0.0] * (p + 1) + [i / ne for i in range(1, ne)] + [1.0] * (p + 1)
    U, V = knots(nu), knots(nv)
    gu = [sum(U[i + 1:i + p + 1]) / p for i in range(nu + p)]
    gv = [sum(V[i + 1:i + p + 1]) / p for i in range(nv + p)]
    X = np.array([[o + a * eu + b * ev for a in gu] for b in gv])
    P = mg.Patch(p, p, U, V, np.concatenate([X, np.ones(X.shape[:2] + (1,))], axis=-1))
    return P if np.dot(np.cross(eu, ev), outward) > 0 else _flip_u(P)


def prism(k):
    """Prism [-1, 1] x [-1, 1] x [0, 2] with the edge crack y = 0, -1 <= x <= 0 (paper Sec. 4.3, Fig. 18), front
    x = 0 along z. 7 outer patches (the x = -1 face split at the crack mouth), one patch per crack face; element size
    1/k. The z faces carry the front-end enrichment and have knot lines at x = 0 and y = 0."""
    e1, e2, e3 = np.eye(3)
    n2 = 2 * k
    outer = [
        rect_patch([-1, -1, 0], 2 * e1, 2 * e2, n2, n2, -e3),   # 0 z = 0
        rect_patch([-1, -1, 2], 2 * e1, 2 * e2, n2, n2, e3),    # 1 z = 2
        rect_patch([-1, 1, 0], 2 * e1, 2 * e3, n2, n2, e2),     # 2 y = 1
        rect_patch([-1, -1, 0], 2 * e1, 2 * e3, n2, n2, -e2),   # 3 y = -1
        rect_patch([1, -1, 0], 2 * e2, 2 * e3, n2, n2, e1),     # 4 x = 1
        rect_patch([-1, 0, 0], 2 * e3, e2, n2, k, -e1),         # 5 x = -1, y > 0
        rect_patch([-1, -1, 0], 2 * e3, e2, n2, k, -e1),        # 6 x = -1, y < 0
    ]
    # crack faces: u along z, v from the mouth (x = -1) to the front (x = 0)
    plus = [rect_patch([-1, 0, 0], 2 * e3, e1, n2, k, -e2)]
    minus = [rect_patch([-1, 0, 0], 2 * e3, e1, n2, k, e2)]
    n0, n1 = len(outer), len(outer) + 1
    meta = {
        "kind": "prism", "outer": [0, n0], "plus": [n0, n1], "minus": [n1, n1 + 1], "crown": [n0],
        "front_edge": "v1", "enriched_outer": [[0, 0], [1, 0]], "dirichlet": [2, 3, 4, 5, 6],
        "plane": {"point": [-0.5, 0.0, 1.0], "normal": [0.0, 1.0, 0.0], "e1": [1.0, 0.0, 0.0], "e3": [0.0, 0.0, 1.0]},
        "front": {"point": [0.0, 0.0, 0.0], "n": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "t": [0.0, 0.0, 1.0]},
        "k": k,
    }
    return outer + plus + minus, meta


def stations(a, b, focus, h0, hmax, ratio=1.5):
    """Breakpoints of [a, b]: element size h0 at each focus point (a breakpoint itself), growing by `ratio` per
    element away from it up to hmax, then uniform elements of at most hmax."""
    pts = {a, b}
    fs = sorted(f for f in focus if a - 1e-12 <= f <= b + 1e-12)
    for k, f in enumerate(fs):
        pts.add(f)
        lo = 0.5 * (fs[k - 1] + f) if k else a          # each run stays in its own half-way cell
        hi = 0.5 * (f + fs[k + 1]) if k + 1 < len(fs) else b
        for sgn, end in ((-1, lo), (1, hi)):
            x, h = f, h0
            while h < hmax and (x + sgn * h - end) * sgn < -0.3 * h0:
                x += sgn * h
                pts.add(x)
                h *= ratio
    pts = sorted(pts)
    out = [pts[0]]
    for x0, x1 in zip(pts[:-1], pts[1:]):
        m = max(1, math.ceil((x1 - x0) / hmax - 1e-9))
        out += [x0 + (x1 - x0) * k / m for k in range(1, m + 1)]
    # drop slivers left where a graded run meets an end or another run
    keep = [out[0]]
    for x in out[1:]:
        if x - keep[-1] < 0.3 * h0 and x != b:
            continue
        keep.append(x)
    if keep[-1] - keep[-2] < 0.3 * h0 and len(keep) > 2 and keep[-2] not in focus:
        del keep[-2]
    return keep


def box_patch(o, eu, ev, su, sv, outward, p=2):
    """Flat degree-p patch over o + s eu + t ev with element breakpoints su, sv (fractions in [0, 1], ends
    included); u reversed if needed so the normal points along `outward`."""
    o, eu, ev = map(np.asarray, (o, eu, ev))
    U = [0.0] * p + list(su) + [1.0] * p
    V = [0.0] * p + list(sv) + [1.0] * p
    gu = [sum(U[i + 1:i + p + 1]) / p for i in range(len(U) - p - 1)]
    gv = [sum(V[i + 1:i + p + 1]) / p for i in range(len(V) - p - 1)]
    X = np.array([[o + a * eu + b * ev for a in gu] for b in gv])
    P = mg.Patch(p, p, U, V, np.concatenate([X, np.ones(X.shape[:2] + (1,))], axis=-1))
    return P if np.dot(np.cross(eu, ev), outward) > 0 else _flip_u(P)


def senb(n, W=10.0, B=4.0, a_W=0.5, S_W=4.0, pad=0.05, overhang=0.5, crown=0.2):
    """Single-edge-notched bend bar with a straight through crack (ASTM D5045 / E399 SENB).

    x along the span (-L/2..L/2, L = S + 2 overhang W), y up (0..W, crack mouth on y = 0), z through the
    thickness (0..B). Crack plane x = 0 (m = e1), crack 0 <= y <= a, front y = a along z. Support pads of width
    pad W on y = 0 at x = -+S/2 (patch group "supports"), load-nose pad on y = W at x = 0 ("nose"), end faces
    ("ends"). Every patch has dofs of its own (patch-discontinuous, as the prism).

    The front enrichment lives in a fixed window, as the penny's crown ring: the crack face nearest the front
    (a - c .. a, c ~ crown a) and on each z face the square |x| <= c, |y - a| <= c, split at x = 0 (the front
    ends on them). Enriching whole faces instead lets K absorb the field's higher-order terms (measured: K_I
    erratic within 5-10 % under refinement). Level n: element size a/(4n) at the front, at most W/(2n) away
    from it, growth 1.5."""
    e1, e2, e3 = np.eye(3)
    a, S = a_W * W, S_W * W
    L2, w2 = 0.5 * S + overhang * W, 0.5 * pad * W
    h0, hmax = a / (4.0 * n), W / (2.0 * n)
    sy = stations(0.0, W, [a], h0, hmax)                                     # y on z faces and crack faces
    sxl, sxr = stations(-L2, 0.0, [0.0], h0, hmax), stations(0.0, L2, [0.0], h0, hmax)
    near = lambda st, v: min(st, key=lambda t: abs(t - v))
    ylo, yhi = near(sy[:-1], a - crown * a), near(sy[1:], a + crown * a)
    xlo, xhi = near(sxl[:-1], -crown * a), near(sxr[1:], crown * a)
    zs = stations(0.0, B, [0.0, B], 2 * h0, hmax)                           # through the thickness
    def cut(st, lo, hi):
        pts = [t for t in st if lo - 1e-12 <= t <= hi + 1e-12]
        return [(t - lo) / (hi - lo) for t in pts]
    def rect(x0, x1, y0, y1, z, nz):
        sx = sxl if x1 <= 0 else sxr
        return box_patch([x0, y0, z], (x1 - x0) * e1, (y1 - y0) * e2, cut(sx, x0, x1), cut(sy, y0, y1), nz)
    outer, names = [], {"supports": [], "nose": [], "bottom": [], "top": [], "ends": [], "z_window": [], "z_far": []}
    def add(P, group):
        names[group].append(len(outer))
        outer.append(P)
    for z, nz in ((0.0, -e3), (B, e3)):
        add(rect(xlo, 0.0, ylo, yhi, z, nz), "z_window")
        add(rect(0.0, xhi, ylo, yhi, z, nz), "z_window")
        for x0, x1, xc in ((-L2, xlo, None), (xhi, L2, None), (xlo, 0.0, 1), (0.0, xhi, 1)):
            if xc is None:
                add(rect(x0, x1, 0.0, W, z, nz), "z_far")
            else:
                add(rect(x0, x1, 0.0, ylo, z, nz), "z_far")
                add(rect(x0, x1, yhi, W, z, nz), "z_far")
    # bottom y = 0 (split at the mouth and the support pads), top y = W (split at the nose), ends
    bx = [-L2, -0.5 * S - w2, -0.5 * S + w2, 0.0, 0.5 * S - w2, 0.5 * S + w2, L2]
    zf = [t / B for t in zs]
    for lo, hi in zip(bx[:-1], bx[1:]):
        sx = sxl if hi <= 0 else sxr
        grp = "supports" if abs(abs(0.5 * (lo + hi)) - 0.5 * S) < w2 else "bottom"
        su = cut(sorted(set(sx) | {lo, hi}), lo, hi)
        add(box_patch([lo, 0, 0], (hi - lo) * e1, B * e3, su, zf, -e2), grp)
    for lo, hi in ((-L2, -w2), (-w2, w2), (w2, L2)):
        su = cut(sorted(set(sxl) | set(sxr) | {lo, hi}), lo, hi)
        add(box_patch([lo, W, 0], (hi - lo) * e1, B * e3, su, zf, e2), "nose" if lo == -w2 else "top")
    for x, nx in ((-L2, -e1), (L2, e1)):
        add(box_patch([x, 0, 0], W * e2, B * e3, cut(stations(0, W, [], hmax, hmax), 0, W), zf, nx), "ends")
    # crack faces (u along z, v toward the front): mouth part 0..a-c, crown a-c..a
    plus = [box_patch([0, 0, 0], B * e3, ylo * e2, zf, cut(sy, 0.0, ylo), -e1),
            box_patch([0, ylo, 0], B * e3, (a - ylo) * e2, zf, cut(sy, ylo, a), -e1)]
    minus = [box_patch([0, 0, 0], B * e3, ylo * e2, zf, cut(sy, 0.0, ylo), e1),
             box_patch([0, ylo, 0], B * e3, (a - ylo) * e2, zf, cut(sy, ylo, a), e1)]
    n0 = len(outer)
    meta = {
        "kind": "senb", "outer": [0, n0], "plus": [n0, n0 + 2], "minus": [n0 + 2, n0 + 4], "crown": [n0 + 1],
        "front_edge": "v1", "enriched_outer": [[p, 0] for p in names["z_window"]], "patches": names,
        "plane": {"point": [0.0, 0.5 * a, 0.5 * B], "normal": [1.0, 0.0, 0.0], "e1": [0.0, 1.0, 0.0], "e3": [0.0, 0.0, 1.0]},
        "W": W, "B": B, "a": a, "S": S, "L": 2 * L2, "pad": pad * W, "crown_c": a - ylo, "n": n,
    }
    return outer + plus + minus, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["penny", "ellipse", "prism", "senb"])
    ap.add_argument("--aspect", type=float, default=0.5, help="ellipse: b / a")
    ap.add_argument("--a-w", type=float, default=0.5, help="senb: crack depth / width")
    ap.add_argument("--width", type=float, default=10.0, help="senb: W")
    ap.add_argument("--thickness", type=float, default=4.0, help="senb: B")
    ap.add_argument("--n", default="3,4,5,6,7")
    ap.add_argument("--incline", type=float, default=0.0)
    ap.add_argument("--layout", choices=["disk", "ogrid"], default="disk")
    ap.add_argument("--out", default=ROOT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for n in map(int, a.n.split(",")):
        if a.kind == "prism":
            patches, meta = prism(n)
            name = "prism_k%d" % n
        elif a.kind == "senb":
            patches, meta = senb(n, W=a.width, B=a.thickness, a_W=a.a_w)
            name = "senb_aw%g_n%d" % (a.a_w, n)
        else:
            asp = a.aspect if a.kind == "ellipse" else 1.0
            patches, meta = penny(n, incline=a.incline, layout=a.layout, aspect=asp)
            if a.kind == "ellipse":
                meta["kind"] = "ellipse"
            name = "%s_%si%g_n%d" % (a.kind, "" if a.layout == "disk" else "og_", a.incline, n)
        path = os.path.join(a.out, name + ".dat")
        mg.write_dat(path, patches, name)
        with open(os.path.join(a.out, name + ".crack.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(path)


if __name__ == "__main__":
    main()
