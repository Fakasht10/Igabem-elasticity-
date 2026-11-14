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
       make_crack.py prism_tension --n 1,2,3,4 [--lateral enriched|split|aligned]   (paper Sec. 4.4; n refines the
                     crack faces, h = 0.625 / n; outer faces h = 0.625, lateral faces 0.3125)
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


def _cells(L, h):
    return max(1, int(round(L / h)))


def prism_tension(k, kc=None, kl=None, lateral="enriched"):
    """Paper Sec. 4.4 (Fig. 23): block x in [0, 5] (crack depth direction), y in [-4.375, 4.375] (load), z in
    [0, 7.5] (front), edge crack y = 0, 0 <= x <= 2.5 (a / b = 0.5), front x = 2.5. Patch-discontinuous: the cut face
    x = 0 split at the crack mouth (the paper uses a Heaviside enrichment there instead), the face x = 5 split into
    the clamped band |y| < 0.4375 and the two free parts, the z faces Williams-enriched with knot lines at x = 2.5 and
    y = 0. Element size 0.625 / k."""
    e1, e2, e3 = np.eye(3)
    h = 0.625 / k
    hc = 0.625 / (kc or k)  # crack faces
    hl = 0.625 / (kl or k)  # lateral (enriched) faces
    W, H, T, a, band = 5.0, 4.375, 7.5, 2.5, 0.4375
    nx, ny, nz = 2 * _cells(a, h), 2 * _cells(H, h), _cells(T, h)
    ylo = H - band
    # lateral faces z = 0, T: one enriched patch each ("enriched", the paper's Fig. 24b), two enriched halves split at
    # y = 0 ("split"), or four plain patches aligned with the crack and the front ("aligned", Fig. 24a)
    lat, enr = [], []
    for z, nrm in ((0.0, -e3), (T, e3)):
        if lateral == "enriched":
            enr.append(len(lat))
            lat.append(rect_patch([0, -H, z], W * e1, 2 * H * e2, 2 * _cells(a, hl), 2 * _cells(H, hl), nrm))
        elif lateral == "split":
            for y0 in (0.0, -H):
                enr.append(len(lat))
                lat.append(rect_patch([0, y0, z], W * e1, H * e2, 2 * _cells(a, hl), _cells(H, hl), nrm))
        else:
            for x0 in (0.0, a):
                for y0 in (0.0, -H):
                    lat.append(rect_patch([x0, y0, z], (W - a if x0 else a) * e1, H * e2, _cells(a, hl), _cells(H, hl), nrm))
    outer = lat + [
        rect_patch([0, H, 0], W * e1, T * e3, nx, nz, e2),                   # top, sigma0 e2
        rect_patch([0, -H, 0], W * e1, T * e3, nx, nz, -e2),                 # bottom, -sigma0 e2
        rect_patch([0, 0, 0], T * e3, H * e2, nz, ny // 2, -e1),             # x = 0, y > 0
        rect_patch([0, -H, 0], T * e3, H * e2, nz, ny // 2, -e1),            # x = 0, y < 0
        rect_patch([W, band, 0], T * e3, ylo * e2, nz, _cells(ylo, h), e1),  # x = W, y > band
        rect_patch([W, -H, 0], T * e3, ylo * e2, nz, _cells(ylo, h), e1),    # x = W, y < -band
        rect_patch([W, -band, 0], T * e3, 2 * band * e2, nz, 2, e1),         # x = W, clamped band
    ]
    plus = [rect_patch([0, 0, 0], T * e3, a * e1, _cells(T, hc), _cells(a, hc), -e2)]
    minus = [rect_patch([0, 0, 0], T * e3, a * e1, _cells(T, hc), _cells(a, hc), e2)]
    n0, n1 = len(outer), len(outer) + 1
    meta = {
        "kind": "prism_tension", "outer": [0, n0], "plus": [n0, n1], "minus": [n1, n1 + 1], "crown": [n0],
        "front_edge": "v1", "enriched_outer": [[i, 0] for i in enr], "dirichlet": [len(lat) + 6],
        "top": len(lat), "bottom": len(lat) + 1, "lateral": lateral,
        "plane": {"point": [1.25, 0.0, 3.75], "normal": [0.0, 1.0, 0.0], "e1": [1.0, 0.0, 0.0], "e3": [0.0, 0.0, 1.0]},
        "a": a, "W": W, "T": T, "k": k,
    }
    return outer + plus + minus, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["penny", "ellipse", "prism", "prism_tension"])
    ap.add_argument("--aspect", type=float, default=0.5, help="ellipse: b / a")
    ap.add_argument("--lateral", choices=["enriched", "split", "aligned"], default="split",
                    help="prism_tension: lateral-face layout")
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
        elif a.kind == "prism_tension":  # n: crack-face refinement; outer h = 0.625, lateral h = 0.3125
            patches, meta = prism_tension(1, n, 2, a.lateral)
            name = "prismt_%s_c%d" % (a.lateral, n)
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
