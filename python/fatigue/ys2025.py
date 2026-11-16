"""Yaren & Susmel (IJF 2025) PLA in-fill specimens as IgBEM quarter models, unit net stress, solved once and cached.

    V r0.15 (35 deg), U r1, U r3: double-edge notched plate, gross 25 mm, net 12 mm, thickness 5 mm, length 85 mm
                                  (fatigue.blocks: exact root-graded block layout)
    plain:                        hourglass, side arcs R 18.5 mm, minimum width 12 mm, gross 25 mm, length 94 mm
                                  (STEP through gmsh, imported once by step_auto with a focus sphere at the root; Kt ~1.05, mild gradient)
Grip holes/notches of the real specimens are left out: the load is a uniform traction at the quarter's end, which is
more than 3 net widths from the root. Every profile is sigma_1 along the notch bisector at thickness stations.
"""
import math
import os

import numpy as np

from fatigue import blocks
from fatigue.specimens import Specimen, case
from failure.igabem_backend import ROOT, Solve, igabem_geometry

WORK = os.path.join(ROOT, "cache", "ys2025")
WG, WN, T = 25.0, 12.0, 5.0
SPECS = {"V_r0.15": Specimen("edge", T=T, rho=0.15, alpha_deg=35.0, WG=WG, WN=WN, L=85.0),
         "U_r1": Specimen("edge", T=T, rho=1.0, alpha_deg=0.0, WG=WG, WN=WN, L=85.0),
         "U_r3": Specimen("edge", T=T, rho=3.0, alpha_deg=0.0, WG=WG, WN=WN, L=85.0)}
R_PLAIN, L_PLAIN = 18.5, 94.0
Z_FRAC = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])   # half thickness (mirror symmetric), 0 = free face


def plain_step(path):
    """Quarter of the hourglass: root (WN/2, 0), arc centre (WN/2 + R, 0) up to the side x = WG/2, then straight."""
    import gmsh
    yt = math.sqrt(R_PLAIN ** 2 - (R_PLAIN - (WG - WN) / 2) ** 2)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("ys_plain"); o = gmsh.model.occ
    p = [o.addPoint(*q, 0) for q in ((0, 0), (WN / 2, 0), (WG / 2, yt), (WG / 2, L_PLAIN / 2), (0, L_PLAIN / 2))]
    c = o.addPoint(WN / 2 + R_PLAIN, 0, 0)
    cv = [o.addLine(p[0], p[1]), o.addCircleArc(p[1], c, p[2]), o.addLine(p[2], p[3]), o.addLine(p[3], p[4]), o.addLine(p[4], p[0])]
    o.remove([(0, c)]); o.extrude([(2, o.addPlaneSurface([o.addCurveLoop(cv)]))], 0, 0, T); o.synchronize()
    gmsh.write(path); gmsh.finalize()


def geometry(name, h=None):
    """(.dat path, info). Notched: blocks with h0 = rho/8 at the root, hz0 = T/60 at both faces."""
    os.makedirs(WORK, exist_ok=True)
    if name == "plain":
        step = os.path.join(WORK, "plain_R18.5.step")
        if not os.path.exists(step):
            plain_step(step)
        return igabem_geometry(step, WORK, h=h or 3.0, curvature=12, degree=2, focus=[(WN / 2, 0.0, T / 2, 6.0, 1.0)])
    s, h = SPECS[name], h or 1.5
    dat = os.path.join(WORK, f"{s.tag}_b{h:g}.dat")
    info = blocks.write(s, dat, h=h, h0=s.rho / 8, hz0=T / 60, nz=12)
    return dat, info


def plain_case():
    plane = lambda p, n: {"plane": {"point": p, "normal": n, "tol": 1e-6}}  # noqa: E731
    return {"supports": [{"label": "sym x", "select": plane([0, 0, 0], [1, 0, 0]), "fix": ["ux"], "snap": True},
                         {"label": "sym y", "select": plane([0, 0, 0], [0, 1, 0]), "fix": ["uy"], "snap": True}],
            "cases": [{"name": "pull", "loads": [{"label": "grip", "select": plane([0, L_PLAIN / 2, 0], [0, 1, 0]),
                                                  "traction": [0, WN / WG, 0]}]}]}


def profiles(name, h=None, nd=160, log=print):
    """sigma_1 / net stress on the bisector y = 0 from the root inwards, at z = Z_FRAC * T: (d, z, s1 (nz, nd))."""
    out = os.path.join(WORK, f"profile_{name}_h{h or 0:g}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    dat, info = geometry(name, h)
    body = plain_case() if name == "plain" else case(SPECS[name])
    d = np.concatenate([[0.0], np.geomspace(1e-3, 0.999 * WN / 2, nd - 1)])
    z = Z_FRAC * T
    with Solve(dat, body, 3000.0, 0.35, workdir=os.path.join(WORK, f"run_{name}_h{h or 0:g}"), log=log) as s:
        pts = np.array([(WN / 2 - dd, 0.0, zz) for zz in z for dd in d])
        on = np.tile(d == 0.0, len(z)) | np.repeat(z == 0.0, len(d))
        sig, _ = s.stress(pts, on)
        s1 = np.linalg.eigvalsh(sig)[..., -1].reshape(len(z), len(d))
        unk = int(next(ln for ln in s.report if "unknowns" in ln).split(" unknowns")[0].split()[-1])
        res = dict(d=d, z=z, s1=s1, unknowns=unk, seconds=s.seconds, from_cache=s.from_cache)
    np.savez(out, **res)
    log(f"{name}: Kt mid {s1[-1, 0]:.3f}, face {s1[0, 0]:.3f}, {unk} unknowns, {res['seconds']:.0f} s")
    return res


def _edge_segments(name):
    """Free in-plane boundary of the quarter from the root to the grip end, as [('line', p, q) | ('arc', p, c, q)]."""
    if name == "plain":
        yt = math.sqrt(R_PLAIN ** 2 - (R_PLAIN - (WG - WN) / 2) ** 2)
        return [("arc", (WN / 2, 0.0), (WN / 2 + R_PLAIN, 0.0), (WG / 2, yt)), ("line", (WG / 2, yt), (WG / 2, L_PLAIN / 2))]
    from fatigue.specimens import outline
    seg = outline(SPECS[name])                       # a -> b (ligament), arc, [flank], side edge, grip end, axis
    i0 = next(i for i, g in enumerate(seg) if g[0] == "arc")
    return seg[i0:-2]                                # drop the grip end and the symmetry axis


def _seg_cells(g, n, grade):
    """Cell centres, lengths and arc-length start of a segment; graded towards its start when grade."""
    t = np.concatenate([[0.0], np.geomspace(2e-3, 1.0, n)]) if grade else np.linspace(0.0, 1.0, n + 1)
    tm = 0.5 * (t[1:] + t[:-1])
    if g[0] == "line":
        p, q = np.asarray(g[1]), np.asarray(g[2]); L = np.linalg.norm(q - p)
        return p + tm[:, None] * (q - p), np.diff(t) * L, L
    p, c, q = map(np.asarray, (g[1], g[2], g[3])); r = np.linalg.norm(p - c)
    a0, a1 = math.atan2(*(p - c)[::-1]), math.atan2(*(q - c)[::-1])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    a = a0 + tm * da
    return c + r * np.c_[np.cos(a), np.sin(a)], np.diff(t) * abs(da) * r, abs(da) * r


def surface(name, log=print):
    """sigma_1 / net stress ON every free face of the specimen, with area weights for the FULL specimen (quarter x 4).
    edge: the notch/hourglass face, flank and gross side edge (outline from the root to the grip end) x thickness;
          s (arc length from the root, mm), z, s1 (ns, nz), dA (ns, nz) mm^2, seg (segment index per s).
    side: flat faces z = 0 and z = T (mirror), triangle centroids x, y, s1, dA (both faces, full specimen).
    Cached in WORK; the solve itself is read from the solution cache."""
    out = os.path.join(WORK, f"surface_{name}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    import triangle
    dat, info = geometry(name)
    body = plain_case() if name == "plain" else case(SPECS[name])
    rho = R_PLAIN if name == "plain" else SPECS[name].rho
    segs = _edge_segments(name)
    P, ds, sseg, s0 = [], [], [], 0.0
    for k, g in enumerate(segs):
        L = _seg_cells(g, 2, False)[2]
        n = 60 if k == 0 else max(8, int(math.ceil(L / 0.5)))
        X, d_, L = _seg_cells(g, n, k == 0 or k == 1)
        P.append(X); ds.append(d_); sseg += [k] * len(X)
    X2, ds = np.vstack(P), np.concatenate(ds); s = np.cumsum(ds) - 0.5 * ds
    ze = np.concatenate([[0.0], np.geomspace(T / 400, T / 2, 14)]); ze = np.concatenate([ze, T - ze[-2::-1]])
    zc, dz = 0.5 * (ze[1:] + ze[:-1]), np.diff(ze)
    pts_e = np.array([(x, y, z) for x, y in X2 for z in zc])
    # side face: graded triangulation of the quarter outline
    poly = [(0.0, 0.0)] + [tuple(x) for x in np.vstack([[segs[0][1]], X2])] + [tuple(segs[-1][-1]), (0.0, segs[-1][-1][1])]
    poly = np.array(poly); sg = np.array([(i, (i + 1) % len(poly)) for i in range(len(poly))])
    h0 = min(rho / 6, 0.4); hf = 1.5
    Tr = triangle.triangulate(dict(vertices=poly, segments=sg), "pq30a%f" % (hf ** 2 / 2))
    for _ in range(14):
        V = Tr["vertices"][Tr["triangles"]]; c = V.mean(1)
        dist = np.hypot(c[:, 0] - WN / 2, c[:, 1]); h = np.minimum(hf, h0 + 0.25 * dist)
        a1, a2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]; A = 0.5 * np.abs(a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0])
        if np.all(A <= h ** 2 / 2 * 1.5):
            break
        Tr = triangle.triangulate(dict(vertices=Tr["vertices"], segments=Tr["segments"], triangles=Tr["triangles"],
                                       triangle_max_area=h ** 2 / 2), "rpq30a")
    V = Tr["vertices"][Tr["triangles"]]; c = V.mean(1)
    a1, a2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]; A = 0.5 * np.abs(a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0])
    pts_s = np.c_[c, np.zeros(len(c))]
    with Solve(dat, body, 3000.0, 0.35, workdir=os.path.join(WORK, f"run_{name}_h0"), log=log) as S:
        se, _ = S.stress(pts_e, True); ss, _ = S.stress(pts_s, True); cached = S.from_cache
    s1e = np.linalg.eigvalsh(se)[..., -1].reshape(len(X2), len(zc)); s1s = np.linalg.eigvalsh(ss)[..., -1]
    res = dict(edge_s=s, edge_xy=X2, edge_z=zc, edge_s1=s1e, edge_dA=4 * np.outer(ds, dz), edge_seg=np.array(sseg),
               side_xy=c, side_s1=s1s, side_dA=8 * A, from_cache=cached,
               area_edge=4 * ds.sum() * T, area_side=8 * A.sum())
    np.savez(out, **res)
    log(f"{name}: surface max s1 edge {s1e.max():.3f}, side {s1s.max():.3f}; {len(pts_e)} + {len(pts_s)} points; solve from cache {cached}")
    return res
