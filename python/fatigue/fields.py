"""Unit-net-stress fields of a specimen, solved once on IgBEM and cached (npz keyed by the specimen tag and mesh).

    f = field(spec, h=1.5, work=...)
    f["d"], f["z"], f["s1_line"]   bisector distances from the root (mm), thickness stations, sigma_1 (nz, nd)
    f["Kt_mid"], f["Kt_mean"]      root sigma_1 / nominal net stress at mid-thickness, and averaged over z
    f["P"], f["w"], f["s1"]        TET10 Gauss points of the quarter, weights (mm^3), positive sigma_1 there
    f["g1"]                        |grad sigma_1| at the Gauss points (for gradient-supported Weibull)
    f["sig"]                       full stress there, Voigt (xx, yy, zz, yz, xz, xy), float32 (ply-frame criteria)

Reuses failure.igabem_backend (STEP -> multipatch, TET10 mesh, `bc_case --serve`) and failure.fields (TET10
quadrature and gradient), exactly as failure.pla_orient does for the static PLA plates. A traction-loaded model's
stress does not depend on E, so one field serves every material and every load level.
"""
import os
import time

import numpy as np

from failure.fields import tet10_gradient, tet10_quadrature
from failure.igabem_backend import ROOT, Solve, fea_mesh, igabem_geometry

from . import blocks
from .specimens import case, quarter_step

WORK = os.path.join(ROOT, "cache", "fatigue")
Z_STATIONS = (0.125, 0.25, 0.5, 0.75, 0.875)       # fractions of the thickness


def _s1(sig):
    return np.linalg.eigvalsh(sig)[..., -1]


def field(spec, h=1.5, work=WORK, volume=True, nd=200, hmat=False, coarsen=0, log=print):
    """h: element size away from the root. An edge notch is built from the designed block layout (fatigue.blocks,
    root-graded, < 1 s); a hole still goes through the STEP import. hmat: H-matrix solver (needed past ~25k unknowns,
    where the dense LU outgrows 15 GB); same cache file. coarsen: nested coarser level of the block mesh (convergence)."""
    if spec.shape == "plain":
        raise ValueError("a plain bar has the unit field everywhere; no solve")
    designed = spec.shape in ("edge", "hole")          # exact block layout (fatigue.blocks)
    path = os.path.join(work, f"{spec.tag}_{'b' if designed else 'h'}{h:g}{f'_c{coarsen}' if coarsen else ''}"
                        f"{'' if volume else '_line'}.npz")
    if os.path.exists(path):
        f = dict(np.load(path))
        if not volume or "sig" in f:
            return f
        log(f"{spec.tag}: cache has no full stress tensor; re-solving")
    os.makedirs(work, exist_ok=True)
    step = os.path.join(work, f"{spec.tag}.step")
    if not os.path.exists(step):
        quarter_step(spec, step)
    t0 = time.perf_counter()
    if designed:
        dat = os.path.join(work, f"{spec.tag}_b{h:g}_c{coarsen}.dat")
        r = 1.5 / h                                    # h is one global mesh scale: 1.5 = the block defaults
        info = blocks.write(spec, dat, h=h, h0=spec.rho / 8 / r, hz0=spec.T / 60 / r, nz=max(4, round(12 * r)),
                            coarsen=coarsen)
    else:
        dat, info = igabem_geometry(step, work, h=h, curvature=12, degree=2, log=log)
    (rx, ry, _), sgn = spec.root
    z = np.array(Z_STATIONS) * spec.T
    d = np.concatenate([[0.0], np.geomspace(2e-3, 0.999 * spec.ligament, nd - 1)])
    out = {"d": d, "z": z, "patches": info.get("patches", 0)}
    body = dict(case(spec), **({"solver": {"backend": "hmat", "eps": 1e-5}} if hmat else {}))
    with Solve(dat, body, 3000.0, 0.35, workdir=os.path.join(work, f"run_{spec.tag}_h{h:g}"), log=log) as s:
        line = np.array([(rx + sgn * dd, ry, zz) for zz in z for dd in d])
        on = np.tile(d == 0.0, len(z))
        sig, _ = s.stress(line, on)
        out["s1_line"] = _s1(sig).reshape(len(z), len(d))
        out["unknowns"] = int(next(ln for ln in s.report if "unknowns" in ln).split(" unknowns")[0].split()[-1])
        out["t_solve_s"] = s.seconds
        if volume:
            rho = max(spec.rho, 0.05)
            mesh = fea_mesh(step, h=1.0, curvature=16, focus=[(rx, ry, spec.T / 2, 3 * rho, rho / 5)])
            pts, cells = np.asarray(mesh.points, float), np.asarray(mesh.cells)
            onv = np.zeros(len(pts), bool)
            onv[np.unique(mesh.surface_tri6)] = True
            sig_n, wind = s.stress(pts, onv)
            bad = ~onv & (wind < 0.5)
            if bad.any():
                sig_n[bad], _ = s.stress(pts[bad], True)
            P, w, _ = tet10_quadrature(pts, cells, np.zeros((len(pts), 1)))
            sq, _ = s.stress(P)
            grad = tet10_gradient(pts, cells, sig_n)                       # (nq, 3, 3, 3) d sigma_ij / d x_k
            lam, vec = np.linalg.eigh(sq)
            n1 = vec[..., -1]                                               # d sigma_1 = n1 . (d sigma) . n1
            g1 = np.einsum("qi,qijk,qj->qk", n1, grad, n1)
            out.update(P=P, w=w, s1=np.maximum(lam[:, -1], 0.0), g1=np.linalg.norm(g1, axis=1),
                       sig=sq[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].astype(np.float32))
    out["Kt_mid"] = float(out["s1_line"][len(z) // 2, 0])
    out["Kt_mean"] = float(out["s1_line"][:, 0].mean())
    out["seconds"] = time.perf_counter() - t0
    np.savez(path, **out)
    log(f"{spec.tag}: Kt mid {out['Kt_mid']:.3f}, mean {out['Kt_mean']:.3f}, {out['unknowns']} unknowns, "
        f"{out['seconds']:.0f} s")
    return out


def root_surface(spec, h=1.5, work=WORK, nth=61, nz=41, nr=30, log=print):
    """sigma_1 ON the free surfaces around an edge-notch root, evaluated directly by IgBEM (no extrapolation from Gauss
    points), cached. Notch face: arc angle th from the root (0) to the flank tangent point, x thickness z in [0, T].
    Side face z = 0 (z = T is its mirror): polar grid around the arc centre, r in [rho, rho + min(rho, ligament/2)].
    Returns th (rad), z, s1_arc (nth, nz), r, s1_side (nth, nr) and the area weights dA_arc, dA_side (mm^2, quarter)."""
    if spec.shape != "edge":
        raise ValueError("root_surface: edge notches only")
    path = os.path.join(work, f"{spec.tag}_b{h:g}_surf.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    os.makedirs(work, exist_ok=True)
    t0 = time.perf_counter()
    r_ = 1.5 / h
    dat = os.path.join(work, f"{spec.tag}_b{h:g}_c0.dat")
    blocks.write(spec, dat, h=h, h0=spec.rho / 8 / r_, hz0=spec.T / 60 / r_, nz=max(4, round(12 * r_)))
    cx = spec.WN / 2 + spec.rho
    th = np.linspace(0.0, np.radians(90 - spec.alpha_deg / 2), nth)
    z = np.linspace(0.0, spec.T, nz)
    r = spec.rho + np.concatenate([[0.0], np.geomspace(1e-3, 1.0, nr - 1)]) * min(spec.rho, spec.ligament / 2)
    arc = np.array([(cx - spec.rho * np.cos(t), spec.rho * np.sin(t), zz) for t in th for zz in z])
    side = np.array([(cx - rr * np.cos(t), rr * np.sin(t), 0.0) for t in th for rr in r])
    with Solve(dat, case(spec), 3000.0, 0.35, workdir=os.path.join(work, f"run_{spec.tag}_surf"), log=log) as s:
        sa, _ = s.stress(arc, True)
        ss, _ = s.stress(side, True)
    tw = np.gradient(th)
    out = dict(th=th, z=z, r=r, s1_arc=_s1(sa).reshape(nth, nz), s1_side=_s1(ss).reshape(nth, nr),
               dA_arc=spec.rho * np.outer(tw, np.gradient(z)), dA_side=np.outer(tw, r * np.gradient(r)),
               seconds=time.perf_counter() - t0)
    np.savez(path, **out)
    log(f"{spec.tag}: root surface sigma_1 max {out['s1_arc'].max():.3f}, {out['seconds']:.0f} s")
    return out
