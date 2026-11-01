"""3D views of a notch-window solution (rve.notch_coupling): bead pieces translucent, weld and filament-joint faces
coloured by their normal traction (MPa per N of SENB load). Offscreen pyvista.

usage (from python/): python3 -m rve.notch_figures <manifest> <tractions.npy> <out.png> [--rho 1] [--clip 3]
"""
import argparse
import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_geometries import read_dat  # noqa: E402


def _tess(P, n=5):
    """Vertices (n*n, 3) and quads of a rational biquadratic Bezier patch (homogeneous Pw (3, 3, 4))."""
    t = np.linspace(0, 1, n)
    B = np.stack([(1 - t) ** 2, 2 * t * (1 - t), t ** 2], 1)          # (n, 3)
    H = np.einsum("vj,ui,jik->vuk", B, B, P.Pw)
    X = (H[..., :3] / H[..., 3:]).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    return X, np.column_stack([np.full(len(a), 4), a, a + 1, a + n + 1, a + n])


def scene(manifest, T, n=5):
    """(beads, faces): one PolyData of every patch; the weld/joint ones carry 'tn' (nan elsewhere)."""
    rows = [l.split() for l in open(manifest)]
    kind, x, nrm, t = T[:, 0].astype(int), T[:, 3:6], T[:, 7:10], T[:, 10:13]
    m = kind < 3
    tn = np.einsum("pc,pc->p", t[m], nrm[m])
    tree, xs = cKDTree(x[m]), x[m]
    pts, cells, val = [], [], []
    nv = 0
    cache = {}
    for r in rows:
        f, o = r[0], np.array(list(map(float, r[2:5])))
        if f not in cache:
            cache[f] = read_dat(f)
        for P in cache[f]:
            X, C = _tess(P, n)
            X = X + o
            v = np.full(len(X), np.nan)
            ext = np.ptp(X, 0)
            if ext.min() < 1e-9:                      # planar: a weld / joint patch holds its own Gauss points
                lo_, hi_ = X.min(0) - 1e-9, X.max(0) + 1e-9
                k = tree.query_ball_point(X.mean(0), np.linalg.norm(ext) / 2 + 1e-9)
                k = [q for q in k if np.all(xs[q] >= lo_) and np.all(xs[q] <= hi_)]
                if len(k) >= 9:
                    d, j = cKDTree(xs[k]).query(X)
                    v = tn[np.asarray(k)[j]]
            C = C.copy()
            C[:, 1:] += nv
            pts.append(X), cells.append(C), val.append(v)
            nv += len(X)
    mesh = pv.PolyData(np.vstack(pts), np.vstack(cells).ravel())
    mesh["tn"] = np.concatenate(val)
    has = np.isfinite(mesh["tn"])
    cell_has = np.array([has[c[1:]].all() for c in np.vstack(cells)])
    beads = mesh.extract_cells(np.flatnonzero(~cell_has)).extract_surface()
    faces = mesh.extract_cells(np.flatnonzero(cell_has)).extract_surface()
    return beads, faces, xs, tn


def figure(manifest, tractions, out, rho=1.0, clip=3.0):
    T = np.load(tractions)
    beads, faces, _, tn = scene(manifest, T)
    lo = faces.bounds
    c = np.array([(lo[0] + lo[1]) / 2, (lo[2] + lo[3]) / 2, (lo[4] + lo[5]) / 2])
    root = np.array([0.0, rho, c[2]])
    w = 0.4
    xcut = min(np.unique(np.round(T[(T[:, 0] == 2) & (T[:, 1] == 0), 3], 6)), key=lambda v: abs(v - 0.3) if v > 0 else 9)   # joint plane by the root
    ycut = min(np.unique(np.round(T[(T[:, 0] == 2) & (T[:, 1] == 1), 4], 6)), key=lambda v: abs(v - (rho + 0.3)))
    views = [("Isometric", None, (c + [-3.2, -4.5, 3.5], c, (0, 0, 1)), 1.0),
             ("Inside the notch, looking at the root face", None, (np.array([0.0, -1.2, c[2] + 1.6]), root, (0, 0, 1)), 1.0),
             ("Section x = %.3f (A filament joints, B lateral welds), looking +x" % xcut, ("x", xcut),
              (np.array([xcut - 6, c[1], c[2] + 0.6]), np.array([xcut, c[1], c[2]]), (0, 0, 1)), 1.3),
             ("Section y = %.3f, %.2f mm ahead of the root (B joints, A lateral welds), looking -y" % (ycut, ycut - rho),
              ("y", ycut), (np.array([c[0], ycut + 6, c[2] + 0.6]), np.array([c[0], ycut, c[2]]), (0, 0, 1)), 1.3)]
    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=(2200, 1700), border=False)
    p.set_background("white")
    sbar = dict(title="weld / joint normal traction (MPa per N)", vertical=False, position_x=0.2, position_y=0.04,
                width=0.6, height=0.06, title_font_size=18, label_font_size=16, color="black", fmt="%.1f")
    for k, (title, cut, cam, zoom) in enumerate(views):
        p.subplot(k // 2, k % 2)
        B_, F_ = beads, faces
        if cut is not None:
            a, v = cut
            nrm = {"x": (-1, 0, 0), "y": (0, 1, 0)}[a]
            org = [v - 1e-4, 0, 0] if a == "x" else [0, v + 1e-4, 0]
            B_, F_ = beads.clip(normal=nrm, origin=org), faces.clip(normal=nrm, origin=org)
        p.add_mesh(B_, color="#c8c8c8", opacity=0.35 if cut is None else 0.6, show_edges=False, smooth_shading=True)
        p.add_mesh(F_, scalars="tn", cmap="coolwarm", clim=(-clip, clip), show_scalar_bar=(k == 0),
                   scalar_bar_args=sbar, interpolate_before_map=True)
        p.add_text(title, font_size=11, color="black", position="upper_left")
        p.camera_position = cam
        p.camera.zoom(zoom)
    p.add_text("SENB rho = %g mm, 0/90 ABS beads 0.4 x 0.3 mm, perfect welds; colour clipped at +-%g (peak %.1f at the "
               "weld-neck / notch-face corner)" % (rho, clip, tn.max()), font_size=10, color="black",
               position="lower_left")
    p.screenshot(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("tractions")
    ap.add_argument("out")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=3.0)
    a = ap.parse_args(argv)
    print(figure(a.manifest, a.tractions, a.out, a.rho, a.clip))


if __name__ == "__main__":
    main()
