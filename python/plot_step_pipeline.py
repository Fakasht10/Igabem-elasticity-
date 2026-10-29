"""Figures for the STEP -> .dat pipeline (python/step2dat.py).

fig_step_plate:  (a) the CAD solid read from STEP (7 trimmed B-rep faces), (b) the 16
                 conforming patches after splitting and untrimming, element lines at r = 1.
fig_step_census: public OCCT parts, faces coloured by whether they are already four-sided
                 (convertible as is) or need a quad layout first.

usage: python3 python/plot_step_pipeline.py plate.step patches.dat out_dir [part.step ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import read_dat  # noqa: E402
from step2dat import evaluate, face_census, faces, is_quad, read_step  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
LIGHT = np.array([0.4, -0.5, 0.75]) / np.linalg.norm([0.4, -0.5, 0.75])


def shade(tris, rgb):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    k = 0.45 + 0.55 * np.abs(n @ LIGHT)
    return np.clip(np.asarray(rgb)[None, :3] * k[:, None], 0, 1)


def face_triangles(shape, defl):
    BRepMesh_IncrementalMesh(shape, defl, False, 0.3, True)
    out = []
    for f in faces(shape):
        loc = TopLoc_Location()
        T = BRep_Tool.Triangulation_s(f, loc)
        X = np.array([[T.Node(i).Transformed(loc.Transformation()).Coord()] for i in range(1, T.NbNodes() + 1)])[:, 0]
        I = np.array([T.Triangle(i).Get() for i in range(1, T.NbTriangles() + 1)]) - 1
        out.append((f, X[I]))
    return out


def setup(ax, X, elev=28, azim=-58):
    lo, hi = X.min(0), X.max(0)
    c, r = (lo + hi) / 2, 0.42 * (hi - lo).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()


def patch_grid(P, m=12):
    ku, kv = np.unique(P.U), np.unique(P.V)
    us = np.unique(np.concatenate([np.linspace(a, b, m // len(ku) + 3) for a, b in zip(ku[:-1], ku[1:])]))
    vs = np.unique(np.concatenate([np.linspace(a, b, m // len(kv) + 3) for a, b in zip(kv[:-1], kv[1:])]))
    X = np.array([[evaluate(P, u, v)[0] for u in us] for v in vs])
    lines = [np.array([evaluate(P, u, v)[0] for v in np.linspace(0, 1, 30)]) for u in ku] + \
            [np.array([evaluate(P, u, v)[0] for u in np.linspace(0, 1, 30)]) for v in kv]
    return X, lines


def fig_plate(step, dat, out):
    cmap = plt.get_cmap("tab20")
    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    tri = face_triangles(read_step(step), 2e-3)
    allX = np.concatenate([t.reshape(-1, 3) for _, t in tri])
    for k, (f, t) in enumerate(tri):
        ax.add_collection3d(Poly3DCollection(t, facecolors=shade(t, cmap(2 * k % 20)), edgecolor="none", linewidth=0, antialiased=False))
    setup(ax, allX)
    ax.set_title("(a) STEP B-rep: %d trimmed faces" % len(tri))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    P = read_dat(dat)
    for k, Q in enumerate(P):
        X, lines = patch_grid(Q)
        t = np.concatenate([np.stack([X[:-1, :-1], X[1:, :-1], X[1:, 1:]], 2).reshape(-1, 3, 3),
                            np.stack([X[:-1, :-1], X[1:, 1:], X[:-1, 1:]], 2).reshape(-1, 3, 3)])
        ax.add_collection3d(Poly3DCollection(t, facecolors=shade(t, cmap(k % 20)), edgecolor="none", linewidth=0, antialiased=False))
        ax.add_collection3d(Line3DCollection(lines, colors="k", linewidths=0.35))
    setup(ax, allX)
    ax.set_title("(b) %d untrimmed conforming patches (r = 1)" % len(P))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "fig_step_plate." + ext), dpi=220, bbox_inches="tight")


def fig_census(parts, out):
    fig = plt.figure(figsize=(3.3 * len(parts), 3.6))
    for j, step in enumerate(parts):
        ax = fig.add_subplot(1, len(parts), j + 1, projection="3d")
        sh = read_step(step)
        tri = face_triangles(sh, 0.05 * 1.0)
        allX = np.concatenate([t.reshape(-1, 3) for _, t in tri])
        tri = face_triangles(sh, 2e-3 * np.ptp(allX, 0).max())
        for f, t in tri:
            col = (0.30, 0.65, 0.35) if is_quad(f) else (0.85, 0.33, 0.25)
            ax.add_collection3d(Poly3DCollection(t, facecolors=shade(t, col), edgecolor="none", linewidth=0, antialiased=False))
        setup(ax, allX, 22, -50)
        nf, nq = face_census(sh)
        ax.set_title("%s\n%d of %d faces four-sided" % (os.path.basename(step), nq, nf))
    fig.text(0.5, 0.06, "green: four-sided, one wire (Coons-untrimmable as is);  red: needs a quad layout first",
             ha="center")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "fig_step_census." + ext), dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    fig_plate(sys.argv[1], sys.argv[2], sys.argv[3])
    if len(sys.argv) > 4:
        fig_census(sys.argv[4:], sys.argv[3])
