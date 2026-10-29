"""Figure of STEP -> face-tagged triangle meshes (python/geomio): one panel per part with every CAD face in its own
colour (the id that boundary conditions select), and the volume error against the CAD volume as the curvature
sizing is refined (expected second order: chord error ~ R pi^2 / (2 n^2)).

usage: python3 python/plot_geomio.py out.png [csv]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geomio  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
STEP = os.path.join(HERE, "..", "geo", "step")
PARTS = [("bracket", "bracket.step", 28, -58), ("bolt (OCCT screw)", "occt_screw.step", 18, -60),
         ("plate with hole", "plate_hole_t0.2.step", 35, -60)]
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def panel(ax, m, title, elev, azim):
    cmap = plt.get_cmap("tab20")
    faces = np.unique(m.fid)
    rgb = np.array([cmap(np.searchsorted(faces, f) % 20)[:3] for f in m.fid])
    k = 0.45 + 0.55 * np.abs(m.normals() @ LIGHT)
    ax.add_collection3d(Poly3DCollection(m.V[m.T], facecolors=rgb * k[:, None], edgecolors=(0, 0, 0, 0.25),
                                         linewidths=0.15))
    lo, hi = m.V.min(0), m.V.max(0)
    c, r = (lo + hi) / 2, 0.36 * (hi - lo).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev, azim); ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main(out, csv=None):
    fig = plt.figure(figsize=(11, 3.2))
    rows = []
    for k, (title, f, el, az) in enumerate(PARTS):
        m, info = geomio.cad_to_mesh(os.path.join(STEP, f), curvature=16)
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        panel(ax, m, "%s\n%d faces, %d triangles, %.2f s" % (title, info["faces"], info["triangles"],
                                                             info["t_total"]), el, az)
    ax = fig.add_subplot(1, 4, 4)
    ns = [8, 12, 16, 24, 32, 48, 64]
    for (title, f, _, _), mk in zip(PARTS, "osd"):
        e = []
        for n in ns:
            m, info = geomio.cad_to_mesh(os.path.join(STEP, f), curvature=n)
            e.append(info["vol_rel_err"])
            rows.append((f, n, info["triangles"], info["faces"], info["watertight"], info["vol_rel_err"],
                         info["t_total"]))
        ax.loglog(ns, e, mk + "-", ms=4, lw=1, label=title.split(" (")[0])
    ax.loglog(ns, 0.2 * np.array(ns, float) ** -2, "k--", lw=0.8, label=r"$n^{-2}$")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_xlabel("elements per $2\\pi$ of curvature, $n$")
    ax.set_ylabel(r"$|V_h - V_{CAD}| / V_{CAD}$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    if csv:
        with open(csv, "w") as fh:
            fh.write("part,curvature,triangles,faces,watertight,vol_rel_err,seconds\n")
            for r in rows:
                fh.write("%s,%d,%d,%d,%d,%.6e,%.3f\n" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))


if __name__ == "__main__":
    main(*sys.argv[1:])
