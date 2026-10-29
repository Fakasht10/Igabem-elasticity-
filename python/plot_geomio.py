"""Figure of STEP -> face-tagged triangle meshes (python/geomio): one panel per part with every CAD face in its own
colour (the id that boundary conditions select), and the volume error against the CAD volume under uniform
refinement h, for flat tri3 (expected h^2) and curved tri6 with mid-edge nodes on the CAD (geomio.highorder; h^3 at
least, h^4 observed).

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
import gmsh  # noqa: E402
import geomio  # noqa: E402
from geomio.highorder import cad_to_mesh2  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
STEP = os.path.join(HERE, "..", "geo", "step")
PARTS = [("bracket", "bracket.step", 28, -58), ("bolt (OCCT screw)", "occt_screw.step", 18, -60),
         ("plate with hole", "plate_hole_t0.2.step", 35, -60)]
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def diag(f):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.occ.importShapes(os.path.join(STEP, f)); gmsh.model.occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
    finally:
        gmsh.finalize()
    return float(np.linalg.norm(bb[3:] - bb[:3]))


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
    fig = plt.figure(figsize=(12, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35])
    rows = []
    for k, (title, f, el, az) in enumerate(PARTS):
        m, info = geomio.cad_to_mesh(os.path.join(STEP, f), curvature=16)
        ax = fig.add_subplot(gs[k], projection="3d")
        panel(ax, m, "%s\n%d faces, %d triangles, %.2f s" % (title, info["faces"], info["triangles"],
                                                             info["t_total"]), el, az)
    ax = fig.add_subplot(gs[3])
    ks = [8, 12, 16, 24, 32, 48]
    for (title, f, _, _), mk, col in zip(PARTS, "osd", ("C0", "C1", "C2")):
        e3, e6, hs = [], [], []
        for k in ks:
            m, info = cad_to_mesh2(os.path.join(STEP, f), h=diag(f) / k, curvature=0, check_cad=False)
            hs.append(1.0 / k); e3.append(info["vol_rel_err_tri3"]); e6.append(info["vol_rel_err"])
            rows.append((f, k, info["tri6"], info["faces"], info["watertight"], e3[-1], e6[-1], info["t_total"]))
        lab = title.split(" (")[0]
        ax.loglog(hs, e3, mk + "--", color=col, mfc="none", ms=4, lw=0.8, label=lab + ", tri3")
        ax.loglog(hs, e6, mk + "-", color=col, ms=4, lw=1, label=lab + ", tri6")
    hh = np.array([1 / 8, 1 / 48])
    ax.loglog(hh, 2.0 * hh ** 2, "k:", lw=0.8); ax.text(hh[0] * 1.08, 2.0 * hh[0] ** 2, r"$h^2$", fontsize=8, va="center")
    ax.loglog(hh, 1.2 * hh ** 4, "k:", lw=0.8); ax.text(hh[0] * 1.08, 1.2 * hh[0] ** 4, r"$h^4$", fontsize=8, va="center")
    ax.set_xticks([1 / k for k in ks]); ax.set_xticklabels(["1/%d" % k for k in ks]); ax.minorticks_off()
    ax.set_xlim(1 / 60, 1 / 6.5)
    ax.set_xlabel(r"element size $h$ / bounding-box diagonal")
    ax.set_ylabel(r"$|V_h - V_{CAD}| / V_{CAD}$")
    ax.legend(frameon=False, fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    if csv:
        with open(csv, "w") as fh:
            fh.write("part,diag_over_h,triangles,faces,watertight,vol_rel_err_tri3,vol_rel_err_tri6,seconds\n")
            for r in rows:
                fh.write("%s,%d,%d,%d,%d,%.6e,%.6e,%.3f\n" % r)


if __name__ == "__main__":
    main(*sys.argv[1:])
