"""Static figure of the 3D fields written by src/vtk.hpp (ParaView is the
interactive route; this is the paper figure).

  render_vtu.py <prefix> --out fig.png [--field von_mises|s_xx|...] [--title ...]

Reads <prefix>_surface.vtu and <prefix>_section.vtu and draws
  (a) the whole surface coloured by the chosen field (default von Mises)
  (b) the mid-plane section, sigma_xx (diverging scale centred on 0)
  (c) the section's stress error against the exact field, when present
      (|sigma_h - sigma| / max |sigma|, log scale), otherwise sigma_yy
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# One-hue sequential (blue, light -> dark) and blue <-> red diverging with a gray midpoint.
SEQ = mcolors.LinearSegmentedColormap.from_list(
    "seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
DIV = mcolors.LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#3987e5", "#b7d3f6", "#f0efec", "#f3b9a8", "#e0603d", "#8f2a14"])
COMP = {"xx": 0, "xy": 1, "xz": 2, "yy": 4, "yz": 5, "zz": 8}


def read_vtu(path):
    txt = open(path).read()
    arrays = {}
    for m in re.finditer(r'<DataArray[^>]*Name="([^"]+)"[^>]*>(.*?)</DataArray>', txt, re.S):
        arrays[m.group(1)] = np.array(m.group(2).split(), dtype=float)
    pts = re.search(r"<Points>\s*<DataArray[^>]*>(.*?)</DataArray>", txt, re.S).group(1)
    arrays["points"] = np.array(pts.split(), dtype=float).reshape(-1, 3)
    arrays["quads"] = arrays["connectivity"].astype(int).reshape(-1, 4)
    return arrays


def field(v, name):
    n = len(v["points"])
    if name == "von_mises":
        return v["von_mises"]
    if name.startswith("s_"):
        return v["stress"].reshape(n, 9)[:, COMP[name[2:]]]
    raise ValueError(name)


def tri(v):
    q = v["quads"]
    t = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return mtri.Triangulation(v["points"][:, 0], v["points"][:, 1], t)


def label(name):
    if name == "von_mises":
        return r"$\sigma_{\mathrm{vM}}\,/\,S$"
    return r"$\sigma_{%s}\,/\,S$" % name[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="von_mises")
    ap.add_argument("--title", default="")
    o = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    surf = read_vtu(o.prefix + "_surface.vtu")
    sec = read_vtu(o.prefix + "_section.vtu")

    fig = plt.figure(figsize=(11.5, 3.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

    ax = fig.add_subplot(gs[0], projection="3d")
    P, Q = surf["points"], surf["quads"]
    f = field(surf, o.field)
    fq = f[Q].mean(axis=1)
    norm = mcolors.Normalize(vmin=fq.min(), vmax=fq.max())
    pc = Poly3DCollection(P[Q], facecolors=SEQ(norm(fq)), edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.0, label=label(o.field))
    ax.set_title("(a) surface, " + label(o.field), fontsize=10, loc="left")

    T = tri(sec)
    ext = np.ptp(sec["points"][:, 0]) / max(np.ptp(sec["points"][:, 1]), 1e-12)
    cb = dict(orientation="horizontal", shrink=0.9, pad=0.02) if ext > 2 else dict(shrink=0.85)
    ax = fig.add_subplot(gs[1])
    sxx = field(sec, "s_xx")
    m = max(abs(sxx.min()), abs(sxx.max()))
    nrm = mcolors.TwoSlopeNorm(vmin=min(sxx.min(), -1e-9), vcenter=0.0, vmax=max(sxx.max(), 1e-9))
    tc = ax.tripcolor(T, sxx, shading="gouraud", cmap=DIV, norm=nrm)
    fig.colorbar(tc, ax=ax, label=label("s_xx"), **cb)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(r"(b) mid-plane section, $\sigma_{xx}$", fontsize=10, loc="left")

    ax = fig.add_subplot(gs[2])
    if "stress_error" in sec:
        e = np.maximum(sec["stress_error"], 1e-8)
        tc = ax.tripcolor(T, e, shading="gouraud", cmap=SEQ,
                          norm=mcolors.LogNorm(vmin=max(e.min(), 1e-7), vmax=e.max()))
        fig.colorbar(tc, ax=ax, label=r"$\|\sigma_h-\sigma\|\,/\,\max\|\sigma\|$", **cb)
        ax.set_title("(c) section, stress error vs exact", fontsize=10, loc="left")
    else:
        syy = field(sec, "s_yy")
        nrm = mcolors.TwoSlopeNorm(vmin=min(syy.min(), -1e-9), vcenter=0.0, vmax=max(syy.max(), 1e-9))
        tc = ax.tripcolor(T, syy, shading="gouraud", cmap=DIV, norm=nrm)
        fig.colorbar(tc, ax=ax, label=label("s_yy"), **cb)
        ax.set_title(r"(c) mid-plane section, $\sigma_{yy}$", fontsize=10, loc="left")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    if o.title:
        fig.suptitle(o.title, fontsize=10)
    fig.savefig(o.out, dpi=200)


if __name__ == "__main__":
    main()
