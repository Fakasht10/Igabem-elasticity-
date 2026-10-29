"""Figure of automatic STEP -> patch layouts (python/step_auto.py): one panel per part,
patches in random colours with their element (knot) lines.

usage: python3 python/plot_step_auto.py out.png title1:file1.dat[:elev:azim] [title2:file2.dat ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import read_dat  # noqa: E402
from plot_step_pipeline import setup, shade  # noqa: E402
from spline_checks import evaluate  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})


def patch_mesh(P, per_span=3):
    ku, kv = np.unique(P.U), np.unique(P.V)
    us = np.unique(np.concatenate([np.linspace(a, b, per_span + 1) for a, b in zip(ku[:-1], ku[1:])]))
    vs = np.unique(np.concatenate([np.linspace(a, b, per_span + 1) for a, b in zip(kv[:-1], kv[1:])]))
    X = np.array([[evaluate(P, u, v)[0] for u in us] for v in vs])
    iu = [np.argmin(abs(us - k)) for k in ku]
    iv = [np.argmin(abs(vs - k)) for k in kv]
    lines = [X[:, i] for i in iu] + [X[j, :] for j in iv]
    t = np.concatenate([np.stack([X[:-1, :-1], X[1:, :-1], X[1:, 1:]], 2).reshape(-1, 3, 3),
                        np.stack([X[:-1, :-1], X[1:, 1:], X[:-1, 1:]], 2).reshape(-1, 3, 3)])
    return t, lines, [X[0], X[-1], X[:, 0], X[:, -1]]


def main(out, panels):
    rng = np.random.default_rng(3)
    fig = plt.figure(figsize=(3.3 * len(panels), 3.4))
    for k, spec in enumerate(panels):
        parts = spec.split(":")
        title, dat = parts[0], parts[1]
        elev, azim = (float(parts[2]), float(parts[3])) if len(parts) > 3 else (28, -58)
        ax = fig.add_subplot(1, len(panels), k + 1, projection="3d")
        P = read_dat(dat)
        allX, el, bd = [], [], []
        for Q in P:
            t, lines, border = patch_mesh(Q)
            col = 0.55 + 0.4 * rng.random(3)
            ax.add_collection3d(Poly3DCollection(t, facecolors=shade(t, col), edgecolor="none", linewidth=0,
                                                 antialiased=False))
            el += lines
            bd += border
            allX.append(t.reshape(-1, 3))
        ax.add_collection3d(Line3DCollection(el, colors="0.35", linewidths=0.15))
        ax.add_collection3d(Line3DCollection(bd, colors="k", linewidths=0.45))
        setup(ax, np.concatenate(allX), elev, azim)
        nel = sum((len(np.unique(Q.U)) - 1) * (len(np.unique(Q.V)) - 1) for Q in P)
        ax.set_title("%s\n%d patches, %d elements, p = %d" % (title, len(P), nel, P[0].p))
    fig.text(0.5, 0.04, "black: patch boundaries (C$^0$);  grey: knot lines inside patches (C$^{p-1}$)",
             ha="center")
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.splitext(out)[0] + ext, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
