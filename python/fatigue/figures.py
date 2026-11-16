"""Setup figure of a dataset's notched specimens: one row per notch.
    (a) the specimen, its modelled quarter, the boundary conditions and the dimensions;
    (b) the IgBEM quarter mesh (patches and knot lines) as solved;
    (c) the mesh at the notch root (top face, in plane), with the root radius.

usage (from python/): python3 -m fatigue.figures [--data fatigue/data/ezeh_susmel.json] [--h 1.5]
       python3 -m fatigue.figures --convergence 1.5 1.25 1.0 0.75     (mesh study of Kt and the TCD stress)
"""
import argparse
import json
import os
import sys

import matplotlib
import matplotlib.collections
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from failure.igabem_backend import igabem_geometry  # noqa: E402
from fatigue import blocks  # noqa: E402
from make_geometries import read_dat  # noqa: E402
from spline_checks import evaluate  # noqa: E402

from .fields import WORK  # noqa: E402
from .specimens import Specimen, outline, quarter_step  # noqa: E402
from .transfer import HERE, OUT  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
LIGHT = np.array([0.35, -0.45, 0.82]) / np.linalg.norm([0.35, -0.45, 0.82])


def shade(tris, rgb):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-300
    return np.clip(np.asarray(rgb)[None, :3] * (0.45 + 0.55 * np.abs(n @ LIGHT))[:, None], 0, 1)


def patch_mesh(P, per_span=2):
    """Triangles, knot lines and border of one patch (as plot_step_auto, without its CAD imports)."""
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


def setup(ax, X, elev, azim):
    lo, hi = X.min(0), X.max(0)
    c = (lo + hi) / 2
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(c[2] - 1, c[2] + 1)
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-9))); ax.view_init(elev, azim); ax.set_axis_off()


def _poly(s, n=40):
    pts = []
    for seg in outline(s):
        if seg[0] == "line":
            pts.append(seg[1])
        else:
            (x0, y0), (cx, cy), (x1, y1) = seg[1], seg[2], seg[3]
            a0, a1 = np.arctan2(y0 - cy, x0 - cx), np.arctan2(y1 - cy, x1 - cx)
            if a1 - a0 > np.pi:
                a1 -= 2 * np.pi
            if a0 - a1 > np.pi:
                a1 += 2 * np.pi
            r = np.hypot(x0 - cx, y0 - cy)
            pts += [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a1, n)[:-1]]
    return np.array(pts)


def _full(q):
    """Quarter outline mirrored to the whole specimen (four copies, drawn as separate polygons)."""
    return [q * np.array(m) for m in ((1, 1), (-1, 1), (1, -1), (-1, -1))]


def specimen_panel(ax, s, name):
    q = _poly(s)
    for k, P in enumerate(_full(q)):
        ax.add_patch(Polygon(P, closed=True, fc="#c9d7ea" if k == 0 else "#eeeeee", ec="k", lw=0.6))
    W, L = s.gross_width, s.L
    # symmetry: rollers on x = 0 and y = 0 of the quarter
    for t in np.linspace(0.12, 0.88, 6):
        ax.plot(0, t * L / 2, marker=">", ms=3.0, color="#2a6", mec="none")
    xs = np.linspace(0.08, 0.92, 5) * (s.WN / 2 if s.shape == "edge" else (W / 2 - s.rho))
    xs = xs + (0 if s.shape == "edge" else s.rho)
    for x in xs:
        ax.plot(x, 0, marker="^", ms=3.0, color="#2a6", mec="none")
    # grip traction
    for x in np.linspace(-W / 2, W / 2, 7)[1:-1]:
        for sg in (1, -1):
            ax.annotate("", xy=(x, sg * (L / 2 + 0.16 * W)), xytext=(x, sg * L / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#c33"))
    ax.text(0, L / 2 + 0.22 * W, r"$\sigma_g = \sigma_n\,W_N/W_G$", ha="center", va="bottom", color="#c33")
    ax.text(W / 2 + 0.06 * W, 0, f"$W_G$ = {W:g}\n$W_N$ = {s.net_width:g}\nt = {s.T:g}\nL = {L:g}\n"
            + (rf"$\rho$ = {s.rho:g}" + (rf", {s.alpha_deg:g}°" if s.alpha_deg else " (U)")), va="center")
    ax.set_xlim(-W / 2 - 0.1 * W, W / 2 + 0.75 * W)
    ax.set_ylim(-L / 2 - 0.35 * W, L / 2 + 0.45 * W)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{name}: quarter (blue), symmetry x = 0, y = 0 (green),\nfree faces z = 0, t; dimensions in mm",
                 fontsize=8)


def mesh_panels(ax3, ax2, dat, s):
    rng = np.random.default_rng(3)
    P = read_dat(dat)
    allX, el, bd, el2 = [], [], [], []
    for Q in P:
        t, lines, border = patch_mesh(Q, per_span=2)
        col = 0.55 + 0.4 * rng.random(3)
        ax3.add_collection3d(Poly3DCollection(t, facecolors=shade(t, col), edgecolor="none", lw=0, antialiased=False))
        el += lines
        bd += border
        allX.append(t.reshape(-1, 3))
        top = np.allclose(t[..., 2], s.T, atol=1e-6)
        if top:
            ax2.add_collection(matplotlib.collections.PolyCollection(t[..., :2], facecolors=[col],
                                                                     edgecolors="none"))
            el2 += [ln[:, :2] for ln in lines] + [b[:, :2] for b in border]
    ax3.add_collection3d(Line3DCollection(el, colors="0.3", linewidths=0.12))
    ax3.add_collection3d(Line3DCollection(bd, colors="k", linewidths=0.35))
    setup(ax3, np.concatenate(allX), 28, -58)
    nel = sum((len(np.unique(Q.U)) - 1) * (len(np.unique(Q.V)) - 1) for Q in P)
    ax3.set_title(f"IgBEM quarter: {len(P)} patches, {nel} elements, p = {P[0].p}", fontsize=8)
    ax2.add_collection(matplotlib.collections.LineCollection(el2, colors="k", linewidths=0.3))
    (rx, ry, _), sgn = s.root
    r = max(1.5 * s.rho, 0.5)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.set_xlim(rx - 1.5 * r, rx + 1.5 * r)
    ax2.set_ylim(-0.1 * r, 2.4 * r)
    ax2.plot(rx, ry, "o", ms=3, color="#c33")
    ax2.set_title(rf"root, top face (root marked), $\rho$ = {s.rho:g} mm", fontsize=8)
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "ezeh_susmel.json"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--convergence", type=int, nargs="*", metavar="C",
                    help="nested coarsening levels of the graded block mesh, coarse to fine (default 4 3 2 1 0; level 0, "
                         "the default mesh, is the reference); line fields only")
    a = ap.parse_args(argv)
    if a.convergence is not None:
        print(convergence(a.data, a.convergence or [4, 3, 2, 1, 0], a.out, h=a.h)[0])
        return
    data = json.load(open(a.data))
    specs = {k: Specimen.from_dict(v) for k, v in data["specimens"].items()}
    fig = plt.figure(figsize=(9.6, 3.3 * len(specs)))
    for i, (name, s) in enumerate(specs.items()):
        step = os.path.join(WORK, f"{s.tag}.step")
        if not os.path.exists(step):
            os.makedirs(WORK, exist_ok=True)
            quarter_step(s, step)
        if s.shape == "edge":
            dat = os.path.join(WORK, f"{s.tag}_b{a.h:g}.dat")
            blocks.write(s, dat, h=a.h)
        else:
            dat, _ = igabem_geometry(step, WORK, h=a.h, curvature=12, degree=2, log=lambda *_: None)
        specimen_panel(fig.add_subplot(len(specs), 3, 3 * i + 1), s, name)
        mesh_panels(fig.add_subplot(len(specs), 3, 3 * i + 2, projection="3d"),
                    fig.add_subplot(len(specs), 3, 3 * i + 3), dat, s)
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"fig_specimens_{data['id']}_h{a.h:g}")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(out + ".png")



def convergence(data_path, levels, out_dir=OUT, h=1.5):
    """Mesh convergence of the line quantities the criteria use: root Kt and the TCD point stress at L/2, against
    the finest level, for every notch; plus the sharp notch's bisector profile per level. The levels are nested:
    level c keeps every 2^c-th knot of the graded default mesh (level 0, the reference)."""
    from .criteria import line_profile
    from .fields import field
    data = json.load(open(data_path))
    tcd = data.get("tcd_reference")
    L = tcd["A_mm"] * data["N_ref"] ** tcd["b"] if tcd else 0.5
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.6, 3.0))
    rows = {}
    colours = {"blunt": "#1f77b4", "intermediate": "#2ca02c", "sharp": "#d62728"}
    for name, v in data["specimens"].items():
        s = Specimen.from_dict(v)
        F = []
        for c in levels:
            f = field(s, h=h, volume=False, hmat=True, coarsen=c, log=lambda *_: None)
            d, sl = line_profile(f)
            F.append((c, int(f["unknowns"]), float(np.asarray(f["s1_line"])[:, 0].mean()), float(np.interp(L / 2, d, sl)),
                      d, sl))
        ref = F[-1]
        rows[name] = [{"coarsen": c, "unknowns": n, "Kt": kt, "s_L2": sp} for c, n, kt, sp, *_ in F]
        n = np.array([x[1] for x in F[:-1]])
        a1.loglog(n, [abs(x[2] / ref[2] - 1) for x in F[:-1]], "o-", color=colours.get(name), label=name, ms=4)
        a2.loglog(n, [abs(x[3] / ref[3] - 1) for x in F[:-1]], "o-", color=colours.get(name), label=name, ms=4)
        if name == "sharp":
            for (c, nn, kt, sp, d, sl), ls, lc in zip(F, (":", "-.", "--", "-", "-"), ("0.55", "0.45", "0.35", "0.6", "k")):
                a3.semilogx(np.maximum(d, 1e-3), sl, ls, color=lc, lw=0.8 if c else 1.3,
                            label=("graded default" if c == 0 else f"every {2 ** c}{'nd' if c == 1 else 'th'} knot") + f" ({nn / 1000:.1f}k)")
            a3.axvline(L / 2, color="0.5", lw=0.6)
            a3.text(L / 2, a3.get_ylim()[1] * 0.95, " L/2", color="0.4", va="top")
    for ax, t in ((a1, r"root $K_t$"), (a2, r"$\sigma_1(L/2)$, TCD point")):
        ax.set_xlabel("unknowns")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([500, 1000, 2000, 5000, 10000, 20000]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylabel("|relative change| vs the graded default")
        ax.set_title(t + " (thickness mean)", fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
        ax.legend(fontsize=7, frameon=False)
    a3.set_xlim(1e-3, 3)
    a3.set_xlabel("distance from the root (mm)")
    a3.set_ylabel(r"$\sigma_1 / \sigma_n$")
    a3.set_title("sharp V (r 0.15): bisector profile per mesh", fontsize=8)
    a3.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, f"fig_convergence_{data['id']}")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    json.dump(rows, open(out + ".json", "w"), indent=1)
    return out + ".png", rows


if __name__ == "__main__":
    main()
