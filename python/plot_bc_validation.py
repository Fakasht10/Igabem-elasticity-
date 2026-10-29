"""Paper-style plots for local supports and loads (BcModel, app/bc_validate, app/bc_case).

Reads results/bc_*.csv and the bracket surface .vtu files and writes
results/fig_bc_validation.{pdf,png}:
  (a) half-space, pressure p on a disk the mesh does not follow: sigma_zz and sigma_rr
      under the load against the exact (Love) field, coarsest and finest mesh
  (b) the same case: boundary displacement error and max stress error on that line
      against unknowns
  (c) clamped cantilever (L/h = 10), tip shear: deflection against Timoshenko and
      Euler-Bernoulli
  (d) clamp shear reaction / applied force against unknowns: cantilever, thin plate
      (t = 0.2) uniform and graded towards the clamp, and the bracket from STEP
and results/fig_bc_bracket.{pdf,png}: von Mises on the bracket surface for four load
cases (clamped end, loads placed by CAD face id or by a box through the patches).
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from render_vtu import SEQ, field, read_vtu

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]


def rows(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def col(r, k):
    return np.array([float(x[k]) for x in r])


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def slope(ax, x0, x1, y0, k, text):
    ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** k], color="0.45", lw=0.9, ls="--")
    ax.text(x1 * 1.05, y0 * (x1 / x0) ** k, text, fontsize=8, color="0.35", va="center")


def reaction(csvname):
    b = [r for r in rows(csvname) if r["case"] == "bend"]
    load = sum(float(r["Fz"]) for r in b if r["kind"] == "load")
    react = sum(float(r["Fz"]) for r in b if r["kind"] == "reaction")
    return react / -load


def validation():
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.8), constrained_layout=True)

    ax = axs[0, 0]
    for i, r in enumerate((0, 2)):
        p = rows("bc_halfspace_probe_r%d.csv" % r)
        z = col(p, "zeta")
        ax.plot(z, col(p, "szz"), color=C[0], lw=1.4 if r else 0, marker="" if r else "o", ms=3.5,
                mfc="white", label=r"$\sigma_{zz}$, IgBEM %s" % ("r2" if r else "r0"))
        ax.plot(z, col(p, "srr"), color=C[1], lw=1.4 if r else 0, marker="" if r else "s", ms=3.5,
                mfc="white", label=r"$\sigma_{rr}$, IgBEM %s" % ("r2" if r else "r0"))
    p = rows("bc_halfspace_probe_r2.csv")
    ax.plot(col(p, "zeta"), col(p, "szz_exact"), color="0.15", lw=0.9, ls="--", label="exact (Love)")
    ax.plot(col(p, "zeta"), col(p, "srr_axis"), color="0.15", lw=0.9, ls="--")
    ax.set_xlabel(r"depth $\zeta / a$")
    ax.set_ylabel(r"$\sigma / p$")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, "(a) half-space, disk load: stress under the load")

    ax = axs[0, 1]
    h = rows("bc_halfspace.csv")
    N = col(h, "unknowns")
    ax.loglog(N, col(h, "u_l2_err"), marker=M[0], color=C[0], ms=4.5, lw=1.4,
              label=r"$\|u_h-u\|_{L^2(\Gamma)}/\|u\|$")
    ax.loglog(N, col(h, "sigma_line_max_err"), marker=M[1], color=C[1], ms=4.5, lw=1.4,
              label=r"$\max|\sigma_h-\sigma|/p$ under the load")
    slope(ax, N[0], N[-1], 0.6 * col(h, "u_l2_err")[0], -0.75, r"$h^{1.5}$")
    ax.set_xlabel("unknowns")
    ax.set_ylabel("relative error")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style(ax, "(b) half-space: convergence")

    ax = axs[1, 0]
    c = rows("bc_cantilever_probe_n40.csv")
    x = col(c, "x")
    ax.plot(x, -col(c, "w"), color=C[0], lw=0, marker="o", ms=3.5, mfc="white", label="IgBEM, 3246 unknowns")
    ax.plot(x, -col(c, "w_timoshenko"), color="0.15", lw=0.9, ls="--", label="Timoshenko")
    ax.plot(x, -col(c, "w_euler"), color="0.55", lw=0.9, ls=":", label="Euler-Bernoulli")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"deflection $-w$ ($F = 1$, $E = 1$, $h = 1$)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(ax, "(c) clamped cantilever, tip shear")

    ax = axs[1, 1]
    cant = rows("bc_cantilever.csv")
    ax.semilogx(col(cant, "unknowns"), col(cant, "reaction_Fz"), marker=M[0], color=C[0], ms=4.5, lw=1.4,
                label="cantilever $L/h = 10$, uniform")
    pu = [(2430, reaction("bc_plate_clamped_uniform.csv"))]
    pg = [(1422, reaction("bc_plate_clamped_graded.csv"))]
    ax.semilogx(*zip(*pu), marker=M[1], color=C[1], ms=5, lw=0, label="plate $t = 0.2$, uniform")
    ax.semilogx(*zip(*pg), marker=M[2], color=C[2], ms=5, lw=0, label="plate $t = 0.2$, graded to clamp")
    ax.semilogx([6708], [reaction("bc_bracket_clamped.csv")], marker=M[3], color=C[3], ms=5, lw=0,
                label="bracket (STEP), not graded")
    ax.axhline(1.0, color="0.15", lw=0.9, ls="--")
    ax.set_xlim(900, 1e4)
    ax.set_ylim(0.8, 1.05)
    ax.set_xlabel("unknowns")
    ax.set_ylabel(r"clamp reaction $R_z / F$")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    style(ax, "(d) bending: clamp shear reaction")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_bc_validation." + ext), dpi=200)


def bracket():
    cases = [("pull", "pull, 10 N on face 7"), ("bend", "bend, 1 N on face 7"),
             ("press", "pad pressure in a box"), ("pin", "pin bearing on face 12")]
    fig = plt.figure(figsize=(10, 6.2))
    fig.subplots_adjust(left=0.0, right=0.95, bottom=0.02, top=0.9, wspace=0.05, hspace=0.12)
    for k, (c, t) in enumerate(cases):
        v = read_vtu(os.path.join(RES, "bc_bracket_clamped_%s_surface.vtu" % c))
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        P, Q = v["points"], v["quads"]
        f = field(v, "von_mises")[Q].mean(axis=1)
        norm = mcolors.Normalize(vmin=0, vmax=np.percentile(f, 99.5))
        ax.add_collection3d(Poly3DCollection(P[Q], facecolors=SEQ(norm(f)), edgecolors="none", linewidths=0))
        lo, hi = P.min(axis=0), P.max(axis=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.15)
        ax.view_init(elev=32, azim=-60)
        ax.set_axis_off()
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=ax, shrink=0.55, pad=0.0,
                     label="von Mises (MPa)")
        ax.set_title("(%s) %s" % ("abcd"[k], t), fontsize=10, loc="left")
    fig.suptitle("Bracket from STEP, end face 6 clamped; one LU for all cases (6708 unknowns)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_bc_bracket." + ext), dpi=170, bbox_inches="tight")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    validation()
    bracket()


if __name__ == "__main__":
    main()
