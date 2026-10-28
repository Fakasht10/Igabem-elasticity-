"""Paper-style plots for the plate with a circular through-hole (traction-only SBIE).

Reads results/plate_tests.csv and results/plate_profile_*.csv and writes
results/fig_plate_tests.{pdf,png}:
  (a) Kirsch field: displacement error against unknowns, p = 2 and p = 4
  (b) the same error against wall time (assembly of H and G + LU + solve)
  (c) error of the stress concentration factor sigma_xx(0, a, 0) against 3
  (d) sigma_xx through the thickness at the hole edge, free faces (thickness effect)
"""

import csv
import math
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]


def load():
    d = OrderedDict()  # (label, case, data) -> row
    with open(os.path.join(RES, "plate_tests.csv")) as f:
        for r in csv.DictReader(f):
            d[(r["label"], r["case"], r["data"])] = r
    return d


def series(d, p, case, t="0.2", data="interp"):
    rows = [d.get(("plate_p%d_t%s_r%d" % (p, t, r), case, data)) for r in range(4)]
    return [r for r in rows if r is not None]


def profile(label, case):
    with open(os.path.join(RES, "plate_profile_%s_%s.csv" % (label, case))) as f:
        rows = list(csv.DictReader(f))
    return [float(x["z_over_t"]) for x in rows], [float(x["sxx"]) for x in rows]


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    if ax.get_xscale() == "log":
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def slope(ax, x0, x1, y0, k, text):
    ax.plot([x0, x1], [y0, y0 * (x1 / x0) ** k], color="0.45", lw=0.9, ls="--")
    ax.text(x1 * 1.05, y0 * (x1 / x0) ** k, text, fontsize=8, color="0.35", va="center")


def wall(r):
    return float(r["t_assemble"]) + float(r["t_factor"]) + float(r["t_solve"])


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    d = load()
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.8), constrained_layout=True)
    ylab = r"$\|u_h-u\|_{L^2(\Gamma)}\,/\,\|u\|_{L^2(\Gamma)}$"

    ax = axs[0, 0]
    for i, p in enumerate((2, 4)):
        s = series(d, p, "kirsch")
        N = [int(r["unknowns"]) for r in s]
        ax.loglog(N, [float(r["rel_L2_err"]) for r in s], marker=M[i], color=C[i], ms=4.5, lw=1.4,
                  label=r"$p = %d$" % p)
    s = series(d, 2, "kirsch", data="exact")
    ax.loglog([int(r["unknowns"]) for r in s], [float(r["rel_L2_err"]) for r in s], marker="x",
              color=C[0], ls="none", ms=6, label=r"$p = 2$, data integrated exactly")
    # unknowns ~ h^-2 on a surface: O(h^3) -> N^-3/2, O(h^5) -> N^-5/2
    slope(ax, 1.5e3, 4e3, 3e-2, -1.5, r"$h^{3}$")
    slope(ax, 2.2e3, 5e3, 2.5e-4, -2.5, r"$h^{5}$")
    ax.set_xlabel("unknowns $3M$")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "(a) Kirsch field, $t/a = 1$: error against unknowns")

    ax = axs[0, 1]
    for i, p in enumerate((2, 4)):
        s = series(d, p, "kirsch")
        T = [wall(r) for r in s]
        ax.loglog(T, [float(r["rel_L2_err"]) for r in s], marker=M[i], color=C[i], ms=4.5, lw=1.4,
                  label=r"$p = %d$" % p)
        for r, x in zip(s, T):
            ax.annotate("%s" % r["unknowns"], (x, float(r["rel_L2_err"])), textcoords="offset points",
                        xytext=(5, 3), fontsize=7, color=C[i])
    ax.set_xlabel("wall time [s]: assembly of $H, G$ + LU + solve (4 cores)")
    ax.set_ylabel(ylab)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, "(b) error against time (labels: unknowns)")

    ax = axs[1, 0]
    for i, p in enumerate((2, 4)):
        s = series(d, p, "kirsch")
        ax.loglog([int(r["unknowns"]) for r in s], [abs(float(r["sxx_mid"]) - 3.0) / 3.0 for r in s],
                  marker=M[i], color=C[i], ms=4.5, lw=1.4, label=r"$p = %d$, mid-plane" % p)
        ax.loglog([int(r["unknowns"]) for r in s], [abs(float(r["sxx_face"]) - 3.0) / 3.0 for r in s],
                  marker=M[i], color=C[i], ms=4.5, lw=1.0, ls=":", mfc="white", label=r"$p = %d$, face" % p)
    ax.axhline(1e-3, color="0.6", lw=0.8, ls="--")
    ax.text(200, 1.15e-3, "$10^{-3}$", fontsize=8, color="0.4")
    ax.set_xlabel("unknowns $3M$")
    ax.set_ylabel(r"$|K_t - 3|\,/\,3$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, r"(c) stress concentration $\sigma_{xx}(0, a, z)$, Kirsch")

    ax = axs[1, 1]
    for i, t in enumerate(("0.1", "0.2", "0.4")):
        z, s = profile("p4_t%s_r3" % t, "kirsch_free")
        ax.plot(z, s, color=C[i], lw=1.5, label=r"$t/a = %g$" % (float(t) / 0.2))
    z, s = profile("p4_t0.2_r2", "kirsch_free")
    ax.plot(z, s, color=C[1], lw=0.9, ls="--", label=r"$t/a = 1$, coarser mesh")
    ax.axhline(3.0, color="0.3", lw=0.9, ls=":", label="Kirsch (plane), 3")
    ax.set_xlabel(r"$z/t$")
    ax.set_ylabel(r"$\sigma_{xx}(0, a, z)\,/\,S$")
    ax.set_xlim(-0.5, 0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    style(ax, "(d) thickness effect, free faces, $p = 4$")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_plate_tests." + ext), dpi=200)


if __name__ == "__main__":
    main()
