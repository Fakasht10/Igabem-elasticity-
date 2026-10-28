"""Paper-style plots for the interior Dirichlet (HSBIE) tests.

Reads results/dirichlet_tests.csv and results/dirichlet_interp.csv and writes
results/fig_dirichlet_tests.{pdf,png}:
  (a) linear patch test u = alpha x (cf. Taus et al. 2019, Fig. 5)
  (b) Kelvin point-load solution: traction error against n for 6 to 384 elements
  (c) the same at n = 10 against element size, with the interpolation error
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]


def load():
    d = defaultdict(lambda: ([], []))
    with open(os.path.join(RES, "dirichlet_tests.csv")) as f:
        for r in csv.DictReader(f):
            d[r["label"]][0].append(int(r["n"]))
            d[r["label"]][1].append(max(float(r["rel_L2_err"]), 1e-17))
    interp = []
    with open(os.path.join(RES, "dirichlet_interp.csv")) as f:
        for r in csv.DictReader(f):
            interp.append((int(r["elements"]), float(r["t_interp_err"])))
    return d, interp


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    d, interp = load()
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    ylab = r"$\|t_h-t\|_{L^2(\Gamma)}\,/\,\|t\|_{L^2(\Gamma)}$"

    ax = axs[0]
    for k, (lab, name) in enumerate([("cube_p2", "Cube (quadratic)"), ("sphere_r0", "Sphere"),
                                     ("spheroid_k2_r0", r"Spheroid $\kappa=2$")]):
        if lab in d:
            ax.semilogy(*d[lab], color=C[k], marker=M[k], ms=4.5, lw=1.6, label=name)
    ax.set_ylim(1e-16, 1e5)
    ax.set_yticks([10.0 ** k for k in range(-16, 1, 2)])
    ax.minorticks_off()
    ax.set_xlabel("integration points per direction, $n$")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    style(ax, r"(a) Linear patch test, $u=\alpha x$")

    ax = axs[1]
    elems = {0: 6, 1: 24, 2: 96, 3: 384}
    for k, r in enumerate(range(4)):
        lab = "src_sphere_r%d" % r
        if lab in d:
            ax.semilogy(*d[lab], color=C[k], marker=M[k], ms=4.5, lw=1.6, label="%d elements" % elems[r])
    ax.set_xlabel("integration points per direction, $n$")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(1e-6, 1e1)
    style(ax, "(b) Point-load solution: quadrature")

    ax = axs[2]
    h, eb, ei = [], [], []
    for k, r in enumerate(range(4)):
        lab = "src_sphere_r%d" % r
        if lab in d and 10 in d[lab][0]:
            h.append(1.0 / 2 ** r)
            eb.append(d[lab][1][d[lab][0].index(10)])
            ei.append(interp[r][1])
    ax.loglog(h, eb, color=C[0], marker="o", ms=5, lw=1.6, label="HSBIE, $n=10$")
    ax.loglog(h, ei, color="#7a7974", marker="s", mfc="white", ms=5, lw=1.2, ls="--",
              label="traction interpolation")
    if len(h) >= 2:
        x0, x1 = h[-2], h[-1]
        y1 = eb[-1] * 0.4
        ax.loglog([x0, x1], [y1 * (x0 / x1) ** 5, y1], color="#3d3d3a", lw=0.9)
        ax.text((x0 * x1) ** 0.5 * 1.15, y1 * (x0 / x1) ** 2.5 * 0.5, "slope 5", fontsize=8,
                color="#3d3d3a", ha="right")
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.set_xticks(h, ["1", "1/2", "1/4", "1/8"][: len(h)])
    ax.set_xlabel("element size $h/h_0$ (6, 24, 96, 384 elements)")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style(ax, "(c) Point-load solution: h-refinement")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_dirichlet_tests." + ext), dpi=200)


if __name__ == "__main__":
    main()
