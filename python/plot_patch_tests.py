"""Paper-style convergence plots for the interior Neumann patch tests.

Reads results/patch_tests.csv and writes results/fig_patch_tests.{pdf,png}:
  (a) unit cube, watertight vs. imperfect (cf. Taus et al. 2019, Fig. 3)
  (b) curved exact geometries with App. B partitioning (cf. Fig. 2)
  (c) partitioning on vs. off (cf. Figs. 6-7)
"""

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# Categorical slots 1-4 of the dataviz reference palette, plus distinct markers
# so identity never relies on colour alone.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]

NAMES = {
    "cube_p2": "Cube (watertight)",
    "cube_p2_imperfect": "Cube, one CP moved by 1e-3",
    "sphere_r0": "Sphere, 6 elements",
    "sphere_r1": "Sphere, 24 elements",
    "spheroid_k2_r0": r"Spheroid $\kappa=2$",
    "sphere_perturbed_r0": "Perturbed sphere",
}


def load():
    data = defaultdict(lambda: ([], []))
    with open(os.path.join(RES, "patch_tests.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["label"], r["partition"] == "1")
            data[key][0].append(int(r["n"]))
            data[key][1].append(max(float(r["rel_L2_err"]), 1e-17))
    return data


def panel(ax, data, series, title, legend_loc="upper right"):
    for k, item in enumerate(series):
        key, label = item[0], item[1]
        ci = item[2] if len(item) > 2 else k
        ls = item[3] if len(item) > 3 else "-"
        if key not in data:
            continue
        n, e = data[key]
        ax.semilogy(n, e, color=C[ci], marker=M[ci], ms=4.5, lw=1.6, ls=ls, label=label,
                    mfc=C[ci] if ls == "-" else "white")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("integration points per direction, $n$")
    ax.set_ylim(1e-16, 1e5)
    ax.set_yticks([10.0 ** k for k in range(-16, 1, 2)])
    ax.minorticks_off()
    ax.set_xlim(0, 25)
    ax.axhline(1e-14, color="#9a9a96", lw=0.8, ls=":")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc=legend_loc, ncol=2, columnspacing=1.0, handlelength=2.2)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    d = load()
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True, constrained_layout=True)
    panel(axs[0], d, [(("cube_p2", True), NAMES["cube_p2"]),
                      (("cube_p2_imperfect", True), NAMES["cube_p2_imperfect"])],
          "(a) Quadratic cube, 6 patches")
    panel(axs[1], d, [((g, True), NAMES[g]) for g in
                      ("sphere_r0", "spheroid_k2_r0", "sphere_perturbed_r0", "sphere_r1")],
          "(b) Curved rational quartic surfaces")
    panel(axs[2], d, [(("sphere_r0", True), "Sphere, partitioned", 0, "-"),
                      (("sphere_r0", False), "Sphere, not partitioned", 0, "--"),
                      (("spheroid_k2_r0", True), r"Spheroid $\kappa=2$, partitioned", 1, "-"),
                      (("spheroid_k2_r0", False), r"Spheroid $\kappa=2$, not partitioned", 1, "--")],
          "(c) Effect of App. B partitioning", legend_loc="upper right")
    axs[0].set_ylabel(r"$\|u_h-u\|_{L^2(\Gamma)}\,/\,\|u\|_{L^2(\Gamma)}$")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_patch_tests." + ext), dpi=200)


if __name__ == "__main__":
    main()
