"""Scaling of the hierarchical operator (src/hmat, results/hmat_gate.csv) on the degree-4 plate with a hole:

  (a) wall time against unknowns: operator build, preconditioner (LU + coarse level) and GMRES solve, with the
      speed gate (build, solve) at 50k and 100k and bemkit's 55k run (59 s, 5 cores) for reference
  (b) peak memory and stored operator (near + far, float) against unknowns, with the memory gate
  (c) GMRES iterations and the error against the 3D Kirsch field (eps 1e-5 and 1e-6), dense LU references

usage: python3 python/plot_hmat_gate.py [results/fig_hmat_gate.png]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = ["#0969da", "#cf222e", "#1a7f37", "#8250df"]


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "fig_hmat_gate.png")
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "hmat_gate.csv"))))
    main_ = sorted([r for r in rows if r["label"].endswith("eps1e-5")], key=lambda r: int(r["unknowns"]))
    tight = [r for r in rows if r["label"].endswith("eps1e-6")]
    n = [int(r["unknowns"]) / 1e3 for r in main_]
    f = lambda k, rs=main_: [float(r[k]) for r in rs]  # noqa: E731

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
    a = ax[0]
    a.plot(n, f("t_build"), "o-", color=C[0], label="operator build")
    a.plot(n, [p + c for p, c in zip(f("t_precond"), f("t_coarse"))], "s-", color=C[2], label="preconditioner")
    a.plot(n, f("t_solve"), "^-", color=C[1], label="GMRES solve")
    for x, b, s in ((50, 60, 20), (100, 150, 45)):
        a.plot([x], [b], "_", ms=16, mew=2, color=C[0])
        a.plot([x], [s], "_", ms=16, mew=2, color=C[1])
    a.plot([55], [59], "*", ms=10, color=MUTED, label="bemkit 55k build (5 cores)")
    a.set_xlabel("unknowns (thousands)", color=INK, fontsize=9)
    a.set_ylabel("wall time (s), 4 cores", color=INK, fontsize=9)
    a.set_title("(a) time; bars = gate", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=7.5)
    style(a)

    a = ax[1]
    a.plot(n, [m / 1024 for m in f("rss_MB")], "o-", color=C[3], label="peak RSS")
    a.plot(n, [x + y for x, y in zip(f("near_GB"), f("far_GB"))], "s--", color=C[0], label="stored operator (float)")
    a.plot(n, f("near_GB"), ":", color=MUTED, label="near field")
    for x, g in ((50, 4), (100, 8)):
        a.plot([x], [g], "_", ms=16, mew=2, color=C[3])
    a.plot([55], [4.2], "*", ms=10, color=MUTED, label="bemkit 55k")
    a.set_xlabel("unknowns (thousands)", color=INK, fontsize=9)
    a.set_ylabel("GB", color=INK, fontsize=9)
    a.set_title("(b) memory; bars = gate", color=INK, fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=7.5)
    style(a)

    a = ax[2]
    # square plate (L = W) and long plate (L = 3W) separately; dense LU references from results/dense_fast.csv
    for key, mk, name in (("plate_hole_p4", "o", "square plate"), ("plate_hole_L3", "s", "long plate (L3)")):
        rs = [r for r in main_ if r["label"].startswith(key)]
        a.semilogy([int(r["unknowns"]) / 1e3 for r in rs], f("err", rs), mk + "-", color=C[0], label=name + ", eps 1e-5")
    if tight:
        a.semilogy([int(r["unknowns"]) / 1e3 for r in tight], f("err", tight), "s", color=C[2], label="long plate, eps 1e-6")
    a.semilogy([17.3, 26.0], [6.19e-7, 1.905e-7], "o", mfc="none", color=MUTED, label="dense LU (17k square, 26k long)")
    a.set_xlabel("unknowns (thousands)", color=INK, fontsize=9)
    a.set_ylabel("relative L2 displacement error", color=INK, fontsize=9)
    a.set_ylim(1e-7, 1e-4)
    a.set_title("(c) accuracy and iterations", color=INK, fontsize=10, loc="left")
    b = a.twinx()
    b.plot(n, f("iterations"), "x--", color=C[1], label="GMRES iterations")
    b.set_ylim(0, 60)
    b.set_ylabel("iterations (to 1e-8)", color=C[1], fontsize=9)
    b.tick_params(colors=C[1], labelsize=8)
    b.spines["top"].set_visible(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="upper left", ncol=2)
    style(a)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
