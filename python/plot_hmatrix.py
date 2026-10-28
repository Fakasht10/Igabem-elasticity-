"""Paper-style plots for the hierarchical compression of H and G (results/hmatrix_tests.csv,
written by results/run_hmatrix.sh) -> results/fig_hmatrix.{pdf,png}:
  (a) storage of H + G against unknowns: dense (N x (N + N_t) doubles) and compressed, plate
      thickness t = 0.2 and the thin plate t = 0.02, with O(N^2) and O(N log N) guides
  (b) compressed / dense storage
  (c) build time of the compressed operators, and the dense assembly where it was run
  (d) error against the ACA tolerance (5.8k plate): products H u, G t and the Kirsch solution
      of the compressed system relative to the dense one
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def f(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError):
        return math.nan


def main():
    with open(os.path.join(RES, "hmatrix_tests.csv")) as fh:
        rows = list(csv.DictReader(fh))
    main_rows = [r for r in rows if not r["label"].endswith("_sweep") and abs(f(r, "eps") - 1e-6) < 1e-12]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))
    for i, t in enumerate(["0.2", "0.02"]):
        rs = sorted([r for r in main_rows if r["t"] == t], key=lambda r: f(r, "unknowns"))
        if not rs:
            continue
        N = [f(r, "unknowns") for r in rs]
        lab = "t = %s" % t
        ax[0].loglog(N, [f(r, "hmat_MB") / 1e3 for r in rs], "o-", color=C[i], label="compressed, " + lab)
        if i == 0:
            ax[0].loglog(N, [f(r, "dense_MB") / 1e3 for r in rs], "k--", label="dense (N$^2$)")
        ax[1].semilogx(N, [100 * f(r, "ratio") for r in rs], "o-", color=C[i], label=lab)
        ax[2].loglog(N, [f(r, "t_assemble") for r in rs], "o-", color=C[i], label="compressed, " + lab)
        d = [(f(r, "unknowns"), f(r, "t_assemble_dense")) for r in rs if not math.isnan(f(r, "t_assemble_dense"))]
        if d:
            ax[2].loglog(*zip(*d), "s--", color=C[i], mfc="none", label="dense, " + lab)
    rs = sorted([r for r in main_rows if r["t"] == "0.2"], key=lambda r: f(r, "unknowns"))
    if rs:
        n0, m0 = f(rs[-1], "unknowns"), f(rs[-1], "hmat_MB") / 1e3
        xs = [f(rs[0], "unknowns"), n0]
        ax[0].loglog(xs, [m0 * (x * math.log(x)) / (n0 * math.log(n0)) for x in xs], ":", color="gray",
                     label="N log N")
    ax[0].axhline(15, color="gray", lw=0.8)
    ax[0].text(ax[0].get_xlim()[0] * 1.1, 17, "RAM of this machine", fontsize=8, color="gray")
    ax[0].set_xlabel("unknowns N")
    ax[0].set_ylabel("storage of H + G [GB]")
    ax[0].set_title("(a) memory")
    ax[1].set_xlabel("unknowns N")
    ax[1].set_ylabel("compressed / dense [%]")
    ax[1].set_ylim(0, 105)
    ax[1].set_title("(b) compression ratio")
    ax[2].set_xlabel("unknowns N")
    ax[2].set_ylabel("build time [s] (4 cores)")
    ax[2].set_title("(c) assembly time")
    sw = [r for r in rows if r["label"].startswith("p4_t0.2_r3") and not math.isnan(f(r, "err_Hx"))]
    sw = sorted(sw, key=lambda r: f(r, "eps"))
    if sw:
        e = [f(r, "eps") for r in sw]
        ax[3].loglog(e, [f(r, "err_Hx") for r in sw], "o-", color=C[0], label="|H$_h$u - Hu| / |Hu|")
        ax[3].loglog(e, [f(r, "err_Gt") for r in sw], "s-", color=C[1], label="|G$_h$t - Gt| / |Gt|")
        ax[3].loglog(e, [f(r, "du_rel") for r in sw], "^-", color=C[2], label="Kirsch u: |u$_h$ - u| / |u|")
        ax[3].loglog(e, [f(r, "err_kirsch_dense") for r in sw], "k--", label="discretization error (dense)")
        ax[3].loglog(e, e, ":", color="gray", label="ACA tolerance")
        ax[3].set_xlabel("ACA tolerance")
        ax[3].set_ylabel("relative error")
        ax[3].set_title("(d) accuracy, 5.8k unknowns")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_hmatrix." + ext), dpi=150)


if __name__ == "__main__":
    main()
