"""Paper style (Ezeh-Susmel 2020 Fig. 10): sigma_eff,max/sigma_UTS, no mean-stress correction, one panel, their markers,
their lines as drawn. Left: authors' points (extracted from the PDF). Right: ours, same specimens and same L_M(N)."""
import json, os, sys, numpy as np

from fatigue import calib as c
from fatigue.hsv import load
from fatigue.transfer import OUT
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm"})
g, plain, spec = load()
uts = {p: 42.9 for (p, _), q in plain.items()}   # the paper normalises by one UTS, 42.9 MPa (2020 Sec. 6)
no = {k: c.Notch(g[k]["f"], g[k]) for k in g}
th = json.load(open(os.path.join(os.path.dirname(__file__), "data", "ezeh_susmel_2020_fig10_point_method.json")))["points"]
def drawn(y0, y1, N0=149.5):
    N = np.geomspace(10, 2e6, 300); s = np.log(y1 / y0) / np.log(2e6 / N0)
    return N, np.where(N < N0, y0, y0 * (N / N0) ** s)
sty = {"blunt": dict(marker="D", mfc="white", mec="k", ms=5.5, mew=0.9), "intermediate": dict(marker="o", mfc="0.6", mec="k", ms=6, mew=0.8),
       "sharp": dict(marker="^", mfc="k", mec="k", ms=6)}
lab = {"blunt": "Blunt (U r3)", "intermediate": "Intermediate (U r1)", "sharp": "Sharp (V r0.15)"}
ours = {k: [(N, no[k].pm(16.4 * N ** -0.242) * s * 2 / (1 - R) / uts[p], ro) for kk, p, R, s, N, ro in spec if kk == k] for k in sty}
theirs = {k: [(q["x"], q["y"], q["runout"]) for q in th if q["type"] == k] for k in sty}
fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True, layout="constrained")
for ax, D, title in zip(axs, (theirs, ours), ("(a) Authors: Fig. 10 Point Method markers\nextracted from the PDF",
                                              "(b) Ours: same 165 specimens (157 + 8 run-outs)\nIgBEM stress, UTS 42.9 MPa, same $L_M = 16.4\\,N_f^{-0.242}$ mm")):
    ax.plot(*drawn(0.9965, 0.1551), "k-", lw=1.6); ax.plot(*drawn(1.5591, 0.2427), "k--", lw=1.1); ax.plot(*drawn(0.6369, 0.0991), "k--", lw=1.1)
    ax.plot(*drawn(0.5628, 0.1), color="0.6", lw=2.2)
    for k in ("blunt", "intermediate", "sharp"):
        P = np.array(D[k]); f = P[:, 2] < 0.5
        ax.plot(P[f, 0], P[f, 1], ls="", label=lab[k], **sty[k])
        for N, y, _ in P[~f]:
            ax.plot(N, y, ls="", **sty[k]); ax.annotate("", xy=(N * 1.7, y * 1.3), xytext=(N, y), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(14, 1.70, "$P_S$=10%", fontsize=9); ax.text(14, 1.08, "$P_S$=50%", fontsize=9); ax.text(14, 0.69, "$P_S$=90%", fontsize=9)
    ax.text(25, 0.042, "Plain $P_{50}$: 0.16 at $2\\cdot10^6$, k 5.1, $T_\\sigma$ 2.451 (lines as drawn in the PDF)\n"
            "Grey: unifying design curve 0.1 $\\sigma_{UTS}$, k 5.5\nR = -1 and 0, print angles 0, 30, 45 deg; arrows: run-outs", fontsize=8.8)
    ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.01, 10), xlabel="$N_f$ [cycles to failure]"); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.35, which="both"); ax.legend(loc="upper right", fontsize=9)
axs[0].set_ylabel(r"$\sigma_{eff,max}/\sigma_{UTS}$")
fig.suptitle("TCD Point Method in the paper's own variable (no mean-stress correction): authors vs ours", fontsize=12.5)
out = os.path.join(OUT, "fig_tcd_point_paper_style")
for e in ("png", "pdf"): fig.savefig(f"{out}.{e}", dpi=170)
for D, who in ((theirs, "authors"), (ours, "ours")):
    for k in sty:
        P = np.array([x for x in D[k] if not x[2]]); r = np.log10(P[:, 1] / np.minimum(0.155 * (P[:, 0] / 2e6) ** (-1 / 5.1), 1))
        print(f"{who:8s} {k:13s} n {len(P)} mean {r.mean():+.3f} in band {100*np.mean(abs(r) < np.log10(2.451)/2):.0f}%")
