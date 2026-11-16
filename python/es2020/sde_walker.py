"""Walker-collapsed view of the SDE results (DEPARTURE from Ezeh & Susmel 2020, who use sigma_max only).

sigma_W = sigma_max ((1 - R)/2)^gamma; gamma fitted on the Fig. 1 plain points with R = -1 and R = 0 only
(minimum pooled residual about one Basquin line in log N). Notched specimens at net sigma_max.
    python -m es2020.sde_walker chain.npz out.png
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import paper, sde
from .sde_figs import COL, TITLE, predictive_band


def walker(s, R, g):
    return np.asarray(s) * ((1 - np.asarray(R)) / 2) ** g


def pooled_rms(N, s, R, g):
    x, y = np.log10(N), np.log10(walker(s, R, g))
    p = np.polyfit(x, y, 1)
    return math.sqrt(np.mean((y - np.polyval(p, x)) ** 2)), p


def fit_gamma(rows):
    sel = [r for r in rows if r["notch"] == "plain" and r["R"] in (-1.0, 0.0) and not r["runout"]]
    N, s, R = (np.array([r[k] for r in sel]) for k in ("N", "s_max", "R"))
    gs = np.linspace(0, 1, 201)
    v = [pooled_rms(N, s, R, g)[0] for g in gs]
    return float(gs[int(np.argmin(v))])


def collapse_table(rows, g):
    out = {}
    for n in ("plain", "sharp", "intermediate", "blunt"):
        sel = [r for r in rows if r["notch"] == n and not r["runout"]]
        N, s, R = (np.array([r[k] for r in sel]) for k in ("N", "s_max", "R"))
        out[n] = (pooled_rms(N, s, R, 0.0)[0], pooled_rms(N, s, R, g)[0], len(sel))
    return out


def fig(draws, g, path):
    rows = sde.dataset()
    tab = collapse_table(rows, g)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, notch in zip(axs.ravel(), ("plain", "sharp", "intermediate", "blunt")):
        rs = [r for r in rows if r["notch"] == notch]
        for R in sorted({r["R"] for r in rs}):
            sel = [r for r in rs if r["R"] == R]
            N = np.array([r["N"] for r in sel])
            y = walker([r["s_max"] for r in sel], R, g) / paper.UTS
            ro = np.array([r["runout"] for r in sel])
            ax.scatter(N[~ro], y[~ro], s=14, color=COL[R], alpha=0.8, label=f"specimen R {R:g}")
            if ro.any():
                ax.scatter(N[ro], y[ro], s=30, marker=">", facecolor="none", edgecolor=COL[R], label=f"run-out R {R:g}")
            if R in (-1.0, 0.0):
                smax = np.array([r["s_max"] for r in sel])
                S = np.geomspace(smax.min() * 0.7, min(smax.max() * 1.3, paper.UTS), 14)
                b = predictive_band(draws, notch, R, S)
                ax.plot(b[1], walker(S, R, g) / paper.UTS, color=COL[R], lw=2, label=f"model median R {R:g}")
                ax.plot(b[0], walker(S, R, g) / paper.UTS, color=COL[R], lw=0.8, ls="--")
                ax.plot(b[2], walker(S, R, g) / paper.UTS, color=COL[R], lw=0.8, ls="--",
                        label=f"model P10/P90 R {R:g}")
        r0, rw, n = tab[notch]
        ax.set_title(f"{TITLE[notch]}\nscatter about one line: sigma_max {r0:.3f} -> Walker {rw:.3f} (n {n})", fontsize=10)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e2, 5e6); ax.set_ylim(0.04, 1.0)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    for ax in axs[1]:
        ax.set_xlabel("cycles to failure N")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\sigma_{\max}((1-R)/2)^{\gamma}$ / UTS  (UTS 42.9 MPa; notched: net)")
    fig.suptitle(f"Walker-collapsed view, gamma {g:.2f} fitted on plain R 0 vs R -1 only. DEPARTURE from Ezeh & Susmel 2020 "
                 "(sigma_max only).\nSame SDE posterior as the sigma_max figure (E' 3479 MPa, pointwise gradient); "
                 "the model itself has no Walker term. Scatter = rms log10 about one Basquin line per panel.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))
    return tab


if __name__ == "__main__":
    c = np.load(sys.argv[1])
    ch = c["chain"][c["chain"].shape[0] // 2:].reshape(-1, c["chain"].shape[-1])
    draws = ch[np.random.default_rng(1).choice(len(ch), 120, replace=False)]
    g = fit_gamma(sde.dataset())
    print("gamma", g, fig(draws, g, sys.argv[2]))
