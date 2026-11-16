"""Figures for the latent-damage SDE (paper variables: sigma_max / UTS, UTS 42.9 MPa, against N).

    python -m es2020.sde_figs chain.npz resid.npz scores.json out_dir

Fig. A  S-N per specimen set (plain, sharp V fitted; U r1, U r3 predicted): specimens (labelled), run-outs as
        arrows, posterior-predictive median and P10-P90 per R. Notched specimens at net sigma_max (the paper's Fig. 8
        variable). Departure from the paper stated on the figure: R enters through the local Masing loop.
Fig. B  held-out residual log10(S*/S) against log10 N, with the fitted slope.
Fig. C  ranking of held-out rms against the field-only baselines.
"""
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import paper
from . import sde

COL = {-1.0: "#1f5fa8", 0.0: "#c2410c", -0.5: "#6b7280", 0.3: "#15803d"}
TITLE = {"plain": "Plain, Fig. 1 pool (fitted)", "sharp": "Sharp V r0.15 (fitted)",
         "intermediate": "U r1 (predicted, held out)", "blunt": "U r3 (predicted, held out)"}


def predictive_band(draws, notch, R, S, mode="point", nsamp=40, seed=0):
    """Posterior-predictive P10/P50/P90 of N at net (or plain) stresses S: inverse-CDF samples from each draw's
    first-passage law, pooled."""
    rng = np.random.default_rng(seed)
    rows = [dict(set="x", notch=notch, R=R, s_max=float(s), N=1.0, runout=False) for s in S]
    m = sde.Model(rows, mode)
    q = sde.table()["qlev"]
    out = []
    for th in draws:
        lc, Q = m.predict(th)
        u = rng.uniform(q[0], q[-1], (nsamp, len(S)))
        y = np.array([np.interp(u[:, i], q, np.maximum.accumulate(Q[i])) for i in range(len(S))]).T
        out.append(lc[None, :] + y)
    lnN = np.concatenate(out, axis=0)
    return np.exp(np.percentile(lnN, [10, 50, 90], axis=0))


def fig_sn(draws, path):
    rows = sde.dataset()
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, notch in zip(axs.ravel(), ("plain", "sharp", "intermediate", "blunt")):
        rs = [r for r in rows if r["notch"] == notch]
        for R in sorted({r["R"] for r in rs}):
            sel = [r for r in rs if r["R"] == R]
            N = np.array([r["N"] for r in sel])
            y = np.array([r["s_max"] for r in sel]) / paper.UTS
            ro = np.array([r["runout"] for r in sel])
            ax.scatter(N[~ro], y[~ro], s=14, color=COL[R], alpha=0.8, label=f"specimen R {R:g}")
            if ro.any():
                ax.scatter(N[ro], y[ro], s=30, marker=">", facecolor="none", edgecolor=COL[R], label=f"run-out R {R:g}")
            if R in (-1.0, 0.0):
                S = np.geomspace(y.min() * 0.7, min(y.max() * 1.3, 1.0), 14) * paper.UTS
                b = predictive_band(draws, notch, R, S)
                ax.plot(b[1], S / paper.UTS, color=COL[R], lw=2, label=f"model median R {R:g}")
                ax.fill_betweenx(S / paper.UTS, b[0], b[2], color=COL[R], alpha=0.12, lw=0,
                                 label=f"model P10-P90 R {R:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e2, 5e6); ax.set_ylim(0.04, 1.0)
        ax.set_title(TITLE[notch], fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    for ax in axs[1]:
        ax.set_xlabel("cycles to failure N")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\sigma_{\max}$ / UTS  (UTS 42.9 MPa; notched: net section)")
    fig.suptitle("Latent-damage SDE, pointwise gradient support, E' fixed at 3479 MPa: posterior predictive\n"
                 "Departure from Ezeh & Susmel 2020: R enters through the local Masing loop, not sigma_max alone. "
                 "Fit: plain + sharp V only.", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.replace(".png", ".pdf"))


def fig_resid(models, path):
    """models: list of (label, resid dict) on dataset() rows."""
    rows = sde.dataset()
    held = np.array([r["set"] == "held" and not r["runout"] for r in rows])
    notch = np.array([r["notch"] for r in rows])
    logN = np.log10([r["N"] for r in rows])
    fig, axs = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.2), sharey=True)
    for ax, (lab, r) in zip(np.atleast_1d(axs), models):
        d = np.asarray(r["dlogS"])
        for n, mk in (("intermediate", "o"), ("blunt", "s")):
            m = held & (notch == n)
            ax.scatter(logN[m], d[m], s=16, marker=mk, alpha=0.75, label=f"specimen, {TITLE[n].split(' (')[0]}")
        p = np.polyfit(logN[held], d[held], 1)
        x = np.linspace(logN[held].min(), logN[held].max(), 2)
        ax.plot(x, np.polyval(p, x), "k-", lw=1.5, label=f"fit: slope {p[0]:+.3f} / decade")
        ax.axhline(0, color="0.4", lw=0.8)
        for s in (-1, 1):
            ax.axhline(s * sde.HALF_BAND, color="0.6", ls="--", lw=0.8)
        rms = math.sqrt(np.mean(d[held] ** 2))
        ax.set_title(f"{lab}\nheld-out rms {rms:.3f}, bias {np.mean(d[held]):+.3f}", fontsize=10)
        ax.set_xlabel("log10 N (observed)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    np.atleast_1d(axs)[0].set_ylabel("log10(S* / S_obs);  < 0: predicted life too short")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def fig_rank(entries, path):
    """entries: list of (label, held-out rms, kind) with kind 'ours' or 'baseline'."""
    entries = sorted(entries, key=lambda e: e[1])
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(entries) + 1.2))
    y = np.arange(len(entries))
    ax.barh(y, [e[1] for e in entries], color=["#1f5fa8" if e[2] == "ours" else "#9ca3af" for e in entries])
    for i, e in enumerate(entries):
        ax.text(e[1] + 0.003, i, f"{e[1]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y, [e[0] for e in entries], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("held-out rms log10 S (U r1 + U r3 failures, fit on plain + sharp V)")
    ax.set_title("Blue: this thread. Grey: field-only baselines (sibling thread, same protocol)", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
