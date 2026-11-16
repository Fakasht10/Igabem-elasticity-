"""Step 2: every method calibrated the paper's way (plain P50 + sharp V Fig. 8a curve), U r1 and U r3 held out.

    python -m es2020.compare [out_dir]   -> compare_methods.md, fig_compare_methods.png/pdf

Stresses: sigma_n,max of each specimen as plotted in Fig. 8 (the paper's chain). Residual = log10(sigma_eq / plain P50
at the specimen's life), failures only (8 run-outs left out). Band = inside the plain P10-P90 band.
"""
import os
import sys

import numpy as np

from . import methods as M
from . import paper

PANELS = {"paper": ("point", "area", "weibull", "mhsv"),          # fig_compare_methods: paper's TCD forms + size methods
          "more": ("line", "gradient", "sed", "swt_point"),       # fig_compare_more: Aleksey's 2026-11-16 additions
          "sed": ("net", "point", "sed_r0", "sed")}               # fig_compare_sed: SED vs TCD (Aleksey, 2026-11-16)
TITLE = {"point": "TCD Point Method (Eq. 2)", "area": "TCD Area Method (Eq. 4)", "line": "TCD Line Method (Eq. 3)",
         "weibull": "Weibull weakest link (whole specimen volume)", "mhsv": "Modified HSV (Zhang et al. 2023)",
         "net": "Net stress, no method", "netfit": "Net stress x one constant fitted on sharp V",
         "gradient": "Stress gradient (Siebel-Stieler support)", "sed": "Averaged SED (Lazzarin-Berto), R0 life-dependent",
         "sed_r0": "Averaged SED (Lazzarin-Berto), constant R0",
         "swt_point": "TCD Point Method in SWT variable"}


def run():
    specs = paper.fig8_specimens()
    out = {}
    for name in [n for n in M.METHODS if n in TITLE]:      # per-measure variants: es2020.measures
        p, c = M.calibrate(name)
        R = M.residuals(name, p, specs)
        T = M.basis(name)["plain"].T
        out[name] = dict(p=p, cal=c, R=R, T=T, S={k: M.stats(*R[k], T) for k in paper.NOTCHES},
                         const=M.METHODS[name][2].format(*p), basis=M.basis(name))
    R = {}
    for k in paper.NOTCHES:
        S = np.array([(s.N, s.s_max) for s in specs if s.notch == k and not s.runout])
        R[k] = (S[:, 0], np.log10(S[:, 1] / paper.PLAIN(S[:, 0])))
    out["net"] = dict(p=(), cal=np.nan, R=R, T=paper.PLAIN.T, S={k: M.stats(*R[k]) for k in paper.NOTCHES},
                      const="none", basis=M.basis("net"))
    return out


def held(o):
    r = np.concatenate([o["R"][k][1] for k in ("intermediate", "blunt")])
    N = np.concatenate([o["R"][k][0] for k in ("intermediate", "blunt")])
    return M.stats(N, r, o["T"])


def table(out):
    L = ["| method | constants (fit on plain + sharp V) | fitted sharp V: bias / rms / slope / in band | "
         "held-out U r1 | held-out U r3 | held-out both: rms / in band |", "|---|---|---|---|---|---|"]
    f = lambda s: f"{s['bias']:+.3f} / {s['rms']:.3f} / {s['slope']:+.3f} / {100*s['band']:.0f}%"
    for name in sorted(out, key=lambda n: held(out[n])["rms"]):
        o, h = out[name], held(out[name])
        L.append(f"| {TITLE[name]} | {o['const']} | {f(o['S']['sharp'])} | {f(o['S']['intermediate'])} | "
                 f"{f(o['S']['blunt'])} | {h['rms']:.3f} / {100*h['band']:.0f}% |")
    L.append("")
    L.append("SWT row is a departure from the paper and not like-for-like: its plain curve is our refit of the Fig. 1 "
             "points in sigma_SWT (T 1.96, narrower than 2.451), so its rms and in-band share refer to a different band.")
    return "\n".join(L)


def plot(out, path, show):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6), sharex=True, sharey=True, layout="constrained")
    N = np.geomspace(10, 2e6, 300)
    mk = {"sharp": ("^", "k", "k"), "intermediate": ("o", "none", "#1f3f8f"), "blunt": ("D", "none", "#c0392b")}
    for ax, name in zip(axs.ravel(), show):
        o = out[name]
        pl = o["basis"]["plain"]
        ax.plot(N, pl(N) / paper.UTS, "k-", lw=1.3)
        for ps in (0.1, 0.9):
            ax.plot(N, pl(N, ps) / paper.UTS, "k--", lw=0.8)
        if pl is paper.PLAIN:
            ax.plot(N, paper.DESIGN(N) / paper.UTS, color="0.6", lw=1.8)
        for k in paper.NOTCHES:
            n, r = o["R"][k]
            m, fc, ec = mk[k]
            ax.scatter(n, 10 ** r * pl(n) / paper.UTS, marker=m, s=26, facecolor=fc, edgecolor=ec, lw=0.9, zorder=3)
        h = held(o)
        s = o["S"]["sharp"]
        note = {"area": "\nL > half net width (3 mm) below ~1e3 cycles: semicircle mirrored past the centre line",
                "weibull": "\nnotch factor vs sharp V: U r1 0.996, U r3 1.020 (acts like net stress)",
                "line": "\nnot used in the paper: 2L > half net width below ~1e4 cycles (profile mirrored),\nL refitted (Eq. 3 would use the point L_M); 2L capped at 6 mm below ~800 cycles",
                "gradient": "\nchi at root 12.2 / 1.85 / 0.59 1/mm (V, U r1, U r3); not in the paper",
                "sed": "\nW: all 6 stress components, nu 0.35; crescent r0 0.45 rho (V), 0.5 rho (U)"
                       "\nR0 > 3 mm (half net width) below ~900 cycles: control area extends past the centre plane",
                "sed_r0": "\nW: all 6 stress components, nu 0.35; crescent r0 0.45 rho (V), 0.5 rho (U)"
                          "\nK_eq " + " / ".join(f"{float(M.k_sed(k, o['p'][0])[0]):.2f}" for k in paper.NOTCHES if name == "sed_r0")
                          + f" (V, U r1, U r3); calibration rms {o['cal']:.3f}",
                "swt_point": f"\nDEPARTURE: plain P50 refit on Fig. 1 points in SWT (k {pl.k:.2f}, {pl.s_ref:.2f} MPa, T {pl.T:.2f}),"
                             f"\nsharp curve refit on Fig. 8 V in SWT; no design curve"}.get(name, "")
        ax.text(0.02, 0.02, f"{TITLE[name]}\n{o['const']}{note}\n"
                f"fitted (sharp V): bias {s['bias']:+.3f}, rms {s['rms']:.3f}, slope {s['slope']:+.3f}\n"
                f"held-out (U r1 + U r3, n {h['n']}): bias {h['bias']:+.3f}, rms {h['rms']:.3f},\n"
                f"   slope {h['slope']:+.3f}, in P10-P90 band {round(h['band']*h['n'])}/{h['n']} ({100*h['band']:.0f}%)   (log10 stress)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.3, bbox=dict(fc="white", ec="0.6", lw=0.6))
        ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.03, 3))
        ax.grid(alpha=0.3, which="both")
    for ax in axs[1]:
        ax.set_xlabel("$N_f$ [cycles to failure]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\sigma_{eq}/\sigma_{UTS}$ ($\sigma_{max}$; SWT panel: $\sigma_{SWT}$)" if "swt_point" in show
                      else r"$\sigma_{eq,max}/\sigma_{UTS}$")
    h = [Line2D([], [], marker=mk[k][0], ls="", mfc=mk[k][1] if mk[k][1] != "none" else "none", mec=mk[k][2], ms=6)
         for k in paper.NOTCHES] + [Line2D([], [], color="k", lw=1.3), Line2D([], [], color="k", ls="--", lw=0.8),
                                    Line2D([], [], color="0.6", lw=1.8)]
    lb = ["Specimens, sharp V r0.15 (calibration set)", "Specimens, U r1 (held out)", "Specimens, U r3 (held out)",
          "Plain P50 6.7 MPa at 2e6, k 5.1" + (" (SWT panel: refit, see box)" if "swt_point" in show else ""),
          "Plain P10 / P90, $T_\\sigma$ 2.451" + (" (SWT: 1.96)" if "swt_point" in show else ""), "Design 0.1 UTS, k 5.5"]
    fig.legend(h, lb, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Ezeh & Susmel 2020 data (Fig. 8 $\\sigma_{n,max}$, UTS 42.9 MPa, R = -1 and 0 pooled, failures only): "
                 "free constants fitted on plain P50 + sharp V (Fig. 9e), U notches predicted"
                 + ("\nSWT panel: plain band refit on Fig. 1 points at R -1, -0.5, 0, 0.3" if "swt_point" in show else ""), fontsize=10.5)
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=190)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = run()
    t = table(out)
    print(t)
    open(os.path.join(d, "compare_methods.md"), "w").write(__doc__ + "\n" + t + "\n")
    plot(out, os.path.join(d, "fig_compare_methods"), PANELS["paper"])
    plot(out, os.path.join(d, "fig_compare_more"), PANELS["more"])
    plot(out, os.path.join(d, "fig_compare_sed"), PANELS["sed"])
