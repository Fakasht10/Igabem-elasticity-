"""Local stress measures along the notch bisector (IgBEM, thickness-mean, unit net stress) and the point / line
method built on each, calibrated on plain P50 + sharp V (Fig. 8a), U notches held out.

    python -m es2020.measures [out_dir]  -> fig_bisector_measures.png/pdf, bisector_measures.md
"""
import os
import sys

import numpy as np

from . import methods as M
from . import paper

ROWS = (("net", "Net stress (no local measure)"), ("point", "Point, max principal stress"),
        ("point_vm", "Point, von Mises = octahedral"), ("point_eps1", "Point, E x max principal strain"),
        ("line", "Line, max principal stress"), ("line_vm", "Line, von Mises = octahedral"),
        ("line_eps1", "Line, E x max principal strain"))
CURVE = {"s1": ("max principal stress $\\sigma_1$ (= $\\sigma_{yy}$ on the bisector)", "k", "-"),
         "vm": ("von Mises $\\sigma_{vM}$ (= $3\\tau_{oct}/\\sqrt{2}$)", "#1f3f8f", "--"),
         "eps1": ("$E\\,\\varepsilon_1$, max principal strain ($\\nu$ 0.35)", "#c0392b", "-.")}
PT = {"s1": "point", "vm": "point_vm", "eps1": "point_eps1"}


def scores():
    specs = paper.fig8_specimens()
    out = {}
    for name, _ in ROWS:
        if name == "net":
            R = {k: (np.array([s.N for s in specs if s.notch == k and not s.runout]),
                     np.log10([s.s_max / paper.PLAIN(s.N) for s in specs if s.notch == k and not s.runout]))
                 for k in paper.NOTCHES}
            p = ()
        else:
            p, _ = M.calibrate(name)
            R = M.residuals(name, p, specs)
        N = np.concatenate([R[k][0] for k in ("intermediate", "blunt")])
        r = np.concatenate([R[k][1] for k in ("intermediate", "blunt")])
        out[name] = dict(p=p, fit=M.stats(*R["sharp"]), u1=M.stats(*R["intermediate"]), u3=M.stats(*R["blunt"]),
                         held=M.stats(N, r))
    return out


def table(S):
    f = lambda s: f"{s['bias']:+.3f} / {s['rms']:.3f}"
    L = ["| measure | L (mm) | fitted sharp V bias / rms | U r1 bias / rms | U r3 bias / rms | held-out rms / in band |",
         "|---|---|---|---|---|---|"]
    for name, lab in ROWS:
        o = S[name]
        c = "-" if not len(o["p"]) else f"{o['p'][0]:.3g} N^{o['p'][1]:.3f}"
        L.append(f"| {lab} | {c} | {f(o['fit'])} | {f(o['u1'])} | {f(o['u3'])} | "
                 f"{o['held']['rms']:.3f} / {100*o['held']['band']:.0f}% |")
    return "\n".join(L)


SHORT = {"max principal stress": "$\\sigma_1$", "von Mises = octahedral": "vM", "E x max principal strain": "$E\\varepsilon_1$"}


def plot(S, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True, layout="constrained")
    for ax, k in zip(axs, paper.NOTCHES):
        for m, (lab, col, ls) in CURVE.items():
            p = M.measure_profile(k, m)
            ax.plot(p.d[1:], p.s[1:], color=col, ls=ls, lw=1.5)
            A, B = S[PT[m]]["p"]
            for N, mk in ((1e3, "s"), (2e6, "o")):
                r = A * N ** B / 2
                ax.plot(r, p(r), mk, color=col, ms=6, mfc="white", mew=1.3)
        ax.axhline(1.0, color="0.55", lw=1.0)
        ax.set(xscale="log", xlim=(2e-3, 3.05), ylim=(0, 6.8), xlabel="distance from the notch root on the bisector, d (mm)",
               title=paper.LABEL[k] + f"  (K$_t$ {M.measure_profile(k, 's1').s[0]:.2f})")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("local measure / nominal net stress")
    h = [Line2D([], [], color=c, ls=l, lw=1.5) for _, c, l in CURVE.values()] + \
        [Line2D([], [], color="0.55", lw=1.0), Line2D([], [], marker="s", ls="", color="k", mfc="white"),
         Line2D([], [], marker="o", ls="", color="k", mfc="white")]
    lb = [v[0] for v in CURVE.values()] + ["net stress", "point-method sampling point L/2 at N = 1e3 (own fit per measure)",
                                          "... at N = 2e6"]
    fig.legend(h, lb, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Local measures on the notch bisector: IgBEM 3D, thickness mean, 200 stations per notch "
                 "(mesh-converged within 0.4% for d >= 0.1 mm), read from the stored solution\n"
                 "held-out rms, log10 stress (U r1 + U r3; fit on plain P50 + sharp V Fig. 8a):  net " +
                 f"{S['net']['held']['rms']:.3f}  |  " +
                 "  |  ".join(f"{lab.split(', ')[0].lower()} {SHORT[lab.split(', ')[1]]} {S[n]['held']['rms']:.3f}"
                              for n, lab in ROWS[1:]), fontsize=9.6)
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=190)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    S = scores()
    t = table(S)
    print(t)
    open(os.path.join(d, "bisector_measures.md"), "w").write(__doc__ + "\n" + t + "\n")
    plot(S, os.path.join(d, "fig_bisector_measures"))
