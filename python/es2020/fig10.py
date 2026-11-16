"""Fig. 10 (Point Method) of Ezeh & Susmel 2020, recomputed on IgBEM stresses and compared marker by marker.

    python -m es2020.fig10 [out_dir]

Inputs exactly as the paper: sigma_n,max of every specimen as plotted in Fig. 8, L_M = 16.4 N^-0.242 mm (Eq. 6),
sigma_eff,max = K(L_M/2) sigma_n,max (Eq. 2), normalised by one UTS of 42.9 MPa; curves of Fig. 1.
The only departure: K(d) comes from our 3D IgBEM solve (sigma_1 on the bisector, thickness-mean), not 2D ANSYS.
"""
import os
import sys

import numpy as np

from . import paper, point


def compare(specs=None, lm=paper.lm_paper, z=point.ZMEAN):
    """Per notch: (N, ours, theirs, runout) for every marker, paired with the authors' Fig. 10 markers."""
    specs = paper.fig8_specimens() if specs is None else specs
    ours = point.fig10(point.profiles(z), specs, lm)
    M = paper.fig10_markers()
    out = {}
    for k in paper.NOTCHES:
        o = [(s.N, y, s.runout) for s, y in ours if s.notch == k]
        t = [(n, y, ro) for kk, n, y, ro in M if kk == k]
        i, j = paper.pair([a[0] for a in o], [a[1] for a in o], [a[0] for a in t], [a[1] for a in t])
        out[k] = np.array([(o[a][0], o[a][1], t[b][1], t[b][0], o[a][2]) for a, b in zip(i, j)])
    return out


def stats(r):
    return dict(n=len(r), median=10 ** np.median(r), rms=np.sqrt(np.mean(r ** 2)), max=10 ** np.abs(r).max(),
                w2=np.mean(np.abs(r) < np.log10(1.02)))


def plot(C, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10.5, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(8.6, 6.6), layout="constrained")
    N = np.geomspace(10, 2e6, 400)
    ax.plot(N, paper.PLAIN(N) / paper.UTS, "k-", lw=1.4)
    for ps in (0.1, 0.9):
        ax.plot(N, paper.PLAIN(N, ps) / paper.UTS, "k--", lw=0.9)
    ax.plot(N, paper.DESIGN(N) / paper.UTS, color="0.6", lw=2.0)
    ax.plot([2e6, 2e6], [0.01, 0.1], "k-.", lw=0.9)
    ax.text(1.9e6, 0.035, "$\\sigma_{MAX,50\\%}/\\sigma_{UTS}$=0.16\n$T_\\sigma$=2.451\n$N_{Ref}=2\\cdot10^6$", ha="right", fontsize=8.5)
    mk = {"blunt": "D", "intermediate": "o", "sharp": "^"}
    for k in paper.NOTCHES:
        P = C[k]
        ax.scatter(P[:, 3], P[:, 2], marker=mk[k], s=62, facecolor="none", edgecolor="#c0392b", lw=1.0, zorder=3)
        ax.scatter(P[:, 0], P[:, 1], marker=mk[k], s=14, color="#1f3f8f", zorder=4)
        for n, y in P[P[:, 4] > 0.5][:, :2]:
            ax.annotate("", xy=(n * 1.6, y * 1.25), xytext=(n, y), arrowprops=dict(arrowstyle="->", lw=0.7))
    r = np.concatenate([np.log10(C[k][:, 1] / C[k][:, 2]) for k in paper.NOTCHES])
    s = stats(r)
    ax.text(14, 1.72, "$P_S$=10%", fontsize=8.5); ax.text(14, 1.08, "$P_S$=50%", fontsize=8.5)
    ax.text(14, 0.70, "$P_S$=90%", fontsize=8.5); ax.text(14, 0.47, "design", fontsize=8.5, color="0.45")
    h = [Line2D([], [], marker=mk[k], ls="", mfc="none", mec="#c0392b", ms=8) for k in paper.NOTCHES] + \
        [Line2D([], [], marker=mk[k], ls="", color="#1f3f8f", ms=4.5) for k in paper.NOTCHES] + \
        [Line2D([], [], color="k", lw=1.4), Line2D([], [], color="k", ls="--", lw=0.9), Line2D([], [], color="0.6", lw=2)]
    lb = [f"Authors, extracted: {paper.LABEL[k]}" for k in paper.NOTCHES] + \
         [f"Ours (IgBEM): {paper.LABEL[k]}" for k in paper.NOTCHES] + \
         ["Plain P50: 6.7 MPa (0.156 UTS) at 2e6, k 5.1 (Sec. 6)", "Plain P10 / P90: $T_\\sigma$ 2.451",
          "Design curve 0.1 UTS, k 5.5"]
    ax.legend(h, lb, fontsize=7.8, loc="lower left", ncol=1, framealpha=0.95)
    ax.text(0.985, 0.985,
            f"Same inputs as the paper: $\\sigma_{{n,max}}$ of each specimen from Fig. 8,\n"
            f"$L_M$ = 16.4 $N_f^{{-0.242}}$ mm (Eq. 6), UTS 42.9 MPa, R = -1 and 0 pooled.\n"
            f"Departure: K(r) from 3D IgBEM (thickness-mean $\\sigma_1$), paper used 2D FE.\n"
            f"Caveat (paper): Fig. 8 $\\sigma_{{n,max}}$ differs from Figs 5-7 $\\times 2/(1-R)$ by >3% for\n"
            f"73 of 157 specimens, mostly past $10^5$ cycles (worst -32%); Fig. 10 follows Fig. 8.\n"
            f"Ours / authors, all {s['n']} markers: median {s['median']:.3f}, rms {100*(10**s['rms']-1):.1f}%,\n"
            f"max {100*(s['max']-1):.1f}%, within 2%: {100*s['w2']:.0f}%. Arrows: run-outs.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            bbox=dict(fc="white", ec="0.6", lw=0.6))
    ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.01, 10), xlabel="$N_f$ [cycles to failure]",
           ylabel=r"$\sigma_{eff,max}/\sigma_{UTS}$")
    ax.set_title("Point Method, Ezeh & Susmel 2020 Fig. 10: authors' markers vs our recomputation", fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig10_point_reproduction.{e}"), dpi=200)
    return s


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    C = compare()
    for k in paper.NOTCHES:
        s = stats(np.log10(C[k][:, 1] / C[k][:, 2]))
        print(f"{k:12s} n {s['n']}  ours/authors median {s['median']:.4f}  rms {s['rms']:.4f} dex  max {s['max']:.3f}  within 2% {100*s['w2']:.0f}%")
    print(plot(C, out))
