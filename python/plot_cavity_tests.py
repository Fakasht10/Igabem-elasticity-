"""Paper-style plots for the cube with a spherical cavity (traction-only SBIE).

Reads results/cavity_tests.csv and results/cavity_profile_*.csv and writes
results/fig_cavity_tests.{pdf,png}:
  (a) quadrature: displacement error against n
  (b) h-refinement at n = 12: displacement error and stress-concentration error
  (c) meridional stress on the cavity, exact Goodier field and finite cube in tension
"""

import csv
import math
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]
NG = 12  # production quadrature
SCF = 93.0 / 46.0  # Goodier, nu = 1/4


def load():
    d = OrderedDict()  # (label, case) -> {n: row}; later rows win
    with open(os.path.join(RES, "cavity_tests.csv")) as f:
        for r in csv.DictReader(f):
            d.setdefault((r["label"], r["case"]), {})[int(r["n"])] = r
    return d


def profile(case, r):
    p = os.path.join(RES, "cavity_profile_%s_r%d.csv" % (case, r))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: [float(x[k]) if x[k] else math.nan for x in rows]
    return col("theta_deg"), col("s_mer"), col("s_mer_exact")


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    d = load()
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    ylab = r"$\|u_h-u\|_{L^2(\Gamma)}\,/\,\|u\|_{L^2(\Gamma)}$"

    ax = axs[0]
    key = ("cube_cavity_r0", "patch")
    if key in d:
        ns = sorted(d[key])
        ax.semilogy(ns, [max(float(d[key][n]["rel_L2_err"]), 1e-17) for n in ns], color=C[0],
                    marker=M[0], ms=4.5, lw=1.6, label=r"patch test $u=\alpha x$: $u$ error, 12 el.")
    # Load cases without a polynomial solution: change of the equator stress against n = 12.
    for k, (case, name) in enumerate([("pressure", "pressurized cavity"), ("goodier", "Goodier field"),
                                      ("tension", "finite cube in tension")]):
        key = ("cube_cavity_r1", case)
        if key in d:
            ns = [n for n in sorted(d[key]) if n < NG]
            ref = float(d[key][NG]["szz_eq"])
            ax.semilogy(ns, [max(abs(float(d[key][n]["szz_eq"]) - ref) / abs(ref), 1e-17) for n in ns],
                        color=C[k + 1], marker=M[k + 1], ms=4.5, lw=1.2, ls="--",
                        label=r"%s: $\sigma_{zz}$ at equator, 48 el." % name)
    ax.set_ylim(1e-14, 1e0)
    ax.set_yticks([10.0 ** k for k in range(-14, 1, 2)])
    ax.minorticks_off()
    ax.set_xlabel("integration points per direction, $n$")
    ax.set_ylabel(r"relative error ($u$) or change from $n=%d$ ($\sigma_{zz}$)" % NG)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style(ax, "(a) Quadrature convergence")

    ax = axs[1]
    rs = [r for r in range(4) if ("cube_cavity_r%d" % r, "goodier") in d]
    h = [1.0 / 2 ** r for r in rs]
    get = lambda case, key: [float(d[("cube_cavity_r%d" % r, case)][NG][key]) for r in rs]
    eu_p, eu_g = get("pressure", "rel_L2_err"), get("goodier", "rel_L2_err")
    es = [abs(s - SCF) / SCF for s in get("goodier", "szz_eq")]
    ax.loglog(h, eu_p, color=C[1], marker="s", ms=5, lw=1.6, label="pressurized cavity: $u$")
    ax.loglog(h, eu_g, color=C[2], marker="^", ms=5, lw=1.6, label="Goodier field: $u$")
    ax.loglog(h, es, color=C[2], marker="^", mfc="white", ms=5, lw=1.2, ls="--",
              label=r"Goodier field: SCF")
    if len(h) >= 2:
        x0, x1 = h[-2], h[-1]
        y1 = min(eu_g[-1], es[-1]) * 0.3
        ax.loglog([x0, x1], [y1 * (x0 / x1) ** 4, y1], color="#3d3d3a", lw=0.9)
        ax.text((x0 * x1) ** 0.5 * 1.08, y1 * (x0 / x1) ** 2, "slope 4", fontsize=8,
                color="#3d3d3a", ha="right")
    ax.invert_xaxis()
    ax.minorticks_off()
    ax.set_xticks(h, ["1", "1/2", "1/4", "1/8"][: len(h)])
    el = [d[("cube_cavity_r%d" % r, "goodier")][NG]["elements"] for r in rs]
    ax.set_xlabel("element size $h/h_0$ (%s elements)" % ", ".join(el))
    ax.set_ylabel("relative error")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    style(ax, "(b) h-refinement, $n=%d$" % NG)

    ax = axs[2]
    pg = profile("goodier", 0)
    if pg:
        ax.plot(pg[0], pg[2], color="#3d3d3a", lw=1.4, label="exact, infinite medium")
    for k, r in enumerate([0, 3]):
        pg = profile("goodier", r)
        if pg:
            ax.plot(pg[0][::6], pg[1][::6], ls="none", color=[C[3], C[2]][k], marker=["D", "^"][k],
                    ms=4, mfc="white" if r == 0 else None,
                    label="IgBEM, Goodier data, %s el." % d[("cube_cavity_r%d" % r, "goodier")][NG]["elements"])
    pt = profile("tension", 3)
    if pt:
        ax.plot(pt[0], pt[1], color=C[0], lw=1.6, ls="--", label="IgBEM, finite cube in tension")
        scf_t = float(d[("cube_cavity_r3", "tension")][NG]["szz_eq"])
        ax.annotate("%.3f" % scf_t, (90, scf_t), xytext=(110, scf_t + 0.2), fontsize=8, color=C[0],
                    arrowprops=dict(arrowstyle="-", color=C[0], lw=0.6))
    ax.annotate("%.3f" % SCF, (90, SCF), xytext=(35, SCF + 0.35), fontsize=8, color="#3d3d3a",
                arrowprops=dict(arrowstyle="-", color="#3d3d3a", lw=0.6))
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"polar angle on the cavity $\theta$ (deg), $0$ = load axis")
    ax.set_ylabel(r"meridional stress $\sigma_{\theta\theta}/S$")
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="lower center")
    ax.set_ylim(-1.6, 2.8)
    style(ax, "(c) Stress on the cavity, uniaxial tension")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_cavity_tests." + ext), dpi=200)


if __name__ == "__main__":
    main()
