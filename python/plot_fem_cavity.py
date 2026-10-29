"""IGA-FEM (JAXIGA, octant) against the collocation IgBEM for the cube with a spherical cavity.

Reads results/fem_cavity.csv, results/fem_profile_*.csv, results/cavity_tests.csv and
results/cavity_profile_tension_r3.csv, and writes results/fig_fem_cavity.{pdf,png}:
  (a) Goodier field: SCF error against unknowns, FEM p = 4, 5 and IgBEM r0..r3
  (b) finite cube in tension: sigma_zz(a, 0, 0) / S against unknowns, both methods
  (c) meridional stress on the cavity in tension, FEM and IgBEM, and their difference
Unknowns: FEM counts the octant model's 3 x control points; IgBEM the full surface's.
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
SCF = 93.0 / 46.0  # Goodier, nu = 1/4
NG = 12  # IgBEM production quadrature


def fem_rows():
    with open(os.path.join(RES, "fem_cavity.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:  # later rows win
        out[(r["case"], int(r["p"]), int(r["n"]))] = r
    return out


def bem_rows():
    out = {}
    with open(os.path.join(RES, "cavity_tests.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["n"]) == NG:
                out[(r["case"], int(r["label"][-1]))] = r
    return out


def series(fem, case, p):
    ns = sorted(n for (c, q, n) in fem if c == case and q == p)
    return ([int(fem[(case, p, n)]["ndof"]) for n in ns],
            [float(fem[(case, p, n)]["szz_eq"]) for n in ns], ns)


def profile(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: [float(x[k]) if x[k] not in ("", "nan") else math.nan for x in rows]
    return col("theta_deg"), col("s_mer")


def interp(xq, x, y):
    out, j = [], 0
    for t in xq:
        while j < len(x) - 2 and x[j + 1] < t:
            j += 1
        out.append(y[j] + (y[j + 1] - y[j]) * (t - x[j]) / (x[j + 1] - x[j]))
    return out


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fem, bem = fem_rows(), bem_rows()
    ps = sorted({p for (_, p, _) in fem})
    rs = sorted(r for (c, r) in bem if c == "goodier")
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

    ax = axs[0]
    for k, p in enumerate(ps):
        nd, s, _ = series(fem, "goodier", p)
        ax.loglog(nd, [abs(v - SCF) / SCF for v in s], color=C[k], marker="os"[k], ms=4.5,
                  lw=1.6, label="IGA-FEM octant, $p=%d$" % p)
    nb = [3 * int(bem[("goodier", r)]["ndof"]) for r in rs]
    ax.loglog(nb, [abs(float(bem[("goodier", r)]["szz_eq"]) - SCF) / SCF for r in rs],
              color=C[2], marker="^", ms=5, lw=1.2, ls="--", label="IgBEM full surface, $p=4$")
    ax.set_xlabel("unknowns")
    ax.set_ylabel(r"$|\mathrm{SCF}_h-\mathrm{SCF}|\,/\,\mathrm{SCF}$")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.minorticks_off()
    style(ax, "(a) Goodier field, exact SCF %.4f" % SCF)

    ax = axs[1]
    finest = max(fem.values(), key=lambda r: int(r["ndof"]) if r["case"] == "tension" else 0)
    for k, p in enumerate(ps):
        nd, s, _ = series(fem, "tension", p)
        ax.semilogx(nd, s, color=C[k], marker="os"[k], ms=4.5, lw=1.6,
                    label="IGA-FEM octant, $p=%d$" % p)
    sb = [float(bem[("tension", r)]["szz_eq"]) for r in rs]
    ax.semilogx([3 * int(bem[("tension", r)]["ndof"]) for r in rs], sb, color=C[2], marker="^",
                ms=5, lw=1.2, ls="--", label="IgBEM full surface, $p=4$")
    best = float(finest["szz_eq"])
    ax.axhline(best, color="#3d3d3a", lw=0.8, ls=":")
    ax.text(1.2 * min(nb), best + 0.003, "finest FEM %.4f" % best, fontsize=7.5, color="#3d3d3a")
    ax.set_xlabel("unknowns")
    ax.set_ylabel(r"$\sigma_{zz}(a,0,0)\,/\,S$")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.minorticks_off()
    style(ax, "(b) Finite cube in tension")

    ax = axs[2]
    pmax, nmax = int(finest["p"]), int(finest["n"])
    th, sm = profile(os.path.join(RES, "fem_profile_tension_p%d_n%d.csv" % (pmax, nmax)))
    ax.plot(th, sm, color=C[ps.index(pmax)], lw=1.6,
            label="IGA-FEM, $p=%d$, %s el." % (pmax, finest["elements"]))
    tb, sbm = profile(os.path.join(RES, "cavity_profile_tension_r3.csv"))
    keep = [i for i, t in enumerate(tb) if t <= 90]
    tb, sbm = [tb[i] for i in keep], [sbm[i] for i in keep]
    ax.plot(tb[::5], sbm[::5], ls="none", color=C[2], marker="^", ms=4, mfc="white",
            label="IgBEM, 768 el.")
    diff = [b - f for b, f in zip(sbm, interp(tb, th, sm))]
    ins = ax.inset_axes([0.58, 0.12, 0.38, 0.32])
    ins.plot(tb, diff, color="#3d3d3a", lw=1.0)
    ins.axhline(0, color="#c3c2b7", lw=0.6)
    ins.set_title("IgBEM $-$ FEM", fontsize=7)
    ins.tick_params(labelsize=6.5)
    ins.set_xticks([0, 45, 90])
    ins.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ins.yaxis.get_offset_text().set_fontsize(6.5)
    ax.set_xlim(0, 90)
    ax.set_xticks([0, 30, 60, 90])
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.set_xlabel(r"polar angle on the cavity $\theta$ (deg), $0$ = load axis")
    ax.set_ylabel(r"meridional stress $\sigma_{\theta\theta}/S$")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    style(ax, "(c) Cavity stress, finite cube in tension")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, "fig_fem_cavity." + ext), dpi=200)


if __name__ == "__main__":
    main()
