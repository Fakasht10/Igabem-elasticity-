"""Figures of the two-region validation (results/run_regions.sh):

  (a) Eshelby inclusion: boundary errors against unknowns, with the patch and Lame
      cases (exact in the space: quadrature floor)
  (b) interior stress through shell and core along the z axis, against the exact field
  (c) matrix stress at the interface against the stiffness contrast, exact curve
  (d) printed coupon: effective modulus against infill density, with bounds

usage: python3 python/plot_regions.py [results/fig_regions.png]
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "results", "regions")
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C1, C2, C3, C4 = "#0969da", "#cf222e", "#1a7f37", "#8250df"


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def scf_exact(k, nu1=0.35, nu2=0.3):
    """Matrix sigma_zz(a, 0, 0) / S for E2 = k E1 (gen_inclusion_exact.py)."""
    mu1 = 1.0 / (2 * (1 + nu1))
    mu2 = k / (2 * (1 + nu2))
    num = 3 * (10 * mu1**2 * nu1**2 * nu2 - 5 * mu1**2 * nu1**2 - 8 * mu1**2 * nu1 * nu2 + 4 * mu1**2 * nu1
               - 18 * mu1**2 * nu2 + 9 * mu1**2 + 10 * mu1 * mu2 * nu1**2 * nu2 - 10 * mu1 * mu2 * nu1**2
               + 12 * mu1 * mu2 * nu1 * nu2 - 2 * mu1 * mu2 * nu2 + 6 * mu1 * mu2 - 5 * mu2**2 * nu1**2 * nu2
               - 5 * mu2**2 * nu1**2 + mu2**2 * nu1 * nu2 + mu2**2 * nu1)
    den = ((nu1 + 1) * (5 * mu1 * nu1 - 7 * mu1 + 10 * mu2 * nu1 - 8 * mu2)
           * (4 * mu1 * nu2 - 2 * mu1 - mu2 * nu2 - mu2))
    return num / den


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "fig_regions.png")
    S = rows(os.path.join(D, "spheres.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))

    # (a) convergence
    a = ax[0, 0]
    es = [r for r in S if r["case"] == "eshelby" and float(r["contrast"]) == 0.1]
    N = np.array([int(r["host_unknowns"]) + int(r["core_unknowns"]) for r in es])
    for key, lab, c, m in [("eu_shell", "u, shell", C1, "o"), ("eu_core", "u, core", C3, "s"),
                           ("et_interface", "t, interface", C2, "^"), ("es_interior", "interior σ (max)", C4, "D")]:
        a.loglog(N, [float(r[key]) for r in es], "-" + m, color=c, ms=5, lw=1.4, label=lab)
    ref = float(es[-2]["eu_shell"]) * (N / N[-2]) ** (-2.5)
    a.loglog(N[1:], ref[1:], "--", color=MUTED, lw=1, label="N$^{-5/2}$ (h$^5$, p = 4)")
    for case, m in [("patch", "x"), ("lame", "+")]:
        rr = [r for r in S if r["case"] == case]
        a.loglog([int(r["host_unknowns"]) + int(r["core_unknowns"]) for r in rr], [float(r["eu_shell"]) for r in rr],
                 m, color=INK, ms=7, ls="none", label="%s (exact in space), u shell" % case)
    a.set_xlabel("unknowns (shell + core)", color=INK)
    a.set_ylabel("relative L2 error", color=INK)
    a.set_title("(a) Eshelby inclusion, E$_c$/E$_s$ = 0.1", color=INK, fontsize=10, loc="left")
    a.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    style(a)

    # (b) interior profile along z (and x)
    b = ax[0, 1]
    P = rows(os.path.join(D, "probe_eshelby_r3.csv"))
    for axis, comp, c, lab in [("z", "szz", C1, "σ$_{zz}$ on z axis"), ("x", "szz", C2, "σ$_{zz}$ on x axis"),
                               ("x", "sxx", C3, "σ$_{xx}$ on x axis")]:
        pts = [r for r in P if r["axis"] == axis]
        s = np.array([float(r["s"]) for r in pts])
        b.plot(s, [float(r[comp]) for r in pts], "o", color=c, ms=3.5, label=lab + " (IgBEM)")
        ex = np.array([float(r[comp + "_exact"]) for r in pts])
        for sel in (s < 0.5, s > 0.5):  # the exact field jumps at the interface
            b.plot(s[sel], ex[sel], "-", color=c, lw=0.8, alpha=0.7)
    b.axvspan(0, 0.5, color=GRID, alpha=0.35, lw=0)
    b.text(0.25, 0.97, "core", transform=b.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=8)
    b.text(0.75, 0.97, "shell", transform=b.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=8)
    b.set_xlabel("distance from centre", color=INK)
    b.set_ylabel("stress / S", color=INK)
    b.set_title("(b) interior stress, r3 (lines: exact)", color=INK, fontsize=10, loc="left")
    b.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    style(b)

    # (c) contrast
    c = ax[1, 0]
    k = np.logspace(-2.2, 1.2, 200)
    c.semilogx(k, scf_exact(k), "-", color=INK, lw=1, label="exact (Goodier 1933)")
    cr = [r for r in S if r["case"] == "eshelby" and "r2" in r["geometry"]]
    c.semilogx([float(r["contrast"]) for r in cr], [float(r["szz_matrix"]) for r in cr], "o", color=C2, ms=6,
               label="IgBEM r2, matrix side")
    c.axhline(scf_exact(1e-9), color=MUTED, lw=0.8, ls=":")
    c.text(0.008, scf_exact(1e-9) + 0.02, "cavity limit", color=MUTED, fontsize=7)
    c.set_xlabel("E$_{core}$ / E$_{shell}$", color=INK)
    c.set_ylabel("σ$_{zz}$(a, 0, 0) / S", color=INK)
    c.set_title("(c) interface stress vs stiffness contrast", color=INK, fontsize=10, loc="left")
    c.legend(fontsize=7, frameon=False)
    style(c)

    # (d) coupon
    d = ax[1, 1]
    for n, m, col in [(2, "s", C3), (4, "^", C1), (6, "o", C2)]:
        f = os.path.join(D, "coupon_n%d.csv" % n)
        if not os.path.exists(f):
            continue
        R = rows(f)
        d.plot([float(r["rho"]) for r in R], [float(r["E_eff"]) for r in R], m + "-", color=col, ms=5, lw=1,
               label="IgBEM, %d el./unit" % n)
    R = rows(os.path.join(D, "coupon_n2.csv"))
    rho = [float(r["rho"]) for r in R]
    for key, lab, ls in [("E_voigt", "Voigt", "--"), ("E_series_parallel", "series-parallel", ":"),
                         ("E_reuss", "Reuss", "-.")]:
        d.plot(rho, [float(r[key]) for r in R], ls, color=MUTED, lw=1, label=lab)
    d.set_xlabel("infill density ρ (E$_{core}$ = ρ² E$_s$)", color=INK)
    d.set_ylabel("E$_{eff}$ / E$_s$", color=INK)
    d.set_title("(d) printed coupon, walls 0.15 of 1", color=INK, fontsize=10, loc="left")
    d.legend(fontsize=7, frameon=False, loc="upper left")
    style(d)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("wrote", out)


if __name__ == "__main__":
    main()
