"""Paper figures and tables of the bead-cell module (PDF + PNG + Markdown table).

  fig_stiffness   engineering constants of the identified 0/90 cell rotated to 45/-45 vs Biswas 2019 Table 7
                  (CAD-FEM) for the base geometry and the four parametric variants; one panel per constant
  fig_profiles    bond-plane normal traction ahead of the tips (log-log) with the Williams slope lambda - 1
  fig_envelope    sigma_22 - sigma_33 failure envelope of the 0/0 cell with the controlling bond family
  table.md        stiffness and strength tables with the thesis values and the deviations

usage (from python/): python3 -m rve.figures outdir [--runs dir_with_base/LH1/FW1/BW1/BW2.json]
                      [--str00 s.json] [--str090 s.json]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rve.homog import cross_ply, engineering  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (fixed order)
COLS = ["E1", "E2", "E3", "G12", "G13", "G23"]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)))
    plt.close(fig)


def thesis():
    with open(os.path.join(HERE, "data", "biswas2019.json")) as f:
        return json.load(f)


def ours_45(path):
    with open(path) as f:
        r = json.load(f)
    e = engineering(cross_ply(np.array(r["bc"]["pbc"]["C"]), 45.0))
    return r["porosity"] * 100, e


def fig_stiffness(runs, out, rows):
    t = thesis()["table7_param_45"]["rows"]
    names = ["base", "LH1", "FW1", "BW1", "BW2"]
    ours = {n: ours_45(os.path.join(runs, n + ".json")) for n in names}
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.2), sharex=True)
    fig.subplots_adjust(wspace=0.45)
    x = np.arange(len(names))
    for ax, (c, k) in zip(axs, (("E1", 1), ("E3", 3), ("G12", 4), ("G13", 5))):
        ax.bar(x - 0.19, [t[n][k] for n in names], 0.36, color=S1, label="Biswas 2019, CAD-FEM")
        ax.bar(x + 0.19, [ours[n][1][c] for n in names], 0.36, color=S2, label="IgBEM bead cell (PBC)")
        ax.set_title(c + " (MPa)", fontsize=9, color=INK)
        ax.set_xticks(x, names, rotation=45)
        ax.grid(axis="x", visible=False)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0, 1.15), ncol=2)
    save(fig, out, "fig_stiffness")
    rows.append("### Stiffness, 45/-45 layup (0/90 cell rotated), vs Biswas 2019 Table 7 (p. 23)\n")
    rows.append("| case | porosity % (thesis / ours) | " + " | ".join("%s ours (dev)" % c for c in COLS) + " |")
    rows.append("|---|---|" + "---|" * len(COLS))
    for n in names:
        ph, e = ours[n]
        cells = ["%.0f (%+.1f%%)" % (e[c], 100 * (e[c] / t[n][i + 1] - 1)) for i, c in enumerate(COLS)]
        rows.append("| %s | %.2f / %.2f | %s |" % (n, t[n][0], ph, " | ".join(cells)))
    rows.append("")


def fig_profiles(s00, out):
    rho = np.array(s00["rho"])
    lam = s00["williams"][0]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (name, j, col, lab) in (("lat", 1, S1, "lateral bond, $\\sigma_{22}$"), ("int", 2, S2, "interlayer bond, $\\sigma_{33}$")):
        F = s00["families"][name]
        p = np.array(F["profile"])[j][:, 0]
        r = rho * F["half"]
        ok = r >= s00["h_tip"][name]
        ax.loglog(r[ok], p[ok], "-o", color=col, ms=3, lw=1.5, label=lab)
        ax.loglog(r[~ok], p[~ok], "o", color=col, ms=3, mfc="none", lw=0)
    r = np.geomspace(2e-3, 2e-2, 10)
    ax.loglog(r, 0.35 * r ** (lam - 1), "--", color=INK2, lw=1)
    ax.text(6e-3, 0.35 * 6e-3 ** (lam - 1) * 1.12, "slope $\\lambda_I - 1 = %.3f$" % (lam - 1), color=INK2, fontsize=8)
    ax.set_xlabel("distance from the bond tip $r$ (mm)")
    ax.set_ylabel("$t_n$ per unit macro stress")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.text(0.02, -0.3, "open: within one tip element (replaced by the fitted asymptote)", transform=ax.transAxes,
            fontsize=6.5, color=INK2)
    save(fig, out, "fig_profiles")


def fig_envelope(s00, out, T):
    env = s00["results"]["T=%g" % T]["envelope"]["23"]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    colors = {"lat": S1, "int": S2}
    cap = 2.0 * max(np.hypot(p[0], p[1]) for p in env if p[0] >= 0 and p[1] >= 0)
    ok = [np.hypot(p[0], p[1]) <= cap for p in env]   # bonds never fail in biaxial compression
    i0 = ok.index(False)                                # start the curve after a dropped direction: no false chord
    order = list(range(i0, len(env))) + list(range(i0))
    line = [(env[i][0], env[i][1]) if ok[i] else (np.nan, np.nan) for i in order]
    ax.plot([p[0] for p in line], [p[1] for p in line], color=INK2, lw=1, zorder=1)
    keep = [p for p, k in zip(env, ok) if k]
    xs, ys, fam = [p[0] for p in keep], [p[1] for p in keep], [p[2] for p in keep]
    ax.text(0.03, 0.03, "no bond failure in\nbiaxial compression\n(filament limit)", transform=ax.transAxes,
            fontsize=7, color=INK2)
    for f, lab in (("lat", "lateral bond (T mode)"), ("int", "interlayer bond (L mode)")):
        m = [i for i, v in enumerate(fam) if v == f]
        ax.scatter([xs[i] for i in m], [ys[i] for i in m], s=10, color=colors[f], label=lab, zorder=2)
    ax.axhline(0, color=INK2, lw=0.5)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlabel("$\\bar\\sigma_{22}$ (MPa)")
    ax.set_ylabel("$\\bar\\sigma_{33}$ (MPa)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", fontsize=7, bbox_to_anchor=(0, 1.0))
    save(fig, out, "fig_envelope")


def strength_rows(s00, s090, rows):
    t = thesis()
    tz = t["directional_Tz"]
    rows.append("### Bond strength (finite fracture mechanics), Biswas Table 8 cell, weld T and Gc %.2f N/mm\n" % s00["Gc"])
    rows.append("| cell | T (MPa) | Y | Z | S23 | S13 | S12 | thesis Z (p. 33 / 37) |")
    rows.append("|---|---|---|---|---|---|---|---|")
    for lab, s, key in (("0/0", s00, "0/0"), ("0/90", s090, "0/90")):
        if s is None:
            continue
        for Tk, r in s["results"].items():
            u = r["uniaxial"]
            v = [u[k]["strength"] for k in ("Y (22)", "Z (33)", "S23", "S13", "S12")]
            T = float(Tk[2:])
            ref = tz[key + " L33"][0] if abs(T - tz["T_z"]) < 1e-6 else t["table9_strength"][key]["L33"][0]
            rows.append("| %s | %g | %s | %.2f |" % (lab, T, " | ".join("%.2f" % x if x else "-" for x in v), ref))
    rows.append("")
    rows.append("Ligament equilibrium check (tip-corrected mean traction over the whole bond / exact net section): "
                + ", ".join("%s %s %.4f" % (lab, n, f["ligament_check"]) for lab, s in (("0/0", s00), ("0/90", s090))
                            if s for n, f in s["families"].items() if f["ligament_check"]) + ".\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    res = os.path.join(HERE, "data", "results")
    ap.add_argument("--runs", default=res)
    ap.add_argument("--str00", default=os.path.join(res, "strength_00.json"))
    ap.add_argument("--str090", default=os.path.join(res, "strength_090.json"))
    ap.add_argument("--T", type=float, default=15.26)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = ["# Bead-cell RVE results\n"]
    fig_stiffness(a.runs, a.out, rows)
    with open(a.str00) as f:
        s00 = json.load(f)
    s090 = None
    if os.path.exists(a.str090):
        with open(a.str090) as f:
            s090 = json.load(f)
    fig_profiles(s00, a.out)
    fig_envelope(s00, a.out, a.T)
    strength_rows(s00, s090, rows)
    with open(os.path.join(a.out, "table.md"), "w") as f:
        f.write("\n".join(rows))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
