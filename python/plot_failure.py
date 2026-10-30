"""Figures of the failure layer on IgBEM (python/failure):

  (a) Cicero SENB: first principal stress per newton along the extraction path from the notch root (exact IgBEM
      line), the four root radii, with the probe distance L/2 of the 0/90 raster
  (b) Cicero SENB: failure loads, measured (Table 2) against the TCD point method on IgBEM with sigma_0 from the
      paper's K_mat and with the material card's K_Ic and sigma_uts (results/failure/cicero_card.json; without it,
      sigma_0 calibrated on rho = 1 mm)
  (c) Weibull card parity: log J(m) of the PLA UB and US quarter specimens on IgBEM against the NGSolve TET4
      reductions the frozen PLA cards were fitted on; the card's m marked

usage: python3 python/plot_failure.py [results/fig_failure.png]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "results", "failure")
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = ["#0969da", "#cf222e", "#1a7f37", "#8250df"]


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "fig_failure.png")
    cic = json.load(open(os.path.join(D, "cicero.json")))
    par_path = os.path.join(D, "weibull_parity.json")
    par = json.load(open(par_path)) if os.path.exists(par_path) else None
    fig, ax = plt.subplots(1, 3 if par else 2, figsize=(14 if par else 9.5, 4.2))

    a = ax[0]
    for k, r in enumerate(cic["rows"]):
        x, y = np.asarray(r["line_r"]), np.asarray(r["line_s1"])
        a.plot(x[x > 0], y[x > 0], color=C[k], lw=1.4, label=f"$\\rho$ = {r['rho']:g} mm")
    L = cic["rows"][0]["L"]
    a.axvline(L / 2, color=MUTED, lw=0.8, ls="--")
    a.text(L / 2 * 1.06, a.get_ylim()[1] * 0.92, "L/2", color=MUTED, fontsize=8)
    a.set_xscale("log")
    a.set_xlabel("distance from the notch root r (mm)", fontsize=9)
    a.set_ylabel("$\\sigma_1$ per unit load (MPa/N)", fontsize=9)
    a.set_title("(a) SENB: stress along the extraction path", fontsize=9, loc="left")
    a.legend(fontsize=8, frameon=False)
    style(a)

    b = ax[1]
    rho = [r["rho"] for r in cic["rows"]]
    xs = np.arange(len(rho))
    w = 0.26
    meas = [r["measured_N"] for r in cic["rows"]]
    pred = [r["predicted_N"] for r in cic["rows"]]
    card_path = os.path.join(D, "cicero_card.json")
    if os.path.exists(card_path):    # third bar: the material card's K_Ic and sigma_uts, no calibration
        cc = json.load(open(card_path))
        p = cc["props"]
        third = [r["predicted_N"] for r in cc["rows"]]
        third_err = [r["error_pct"] for r in cc["rows"]]
        third_label = (f"TCD PM, {p['source'].split(',')[0]} ($K_{{Ic}}$ {p['K']:g} MPa$\\sqrt{{mm}}$, "
                       f"$\\sigma_0$ {p['sigma_0']:g} MPa)")
    else:                            # third bar: sigma_0 calibrated on one radius
        third = [r.get("predicted_cal_N", np.nan) for r in cic["rows"]]
        third_err = [r.get("error_cal_pct", np.nan) for r in cic["rows"]]
        third_label = f"TCD PM, $\\sigma_0$ calibrated on $\\rho$ = {cic['calibrate_on']:g}"
    b.bar(xs - w, meas, w, color=MUTED, label="measured (Cicero 2020, Table 2)")
    b.bar(xs, pred, w, color=C[1], label="TCD PM, $\\sigma_0$ from the paper's $K_{mat}$")
    b.bar(xs + w, third, w, color=C[0], label=third_label)
    for i, r in enumerate(cic["rows"]):
        b.text(xs[i] + w, third[i] + 2, f"{third_err[i]:+.0f}%", ha="center", fontsize=7, color=C[0])
        b.text(xs[i], pred[i] + 2, f"{r['error_pct']:+.0f}%", ha="center", fontsize=7, color=C[1])
    b.set_xticks(xs, [f"{v:g}" for v in rho])
    b.set_xlabel("notch root radius $\\rho$ (mm)", fontsize=9)
    b.set_ylabel("failure load (N)", fontsize=9)
    b.set_title(f"(b) SENB {cic['orientation']}: failure load, IgBEM", fontsize=9, loc="left")
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.set_ylim(0, 1.35 * max(pred))
    style(b)

    if par:
        c = ax[2]
        for k, r in enumerate(par["rows"]):
            m = np.asarray(r["m"])
            ratio = np.exp(-(np.asarray(r["logJ_igabem"]) - np.asarray(r["logJ_ngsolve_tet4"])) / m)
            c.plot(m, ratio, color=C[k + 2], lw=1.4, marker="o", ms=3,
                   label=f"{r['family']} ($\\rho$ = {r['rho']:g} mm): x{r['mean_load_ratio_at_card_m']:.3f} at card m")
        mc = par["rows"][0]["m_card"]
        c.axvline(mc, color=MUTED, lw=0.8, ls=":")
        c.axhline(1.0, color=INK, lw=0.6)
        c.text(mc * 1.05, 0.985, f"card m = {mc:.2f}", color=MUTED, fontsize=8)
        c.set_xscale("log")
        c.set_xlim(1, 30)
        c.set_ylim(0.97, 1.3)
        c.set_xlabel("Weibull modulus m", fontsize=9)
        c.set_ylabel("mean failure load, IgBEM / NGSolve TET4", fontsize=9)
        c.set_title("(c) PLA Weibull cards: load parity, exp(-$\\Delta$log J / m)", fontsize=9, loc="left")
        c.legend(fontsize=7, frameon=False, loc="upper left")
        style(c)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print(out)


if __name__ == "__main__":
    main()
