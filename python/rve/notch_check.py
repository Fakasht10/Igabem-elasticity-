"""Weld-onset card on the Cicero et al. 2020 ABS SENB (three cross-ply layups x four root radii, 12 mean loads):
rve.onset (bead-cell weld families + filament, TCD) against the ply-mode card of failure.cicero_orient.

Both cards have four parameters fitted to the same 12 loads (log least squares), scored in sample and leave one
layup out:
  ply modes   X, Y, Y/S12, L          modes F and T on the ply stress (failure.aniso)
  weld modes  X, T, T/S, L            the weld families of the 0/90 bead cell on the macro stress (rve.onset)
The print geometry of the coupons is not reported; the bead cell is the module's ABS base geometry (w 0.6, h 0.3,
b_int 0.36, b_lat 0.10 mm, notch 15 deg) and --blat / --bint vary it. The SENB lines are failure.cicero_orient's
(one IgBEM solve per radius; the stress does not depend on the layup).

usage (from python/): python3 -m rve.notch_check [--D net|0.3] [--method point|line] [--kappa k]
  runtime: ~4 min the first time (the four SENB lines), then ~40 s (fits; the cell response is in the library)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from failure.cicero_orient import LAYUPS, RADII, line  # noqa: E402
from failure.validate_cicero import MEASURED_P_CRIT  # noqa: E402
from rve import onset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def predict(lines, wmap, x, layups, method):
    X, T, r, L = x
    out = {}
    for name in layups:
        R = onset.cell_frame(LAYUPS[name][0])
        for rho in RADII:
            f, gov, _ = onset.onset(lines[rho]["sig"], lines[rho]["d"], wmap, R, T, r, L, X, method)
            out[(name, rho)] = (f, gov)
    return out


def fit(lines, wmap, layups, method):
    keys = [(n, r) for n in layups for r in RADII]
    obs = np.log([MEASURED_P_CRIT[n][r] for n, r in keys])

    def loss(z):
        p = predict(lines, wmap, np.exp(z), layups, method)
        return float(np.sum((obs - np.log([p[k][0] for k in keys])) ** 2))

    best = None
    for x0 in ([60.0, 150.0, 1.0, 0.4], [80.0, 120.0, 2.0, 0.3], [55.0, 200.0, 0.7, 0.6]):
        res = minimize(loss, np.log(x0), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 6000})
        best = res if best is None or res.fun < best.fun else best
    return np.exp(best.x)


def run(a):
    work = os.path.abspath(os.path.join(ROOT, "cache", "failure"))
    lines = {rho: line(rho, work) for rho in RADII}
    g = dict(w=a.w, h=a.h, b_int=a.bint, b_lat=a.blat, alpha=15.0, wc=math.sqrt(0.5))
    wmap = onset.weld_map("090", g, 0.35, a.kappa, a.D if a.D == "net" else float(a.D), grade=2, work=a.work)
    names = list(LAYUPS)
    x = fit(lines, wmap, names, a.method)
    ins = predict(lines, wmap, x, names, a.method)
    loo = {}
    for n in names:
        loo.update(predict(lines, wmap, fit(lines, wmap, [m for m in names if m != n], a.method), [n], a.method))
    rows = [dict(layup=n, rho=rho, measured_N=MEASURED_P_CRIT[n][rho], weld_N=ins[(n, rho)][0],
                 governing=ins[(n, rho)][1], weld_loo_N=loo[(n, rho)][0]) for n in names for rho in RADII]
    err = lambda k: [100 * (r[k] / r["measured_N"] - 1) for r in rows]  # noqa: E731
    summary = {k: dict(min_pct=min(err(k)), max_pct=max(err(k)), rms_pct=float(np.sqrt(np.mean(np.square(err(k))))))
               for k in ("weld_N", "weld_loo_N")}
    ref = json.load(open(os.path.join(ROOT, "results", "failure", "cicero_orient.json")))
    return dict(params=dict(zip(["X_MPa", "T_MPa", "T_over_S", "L_mm"], x.tolist())), geometry=g, kappa=a.kappa,
                D=a.D, method=a.method, rows=rows, summary=summary,
                ply_modes=dict(params=ref["params"], summary={k: ref["summary"][k] for k in ("modes_N", "modes_loo_N")}),
                maps={f: dict(N=m["N"].tolist(), S=m["S"].tolist()) for f, m in wmap.items()})


def gradient_terms(lines, a=0.3, comp=(0, 0)):
    """Across one cell of half size a centred at d on the bisector: the linear term (s(d+a) - s(d-a)) / 2 s(d) and the
    curvature term (s(d+a) + s(d-a)) / 2 s(d) - 1 of the stress component comp (default s_xx, the SENB opening
    stress). The second-order (gradient) correction of a periodic cell is an expansion in these; d >= a only."""
    out = {}
    for rho, ln in lines.items():
        dd = np.geomspace(a, 1.5, 50)   # beyond ~2 mm s_xx turns towards the neutral axis
        f = lambda x: np.interp(x, ln["d"], ln["sig"][:, comp[0], comp[1]])  # noqa: E731
        out[rho] = dict(d=dd, lin=(f(dd + a) - f(dd - a)) / (2 * f(dd)), curv=(f(dd + a) + f(dd - a)) / (2 * f(dd)) - 1)
    return out


def figure(res, lines, out):
    from rve.figures import INK, INK2, S1, S2, plt, save
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    fig.subplots_adjust(wspace=0.32)
    ax = axs[0]
    ref = {(r["layup"], r["rho"]): r for r in json.load(open(os.path.join(ROOT, "results", "failure",
                                                                           "cicero_orient.json")))["rows"]}
    marks = dict(zip(LAYUPS, ("o", "^", "s")))
    for r in res["rows"]:
        m = r["measured_N"]
        ax.plot(m, r["weld_loo_N"], marks[r["layup"]], color=S1, ms=4.5)
        ax.plot(m, ref[(r["layup"], r["rho"])]["modes_loo_N"], marks[r["layup"]], color=S2, ms=4.5, mfc="white")
    lo, hi = 80, 125
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
    for f in (0.9, 1.1):
        ax.plot([lo, hi], [lo * f, hi * f], color=INK2, lw=0.6, ls=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("measured critical load (N)")
    ax.set_ylabel("predicted, leave one layup out (N)")
    for name, mk in marks.items():
        ax.plot([], [], mk, color=INK2, ms=4, label=name)
    ax.plot([], [], "o", color=S1, ms=4.5, label="weld modes  %.1f %%" % res["summary"]["weld_loo_N"]["rms_pct"])
    ax.plot([], [], "o", color=S2, mfc="white", ms=4.5,
            label="ply modes  %.1f %%" % res["ply_modes"]["summary"]["modes_loo_N"]["rms_pct"])
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axs[1]
    gt = gradient_terms(lines)
    shades = dict(zip(RADII, ["#0f2f57", "#1d559a", S1, "#9cc3ee"]))
    for rho, v in gt.items():
        ax.loglog(v["d"], np.abs(v["curv"]), color=shades[rho], lw=1.3, label="$\\rho$ = %g mm" % rho)
    L2 = 0.5 * res["params"]["L_mm"]
    ax.axvspan(0.1, 0.3, color=INK2, alpha=0.08, lw=0)
    ax.axvline(L2, color=S2, lw=1.0, ls="--")
    ax.text(L2 * 0.93, 0.5, "TCD point L/2", color=S2, fontsize=7, rotation=90, va="center", ha="right")
    ax.text(0.11, 0.012, "a cell here crosses\nthe notch surface", color=INK2, fontsize=6.5, va="bottom")
    ax.set_xlim(0.1, 1.5)
    ax.set_xticks([0.1, 0.3, 0.5, 1.0, 1.5], ["0.1", "0.3", "0.5", "1", "1.5"])
    ax.minorticks_off()
    ax.set_ylim(0.01, 1.5)
    ax.set_xlabel("distance from the notch root  d (mm)")
    ax.set_ylabel("$|(s(d+a) + s(d-a)) / 2 s(d) - 1|$")
    ax.legend(fontsize=6.5, loc="upper right", title="cell width w = 0.6 mm, a = w/2", title_fontsize=6.5)
    for a_, t in zip(axs, "ab"):
        a_.set_title("(%s)" % t, loc="left", fontsize=8.5, color=INK)
    save(fig, out, "fig_notch_onset")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--bint", type=float, default=0.36)
    ap.add_argument("--blat", type=float, default=0.10)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--D", default="net")
    ap.add_argument("--method", choices=["point", "line"], default="point")
    ap.add_argument("--work", default="/tmp/rve_onset")
    ap.add_argument("--out", default=None, help="directory for notch_onset.json and fig_notch_onset.*")
    a = ap.parse_args(argv)
    res = run(a)
    print("weld card:", {k: round(v, 4) for k, v in res["params"].items()})
    for r in res["rows"]:
        print("%-7s rho %-4g measured %6.1f  weld %6.1f (%s)  LOO %6.1f" % (
            r["layup"], r["rho"], r["measured_N"], r["weld_N"], r["governing"], r["weld_loo_N"]))
    for k, v in list(res["summary"].items()) + [("ply " + k, v) for k, v in res["ply_modes"]["summary"].items()]:
        print("%-18s %+6.1f .. %+6.1f %%  rms %.1f %%" % (k, v["min_pct"], v["max_pct"], v["rms_pct"]))
    if a.out:
        with open(os.path.join(a.out, "notch_onset.json"), "w") as f:
            json.dump(res, f, indent=1)
        lines = {rho: line(rho, os.path.abspath(os.path.join(ROOT, "cache", "failure"))) for rho in RADII}
        figure(res, lines, a.out)
    return res


if __name__ == "__main__":
    main()
