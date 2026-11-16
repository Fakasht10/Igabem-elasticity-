"""Master S-N collapse of every PLA test, plain and notched, from both data sets (Seifollahi & Kabir 2026, every
specimen; Ezeh & Susmel 2019/2020, every series) onto ONE dimensionless curve.

    python3 -m fatigue.master [--out results/fatigue]

Master variable, after the independent review of 2026-11-16 (review_master_collapse.md):
    S* = f_eff sigma_a,net (2/(1 - R))^(1 - gamma) / sigma_plain,ds(N_ref = 1e5)
    log10 S* = b log10 N + c    one slope for all data, residual in log10 STRESS at the observed life
f_eff = sigma_eff / sigma_net is the criterion's computed notch factor from the linear-elastic unit field (pure number):
    net 1 | kt Kt | hotspot max(1, Kt / (1 + sqrt(l chi_0))) | tcd_point sigma_1(L/2) | tcd_line mean over 2L |
    weibull (J_m / V_ref)^(1/m)
sigma_plain,ds(N_ref) is each data set's own plain level at 1e5 (one intercept per data set, fitted with the slope);
the ratio of the two levels is reported, not hidden in the residual: printer, raster and filament differ.
Why stress direction: fitting life on stress across many series halves the slope (regression dilution).
gamma: one Walker exponent for all tests, fitted (only Ezeh-Susmel spans R). A separate gamma for plain and notched
would change Kf between plain and notched specimens at the SAME R, so it is not a stress parameter.
Outliers, flagged and excluded from the fit, residuals reported: Ezeh-Susmel blunt notch (notched stronger than plain,
Kf 0.74, in 5 of 6 matched p/R pairs; no elastic criterion gives Kf < 1) and plain p30 R-1 (weaker than p30 R0).
Ezeh-Susmel series are Basquin lines sampled at 4 lives in 2e4..2e6 (weight n/4); their plain gauge length (Weibull
only) is assumed 30 mm.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from . import criteria as C
from .fields import field
from .specimens import Specimen
from .transfer import HERE, OUT

V_REF = 1000.0                     # mm^3, reference volume of the Weibull effective stress (cancels in the collapse)
N_EZ = np.geomspace(2e4, 2e6, 4)


def load(gauge_ezeh=30.0):
    """Rows: (data set, group, field key or None, s_a net, R, N, weight, plain volume, outlier, series id)."""
    rows, F = [], {}
    S = json.load(open(os.path.join(HERE, "data", "seifollahi_kabir.json")))
    ps = S["plain_specimen"]
    Vs = ps["W"] * ps["T"] * 50.0
    for k, v in S["specimens"].items():
        F[("S", k)] = field(Specimen.from_dict(v), h=1.5, log=lambda *_: None)
    for k, pts in S["raw_lives"].items():
        for s, *ns in pts:
            for n in ns:
                rows.append(("Seifollahi", k, None if k == "smooth" else ("S", k), s, 0.05, n, 1.0, Vs, False, None))
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    pe = E["plain_specimen"]
    Ve = pe["W"] * pe["T"] * gauge_ezeh
    for k, v in E["specimens"].items():
        F[("E", k)] = field(Specimen.from_dict(v), h=1.5, log=lambda *_: None)
    for i, q in enumerate(E["plain"] + E["notched"]):
        g = q.get("notch", "plain")
        out = g == "blunt" or (g == "plain" and q["p"] == 30 and q["R"] == -1)
        s50 = q.get("sigma_nA50", q.get("sigma_A50"))
        for N in N_EZ:
            rows.append(("Ezeh-Susmel", g, None if g == "plain" else ("E", g), s50 * (N / E["N_ref"]) ** (-1 / q["k"]),
                         q["R"], N, q["n"] / len(N_EZ), Ve, out, (i, q["T"], s50, q["p"])))
    return rows, F


def factor(name, f, th, Vp):
    """f_eff of a notched field f (None = plain bar of volume Vp) for criterion `name`, parameters th."""
    if name == "net":
        return 1.0
    if name in ("weibull", "sweibull"):
        m = th[0]
        if f is None:
            return (Vp / V_REF) ** (1 / m)
        return 1.0 / (C.weibull(f, 1.0, m, V_REF) if name == "weibull" else C.weibull_support(f, 1.0, m, V_REF, th[1]))
    if f is None:
        return 1.0
    if name == "kt":
        return float(C.line_profile(f)[1][0])
    if name == "hotspot":                      # capped: gradient support cannot make a notch stronger than plain
        return max(1.0, 1.0 / C.supported_kt(f, 1.0, th[0]))
    lig = float(C.line_profile(f)[0][-1])
    if name == "tcd_point":
        return 1.0 / C.tcd_point(f, 1.0, min(th[0], 1.9 * lig))
    if name == "tcd_line":
        return 1.0 / C.tcd_line(f, 1.0, min(th[0], 0.95 * lig))
    raise KeyError(name)


SPEC = {   # name: (parameter names, start, log-transformed bounds)
    "net": ([], []), "kt": ([], []), "hotspot": (["l"], [1.0]), "tcd_point": (["L"], [1.0]),
    "tcd_line": (["L"], [1.0]), "weibull": (["m"], [10.0]), "sweibull": (["m", "l"], [20.0, 0.5])}



SPEC = {   # name: (parameter names, start values)
    "net": ([], []), "kt": ([], []), "hotspot": (["l"], [1.0]), "tcd_point": (["L"], [1.0]),
    "tcd_line": (["L"], [1.0]), "weibull": (["m"], [10.0]), "sweibull": (["m", "l"], [20.0, 0.5])}
DATASETS = ("Seifollahi", "Ezeh-Susmel")
N_REF = 1e5


def logS(name, th, gamma, rows, F, cache):
    out = []
    for d, g, key, s, R, n, wt, Vp, *_ in rows:
        ck = (name, tuple(np.round(th, 9)), key, Vp)
        if ck not in cache:
            fe = factor(name, F[key] if key else None, th, Vp)
            cache[ck] = fe
        out.append(math.log10(cache[ck] * s * (2 / (1 - R)) ** (1 - gamma)))
    return np.array(out)


def score(name, th, gamma, rows, F, cache):
    """One slope b, one intercept per data set, least squares in log10 S over the non-outlier rows (each data set
    weighted equally). Returns objective, rms per data set (log10 S), (b, intercepts), all residuals."""
    y = logS(name, th, gamma, rows, F, cache)
    x = np.log10([r[5] for r in rows]) - math.log10(N_REF)
    ds = np.array([r[0] for r in rows])
    w = np.array([r[6] for r in rows])
    keep = ~np.array([r[8] for r in rows])
    for d in DATASETS:                                         # each data set carries total weight 1
        k = ds == d
        w[k] = w[k] / w[k & keep].sum()
    X = np.column_stack([x] + [(ds == d).astype(float) for d in DATASETS])
    W = np.sqrt(w * keep)
    coef = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
    r = y - X @ coef
    per = {d: float(np.sqrt(np.average(r[(ds == d) & keep] ** 2, weights=w[(ds == d) & keep]))) for d in DATASETS}
    return float(np.mean([v ** 2 for v in per.values()])), per, coef, r


def fit(name, rows, F, gamma=None):
    """Parameters (log space) and gamma (logistic, if None) minimising the objective."""
    cache = {}
    names, th0 = SPEC[name]
    free_g = gamma is None
    def unpack(z):
        lo, hi = {"m": 3.0, "l": 1e-3, "L": 1e-2}, {"m": 1e3, "l": 20.0, "L": 10.0}      # physical ranges
        th = [min(max(math.exp(v), lo[n]), hi[n]) for v, n in zip(z[:len(th0)], names)]
        return th, (1 / (1 + math.exp(-z[-1])) if free_g else gamma)
    z0 = [math.log(v) for v in th0] + ([0.0] if free_g else [])
    if not z0:
        return [], gamma, score(name, [], gamma, rows, F, cache)
    obj = lambda z: score(name, *unpack(z), rows, F, cache)[0]
    best = min((minimize(obj, [v + math.log(sc) for v in z0[:len(th0)]] + z0[len(th0):], method="Nelder-Mead",
                         options=dict(xatol=1e-3, fatol=1e-8, maxiter=500)) for sc in (0.3, 1.0, 3.0)),
               key=lambda o: o.fun)
    th, g = unpack(best.x)
    return th, g, score(name, th, g, rows, F, cache)


def plain_floor(rows, F, gamma):
    """Same fit with plain rows only: the scatter no notch criterion can remove."""
    pr = [r for r in rows if r[2] is None]
    return score("net", [], gamma, pr, F, {})[1]


def figure(res, rows, path, best, ref="net"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(13.5, 6.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    th, g, J, per, coef, r, y = res[best]
    b, cS, cE = coef
    lev = {"Seifollahi": cS, "Ezeh-Susmel": cE}
    sd = math.sqrt(np.mean([v ** 2 for v in per.values()]))
    Ns = np.geomspace(3e3, 3e6, 40)
    line = 10 ** (b * (np.log10(Ns) - math.log10(N_REF)))
    ax.fill_between(Ns, line * 10 ** (-2 * sd), line * 10 ** (2 * sd), color="0.9", lw=0)
    ax.plot(Ns, line, "k-", lw=1.6)
    col = {"Seifollahi": "#d62728", "Ezeh-Susmel": "#1f77b4"}
    mk = {"smooth": "o", "plain": "o", "edge": "^", "hole": "D", "intermediate": "s", "sharp": "v", "blunt": "P"}
    for i, row in enumerate(rows):
        d, gname, key, s, R, n, wt, Vp, out, sid = row
        yy = 10 ** (y[i] - lev[d])
        if d == "Seifollahi":
            ax.scatter(n, yy, marker=mk[gname], s=34, facecolor="none" if key is None else col[d], edgecolors=col[d],
                       linewidths=1.1, zorder=5)
    series = {}
    for i, row in enumerate(rows):
        if row[0] == "Ezeh-Susmel":
            series.setdefault(row[9][0], []).append(i)
    for idx in series.values():
        row = rows[idx[0]]
        d, gname, key, *_ , out, sid = row
        Nn = [rows[i][5] for i in idx]
        yy = [10 ** (y[i] - lev[d]) for i in idx]
        c = "0.55" if out else col[d]
        T = sid[1]
        xj = Nn[-1] * math.exp(0.25 * (((sid[0] * 7) % 11) / 10 - 0.5))      # spread the 2e6 column
        ax.errorbar(xj, yy[-1], yerr=[[yy[-1] * (1 - T ** -0.5)], [yy[-1] * (T ** 0.5 - 1)]], fmt=mk[gname],
                    ms=5, mfc="none" if key is None else c, mec=c, ecolor=c, elinewidth=0.6, capsize=0, zorder=4)
    ax.set(xscale="log", yscale="log", xlim=(3e3, 3e6), ylim=(0.25, 4),
           xlabel="cycles to failure $N_f$", ylabel=r"$S^* = f_{eff}\,\sigma_{a,n}(2/(1-R))^{1-\gamma}\,/\,\sigma_{plain}(10^5)$",
           title=f"one master curve: {TITLE[best]}, $\\gamma$ {g:.2f}, slope {b:.3f} (k {-1 / b:.1f})")
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 0.7, 1, 1.5, 2, 3]))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.3, which="both")
    h = [Line2D([], [], color="k", lw=1.6), Line2D([], [], color="0.9", lw=8),
         Line2D([], [], color=col["Seifollahi"], marker="o", mfc="none", ls=""),
         Line2D([], [], color=col["Seifollahi"], marker="^", ls=""), Line2D([], [], color=col["Seifollahi"], marker="D", ls=""),
         Line2D([], [], color=col["Ezeh-Susmel"], marker="o", mfc="none"),
         Line2D([], [], color=col["Ezeh-Susmel"], marker="s"), Line2D([], [], color=col["Ezeh-Susmel"], marker="v"),
         Line2D([], [], color="0.55", ls="", marker="P")]
    lb = ["master curve", "±2 SD", "Seifollahi smooth", "Seifollahi edge R2", "Seifollahi hole R2",
          "Ezeh-Susmel plain: one point per series (reported 2e6 value, 10-90 % bar)", "Ezeh-Susmel U r1", "Ezeh-Susmel V r0.15",
          "flagged outliers: Ezeh blunt U r3, plain p30 R-1"]
    ax.legend(h, lb, fontsize=8, loc="lower left", ncol=2, framealpha=0.95)
    # residual strip: mean residual per notch geometry against Kt, best criterion vs no notch factor
    for name, mstyle in ((ref, "o"), (best, "s")):
        _, _, _, _, _, rr, _ = res[name]
        pts = {}
        for i, row in enumerate(rows):
            if row[2] is not None:
                pts.setdefault((row[0], row[1], row[8]), []).append(rr[i])
        for (d, gname, out), v in pts.items():
            kt = KT[(d, gname)]
            axr.scatter(kt, np.mean(v), marker=mstyle, s=40, color=col[d] if not out else "0.55",
                        facecolor="none" if name == ref else (col[d] if not out else "0.55"), zorder=3)
            axr.annotate(gname, (kt, np.mean(v)), fontsize=7, xytext=(4, 2), textcoords="offset points")
    axr.axhline(0, color="k", lw=0.8)
    axr.set(xscale="log", xlabel="$K_t$", ylabel="mean residual, log10 S*",
            title=f"residual vs Kt: hollow = net stress, filled = {TITLE[best]}")
    axr.title.set_fontsize(9)
    from matplotlib.ticker import FixedLocator as FL, ScalarFormatter as SF, NullFormatter as NF
    axr.xaxis.set_major_locator(FL([1.5, 2, 3, 4, 6])); axr.xaxis.set_major_formatter(SF())
    axr.xaxis.set_minor_formatter(NF())
    axr.grid(alpha=0.3, which="both")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


TITLE = {"net": "net stress", "kt": "Kt sigma_n", "hotspot": "supported hot spot (Kf >= 1)",
         "tcd_point": "TCD point", "tcd_line": "TCD line", "weibull": "Weibull", "sweibull": "supported Weibull"}
KT = {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    rows, F = load()
    for (d, k), f in F.items():
        KT[("Seifollahi" if d == "S" else "Ezeh-Susmel", k)] = float(C.line_profile(f)[1][0])
    lines = ["# One master S-N curve for all PLA tests: stress-direction fit, one slope, one level per data set", "",
             "| criterion | gamma | parameters | rms log10 S Seifollahi | rms log10 S Ezeh-Susmel | slope (k) | "
             "level Seif/Ezeh | outliers mean residual (blunt; plain p30 R-1) |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for name in ("net", "kt", "hotspot", "tcd_point", "tcd_line", "weibull"):
        th, g, (J, per, coef, r) = fit(name, rows, F)
        y = logS(name, th, g, rows, F, {})
        res[name] = (th, g, J, per, coef, r, y)
        par = ", ".join(f"{p} {v:.3g}" for p, v in zip(SPEC[name][0], th)) or "-"
        ob = np.mean([ri for ri, row in zip(r, rows) if row[8] and row[1] == "blunt"])
        op = np.mean([ri for ri, row in zip(r, rows) if row[8] and row[1] == "plain"])
        lines.append(f"| {name} | {g:.2f} | {par} | {per['Seifollahi']:.3f} | {per['Ezeh-Susmel']:.3f} | "
                     f"{coef[0]:.3f} ({-1 / coef[0]:.1f}) | {10 ** (coef[1] - coef[2]):.2f} | {ob:+.2f}; {op:+.2f} |")
    best = min((n for n in res if n != "net"), key=lambda n: res[n][2])
    fl = plain_floor(rows, F, res[best][1])
    lines += ["", f"Plain-only floor (same fit, plain tests only): Seifollahi {fl['Seifollahi']:.3f}, Ezeh-Susmel "
              f"{fl['Ezeh-Susmel']:.3f} log10 S. Best criterion by the objective: {best}. rms log10 life ~ k x rms log10 S."]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "master_collapse.md"), "w").write("\n".join(lines) + "\n")
    json.dump({n: dict(th=v[0], gamma=v[1], rms=v[3], slope=v[4][0], levels=list(v[4][1:])) for n, v in res.items()},
              open(os.path.join(a.out, "master_collapse.json"), "w"), indent=1)
    figure(res, rows, os.path.join(a.out, "fig_master_collapse"), best)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
