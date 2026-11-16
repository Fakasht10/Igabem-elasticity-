"""Why the modified HSV (Zhang, Zhang, Hu, Chen, Susmel & Wei, IJF 172 (2023) 107644) fails on Ezeh & Susmel 2020
notched printed PLA, and its best honest version, under the common protocol of fatigue.calib (fit on plain Fig. 1
P50 + the 47 failed sharp V r0.15 specimens; predict the 102 failed U r1 and U r3 specimens, held out).

    python3 -m fatigue.hsv_fix [--out results/fatigue] [--gauge 30] [--L-fci 0.4]

The paper's prediction (eq. 10, with eqs 3, 5, 7, 8, 9):
    N_N(sigma_eff) = N_S(sigma_eff) (S_S L_FCI / (S_N L_CD))^(1/k),   sigma_eff = sigma_1(L_CD / 2)  (point method)
with k the Weibull shape of the plain LIFE distribution at one stress level (Sec. 3.2.2: smooth lives moved to one
stress level along the P50 S-N curve, then fitted with a Weibull law). N_S comes from the plain P50 curve, here the
Fig. 1 unified curve of slope k_SN = 5.1, so in the protocol's stress residual the size term is a stress factor
    f_eff = sigma_1(L_CD / 2) / sigma_n x (S_S L_FCI / (S_N L_CD))^(-e),   e = 1 / (k k_SN)            (exact: power law)

Inputs (provenance):
    L_CD 0.489 mm      authors' L_M(N) = 16.4 N^-0.242 at 2e6 (2020 eq. 6), fitted by them on plain + sharp V (eq. 4
                       cannot be evaluated: dK_th of this PLA is not reported). Variant: L_CD fitted here on sharp V.
    L_FCI              NOT measured (no plain fractography in either PLA paper). ASSUMED 0.4 mm (shell thickness =
                       embryonic crack length on the notched fracture surfaces, 2020 Sec. 3), or FITTED on sharp V.
    S_N                IgBEM sigma_1 on the notch-root surfaces, area >= 0.9 peak, full specimen (hsv.S90_surface).
    S_S                2 (W + T) L_g = 540 mm^2, W 6, T 3 (2019 paper), gauge L_g 30 mm ASSUMED. With L_FCI fitted,
                       only the product S_S L_FCI matters, so L_g drops out.
    k (life shape)     from PLAIN data only, five sources (each a variant, every one reported):
        band           T_sigma 2.451 of the Fig. 1 unified band (what hsv.py / calib.py use): k = ln 21.85 / (5.1 ln T)
        fig1_mle       the paper's own procedure on the Fig. 1 points: each life moved to one stress level along the P50
                       curve (slope 5.1), two-parameter Weibull MLE of those lives
        series         2019 Table 1 plain series at the specimen's own print angle p and R (same printer, filament,
                       settings): k = ln 21.85 / (k_SN,series ln T_series), one k per specimen
        series_stress  same series, but the size term mapped with the series' own stress Weibull modulus
                       m = ln 21.85 / ln T (e = 1/m) instead of through the life domain and the 5.1 slope
        series_pooled  the twelve 2019 series pooled (life log-variance weighted by n - 1)
        sharp_scatter  NOT the paper's source (it uses smooth lives): Weibull MLE of the sharp-V lives about each
                       sharp series' own log-log fit (fitted on sharp V only)
    Ln 21.85 = ln(ln 10 / ln(10/9)): a Weibull life law has N(P_S 10 %) / N(P_S 90 %) = 21.85^(1/k).
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import weibull_min

from .band import K_BAND, P50, T_BAND
from .calib import Notch
from .criteria import Z90
from .hsv import L_CD, W21, load
from .transfer import HERE, OUT

NAMES = {"sharp": "V r0.15", "intermediate": "U r1", "blunt": "U r3"}
HELD = ("intermediate", "blunt")
REF = {"net stress, no notch factor": 0.144, "TCD line, constant L": 0.139, "Weibull, m fitted on sharp V": 0.123}
KSRC = {"band": "band T 2.451 (Fig. 1, as in hsv.py)", "fig1_mle": "Fig. 1 plain points, Weibull MLE (paper's procedure)",
        "series": "2019 series at own p, R", "series_stress": "2019 series at own p, R, stress modulus",
        "series_pooled": "2019 series pooled",
        "sharp_scatter": "sharp-V life scatter (NOT paper's source)"}


# ----------------------------------------------------------------------------------------------- life Weibull shape
def k_from_T(T, k_sn):
    """Life Weibull shape from a stress scatter ratio T = s10/s90 of a parallel band of S-N slope k_sn: T_N = T^k_sn."""
    return W21 / (k_sn * math.log(T))


def k_weibull_mle(lives):
    """Two-parameter Weibull MLE shape of lives at one stress level (any common scale)."""
    x = np.asarray(lives, float)
    return float(weibull_min.fit(x / np.median(x), floc=0)[0])


def k_fig1(points):
    """Paper Sec. 3.2.2 on the Fig. 1 plain points: N_i (S_i / P50(N_i))^5.1 = life of point i at its P50-equivalent
    stress level, the same for all points; Weibull MLE of those lives."""
    return k_weibull_mle([math.exp(K_BAND * math.log(q["smax_uts"] / P50(q["N"]))) for q in points])


def k_pooled(series):
    """Pooled life log-std over plain series [(T, k_sn, n)], s_i = k_sn ln T / (2 z90); k = ln 21.85 / (2 z90 s)."""
    v = sum((n - 1) * (k * math.log(T) / (2 * Z90)) ** 2 for T, k, n in series) / sum(n - 1 for _, _, n in series)
    return W21 / (2 * Z90 * math.sqrt(v))


def k_sharp(rows):
    """Weibull MLE of sharp-V lives about each (p, R) series' own least-squares log-log line (sharp V only)."""
    r = []
    for key in sorted({(p, R) for _, p, R, _, _ in rows}):
        s = np.array([x[3] for x in rows if (x[1], x[2]) == key])
        N = np.array([x[4] for x in rows if (x[1], x[2]) == key])
        c = np.polyfit(np.log(s), np.log(N), 1)
        r += list(np.exp(np.log(N) - np.polyval(c, np.log(s))))
    return k_weibull_mle(r)


# ------------------------------------------------------------------------------------------------------ criterion
def feff(n, S90, e, L_cd, S_S_Lfci):
    """Stress factor of eq. 10 at unit net stress: sigma_1(L_CD/2) (S_S L_FCI / (S_N L_CD))^(-e)."""
    return n.pm(L_cd) * (S_S_Lfci / (S90 * L_cd)) ** (-e)


def resid(rows, notches, S90, kmap, L_cd, S_S_Lfci, uts):
    """log10(S_eff / P50(N)) per specimen; kmap(p, R) = life Weibull shape for that specimen."""
    return np.array([math.log10(feff(notches[k], S90[k], 1.0 / (kmap(p, R) * K_BAND), L_cd, S_S_Lfci)
                                * s * 2 / (1 - R) / uts[p] / P50(N)) for k, p, R, s, N in rows])


def fit_variant(cal, notches, S90, kmap, uts, mode, S_S, L_fci):
    """mode: 'assumed' (L_FCI and L_CD fixed: no fit), 'lfci' (L_FCI fitted), 'lfci_lcd' (both fitted on sharp V)."""
    if mode == "assumed":
        return L_CD, S_S * L_fci
    if mode == "lfci":
        o = minimize_scalar(lambda z: np.mean(resid(cal, notches, S90, kmap, L_CD, math.exp(z), uts) ** 2),
                            bounds=(-30, 30), method="bounded", options=dict(xatol=1e-8))
        return L_CD, math.exp(o.x)
    obj = lambda z: np.mean(resid(cal, notches, S90, kmap, math.exp(z[0]), math.exp(z[1]), uts) ** 2)
    o = min((minimize(obj, [math.log(L0), math.log(S_S * L_fci)], method="Nelder-Mead",
                      options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000)) for L0 in (0.1, 0.49, 2.0)), key=lambda o: o.fun)
    return math.exp(o.x[0]), math.exp(o.x[1])


def stats(r):
    return dict(bias=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))),
                band=float(np.mean(abs(r) < math.log10(T_BAND) / 2)))


def setup():
    g, plain, spec = load()
    uts = {p: q["UTS"] for (p, _), q in plain.items()}
    notches = {k: Notch(g[k]["f"], g[k]) for k in g}
    rows = [(k, p, R, s, N) for k, p, R, s, N, ro in spec if not ro]
    fig1 = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_fig1_plain.json")))
    sharp = [r for r in rows if r[0] == "sharp"]
    ks = {"band": k_from_T(T_BAND, K_BAND), "fig1_mle": k_fig1(fig1),
          "series_pooled": k_pooled([(q["T"], q["k"], q["n"]) for q in plain.values()]), "sharp_scatter": k_sharp(sharp)}
    kmaps = {n: (lambda p, R, v=v: v) for n, v in ks.items()}
    kmaps["series"] = lambda p, R: k_from_T(plain[(p, R)]["T"], plain[(p, R)]["k"])
    kmaps["series_stress"] = lambda p, R: k_from_T(plain[(p, R)]["T"], K_BAND)
    return g, plain, rows, uts, notches, ks, kmaps


def run(gauge=30.0, L_fci=0.4):
    g, plain, rows, uts, notches, ks, kmaps = setup()
    S_S = 2 * (6.0 + 3.0) * gauge
    cal = [r for r in rows if r[0] == "sharp"]
    S90 = {k: g[k]["S90"] for k in g}
    out = []
    for src in KSRC:
        for mode in ("assumed", "lfci", "lfci_lcd"):
            Lc, SL = fit_variant(cal, notches, S90, kmaps[src], uts, mode, S_S, L_fci)
            r = {k: resid([x for x in rows if x[0] == k], notches, S90, kmaps[src], Lc, SL, uts) for k in NAMES}
            out.append(dict(src=src, mode=mode, L_CD=Lc, L_FCI=SL / S_S, r=r))
    # e -> 0 limit (k -> infinity): no size term; the fitted L_FCI degenerates into a free global scale on the TCD
    # point stress at L_CD 0.489. NOT the paper's method: reported as the diagnostic limit only.
    sc = 10 ** -np.mean([math.log10(notches["sharp"].pm(L_CD) * s * 2 / (1 - R) / uts[p] / P50(N)) for _, p, R, s, N in cal])
    r = {k: np.array([math.log10(sc * notches[k].pm(L_CD) * s * 2 / (1 - R) / uts[p] / P50(N))
                      for kk, p, R, s, N in rows if kk == k]) for k in NAMES}
    limit = dict(src="none", mode="scale", L_CD=L_CD, scale=sc, r=r)
    return out, limit, ks, g, notches, rows, uts, plain


def decomposition(g, notches, rows, uts, e):
    """Per notch, relative to sharp V (log10): point-method part, size part at exponent e, and what the data show
    (minus the net-stress residual shift; U-notch rows are held-out data, used here for diagnosis only)."""
    net = {k: np.mean([math.log10(s * 2 / (1 - R) / uts[p] / P50(N)) for kk, p, R, s, N in rows if kk == k]) for k in NAMES}
    d = {}
    for k in NAMES:
        d[k] = dict(tcd=math.log10(notches[k].pm(L_CD) / notches["sharp"].pm(L_CD)),
                    size=e * math.log10(g[k]["S90"] / g["sharp"]["S90"]), data=-(net[k] - net["sharp"]))
    return d


def rms_vs_e(notches, g, rows, uts, es):
    """Held-out rms when e is imposed and only the global scale (L_FCI) is fitted on sharp V. Diagnostic curve."""
    out = []
    for e in es:
        base = lambda k, p, R, s, N: math.log10(notches[k].pm(L_CD) * g[k]["S90"] ** e * s * 2 / (1 - R) / uts[p] / P50(N))
        c = np.mean([base(*x) for x in rows if x[0] == "sharp"])
        h = np.array([base(*x) - c for x in rows if x[0] != "sharp"])
        out.append(float(np.sqrt(np.mean(h ** 2))))
    return np.array(out)


BEST = ("series", "lfci")           # chosen BEFORE looking at U notches: the paper's k source (plain lives, same population)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gauge", type=float, default=30.0, help="ASSUMED plain gauge length (mm)")
    ap.add_argument("--L-fci", type=float, default=0.4, help="ASSUMED L_FCI (mm) for the no-fit variants")
    a = ap.parse_args(argv)
    out, limit, ks, g, notches, rows, uts, plain = run(a.gauge, a.L_fci)
    kser = {pr: k_from_T(q["T"], q["k"]) for pr, q in plain.items()}
    L = ["# Modified HSV (Zhang et al. 2023) on Ezeh-Susmel PLA: diagnosis and best honest version", "",
         "Protocol of fatigue.calib: fitted on plain (Fig. 1 P50) + 47 failed sharp V; held out 102 failed U r1 + U r3. "
         "Residual log10(S_eff / P50(N)); band = P10-P90 of T 2.451.", "",
         f"S90 (trapezoid weights, hsv.S90_surface): " + ", ".join(f"{NAMES[k]} {g[k]['S90']:.3f}" for k in NAMES) +
         f" mm^2; sigma_1(L_CD/2)/sigma_n: " + ", ".join(f"{NAMES[k]} {notches[k].pm(L_CD):.3f}" for k in NAMES) + ".", "",
         "Life Weibull shape k (plain data unless marked) and the stress exponent e = 1/(5.1 k) of the size term:", ""]
    for n, v in ks.items():
        L.append(f"- {KSRC[n]}: k {v:.3f}, e {1 / (5.1 * v):.3f}")
    L.append(f"- {KSRC['series']}: k {min(kser.values()):.2f}-{max(kser.values()):.2f}, e "
             f"{1 / (5.1 * max(kser.values())):.3f}-{1 / (5.1 * min(kser.values())):.3f}")
    L += ["", "## Root cause: decomposition relative to the sharp V (log10 of the stress factor)", "",
          "| notch | point-method part | size part, band k (e 0.290) | size part, series-pooled k | what the data need "
          "(held-out, diagnosis only) |", "|---|---|---|---|---|"]
    db, dp = (decomposition(g, notches, rows, uts, 1 / (5.1 * ks[s])) for s in ("band", "series_pooled"))
    for k in HELD:
        L.append(f"| {NAMES[k]} | {db[k]['tcd']:+.3f} | {db[k]['size']:+.3f} | {dp[k]['size']:+.3f} | {db[k]['data']:+.3f} |")
    L += ["", "## Every variant tried (all fitted on plain + sharp V only)", "",
          "| k source | L_FCI / L_CD | fitted values | sharp V (fitted): bias / rms | U r1 held out: bias / rms | "
          "U r3 held out: bias / rms | HELD OUT: rms, in band |", "|---|---|---|---|---|---|---|"]
    lab = {"assumed": f"L_FCI {a.L_fci:g} ASSUMED, L_CD 0.489 (no fit)", "lfci": "L_FCI fitted, L_CD 0.489",
           "lfci_lcd": "L_FCI and L_CD fitted"}
    res = {}
    for v in out:
        r = v["r"]
        h = stats(np.concatenate([r[k] for k in HELD]))
        res[(v["src"], v["mode"])] = v
        fv = f"L_FCI {v['L_FCI']:.3g} mm, L_CD {v['L_CD']:.3g} mm"
        best = " **(chosen)**" if (v["src"], v["mode"]) == BEST else ""
        L.append(f"| {KSRC[v['src']]}{best} | {lab[v['mode']]} | {fv} | " +
                 " | ".join(f"{stats(r[k])['bias']:+.3f} / {stats(r[k])['rms']:.3f}" for k in NAMES) +
                 f" | {h['rms']:.3f}, {100 * h['band']:.0f} % |")
    h = stats(np.concatenate([limit["r"][k] for k in HELD]))
    L.append(f"| limit e -> 0 (NOT the paper: no size term) | global scale fitted | scale {limit['scale']:.3f} | " +
             " | ".join(f"{stats(limit['r'][k])['bias']:+.3f} / {stats(limit['r'][k])['rms']:.3f}" for k in NAMES) +
             f" | {h['rms']:.3f}, {100 * h['band']:.0f} % |")
    L += ["", "Sensitivity of the no-fit variant (k from 2019 series at own p, R; L_CD 0.489) to the two ASSUMED inputs "
          "(held-out rms, in band; V = bias on the fitted sharp V):", "", "| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |",
          "|---|---|---|---|"]
    _, plain_, rows_, uts_, notches_, _, kmaps_ = setup()
    S90_ = {k: g[k]["S90"] for k in g}
    for lf in (0.1, 0.2, 0.4, 0.8):
        cells = []
        for lg in (20.0, 30.0, 50.0):
            r = {k: resid([x for x in rows_ if x[0] == k], notches_, S90_, kmaps_["series"], L_CD, 18 * lg * lf, uts_)
                 for k in NAMES}
            h = stats(np.concatenate([r[k] for k in HELD]))
            cells.append(f"{h['rms']:.3f}, {100 * h['band']:.0f} %; V {r['sharp'].mean():+.3f}")
        L.append(f"| {lf:g} | " + " | ".join(cells) + " |")
    L += ["", "References (fatigue.calib, same protocol): " + ", ".join(f"{k} {v:.3f}" for k, v in REF.items()) + "."]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "hsv_fix.md"), "w").write("\n".join(L) + "\n")
    json.dump([dict({k: v for k, v in x.items() if k != "r"}, r={n: q.tolist() for n, q in x["r"].items()})
               for x in out + [limit]], open(os.path.join(a.out, "hsv_fix.json"), "w"), indent=1)
    print("\n".join(L))
    figure(res, limit, ks, g, notches, rows, uts, os.path.join(a.out, "fig_hsv_fix"))


def figure(res, limit, ks, g, notches, rows, uts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.5), layout="constrained")
    # (a) decomposition
    a = ax[0, 0]
    x = np.arange(2)
    db = decomposition(g, notches, rows, uts, 1 / (5.1 * ks["band"]))
    ds = decomposition(g, notches, rows, uts, 1 / (5.1 * ks["series_pooled"]))
    a.bar(x - 0.27, [db[k]["tcd"] for k in HELD], 0.18, color="#1f77b4", label=r"point method $\sigma_1(L_{CD}/2)$")
    a.bar(x - 0.09, [db[k]["size"] for k in HELD], 0.18, color="#d62728", label="size term, band k (current)")
    a.bar(x + 0.09, [ds[k]["size"] for k in HELD], 0.18, color="#ff9896", label="size term, 2019 series k (pooled)")
    a.bar(x + 0.27, [db[k]["data"] for k in HELD], 0.18, color="0.3", label="data: held-out U vs fitted V (net shift)")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x, [f"{NAMES[k]} (held out)\nvs V r0.15 (fitted)" for k in HELD])
    a.set_ylabel(r"$\log_{10}$ of stress factor relative to V r0.15")
    a.set_title("(a) Contributions to the notch-to-notch factor", fontsize=10)
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3, axis="y")
    # (b) held-out rms vs size exponent
    a = ax[0, 1]
    es = np.linspace(0, 0.4, 81)
    a.plot(es, rms_vs_e(notches, g, rows, uts, es), "k-", lw=1.5, label="L_FCI fitted on V r0.15, e imposed")
    mk = {"band": "o", "fig1_mle": "s", "series_pooled": "D", "sharp_scatter": "^"}
    for n, v in ks.items():
        e = 1 / (5.1 * v)
        a.plot(e, rms_vs_e(notches, g, rows, uts, [e])[0], mk[n], ms=7, mfc="w", mec="k", label=f"k: {KSRC[n]}")
    for (lbl, v), ls in zip(REF.items(), (":", "--", "-.")):
        a.axhline(v, color="0.5", ls=ls, lw=1, label=f"{lbl} ({v:.3f})")
    a.set(xlabel=r"size exponent $e = 1/(k\,k_{SN})$ on $S_S L_{FCI} / (S_N L_{CD})$", ylabel="held-out rms log10 S (U r1 + U r3)")
    a.set_title("(b) Held-out error vs the size exponent (paper: e > 0)", fontsize=10)
    a.legend(fontsize=7, loc="upper left")
    a.grid(alpha=0.3)
    # (c), (d) S-N: current (band k) and the chosen honest version
    Ns = np.geomspace(10, 1e7, 200)
    sty = {"sharp": dict(marker="v", s=30, facecolor="none", edgecolors="#9467bd", linewidths=1.1, label="V r0.15: FITTED"),
           "intermediate": dict(marker="s", s=22, color="#1f77b4", label="U r1: HELD OUT"),
           "blunt": dict(marker="P", s=32, color="#d62728", label="U r3: HELD OUT")}
    for a, key, t in ((ax[1, 0], ("band", "lfci"), "(c) Current: k from band T 2.451, L_FCI fitted"),
                      (ax[1, 1], BEST, "(d) Chosen: k from 2019 series (own p, R), L_FCI fitted")):
        v = res[key]
        a.plot(Ns, P50(Ns), "k-", lw=1.5, label="plain P50 (Fig. 1)")
        for f_ in (math.sqrt(T_BAND), 1 / math.sqrt(T_BAND)):
            a.plot(Ns, P50(Ns) * f_, "k--", lw=0.9)
        for k in ("sharp", "intermediate", "blunt"):
            N = np.array([x[4] for x in rows if x[0] == k])
            a.scatter(N, P50(N) * 10 ** v["r"][k], zorder=3, **sty[k])
        h = stats(np.concatenate([v["r"][k] for k in HELD]))
        a.set_title(f"{t}\nL_FCI {v['L_FCI']:.3g} mm; held out rms {h['rms']:.3f}, {100 * h['band']:.0f} % in band",
                    fontsize=10)
        a.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.05, 3), xlabel="cycles to failure $N_f$",
              ylabel=r"$\sigma_{eff,max}/\sigma_{UTS}$")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
