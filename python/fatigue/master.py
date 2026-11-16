"""Master S-N collapse: which effective-stress parameter puts every PLA test, plain and notched, of both data sets on one
dimensionless S-N curve?

    python3 -m fatigue.master [--out results/fatigue]

Master variable (dimensionless, Walker mean-stress form):
    S = f_eff * sigma_a,net * (2 / (1 - R))^(1 - gamma) / sigma_UTS
f_eff = sigma_eff / sigma_net is the criterion's computed notch factor from the linear-elastic unit field (pure number):
    net        1                                          (no notch correction)
    kt         Kt                                         (elastic peak, sigma_max)
    hotspot    Kt / (1 + sqrt(l chi_0))                   (supported hot spot, FKM / Siebel-Stieler; length l)
    tcd_point  sigma_1(L/2)                               (TCD point method; length L, constant in N)
    tcd_line   mean sigma_1 over 2L                       (TCD line method)
    weibull    (J_m / V_ref)^(1/m)                        (weakest link; a plain bar of volume V gets (V/V_ref)^(1/m))
    sweibull   as weibull with sigma_1 / (1 + sqrt(l chi))(supported Weibull; m and l)
Data: Seifollahi & Kabir 2026, every specimen (R 0.05); Ezeh & Susmel 2019/2020, every plain and notched series as its
Basquin line sigma_A50 (N / 2e6)^(-1/k) sampled at 4 lives in 2e4..2e6, weight n/4. Print angles pooled.
Score: one master line log10 N = c + d log10 S fitted to everything; rms of the log10-life residual, per data set, and
the objective is the mean of the two data sets' mean-square residuals (each data set counts equally).
Ezeh-Susmel's plain gauge length is not given: 30 mm assumed (only the Weibull volume uses it).
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
    """Rows: (data set, group, notch or None, s_a net, R, N, weight, UTS, plain volume)."""
    rows, F = [], {}
    S = json.load(open(os.path.join(HERE, "data", "seifollahi_kabir.json")))
    ps = S["plain_specimen"]
    Vs = ps["W"] * ps["T"] * 50.0
    for k, v in S["specimens"].items():
        F[("S", k)] = field(Specimen.from_dict(v), h=1.5, log=lambda *_: None)
    for k, pts in S["raw_lives"].items():
        for s, *ns in pts:
            for n in ns:
                rows.append(("Seifollahi", k, None if k == "smooth" else ("S", k), s, 0.05, n, 1.0, S["UTS_MPa"], Vs))
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    pe = E["plain_specimen"]
    Ve = pe["W"] * pe["T"] * gauge_ezeh
    for k, v in E["specimens"].items():
        F[("E", k)] = field(Specimen.from_dict(v), h=1.5, log=lambda *_: None)
    uts = {q["p"]: q["UTS"] for q in E["plain"]}
    for q in E["plain"]:
        for N in N_EZ:
            rows.append(("Ezeh-Susmel", "plain", None, q["sigma_A50"] * (N / E["N_ref"]) ** (-1 / q["k"]), q["R"], N,
                         q["n"] / len(N_EZ), q["UTS"], Ve))
    for q in E["notched"]:
        for N in N_EZ:
            rows.append(("Ezeh-Susmel", q["notch"], ("E", q["notch"]), q["sigma_nA50"] * (N / E["N_ref"]) ** (-1 / q["k"]),
                         q["R"], N, q["n"] / len(N_EZ), uts[q["p"]], Ve))
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
    if name == "hotspot":
        return 1.0 / C.supported_kt(f, 1.0, th[0])
    lig = float(C.line_profile(f)[0][-1])
    if name == "tcd_point":
        return 1.0 / C.tcd_point(f, 1.0, min(th[0], 1.9 * lig))
    if name == "tcd_line":
        return 1.0 / C.tcd_line(f, 1.0, min(th[0], 0.95 * lig))
    raise KeyError(name)


SPEC = {   # name: (parameter names, start, log-transformed bounds)
    "net": ([], []), "kt": ([], []), "hotspot": (["l"], [1.0]), "tcd_point": (["L"], [1.0]),
    "tcd_line": (["L"], [1.0]), "weibull": (["m"], [10.0]), "sweibull": (["m", "l"], [20.0, 0.5])}



def score(name, th, gamma, rows, F, cache):
    S, N, w, ds, nt = [], [], [], [], []
    for d, g, key, s, R, n, wt, uts, Vp in rows:
        nt.append(key is not None)
        ck = (name, tuple(np.round(th, 9)), key, Vp)
        if ck not in cache:
            cache[ck] = factor(name, F[key] if key else None, th, Vp)
        S.append(cache[ck] * s * (2 / (1 - R)) ** (1 - gamma) / uts)
        N.append(n); w.append(wt); ds.append(d)
    x, y, w = np.log10(S), np.log10(N), np.array(w)
    W = np.sqrt(w)
    d_, c_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]) * W[:, None], y * W, rcond=None)[0]
    r = y - (c_ + d_ * x)
    ds = np.array(ds)
    per = {k: float(np.sqrt(np.average(r[ds == k] ** 2, weights=w[ds == k]))) for k in ("Seifollahi", "Ezeh-Susmel")}
    return float(np.mean([v ** 2 for v in per.values()])), per, (c_, d_), (x, y, w, ds, np.array(nt))


def fit(name, rows, F, gamma=None):
    """Minimise the objective over the criterion's parameters (log space) and gamma (if None: fitted, else fixed)."""
    cache = {}
    names, th0 = SPEC[name]
    free_g = gamma is None
    def unpack(z):
        th = [math.exp(v) for v in z[:len(th0)]]
        g = (1 / (1 + math.exp(-z[-1]))) if free_g else gamma          # gamma in (0, 1)
        return th, g
    z0 = [math.log(v) for v in th0] + ([0.0] if free_g else [])
    if not z0:
        return [], gamma, score(name, [], gamma, rows, F, cache)
    obj = lambda z: score(name, *unpack(z), rows, F, cache)[0]
    best = None
    for scale in (0.3, 1.0, 3.0):
        zs = [v + math.log(scale) for v in z0[:len(th0)]] + z0[len(th0):]
        o = minimize(obj, zs, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
        if best is None or o.fun < best.fun:
            best = o
    th, g = unpack(best.x)
    return th, g, score(name, th, g, rows, F, cache)


def plain_floor(rows, gamma=0.5):
    out = {}
    for ds in ("Seifollahi", "Ezeh-Susmel"):
        rr = [r for r in rows if r[0] == ds and r[2] is None]
        x = np.log10([s * (2 / (1 - R)) ** (1 - gamma) / u for _, _, _, s, R, _, _, u, _ in rr])
        y, w = np.log10([r[5] for r in rr]), np.array([r[6] for r in rr])
        A = np.column_stack([x, np.ones_like(x)]) * np.sqrt(w)[:, None]
        c = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)[0]
        out[ds] = float(np.sqrt(np.average((y - A @ c / np.sqrt(w)) ** 2, weights=w)))
    return out


def figure(res, path, show=(("net", "fitted"), ("kt", "fitted"), ("tcd_point", "fitted"), ("hotspot", "fitted"))):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(1, len(show), figsize=(4.1 * len(show), 4.7), sharey=True, layout="constrained")
    title = {"net": "net stress (no notch factor)", "kt": r"elastic peak $K_t\sigma_n$ ($\sigma_{max}$)",
             "tcd_point": "TCD point, constant L", "hotspot": r"supported hot spot $K_t/(1+\sqrt{l\chi_0})$"}
    for ax, key in zip(axs, show):
        th, g, per, (c, d), (x, y, w, ds, nt) = res[key]
        groups = [("Ezeh-Susmel", False, "0.6", "o", "Ezeh-Susmel plain (series lines)"),
                  ("Ezeh-Susmel", True, "#9467bd", "s", "Ezeh-Susmel notched (series lines)"),
                  ("Seifollahi", False, "k", "o", "Seifollahi smooth"),
                  ("Seifollahi", True, "#d62728", "^", "Seifollahi notched")]
        for dsn, isn, col, mk, lab in groups:
            k = (ds == dsn) & (nt == isn)
            if dsn == "Ezeh-Susmel":                                  # 4 samples per series: draw the series line
                xs, ys = x[k].reshape(-1, len(N_EZ)), y[k].reshape(-1, len(N_EZ))
                for i, (xx, yy) in enumerate(zip(xs, ys)):
                    ax.plot(10 ** yy, 10 ** xx, color=col, lw=1.0, alpha=0.8, label=lab if i == 0 else None)
            else:
                ax.scatter(10 ** y[k], 10 ** x[k], marker=mk, s=26, color=col, edgecolors="k", linewidths=0.3,
                           zorder=4, label=lab)
        ax.grid(alpha=0.3, which="both")
        Ns = np.geomspace(1e3, 3e6, 20)
        ax.plot(Ns, 10 ** ((np.log10(Ns) - c) / d), "k-", lw=1.4, label="master line")
        ax.set(xscale="log", yscale="log", xlabel="cycles to failure $N_f$",
               title=f"{title[key[0]]}\n$\\gamma$ {g:.2f}; rms Seif {per['Seifollahi']:.2f}, Ezeh {per['Ezeh-Susmel']:.2f}")
        ax.title.set_fontsize(9.5)
    from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
    axs[0].yaxis.set_major_locator(FixedLocator([0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]))
    axs[0].yaxis.set_major_formatter(ScalarFormatter()); axs[0].yaxis.set_minor_formatter(NullFormatter())
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=5, fontsize=9, frameon=False)
    axs[0].set_ylim(0.08, 3.0)
    axs[0].set_ylabel(r"$S = f_{eff}\,\sigma_{a,n}\,(2/(1-R))^{1-\gamma}/\sigma_{UTS}$")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    rows, F = load()
    lines = ["# Master S-N collapse of all PLA tests (Seifollahi & Kabir + Ezeh & Susmel), rms log10 life about one line",
             "", "| criterion | gamma | parameters | rms Seifollahi | rms Ezeh-Susmel | master slope 1/d |",
             "|---|---|---|---|---|---|"]
    res = {}
    for name in SPEC:
        for gam in (0.5, None):
            th, g, (J, per, (c, d), dat) = fit(name, rows, F, gamma=gam)
            res[(name, "fitted" if gam is None else "SWT")] = (th, g, per, (c, d), dat)
            par = ", ".join(f"{p} {v:.3g}" for p, v in zip(SPEC[name][0], th)) or "-"
            lines.append(f"| {name} | {g:.2f}{' (fitted)' if gam is None else ' (SWT)'} | {par} | {per['Seifollahi']:.3f} | "
                         f"{per['Ezeh-Susmel']:.3f} | {1 / d:.3f} |")
    lines += ["", "Floors (no notch involved): plain tests alone about their own master line, rms log10 life: "
              + ", ".join(f"{k} {v:.3f}" for k, v in plain_floor(rows).items())
              + ". Ezeh-Susmel's plain series alone scatter more than the whole collapse (print angle and R), so only "
              "Seifollahi discriminates between criteria. sweibull drives m to infinity: it becomes the hotspot."]
    figure(res, os.path.join(a.out, "fig_master_collapse"))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "master_collapse.md"), "w").write("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}": dict(th=v[0], gamma=v[1], rms=v[2], line=v[3]) for k, v in res.items()},
              open(os.path.join(a.out, "master_collapse.json"), "w"), indent=1)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
