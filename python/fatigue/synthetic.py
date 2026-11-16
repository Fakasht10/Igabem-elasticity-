"""Statistically equivalent SYNTHETIC specimens from Ezeh & Susmel's summary tables, checked against the real specimens,
and placed on their unifying design curve with each effective-stress criterion.

    python3 -m fatigue.synthetic [--reps 400] [--out results/fatigue]

Generator, per series (n, k, sigma_A50 at 2e6, T_sigma = sigma_10%/sigma_90%, from the paper's table):
    stress levels   n amplitudes, log-spaced between the series' median-curve stresses at N_lo and N_hi;
    specimen        strength multiplier X (median 1, P10/P90 ratio = T_sigma), life N = 2e6 (s / (X sigma_A50))^-k;
                    N > 2e6 is a run-out (censored, left out of the score like the real run-outs).
    X distribution  log-normal (Susmel's own assumption: sd log10 X = log10 T / 2.563), Weibull (shape set by T),
                    or normal in stress (same P10/P90 ratio).
Inputs and their source:
    notched 18 series: 2020 Table 1; [N_lo, N_hi] = life range of that series' failures in Figs 5-7 (extracted).
    plain 12 series:   2019 Table 1 (via data/ezeh_susmel.json); [N_lo, N_hi] = 5-95 % life range of the Fig. 1 failures
                       at the same R (Fig. 1 has no per-series plain points; it pools Refs 7, 8, 10).
Check: the same criteria applied to the REAL specimens (fatigue.band) and to each synthetic replicate; the synthetic
in-band share must bracket the real one. Band and design curve as printed in 2020 Fig. 1.
"""
import argparse
import json
import math
import os

import numpy as np

from .band import P50, T_BAND, feff, load
from .transfer import HERE, OUT

Z90 = 1.2815516
DESIGN = lambda N: np.minimum(0.1 * (np.asarray(N, float) / 2e6) ** (-1 / 5.5), 1.0)   # P_S > 90 %, printed
CRIT = [("net", None, "net (no notch factor)"), ("kt", None, "Kt x net"),
        ("tcd_point", None, "TCD point, authors' L(N)"), ("tcd_line", None, "TCD line, authors' L(N)"),
        ("tcd_line", 2.13, "TCD line, L 2.13 fitted on real")]


def multiplier(dist, T, z):
    """Strength multiplier with median 1 and X(P10)/X(P90) = T, from standard-normal z (so all three share the draw)."""
    if dist == "lognormal":
        return 10 ** (z * math.log10(T) / (2 * Z90))
    if dist == "normal":
        c = (T - 1) / (Z90 * (T + 1))
        return np.maximum(1 + c * z, 0.05)
    if dist == "weibull":
        beta = math.log(math.log(10) / math.log(10 / 9)) / math.log(T)
        u = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))         # same quantile as z
        return (-np.log1p(-u) / math.log(2)) ** (1 / beta)
    raise KeyError(dist)


def series_tables():
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    sp = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_specimens.json")))
    plain_pts = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_fig1_plain.json")))
    uts = {q["p"]: q["UTS"] for q in E["plain"]}
    notched = []
    for r in E["notched"]:
        real = [s for q in sp["series"] if (q["notch"], q["p"], q["R"]) == (r["notch"], r["p"], r["R"])
                for s in q["specimens"] if not s["runout"]]
        N = [s["N"] for s in real]
        notched.append(dict(r, s50=r["sigma_nA50"], N_lo=min(N), N_hi=max(N)))
    plain = []
    for r in E["plain"]:
        N = np.array([q["N"] for q in plain_pts if q["R"] == r["R"] and q["N"] < 1.8e6])
        plain.append(dict(r, s50=r["sigma_A50"], N_lo=float(np.percentile(N, 5)), N_hi=float(np.percentile(N, 95))))
    return notched, plain, uts, plain_pts


def generate(series, dist, rng):
    """One synthetic replicate: list of (series index, s_a, N, runout)."""
    out = []
    for i, r in enumerate(series):
        s_hi = r["s50"] * (r["N_lo"] / 2e6) ** (-1 / r["k"])
        s_lo = r["s50"] * (r["N_hi"] / 2e6) ** (-1 / r["k"])
        s = np.geomspace(s_lo, s_hi, r["n"])
        X = multiplier(dist, r["T"], rng.standard_normal(r["n"]))
        N = 2e6 * (s / (X * r["s50"])) ** (-r["k"])
        out += [(i, float(a), float(min(n, 2e6)), bool(n > 2e6)) for a, n in zip(s, N)]
    return out


class Factors:
    """f_eff(N) per notch and criterion, tabulated once on a log-N grid (L(N) makes TCD depend on life)."""
    def __init__(self, F):
        self.g = np.linspace(0, 7, 57)
        self.t = {(n, c, p): np.array([feff(c, f, 10 ** x, p) for x in self.g]) for n, f in F.items() for c, p, _ in CRIT}

    def __call__(self, notch, crit, par, N):
        return np.interp(np.log10(N), self.g, self.t[(notch, crit, par)])


def resid(rows, series, uts, fac, crit, par, plain=False):
    """log10(S / P50) of the failures; S = f_eff sigma_a 2/(1-R) / UTS(p)."""
    r = []
    for i, s, N, ro in rows:
        if ro:
            continue
        q = series[i]
        f = 1.0 if plain else fac(q["notch"], crit, par, N)
        r.append(math.log10(f * s * 2 / (1 - q["R"]) / uts[q["p"]] / P50(N)))
    return np.array(r)


def share(r):
    return float(np.mean(abs(r) < math.log10(T_BAND) / 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    notched, plain, uts, plain_pts = series_tables()
    E, _, F, spec, _, _, _ = load()
    fac = Factors(F)
    key = {(r["notch"], r["p"], r["R"]): i for i, r in enumerate(notched)}
    real = [(key[(n, p, R)], s, N, ro) for n, p, R, s, N, ro in spec]
    real_r = {lab: resid(real, notched, uts, fac, c, p) for c, p, lab in CRIT}
    pr = np.array([math.log10(q["smax_uts"] / P50(q["N"])) for q in plain_pts if q["N"] < 1.8e6])
    dists = ("lognormal", "weibull", "normal")
    syn, example = {}, {}
    for d in dists:
        rng = np.random.default_rng(a.seed)
        acc = {lab: [] for *_, lab in CRIT}
        acc["plain"] = []
        for k in range(a.reps):
            rows = generate(notched, d, rng)
            for c, p, lab in CRIT:
                r = resid(rows, notched, uts, fac, c, p)
                acc[lab].append((share(r), r.mean(), r.std()))
            prow = generate(plain, d, rng)
            r = resid(prow, plain, uts, fac, None, None, plain=True)
            acc["plain"].append((share(r), r.mean(), r.std()))
            if k == 0:
                example[d] = (rows, prow)
        syn[d] = {lab: np.array(v) for lab, v in acc.items()}
    q = lambda v: f"{100 * np.median(v):.0f} % [{100 * np.percentile(v, 5):.0f}-{100 * np.percentile(v, 95):.0f}]"
    L = ["# Synthetic (statistically equivalent) Ezeh-Susmel specimens vs the real ones, on the unifying band", "",
         f"{a.reps} synthetic replicates per distribution, each the same size as the paper's series (Table 1 n). "
         "Cells: share of failures inside the printed P10-P90 band, median [5-95 % over replicates]; real = the "
         "vector-extracted specimens.", "",
         "| set / criterion | REAL specimens | synthetic log-normal | synthetic Weibull | synthetic normal |",
         "|---|---|---|---|---|",
         f"| plain (synthetic from 2019 Table 1; real = Fig. 1 points, 3 printers) | {100 * share(pr):.0f} % | " +
         " | ".join(q(syn[d]["plain"][:, 0]) for d in dists) + " |"]
    for *_, lab in CRIT:
        L.append(f"| notched, {lab} | {100 * share(real_r[lab]):.0f} % | " +
                 " | ".join(q(syn[d][lab][:, 0]) for d in dists) + " |")
    L += ["", "Mean / sd of log10(S/P50), real vs synthetic log-normal median (does the synthetic set reproduce the real "
          "cloud, not just the share?):", "",
          "| set / criterion | real mean / sd | synthetic mean / sd |", "|---|---|---|",
          f"| plain | {pr.mean():+.3f} / {pr.std():.3f} | {np.median(syn['lognormal']['plain'][:, 1]):+.3f} / "
          f"{np.median(syn['lognormal']['plain'][:, 2]):.3f} |"]
    for *_, lab in CRIT:
        v = syn["lognormal"][lab]
        L.append(f"| {lab} | {real_r[lab].mean():+.3f} / {real_r[lab].std():.3f} | {np.median(v[:, 1]):+.3f} / "
                 f"{np.median(v[:, 2]):.3f} |")
    # design curve (P_S > 90 %): share of real notched failures at or above it
    for c, p, lab in CRIT[:1] + CRIT[3:4]:
        S = np.array([fac(n, c, p, N) * s * 2 / (1 - R) / uts[pp] for n, pp, R, s, N, ro in spec if not ro])
        Nf = np.array([N for *_, N, ro in spec if not ro])
        L.append(f"\nReal notched failures on the safe side of the printed design curve (0.1 UTS, k 5.5) with {lab}: "
                 f"{100 * np.mean(S >= DESIGN(Nf)):.0f} %")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "synthetic_band.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(real, example["lognormal"], notched, plain, uts, fac, plain_pts, syn, real_r, pr,
           os.path.join(a.out, "fig_synthetic_band"))


def figure(real, ex, notched, plain, uts, fac, plain_pts, syn, real_r, pr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.8), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    Ns = np.geomspace(10, 1e7, 200)
    col = {"blunt": "0.55", "intermediate": "#1f77b4", "sharp": "#9467bd"}
    mk = {"blunt": "P", "intermediate": "s", "sharp": "v"}

    def cloud(ax, rows, prow, title, filled):
        ax.plot(Ns, P50(Ns), "k-", lw=1.6, label="band P50 (paper)")
        ax.plot(Ns, P50(Ns) * math.sqrt(T_BAND), "k--", lw=1, label="band P10 / P90 (paper)")
        ax.plot(Ns, P50(Ns) / math.sqrt(T_BAND), "k--", lw=1)
        ax.plot(Ns, DESIGN(Ns), color="#d62728", lw=1.2, label="design curve 0.1 UTS, k 5.5 (paper)")
        if prow is None:
            ax.scatter([q["N"] for q in plain_pts], [q["smax_uts"] for q in plain_pts], s=12, facecolor="none",
                       edgecolors="0.6", lw=0.6, label="plain: Fig. 1 points (extracted)")
        else:
            ax.scatter([n for i, s, n, ro in prow], [s * 2 / (1 - plain[i]["R"]) / uts[plain[i]["p"]] for i, s, n, ro in prow],
                       s=12, marker="x", color="0.6", lw=0.7, label="plain: SYNTHETIC (2019 Table 1)")
        for nt in ("blunt", "intermediate", "sharp"):
            sel = [(s * 2 / (1 - notched[i]["R"]) / uts[notched[i]["p"]], n) for i, s, n, ro in rows if notched[i]["notch"] == nt]
            ax.scatter([n for _, n in sel], [y for y, _ in sel], marker=mk[nt], s=20, color=col[nt] if filled else "none",
                       edgecolors=col[nt], lw=0.9, label=f"{nt} notched, net: {'SPECIMEN' if filled else 'SYNTHETIC'}")
        ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.03, 3), xlabel="cycles to failure $N_f$",
               ylabel=r"$\sigma_{max}/\sigma_{UTS}$ (net, no notch factor)", title=title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="lower left")
    cloud(axs[0], real, None, "(a) REAL: specimens extracted from Figs 1, 5-7", True)
    cloud(axs[1], ex[0], ex[1], "(b) SYNTHETIC: one log-normal replicate\nfrom Table 1 (n, k, $\\sigma_{A50}$, $T_\\sigma$)", False)
    ax = axs[2]
    labs = ["plain"] + [lab for *_, lab in CRIT]
    realv = [share(pr)] + [share(real_r[l]) for l in labs[1:]]
    y = np.arange(len(labs))
    for j, (d, c) in enumerate((("lognormal", "#1f77b4"), ("weibull", "#2ca02c"), ("normal", "#ff7f0e"))):
        v = [syn[d][l][:, 0] for l in labs]
        med = [np.median(x) for x in v]
        lo = [np.median(x) - np.percentile(x, 5) for x in v]
        hi = [np.percentile(x, 95) - np.median(x) for x in v]
        ax.errorbar(100 * np.array(med), y + 0.18 * (j - 1), xerr=100 * np.array([lo, hi]), fmt="o", color=c, ms=4,
                    capsize=2, label=f"SYNTHETIC {d}: median, 5-95 %")
    ax.scatter(100 * np.array(realv), y, marker="*", s=110, color="k", zorder=5, label="REAL specimens")
    ax.set_yticks(y, ["plain (real = Fig. 1, 3 printers)"] + labs[1:], fontsize=8.5)
    ax.invert_yaxis()
    ax.set(xlabel="failures inside the printed P10-P90 band (%)", xlim=(-2, 102),
           title="(c) share in band: real vs synthetic, three scatter models")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.12, 0.62))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
