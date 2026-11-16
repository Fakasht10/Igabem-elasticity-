"""Full S-N transfer: predict the life of every notched specimen from the smooth S-N curve and a cached field.

    python3 -m fatigue.sn_transfer [--data fatigue/data/seifollahi_kabir.json] [--gauge 50] [--out results/fatigue]

The smooth curve is the Basquin fit of the raw smooth lives, s_a = A N^B. A notched specimen at nominal net amplitude
s_n fails at the N where its effective stress equals the smooth curve:
    kt        Kt s_n = A N^B
    weibull   s_n (J_notch / V_plain)^(1/m) = A N^B            m given (fixed for every N)
    tcd       s_n sigma(L(N)/2) = A N^B (point) or the line mean over 2 L(N), with L(N) = A_L N^b_L (Susmel and Taylor's
              L-N law; here fitted on another data set, Ezeh & Susmel 2020 PLA: a cross-dataset transfer)
Scores are on log10 life: mean bias, rms, and the share within a factor 2 and 3. Every specimen is one point.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import brentq

from . import criteria as C
from .fields import field
from .specimens import Specimen
from .transfer import HERE, OUT


def basquin(pts):
    S = np.array([s for s, *ns in pts for _ in ns]); N = np.array([n for s, *ns in pts for n in ns], float)
    B, lnA = np.polyfit(np.log(N), np.log(S), 1)
    return math.exp(lnA), B


def life(eff_of_N, s_n, A, B):
    """N with s_n eff(N) = A N^B (eff may depend on N through L(N)); log-space root."""
    g = lambda lg: math.log(s_n * eff_of_N(10 ** lg)) - math.log(A) - B * lg * math.log(10)
    return 10 ** brentq(g, 0.0, 8.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "seifollahi_kabir.json"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--gauge", type=float, default=50.0)
    ap.add_argument("--m", type=float, nargs="*", default=[7.5], help="Weibull moduli fitted on another data set (Ezeh-Susmel PLA)")
    ap.add_argument("--m-fit", type=float, nargs="*", default=[12.0], help="Weibull moduli fitted on these notches")
    ap.add_argument("--L", type=float, nargs=2, default=[16.4, -0.242], metavar=("A_L", "b_L"),
                    help="TCD L(N) = A_L N^b_L mm fitted on another data set (default Ezeh & Susmel 2020 PLA)")
    ap.add_argument("--support", type=float, nargs=2, default=[0.713, 119.0], metavar=("l_mm", "m"),
                    help="gradient-supported Weibull (Maede et al.: n = 1 + sqrt(l chi)); default l fitted on Ezeh-Susmel "
                         "(fatigue.support_fit) at m 119, which is THIS data set's own smooth scatter (T 1.028): one "
                         "number taken from the target. With Ezeh-Susmel's own m 7.5 the fitted l is ~0 and it fails.")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    d = json.load(open(a.data))
    raw, ps = d["raw_lives"], d["plain_specimen"]
    V = ps["W"] * ps["T"] * a.gauge
    A, B = basquin(raw["smooth"])
    F = {k: field(Specimen.from_dict(v), h=a.h, log=lambda *_: None) for k, v in d["specimens"].items()}
    lig = min(Specimen.from_dict(v).ligament for v in d["specimens"].values())
    LN = lambda N: min(a.L[0] * N ** a.L[1], 0.95 * lig)          # held inside the ligament at very short lives
    crit = {"Kt (blind)": lambda f: (lambda N: float(f["Kt_mean"]))}
    for m, tag in [(m, "fitted on Ezeh-Susmel") for m in a.m] + [(m, "fitted here") for m in a.m_fit]:
        crit[f"Weibull m {m:g} ({tag})"] = lambda f, m=m: (lambda N, w=1.0 / C.weibull(f, 1.0, m, V): w)
    l_s, m_s = a.support
    crit[SUP] = (
        lambda f: (lambda N, w=1.0 / C.weibull_support(f, 1.0, m_s, V, l_s): w))
    crit["TCD point, L(N) fitted on Ezeh-Susmel"] = lambda f: (lambda N: 1.0 / C.tcd_point(f, 1.0, LN(N)))
    crit["TCD line, L(N) fitted on Ezeh-Susmel"] = lambda f: (lambda N: 1.0 / C.tcd_line(f, 1.0, LN(N)))
    # in-sample reference: the measured fatigue notch factor of each notch at N_ref (plain / notched strength), constant in
    # N. It uses the notched data themselves, so it is the ceiling a constant-factor criterion can reach, not a prediction.
    Kf = {r["notch"]: d["plain"][0]["sigma_A50"] / r["sigma_nA50"] for r in d["notched"]}
    crit[KF] = lambda f: (lambda N, w=Kf[f["_notch"]]: w)
    for k in F:
        F[k]["_notch"] = k
    rows = []
    for k, pts in raw.items():
        if k == "smooth":
            continue
        for s, *ns in pts:
            for n in ns:
                r = dict(notch=k, s_a=s, N=n)
                for c, mk in crit.items():
                    r[c] = life(mk(F[k]), s, A, B)
                rows.append(r)
    # fitted on smooth + one notch (every specimen of it), predicted on the other notch: L(N) = A_L N^b_L per TCD
    # method (Susmel & Taylor's two-curve calibration) and the Weibull m
    from scipy.optimize import minimize, minimize_scalar
    notches = [k for k in raw if k != "smooth"]
    spec = lambda k: [(s, n) for s, *ns in raw[k] for n in ns]
    tcd = {"point": C.tcd_point, "line": C.tcd_line}
    cross = {}
    for cal in notches:
        for meth, fn in tcd.items():
            def err(th, k=cal, fn=fn):
                LNc = lambda N: min(math.exp(th[0]) * N ** th[1], 0.95 * lig)
                return float(np.mean([math.log10(life(lambda N: 1.0 / fn(F[k], 1.0, LNc(N)), s, A, B) / n) ** 2
                                      for s, n in spec(k)]))
            o = min((minimize(err, [math.log(L0), b0], method="Nelder-Mead") for L0 in (0.5, 2.0, 8.0)
                     for b0 in (-0.3, 0.0)), key=lambda o: o.fun)
            cross[(cal, f"TCD {meth}, L(N) fitted on {cal}")] = (
                lambda f, th=o.x, fn=fn: (lambda N: 1.0 / fn(f, 1.0, min(math.exp(th[0]) * N ** th[1], 0.95 * lig))),
                f"A_L {math.exp(o.x[0]):.3g} mm, b_L {o.x[1]:.3f}")
        def errm(lm, k=cal):
            return float(np.mean([math.log10(life(lambda N, w=1.0 / C.weibull(F[k], 1.0, math.exp(lm), V): w, s, A, B)
                                              / n) ** 2 for s, n in spec(k)]))
        mfit = math.exp(minimize_scalar(errm, bounds=(math.log(2), math.log(200)), method="bounded").x)
        cross[(cal, f"Weibull m fitted on {cal}")] = (
            lambda f, m=mfit: (lambda N, w=1.0 / C.weibull(f, 1.0, m, V): w), f"m {mfit:.2f}")
    xlines = ["", "## Fitted on smooth + one notch, every specimen of both; scored on that notch (fit) and the other (predicted)",
              "", "| criterion | parameters | fitted notch: bias / rms log10, within x2 | predicted notch: bias / rms log10, within x2 |",
              "|---|---|---|---|"]
    for (cal, c), (mk, par) in cross.items():
        cells = []
        for k in (cal, *[n for n in notches if n != cal]):
            e = np.log10([life(mk(F[k]), s, A, B) / n for s, n in spec(k)])
            cells.append(f"{k}: {e.mean():+.2f} / {np.sqrt(np.mean(e ** 2)):.2f}, "
                         f"{int((abs(e) < math.log10(2)).sum())}/{len(e)}")
        xlines.append(f"| {c} | {par} | {cells[0]} | {cells[1]} |")
    lines = [f"# S-N transfer, {d['id']}: smooth Basquin A {A:.3f}, B {B:.4f}; every notched specimen", "",
             "| criterion | n | mean log10(N_pred/N) | rms log10 | within x2 | within x3 |", "|---|---|---|---|---|---|"]
    sc = {}
    order = [c for c in crit if not insample(c)] + [c for c in crit if insample(c)]
    for i, c in enumerate(order):
        if i and insample(c) and not insample(order[i - 1]):
            lines.append("| *in-sample references (fitted on these notches, not predictions)* | | | | | |")
        e = np.log10([r[c] / r["N"] for r in rows])
        sc[c] = dict(bias=float(e.mean()), rms=float(np.sqrt(np.mean(e ** 2))), x2=int((abs(e) < math.log10(2)).sum()),
                     x3=int((abs(e) < math.log10(3)).sum()), n=len(e))
        lines.append(f"| {c} | {len(e)} | {e.mean():+.2f} | {sc[c]['rms']:.2f} | {sc[c]['x2']}/{len(e)} | "
                     f"{sc[c]['x3']}/{len(e)} |")
    lines += ["", "| notch | s_a (MPa) | N test | " + " | ".join(crit) + " |", "|---" * (3 + len(crit)) + "|"]
    for r in rows:
        lines.append(f"| {r['notch']} | {r['s_a']:.2f} | {r['N']:,.0f} | " + " | ".join(f"{r[c]:,.0f}" for c in crit) + " |")
    lines += xlines
    os.makedirs(a.out, exist_ok=True)
    tag = f"sn_transfer_{d['id']}_h{a.h:g}"
    open(os.path.join(a.out, tag + ".md"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(A=A, B=B, scores=sc, rows=rows), open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    figure(d, raw, F, crit, sc, os.path.join(a.out, "fig_" + tag))
    figure_overview(d, raw, F, crit, rows, sc, os.path.join(a.out, "fig_" + tag + "_overview"))
    print("\n".join(lines))


SUP = "Weibull + support: l 0.713 fitted on Ezeh-Susmel, m 119 from this smooth scatter"
KF = "measured Kf per notch (in-sample)"


def insample(c):
    return "fitted here" in c or "in-sample" in c


def paper_fits(d):
    """Basquin fits sigma_a = A N^B of each group's tests, as stored in the data file: a refit of Table 6 that equals the
    paper's Table 7 constants to 0.04 % (68.159 / 57.147 / 48.436 printed)."""
    out = {"smooth": tuple(d["plain"][0]["basquin"])}
    out.update({r["notch"]: tuple(r["basquin"]) for r in d["notched"]})
    return out


def passing(sc):
    """Transferred criteria worth drawing: not Kt, not in-sample, and at least half of the specimens within a factor 2
    (failed and in-sample ones stay in the table, not on the plot)."""
    return [c for c, v in sc.items() if not c.startswith("Kt") and not insample(c) and v["x2"] >= v["n"] / 2]


STY = {"Weibull + support": ("--", "#2ca02c"), "TCD line": (":", "#ff7f0e"), "TCD point": ("-.", "#9467bd"),
       "Weibull m": ((0, (5, 1, 1, 1, 1, 1)), "#17becf")}


def sty(c):
    return next(v for k, v in STY.items() if c.startswith(k))


def short(c):
    return {SUP: "supported Weibull (l fitted on Ezeh-Susmel, m 119 from this smooth scatter)"}.get(c, c)


def tiny(c):
    """Legend name; provenance goes in the caption (README)."""
    return ("supported Weibull" if c == SUP else c.split(",")[0]) + " (predicted)"


def figure_overview(d, raw, F, crit, rows, sc, path):
    """(a) the three data sets with the paper's own fits (solid) and the transferred predictions (broken);
    (b) predicted vs measured life of every notched specimen for every criterion that did not fail outright."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    col = {"smooth": "k", "edge": "#d62728", "hole": "#1f77b4"}
    mk = {"smooth": "o", "edge": "^", "hole": "D"}
    name = {"smooth": "smooth", "edge": "edge notch R2, Kt %.2f", "hole": "central hole R2, Kt %.2f"}
    PF = paper_fits(d)
    A, B = PF["smooth"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 6.6), gridspec_kw={"width_ratios": [1.25, 1]})
    for k, pts in raw.items():
        lab = name[k] % float(F[k]["Kt_mean"]) if k != "smooth" else name[k]
        n = [n for s, *ns in pts for n in ns]
        Ns = np.geomspace(min(n) / 1.5, max(n) * 1.5, 60)
        a.scatter(n, [s for s, *ns in pts for _ in ns], marker=mk[k], s=36, color=col[k], zorder=4,
                  label=f"{lab}")
        Ak, Bk = PF[k]
        a.plot(Ns, Ak * Ns ** Bk, color=col[k], lw=1.8)
    show = passing(sc)
    Ns = np.geomspace(3e3, 2e5, 80)
    for k in raw:
        if k == "smooth":
            continue
        for c in show:
            eff = crit[c](F[k])
            a.plot(Ns, [A * N ** B / eff(N) for N in Ns], color=col[k], ls=sty(c)[0], lw=1.3)
    a.set(xscale="log", xlim=(3e3, 2e5), ylim=(8, 17), xlabel="cycles to failure $N_f$",
          ylabel=r"net stress amplitude $\sigma_a$ (MPa)", title="(a) Seifollahi & Kabir 2026, YZ PLA, R = 0.05")
    a.grid(alpha=0.3, which="both")
    from matplotlib.lines import Line2D
    h, lb = a.get_legend_handles_labels()
    h += [Line2D([], [], color="k", lw=1.8)] + [Line2D([], [], color="k", ls=sty(c)[0], lw=1.3) for c in show]
    lb += ["Basquin fit of the tests (paper Table 7 to 0.04 %)"] + [tiny(c) for c in show]
    a.legend(h, lb, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    lo, hi = 3e3, 2e5
    x = np.geomspace(lo, hi, 2)
    b.fill_between(x, x / 2, x * 2, color="0.9", lw=0, label="factor 2")
    b.plot(x, x, "k-", lw=0.8)
    for c in passing(sc):
        for k in ("edge", "hole"):
            rr = [r for r in rows if r["notch"] == k]
            b.scatter([r["N"] for r in rr], [r[c] for r in rr], marker=mk[k], s=28, color=sty(c)[1],
                      edgecolors="k", linewidths=0.3, zorder=3,
                      label=f"{tiny(c)}: {sc[c]['x2']}/{sc[c]['n']} within x2, rms {sc[c]['rms']:.2f}"
                      if k == "edge" else None)
    b.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel="measured life",
          ylabel="predicted life", title="(b) every notched specimen (\u25b2 edge, \u25c6 hole)")
    b.grid(alpha=0.3, which="both")
    b.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def figure(d, raw, F, crit, sc, path):
    """One panel per notch: tests, the paper's own fits (solid), every criterion that did not fail outright (broken)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
    PF = paper_fits(d)
    A, B = PF["smooth"]
    notches = [k for k in raw if k != "smooth"]
    fig, axs = plt.subplots(1, len(notches), figsize=(5.6 * len(notches), 6.4), sharey=True)
    Ns = np.geomspace(3e3, 2e5, 60)
    for ax, k in zip(np.atleast_1d(axs), notches):
        for g, m, fc in (("smooth", "o", "k"), (k, {"edge": "^", "hole": "D"}[k], "white")):
            ax.scatter([n for s, *ns in raw[g] for n in ns], [s for s, *ns in raw[g] for _ in ns], marker=m, s=24,
                       facecolor=fc, edgecolor="k", lw=1.0, zorder=4, label=f"{g}: test")
            Ag, Bg = PF[g]
            ax.plot(Ns, Ag * Ns ** Bg, "k-", lw=1.4 if g == k else 0.8,
                    label=f"{g}: fit of the tests (paper), {Ag:.2f} N^{Bg:.3f}")
        for c in passing(sc):
            eff = crit[c](F[k])
            ls, col = sty(c)
            ax.plot(Ns, [A * N ** B / eff(N) for N in Ns], color=col, ls=ls, lw=1.3, label=tiny(c))
        ax.set(xscale="log", xlabel="cycles to failure $N_f$", title=f"{k}: Kt {float(F[k]['Kt_mean']):.2f}",
               xlim=(3e3, 2e5), ylim=(8, 17))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    np.atleast_1d(axs)[0].set_ylabel(r"net stress amplitude $\sigma_a$ (MPa)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
