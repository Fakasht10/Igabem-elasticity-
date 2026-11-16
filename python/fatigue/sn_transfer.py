"""Full S-N transfer: predict the life of every notched specimen from the smooth S-N curve and a cached field.

    python3 -m fatigue.sn_transfer [--data fatigue/data/seifollahi_kabir.json] [--gauge 50] [--out results/fatigue]

The smooth curve is the Basquin fit of the raw smooth lives, s_a = A N^B. A notched specimen at nominal net amplitude
s_n fails at the N where its effective stress equals the smooth curve:
    kt        Kt s_n = A N^B
    weibull   s_n (J_notch / V_plain)^(1/m) = A N^B            m given (fixed for every N)
    tcd       s_n sigma(L(N)/2) = A N^B (point) or the line mean over 2 L(N), with L(N) = A_L N^b_L (Susmel and Taylor's
              L-N law; here borrowed from another data set: blind)
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
    ap.add_argument("--m", type=float, nargs="*", default=[7.5], help="Weibull moduli taken from elsewhere (blind)")
    ap.add_argument("--m-fit", type=float, nargs="*", default=[12.0], help="Weibull moduli fitted on these notches")
    ap.add_argument("--L", type=float, nargs=2, default=[16.4, -0.242], metavar=("A_L", "b_L"),
                    help="TCD L(N) = A_L N^b_L mm (default Ezeh & Susmel 2020 PLA)")
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
    for m, tag in [(m, "blind") for m in a.m] + [(m, "fitted") for m in a.m_fit]:
        crit[f"Weibull m {m:g} ({tag})"] = lambda f, m=m: (lambda N, w=1.0 / C.weibull(f, 1.0, m, V): w)
    crit["TCD point, L(N) (blind)"] = lambda f: (lambda N: 1.0 / C.tcd_point(f, 1.0, LN(N)))
    crit["TCD line, L(N) (blind)"] = lambda f: (lambda N: 1.0 / C.tcd_line(f, 1.0, LN(N)))
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
    lines = [f"# S-N transfer, {d['id']}: smooth Basquin A {A:.3f}, B {B:.4f}; every notched specimen", "",
             "| criterion | n | mean log10(N_pred/N) | rms log10 | within x2 | within x3 |", "|---|---|---|---|---|---|"]
    sc = {}
    for c in crit:
        e = np.log10([r[c] / r["N"] for r in rows])
        sc[c] = dict(bias=float(e.mean()), rms=float(np.sqrt(np.mean(e ** 2))), x2=int((abs(e) < math.log10(2)).sum()),
                     x3=int((abs(e) < math.log10(3)).sum()), n=len(e))
        lines.append(f"| {c} | {len(e)} | {e.mean():+.2f} | {sc[c]['rms']:.2f} | {sc[c]['x2']}/{len(e)} | "
                     f"{sc[c]['x3']}/{len(e)} |")
    lines += ["", "| notch | s_a (MPa) | N test | " + " | ".join(crit) + " |", "|---" * (3 + len(crit)) + "|"]
    for r in rows:
        lines.append(f"| {r['notch']} | {r['s_a']:.2f} | {r['N']:,.0f} | " + " | ".join(f"{r[c]:,.0f}" for c in crit) + " |")
    os.makedirs(a.out, exist_ok=True)
    tag = f"sn_transfer_{d['id']}_h{a.h:g}"
    open(os.path.join(a.out, tag + ".md"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(A=A, B=B, scores=sc, rows=rows), open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    figure(d, raw, A, B, F, crit, os.path.join(a.out, "fig_" + tag))
    print("\n".join(lines))


def figure(d, raw, A, B, F, crit, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
    notches = [k for k in raw if k != "smooth"]
    fig, axs = plt.subplots(1, len(notches), figsize=(5.2 * len(notches), 4.0), sharey=True)
    Ns = np.geomspace(2e3, 3e5, 60)
    sty = {"Kt (blind)": ("0.5", ":"), "TCD point, L(N) (blind)": ("#d62728", "-"), "TCD line, L(N) (blind)": ("#ff7f0e", "--")}
    wc = iter(["#1f77b4", "#17becf", "#9467bd", "#2ca02c"])
    sty.update({c: (next(wc), "-.") for c in crit if c not in sty})
    for ax, k in zip(np.atleast_1d(axs), notches):
        S = [s for s, *_ in raw["smooth"]]
        ax.scatter([n for s, *ns in raw["smooth"] for n in ns], [s for s, *ns in raw["smooth"] for _ in ns], c="k",
                   s=18, label="smooth (test)")
        ax.plot(Ns, A * Ns ** B, "k-", lw=0.9, label="smooth fit")
        ax.scatter([n for s, *ns in raw[k] for n in ns], [s for s, *ns in raw[k] for _ in ns], marker="D", s=22,
                   facecolor="white", edgecolor="k", lw=1.1, zorder=4, label=f"{k} (test)")
        for c, mk in crit.items():
            eff = mk(F[k])
            col, ls = sty[c]
            ax.plot(Ns, [A * N ** B / eff(N) for N in Ns], color=col, ls=ls, lw=1.2, label=c)
        ax.set(xscale="log", xlabel="cycles to failure $N_f$", title=f"{k}: Kt {float(F[k]['Kt_mean']):.2f}", ylim=(4, 20))
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axs)[0].set_ylabel(r"net stress amplitude $\sigma_a$ (MPa)")
    np.atleast_1d(axs)[-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
