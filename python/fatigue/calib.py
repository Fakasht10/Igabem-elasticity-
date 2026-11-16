"""One calibration protocol for every notch criterion, the one Ezeh & Susmel 2020 (Sec. 6) used for their TCD:
fit on the plain curve + the sharp V-notch specimens, predict the two U notches (held out).

    python3 -m fatigue.calib [--out results/fatigue]

Plain curve: the 2020 Fig. 1 unifying P50, sigma_max/UTS = 0.16 (N/2e6)^(-1/5.1), as the authors used it.
Notched specimen (vector-extracted, 2020 Figs 5-7): S = f_eff sigma_a,net 2/(1-R) / UTS(p).
Fit: least squares of log10(S / P50(N)) over the 47 failed sharp-V specimens, each criterion's free parameters only.
Held out: the 102 failed U r1 and U r3 specimens. Score: bias / rms of the same residual, share inside the P10-P90 band.
Criteria and free parameters:
    net                none (no notch factor)                       kt        none (elastic peak)
    tcd_ln_paper       none here: the authors' published L_M(N) = 16.4 N^-0.242 (fitted by them on the same data)
    tcd_ln             A, B of L(N) = A N^B, refit here on our IgBEM field
    tcd_const          constant L
    weibull            m (plain gauge volume 6 x 3 x 30 mm, gauge length ASSUMED)
    hotspot            l of Kt / (1 + sqrt(l chi0)), capped at f >= 1
    mhsv               L_FCI of the modified HSV (Zhang et al. 2023); L_CD 0.49 mm, life Weibull shape from T 2.451
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from . import criteria as C
from .band import P50, T_BAND
from .hsv import L_CD, W21, load as hsv_load
from .transfer import HERE, OUT

V_PLAIN = 6.0 * 3.0 * 30.0
NAMES = {"blunt": "U r3", "intermediate": "U r1", "sharp": "V r0.15"}
CRIT = {"net": ([], []), "kt": ([], []), "tcd_ln_paper": ([], []), "tcd_ln": (["A", "B"], [16.4, -0.242]),
        "tcd_const": (["L"], [0.5]), "tcd_line": (["L"], [0.5]),
        "tcd_line_ln": (["A", "B"], [16.4, -0.242]), "sup_tcd": (["L", "l"], [0.5, 1.0]), "sup_tcd_line": (["L", "l"], [0.5, 1.0]),
        "weibull": (["m"], [10.0]), "hotspot": (["l"], [1.0]), "mhsv": (["L_FCI"], [0.4])}
TITLE = {"net": "net stress (no notch factor)", "kt": "elastic peak Kt x net", "tcd_ln_paper": "TCD point, authors' L(N)",
         "tcd_ln": "TCD point, L(N) = A N^B", "tcd_const": "TCD point, constant L", "tcd_line": "TCD line, constant L",
         "tcd_line_ln": "TCD line, L(N) = A N^B", "sup_tcd": "supported TCD point, L and l",
         "sup_tcd_line": "supported TCD line, L and l", "weibull": "Weibull, m",
         "hotspot": "supported hot spot, l", "mhsv": "modified HSV, L_FCI"}


class Notch:
    def __init__(self, f, G):
        self.d, self.s = C.line_profile(f)
        self.lig = float(self.d[-1])
        self.kt = float(self.s[0])
        self.chi = C.root_chi(f)
        self.f, self.G = f, G
        self._J = {}

    def pm(self, L):
        return float(np.interp(np.clip(L / 2, 0, self.lig), self.d, self.s))

    def lm(self, L):
        """Line method: mean over [0, 2L], mirrored past the ligament (criteria.tcd_line)."""
        L = min(L, 0.99 * self.lig)
        return 1.0 / C.tcd_line(self.f, 1.0, L)

    def weib(self, m):
        k = round(m, 4)
        if k not in self._J:
            self._J[k] = math.exp((C.log_J(self.f, k) - math.log(V_PLAIN)) / k)
        return self._J[k]


def feff(name, n, N, th):
    if name == "net":
        return 1.0
    if name == "kt":
        return n.kt
    if name == "tcd_ln_paper":
        return n.pm(16.4 * N ** -0.242)
    if name == "tcd_ln":
        return n.pm(th[0] * N ** th[1])
    if name == "tcd_const":
        return n.pm(th[0])
    if name == "tcd_line":
        return n.lm(th[0])
    if name == "tcd_line_ln":
        return n.lm(th[0] * N ** th[1])
    if name == "sup_tcd":                             # supported TCD: sigma(L/2) / (1 + sqrt(l chi0)), f >= 1
        return max(1.0, n.pm(th[0]) / (1 + math.sqrt(th[1] * n.chi)))
    if name == "sup_tcd_line":
        return max(1.0, n.lm(th[0]) / (1 + math.sqrt(th[1] * n.chi)))
    if name == "weibull":
        return n.weib(th[0])
    if name == "hotspot":
        return max(1.0, n.kt / (1 + math.sqrt(th[0] * n.chi)))
    if name == "mhsv":                                # life-domain size factor mapped to stress with the plain k 5.1
        kl = W21 / (5.1 * math.log(T_BAND))
        ratio = 18 * 30.0 * th[0] / (n.G["S90"] * L_CD)
        return n.pm(L_CD) * ratio ** (-1 / (kl * 5.1))
    raise KeyError(name)


def resid(name, notches, rows, uts, th):
    return np.array([math.log10(feff(name, notches[k], N, th) * s * 2 / (1 - R) / uts[p] / P50(N))
                     for k, p, R, s, N in rows])


def fit(name, notches, rows, uts):
    names, th0 = CRIT[name]
    if not names:
        return []
    bounds = {"A": (0.1, 200), "B": (-1.0, 0.2), "L": (1e-3, 6.0), "m": (2.0, 200.0), "l": (1e-4, 50.0),
              "L_FCI": (1e-6, 100.0)}

    def unpack(z):
        out = []
        for v, nm in zip(z, names):
            lo, hi = bounds[nm]
            out.append(min(max(v if nm == "B" else math.exp(v), lo), hi))
        return out

    obj = lambda z: float(np.mean(resid(name, notches, rows, uts, unpack(z)) ** 2))
    z0 = [v if nm == "B" else math.log(v) for v, nm in zip(th0, names)]
    best = min((minimize(obj, [z + (s if nm != "B" else 0) for z, nm in zip(z0, names)], method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000)) for s in (-1.5, 0.0, 1.5)), key=lambda o: o.fun)
    return unpack(best.x)


def stats(r):
    return dict(bias=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))),
                band=float(np.mean(abs(r) < math.log10(T_BAND) / 2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    g, plain, spec = hsv_load()
    uts = {p: q["UTS"] for (p, _), q in plain.items()}
    notches = {k: Notch(g[k]["f"], g[k]) for k in g}
    rows = [(k, p, R, s, N) for k, p, R, s, N, ro in spec if not ro]
    cal = [r for r in rows if r[0] == "sharp"]
    held = [r for r in rows if r[0] != "sharp"]
    L = ["# One protocol for every criterion: fit on plain (Fig. 1 P50) + sharp V specimens, predict U r1 and U r3", "",
         f"Fitted on {len(cal)} failed sharp-V specimens; held out {len(held)} failed U-notch specimens. Residual "
         "log10(S_eff / P50(N)); band = printed P10-P90 (T 2.451).", "",
         "| criterion | fitted parameters | sharp V (fitted): bias / rms | HELD OUT U r1: bias / rms, in band | "
         "HELD OUT U r3: bias / rms, in band | HELD OUT both: rms, in band |", "|---|---|---|---|---|---|"]
    res = {}
    for name in CRIT:
        th = fit(name, notches, cal, uts)
        r = {k: resid(name, notches, [x for x in rows if x[0] == k], uts, th) for k in NAMES}
        res[name] = (th, r)
        h = stats(np.concatenate([r["intermediate"], r["blunt"]]))
        si, sb, ss = stats(r["intermediate"]), stats(r["blunt"]), stats(r["sharp"])
        par = ", ".join(f"{n} {v:.3g}" for n, v in zip(CRIT[name][0], th)) or (
            "A 16.4, B -0.242 (published, fitted by the authors)" if name == "tcd_ln_paper" else "none")
        L.append(f"| {TITLE[name]} | {par} | {ss['bias']:+.3f} / {ss['rms']:.3f} | {si['bias']:+.3f} / {si['rms']:.3f}, "
                 f"{100 * si['band']:.0f} % | {sb['bias']:+.3f} / {sb['rms']:.3f}, {100 * sb['band']:.0f} % | "
                 f"{h['rms']:.3f}, {100 * h['band']:.0f} % |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "calib_sharp_holdout.md"), "w").write("\n".join(L) + "\n")
    json.dump({k: dict(theta=v[0], resid={n: r.tolist() for n, r in v[1].items()}) for k, v in res.items()},
              open(os.path.join(a.out, "calib_sharp_holdout.json"), "w"), indent=1)
    print("\n".join(L))
    figure(res, notches, rows, uts, os.path.join(a.out, "fig_calib_sharp_holdout"))


GROUPS = [("baselines", "Baselines (no free parameter)", ["net", "kt"]),
          ("tcd_point", "TCD point family", ["tcd_ln_paper", "tcd_ln", "tcd_const", "sup_tcd"]),
          ("tcd_line", "TCD line family", ["tcd_line", "tcd_line_ln", "sup_tcd_line"]),
          ("volume", "Volume and gradient methods", ["weibull", "hotspot", "mhsv"])]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm"})
    return plt


def figure(res, notches, rows, uts, path):
    """One ranking chart plus one S-N figure per method family (at most 2 x 2 panels each)."""
    plt = _style()
    from matplotlib.lines import Line2D
    plain = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_fig1_plain.json")))
    Ns = np.geomspace(10, 1e7, 200)
    sty = {"sharp": dict(marker="v", s=30, facecolor="none", edgecolors="#9467bd", linewidths=1.1),
           "intermediate": dict(marker="s", s=24, color="#1f77b4"), "blunt": dict(marker="P", s=34, color="#d62728")}
    hd = [Line2D([], [], color="k", lw=1.6), Line2D([], [], color="k", ls="--"),
          Line2D([], [], marker="o", mfc="none", mec="0.7", ls=""),
          Line2D([], [], marker="v", mfc="none", mec="#9467bd", ls="", ms=8),
          Line2D([], [], marker="s", color="#1f77b4", ls=""), Line2D([], [], marker="P", color="#d62728", ls="", ms=8)]
    lb = ["plain P50, Fig. 1 (calibration)", "plain P10 / P90 (paper)", "plain points, Fig. 1 (extracted, calibration)",
          "V r0.15: FITTED", "U r1: HELD OUT", "U r3: HELD OUT"]
    for tag, gtitle, names in GROUPS:
        nc = len(names) if len(names) <= 3 else 2
        nr = -(-len(names) // nc)
        fig, axs = plt.subplots(nr, nc, figsize=(5.6 * nc + 0.6, 5.0 * nr + 1.0), sharex=True, sharey=True, layout="constrained",
                                squeeze=False)
        for ax, name in zip(axs.flat, names):
            th, r = res[name]
            ax.plot(Ns, P50(Ns), "k-", lw=1.6)
            ax.plot(Ns, P50(Ns) * math.sqrt(T_BAND), "k--", lw=1)
            ax.plot(Ns, P50(Ns) / math.sqrt(T_BAND), "k--", lw=1)
            ax.scatter([q["N"] for q in plain], [q["smax_uts"] for q in plain], s=9, facecolor="none", edgecolors="0.7",
                       linewidths=0.5, zorder=1)
            for k in ("sharp", "intermediate", "blunt"):
                N = np.array([x[4] for x in rows if x[0] == k])
                ax.scatter(N, P50(N) * 10 ** r[k], zorder=3 if k == "sharp" else 4, **sty[k])
            h = stats(np.concatenate([r["intermediate"], r["blunt"]]))
            par = ", ".join(f"{n} {v:.3g}" for n, v in zip(CRIT[name][0], th)) or (
                "published by the authors" if name == "tcd_ln_paper" else "no free parameter")
            ax.set_title(f"{TITLE[name]} ({par})\nheld out: rms {h['rms']:.3f} log10 S, {100 * h['band']:.0f} % in band",
                         fontsize=10.5)
            ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.05, 10) if tag == "baselines" else (0.08, 2.5))
            ax.grid(alpha=0.3, which="both")
        for ax in axs.flat[len(names):]:
            ax.set_visible(False)
        for ax in axs[-1]:
            ax.set_xlabel("cycles to failure $N_f$ (specimen)")
        for ax in axs[:, 0]:
            ax.set_ylabel(r"$\sigma_{eff,max}/\sigma_{UTS}$")
        fig.suptitle(f"{gtitle}: Ezeh-Susmel PLA, fitted on plain + sharp V, U notches held out", fontsize=12)
        fig.legend(hd, lb, loc="outside lower center", ncol=3, fontsize=9.5, frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(f"{path}_{tag}.{ext}", dpi=170)
        plt.close(fig)
    ranking(res, path + "_ranking")


def ranking(res, path):
    plt = _style()
    rows = []
    for name, (th, r) in res.items():
        h = np.concatenate([r["intermediate"], r["blunt"]])
        rows.append((float(np.sqrt(np.mean(h ** 2))), name, float(np.sqrt(np.mean(r["sharp"] ** 2))),
                     float(np.sqrt(np.mean(r["intermediate"] ** 2))), float(np.sqrt(np.mean(r["blunt"] ** 2)))))
    rows.sort()
    fig, ax = plt.subplots(figsize=(10, 6.2), layout="constrained")
    y = np.arange(len(rows))
    fam = {n: g for g, _, ns in GROUPS for n in ns}
    col = {"baselines": "0.55", "tcd_point": "#1f77b4", "tcd_line": "#2ca02c", "volume": "#d62728"}
    ax.barh(y, [r[0] for r in rows], color=[col[fam[r[1]]] for r in rows], height=0.6)
    ax.scatter([r[3] for r in rows], y, marker="s", color="k", s=18, zorder=3, label="U r1 alone (held out)")
    ax.scatter([r[4] for r in rows], y, marker="P", color="k", s=26, zorder=3, label="U r3 alone (held out)")
    ax.scatter([r[2] for r in rows], y, marker="v", facecolor="none", edgecolors="k", s=26, zorder=3,
               label="V r0.15 (fitted)")
    net = next(r[0] for r in rows if r[1] == "net")
    ax.axvline(net, color="0.4", ls=":", lw=1.2)
    ax.text(net, len(rows) - 0.4, " net stress, no fit", fontsize=9, color="0.3", va="top")
    ax.set_yticks(y, [TITLE[r[1]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("rms log10(S_eff / P50), bar = both held-out U notches")
    ax.set_title("Held-out error by method: Ezeh-Susmel PLA, fitted on plain + sharp V\n"
                 "colour: grey baselines, blue TCD point, green TCD line, red volume / gradient", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 0.9)
    ax.legend(fontsize=9, loc="center right")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
