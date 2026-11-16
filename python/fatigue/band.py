"""Ezeh & Susmel's own unifying plain band (2020 Fig. 1, from Ref. [10] = their 2019 paper) as the master curve, and
where the notched specimens land on it with each effective-stress criterion.

    python3 -m fatigue.band [--out results/fatigue]

Band, as printed in the figure: sigma_max/sigma_UTS = 0.16 (N/2e6)^(-1/5.1) at P_S 50 %, capped at 1 (static), scatter
T_sigma 2.451 (P_S 10 % and 90 % lines = P50 x/÷ sqrt(T)).
Data, every point marked by its origin:
    plain (data/ezeh_susmel_2020_fig1_plain.json): EXTRACTED from the vector plot of Fig. 1, 195 points, R from the
        marker shape. The figure pools Refs [7] Letcher 2014, [8] Afrose 2016 and [10] Ezeh & Susmel 2019: other
        printers, and no print angle per point.
    notched (data/ezeh_susmel_2020_specimens.json): EXTRACTED from Figs 5-7, 157 specimens (8 run-outs).
    Seifollahi & Kabir: every specimen from their Table 6 (another printer and PLA), for comparison.
Notched specimen: S = f_eff sigma_a,net 2/(1-R) / sigma_UTS(p), UTS per print angle from Ezeh & Susmel 2019 Table 1.
Score: residual log10(S / P50(N)) of every failed specimen; share inside the printed P10-P90 band.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize_scalar

from . import criteria as C
from .fields import field
from .specimens import Specimen
from .transfer import HERE, OUT

T_BAND, K_BAND, S50_BAND = 2.451, 5.1, 0.16
P50 = lambda N: np.minimum(S50_BAND * (np.asarray(N, float) / 2e6) ** (-1 / K_BAND), 1.0)
L_N = lambda N: 16.4 * N ** -0.242                         # authors' TCD L(N), mm (2020 eq. 6)


def feff(name, f, N, par=None):
    d, s = C.line_profile(f)
    lig = float(d[-1])
    if name == "net":
        return 1.0
    if name == "kt":
        return float(s[0])
    L = L_N(N) if par is None else par
    if name == "tcd_point":
        return 1.0 / C.tcd_point(f, 1.0, min(L, 1.9 * lig))
    if name == "tcd_line":
        return 1.0 / C.tcd_line(f, 1.0, min(L, 0.95 * lig))
    if name == "hotspot":
        return max(1.0, 1.0 / C.supported_kt(f, 1.0, par))
    raise KeyError(name)


def load():
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    uts = {q["p"]: q["UTS"] for q in E["plain"]}
    F = {k: field(Specimen.from_dict(v), h=1.5, log=lambda *_: None) for k, v in E["specimens"].items()}
    sp = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_specimens.json")))
    spec = [(q["notch"], q["p"], q["R"], t["sigma_na"], t["N"], t["runout"]) for q in sp["series"] for t in q["specimens"]]
    plain = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_fig1_plain.json")))
    S = json.load(open(os.path.join(HERE, "data", "seifollahi_kabir.json")))
    FS = {k: field(Specimen.from_dict(v), h=1.5, log=lambda *_: None) for k, v in S["specimens"].items()}
    return E, uts, F, spec, plain, S, FS


def S_notched(name, F, uts, spec, par=None):
    return np.array([feff(name, F[n], N, par) * s * 2 / (1 - R) / uts[p] for n, p, R, s, N, ro in spec])


def stats(r):
    return dict(mean=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))),
                in_band=float(np.mean(abs(r) < math.log10(T_BAND) / 2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    E, uts, F, spec, plain, S, FS = load()
    N = np.array([t[4] for t in spec], float)
    fail = ~np.array([t[5] for t in spec])
    blunt = np.array([t[0] == "blunt" for t in spec])
    res, lines = {}, ["# Notched specimens on Ezeh & Susmel's unifying plain band (sigma_max/sigma_UTS, P50 0.16 at 2e6, "
                      "k 5.1, T 2.451)", "", "Residual log10(S/P50) of the 149 failed notched specimens.", "",
                      "| criterion | parameter | all: mean / rms, in band | without blunt: mean / rms, in band |",
                      "|---|---|---|---|"]
    cases = [("net", None, "-"), ("kt", None, "-"), ("tcd_point", None, "authors' L(N) = 16.4 N^-0.242"),
             ("tcd_line", None, "authors' L(N) = 16.4 N^-0.242")]
    for name, lo, hi in (("tcd_point", 0.05, 8.0), ("tcd_line", 0.05, 4.0), ("hotspot", 0.01, 20.0)):
        o = minimize_scalar(lambda z: float(np.mean(np.log10(S_notched(name, F, uts, spec, math.exp(z))[fail]
                                                             / P50(N[fail])) ** 2)),
                            bounds=(math.log(lo), math.log(hi)), method="bounded")
        cases.append((name, math.exp(o.x), f"{'L' if name != 'hotspot' else 'l'} {math.exp(o.x):.2f} mm fitted here"))
    for name, par, lab in cases:
        Sx = S_notched(name, F, uts, spec, par)
        r = np.log10(Sx / P50(N))
        sa, sb = stats(r[fail]), stats(r[fail & ~blunt])
        res[(name, lab)] = (Sx, sa, sb)
        lines.append(f"| {name} | {lab} | {sa['mean']:+.3f} / {sa['rms']:.3f}, {100 * sa['in_band']:.0f} % | "
                     f"{sb['mean']:+.3f} / {sb['rms']:.3f}, {100 * sb['in_band']:.0f} % |")
    # plain points: refit check of the printed band (failures past the static knee)
    Np = np.array([q["N"] for q in plain]); sp_ = np.array([q["smax_uts"] for q in plain])
    ok = (Np < 1.9e6) & (Np > 150)
    b, c = np.polyfit(np.log10(sp_[ok]), np.log10(Np[ok]), 1)
    lines += ["", f"Check of the extraction: refit of the {ok.sum()} failed plain points past the static knee gives "
              f"k {-b:.2f}, sigma_max,50/UTS {10 ** ((math.log10(2e6) - c) / b):.3f} at 2e6 (printed 5.1, 0.16)."]
    # Walker exponent on their own plain points
    R = np.array([q["R"] for q in plain])
    for g in (0.0, 0.45, 1.0):
        Sg = sp_ * (1 - R) / 2 * (2 / (1 - R)) ** (1 - g)
        bb, cc = np.polyfit(np.log10(Np[ok]), np.log10(Sg[ok]), 1)
        rr = np.log10(Sg[ok]) - (cc + bb * np.log10(Np[ok]))
        lines.append(f"- plain, Walker gamma {g:g}: rms log10 S {np.sqrt(np.mean(rr ** 2)):.3f}; mean residual by R " +
                     ", ".join(f"R {x:g}: {rr[R[ok] == x].mean():+.3f}" for x in (-1, -0.5, 0, 0.3)))
    # Seifollahi on the same band
    for k in ("smooth", "edge", "hole"):
        rr = []
        for s, *ns in S["raw_lives"][k]:
            for n in ns:
                f = 1.0 if k == "smooth" else feff("tcd_line", FS[k], n)
                rr.append(math.log10(f * s * 2 / 0.95 / S["UTS_MPa"] / P50(n)))
        st = stats(np.array(rr))
        lines.append(f"- Seifollahi {k}{' (TCD line, authors L(N))' if k != 'smooth' else ''}: mean {st['mean']:+.3f}, "
                     f"in band {100 * st['in_band']:.0f} %")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "unifying_band.md"), "w").write("\n".join(lines) + "\n")
    figure(res, spec, plain, S, FS, os.path.join(a.out, "fig_unifying_band"))
    print("\n".join(lines))


def figure(res, spec, plain, S, FS, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    show = [k for k in res if k[0] == "net"] + [k for k in res if k[0] == "tcd_line" and "authors" in k[1]] + \
        [k for k in res if k[0] == "kt"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.0), sharey=True, layout="constrained")
    Ns = np.geomspace(10, 1e7, 200)
    col = {"blunt": "0.55", "intermediate": "#1f77b4", "sharp": "#9467bd"}
    mk = {"blunt": "P", "intermediate": "s", "sharp": "v"}
    for ax, key in zip(axs, show):
        ax.plot(Ns, P50(Ns), "k-", lw=1.6)
        for f_, ls in ((math.sqrt(T_BAND), "--"), (1 / math.sqrt(T_BAND), "--")):
            ax.plot(Ns, P50(Ns) * f_, "k", ls=ls, lw=1.0)
        ax.scatter([q["N"] for q in plain], [q["smax_uts"] for q in plain], s=12, facecolor="none", edgecolors="0.6",
                   linewidths=0.6, zorder=2)
        Sx, sa, sb = res[key]
        for n in ("blunt", "intermediate", "sharp"):
            k = np.array([t[0] == n for t in spec])
            ro = np.array([t[5] for t in spec])
            ax.scatter(np.array([t[4] for t in spec])[k & ~ro], Sx[k & ~ro], marker=mk[n], s=20, color=col[n], zorder=4)
            ax.scatter(np.array([t[4] for t in spec])[k & ro], Sx[k & ro], marker=">", s=22, color=col[n], zorder=4)
        for k, m in (("smooth", "o"), ("edge", "^"), ("hole", "D")):
            xs, ys = [], []
            for s, *ns in S["raw_lives"][k]:
                for n in ns:
                    f = 1.0 if k == "smooth" or key[0] == "net" else (
                        float(C.line_profile(FS[k])[1][0]) if key[0] == "kt" else feff("tcd_line", FS[k], n))
                    xs.append(n); ys.append(f * s * 2 / 0.95 / S["UTS_MPa"])
            ax.scatter(xs, ys, marker=m, s=26, facecolor="none" if k == "smooth" else "#d62728", edgecolors="#d62728",
                       linewidths=1.0, zorder=5)
        name = {"net": "notched at NOMINAL net stress (no notch factor)", "tcd_line": "notched at TCD line stress, "
                "authors' L(N)", "kt": "notched at elastic peak Kt x net"}[key[0]]
        ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.03, 10), xlabel="cycles to failure $N_f$",
               title=f"{name}\nEzeh notched in band: {100 * sa['in_band']:.0f} % ({100 * sb['in_band']:.0f} % without blunt)")
        ax.title.set_fontsize(9.5)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"$\sigma_{max,eff}/\sigma_{UTS}$")
    axs[0].text(0.02, 0.02, "Band: Ezeh & Susmel 2020 Fig. 1 as printed (P50, P10, P90).\nGrey circles: Fig. 1 plain "
                "points, EXTRACTED from the vector plot\n(pools Refs 7, 8, 10: three printers, no print angle).\n"
                "Blue/purple/grey: Ezeh notched SPECIMENS, extracted from Figs 5-7 (> run-out).\n"
                "Red: Seifollahi SPECIMENS (Table 6), another printer and PLA.", transform=axs[0].transAxes, fontsize=7.5,
                va="bottom", bbox=dict(fc="white", ec="0.7"))
    h = [Line2D([], [], color="k", lw=1.6), Line2D([], [], color="k", ls="--"),
         Line2D([], [], marker="o", mfc="none", mec="0.6", ls=""), Line2D([], [], marker="s", color=col["intermediate"], ls=""),
         Line2D([], [], marker="v", color=col["sharp"], ls=""), Line2D([], [], marker="P", color=col["blunt"], ls=""),
         Line2D([], [], marker="o", mfc="none", mec="#d62728", ls=""), Line2D([], [], marker="^", color="#d62728", ls=""),
         Line2D([], [], marker="D", color="#d62728", ls="")]
    lb = ["band P50 (paper)", "band P10 / P90 (paper)", "plain, Fig. 1 points (extracted)", "Ezeh U r1 (specimen)",
          "Ezeh V r0.15 (specimen)", "Ezeh U r3 blunt (specimen)", "Seifollahi smooth (specimen)",
          "Seifollahi edge R2 (specimen)", "Seifollahi hole R2 (specimen)"]
    fig.legend(h, lb, loc="outside lower center", ncol=5, fontsize=8.5, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
