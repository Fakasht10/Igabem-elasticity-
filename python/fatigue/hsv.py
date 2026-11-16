"""Modified highly-stressed-volume (HSV) method of Zhang, Zhang, Hu, Chen, Susmel & Wei, IJF 172 (2023) 107644, applied
to every Ezeh & Susmel 2020 notched PLA specimen, next to the traditional HSV and the plain point method.

    python3 -m fatigue.hsv [--L-fci 0.4] [--gauge 30] [--out results/fatigue]

The method, as written in the paper (eq. numbers theirs):
    (8)  sigma_eff = sigma_1(L_CD / 2) on the notch bisector (point method);
    (9)  N_S = plain S-N life at sigma_eff;
    (3)  N_N = N_S (V_S / V_N)^(1/k), k = Weibull shape of the LIFE distribution at one stress level;
    (5)  V_N = S_N L_CD,  (7) V_S = S_S L_FCI;  S = surface area where sigma_1 >= 0.9 max sigma_1 (highly stressed surface);
    (4)  L_CD = (dK_th / d sigma_0)^2 / pi;  L_FCI = depth of the crack-initiation region on the smooth fracture surface.
Traditional HSV (their Fig. 10a): sigma = peak sigma_1 = Kt sigma_n, N_N = N_S(sigma_max) (V90_S / V90_N)^(1/k), V90 = volume
where sigma_1 >= 0.9 max.

Inputs for Ezeh-Susmel PLA and where each one comes from:
    plain S-N per print angle and R       2019 Table 1 (n, k, sigma_A50 at 2e6, T): same printer, same R as the notch.
    k (life Weibull shape)                from that plain series: life scatter T_N = T_sigma^k_SN, k = ln(21.85)/ln T_N
                                          (Weibull P_f 10 %/90 % life ratio); no notched data.
    L_CD                                  0.49 mm = the authors' L_M(N) = 16.4 N^-0.242 at 2e6 (2020 eq. 6). dK_th of
                                          PLA is not given; the authors calibrated L on the sharp notch, so L_CD is
                                          fitted on Ezeh-Susmel (sharp V + plain), not independent.
    L_FCI                                 NOT measured for PLA plain specimens (no smooth fractography in either paper).
                                          Default 0.4 mm = the 0.4 mm shell (perimeter) thickness the 2020 paper reports as
                                          the embryonic crack length on the NOTCHED fracture surfaces. ASSUMED; swept.
    S_N, V90_N                            IgBEM: sigma_1 evaluated ON the root surface (fields.root_surface), V90 from the
                                          TET10 volume field (fields.field).
    S_S, V90_S                            plain gauge, uniform stress: S_S = 2 (W + T) Lg, V_S = W T Lg; gauge length Lg
                                          is not given in the paper (ASSUMED 30 mm, swept).
Score: log10(N_pred / N) for each of the 149 failed specimens (vector-extracted from 2020 Figs 5-7), and the share inside
the predicted P_S 10-90 % life band (the paper's own validation criterion).
"""
import argparse
import json
import math
import os

import numpy as np

from .fields import field, root_surface
from .specimens import Specimen
from .transfer import HERE, OUT

L_CD = 16.4 * 2e6 ** -0.242                      # 0.489 mm
W21 = math.log(math.log(10) / math.log(10 / 9))  # ln 21.85: Weibull life ratio P_f 90 % / 10 % = 21.85^(1/k)
NAMES = {"blunt": "U r3 (blunt)", "intermediate": "U r1", "sharp": "V r0.15 (sharp)"}


def geometry(F, S):
    """Per notch: peak surface sigma_1 (unit net stress), S90 (mm^2), V90 (mm^3), sigma_1 at L_CD/2 on the bisector."""
    g = {}
    for k in F:
        s, f = S[k], F[k]
        sm = max(s["s1_arc"].max(), s["s1_side"].max())
        S90 = 4 * (s["dA_arc"] * (s["s1_arc"] >= 0.9 * sm)).sum() + 8 * (s["dA_side"] * (s["s1_side"] >= 0.9 * sm)).sum()
        V90 = 4 * f["w"][f["s1"] >= 0.9 * f["s1"].max()].sum()
        prof = f["s1_line"].mean(axis=0)                 # thickness-averaged bisector profile
        g[k] = dict(smax=float(sm), S90=float(S90), V90=float(V90), seff=float(np.interp(L_CD / 2, f["d"], prof)),
                    prof=lambda x, d=f["d"], p=prof: float(np.interp(x, d, p)))
    return g


def plain_curve(src, pl, R):
    """(sigma_A50 at 2e6 as an amplitude at this R, S-N inverse slope, T_sigma) of the plain curve used for N_S.
    series:  2019 Table 1 at the same print angle and R.
    unified: 2020 Fig. 1 unifying band in sigma_max/UTS (P50 0.16 at 2e6, k 5.1, T 2.451), as the authors use it."""
    if src == "series":
        return pl["sigma_A50"], pl["k"], pl["T"]
    return 0.16 * pl["UTS"] * (1 - R) / 2, 5.1, 2.451


def predict(method, g, pl, notch, s_a, L_fci, gauge, W=6.0, T=3.0, src="series", R=-1):
    """Median life of a notched specimen at net amplitude s_a, and the life Weibull shape k used."""
    G = g[notch]
    s0, kb, Ts = plain_curve(src, pl, R)
    k = W21 / (kb * math.log(Ts))
    if method == "tcd_pm":
        return 2e6 * (G["seff"] * s_a / s0) ** -kb, k
    if method == "tcd_pm_LN":                          # authors' own TCD: L_M(N) solved with the life (fixed point)
        N = 1e4
        for _ in range(60):
            N = 2e6 * (G["prof"](16.4 * N ** -0.242 / 2) * s_a / s0) ** -kb
        return N, k
    if method == "hsv":
        NS = 2e6 * (G["smax"] * s_a / s0) ** -kb
        return NS * (W * T * gauge / G["V90"]) ** (1 / k), k
    if method == "mhsv_LN":                            # NOT in the paper: L_M(N) in both sigma_eff and V_N
        N = 1e4
        for _ in range(60):
            Ln = 16.4 * N ** -0.242
            N = 2e6 * (G["prof"](Ln / 2) * s_a / s0) ** -kb * (2 * (W + T) * gauge * L_fci / (G["S90"] * Ln)) ** (1 / k)
        return N, k
    if method == "mhsv":
        NS = 2e6 * (G["seff"] * s_a / s0) ** -kb
        return NS * (2 * (W + T) * gauge * L_fci / (G["S90"] * L_CD)) ** (1 / k), k
    raise KeyError(method)


def score(method, g, plain, spec, L_fci, gauge, src="series"):
    e, inb = [], []
    for notch, p, R, s, N, ro in spec:
        if ro:
            continue
        N50, k = predict(method, g, plain[(p, R)], notch, s, L_fci, gauge, src=src, R=R)
        e.append(math.log10(N50 / N))
        # P_S 10-90 % band: N_p = N50 (ln(1/P_S) / ln 2)^(1/k)
        lo, hi = N50 * (math.log(1 / 0.9) / math.log(2)) ** (1 / k), N50 * (math.log(1 / 0.1) / math.log(2)) ** (1 / k)
        inb.append(lo <= N <= hi)
    return np.array(e), np.array(inb)


def load():
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    sp = {k: Specimen.from_dict(v) for k, v in E["specimens"].items()}
    F = {k: field(v, h=1.5, log=lambda *_: None) for k, v in sp.items()}
    S = {k: root_surface(v, log=lambda *_: None) for k, v in sp.items()}
    plain = {(q["p"], q["R"]): q for q in E["plain"]}
    js = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_specimens.json")))
    spec = [(q["notch"], q["p"], q["R"], t["sigma_na"], t["N"], t["runout"]) for q in js["series"] for t in q["specimens"]]
    return geometry(F, S), plain, spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L-fci", type=float, default=0.4, help="ASSUMED smooth crack-initiation depth (mm)")
    ap.add_argument("--gauge", type=float, default=30.0, help="ASSUMED plain gauge length (mm)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    g, plain, spec = load()
    notch = np.array([t[0] for t in spec if not t[5]])
    ks = {pr: W21 / (q["k"] * math.log(q["T"])) for pr, q in plain.items()}
    L = ["# Modified HSV (Zhang et al. 2023) on every Ezeh-Susmel notched PLA specimen", "",
         f"L_CD {L_CD:.3f} mm (authors' L_M at 2e6, fitted on Ezeh-Susmel), L_FCI {a.L_fci:g} mm (ASSUMED), plain gauge "
         f"{a.gauge:g} mm (ASSUMED): S_S {2 * 9 * a.gauge:.0f} mm^2, V90_S {18 * a.gauge:.0f} mm^3.", "",
         "Life Weibull shape k from each plain series (2019 Table 1): " + ", ".join(
             f"p{p} R{R:g} {v:.2f}" for (p, R), v in sorted(ks.items())), "",
         "| notch | peak sigma_1 / sigma_n (surface) | S90 (mm^2) | V90 (mm^3) | V_N = S90 L_CD (mm^3) | "
         "sigma_1(L_CD/2) / sigma_n |", "|---|---|---|---|---|---|"]
    for k, G in g.items():
        L.append(f"| {NAMES[k]} | {G['smax']:.3f} | {G['S90']:.3f} | {G['V90']:.4f} | {G['S90'] * L_CD:.3f} | "
                 f"{G['seff']:.3f} |")
    L += ["", "Every failed specimen (149). Cells: bias / rms log10(N_pred/N); within x2; within x3; inside the "
          "predicted P_S 10-90 % band. Per notch: bias.", "",
          "| method | all: bias / rms | within x2 | within x3 | in P10-P90 band | blunt | U r1 | V sharp |", "|---|---|---|---|---|---|---|---|"]
    lab = {"tcd_pm_LN": "TCD point, authors' L_M(N) (their own method)", "tcd_pm": "TCD point, L_CD 0.49 (no size term)",
           "hsv": "traditional HSV (V90, peak stress)", "mhsv": f"modified HSV (L_FCI {a.L_fci:g})",
           "mhsv_LN": f"variant, NOT in the paper: modified HSV with L_M(N) (L_FCI {a.L_fci:g})"}
    srcs = {"series": "plain S-N = 2019 Table 1 series at the same p and R",
            "unified": "plain S-N = 2020 Fig. 1 unifying curve (P50 0.16 UTS at 2e6, k 5.1, T 2.451)"}
    res = {}
    for src, sl in srcs.items():
        L.append(f"| **{sl}** | | | | | | | |")
        for m in lab:
            e, ib = score(m, g, plain, spec, a.L_fci, a.gauge, src)
            res[(src, m)] = (e, ib)
            L.append(f"| {lab[m]} | {e.mean():+.2f} / {np.sqrt(np.mean(e ** 2)):.2f} | "
                     f"{np.mean(abs(e) < math.log10(2)) * 100:.0f} % | {np.mean(abs(e) < math.log10(3)) * 100:.0f} % | "
                     f"{100 * ib.mean():.0f} % | " +
                     " | ".join(f"{e[notch == n].mean():+.2f}" for n in ("blunt", "intermediate", "sharp")) + " |")
    for src in srcs:
        L += ["", f"Sensitivity of the modified HSV to the two assumed inputs, {srcs[src]} (bias / rms log10 life, "
              "in P10-P90 band):", "", "| L_FCI (mm) | gauge 20 mm | gauge 30 mm | gauge 50 mm |", "|---|---|---|---|"]
        for lf in (0.05, 0.1, 0.2, 0.4, L_CD, 1.0):
            row = []
            for lg in (20.0, 30.0, 50.0):
                e, ib = score("mhsv", g, plain, spec, lf, lg, src)
                row.append(f"{e.mean():+.2f} / {np.sqrt(np.mean(e ** 2)):.2f}, {100 * ib.mean():.0f} %")
            L.append(f"| {lf:.3g} | " + " | ".join(row) + " |")
        grid = np.geomspace(1e-4, 100, 400)
        b = min(grid, key=lambda lf: np.mean(score("mhsv", g, plain, spec, lf, a.gauge, src)[0] ** 2))
        e, ib = score("mhsv", g, plain, spec, b, a.gauge, src)
        L.append(f"\nIn-sample: L_FCI FITTED on these notched specimens (gauge {a.gauge:g}): {b:.3g} mm, bias "
                 f"{e.mean():+.2f}, rms {np.sqrt(np.mean(e ** 2)):.2f}, per notch " +
                 ", ".join(f"{n} {e[notch == n].mean():+.2f}" for n in ("blunt", "intermediate", "sharp")))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "hsv_ezeh.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    figure(res, lab, srcs, spec, notch, os.path.join(a.out, "fig_hsv_ezeh"))


def figure(res, lab, srcs, spec, notch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    N = np.array([t[4] for t in spec if not t[5]])
    col = {"blunt": "0.45", "intermediate": "#1f77b4", "sharp": "#9467bd"}
    mk = {"blunt": "P", "intermediate": "s", "sharp": "v"}
    ms = ("tcd_pm_LN", "tcd_pm", "hsv", "mhsv")
    fig, axs = plt.subplots(2, 4, figsize=(19, 10), layout="constrained", sharex=True, sharey=True)
    x = np.array([1e1, 1e10])
    for row, src in zip(axs, srcs):
        for ax, m in zip(row, ms):
            e, ib = res[(src, m)]
            ax.plot(x, x, "k-", lw=1)
            for f_, ls in ((2, "--"), (3, ":")):
                ax.plot(x, x * f_, "k", ls=ls, lw=0.9)
                ax.plot(x, x / f_, "k", ls=ls, lw=0.9)
            for n in ("blunt", "intermediate", "sharp"):
                s = notch == n
                ax.scatter(N[s], N[s] * 10 ** e[s], marker=mk[n], s=20, color=col[n], label=f"{NAMES[n]} (specimen)")
            ax.set(xscale="log", yscale="log", xlim=(50, 1e7), ylim=(1, 1e10))
            ax.set_title(f"{lab[m]}\nbias {e.mean():+.2f}, rms {np.sqrt(np.mean(e ** 2)):.2f} log10; "
                         f"{100 * np.mean(abs(e) < math.log10(2)):.0f} % within x2", fontsize=9)
            ax.grid(alpha=0.3, which="both")
        row[0].set_ylabel(f"predicted median life\n({'2019 Table 1 plain series' if src == 'series' else 'Fig. 1 unifying plain curve'})")
    for ax in axs[1]:
        ax.set_xlabel("measured life $N_f$ (specimen, extracted from 2020 Figs 5-7)")
    axs[0, 0].legend(fontsize=8, loc="upper left", title="solid 1:1, dashed x2, dotted x3", title_fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
