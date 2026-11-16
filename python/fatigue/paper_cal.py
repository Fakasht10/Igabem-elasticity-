"""Every criterion calibrated exactly the way Ezeh & Susmel 2020 (Sec. 6, Fig. 9e) calibrate the TCD point method.

Variable sigma_max/UTS, one UTS 42.9 MPa, no mean-stress correction. Calibration curves: plain P50 (6.7 MPa at 2e6,
k 5.1) and the pooled sharp-V net sigma_max S-N curve (log N on log S, R -1 and 0, failures). At each N on a grid the
criterion's parameter p(N) is solved so that factor(p) * S_V(N) = S_plain(N); then p(N) = A N^B is fitted where a
solution exists. Every specimen (all notches, R -1 and 0) is then mapped to sigma_eff,max/UTS with its own N."""
import json, os, numpy as np
from scipy.optimize import brentq
from . import calib as c
from .hsv import load
from .transfer import OUT

UTS = c.UTS_PAPER
S0 = lambda N: np.minimum(6.7 * (np.asarray(N, float) / 2e6) ** (-1 / 5.1), UTS)
NGRID = np.geomspace(1e3, 2e6, 40)


def methods(no, LM):
    """name -> (factor(notch, p, N), p bracket, label). LM(N) is our point-method L used by supported TCD line."""
    return {
        "tcd_point": (lambda n, p, N: n.pm(p), (1e-3, 2 * 0.99 * no["sharp"].lig), "TCD point, L(N)"),
        "tcd_line": (lambda n, p, N: n.lm(p), (1e-3, 0.99 * no["sharp"].lig), "TCD line, L(N)"),
        "weibull": (lambda n, p, N: n.weib(p), (3.2, 200.0), "Weibull, m(N)"),
        "hsv80": (lambda n, p, N: n.kt * (n.vq(0.8) / c.V_PLAIN) ** (1 / p), (1.5, 200.0), "HSV V80, m(N)"),
        "sup_tcd_line": (lambda n, p, N: max(1.0, n.lm(LM(N)) / (1 + np.sqrt(p * n.chi))), (1e-6, 50.0),
                         "supported TCD line, l(N) (L = our L_M(N))")}


def calibrate(fac, br, n, sv):
    out = []
    for N in NGRID:
        h = lambda lp: np.log(fac(n, np.exp(lp), N) * sv(N) / S0(N))
        try:
            out.append((N, np.exp(brentq(h, *np.log(br), xtol=1e-6))))
        except ValueError:
            pass
    P = np.array(out)
    if len(P) < 3:
        return None, P
    B, lA = np.polyfit(np.log10(P[:, 0]), np.log10(P[:, 1]), 1)
    return (10 ** lA, B), P


def main():
    g, plain, spec = load()
    no = {k: c.Notch(g[k]["f"], g[k]) for k in g}
    sh = np.array([(N, s * 2 / (1 - R)) for k, p, R, s, N, ro in spec if k == "sharp" and not ro])
    b, a = np.polyfit(np.log10(sh[:, 1]), np.log10(sh[:, 0]), 1)
    sv = lambda N: 10 ** ((np.log10(N) - a) / b)
    (A0, B0), _ = calibrate(lambda n, p, N: n.pm(p), (1e-3, 2 * 0.99 * no["sharp"].lig), no["sharp"], sv)
    LM = lambda N: A0 * N ** B0
    th = json.load(open(os.path.join(os.path.dirname(__file__), "data", "ezeh_susmel_2020_fig10_point_method.json")))["points"]
    res = {"authors": dict(label="authors' TCD point (Fig. 10 points, their L = 16.4 N^-0.242)",
                           pts={k: np.array([(q["x"], q["y"]) for q in th if q["type"] == k and not q["runout"]]) for k in no})}
    for name, (fac, br, label) in methods(no, LM).items():
        ab, P = calibrate(fac, br, no["sharp"], sv)
        if ab is None:
            res[name] = dict(label=label + ": no solution on the V curve", pts=None); continue
        A, B = ab
        lo, hi = br
        par = lambda N: min(max(A * N ** B, lo), hi)
        pts = {k: np.array([(N, fac(no[k], par(N), N) * s * 2 / (1 - R) / UTS) for kk, p, R, s, N, ro in spec
                            if kk == k and not ro]) for k in no}
        res[name] = dict(label=label, pts=pts, A=A, B=B, solved=(P[0, 0], P[-1, 0], len(P)))
    L = ["# Every criterion calibrated the paper's way (plain P50 + pooled sharp-V curve), sigma_max/UTS, UTS 42.9", "",
         f"Sharp-V net sigma_max S-N: k {-b:.2f}, S50(2e6) {sv(2e6):.2f} MPa. Plain: 6.7 MPa at 2e6, k 5.1.", "",
         "| method | p(N) = A N^B | solved on N grid | trend k (all notches) | trend at 2e6 | U r1 median / in band | "
         "U r3 median / in band | V median / in band |", "|---|---|---|---|---|---|---|---|"]
    half = np.log10(2.451) / 2
    for name, r in res.items():
        if r["pts"] is None:
            L.append(f"| {r['label']} | - | none | - | - | - | - | - |"); continue
        allp = np.vstack(list(r["pts"].values())); bb, ii = np.polyfit(np.log10(allp[:, 0]), np.log10(allp[:, 1]), 1)
        r["trend"] = (-1 / bb, 10 ** (ii + bb * np.log10(2e6)), bb, ii)
        cells = []
        for k in ("intermediate", "blunt", "sharp"):
            q = r["pts"][k]; rr = np.log10(q[:, 1] / np.minimum(0.155 * (q[:, 0] / 2e6) ** (-1 / 5.1), 1))
            cells.append(f"{np.median(rr):+.3f} / {100 * np.mean(abs(rr) < half):.0f} %")
        pn = f"{r['A']:.3g} N^{r['B']:+.3f}" if "A" in r else "16.4 N^-0.242 (published)"
        sv_ = f"{r['solved'][0]:.0f}-{r['solved'][1]:.0f} ({r['solved'][2]}/40)" if "solved" in r else "-"
        L.append(f"| {r['label']} | {pn} | {sv_} | {r['trend'][0]:.2f} | {r['trend'][1]:.3f} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "paper_cal.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
    figure(res, os.path.join(OUT, "fig_paper_cal_trends"))


def figure(res, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm"})
    def drawn(y0, y1, N0=149.5):
        N = np.geomspace(10, 2e6, 300); sl = np.log(y1 / y0) / np.log(2e6 / N0)
        return N, np.where(N < N0, y0, y0 * (N / N0) ** sl)
    fig, ax = plt.subplots(figsize=(9.5, 6.8), layout="constrained")
    ax.plot(*drawn(0.9965, 0.1551), "k-", lw=1.6, label="plain P50 (paper, as drawn): 0.155 at 2e6, k 5.1")
    ax.plot(*drawn(1.5591, 0.2427), "k:", lw=1.1, label="plain P10 / P90")
    ax.plot(*drawn(0.6369, 0.0991), "k:", lw=1.1)
    sty = {"authors": ("#000000", (0, (1, 0)), 3.0), "tcd_point": ("#d62728", "--", 2.2), "tcd_line": ("#2ca02c", "--", 2.2),
           "weibull": ("#1f77b4", "--", 2.2), "hsv80": ("#9467bd", "--", 2.2), "sup_tcd_line": ("#ff7f0e", "--", 2.2)}
    Nf = np.geomspace(1e2, 2e6, 60)
    for name, r in res.items():
        if "trend" not in r: continue
        k, s2, bb, ii = r["trend"]; col, ls, lw = sty[name]
        ax.plot(Nf, 10 ** (ii + bb * np.log10(Nf)), color=col, ls=ls, lw=lw, alpha=0.6 if name == "authors" else 1,
                label=f"{r['label']}: k {k:.2f}, {s2:.3f} at 2e6")
    ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.01, 10), xlabel="$N_f$ [cycles to failure]",
           ylabel=r"$\sigma_{eff,max}/\sigma_{UTS}$")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.3, loc="lower left")
    ax.set_title("Trend line through all 149 failed notched specimens (V, U r1, U r3; R -1 and 0), one per method.\n"
                 "Each method calibrated the paper's way on our IgBEM stresses; UTS 42.9 MPa; no mean-stress term", fontsize=10.5)
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=170)


if __name__ == "__main__":
    main()
