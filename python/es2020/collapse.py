"""Dimensionless collapse of the notched data (Buckingham-Pi study, 2026-11-16).

    python -m es2020.collapse [out_dir]  -> collapse.md, fig_collapse_groups.png/pdf, fig_collapse_closures.png/pdf

Response group Pi_S = sigma_n,max / sigma_0(N), sigma_0 = plain P50 (paper: 6.7 MPa at 2e6, k 5.1, capped at UTS).
Per specimen K_f,i = sigma_0(N_i) / sigma_n,max,i (Fig. 8 stresses, the paper's chain). Groups:
    rho / L(N)      notch number          L(N) = paper Eq. 6, 16.4 N^-0.242 mm
    Lambda = b / L  ligament number       b = 3 mm half net width
    sigma_n,max/UTS static group          UTS 42.9 MPa
Closures K_f(N[, sigma]) are calibrated the paper's way (plain P50 + Fig. 8a sharp curve on 1e3..2e6) and U r1,
U r3 are never used in a fit. Post-processing only: K_t, chi and the profiles are the stored IgBEM fields.
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

from . import methods as M
from . import paper

RHO = {"sharp": 0.15, "intermediate": 1.0, "blunt": 3.0}
B_HALF = 3.0                       # half net width (mm)
W_HALF, A_DEPTH = 12.5, 9.5        # half gross width, notch depth (mm)


def kt(notch):
    return float(M._prof(notch).s[0])


def dent_g():
    """K_I / sigma_net (sqrt mm) of a double-edge crack as deep as the notches, Tada's DENT formula
    (Tada, Paris & Irwin, Stress Analysis of Cracks Handbook, 2.6): K = sigma_g sqrt(pi a) F(a/W_half)."""
    x = A_DEPTH / W_HALF
    F = (1.122 - 0.561 * x - 0.205 * x ** 2 + 0.471 * x ** 3 - 0.190 * x ** 4) / math.sqrt(1 - x)
    return math.sqrt(math.pi * A_DEPTH) * F * (2 * B_HALF) / (2 * W_HALF)


def _len(p, N):
    return p[0] * np.asarray(N, float) ** p[1]


def _lpaper(p, N):
    return p[0] * paper.lm_paper(N)


# K_f closures: f(notch, N, s_max, p). Each comes with start p0 and how p maps to a length.
def peterson(n, N, s, p, L=_len):
    return 1 + (kt(n) - 1) / (1 + L(p, N) / RHO[n])


def neuber(n, N, s, p, L=_len):
    return 1 + (kt(n) - 1) / (1 + np.sqrt(L(p, N) / RHO[n]))


def gradient(n, N, s, p, L=_len):
    return kt(n) / (1 + np.sqrt(L(p, N) * M.chi(n)))


def elhaddad(n, N, s, p, L=_len):
    """Atzori-Lazzarin-Meneghetti: K_f = min(K_t, crack branch); crack branch = El Haddad with the finite-width
    DENT K of a crack as deep as the notch, in net stress: sqrt(1 + g^2 / (pi L)). Meets the blunt (K_t), crack
    (LEFM) and ligament (L >> b -> 1) limits."""
    return np.minimum(kt(n), np.sqrt(1 + dent_g() ** 2 / (math.pi * L(p, N))))


def constraint(base):
    """Static group: K_f = K_f,base * NSR^(-sigma_n,max / UTS); NSR = notch-strengthening ratio (last constant)."""
    return lambda n, N, s, p, L=_len: base(n, N, s, p[:-1], L) * p[-1] ** (-np.asarray(s) / paper.UTS)


# ---------------------------------------------------------------- stress triaxiality on the bisector (stored tensor)
NU = 0.35


def eta_profile(notch):
    """eta = sigma_h / sigma_vM along the bisector (thickness mean of each measure), from the stored IgBEM full-tensor
    measures: tr(sigma) = s1 + (s1 - E eps1) / nu (isotropic Hooke, principal axes shared). Uniaxial = 1/3."""
    P = M.measure_profile
    d = P(notch, "s1").d
    s1, vm, e1 = P(notch, "s1").s, P(notch, "vm").s, P(notch, "eps1").s
    return d, (s1 + (s1 - e1) / NU) / 3 / vm


def eta_at(notch, x):
    d, e = eta_profile(notch)
    return np.interp(np.clip(x, 0, d[-1]), d, e)


def triax(base_len):
    """K_f = base * (3 eta(L/2))^beta, beta = ln(last constant): eta at the point-method distance L/2."""
    def f(n, N, s, p, L=_len):
        Lp = base_len(p, N)
        return (p[-1] ** np.log(3 * eta_at(n, Lp / 2)))
    return f


CLOSURES = {   # name: (f, p0, length map, label, kind) ; kind: 'free' fit, 'pred' no fit
    "net": (lambda n, N, s, p, L=None: np.ones(np.shape(N)), (), None, "C0 net stress, K_f = 1", "pred"),
    "kt": (lambda n, N, s, p, L=None: np.full(np.shape(N), kt(n)), (), None, "C1 K_f = K_t (elastic)", "pred"),
    "peterson": (peterson, (5.0, -0.25), _len, "C2 Peterson, a = {:.3g} N^{:.3f} mm", "free"),
    "neuber": (neuber, (5.0, -0.25), _len, "C3 Neuber, rho' = {:.3g} N^{:.3f} mm", "free"),
    "gradient": (gradient, (5.0, -0.25), _len, "C4 Siebel-Stieler, s = {:.3g} N^{:.3f} mm", "free"),
    "elhaddad": (elhaddad, (16.4, -0.242), _len, "C5 El Haddad + ligament, L = {:.3g} N^{:.3f} mm", "free"),
    "elhaddad_paper": (elhaddad, (1.0,), _lpaper, "C5 El Haddad + ligament, paper's L(N), no fit", "pred"),
    "neuber_nsr": (constraint(neuber), (5.0, -0.25, 1.5), _len,
                   "C6 Neuber x constraint, rho' = {:.3g} N^{:.3f} mm, NSR {:.3g}", "free"),
    "net_nsr": (lambda n, N, s, p, L=None: p[0] * p[1] ** (-np.asarray(s) / paper.UTS) * np.ones(np.shape(N)),
                (1.0, 1.5), None, "C6'' net x constraint only, c {:.3g}, NSR {:.3g}", "free2"),
    "net_triax": (lambda n, N, s, p, L=None: p[0] * p[1] ** np.log(3 * eta_at(n, paper.lm_paper(N) / 2)),
                  (1.0, 2.0), None, "T1 net x c (3 eta)^beta at paper L_M/2, c {:.3g}, exp(beta) {:.3g}", "free2"),
    "neuber_triax": (lambda n, N, s, p, L=_len: neuber(n, N, s, p[:2], L) * p[2] ** np.log(3 * eta_at(n, L(p[:2], N) / 2)),
                     (5.0, -0.25, 2.0), _len, "T2 Neuber x (3 eta(rho'/2))^beta, rho' = {:.3g} N^{:.3f} mm, exp(beta) {:.3g}", "free"),
    "neuber_nsr_triax": (lambda n, N, s, p, L=_len: neuber(n, N, s, p[:2], L) * p[2] ** (-np.asarray(s) / paper.UTS)
                         * p[3] ** np.log(3 * eta_at(n, paper.lm_paper(N) / 2)),
                         (5.0, -0.25, 1.5, 2.0), _len,
                         "T3 Neuber x constraint x triaxiality, rho' = {:.3g} N^{:.3f} mm, NSR {:.3g}, exp(beta) {:.3g}", "free"),
    "elhaddad_nsr": (constraint(elhaddad), (16.4, -0.242, 1.5), _len,
                     "C6' El Haddad x constraint, L = {:.3g} N^{:.3f} mm, NSR {:.3g}", "free"),
}


def keq(name, n, N, s, p):
    f, _, L, _, _ = CLOSURES[name]
    return f(n, N, s, p, L) if L is not None else f(n, N, s, p)


def calibrate(name):
    """Least squares in log stress: K_f(sharp, N, sigma_sharp(N)) = plain(N) / sharp(N) on N in [1e3, 2e6]."""
    f, p0, L, _, kind = CLOSURES[name]
    N = M.N_CAL
    s = paper.SHARP_FIG8A(N)
    target = np.log10(paper.PLAIN(N) / s)
    if kind == "pred":
        p = np.array(p0, float)
        return p, float(np.sqrt(np.mean((np.log10(keq(name, "sharp", N, s, p)) - target) ** 2)))

    def unpack(x):
        if kind == "free2":
            return np.exp(x)
        return np.array([math.exp(x[0]), x[1], *np.exp(x[2:])])

    def cost(x):
        k = keq(name, "sharp", N, s, unpack(x))
        return np.sum((np.log10(np.maximum(k, 1e-9)) - target) ** 2)

    x0 = np.log(p0) if kind == "free2" else np.array([math.log(p0[0]), p0[1], *np.log(p0[2:])])
    best = None
    for dA in (-2, 0, 2):
        for dB in (-0.15, 0, 0.15):
            o = minimize(cost, x0 + np.r_[dA, dB, np.zeros(len(p0) - 2)], method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
            best = o if best is None or o.fun < best.fun else best
    return unpack(best.x), float(math.sqrt(best.fun / len(N)))


def residuals(name, p, specs):
    out = {}
    for k in paper.NOTCHES:
        S = np.array([(s.N, s.s_max) for s in specs if s.notch == k and not s.runout])
        out[k] = (S[:, 0], np.log10(keq(name, k, S[:, 0], S[:, 1], p) * S[:, 1] / paper.PLAIN(S[:, 0])))
    return out


def held(R):
    N = np.concatenate([R[k][0] for k in ("intermediate", "blunt")])
    r = np.concatenate([R[k][1] for k in ("intermediate", "blunt")])
    return M.stats(N, r)


def run():
    specs = paper.fig8_specimens()
    out = {}
    for name, (_, _, _, lab, kind) in CLOSURES.items():
        p, cal = calibrate(name)
        R = residuals(name, p, specs)
        lab = lab.format(*p) if kind != "pred" else lab
        out[name] = dict(p=p, cal=cal, R=R, S={k: M.stats(*R[k]) for k in paper.NOTCHES}, H=held(R), label=lab,
                         byR=by_R(name, p, specs))
    return out


def by_group(specs, key):
    """Residual split by R or raster angle p (net stress), all notches."""
    rows = []
    for v in sorted({getattr(s, key) for s in specs}):
        r = [math.log10(s.s_max / paper.PLAIN(s.N)) for s in specs if getattr(s, key) == v and not s.runout]
        rows.append((v, len(r), float(np.mean(r))))
    return rows


def by_R(name, p, specs):
    """Held-out (U r1 + U r3) mean residual at R = -1 and R = 0: what sigma_max (the paper's variable) leaves in."""
    out = []
    for R in (-1.0, 0.0):
        sub = [s for s in specs if s.R == R]
        r = residuals(name, p, sub)
        out.append(float(np.mean(np.concatenate([r[k][1] for k in ("intermediate", "blunt")]))))
    return out


def table(out):
    f = lambda s: f"{s['bias']:+.3f} / {s['rms']:.3f} / {s['slope']:+.3f} / {100*s['band']:.0f}%"
    L = ["| closure | fit rms (curve) | fitted sharp V: bias / rms / slope / in band | held-out U r1 | held-out U r3 | "
         "held-out both: rms / in band | held-out bias R -1 / R 0 |", "|---|---|---|---|---|---|---|"]
    for name in sorted(out, key=lambda n: out[n]["H"]["rms"]):
        o = out[name]
        L.append(f"| {o['label']} | {o['cal']:.3f} | {f(o['S']['sharp'])} | {f(o['S']['intermediate'])} | "
                 f"{f(o['S']['blunt'])} | {o['H']['rms']:.3f} / {100*o['H']['band']:.0f}% | "
                 f"{o['byR'][0]:+.3f} / {o['byR'][1]:+.3f} |")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ figures
MK = {"sharp": ("^", "k"), "intermediate": ("o", "#1f3f8f"), "blunt": ("D", "#c0392b")}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "serif", "mathtext.fontset": "cm"})
    return plt


def _binned(x, y, nb=6):
    e = np.quantile(np.log10(x), np.linspace(0, 1, nb + 1))
    out = []
    for a, b in zip(e[:-1], e[1:]):
        m = (np.log10(x) >= a) & (np.log10(x) <= b)
        if m.sum() >= 4:
            out.append((10 ** np.median(np.log10(x[m])), *np.quantile(y[m], [0.25, 0.5, 0.75])))
    return np.array(out)


def v90(notch, frac=0.9):
    """Kuguel highly stressed volume (mm^3): sigma_1 >= frac * K_t, whole specimen, stored IgBEM volume histogram."""
    v = M.fields()[notch]["volume"]
    w, s1 = np.asarray(v["w"]), np.asarray(v["s1"])
    return float(w[s1 >= frac * kt(notch)].sum())


def kf_points():
    """Per-specimen K_f,i = plain P50(N_i) / sigma_n,max,i and q_i, failures only (Fig. 8 stresses)."""
    out = {}
    for k in paper.NOTCHES:
        S = np.array([(s.N, s.s_max) for s in paper.fig8_specimens() if s.notch == k and not s.runout])
        N, s = S[:, 0], S[:, 1]
        Kf = paper.PLAIN(N) / s
        b, a = np.polyfit(np.log10(N), np.log10(s), 1)          # the notch's own median S-N line (log S on log N)
        out[k] = dict(N=N, s=s, s50=10 ** (a + b * np.log10(N)), Kf=Kf, q=(Kf - 1) / (kt(k) - 1), L=paper.lm_paper(N))
    return out


GROUPS = {   # name: (x per notch from the specimen dict, axis label)
    "rho_L": (lambda k, d: RHO[k] / d["L"], r"notch number $\rho/L(N_f)$"),
    "chi_L": (lambda k, d: M.chi(k) * d["L"], r"gradient group $\chi^* L(N_f)$ (Siebel-Stieler / FKM)"),
    "static": (lambda k, d: d["s50"] / paper.UTS,
               r"static group $\sigma_{n,max,50}(N_f)/\sigma_{UTS}$ (notch's own median S-N line at $N_f$)"),
    "v90_L3": (lambda k, d: v90(k) / d["L"] ** 3, r"volume group $V_{90}/L(N_f)^3$ (Kuguel $V_{90}$, IgBEM)"),
}


def pooled_rms(x, y, deg=2):
    """rms of y about ONE quadratic in log10 x fitted to all notches pooled."""
    X = np.concatenate([np.log10(x[k]) for k in x])
    Y = np.concatenate([y[k] for k in x])
    return float(np.sqrt(np.mean((Y - np.polyval(np.polyfit(X, Y, deg), X)) ** 2)))


def collapse_rows(yname="logKf"):
    """Pooled rms of y = log10 K_f (or q) about one curve in each group, against two baselines: one curve in N for all
    notches (Lambda = b/L(N) is the same for every notch, so it is this row) and a separate curve in N per notch
    (floor: the best any one-variable group can do with three notches)."""
    P = kf_points()
    y = {k: (np.log10(P[k]["Kf"]) if yname == "logKf" else P[k]["q"]) for k in P}
    rows = [("ligament b/L(N) (= one curve in N, no notch group)", pooled_rms({k: B_HALF / P[k]["L"] for k in P}, y))]
    rows += [(g, pooled_rms({k: fx(k, P[k]) for k in P}, y)) for g, (fx, _) in GROUPS.items()]
    floor = np.sqrt(np.mean(np.concatenate([y[k] - np.polyval(np.polyfit(np.log10(P[k]["N"]), y[k], 2),
                                                               np.log10(P[k]["N"])) for k in P]) ** 2))
    rows.append(("floor: separate curve in N per notch", float(floor)))
    return rows


def collapse_table():
    a, b = collapse_rows("logKf"), collapse_rows("q")
    L = ["| coordinate x (one quadratic in log x, all notches) | rms of log10 K_f | rms of q |", "|---|---|---|"]
    L += [f"| {n} | {u:.3f} | {v:.3f} |" for (n, u), (_, v) in zip(a, b)]
    return "\n".join(L)


def plot_groups(out, path):
    """Model-free collapse test: per-specimen notch sensitivity q in four candidate coordinates."""
    plt = _style()
    P = kf_points()
    rq = collapse_rows("q")
    R0q, Fq = rq[0][1], rq[-1][1]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.4), layout="constrained")
    for ax, (g, (fx, lab)) in zip(axs.ravel(), GROUPS.items()):
        xs = {k: fx(k, P[k]) for k in P}
        for k in paper.NOTCHES:
            m, c = MK[k]
            x, q = xs[k], P[k]["q"]
            tag = f"{paper.LABEL[k]}, K_t {kt(k):.2f}" + (" (calibration set)" if k == "sharp" else " (held out)")
            ax.scatter(x, q, marker=m, s=20, facecolor="none", edgecolor=c, lw=0.8, alpha=0.75, label=tag)
            bq = _binned(x, q)
            ax.errorbar(bq[:, 0], bq[:, 2], yerr=[bq[:, 2] - bq[:, 1], bq[:, 3] - bq[:, 2]], color=c, marker=m,
                        ms=7, lw=1.6, capsize=3, mfc=c)
        X = np.concatenate([np.log10(xs[k]) for k in P])
        Y = np.concatenate([P[k]["q"] for k in P])
        if True:
            xx = np.linspace(X.min(), X.max(), 100)
            ax.plot(10 ** xx, np.polyval(np.polyfit(X, Y, 2), xx), color="0.35", lw=1.2, ls="-.",
                    label="one pooled curve (quadratic in log x)")
        a = pooled_rms(xs, {k: P[k]["q"] for k in P})
        ax.text(0.98, 0.03, f"rms of q about the pooled curve {a:.3f}\n(one curve in N: {R0q:.3f}; per-notch floor: {Fq:.3f})",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, bbox=dict(fc="white", ec="0.6", lw=0.6))
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(1, color="0.5", lw=0.8, ls=":")
        ax.set(xscale="log", xlabel=lab, ylabel=r"$q = (K_f - 1)/(K_t - 1)$", ylim=(-1.3, 1.1))
        ax.grid(alpha=0.3, which="both")
    axs[0, 0].plot(*(lambda x: (x, 1 / (1 + 1 / x)))(np.geomspace(0.05, 6, 100)), "k--", lw=1, label="Peterson, a = L")
    axs[0, 0].legend(fontsize=7.8, loc="upper left")
    fig.suptitle("Model-free collapse test, Ezeh & Susmel 2020 notched PLA, 149 failures (Fig. 8 $\\sigma_{n,max}$, R -1 and 0 "
                 "pooled). Open: specimens; filled with bars: median and IQR per life bin.\n$K_{f,i}$ = plain P50($N_i$) / "
                 "$\\sigma_{n,max,i}$ (6.7 MPa at 2e6, k 5.1, capped at UTS 42.9); $L(N)$ = paper Eq. 6; $K_t$, $\\chi^*$, "
                 "$V_{90}$ from IgBEM 3D. q < 0 means stronger than plain at net stress.", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=190)


def plot_closures(out, path, show):
    """Closures in the paper's own axes (sigma_eq,max / UTS vs N) against the plain band."""
    plt = _style()
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6), sharex=True, sharey=True, layout="constrained")
    N = np.geomspace(10, 2e6, 300)
    for ax, name in zip(axs.ravel(), show):
        o = out[name]
        ax.plot(N, paper.PLAIN(N) / paper.UTS, "k-", lw=1.3)
        for ps in (0.1, 0.9):
            ax.plot(N, paper.PLAIN(N, ps) / paper.UTS, "k--", lw=0.8)
        for k in paper.NOTCHES:
            n, r = o["R"][k]
            m, c = MK[k]
            ax.scatter(n, 10 ** r * paper.PLAIN(n) / paper.UTS, marker=m, s=24, facecolor="k" if k == "sharp" else "none",
                       edgecolor=c, lw=0.9, zorder=3)
        s, h = o["S"]["sharp"], o["H"]
        ax.text(0.02, 0.02, f"{o['label']}\nfitted (sharp V): bias {s['bias']:+.3f}, rms {s['rms']:.3f}, slope {s['slope']:+.3f}\n"
                f"held-out U r1 bias {o['S']['intermediate']['bias']:+.3f}, U r3 bias {o['S']['blunt']['bias']:+.3f}\n"
                f"held-out both (n {h['n']}): rms {h['rms']:.3f}, slope {h['slope']:+.3f}, in band {100*h['band']:.0f}%",
                transform=ax.transAxes, fontsize=8.3, va="bottom", bbox=dict(fc="white", ec="0.6", lw=0.6))
        ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.03, 3))
        ax.grid(alpha=0.3, which="both")
    for ax in axs[1]:
        ax.set_xlabel("$N_f$ [cycles to failure]")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$K_f\,\sigma_{n,max}/\sigma_{UTS}$")
    h = [Line2D([], [], marker=MK[k][0], ls="", mfc="k" if k == "sharp" else "none", mec=MK[k][1], ms=6)
         for k in paper.NOTCHES] + [Line2D([], [], color="k", lw=1.3), Line2D([], [], color="k", ls="--", lw=0.8)]
    fig.legend(h, ["Specimens, sharp V (calibration set)", "Specimens, U r1 (held out)", "Specimens, U r3 (held out)",
                   "Plain P50 6.7 MPa at 2e6, k 5.1", "Plain P10 / P90, T 2.451"], loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Dimensionless closures in the paper's variable (Fig. 8 $\\sigma_{n,max}$, UTS 42.9, R pooled, failures only). "
                 "Constants fitted on plain P50 + Fig. 8a sharp curve; U notches predicted.\nNot in the paper: all four "
                 "closures (K_f laws); C5 uses Tada's DENT K for a crack as deep as the notch (g = "
                 f"{dent_g():.2f} sqrt(mm))", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=190)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = run()
    t = table(out)
    specs = paper.fig8_specimens()
    extra = ["", "Net-stress residual by R and raster angle (all notches, failures): value / n / mean log10(sigma_n,max/P50)"]
    for key in ("R", "p"):
        extra.append(f"- {key}: " + ", ".join(f"{v}: n {n}, {m:+.3f}" for v, n, m in by_group(specs, key)))
    extra.append(f"- K_t (net, IgBEM): " + ", ".join(f"{k} {kt(k):.2f}" for k in paper.NOTCHES)
                 + f"; DENT g = K/sigma_net = {dent_g():.3f} sqrt(mm)")
    ct = collapse_table()
    extra += ["", "Model-free collapse (per-specimen K_f, all three notches):", ct]
    print(t + "\n".join(extra))
    open(os.path.join(d, "collapse.md"), "w").write(__doc__ + "\n" + t + "\n" + "\n".join(extra) + "\n")
    plot_groups(out, os.path.join(d, "fig_collapse_groups"))
    plot_closures(out, os.path.join(d, "fig_collapse_closures"), ("neuber", "gradient", "elhaddad", "neuber_nsr"))
