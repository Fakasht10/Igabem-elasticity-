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
import json
import math
from functools import lru_cache
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


# ---------------------------------------------------------------- SED split on the Lazzarin-Berto crescent
@lru_cache(None)
def _sed_grid(notch):
    """Stored crescent grid (es2020/sed_split.py on the comparison branch, post-process of the cached solve): per point
    thickness-mean (tr sigma)^2 and sigma_vM^2 at unit net stress; reproduces the stored w2E to 5e-16."""
    g = json.load(open(os.path.join(paper.DATA, "igabem_sed_split.json")))
    g = g.get("notches", g)[notch]
    return g["r0"], np.asarray(g["rc"]), np.asarray(g["dA"]), np.asarray(g["tr2"]), np.asarray(g["vm2"])


def sed_parts(notch, R0):
    """Crescent averages of 2E W_dev = 2(1+nu)/3 vM^2 and 2E W_vol = (1-2nu)/3 tr^2 over rc <= R0 + r0."""
    r0, rc, dA, tr2, vm2 = _sed_grid(notch)
    out = []
    for R in np.atleast_1d(R0):
        m = rc <= R + r0
        w = dA[m] / dA[m].sum()
        out.append((2 * (1 + NU) / 3 * (w @ vm2[m]), (1 - 2 * NU) / 3 * (w @ tr2[m])))
    return np.array(out)


def sed_split(notch, R0, alpha):
    """K_eq = sqrt((W_dev + alpha W_vol) / same for uniaxial unit stress); alpha = 1 is Lazzarin-Berto."""
    wd, wv = sed_parts(notch, R0).T
    return np.sqrt((wd + alpha * wv) / (2 * (1 + NU) / 3 + alpha * (1 - 2 * NU) / 3))


def eta_line(notch, L):
    """eta averaged over [0, 2L] on the bisector (mirrored past the centre line, as the line method)."""
    d, e = eta_profile(notch)
    lig = d[-1]
    out = []
    for l2 in np.atleast_1d(np.minimum(2 * np.asarray(L, float), 2 * lig)):
        x = np.linspace(0, l2, 400)
        out.append(np.mean(np.interp(np.where(x <= lig, x, 2 * lig - x), d, e)))
    return np.array(out)


def eta_weibull(notch, R, m=M.M_STRENGTH):
    """eta weighted by sigma_vM^m over the stored crescent of radius R(N) at the root: eta = |tr|/(3 vM) (tension at
    the root, so tr > 0 there; sign is lost in the stored tr^2). DEPARTURE: crescent area, not the whole volume,
    because the whole-volume histogram holds sigma_1 only; vM^m stands in for sigma_1^m as the weight."""
    r0, rc, dA, tr2, vm2 = _sed_grid(notch)
    e = np.sqrt(tr2) / 3 / np.sqrt(vm2)
    w = dA * vm2 ** (m / 2)
    o = np.argsort(rc)
    cw, cwe = np.cumsum(w[o]), np.cumsum((w * e)[o])
    i = np.searchsorted(rc[o], np.atleast_1d(R) + r0, side="right") - 1
    return cwe[i] / cw[i]


def _with_eta(base, eta):
    """K = base(L) * (3 eta(L))^beta, L = A N^B shared, beta = ln(p[2])."""
    return lambda n, N, s, p, L=_len: base(n, L(p[:2], N)) * p[2] ** np.log(3 * eta(n, L(p[:2], N)))


def weibull_mN(n, N, s, p, L=None):
    """Weibull whole volume with life-dependent modulus m = A N^B (one extra parameter), V0 = p[2]."""
    m = np.atleast_1d(p[0] * np.asarray(N, float) ** p[1])
    v = M.fields()[n]["volume"]
    w, s1 = np.asarray(v["w"]), np.asarray(v["s1"])
    S = (w[None, :] * s1[None, :] ** m[:, None]).sum(1)
    return (S / p[2]) ** (1 / m)


# ---------------------------------------------------------------- local strain approach (Neuber + cyclic Ramberg-Osgood)
E_PLA = 3479.0          # MPa, paper Sec. 2 (monotonic E; E' = E assumed)


def ro_strain(sig, Kc, nc):
    return sig / E_PLA + (np.maximum(sig, 0) / Kc) ** (1 / nc)


def neuber_local(sig_el, Kc, nc):
    """Local stress solving sigma * eps(sigma) = sigma_el^2 / E (Neuber), by bisection on [0, sigma_el]."""
    sig_el = np.asarray(sig_el, float)
    lo, hi = np.zeros_like(sig_el), sig_el.copy()
    for _ in range(60):
        mid = (lo + hi) / 2
        big = mid * ro_strain(mid, Kc, nc) > sig_el ** 2 / E_PLA
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return (lo + hi) / 2


def plain_equiv(eps, Kc, nc):
    """Nominal plain stress with the same strain (uniform gauge): inverse Ramberg-Osgood by bisection."""
    eps = np.asarray(eps, float)
    lo, hi = np.zeros_like(eps), eps * E_PLA
    for _ in range(60):
        mid = (lo + hi) / 2
        big = ro_strain(mid, Kc, nc) > eps
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return (lo + hi) / 2


def local_strain(kel):
    """K_eq = S_eq / sigma_n,max: Neuber at the root on the elastic field value kel(n, N) * sigma_n,max (sigma_max, the
    paper's variable), then the plain stress giving the same local strain. Damage parameter = local strain, so the
    plain band is used unchanged. K' and n' fitted on plain + sharp V (no PLA cyclic curve found: DEPARTURE)."""
    def f(n, N, s, p, L=None):
        sig_el = kel(n, N) * np.asarray(s, float)
        loc = neuber_local(sig_el, p[0], p[1])
        return plain_equiv(ro_strain(loc, p[0], p[1]), p[0], p[1]) / np.asarray(s, float)
    return f


K_LIFE_FIX = 1.111      # life-Weibull shape of the 2019 PLA series pooled (fatigue-data/hsv_fix.md): the mHSV fix


def mhsv_fixed(n, N, s, p, L=None):
    """Modified HSV (Zhang et al. 2023 Eq. 10) with the corrected life-scatter shape k_N 1.111 (was 0.675 from the
    band, which over-penalised the U notches); L_CD = point L_M at 2e6 from the point calibration, C_S fitted."""
    Lcd = M.l_cd()
    return np.full(np.shape(N), M.k_point(n, Lcd) * (M.s90(n) * Lcd / p[0]) ** (1 / (K_LIFE_FIX * paper.PLAIN.k)))


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
    "sed": (lambda n, N, s, p, L=_len: sed_split(n, L(p, N), 1.0), (2.0, -0.2), _len,
            "S1 averaged SED (alpha = 1, Lazzarin-Berto), R0 = {:.3g} N^{:.3f} mm", "free"),
    "sed_const": (lambda n, N, s, p, L=None: sed_split(n, np.full(np.shape(N), p[0]), 1.0), (1.0,), None,
                  "S1c averaged SED, constant R0 = {:.3g} mm", "free2"),
    "sed_split": (lambda n, N, s, p, L=_len: sed_split(n, L(p[:2], N), p[2]), (2.0, -0.2, 1.0), _len,
                  "S2 W_dev + alpha W_vol, R0 = {:.3g} N^{:.3f} mm, alpha {:.3g}", "free"),
    "sed_split_const": (lambda n, N, s, p, L=None: sed_split(n, np.full(np.shape(N), p[0]), p[1]), (1.0, 1.0), None,
                        "S2c W_dev + alpha W_vol, constant R0 = {:.3g} mm, alpha {:.3g}", "free2"),
    "point_eta": (_with_eta(M.k_point, eta_line), (16.4, -0.242, 2.0), _len,
                  "F1 point x (3 eta_line[0,2L])^beta, L = {:.3g} N^{:.3f} mm, exp(beta) {:.3g}", "free"),
    "line_eta": (_with_eta(M.k_line, eta_line), (5.0, -0.2, 2.0), _len,
                 "F2 line x (3 eta_line[0,2L])^beta, L = {:.3g} N^{:.3f} mm, exp(beta) {:.3g}", "free"),
    "point_etaw": (_with_eta(M.k_point, eta_weibull), (16.4, -0.242, 2.0), _len,
                   "F3 point x (3 eta_Weibull(R=L))^beta, L = {:.3g} N^{:.3f} mm, exp(beta) {:.3g}", "free"),
    "weibull_mN": (weibull_mN, (3.44, 0.0, 100.0), _len,
                   "F4 Weibull, m = {:.3g} N^{:.3f} (one extra parameter), V0 {:.3g} mm^3", "free"),
    "lsa_peak": (local_strain(lambda n, N: np.full(np.shape(N), kt(n))), (60.0, 0.1), None,
                 "L1 local strain, Neuber on the field peak, K' {:.3g} MPa, n' {:.3g} (fitted)", "free2"),
    "lsa_point": (local_strain(lambda n, N: M.k_point(n, paper.lm_paper(N) / 2)), (60.0, 0.1), None,
                  "L2 local strain, Neuber on sigma_1 at paper L_M(N)/2, K' {:.3g} MPa, n' {:.3g} (fitted)", "free2"),
    "grad_paperL": (lambda n, N, s, p, L=None: kt(n) / (1 + np.sqrt(p[0] * M.chi(n) * paper.lm_paper(N))), (1.0,), None,
                    "G1 peak sigma_1 / (1 + sqrt(c chi L_M(N))), paper L_M, c {:.3g} (one material constant)", "free2"),
    "mhsv_fix": (mhsv_fixed, (50.0,), None, "mHSV corrected (k_N 1.111, 2019 series), C_S {:.3g} mm^3", "free2"),
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
            o = minimize(cost, x0 + (np.full(len(p0), dA / 2) if kind == "free2" else np.r_[dA, dB, np.zeros(len(p0) - 2)]), method="Nelder-Mead",
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


# ------------------------------------------------------------------------------------------------ field-only ranking
FIELD_M = {"point": "TCD point (sigma_1 at L/2 on the steepest-descent path = bisector here)",
           "line": "TCD line (mean sigma_1 over 2L on the steepest-descent path)", "area": "TCD area (point L_M)",
           "gradient": "peak sigma_1 / (1 + sqrt(c chi L)) (FKM-type, peak and chi from field)",
           "sed": "averaged SED, R0(N)", "weibull": "Weibull, whole volume, m 3.44 fixed", "mhsv": "modified HSV"}
FIELD_C = ("grad_paperL", "mhsv_fix", "lsa_peak", "lsa_point", "sed_const", "sed_split", "sed_split_const", "point_eta", "line_eta", "point_etaw", "weibull_mN")


def field_rank(out):
    """Measures defined on any 3D field (no nominal stress, no K_t, no rho). Same protocol and scores."""
    specs = paper.fig8_specimens()
    rows = []
    for name, lab in FIELD_M.items():
        p, cal = M.calibrate(name)
        R = M.residuals(name, p, specs)
        rows.append((lab + ": " + M.METHODS[name][2].format(*p), cal, {k: M.stats(*R[k]) for k in paper.NOTCHES}, held(R)))
    for name in FIELD_C:
        o = out[name]
        rows.append((o["label"], o["cal"], o["S"], o["H"]))
    o = out["net"]
    rows.append(("REFERENCE ONLY (coupon, not defined on a general part): net stress", o["cal"], o["S"], o["H"]))
    f = lambda s: f"{s['bias']:+.3f} / {s['rms']:.3f} / {s['slope']:+.3f}"
    L = ["| field measure | fit rms (curve) | fitted sharp V: bias / rms / slope | U r1 bias / rms / slope | "
         "U r3 bias / rms / slope | held-out both: rms / in band / slope |", "|---|---|---|---|---|---|"]
    for lab, cal, S, H in sorted(rows, key=lambda r: r[3]["rms"]):
        L.append(f"| {lab} | {cal:.3f} | {f(S['sharp'])} | {f(S['intermediate'])} | {f(S['blunt'])} | "
                 f"{H['rms']:.3f} / {100*H['band']:.0f}% / {H['slope']:+.3f} |")
    return "\n".join(L), rows


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
    "wvol_wdev": (lambda k, d: (lambda P: P[:, 1] / P[:, 0])(sed_parts(k, d["L"])),
                  r"SED ratio $W_{vol}/W_{dev}$ on the crescent of radius $L(N)$"),
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



def plot_ranking(rows, path):
    """Held-out rms (bars) and in-band share for the field-only measures; net stress drawn as a reference line."""
    plt = _style()
    rows = sorted(rows, key=lambda r: -r[3]["rms"])
    ref = [r for r in rows if r[0].startswith("REFERENCE")][0][3]
    rows = [r for r in rows if not r[0].startswith("REFERENCE")]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6), layout="constrained")
    y = np.arange(len(rows))
    ax.barh(y, [r[3]["rms"] for r in rows], color="#4a6fa5")
    for i, r in enumerate(rows):
        ax.text(r[3]["rms"] + 0.004, i, f"{r[3]['rms']:.3f}  ({100*r[3]['band']:.0f}% in band, U r3 bias "
                f"{r[2]['blunt']['bias']:+.2f})", va="center", fontsize=8)
    ax.axvline(ref["rms"], color="k", ls="--", lw=1)
    ax.text(ref["rms"], len(rows) - 0.4, " net stress (coupon reference only) 0.141", fontsize=8, va="bottom")
    short = lambda t: (t.replace("peak sigma_1 / (1 + sqrt(c chi L)) (FKM-type, peak and chi from field)", "FKM support, c(N) (2 const)")
                        .replace("G1 peak sigma_1 / (1 + sqrt(c chi L_M(N)))", "FKM support, paper L(N) (1 const)")
                        .replace("L2 local strain", "Neuber at L/2 (n' 1.42, unphysical)")
                        .replace("L1 local strain", "Neuber at root peak")
                        .replace("F4 Weibull", "Weibull, m(N) (1 extra const)")
                        .replace("F1 point x (3 eta_line[0", "point x triaxiality (line)")
                        .replace("F2 line x (3 eta_line[0", "line x triaxiality (line)")
                        .replace("F3 point x (3 eta_Weibull(R=L))^beta", "point x triaxiality (Weibull-wtd)")
                        .replace("S2c W_dev + alpha W_vol", "SED split, const R0")
                        .replace("S1c averaged SED", "averaged SED, const R0")
                        .replace("S2 W_dev + alpha W_vol", "SED split, R0(N)")
                        .replace("averaged SED", "averaged SED")
                        .replace("TCD line (mean sigma_1 over 2L on the steepest-descent path)", "TCD line")
                        .replace("TCD point (sigma_1 at L/2 on the steepest-descent path = bisector here)", "TCD point")
                        .replace("mHSV corrected (k_N 1.111", "mHSV, corrected k_N")
                        .replace("modified HSV", "mHSV, original k_N"))
    ax.set_yticks(y, [short(r[0].split(":")[0].split(",")[0])[:40] for r in rows], fontsize=8.5)
    ax.set(xlabel="held-out rms, log10 stress (U r1 + U r3, 102 failures)", xlim=(0, 0.48))
    ax.set_title("Field-only measures, fitted on plain P50 + sharp V (Fig. 8 sigma_n,max, UTS 42.9, R pooled)", fontsize=10)
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
    ft, frows = field_rank(out)
    plot_ranking(frows, os.path.join(d, "fig_field_ranking"))
    extra += ["", "Field-only ranking (any 3D part; net stress kept only as a coupon reference line):", ft]
    ct = collapse_table()
    extra += ["", "Model-free collapse (per-specimen K_f, all three notches):", ct]
    print(t + "\n".join(extra))
    open(os.path.join(d, "collapse.md"), "w").write(__doc__ + "\n" + t + "\n" + "\n".join(extra) + "\n")
    plot_groups(out, os.path.join(d, "fig_collapse_groups"))
    plot_closures(out, os.path.join(d, "fig_collapse_closures"), ("neuber", "gradient", "elhaddad", "neuber_nsr"))
