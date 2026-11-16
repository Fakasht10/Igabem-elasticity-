"""Notch criteria on the IgBEM fields, all as an equivalent stress factor K_eq = sigma_eq / sigma_n,max, so every
method is scored in the paper's variable sigma_eq / UTS against the plain P50 curve.

Each method has two constants, calibrated the paper's way (Fig. 9e): K_eq(sharp, N) * SHARP_FIG8A(N) = PLAIN(N) on
N in [1e3, 2e6] (least squares in log stress). U r1 and U r3 are never used in a fit.

    point   Eq. 2   sigma_1 at L/2 on the bisector,                  L = A N^B
    line    Eq. 3   mean of sigma_1 on [0, 2L] (mirrored past the centre line; NOT used in the paper), L = A N^B
    area    Eq. 4   mean of sigma_1 over the semicircle of radius L at the root. As in the paper (Sec. 6) it is NOT
            refitted: it uses the point-method L_M(N). L exceeds half the net width (3 mm) below ~1e3 cycles; there the
            semicircle crosses the centre line (field mirrored) and is clipped at 5.9 mm (one specimen).
    netfit  sigma_n,max times one constant fitted on the sharp curve: the fair no-method baseline
    weibull         (sum sigma_1^m dV / V0)^(1/m), weakest link over the whole specimen. K_eq does not depend on N,
            so one notch fixes one constant: V0 is fitted, m = 3.44 is the strength Weibull modulus of the plain
            band (T_sigma 2.451 = P10/P90 ratio: m = ln(ln 0.1 / ln 0.9) / ln T_sigma).
    mhsv    Zhang et al. 2023 Eq. 10: N_N = N_S(sigma_PM(L_CD/2)) (S90 L_CD / C_S)^(-1/k_N), C_S = S_S L_FCI fitted.
            Also N-independent, so L_CD is not fitted here: it is our point-method L_M at N_ref (the TCD critical
            distance of Zhang's Eq. 4, which needs a Delta K_th that PLA lacks). k_N = 0.675 is the life-Weibull
            shape of the plain band (T_sigma^k = 96.7). In stress: K_eq = K(L_CD/2) (S90 L_CD / C_S)^(1/(k_N k_plain)).
"""
import json
import math
import os
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from . import paper, point

M_STRENGTH = math.log(math.log(0.1) / math.log(0.9)) / math.log(paper.PLAIN.T)                # 3.44
K_LIFE = math.log(math.log(0.1) / math.log(0.9)) / math.log(paper.PLAIN.T ** paper.PLAIN.k)   # 0.675
N_CAL = np.geomspace(1e3, 2e6, 60)
BAND = math.log10(math.sqrt(paper.PLAIN.T))               # half-width of the P10-P90 band in log stress


@lru_cache(None)
def fields():
    return json.load(open(os.path.join(paper.DATA, "igabem_fields.json")))["notches"]


@lru_cache(None)
def _prof(notch):
    return point.profiles()[notch]


def _trap(x):
    d = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def k_point(notch, L):
    return _prof(notch)(np.asarray(L) / 2)


def k_line(notch, L):
    p = _prof(notch)
    lig = p.d[-1]
    out = []
    for l2 in np.atleast_1d(np.minimum(2 * np.asarray(L, float), 2 * lig)):   # past 2 lig: the opposite notch
        x = np.linspace(0, l2, 400)
        xm = np.where(x <= lig, x, 2 * lig - np.minimum(x, 2 * lig))    # symmetric about the centre line
        out.append(np.mean(p(xm)))
    return np.array(out)


def k_area(notch, L):
    a = fields()[notch]["area"]
    r, wth, s = np.asarray(a["r"]), np.asarray(a["wth"]), np.asarray(a["s1"])
    q = s @ wth                                               # integral over theta of sigma_1, per r
    out = []
    for l in np.atleast_1d(np.minimum(L, r[-1])):
        rr = np.concatenate([r[r < l], [l]])
        qq = np.interp(rr, r, q)
        out.append(4 / (math.pi * l * l) * np.sum(_trap(rr) * rr * qq))
    return np.array(out)


def _weib_sum(notch, m):
    v = fields()[notch]["volume"]
    return np.sum(np.asarray(v["w"]) * np.asarray(v["s1"]) ** m)


def k_weibull(notch, m, V0):
    return (_weib_sum(notch, m) / V0) ** (1 / m)


def s90(notch, frac=0.9):
    s = fields()[notch]["surface"]
    a, b = np.asarray(s["s1_arc"]), np.asarray(s["s1_side"])
    sm = max(a.max(), b.max())
    return float((np.asarray(s["dA_arc"]) * (a >= frac * sm)).sum() + (np.asarray(s["dA_side"]) * (b >= frac * sm)).sum())


def k_mhsv(notch, Lcd, Cs):
    return k_point(notch, Lcd) * (s90(notch) * Lcd / Cs) ** (1 / (K_LIFE * paper.PLAIN.k))


# name: (K_eq(notch, N, p), start, transform of the free vector to the constants, label of the constants)
def _ab(f):
    return lambda n, N, p: f(n, p[0] * np.asarray(N, float) ** p[1])


METHODS = {
    "point": (_ab(k_point), (16.4, -0.242), "L_M = {:.3g} N^{:.3f} mm"),
    "line": (_ab(k_line), (5.0, -0.2), "L = {:.3g} N^{:.3f} mm"),
    "area": (_ab(k_area), (16.4, -0.242), "L = point L_M = {:.3g} N^{:.3f} mm (not refitted, as the paper)"),
    "netfit": (lambda n, N, p: np.full(np.shape(N), p[0]), (1.0,), "sigma_n,max x {:.3g}"),
    "weibull": (lambda n, N, p: np.full(np.shape(N), k_weibull(n, M_STRENGTH, p[0])), (100.0,), "whole volume; m 3.44 FIXED (plain band), V0 {:.3g} mm^3"),
    "mhsv": (lambda n, N, p: np.full(np.shape(N), k_mhsv(n, p[1], p[0])), (50.0,), "C_S {:.3g} mm^3; FIXED: L_CD {:.3g} mm (point L_M at 2e6), k_N 0.675 (plain band)"),
}


def l_cd():
    """mHSV critical distance: our point-method L_M at N_ref (calibrated on plain + sharp V)."""
    (A, B), _ = calibrate("point")
    return A * paper.N_REF ** B


def calibrate(name, notch="sharp", curve=None):
    f, p0, _ = METHODS[name]
    b = basis(name)
    curve = b["sharp"] if curve is None else curve
    target = np.log10(b["plain"](N_CAL) / curve(N_CAL))
    if name == "area":                                     # the paper's way: area uses the point-method L_M(N)
        p, _ = calibrate("point", notch, curve)
        return p, math.sqrt(np.mean((np.log10(f(notch, N_CAL, p)) - target) ** 2))
    pos = len(p0) == 1                                     # single positive constant: fit in log
    extra = (l_cd(),) if name == "mhsv" else ()

    def unpack(x):
        return np.array([math.exp(x[0]), *extra]) if pos else np.array([math.exp(x[0]), x[1]])

    def cost(x):
        k = f(notch, N_CAL, unpack(x))
        return np.sum((np.log10(np.maximum(k, 1e-12)) - target) ** 2)

    x0 = np.log(p0) if pos else np.array([math.log(p0[0]), p0[1]])
    D = [np.zeros(1), np.ones(1), -np.ones(1)] if pos else [np.zeros(2), np.array([0.7, 0.05]), np.array([-0.7, -0.05])]
    best = min((minimize(cost, x0 + d, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-14, maxiter=8000))
                for d in D), key=lambda o: o.fun)
    return unpack(best.x), math.sqrt(best.fun / len(N_CAL))


def residuals(name, p, specs):
    """log10(sigma_eq / plain P50 at the specimen's life), failures only, in the method's stress variable."""
    f, b = METHODS[name][0], basis(name)
    out = {}
    for k in paper.NOTCHES:
        S = np.array([(s.N, b["var"](s)) for s in specs if s.notch == k and not s.runout])
        out[k] = (S[:, 0], np.log10(f(k, S[:, 0], p) * S[:, 1] / b["plain"](S[:, 0])))
    return out


def stats(N, r, T=paper.PLAIN.T):
    """T: scatter ratio of the plain band the residuals refer to (its P10-P90 half-width is log10 sqrt T)."""
    slope = np.polyfit(np.log10(N), r, 1)[0]
    return dict(n=len(r), bias=float(np.mean(r)), rms=float(np.sqrt(np.mean(r ** 2))), slope=float(slope),
                band=float(np.mean(np.abs(r) <= math.log10(math.sqrt(T)))))


# ---------------------------------------------------------------- stress variable: sigma_max (paper) or SWT (departure)
def _fit_curve(N, S, runout=None):
    """Basquin P50 by the paper's regression (log N on log S, failures), T from the life scatter (log-normal)."""
    f = np.ones(len(N), bool) if runout is None else ~np.asarray(runout, bool)
    b, a = np.polyfit(np.log10(S[f]), np.log10(N[f]), 1)
    sd = np.std(np.log10(N[f]) - a - b * np.log10(S[f]), ddof=2)
    k = -b
    return paper.Basquin(10 ** ((np.log10(paper.N_REF) - a) / b), k, 10 ** (2 * 1.2816 * sd / k), cap=np.inf)


def swt(R):
    """sigma_SWT / sigma_max = sqrt(sigma_a / sigma_max) = sqrt((1 - R) / 2)."""
    return np.sqrt((1 - np.asarray(R, float)) / 2)


@lru_cache(None)
def swt_curves():
    """SWT is NOT in the paper. Plain P50 refitted on the Fig. 1 points (R -1, -0.5, 0, 0.3) in sigma_SWT, and the
    sharp curve refitted on the Fig. 8 sharp specimens in sigma_SWT; points at N >= 1.8e6 treated as run-outs."""
    A = paper.fig1_plain()
    plain = _fit_curve(A[:, 1], A[:, 2] * paper.UTS * swt(A[:, 0]), A[:, 1] >= 1.8e6)
    plain = paper.Basquin(plain.s_ref, plain.k, plain.T, cap=paper.UTS)
    sh = [s for s in paper.fig8_specimens() if s.notch == "sharp"]
    sharp = _fit_curve(np.array([s.N for s in sh]), np.array([s.s_max * swt(s.R) for s in sh]), [s.runout for s in sh])
    return plain, sharp


def basis(name):
    if name.startswith("swt"):
        plain, sharp = swt_curves()
        return dict(var=lambda s: s.s_max * float(swt(s.R)), plain=plain, sharp=sharp, label="sigma_SWT")
    return dict(var=lambda s: s.s_max, plain=paper.PLAIN, sharp=paper.SHARP_FIG8A, label="sigma_max")


# ---------------------------------------------------------------- stress gradient (Siebel-Stieler type support factor)
@lru_cache(None)
def chi(notch):
    """Relative stress gradient at the root, chi = -(d sigma_1/dx)/sigma_1 (1/mm), quadratic fit of the bisector profile
    over d <= rho/10."""
    p = _prof(notch)
    rho = {"sharp": 0.15, "intermediate": 1.0, "blunt": 3.0}[notch]
    m = p.d <= rho / 10
    c = np.polyfit(p.d[m], p.s[m], 2)
    return float(-c[1] / p.s[0])            # c = (a2, a1, a0): a1 is the slope at the root


def k_gradient(notch, c):
    """K_eq = Kt / (1 + sqrt(c chi)), c = A N^B (mm)."""
    return _prof(notch).s[0] / (1 + np.sqrt(np.asarray(c) * chi(notch)))


# ---------------------------------------------------------------- averaged strain energy density (Lazzarin-Berto)
def k_sed(notch, R0):
    """sqrt(2 E W_bar) / sigma_n: W averaged over the crescent of outer radius R0 + r0 centred r0 behind the root
    (r0 = rho (pi - 2a) / (2 pi - 2a)), thickness-mean, nu 0.35; plain uniaxial gives K = 1."""
    g = fields()[notch]["sed"]
    rc, q, w = np.asarray(g["rc"]), np.asarray(g["w2E"]), np.asarray(g["dA"])
    out = []
    for R in np.atleast_1d(R0):
        m = rc <= R + g["r0"]
        out.append(math.sqrt(np.sum((q * w)[m]) / np.sum(w[m])))
    return np.array(out)


METHODS.update({
    "swt_point": (_ab(k_point), (16.4, -0.242), "SWT variable (not in paper); L_M = {:.3g} N^{:.3f} mm"),
    "gradient": (_ab(k_gradient), (0.1, -0.2), "K_t / (1 + sqrt(c chi)), c = {:.3g} N^{:.3f} mm"),
    "sed": (_ab(k_sed), (2.0, -0.2), "R0 = {:.3g} N^{:.3f} mm (life-dependent, as L_M)"),
    # Lazzarin-Berto as published: one material control radius, independent of life
    "sed_r0": (lambda n, N, p: np.full(np.shape(N), k_sed(n, p[0])[0]), (0.5,), "R0 = {:.3g} mm (constant)"),
})


# ---------------------------------------------------------------- local measures along the bisector (Aleksey, 2026-11-16)
MEASURES = {"s1": "max principal stress", "vm": "von Mises (= octahedral shear x 3/sqrt2)",
            "eps1": "E x max principal strain", "syy": "stress normal to the bisector"}


@lru_cache(None)
def measure_profile(notch, meas):
    P = json.load(open(os.path.join(paper.DATA, "igabem_bisector_measures.json")))["profiles"][notch]
    return point.Profile(P["d"], P[meas])


def k_point_m(notch, L, meas):
    return measure_profile(notch, meas)(np.asarray(L) / 2)


def k_line_m(notch, L, meas):
    p = measure_profile(notch, meas)
    lig = p.d[-1]
    out = []
    for l2 in np.atleast_1d(np.minimum(2 * np.asarray(L, float), 2 * lig)):
        x = np.linspace(0, l2, 400)
        out.append(np.mean(p(np.where(x <= lig, x, 2 * lig - x))))
    return np.array(out)


for _m in ("vm", "eps1"):
    METHODS[f"point_{_m}"] = (_ab(lambda n, L, m=_m: k_point_m(n, L, m)), (16.4, -0.242),
                              f"point on {MEASURES[_m]}: L = {{:.3g}} N^{{:.3f}} mm")
    METHODS[f"line_{_m}"] = (_ab(lambda n, L, m=_m: k_line_m(n, L, m)), (5.0, -0.2),
                             f"line on {MEASURES[_m]}: L = {{:.3g}} N^{{:.3f}} mm")
