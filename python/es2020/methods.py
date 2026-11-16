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
    for l2 in np.atleast_1d(2 * np.asarray(L, float)):
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


def calibrate(name, notch="sharp", curve=paper.SHARP_FIG8A):
    f, p0, _ = METHODS[name]
    target = np.log10(paper.PLAIN(N_CAL) / curve(N_CAL))
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
    """log10(sigma_eq / plain P50 at the specimen's life), failures only."""
    f = METHODS[name][0]
    out = {}
    for k in paper.NOTCHES:
        S = np.array([(s.N, s.s_max) for s in specs if s.notch == k and not s.runout])
        out[k] = (S[:, 0], np.log10(f(k, S[:, 0], p) * S[:, 1] / paper.PLAIN(S[:, 0])))
    return out


def stats(N, r):
    slope = np.polyfit(np.log10(N), r, 1)[0]
    return dict(n=len(r), bias=float(np.mean(r)), rms=float(np.sqrt(np.mean(r ** 2))), slope=float(slope),
                band=float(np.mean(np.abs(r) <= BAND)))
