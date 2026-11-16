"""Checks of the Bazant chain model against its own asymptotics (formulation doc, 'Asymptotics')."""
import math
import numpy as np
from scipy.stats import norm
from . import bazant as B, sde, methods as M


def _uniform(nu_area, W=1e4, k=2):
    rows = [dict(notch="plain", R=-1.0, s_max=10.0, N=1e5, runout=False)] * k
    return B.Chain(rows, {"plain": (np.ones(1), np.array([nu_area]))}, {"plain": W}, {"plain": None})


TH = np.array([math.log(80.0), 5.0, 0.4, math.log(0.05), math.log(0.3), math.log(0.01 / 0.99)])


def test_graft_cdf_valid():
    s, p = 0.3, 0.01
    lnr = np.linspace(-8, 3, 4000)
    lH = B.log_neglog_surv(lnr, s, p)
    P = -np.expm1(-np.exp(lH))
    assert np.all(np.diff(P) >= -1e-15) and P[0] < 1e-6 and P[-1] > 0.999
    m = B.tail_modulus(s, p); lrg = s * norm.ppf(p)                     # pdf continuity at the graft
    pdf_tail = m * p; pdf_gauss = norm.pdf(norm.ppf(p)) / s
    assert abs(pdf_tail / pdf_gauss - 1) < 1e-12 and abs(np.interp(lrg, lnr, P) / p - 1) < 1e-2


def test_life_modulus_m_over_n_minus_2():
    """Deep tail, uniform field, a0 << W: ln H vs ln N has slope m/(n-2)."""
    c = _uniform(1.0, k=200)
    g = np.linspace(-10, 30, 200)
    lHg = c.lnH(TH, lnN=g, s=np.full(200, 10.0))
    l0 = np.interp(-8.0, lHg, g)                                  # deep tail, flaw << W
    lnN = np.array([l0, l0 + 0.5])
    lH = _uniform(1.0).lnH(TH, lnN=lnN, s=np.full(2, 10.0))
    m = B.tail_modulus(math.exp(TH[4]), 0.01)
    slope = np.diff(lH)[0] / np.diff(lnN)[0]
    assert abs(slope / (m / (TH[1] - 2)) - 1) < 0.02, (slope, m / (TH[1] - 2))


def test_basquin_slope_n_and_chain_scaling():
    """At fixed P_f in the tail: ln H rises by m n/(n-2) per ln S and by 1 per ln N_eq."""
    c1, c2 = _uniform(1.0, k=1), _uniform(10.0, k=1)
    g = np.linspace(-10, 30, 200)
    lnN = np.full(1, np.interp(-8.0, _uniform(1.0, k=200).lnH(TH, lnN=g, s=np.full(200, 10.0)), g))
    a = c1.lnH(TH, lnN, s=np.array([10.0])); b = c1.lnH(TH, lnN, s=np.array([10.0 * math.e ** 0.01]))
    m = B.tail_modulus(math.exp(TH[4]), 0.01); n = TH[1]
    assert abs((b - a)[0] / 0.01 / (m * n / (n - 2)) - 1) < 0.02
    d = c2.lnH(TH, lnN, s=np.array([10.0])) - a
    assert abs(d[0] - math.log(10.0)) < 1e-9


def test_scaled_profile_vs_radial_path():
    """Off-root bins use the root profile scaled by kappa. Compare sigma_K over the first 0.4 mm with the true
    radial line from the stored polar grid at the angle where sigma_1 at the surface is 0.9 x peak."""
    for n in ("intermediate", "blunt"):
        a = M.fields()[n]["area"]
        r, th, s = np.asarray(a["r"]), np.asarray(a["th"]), np.asarray(a["s1"])
        j = int(np.argmin(np.abs(s[0] / s[0, 0] - 0.9)))
        prof_true = lambda x: np.interp(x, r, s[:, j])
        prof_scaled = lambda x: np.interp(x, r, s[:, 0]) * s[0, j] / s[0, 0]
        for aa in (0.1, 0.4):
            e = sde.crack_stress(prof_true, aa) / sde.crack_stress(prof_scaled, aa) - 1
            assert abs(e) < 0.05, (n, aa, e)
