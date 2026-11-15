"""Transfer criteria: the plain-specimen strength s0 (amplitude at N_ref, same R and print angle) -> the notched
specimen's nominal net strength, from a unit-net-stress field (fatigue.fields). Every function returns the predicted
net strength in the units of s0.

    kt          s0 / Kt                                   no length; the elastic hot-spot bound (Kf = Kt)
    tcd_point   s0 / sigma_1(L/2)                         TCD point method, L a material length (fitted)
    tcd_line    s0 / mean sigma_1 over [0, 2L]            TCD line method
    weibull     s0 (V_plain / J_m)^(1/m),  J_m = int (sigma_1+)^m dV over the specimen
                                                          weakest link, m from the replicate scatter: no length
    weibull_support  as weibull with sigma_1 / n, n = 1 + sqrt(l |grad sigma_1| / sigma_1)
                                                          (Maede et al. 2017, as failure.aniso.support_factor; l fitted)

The bisector quantities are averaged over the thickness stations, as a 2D plane model would give them.
"""
import math

import numpy as np

GUMBEL_STD = math.pi / math.sqrt(6.0)
Z90 = 1.2815515655446004                       # standard normal 90% quantile


def m_from_T(T):
    """Weibull modulus from a scatter ratio T = s_10% / s_90% (log-normal band: ln T = 2 z90 s_ln)."""
    return GUMBEL_STD * 2 * Z90 / math.log(T)


def m_pooled(series):
    """Weibull modulus from several replicate series [(T, n)]: pooled log-std, weighted by n - 1."""
    num = sum((n - 1) * (math.log(T) / (2 * Z90)) ** 2 for T, n in series)
    return GUMBEL_STD / math.sqrt(num / sum(n - 1 for _, n in series))


def walker_fit(series):
    """Walker exponent from plain amplitudes at several R [(R, s_a)]: s_a (2 / (1 - R))^(1 - gamma) = s_ar.
    Least squares in log s_a = log s_ar - (1 - gamma) log(2 / (1 - R)); returns (gamma, s_ar, rms log residual)."""
    x = np.array([math.log(2 / (1 - R)) for R, _ in series])
    y = np.log([a for _, a in series])
    A = np.column_stack([np.ones_like(x), -x])
    (c, one_minus_g), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ np.array([c, one_minus_g])
    return 1 - one_minus_g, math.exp(c), float(np.sqrt((res ** 2).mean()))


def walker_amplitude(s_ar, R, gamma):
    """Amplitude at ratio R equivalent to the fully reversed s_ar."""
    return s_ar / (2 / (1 - R)) ** (1 - gamma)


def line_profile(f):
    return f["d"], np.asarray(f["s1_line"]).mean(0)


def kt(f, s0):
    return s0 / float(np.asarray(f["s1_line"])[:, 0].mean())


def tcd_point(f, s0, L):
    d, s = line_profile(f)
    if L / 2 > d[-1]:
        raise ValueError(f"L/2 = {L / 2:g} mm is past the ligament ({d[-1]:g} mm)")
    return s0 / float(np.interp(L / 2, d, s))


def tcd_line(f, s0, L, mirror=True):
    """Mean over [0, 2L]; past the ligament an edge plate's profile continues as its mirror (x -> -x)."""
    d, s = line_profile(f)
    if 2 * L > d[-1]:
        if not mirror:
            raise ValueError(f"2L = {2 * L:g} mm is past the ligament ({d[-1]:g} mm)")
        lig = d[-1]
        d, s = np.concatenate([d, 2 * lig - d[::-1][1:]]), np.concatenate([s, s[::-1][1:]])
        if 2 * L > d[-1]:
            raise ValueError("2L longer than the whole net section")
    x = np.linspace(0.0, 2 * L, 801)
    return s0 / float(np.trapezoid(np.interp(x, d, s), x) / (2 * L))


def log_J(f, m, quarters=4):
    """log of int (sigma_1+)^m dV over the full specimen at unit net stress."""
    s, w = np.asarray(f["s1"]), np.asarray(f["w"])
    k = s > 0
    a = m * np.log(s[k]) + np.log(w[k])
    top = a.max()
    return math.log(quarters) + top + math.log(np.exp(a - top).sum())


def weibull(f, s0, m, V_plain):
    return s0 * math.exp((math.log(V_plain) - log_J(f, m)) / m)


def weibull_support(f, s0, m, V_plain, l):
    s, g = np.asarray(f["s1"]), np.asarray(f["g1"])
    chi = np.divide(g, s, out=np.zeros_like(s), where=s > 0)
    g2 = dict(f, s1=s / (1.0 + np.sqrt(l * chi)))
    return weibull(g2, s0, m, V_plain)
