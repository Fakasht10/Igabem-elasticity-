"""Failure probability of a predicted load: physical scatter plus model error, for any deterministic predictor.

The geometry exponent of a Weibull volume law (m about 4 for the PLA cards) is fitted to the notch and size effect;
it is not the specimen-to-specimen scatter (log-std 0.028 on the PLA replicates, m_s about 46). So a predictor
(Weibull volume law, gradient-supported Weibull, TCD) gives the mean failure load F_pred, and

    ln F_f = ln F_pred + b + e_fam + e_s,   e_fam ~ N(0, tau^2),   F_f / F_pred given e_fam ~ Weibull(m_s), mean 1

b and tau are the bias and between-family std of the predictor's held-out log residuals (a family offset the model
does not explain, reported per family rather than smeared into the band); m_s comes from the replicate scatter
within groups. P_f(F) and the load at a probability follow by Gauss-Hermite over e_fam.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GUMBEL_STD = math.pi / math.sqrt(6.0)       # std of ln W for W ~ Weibull(m) is GUMBEL_STD / m
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(32)
_GH_W = _GH_W / _GH_W.sum()


def scatter_modulus(groups):
    """Weibull scatter modulus m_s from replicate groups (iterables of loads): pooled within-group std of ln F,
    m_s = (pi / sqrt 6) / s. Returns (m_s, s, dof)."""
    dev, dof = [], 0
    for g in groups:
        x = np.log(np.asarray(list(g), float))
        if len(x) < 2:
            continue
        dev.append(((x - x.mean()) ** 2).sum())
        dof += len(x) - 1
    if not dof:
        raise ValueError("no group with two or more replicates")
    s = math.sqrt(sum(dev) / dof)
    return GUMBEL_STD / s, s, dof


def family_effects(log_res, family, within_s):
    """Bias b, between-family std tau (method of moments, floored at 0) and the per-family offsets of held-out
    log residuals (observed / predicted), with within-specimen std within_s."""
    log_res, family = np.asarray(log_res, float), np.asarray(family)
    fams = sorted(set(family.tolist()))
    means = np.array([log_res[family == f].mean() for f in fams])
    n = np.array([(family == f).sum() for f in fams])
    b = float(means.mean())
    tau2 = means.var(ddof=1) - float(np.mean(within_s ** 2 / n)) if len(fams) > 1 else 0.0
    return b, math.sqrt(max(tau2, 0.0)), dict(zip(fams, (means - b).tolist()))


@dataclass(frozen=True)
class Scatter:
    m_s: float                  # physical scatter modulus
    bias: float = 0.0           # mean held-out log residual of the predictor
    tau: float = 0.0            # between-family std of the log residual (model error); 0 = physical scatter only

    def cdf(self, load, pred):
        """P_f at load(s) for a predicted mean failure load."""
        load = np.atleast_1d(np.asarray(load, float))
        g = math.gamma(1.0 + 1.0 / self.m_s)                               # Weibull scale = mean / g
        z = np.log(load)[:, None] - (math.log(pred) + self.bias + self.tau * _GH_X)[None, :] + math.log(g)
        return (-np.expm1(-np.exp(self.m_s * z))) @ _GH_W

    def load_at(self, p, pred):
        """Load at failure probability p (bisection on ln F; the cdf is monotone)."""
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between zero and one")
        lo, hi = math.log(pred) - 2.0, math.log(pred) + 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.cdf(math.exp(mid), pred)[0] < p else (lo, mid)
        return math.exp(0.5 * (lo + hi))

    def band(self, pred, ps=(0.01, 0.1, 0.5)):
        return {p: self.load_at(p, pred) for p in ps}
