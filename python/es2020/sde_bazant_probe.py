"""Quick probe of the Bazant 2019 Sec. 7 fatigue core (NOT the build; that lives in its own thread).

Per specimen: an intrinsic flaw a0 grows by a Walker-Paris law along the automatic crack path (sde.crack_path)
through the local elastic field, K(a) = S sigma_K(a) F_SENT(a/W) sqrt(pi a), to a_f = 0.7 W:
    ln N = n (ln sigma_D - ln S - gamma ln((1-R)/2)) + ln J(a0),   J(a0) = int_{a0}^{a_f} (g(a) sqrt(a))^-n da
(C, pi folded into sigma_D). a0 is random per specimen through the RVE strength r ~ a0^-1/2, whose cdf is the
Bazant-Pang Gauss-Weibull graft (Gaussian core in ln r, power-law tail below P_gr, pdf-continuous). Life quantile q
<-> strength quantile q, so the life law is computed exactly from the a0 quantiles (frozen, no OU).
Admissible: n > 2 (finite life from a vanishing flaw needs it), 0 <= gamma <= 1, 0 < P_gr < 1, s > 0.
theta = (ln sigma_D, n, gamma, ln a0_med, ln s, logit P_gr)."""
import math
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp, expit
from . import sde
from . import methods as M

NAMES = ("sigma_D", "n", "gamma", "a0_med", "s_lnr", "P_gr")
LO = np.array([math.log(1.0), 2.05, 0.0, math.log(1e-3), math.log(0.02), -9.0])
HI = np.array([math.log(1e5), 40.0, 1.0, math.log(2.0), math.log(1.5), 0.0])


def lnr_quantile(u, s, pgr):
    zg = norm.ppf(pgr)
    lam = norm.pdf(zg) / (s * pgr)                      # pdf continuity at the graft -> tail modulus m = lam
    return np.where(u >= pgr, s * norm.ppf(u), s * zg + np.log(np.minimum(u, pgr) / pgr) / lam)


class ParisChain:
    def __init__(self, rows, profiles, W, ng=320):
        """profiles: kind -> callable sigma_1 per unit net stress along the path (None = uniform); W: kind -> free path."""
        self.notch = np.array([r["notch"] for r in rows]); self.R = np.array([r["R"] for r in rows])
        self.s = np.array([r["s_max"] for r in rows]); self.lnN = np.log([r["N"] for r in rows])
        self.ro = np.array([r["runout"] for r in rows]); self.kinds = sorted(set(self.notch))
        self.grid = {}
        for k in self.kinds:
            af = 0.7 * W[k]; a = np.geomspace(1e-4, af, ng)
            p = profiles[k]
            sk = np.ones_like(a) if p is None else np.array([sde.crack_stress(p, x) for x in a])
            g = sk * sde.f_sent(a / W[k])
            self.grid[k] = (np.log(a), np.log(g) + 0.5 * np.log(a))   # ln a, ln(g sqrt a)

    def lnJ(self, k, n, lna0):
        la, lh = self.grid[k]
        li = -n * lh + la                                  # integrand in d ln a: (g sqrt a)^-n a
        dl = np.diff(la)
        seg = np.logaddexp(li[1:], li[:-1]) + np.log(dl / 2)
        cum = np.r_[np.logaddexp.accumulate(seg[::-1])[::-1], -np.inf]   # int_{a_i}^{a_f}
        return np.interp(np.clip(lna0, la[0], la[-2]), la, cum)

    def predict(self, th, s=None):
        lsD, n, g, la0, ls, lp = th
        S = self.s if s is None else s
        lc = n * (lsD - np.log(S) - g * np.log((1 - self.R) / 2))
        lna0 = la0 - 2 * lnr_quantile(sde.QLEV, math.exp(ls), expit(lp))   # quantile q of life <-> q of strength
        Q = np.zeros((len(S), len(sde.QLEV)))
        for k in self.kinds:
            Q[self.notch == k] = self.lnJ(k, n, lna0)
        return lc, Q

    def loglike(self, th):
        if np.any(th < LO) or np.any(th > HI):
            return -np.inf
        lc, Q = self.predict(th)
        Q = np.maximum.accumulate(Q, axis=1)
        lp, ls = sde.logpdf_cdf(self.lnN - lc, Q)
        v = float(np.sum(np.where(self.ro, ls, lp)))
        return v if np.isfinite(v) else -np.inf
