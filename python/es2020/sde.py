"""Latent-damage SDE for notch fatigue (thread "Model notch fatigue as latent damage SDE", 2026-11-16).

Theory: https://claude.ai/artifact/S8XxRQMfvt4DpG3zCEmg1F. Pointwise, geometry-free process model:
input = elastic stress at a point and its gradient (here sigma_1 and the bisector profile of the stored IgBEM field,
unit net load), material constants only. Pipeline per specimen:

    sigma_eff  = support(sigma_e, grad sigma_e; L)            gradient group chi L (pointwise or averaged over L)
    loop       = Neuber on the Masing / Ramberg-Osgood curve   E', K', n'   (admissible: K' > 0, 0 < n' <= 1)
    Phi        = loop integral of (<sigma>+/sigma_D)^(2s) dp   Lemaitre damage per cycle, s >= 0
    c          = 1 / ((2s + 1) Phi)                            deterministic life (sigma_Z -> 0)
    N_f        = first passage of int exp(-Z - sigma_Z^2/2) dN (mean multiplier 1) to c,  Z = OU(sigma_Z, tau) in cycles

The first-passage law depends only on (sigma_Z, c/tau): one Monte Carlo table (ou_table) serves every specimen and
every parameter draw. Post-processing only: no IgBEM solve anywhere in this module.
"""
import json
import math
import os
from functools import lru_cache

import numpy as np

from . import methods as M
from . import paper

TABLE = os.path.join(paper.DATA, "sde_ou_table.npz")
GL_X, GL_W = np.polynomial.legendre.leggauss(32)

# ----------------------------------------------------------------------------------------------- admissible set
# theta = (ln E' [MPa], ln K' [MPa], n', ln sigma_D [MPa], s, ln sigma_Z, ln tau [cycles], ln L [mm])
NAMES = ("E'", "K'", "n'", "sigma_D", "s", "sigma_Z", "tau", "L")
LO = np.array([math.log(100.0), math.log(1.0), 0.01, math.log(1.0), 0.0, math.log(0.02), math.log(1.0), math.log(1e-3)])
HI = np.array([math.log(1e5), math.log(1e4), 1.0, math.log(1e5), 20.0, math.log(2.5), math.log(1e9), math.log(10.0)])


def admissible(th):
    """A1-A5 of the theory doc as box constraints (the only prior information). n' <= 1 is the second law."""
    th = np.asarray(th)
    return bool(np.all(th >= LO) and np.all(th <= HI))


def unpack(th):
    E, K, n, sD, s, sZ, tau, L = th
    return math.exp(E), math.exp(K), n, math.exp(sD), s, math.exp(sZ), math.exp(tau), math.exp(L)


# ----------------------------------------------------------------------------------------------- local loop
def _ro_newton(rhs, E, K, n, a):
    """Solve x (x/E + a (x/(aK))^(1/n)) ... in the form x*eps(x) = rhs, eps(x) = x/E + a*(x/(a*K))^(1/n),
    a = 1 (monotonic curve) or 2 (doubled Masing curve). Newton on ln x; x*eps(x) is strictly increasing on A2,
    so the root is unique. Start from the elastic solution x = sqrt(rhs E) (an upper bound)."""
    rhs = np.asarray(rhs, float)
    x = np.sqrt(rhs * E)
    for _ in range(60):
        p = a * (x / (a * K)) ** (1 / n)
        f = x * x / E + x * p - rhs
        df = 2 * x * x / E + x * p * (1 + 1 / n)          # d f / d ln x
        step = np.clip(f / np.maximum(df, 1e-300), -2.0, 2.0)
        x = x * np.exp(-step)
        if np.all(np.abs(step) < 1e-13):
            break
    return x


def loop(se_max, R, E, K, n):
    """Local (sigma_max, sigma_min) from the elastic (effective) peak se_max and load ratio R.
    First loading on the monotonic curve, range on the doubled curve (Masing), Neuber both times."""
    se_max = np.asarray(se_max, float)
    smax = _ro_newton(se_max ** 2 / E, E, K, n, 1.0)
    ds = _ro_newton(((1 - np.asarray(R)) * se_max) ** 2 / E, E, K, n, 2.0)
    return smax, smax - ds


def phi(smax, smin, K, n, sD, s):
    """Damage per cycle: integral over both Masing branches of (<sigma>+/sigma_D)^(2s) dp, with
    dp/dsigma = (1/(n K)) (x/(2K))^(1/n - 1), x = distance from the branch origin. Tension only (quasi-unilateral)."""
    a = np.maximum(smin, 0.0)
    b = smax
    h = 0.5 * (b - a)
    sig = 0.5 * (b + a)[..., None] + h[..., None] * GL_X
    dS = (smax - smin)[..., None]
    g = lambda x: (1 / (n * K)) * (np.maximum(x, 0) / (2 * K)) ** (1 / n - 1)
    f = (sig / sD) ** (2 * s) * (g(sig - smin[..., None]) + g(smax[..., None] - sig))
    return h * (f @ GL_W)


def life_det(se_max, R, th, load_control=None):
    """Deterministic life c = 1 / ((2s+1) Phi) (the sigma_Z -> 0 limit). Points flagged load_control (a uniform
    gauge: the local stress is the applied stress) skip Neuber; every other point gets Neuber on its elastic stress."""
    E, K, n, sD, s, *_ = unpack(th)
    smax, smin = loop(se_max, R, E, K, n)
    if load_control is not None:
        se_max = np.asarray(se_max, float)
        smax = np.where(load_control, se_max, smax)
        smin = np.where(load_control, np.asarray(R) * se_max, smin)
    return 1.0 / ((2 * s + 1) * np.maximum(phi(smax, smin, K, n, sD, s), 1e-300))


def dissipation_per_cycle(se_max, R, th):
    """Masing loop area W_c (MJ/m^3 = MPa) = Delta sigma Delta eps^p (1-n')/(1+n'): self-heating diagnostic."""
    E, K, n, *_ = unpack(th)
    smax, smin = loop(se_max, R, E, K, n)
    ds = smax - smin
    return ds * 2 * (ds / (2 * K)) ** (1 / n) * (1 - n) / (1 + n)


# ----------------------------------------------------------------------------------------------- OU first passage
def ein(x):
    """Ein(x) = int_0^x (e^t - 1)/t dt."""
    t = np.linspace(0, x, 2001)[1:]
    v = np.expm1(t) / t
    return float(np.trapezoid(np.concatenate([[1.0], v]), np.linspace(0, x, 2001)))


def _simulate(sig, lv, paths=2000, dt=0.05, umax=2000.0, seed=0):
    """Samples of ln(N_f / c) for levels l = c / tau (array lv), one sigma_Z. Exact AR(1) step of the unit OU process,
    trapezoid integral of exp(-Z - sig^2/2) in u = N / tau. Returns array (len(lv), paths)."""
    rng = np.random.default_rng(seed)
    a = math.exp(-dt)
    b = math.sqrt(1 - a * a)
    out = np.full((len(lv), paths), np.nan)
    chunk = 250
    nstep = int(umax / dt)
    for c0 in range(0, paths, chunk):
        m = min(chunk, paths - c0)
        z = rng.standard_normal(m) * sig
        J = np.zeros(m)
        f0 = np.exp(-z - sig * sig / 2)
        todo = np.ones((len(lv), m), bool)
        res = np.full((len(lv), m), np.nan)
        for k in range(1, nstep + 1):
            z = a * z + b * sig * rng.standard_normal(m)
            f1 = np.exp(-z - sig * sig / 2)
            Jn = J + 0.5 * dt * (f0 + f1)
            hit = todo & (Jn[None, :] >= lv[:, None])
            if hit.any():
                i, j = np.nonzero(hit)
                frac = (lv[i] - J[j]) / (Jn[j] - J[j])          # linear inside the step
                res[i, j] = (k - 1 + frac) * dt
                todo[i, j] = False
                if not todo.any():
                    break
            J, f0 = Jn, f1
        out[:, c0:c0 + m] = res
    return np.log(out / lv[:, None])


QLEV = np.linspace(0.01, 0.99, 99)


def build_table(path=TABLE, sig=None, lgl=None, paths=3000, dt=0.05):
    """Quantiles of ln(N_f / c) on a (sigma_Z, log10 l) grid, l = c / tau. Stored as quantile functions, not moments,
    because the law is skewed (skew down to about -3 at large sigma_Z and intermediate l)."""
    sig = np.linspace(0.02, 2.5, 22) if sig is None else sig
    lgl = np.arange(-3.0, 3.01, 0.25) if lgl is None else lgl
    lv = 10.0 ** lgl
    Q = np.zeros((len(sig), len(lgl), len(QLEV)))
    miss = np.zeros((len(sig), len(lgl)))
    for i, s in enumerate(sig):
        x = _simulate(s, lv, paths=paths, dt=dt, umax=4.0 * lv[-1], seed=i)
        miss[i] = np.mean(np.isnan(x), axis=1)
        Q[i] = np.nanquantile(x, QLEV, axis=1).T
    np.savez(path, sig=sig, lgl=lgl, Q=Q, qlev=QLEV, miss=miss, paths=paths, dt=dt)
    return path


@lru_cache(maxsize=1)
def table():
    t = np.load(TABLE)
    return {k: t[k] for k in t.files}


def quantiles(sZ, l):
    """Quantile functions of ln(N_f / c), one row per (sigma_Z, l) pair: bilinear in (sigma_Z, log10 l) inside the
    table; exact log-normal (frozen) below it; Gaussian averaging asymptote above it, sd^2 = 2 Ein(sigma^2) / l."""
    from scipy.stats import norm
    T = table()
    l = np.atleast_1d(np.asarray(l, float))
    sZ = np.broadcast_to(np.asarray(sZ, float), l.shape)
    ll = np.log10(np.maximum(l, 1e-300))
    S, G = T["sig"], T["lgl"]
    i = np.clip(np.searchsorted(S, sZ) - 1, 0, len(S) - 2)
    j = np.clip(np.searchsorted(G, ll) - 1, 0, len(G) - 2)
    ws = np.clip((sZ - S[i]) / (S[i + 1] - S[i]), 0, 1)[:, None]
    wl = np.clip((ll - G[j]) / (G[j + 1] - G[j]), 0, 1)[:, None]
    A = T["Q"]
    Qr = ((1 - ws) * (1 - wl) * A[i, j] + ws * (1 - wl) * A[i + 1, j]
          + (1 - ws) * wl * A[i, j + 1] + ws * wl * A[i + 1, j + 1])
    z = norm.ppf(T["qlev"])[None, :]
    lo, hi = ll < G[0], ll > G[-1]
    if lo.any():
        Qr[lo] = sZ[lo, None] ** 2 / 2 + sZ[lo, None] * z
    if hi.any():
        sd = np.sqrt(2 * np.array([ein(v) for v in sZ[hi] ** 2]) / l[hi])
        Qr[hi] = 0.5 * sd[:, None] ** 2 + sd[:, None] * z
    return Qr


def logpdf_cdf(y, Qr):
    """log density and log survival of y = ln(N/c) under per-row quantile functions Qr (rows x 99).
    Piecewise-linear CDF between the 1 % and 99 % quantiles, exponential tails matched in density outside them."""
    q = table()["qlev"]
    y = np.asarray(y, float)
    Qv = np.maximum.accumulate(Qr, axis=1)
    dQ = np.maximum(np.diff(Qv, axis=1), 1e-9)
    n = Qv.shape[1]
    k = np.sum(Qv <= y[:, None], axis=1)                      # number of quantiles below y
    rows = np.arange(len(y))
    kk = np.clip(k, 1, n - 1)
    f = (q[kk] - q[kk - 1]) / dQ[rows, kk - 1]
    F = q[kk - 1] + f * (y - Qv[rows, kk - 1])
    lp = np.log(f)
    ls = np.log(np.maximum(1 - F, 1e-300))
    lam0 = (q[1] - q[0]) / dQ[:, 0] / q[0]                      # lower tail: F = q0 exp(-lam0 (Q0 - y))
    lo = k == 0
    lnF = math.log(q[0]) - lam0 * (Qv[:, 0] - y)
    lp = np.where(lo, np.log(lam0) + lnF, lp)
    ls = np.where(lo, np.log1p(-np.exp(np.minimum(lnF, -1e-12))), ls)
    lam1 = (q[-1] - q[-2]) / dQ[:, -1] / (1 - q[-1])            # upper tail: 1 - F = (1 - q99) exp(-lam1 (y - Q99))
    hi = k >= n
    lnG = math.log(1 - q[-1]) - lam1 * (y - Qv[:, -1])
    lp = np.where(hi, np.log(lam1) + lnG, lp)
    ls = np.where(hi, lnG, ls)
    return lp, ls


# ----------------------------------------------------------------------------------------------- support (gradient)
LIG = 3.0   # half net width (mm): the bisector profile ends at the specimen centre line (symmetric beyond)


def _refl(x):
    x = np.asarray(x, float)
    return np.where(x <= LIG, x, 2 * LIG - np.minimum(x, 2 * LIG))


def k_eff(notch, L, mode):
    """Effective elastic stress per unit net stress at the hot spot of a notch.
    'point': Kt / (1 + chi L / 2), chi = relative gradient at the root (first-order process-zone average, Pade-stable).
    'avg'  : Kt (1 - chi_bar L / 2) with chi_bar = (sigma(0) - sigma(L)) / (L sigma(0)), the gradient averaged over L;
             this equals (sigma(0) + sigma(L)) / 2, the trapezoid line average over L.
    'peak' : Kt (no support, L unused)."""
    p = M._prof(notch)
    kt = float(p.s[0])
    if mode == "peak":
        return kt
    if mode == "point":
        return kt / (1 + M.chi(notch) * L / 2)
    if mode == "avg":
        return 0.5 * (kt + float(p(_refl(L))))
    raise ValueError(mode)


# ----------------------------------------------------------------------------------------------- data
FIT_NOTCH = ("sharp",)
HOLD_NOTCH = ("intermediate", "blunt")
PLAIN_RUNOUT_N = 1.8e6          # Fig. 1 draws no run-out flag; 16 points at N >= 1.8e6 (as in fatigue.calib)


def dataset():
    """Rows: dict(set, notch, R, s_max [MPa, net for notches, gross = local for plain], N, runout)."""
    rows = []
    for R, N, su in paper.fig1_plain():
        rows.append(dict(set="plain", notch="plain", R=float(R), s_max=float(su) * paper.UTS, N=float(N),
                         runout=bool(N >= PLAIN_RUNOUT_N)))
    for q in paper.specimens():
        rows.append(dict(set="fit" if q.notch in FIT_NOTCH else "held", notch=q.notch, R=q.R, s_max=q.s_max, N=q.N,
                         runout=q.runout))
    return rows


class Model:
    """Vectorised likelihood over a row subset for a given support mode."""

    def __init__(self, rows, mode="point"):
        self.rows, self.mode = rows, mode
        self.notch = np.array([r["notch"] for r in rows])
        self.R = np.array([r["R"] for r in rows])
        self.s = np.array([r["s_max"] for r in rows])
        self.lnN = np.log(np.array([r["N"] for r in rows]))
        self.ro = np.array([r["runout"] for r in rows])
        self.kinds = sorted(set(self.notch))

    def keff(self, L):
        k = np.ones(len(self.s))
        for n in self.kinds:
            if n != "plain":
                k[self.notch == n] = k_eff(n, L, self.mode)
        return k

    def predict(self, th, s=None, idx=None):
        """(ln c, quantile rows of ln(N_f / c)) per row, optionally at other stresses s."""
        E, K, n, sD, sp, sZ, tau, L = unpack(th)
        k = self.keff(L)
        R = self.R
        if idx is not None:
            k, R = k[idx], R[idx]
        se = (self.s if s is None else s) * k
        lc = np.log(life_det(se, R, th, self.notch == "plain" if idx is None else (self.notch == "plain")[idx]))
        return lc, quantiles(sZ, np.exp(lc) / tau)

    def loglike(self, th):
        if not admissible(th):
            return -np.inf
        lc, Qr = self.predict(th)
        if not np.all(np.isfinite(lc)):
            return -np.inf
        lp, ls = logpdf_cdf(self.lnN - lc, Qr)
        return float(np.sum(np.where(self.ro, ls, lp)))


def log_prior(th):
    return 0.0 if admissible(th) else -np.inf


# ----------------------------------------------------------------------------------------------- scoring
HALF_BAND = 0.5 * math.log10(paper.PLAIN.T) if hasattr(paper.PLAIN, "T") else 0.5 * math.log10(2.451)


def stress_residual(model, th, iters=60):
    """Per row: log10(S* / S_obs), S* = the stress whose predictive median life equals the observed life
    (the same residual definition as fatigue.calib and the sibling thread). Bisection in log S (median life is
    monotone in S). Also returns the life residual ln N_obs - median ln N and coverage of our P10-P90."""
    lo = np.log(model.s) - 4.0
    hi = np.log(model.s) + 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lc, Q = model.predict(th, s=np.exp(mid))
        longer = lc + Q[:, 49] > model.lnN          # predicted life too long -> stress must go up
        lo = np.where(longer, mid, lo)
        hi = np.where(longer, hi, mid)
    s_star = np.exp(0.5 * (lo + hi))
    lc, Q = model.predict(th)
    y = model.lnN - lc
    return dict(dlogS=np.log10(s_star / model.s), dlnN=y - Q[:, 49],
                inside=(y >= Q[:, 9]) & (y <= Q[:, 89]), width10=(Q[:, 89] - Q[:, 9]) / math.log(10))


def score(r, mask, logN):
    d = r["dlogS"][mask]
    x = logN[mask]
    sl = np.polyfit(x, d, 1)[0] if len(d) > 2 else float("nan")
    drift = sl * (x.max() - x.min())
    return dict(n=int(mask.sum()), rms=float(np.sqrt(np.mean(d ** 2))), bias=float(np.mean(d)),
                slope=float(sl), tilted=bool(abs(drift) > 0.05), inband=float(np.mean(np.abs(d) <= HALF_BAND)),
                rms_log10N=float(np.sqrt(np.mean((r["dlnN"][mask] / math.log(10)) ** 2))),
                coverage=float(np.mean(r["inside"][mask])))
