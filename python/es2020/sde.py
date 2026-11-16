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
LO = np.array([math.log(100.0), math.log(1.0), 0.01, math.log(1e-4), 0.0, math.log(0.02), math.log(1.0), math.log(1e-3)])
HI = np.array([math.log(1e5), math.log(1e7), 1.0, math.log(1e5), 20.0, math.log(2.5), math.log(1e9), math.log(10.0)])


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


@lru_cache(maxsize=1)
def _ein_grid():
    x = np.linspace(0.0, 7.0, 701)
    return x, np.array([ein(v) if v > 0 else 0.0 for v in x])


def _ein_fast(x):
    g, v = _ein_grid()
    return np.interp(x, g, v)


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
        sd = np.sqrt(2 * _ein_fast(sZ[hi] ** 2) / l[hi])
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


def field_input(xyz, s1, h):
    """Geometry-free process-model input from ANY elastic field sampled at points xyz (n x d, mm) with values s1
    (max principal stress per unit load): the hot-spot value sigma_peak, the steepest-descent direction e (unit vector,
    -grad sigma / |grad sigma|) and the relative gradient chi = |grad sigma| / sigma_peak (1/mm). Weighted quadratic
    least squares on the points within h of the hot spot. No K_t, nominal stress, notch radius or bisector is used.
    (At a notch root on a free surface the tangential derivative vanishes, so e is the inward normal.)"""
    xyz = np.atleast_2d(np.asarray(xyz, float))
    s1 = np.asarray(s1, float)
    i0 = int(np.argmax(s1))
    d = xyz - xyz[i0]
    rr = np.linalg.norm(d, axis=1)
    m = rr <= h
    D = d[m]
    nd = D.shape[1]
    cols = [np.ones(len(D))] + [D[:, k] for k in range(nd)] + \
           [D[:, k] * D[:, l] for k in range(nd) for l in range(k, nd)]
    A = np.stack(cols, axis=1)
    w = np.exp(-(rr[m] / h) ** 2)
    coef, *_ = np.linalg.lstsq(A * w[:, None], s1[m] * w, rcond=None)
    g = coef[1:1 + nd]
    peak = coef[0]
    return float(peak), -g / np.linalg.norm(g), float(np.linalg.norm(g) / peak)


def notch_field_points(notch):
    """The stored thickness-mean sigma_1 field around a coupon root as a plain point cloud (x, y in mm, apex at 0):
    the polar grid of fields.json (theta from the bisector, one half by symmetry, mirrored here to both halves)."""
    a = M.fields()[notch]["area"]
    r, th, s = np.asarray(a["r"]), np.asarray(a["th"]), np.asarray(a["s1"])
    R, T = np.meshgrid(r, np.concatenate([-th[::-1], th]), indexing="ij")
    S = np.concatenate([s[:, ::-1], s], axis=1)
    return np.stack([R.ravel() * np.cos(T.ravel()), R.ravel() * np.sin(T.ravel())], axis=1), S.ravel()


def hot_spot_auto(xyz, s1, c=0.2, h0=1.0, it=8):
    """field_input with a self-consistent window h = c / chi (the field's own decay length), so no geometric length
    enters. Returns (sigma_peak, e, chi, h)."""
    h = h0
    for _ in range(it):
        peak, e, chi = field_input(xyz, s1, h)
        h_new = c / chi
        if abs(h_new / h - 1) < 1e-3:
            break
        h = h_new
    return peak, e, chi, h


@lru_cache(maxsize=None)
def hot_spot(notch):
    """(sigma_peak, chi) per unit net load for a coupon notch, read from the field cloud only (hot_spot_auto)."""
    xy, s = notch_field_points(notch)
    peak, e, chi, h = hot_spot_auto(xy, s)
    return peak, chi


def k_eff(notch, L, mode):
    """Effective elastic stress per unit net stress at the hot spot of a notch.
    'point': Kt / (1 + chi L / 2), chi = relative gradient at the root (first-order process-zone average, Pade-stable).
    'avg'  : Kt (1 - chi_bar L / 2) with chi_bar = (sigma(0) - sigma(L)) / (L sigma(0)), the gradient averaged over L;
             this equals (sigma(0) + sigma(L)) / 2, the trapezoid line average over L.
    'peak' : Kt (no support, L unused)."""
    p = M._prof(notch)
    kt, chi = hot_spot(notch)            # field-derived: peak and steepest-descent gradient, no K_t or bisector input
    if mode == "peak":
        return kt
    if mode == "point":
        return kt / (1 + chi * L / 2)
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


# ----------------------------------------------------------------------------------------------- weakest link
def surface_bins(notch, nb=40, cut=0.3):
    """Root-surface elastic sigma_1 per unit net stress as (k, area mm^2) bins above cut x peak (full specimen,
    multiplicities in the stored dA). Below the cut the cells add nothing (checked by test_wl_cut)."""
    f = M.fields()[notch]["surface"]
    k = np.concatenate([np.ravel(f["s1_arc"]), np.ravel(f["s1_side"])])
    a = np.concatenate([np.ravel(f["dA_arc"]), np.ravel(f["dA_side"])])
    m = k >= cut * k.max()
    k, a = k[m], a[m]
    edges = np.linspace(k.min(), k.max() + 1e-12, nb + 1)
    idx = np.digitize(k, edges) - 1
    A = np.bincount(idx, a, nb)
    K = np.bincount(idx, a * k, nb) / np.maximum(A, 1e-300)
    keep = A > 0
    return K[keep], A[keep]


class WLModel(Model):
    """Weakest link over surface cells of size l_c (the 8th parameter, read as l_c in mm instead of L):
    ln P(N_f > n) = sum_j (A_j / l_c^2) ln S_j(n). Plain gauge: one cell group, k = 1, area a_plain (mm^2)."""

    def __init__(self, rows, a_plain, nb=40, cut=0.3, support=None, L=None):
        """support='point' scales every root cell by the hot-spot factor 1/(1 + chi L/2), L fixed (from the
        pointwise fit): an approximation, since only the hot-spot gradient is stored, not one per cell."""
        super().__init__(rows, "peak")
        self.support, self.L = support, L
        self.a_plain = a_plain
        self.bins = {n: surface_bins(n, nb, cut) for n in self.kinds if n != "plain"}
        if support == "point":
            self.bins = {n: (K / (1 + M.chi(n) * L / 2), A) for n, (K, A) in self.bins.items()}
        self.bins["plain"] = (np.array([1.0]), np.array([a_plain]))

    def _cells(self, idx=None):
        rows = np.arange(len(self.s)) if idx is None else np.asarray(idx)
        r_of, k, a = [], [], []
        for i in rows:
            K, A = self.bins[self.notch[i]]
            r_of.append(np.full(len(K), i)); k.append(K); a.append(A)
        return np.concatenate(r_of), np.concatenate(k), np.concatenate(a)

    def loglike_rows(self, th, lnN=None, s=None):
        E, K, n, sD, sp, sZ, tau, lc_mm = unpack(th)
        r, k, a = self._cells()
        S = self.s if s is None else s
        se = S[r] * k
        c = life_det(se, self.R[r], th, self.notch[r] == "plain")
        Q = quantiles(sZ, c / tau)
        y = (self.lnN if lnN is None else lnN)[r] - np.log(c)
        lp, ls = logpdf_cdf(y, Q)
        w = a / lc_mm ** 2
        nrow = len(self.s)
        lnS = np.bincount(r, w * ls, nrow)
        haz = np.bincount(r, w * np.exp(lp - ls), nrow)          # hazard in ln N
        return lnS, np.log(np.maximum(haz, 1e-300)) + lnS

    def loglike(self, th):
        if not admissible(th):
            return -np.inf
        lnS, lpdf = self.loglike_rows(th)
        if not np.all(np.isfinite(lpdf)):
            return -np.inf
        return float(np.sum(np.where(self.ro, lnS, lpdf)))

    def median_lnN(self, th, s=None):
        """Specimen median life: solve ln S(n) = ln 0.5 by bisection in ln n."""
        lo = np.full(len(self.s), math.log(1.0))
        hi = np.full(len(self.s), math.log(1e12))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lnS, _ = self.loglike_rows(th, lnN=mid, s=s)
            alive = lnS > math.log(0.5)
            lo = np.where(alive, mid, lo)
            hi = np.where(alive, hi, mid)
        return 0.5 * (lo + hi)

    def band(self, th, p):
        lo = np.full(len(self.s), 0.0)
        hi = np.full(len(self.s), math.log(1e12))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lnS, _ = self.loglike_rows(th, lnN=mid)
            alive = lnS > math.log(1 - p)
            lo = np.where(alive, mid, lo)
            hi = np.where(alive, hi, mid)
        return 0.5 * (lo + hi)


def wl_residual(model, th, iters=40):
    lo = np.log(model.s) - 4.0
    hi = np.log(model.s) + 4.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        longer = model.median_lnN(th, s=np.exp(mid)) > model.lnN
        lo = np.where(longer, mid, lo)
        hi = np.where(longer, hi, mid)
    p10, p90, med = model.band(th, 0.1), model.band(th, 0.9), model.median_lnN(th)
    return dict(dlogS=np.log10(np.exp(0.5 * (lo + hi)) / model.s), dlnN=model.lnN - med,
                inside=(model.lnN >= p10) & (model.lnN <= p90), width10=(p90 - p10) / math.log(10))


# ----------------------------------------------------------------------------------------------- posterior predictive
def mix_cdf(model, draws, s=None):
    """Posterior-predictive P(N_f <= N_obs) per row at stress s (default: observed), averaged over draws."""
    acc = np.zeros(len(model.s))
    for th in draws:
        lc, Q = model.predict(th, s=s)
        _, ls = logpdf_cdf(model.lnN - lc, Q)
        acc += -np.expm1(ls)
    return acc / len(draws)


def predictive_residual(model, draws, iters=40):
    """log10(S*/S_obs) with S* the stress at which the predictive median life equals N_obs, i.e. the predictive
    CDF at N_obs is 0.5 (CDF at fixed N rises with S). Coverage: predictive CDF at the observation in [0.1, 0.9]."""
    lo = np.log(model.s) - 3.0
    hi = np.log(model.s) + 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = mix_cdf(model, draws, s=np.exp(mid)) < 0.5        # predicted life still longer than observed
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    F = mix_cdf(model, draws)
    return dict(dlogS=np.log10(np.exp(0.5 * (lo + hi)) / model.s), F=F, inside=(F >= 0.1) & (F <= 0.9),
                dlnN=np.zeros(len(F)) * np.nan)


# ----------------------------------------------------------------------------------------------- elastic TCD + Walker
class WalkerTCD(Model):
    """Variant adopted after the independent review (2026-11-16, https://claude.ai/artifact/LUrbt2R9Md6iBudjKEHNBf):
    - ONE structural map for every point (no plain/notch switch): elastic, no Neuber.
    - TCD point input: sigma_1 at depth L/2 along the field's steepest-descent direction from the hot spot (on these
      coupons that direction is the bisector, test_field_direction_is_bisector_and_chi_matches), L a material constant.
    - Damage per cycle Phi = (sigma_eff ((1-R)/2)^gamma / sigma_D)^k  (stress-driven Walker damage, rate >= 0 for any
      admissible k > 0, 0 <= gamma <= 1), deterministic life c = 1/Phi; the OU latent process as before.
    - Plain pool: stress scaled by exp(b), an explicit batch factor between the Fig. 1 pool (3 labs) and the notched
      batch. b = 0 ties them.
    theta = (ln sigma_D, k, gamma, ln sigma_Z, ln tau, b); L fixed by the caller."""
    NAMES = ("sigma_D", "k", "gamma", "sigma_Z", "tau", "b")
    LO = np.array([math.log(1.0), 1.0, 0.0, math.log(0.02), math.log(1.0), -2.0])
    HI = np.array([math.log(1e4), 40.0, 1.0, math.log(2.5), math.log(1e9), 2.0])

    def __init__(self, rows, L, plain_free=True):
        super().__init__(rows, "tcd")
        self.L, self.plain_free = L, plain_free
        self.k = np.ones(len(self.s))
        for n in self.kinds:
            if n != "plain":
                p = M._prof(n)
                self.k[self.notch == n] = float(p(_refl(L / 2)))
        self.plain = self.notch == "plain"

    def admissible(self, th):
        return bool(np.all(th >= self.LO) and np.all(th <= self.HI)) and (self.plain_free or th[5] == 0.0)

    def predict(self, th, s=None, idx=None):
        lsD, kk, g, lsZ, ltau, b = th
        S = self.s if s is None else s
        se = S * self.k * np.where(self.plain, math.exp(b), 1.0)
        lc = kk * (lsD - np.log(se) - g * np.log((1 - self.R) / 2))
        return lc, quantiles(math.exp(lsZ), np.exp(lc - ltau))

    def loglike(self, th):
        if not self.admissible(th):
            return -np.inf
        lc, Q = self.predict(th)
        lp, ls = logpdf_cdf(self.lnN - lc, Q)
        return float(np.sum(np.where(self.ro, ls, lp)))


# ----------------------------------------------------------------------------------------------- intrinsic crack
GL_T, GL_TW = np.polynomial.legendre.leggauss(24)


def crack_stress(profile, a):
    """Equivalent stress felt by an intrinsic edge crack of depth a (mm) in a field sigma(x), x = depth along the
    steepest-descent direction: sigma_K = K_I / (1.12 sqrt(pi a)) with the edge-crack weight function
    m = (2 * 1.12 / sqrt(pi a)) / sqrt(1 - (x/a)^2), i.e. sigma_K = (2/pi) int_0^{pi/2} sigma(a sin t) dt.
    Uniform field: sigma_K = sigma (plain gauge needs no special case). Geometry-free: only the field profile."""
    t = 0.25 * math.pi * (GL_T + 1)                   # [0, pi/2]
    return 0.5 * float(np.sum(GL_TW * profile(a * np.sin(t))))


class CrackWalker(WalkerTCD):
    """One rule for every specimen: sigma_eff = sigma_K(a_eq) of the local elastic field, Walker damage, OU latent.
    Plain gauge: uniform field, sigma_K = S. No batch factor. theta = (ln sigma_D, k, gamma, ln sigma_Z, ln tau, ln a_eq)."""
    NAMES = ("sigma_D", "k", "gamma", "sigma_Z", "tau", "a_eq")
    LO = np.array([math.log(1.0), 1.0, 0.0, math.log(0.02), math.log(1.0), math.log(1e-3)])
    HI = np.array([math.log(1e4), 40.0, 1.0, math.log(2.5), math.log(1e9), math.log(2.9)])

    def __init__(self, rows):
        Model.__init__(self, rows, "crack")
        self.plain = self.notch == "plain"
        self._cache = {}

    def admissible(self, th):
        return bool(np.all(th >= self.LO) and np.all(th <= self.HI))

    def keff_a(self, a):
        key = round(a, 6)
        if key not in self._cache:
            k = np.ones(len(self.s))
            for n in self.kinds:
                if n != "plain":
                    p = M._prof(n)
                    k[self.notch == n] = crack_stress(lambda x: p(_refl(x)), a)
            self._cache = {key: k}
        return self._cache[key]

    def predict(self, th, s=None, idx=None):
        lsD, kk, g, lsZ, ltau, la = th
        S = self.s if s is None else s
        se = S * self.keff_a(math.exp(la))
        lc = kk * (lsD - np.log(se) - g * np.log((1 - self.R) / 2))
        return lc, quantiles(math.exp(lsZ), np.exp(lc - ltau))


# ----------------------------------------------------------------------------------------------- finite width
def f_sent(al):
    """Single-edge crack in a strip, tension, a/W = al (Tada 2.10; error < 0.5 % for al <= 0.6)."""
    al = np.minimum(al, 0.9)
    return 1.12 - 0.231 * al + 10.55 * al ** 2 - 21.72 * al ** 3 + 30.39 * al ** 4


def f_dent(al):
    """Symmetric double-edge cracks, crack depth a over half width b, al = a/b (Tada 2.6)."""
    al = np.minimum(al, 0.95)
    return (1.122 - 0.561 * al - 0.205 * al ** 2 + 0.471 * al ** 3 - 0.190 * al ** 4) / np.sqrt(1 - al)


class CrackWalkerFW(CrackWalker):
    """CrackWalker with a finite-width correction: sigma_eq = sigma_K(a) * F(a / l) / 1.12, F for uniform load.
    l is the free path from the hot spot along the steepest-descent direction to the opposite free surface or to
    the symmetry plane, read from the body (coupons: the bisector profile ends at the specimen centre line, l = 3 mm,
    symmetric -> DENT). Plain gauge: uniform field, crack across the thinnest section t_plain (SENT, W = t_plain;
    Ref. [10] gives t 3 or 5 mm, other Fig. 1 labs not stated -> an input, with sensitivity)."""

    def __init__(self, rows, t_plain=3.0, l_notch=LIG):
        super().__init__(rows)
        self.t_plain, self.l_notch = t_plain, l_notch

    def keff_a(self, a):
        key = ("fw", round(a, 6))
        if key not in self._cache:
            k = np.full(len(self.s), float(f_sent(a / self.t_plain)) / 1.12)
            for n in self.kinds:
                if n != "plain":
                    p = M._prof(n)
                    k[self.notch == n] = crack_stress(lambda x: p(_refl(x)), a) * float(f_dent(a / self.l_notch)) / 1.122
            self._cache = {key: k}
        return self._cache[key]


# ------------------------------------------------------------------------------- one automatic path rule
def ray_to_free_surface(box, x0, e):
    """Distance from x0 along unit e to the boundary of an axis-aligned box [(lo, hi), ...] (the body's
    distance field along the ray, for prismatic gauges and ligaments)."""
    t = np.inf
    for (lo, hi), x, d in zip(box, x0, e):
        if d > 1e-12:
            t = min(t, (hi - x) / d)
        elif d < -1e-12:
            t = min(t, (lo - x) / d)
    return float(t)


# Bodies as the crack sees them (mm). Notched coupons: ligament between the two roots, x along the bisector
# (net width 6 mm, Ezeh & Susmel 2020), y through the thickness. Plain dogbone (Ref. [10]): width 6, thickness t.
def coupon_body(kind, t_plain=3.0, t_notch=3.0):
    if kind == "plain":
        return [(0.0, 6.0), (0.0, t_plain)]
    return [(0.0, 2 * LIG), (0.0, t_notch)]


def crack_path(kind, t_plain=3.0, chi_tol=1e-6):
    """One rule for every specimen. Start at the hot spot (a surface point of max sigma_1). The crack runs along
    the steepest-descent direction of sigma_1 into the body. If the field is uniform there (chi = 0, direction
    undefined), every inward direction sees the same sigma_K, so the rule takes the one with the largest
    F(a / W), i.e. the shortest free path. W = distance along the path to the nearest free surface.
    Returns (label, start, direction, W)."""
    box = coupon_body(kind, t_plain)
    chi, e = 0.0, None
    if kind != "plain":
        xy, s = notch_field_points(kind)
        _, e, chi, _ = hot_spot_auto(xy, s)       # direction and chi read from the field cloud only
    if chi > chi_tol:   # root at x = 0, mid-thickness; in-plane field -> no thickness component in e
        x0, e3 = np.array([0.0, box[1][1] / 2]), np.array([e[0], 0.0]) / abs(e[0])
        deg = math.degrees(math.atan2(e[1], e[0]))
        return f"steepest descent ({deg:+.1f} deg from bisector)", x0, e3, ray_to_free_surface(box, x0, e3)
    # uniform: candidates = inward normals of every face, started at the face centre
    cands = []
    for ax, (lo, hi) in enumerate(box):
        for side, x in ((1.0, lo), (-1.0, hi)):
            x0 = np.array([(l + h) / 2 for l, h in box]); x0[ax] = x
            e = np.zeros(len(box)); e[ax] = side
            cands.append((ray_to_free_surface(box, x0, e), ax))
    W, ax = min(cands)
    return f"uniform field: shortest free path ({'width' if ax == 0 else 'thickness'})", None, None, W


class CrackWalkerPath(CrackWalker):
    """CrackWalker + finite width from the automatic path rule (crack_path), same SENT factor for every specimen:
    sigma_eq = sigma_K(a) * F_SENT(a / W) / 1.12. No per-specimen crack configuration."""

    def __init__(self, rows, t_plain=3.0, force_W=None):
        super().__init__(rows)
        self.W = {}
        for n in self.kinds:
            lab, _, _, W = crack_path(n, t_plain)
            if force_W and n in force_W:                       # diagnostic override, labelled
                lab, W = f"forced W (diagnostic)", force_W[n]
            self.W[n] = W
            self.path = getattr(self, "path", {}); self.path[n] = (lab, W)

    def keff_a(self, a):
        key = ("path", round(a, 6))
        if key not in self._cache:
            k = np.ones(len(self.s))
            for n in self.kinds:
                f = float(f_sent(a / self.W[n])) / 1.12
                if n == "plain":
                    k[self.plain] = f
                else:
                    p = M._prof(n)
                    k[self.notch == n] = crack_stress(lambda x: p(_refl(x)), a) * f
            self._cache = {key: k}
        return self._cache[key]
