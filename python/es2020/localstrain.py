"""Field-based local strain approach: deviatoric Neuber under proportionality with Masing reversals.

After Palchoudhary, Ovalle, Maurel & Kerfriden, CFM 2022 (hal-04373346): the deviatoric stress and strain at a point
stay proportional to the elastic deviator (sigma_d = s sigma_bar_d, eps_d = e eps_bar_d), the hydrostatic part stays
elastic, and Neuber's energy equality on the deviator reduces to ONE scalar equation per point,
(s - s_o)(e - e_o) = (f - f_o)^2, with the origin moved to the last reversal. In von Mises equivalents
(sigma_d : eps_d = sigma_vM eps_eq, elastic eps_eq = sigma_vM / 3G):

    first loading      sigma * eps(sigma)        = sigma_bar^2 / 3G,   eps(s)  = s/3G + (s/K')^(1/n')
    Masing branch      dsigma * deps(dsigma)     = dsigma_bar^2 / 3G,  deps(d) = d/3G + 2 (d/2K')^(1/n')

Input per point: the elastic von Mises stress at peak load (any 3D field, unit load times the load scale) and R.
Cyclic curve E' = E, K', n' (no PLA values published: fitted, a departure). Pure numpy, vectorised over any shape.
"""
import numpy as np

NU = 0.35
E = 3479.0                       # MPa, Ezeh & Susmel 2020 Sec. 2
G3 = 3 * E / (2 * (1 + NU))      # 3G


def _bisect(fun, target, hi, it=64):
    lo = np.zeros_like(hi)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        big = fun(mid) > target
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return 0.5 * (lo + hi)


def eps_mono(s, K, n):
    return s / G3 + (np.maximum(s, 0) / K) ** (1 / n)


def deps_branch(ds, K, n):
    return ds / G3 + 2 * (np.maximum(ds, 0) / (2 * K)) ** (1 / n)


def loop(se_max, R, K, n):
    """Stabilised local loop from the elastic peak von Mises stress se_max (MPa) and load ratio R.
    Returns dict: s_max (local peak), ds (range), de (equivalent strain range), dep (plastic strain range),
    w_h (hysteresis energy per cycle, Masing/Morrow: (1-n)/(1+n) ds dep), swt (s_max * de / 2)."""
    se_max = np.asarray(se_max, float)
    dse = se_max * (1 - np.asarray(R, float))
    s_max = _bisect(lambda s: s * eps_mono(s, K, n), se_max ** 2 / G3, se_max.copy())
    ds = _bisect(lambda d: d * deps_branch(d, K, n), dse ** 2 / G3, np.maximum(dse, 1e-12))
    de = deps_branch(ds, K, n)
    dep = de - ds / G3
    return dict(s_max=s_max, ds=ds, de=de, dep=dep, w_h=(1 - n) / (1 + n) * ds * dep, swt=s_max * de / 2)


def plain_loop(S_max, R, K, n):
    """Uniform plain gauge under load control: NO Neuber, the local stress IS the nominal stress; strains from the
    cyclic curve (first loading to S_max, Masing branch of range S_max (1 - R))."""
    S_max = np.asarray(S_max, float)
    ds = S_max * (1 - np.asarray(R, float))
    de = deps_branch(ds, K, n)
    dep = de - ds / G3
    return dict(s_max=S_max, ds=ds, de=de, dep=dep, w_h=(1 - n) / (1 + n) * ds * dep, swt=S_max * de / 2)


def plain_equivalent(D, R, K, n, param="swt", s_hi=200.0):
    """Nominal plain stress S_max giving the damage parameter D (plain loop, monotone in S)."""
    D = np.asarray(D, float)
    R = np.broadcast_to(np.asarray(R, float), D.shape)
    return _bisect(lambda S: plain_loop(S, R, K, n)[param], D, np.full(D.shape, s_hi))


# ------------------------------------------------------------------------------------------------ Ezeh-Susmel scoring
def _field_points(notch, where, R0=None):
    """Elastic von Mises at unit net stress and weights: 'peak' = bisector root value; 'crescent' = stored SED grid
    points with rc <= R0 + r0 (dA weights)."""
    from . import collapse as C, methods as M
    if where == "peak":
        return np.array([M.measure_profile(notch, "vm").s[0]]), np.ones(1)
    r0, rc, dA, tr2, vm2 = C._sed_grid(notch)
    m = rc <= R0 + r0
    return np.sqrt(vm2[m]), dA[m] / dA[m].sum()


def support(notch, N, c):
    """FKM / Siebel-Stieler support from the field only: 1 + sqrt(c chi L(N)), chi = relative gradient of sigma_1 at
    the hot spot (IgBEM), L(N) = paper Eq. 6. Input is (sigma, grad sigma) at the point; c is a material constant."""
    from . import methods as M, paper
    return 1 + np.sqrt(c * M.chi(notch) * paper.lm_paper(N))


def k_eq(specs, K, n, where="peak", param="swt", R0=None, c=None):
    """Equivalent plain sigma_max / specimen sigma_n,max for each specimen. c: gradient support on the peak."""
    out = np.empty(len(specs))
    groups = {}
    for i, s in enumerate(specs):
        groups.setdefault(s.notch, []).append(i)
    for notch, idx in groups.items():
        v, w = _field_points(notch, where, R0)
        S = np.array([specs[i].s_max for i in idx])
        R = np.array([specs[i].R for i in idx])
        sup = support(notch, np.array([specs[i].N for i in idx]), c)[:, None] if c is not None else 1.0
        D = loop(S[:, None] * v[None, :] / sup, R[:, None], K, n)[param] @ w
        out[idx] = plain_equivalent(D, R, K, n, param) / S
    return out


VARIANTS = {   # name: (where, param, fit R0?)
    "peak_swt": ("peak", "swt", False),
    "grad_swt": ("peak", "swt", "c"),
    "grad_wh": ("peak", "w_h", "c"),
    "crescent_swt": ("crescent", "swt", True),
    "crescent_wh": ("crescent", "w_h", True),
}


def fit_and_score(name, specs=None):
    """K', n' (n' in [0.05, 0.5]) and R0 (crescent variants) fitted on the 47 sharp-V failures (log10 stress residual vs
    plain P50 at the specimen's life), U r1 and U r3 held out. Returns params and per-notch residuals."""
    from scipy.optimize import minimize
    from . import paper
    specs = [s for s in (specs or paper.fig8_specimens()) if not s.runout]
    where, param, fr0 = VARIANTS[name]
    sharp = [s for s in specs if s.notch == "sharp"]

    def unpack(x):
        K = np.exp(x[0])
        n = 0.05 + 0.45 / (1 + np.exp(-x[1]))
        return K, n, (np.exp(x[2]) if fr0 else None)

    def resid(sp, x):
        K, n, R0 = unpack(x)
        k = k_eq(sp, K, n, where, param, None, R0) if fr0 == "c" else k_eq(sp, K, n, where, param, R0)
        return np.log10(k * np.array([s.s_max for s in sp]) / paper.PLAIN(np.array([s.N for s in sp])))

    cost = lambda x: np.sum(resid(sharp, x) ** 2)
    starts = [np.r_[np.log(K0), n0] for K0 in (30, 60, 150) for n0 in (-1.5, 0.0, 1.5)]
    if fr0:
        starts = [np.r_[s, np.log(1.0)] for s in starts]
    best = min((minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=1500))
                for x0 in starts), key=lambda o: o.fun)
    K, n, R0 = unpack(best.x)
    best_p = dict(K=K, n=n, c=R0) if fr0 == "c" else dict(K=K, n=n, R0=R0)
    out = {}
    for k in paper.NOTCHES:
        sp = [s for s in specs if s.notch == k]
        out[k] = (np.array([s.N for s in sp]), resid(sp, best.x), np.array([s.R for s in sp]))
    return best_p, out
