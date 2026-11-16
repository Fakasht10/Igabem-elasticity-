"""Bazant statistical fatigue model: finite weakest-link chain of surface RVEs, each with one Gauss-Weibull flaw that
grows by a Paris-Walker law along its own path through the stored elastic field (thread "Build Bazant statistical
fatigue model", 2026-11-16; formulation https://claude.ai/artifact/39ZinRQtVKUUxvMH2ynwGG).

    P1(r)        Gauss core in ln r (width s) + power tail P_gr (r/r_gr)^m below P_gr, m = phi(z_gr)/(s P_gr)
    a0           = a* r^-2                                      (Griffith flaw <-> inert RVE strength)
    ln N_i       = n (ln sigma_D - ln S_eff - ln kappa_i) + ln J(a0),   S_eff = S_max ((1-R)/2)^gamma
    J(a0)        = int_{a0}^{0.7 W} (g(a) sqrt a)^-n da,  g = weight-function sigma_K of the root profile * F_SENT(a/W)
    1 - P_f(N)   = prod_i [1 - P1(rho_i(N))]^nu_i,           nu_i = A_i / l0^2 (surface stress bins, area A_i)

kappa_i = sigma_1(bin i) / sigma_1(root, bisector profile): every bin's path is the root profile scaled by kappa_i
(checked against the true radial line in test_bazant). Plain: uniform field, one bin of the gauge surface.
Departures (labelled): Walker (gamma = 1 is Bazant's Delta K law), m/(n-2) life modulus (inert reference),
Gauss core in ln r, size factor (1 + D0c/D) and Arrhenius absorbed in sigma_D. Post-processing only."""
import math
import numpy as np
from scipy.stats import norm
from scipy.special import expit, logsumexp
from . import sde
from . import methods as M

NAMES = ("sigma_D", "n", "gamma", "a_star", "s_lnr", "P_gr")
LO = np.array([math.log(1.0), 2.05, 0.0, math.log(1e-4), math.log(0.02), -12.0])
HI = np.array([math.log(1e6), 40.0, 1.0, math.log(3.0), math.log(2.0), -0.01])
L0 = 0.4            # RVE size (mm): one shell bead, fixed a priori


def tail_modulus(s, pgr):
    zg = norm.ppf(pgr)
    return norm.pdf(zg) / (s * pgr)


def log_neglog_surv(lnr, s, pgr):
    """ln(-ln(1 - P1)) of the graft, stable in both tails."""
    zg = norm.ppf(pgr); m = norm.pdf(zg) / (s * pgr); lrg = s * zg
    z = lnr / s
    lnP = np.where(lnr < lrg, math.log(pgr) + m * (lnr - lrg), norm.logcdf(z))
    lnS = np.where(lnr < lrg, np.log1p(-np.exp(np.minimum(lnP, math.log(pgr)))), norm.logsf(z))   # ln(1 - P1)
    small = lnP < -18
    out = np.where(small, lnP, np.log(np.maximum(-lnS, 1e-300)))
    return out


class Chain:
    """rows: dict(notch, R, s_max, N, runout). bins: kind -> (kappa array, area array mm^2); W: kind -> free path;
    profiles: kind -> unit-load sigma_1 along the path (None = uniform)."""

    def __init__(self, rows, bins, W, profiles, l0=L0, ng=320, gamma_fixed=None, nu_one=False, dv=None):
        """dv: per-row void size (mm). If given, the median flaw is a* = a_s + kappa_v dv (Yaren-Susmel) and
        theta = (ln sigma_D, n, gamma, ln a_s, ln kappa_v, ln s, logit P_gr)."""
        self.dv = None if dv is None else np.asarray(dv, float)
        self.notch = np.array([r["notch"] for r in rows]); self.R = np.array([r["R"] for r in rows], float)
        self.s = np.array([r["s_max"] for r in rows], float); self.lnN = np.log([r["N"] for r in rows])
        self.ro = np.array([r["runout"] for r in rows], bool); self.kinds = sorted(set(self.notch))
        self.gamma_fixed = gamma_fixed
        self.grid, self.cells, self.lp0 = {}, {}, {}
        for k in self.kinds:
            af = 0.7 * W[k]; a = np.geomspace(1e-4, af, ng); p = profiles[k]
            sk = np.ones_like(a) if p is None else np.array([sde.crack_stress(p, x) for x in a])
            p0 = 1.0 if p is None else float(p(0.0))
            self.lp0[k] = math.log(p0)
            g = sk / p0 * sde.f_sent(a / W[k])              # root-normalised: g(0) = F(0) = 1.12
            self.grid[k] = (np.log(a), np.log(g) + 0.5 * np.log(a))
            kap, A = bins[k]
            kap = np.asarray(kap, float) / p0 if p is not None else np.asarray(kap, float)
            nu = np.ones(1) if nu_one else np.asarray(A, float) / l0 ** 2
            if nu_one:
                kap = np.array([kap.max()])
            self.cells[k] = (np.log(kap), np.log(nu))
        self.idx = {k: np.where(self.notch == k)[0] for k in self.kinds}
        # peak local stress per unit S (for reporting only)
        self.peak = {k: (1.0 if profiles[k] is None else float(np.max(bins[k][0]))) for k in self.kinds}

    def unpack(self, th):
        """Full theta (gamma inserted when fixed). With dv: (lsD, n, g, la_s, lkap, ls, lp)."""
        th = np.array(th, float)
        if self.gamma_fixed is not None:
            th = np.r_[th[:2], self.gamma_fixed, th[2:]]
        return th

    def lna_star(self, th):
        """ln a* per row."""
        th = self.unpack(th)
        if self.dv is None:
            return np.full(len(self.s), th[3])
        return np.log(np.exp(th[3]) + np.exp(th[4]) * self.dv)

    def core(self, th):
        th = self.unpack(th)
        return (th[0], th[1], th[2], th[-2], th[-1])

    def lnJ_table(self, k, n):
        la, lh = self.grid[k]
        li = -n * lh + la
        seg = np.logaddexp(li[1:], li[:-1]) + np.log(np.diff(la) / 2)
        cum = np.logaddexp.accumulate(seg[::-1])[::-1]            # int_{a_i}^{a_f}, i = 0..ng-2
        return la[:-1], cum

    def lnH(self, th, lnN=None, s=None):
        """ln of the cumulative hazard -ln(1 - P_f) per row at lives lnN and stresses s."""
        lsD, n, g, ls, lp = self.core(th)
        las = self.lna_star(th)
        sv, pgr = math.exp(ls), expit(lp)
        lnN = self.lnN if lnN is None else lnN
        S = self.s if s is None else s
        lse = np.log(S) + g * np.log((1 - self.R) / 2)
        out = np.empty(len(S))
        for k in self.kinds:
            ix = self.idx[k]
            la, cum = self.lnJ_table(k, n)
            lk, lnu = self.cells[k]
            y = (lnN[ix] - n * (lsD - lse[ix] - self.lp0[k]))[:, None] + n * lk[None, :]        # ln J(a0) at failure, per cell
            # invert ln J (decreasing in ln a); small-flaw asymptote ln J ~ c + (1 - n/2) ln a below the grid
            lna0 = np.interp(-y, -cum, la)
            lo = y > cum[0]
            lna0 = np.where(lo, la[0] + (y - cum[0]) / (1 - n / 2), lna0)
            lnr = 0.5 * (las[ix][:, None] - lna0)
            v = log_neglog_surv(lnr, sv, pgr) + lnu[None, :]
            out[ix] = logsumexp(v, axis=1)
        return out

    def loglike(self, th, h=1e-3):
        th = np.asarray(th, float)
        lo, hi = self._box()
        if np.any(th < lo) or np.any(th > hi):
            return -np.inf
        l0 = self.lnH(th)
        H = np.exp(l0)
        lp_, lm_ = self.lnH(th, self.lnN + h), self.lnH(th, self.lnN - h)
        dH = (np.exp(lp_) - np.exp(lm_)) / (2 * h)                  # dH/dlnN > 0
        ll = np.where(self.ro, -H, np.log(np.maximum(dH, 1e-300)) - H)
        v = float(np.sum(ll))
        return v if np.isfinite(v) else -np.inf

    def _box(self):
        lo, hi = LO.copy(), HI.copy()
        if self.dv is not None:                      # a_s, kappa_v >= 0 (log scale, tiny lower bound)
            lo = np.r_[lo[:4], math.log(1e-6), lo[4:]]; hi = np.r_[hi[:4], math.log(20.0), hi[4:]]
        if self.gamma_fixed is None:
            return lo, hi
        return np.delete(lo, 2), np.delete(hi, 2)

    def cdf(self, th, s=None, lnN=None):
        return -np.expm1(-np.exp(self.lnH(th, lnN, s)))


def mix_cdf(model, draws, s=None, lnN=None):
    return np.mean([model.cdf(d, s, lnN) for d in draws], axis=0)


def predictive_residual(model, draws, iters=40):
    """log10(S*/S_obs), S* = stress whose posterior-predictive median life equals N_obs; coverage: CDF at obs in
    [0.1, 0.9]."""
    lo = np.log(model.s) - 3.0; hi = np.log(model.s) + 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = mix_cdf(model, draws, s=np.exp(mid)) < 0.5
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    F = mix_cdf(model, draws)
    return dict(dlogS=np.log10(np.exp(0.5 * (lo + hi)) / model.s), F=F, inside=(F >= 0.1) & (F <= 0.9),
                dlnN=np.full(len(F), np.nan))


def band(model, draws, kind, R, svals, q=(0.1, 0.5, 0.9), iters=50, dv=None):
    """Posterior-predictive life quantiles (cycles) for a specimen type at stresses svals."""
    sub = _subset(model, kind, R, svals, dv)
    out = []
    for qq in q:
        lo = np.full(len(svals), math.log(1.0)); hi = np.full(len(svals), math.log(1e12))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            c = mix_cdf(sub, draws, lnN=mid)
            lo = np.where(c < qq, mid, lo); hi = np.where(c < qq, hi, mid)
        out.append(np.exp(0.5 * (lo + hi)))
    return np.array(out)


def _subset(model, kind, R, svals, dv=None):
    import copy
    m = copy.copy(model)
    m.notch = np.array([kind] * len(svals)); m.R = np.full(len(svals), float(R)); m.s = np.asarray(svals, float)
    m.lnN = np.zeros(len(svals)); m.ro = np.zeros(len(svals), bool); m.kinds = [kind]
    if m.dv is not None:
        m.dv = np.full(len(svals), float(dv))
    m.idx = {kind: np.arange(len(svals))}
    return m


# ------------------------------------------------------------------------------------------- Ezeh-Susmel inputs
def es_plain_own(rows, match):
    """Own-lab strict plain rows (review rule, plain_split2.py): R not in (-1, 0), or Fig. 3 marker hit <= 0.3."""
    own = []
    for r in rows:
        if r["notch"] != "plain":
            continue
        if r["R"] not in (-1.0, 0.0):
            own.append(True); continue
        c = [m for m in match if m["R"] == r["R"] and abs(math.log(m["N"] / r["N"])) < 2e-3
             and abs(m["s"] * 42.9 / r["s_max"] - 1) < 2e-3]
        own.append(bool(c) and c[0]["hit"] <= 0.3)
    return own


def es_inputs(gauge, t_plain=3.0, w_plain=6.0, cut=0.3, nb=40):
    """bins, W, profiles for Ezeh-Susmel. Plain gauge: prismatic w x t, length `gauge` (mm, NOT reported:
    sensitivity input), lateral surface 2 (w + t) gauge."""
    prof = {"plain": None}; bins = {"plain": (np.ones(1), np.array([2 * (w_plain + t_plain) * gauge]))}
    W = {"plain": sde.crack_path("plain", t_plain)[3]}
    for n in ("sharp", "intermediate", "blunt"):
        p = M._prof(n)
        prof[n] = (lambda p: (lambda x: p(sde._refl(x))))(p)
        bins[n] = sde.surface_bins(n, nb=nb, cut=cut)
        W[n] = sde.crack_path(n, t_plain)[3]
    return bins, W, prof


# ------------------------------------------------------------------------------------------- Yaren-Susmel inputs
YS = "/mnt/project-files/yaren-susmel-2025/"
YS_KINDS = ("plain", "V_r0.15", "U_r1", "U_r3")


def ys_rows():
    """381 REAL specimens (IJF 2025 Figs 4-7) with the void size dv measured per in-fill (Table 1 plain, Table 2 at
    the notch; 100 % in-fill -> 0)."""
    import csv
    rd = lambda f: [x for x in csv.DictReader(l for l in open(YS + f) if not l.startswith("#"))]
    num = lambda v: float(v) if v not in ("-", "", None) else 0.0
    dvp = {(int(x["infill_pct"]), x["raster_deg"]): num(x["dv_mm"]) for x in rd("table1_plain.csv") if num(x["dv_mm"]) > 0}
    dvn = {}
    for x in rd("table2_notched.csv"):
        g = "V_r0.15" if x["notch"] == "V" else f"U_r{float(x['root_radius_mm']):g}"
        if num(x["dv_mm"]) > 0: dvn[(g, int(x["infill_pct"]))] = num(x["dv_mm"])
    rows = []
    for x in rd("specimens_ijf2025.csv"):
        g, i = x["geometry"], int(x["infill_pct"])
        if i == 100: dv = 0.0
        elif g == "plain": dv = dvp.get((i, x["raster_deg"]), dvp.get((i, "0")))
        else: dv = dvn[(g, i)]
        rows.append(dict(notch=g, R=float(x["R"]), s_max=float(x["sigma_max_MPa"]), N=float(x["N"]),
                         runout=x["runout"] == "True", infill=i, dv=dv, raster=x["raster_deg"]))
    return rows


def ys_inputs(cut=0.3, nb=40, T=5.0):
    """INTERIM fields until the 3D IgBEM surfaces are stored: 2D plane-stress P2 FE (review/ys_wl_fields.npz,
    sigma_1 per unit net stress, w = full-specimen volume weights). Surface cells = the two flat faces
    (area 2 w / T); the notch/edge faces through the thickness are NOT yet included (flagged). Paths: bisector
    profiles (ys profiles.json, finest mesh); plain hourglass path treated as uniform at its own peak."""
    import json, os
    F = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "review", "ys_wl_fields.npz"))
    P = json.load(open(YS + "profiles.json"))
    bins, W, prof = {}, {}, {}
    for k in YS_KINDS:
        w, s1 = F[k + "_w"], F[k + "_s"]
        a = 2 * w / T
        mk = s1 >= cut * s1.max(); s1, a = s1[mk], a[mk]
        e = np.linspace(s1.min(), s1.max() + 1e-12, nb + 1); ix = np.digitize(s1, e) - 1
        A = np.bincount(ix, a, nb); K = np.bincount(ix, a * s1, nb) / np.maximum(A, 1e-300); keep = A > 0
        bins[k] = (K[keep], A[keep])
        if k == "plain":
            prof[k], W[k] = None, T                                     # crack_path rule: thickness 5 < width 12
        else:
            p = P[k][-1]; r, Kp = np.array(p["r"]), np.array(p["K"])
            prof[k] = (lambda r, Kp: (lambda x: np.interp(np.where(x <= 6, x, 12 - np.minimum(x, 12)), r, Kp)))(r, Kp)
            W[k] = 12.0
    return bins, W, prof


def ys_inputs3d(cut=0.3, nb=40):
    """Yaren-Susmel 3D IgBEM surface fields (igabem_fields_3d.json, thread "Extract fatigue data from homogenised-
    material paper", branch claude/project-thread-ramlnd a67e6c3): cells = edge faces (notch/hourglass face, flank,
    side edge, x thickness) + both flat faces, full-specimen dA, sigma_1 per unit net stress. Path: mid-plane
    bisector profile (reflected at the centre line, 6 mm), W = 12 mm (steepest descent runs across the width for
    every specimen, including the hourglass, whose field is not uniform: Kt 1.217)."""
    import json
    d = json.load(open(YS + "igabem_fields_3d.json"))["specimens"]
    bins, W, prof = {}, {}, {}
    for k in YS_KINDS:
        f = d[k]["surface"]
        s1 = np.r_[np.ravel(f["edge_s1"]), np.ravel(f["side_s1"])]
        a = np.r_[np.ravel(f["edge_dA"]), np.ravel(f["side_dA"])]
        mk = s1 >= cut * s1.max(); s1, a = s1[mk], a[mk]
        e = np.linspace(s1.min(), s1.max() + 1e-12, nb + 1); ix = np.digitize(s1, e) - 1
        A = np.bincount(ix, a, nb); K = np.bincount(ix, a * s1, nb) / np.maximum(A, 1e-300); keep = A > 0
        bins[k] = (K[keep], A[keep])
        b = d[k]["bisector"]; z = np.asarray(b["z"]); dd = np.asarray(b["d"]); S = np.asarray(b["s1"])
        sp = S[int(np.argmin(np.abs(z - 2.5)))]
        L = float(dd.max())
        prof[k] = (lambda dd, sp, L: (lambda x: np.interp(np.where(x <= L, x, 2 * L - np.minimum(x, 2 * L)), dd, sp)))(dd, sp, L)
        W[k] = 2 * L
    return bins, W, prof
