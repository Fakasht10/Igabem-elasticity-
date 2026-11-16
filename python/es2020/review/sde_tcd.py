import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Independent check of the latent-damage SDE (es2020.sde) and TCD-based variants. Post-processing of stored IgBEM
bisector fields only. Run: see README.md in this folder.

theta common: sigma_Z (frozen OU, tau = 1e8 cycles fixed, as the SDE thread's posterior), E' = 3479 MPa fixed.
support: 'grad'  sigma_e / (1 + chi L / 2)                  (SDE thread)
         'tcd'   sigma_1 at depth L/2 on the steepest-descent path, constant L
         'tcdN'  same with L(N) = L5 (N/1e5)^B, solved implicitly with the life
damage : 'ro'    Neuber + Masing + Lemaitre loop integral (SDE thread): K', n', sigma_D, s
         'wb'    no plasticity: c = (sigma_D / (sigma_eff ((1-R)/2)^gamma))^k   (stress-driven damage, Walker)
plain  : 'tied'  plain pool on the same curve;  'free' plain stress scaled by exp(b) (batch constant)
"""
import math, sys, json
import numpy as np
from scipy.optimize import minimize
from es2020 import sde, methods as M, paper

E0, TAU = 3479.0, 1e8
rows = sde.dataset()
notch = np.array([r["notch"] for r in rows]); R = np.array([r["R"] for r in rows]); S = np.array([r["s_max"] for r in rows])
lnN = np.log([r["N"] for r in rows]); ro = np.array([r["runout"] for r in rows]); isplain = notch == "plain"
PROF = {n: M._prof(n) for n in paper.NOTCHES}
CHI = {n: M.chi(n) for n in paper.NOTCHES}

def kprof(n, d):
    p = PROF[n]; return np.interp(sde._refl(d), p.d, p.s)

def keff(sup, L, lnNq=None, B=0.0):
    k = np.ones(len(S))
    for n in paper.NOTCHES:
        m = notch == n
        if sup == "grad": k[m] = PROF[n].s[0] / (1 + CHI[n] * L / 2)
        elif sup == "tcd": k[m] = kprof(n, L / 2)
        elif sup == "tcdN": k[m] = kprof(n, 0.5 * L * np.exp(B * (lnNq[m] - math.log(1e5))))
    return k

class V:
    def __init__(self, sup, dmg, plain="tied", fixed=None, pen=None):
        self.sup, self.dmg, self.plain, self.fixed, self.pen = sup, dmg, plain, fixed or {}, pen
        self.names = (["lnK", "n", "lnsD", "s"] if dmg == "ro" else ["lnsD", "k", "gam"]) + ["lnsZ"] + \
                     ([] if sup == "peak" else ["lnL"]) + (["B"] if sup == "tcdN" else []) + (["b"] if plain == "free" else [])
        self.free = [n for n in self.names if n not in self.fixed]
    def par(self, x):
        p = dict(self.fixed); p.update(zip(self.free, x)); return p
    def ok(self, p):
        if self.dmg == "ro" and not (0.01 <= p["n"] <= 1 and p["s"] >= 0 and -1 < p["lnK"] < 16): return False
        if self.dmg == "wb" and not (1 < p["k"] < 40 and 0 <= p["gam"] <= 1): return False
        if "lnL" in p and not (math.log(1e-3) <= p["lnL"] <= math.log(10)): return False
        if "B" in p and not (-0.6 <= p["B"] <= 0.0): return False
        return math.log(0.02) <= p["lnsZ"] <= math.log(2.5)
    def lnc_at(self, p, se, Rr, lc):
        if self.dmg == "ro":
            th = np.array([math.log(E0), p["lnK"], p["n"], p["lnsD"], p["s"], p["lnsZ"], math.log(TAU), 0.0])
            return np.log(sde.life_det(se, Rr, th, lc))
        return p["k"] * (p["lnsD"] - np.log(se) - p["gam"] * np.log((1 - Rr) / 2))
    def lnc(self, p, s=None):
        s = S if s is None else s
        s = np.where(isplain, s * math.exp(p.get("b", 0.0)), s)
        L = math.exp(p["lnL"]) if "lnL" in p else 0.0
        if self.sup != "tcdN":
            return self.lnc_at(p, s * keff(self.sup, L), R, isplain)
        lo, hi = np.full(len(s), math.log(1.0)), np.full(len(s), math.log(1e12))   # implicit: ln c(L(N)) = ln N, decreasing gap
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            g = self.lnc_at(p, s * keff("tcdN", L, mid, p["B"]), R, isplain) > mid
            lo, hi = np.where(g, mid, lo), np.where(g, hi, mid)
        return 0.5 * (lo + hi)
    def nll(self, x, mask):
        p = self.par(x)
        if not self.ok(p): return 1e12
        lc = self.lnc(p)
        if not np.all(np.isfinite(lc)): return 1e12
        sZ = math.exp(p["lnsZ"])
        Q = sde.quantiles(np.full(len(lc), sZ), np.exp(lc) / TAU)
        lp, ls = sde.logpdf_cdf(lnN - lc, Q)
        v = -float(np.sum(np.where(ro, ls, lp)[mask]))
        if self.pen: v += self.pen(p)
        return v if np.isfinite(v) else 1e12
    def resid(self, x):
        """log10(S*/S): stress whose median life equals the observed life (the SDE thread's definition)."""
        p = self.par(x); sZ = math.exp(p["lnsZ"]); med = sZ * sZ / 2       # frozen: median ln(N/c) = sigma_Z^2/2
        lo, hi = np.log(S) - 4, np.log(S) + 4
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            longer = self.lnc(p, np.exp(mid)) + med > lnN
            lo, hi = np.where(longer, mid, lo), np.where(longer, hi, mid)
        return np.log10(np.exp(0.5 * (lo + hi)) / S)

def stats(d, m):
    m = m & ~ro; x = lnN[m] / math.log(10); r = d[m]; sl = np.polyfit(x, r, 1)[0]
    return dict(n=int(m.sum()), bias=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))), slope=float(sl),
                tilted=bool(abs(sl * (x.max() - x.min())) > 0.05))

def report(name, v, x, f):
    d = v.resid(x); p = v.par(x)
    out = {k: stats(d, notch == k) for k in ("plain", "sharp", "intermediate", "blunt")}
    out["held"] = stats(d, np.isin(notch, ("intermediate", "blunt")))
    for Rr in (-1.0, 0.0): out[f"held_R{Rr:g}"] = stats(d, np.isin(notch, ("intermediate", "blunt")) & (R == Rr))
    fm = lambda s: f"{s['bias']:+.3f}/{s['rms']:.3f}/{s['slope']:+.3f}{'T' if s['tilted'] else ''}"
    pp = {k: (round(math.exp(val), 3) if k.startswith("ln") else round(val, 3)) for k, val in p.items()}
    if v.dmg == "ro": pp["s02"] = round(math.exp(p["lnK"]) * 0.002 ** p["n"], 1)
    print(f"{name:34s} nfree {len(v.free)} nll {f:8.2f} | FITTED plain {fm(out['plain'])} V {fm(out['sharp'])} | HELD U1 {fm(out['intermediate'])} U3 {fm(out['blunt'])} both {out['held']['rms']:.3f} | held bias R-1 {out['held_R-1']['bias']:+.3f} R0 {out['held_R0']['bias']:+.3f} | {pp}", flush=True)
    return dict(name=name, par=pp, nll=f, nfree=len(v.free), **out)

def fit(v, starts, mask):
    best = None
    for x0 in starts:
        o = minimize(v.nll, x0, args=(mask,), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-5, fatol=1e-6))
        o = minimize(v.nll, o.x, args=(mask,), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
        if best is None or o.fun < best.fun: best = o
    return best.x, best.fun

if __name__ == "__main__":
    fitmask = np.isin(notch, ("plain", "sharp"))
    res = []
    ro0 = [math.log(84), 0.32, math.log(73), 1.34, math.log(1.4)]
    wb0 = [math.log(60), 6.0, 0.4, math.log(1.4)]
    # 0. the SDE thread's posterior centre, as reported (no fit)
    v = V("grad", "ro"); x = np.array(ro0 + [math.log(0.58)]); res.append(report("0 SDE thread posterior centre", v, x, v.nll(x, fitmask)))
    todo = [
        ("1 grad support + RO (MAP refit)", V("grad", "ro"), [ro0 + [math.log(0.58)], [math.log(40), 0.3, math.log(60), 1.5, math.log(1.2), math.log(0.3)]]),
        ("2 grad + RO, sigma_0.2 >= 41.7", V("grad", "ro", pen=lambda p: 1e3 * max(0.0, math.log(41.7) - (p["lnK"] + p["n"] * math.log(0.002))) ** 2), [[math.log(400), 0.3, math.log(73), 1.34, math.log(1.4), math.log(0.58)], [math.log(1000), 0.2, math.log(60), 2.5, math.log(1.3), math.log(1.0)]]),
        ("3 TCD point const L + RO", V("tcd", "ro"), [ro0 + [math.log(0.6)], ro0 + [math.log(1.5)]]),
        ("4 TCD point L(N) + RO", V("tcdN", "ro"), [ro0 + [math.log(1.0), -0.2], ro0 + [math.log(0.6), -0.05]]),
        ("5 grad + Walker-Basquin", V("grad", "wb"), [wb0 + [math.log(0.58)], wb0 + [math.log(2.0)]]),
        ("6 TCD const L + Walker-Basquin", V("tcd", "wb"), [wb0 + [math.log(0.6)], wb0 + [math.log(1.5)]]),
        ("7 TCD L(N) + Walker-Basquin", V("tcdN", "wb"), [wb0 + [math.log(1.0), -0.2], wb0 + [math.log(0.6), -0.05]]),
        ("8 TCD L=0.4 fixed, plain free, WB", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.4))), [wb0 + [0.0], wb0 + [0.3]]),
        ("9 TCD L=0.4 fixed, plain free, RO", V("tcd", "ro", "free", fixed=dict(lnL=math.log(0.4))), [ro0 + [0.0], ro0 + [0.3]]),
    ]
    for name, v, starts in todo:
        x, f = fit(v, [np.array(s) for s in starts], fitmask); res.append(report(name, v, x, f))
    json.dump(res, open(HERE + "sde_tcd_results.json", "w"), indent=1)
