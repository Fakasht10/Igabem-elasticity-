"""Rank defect / size-effect models that could replace the free plain factor. Post-processing only.
Common damage law for every row: ln c = k (ln sD - ln(S F) - gam ln((1-R)/2)), ln N ~ Normal(ln c, sZ) with run-outs censored (frozen lognormal).
Only F (structural map from net stress to effective stress) differs. L = 0.4 mm fixed where a TCD point is used.
ES  = Ezeh-Susmel 2020: fit own-lab plain (125, fig1_source_match.json) + sharp V; hold out U r1, U r3 (Figs 5-7). Stored IgBEM fields.
YS  = Yaren-Susmel 2025: fit all plain + all V (5 in-fill levels); hold out U r1, U r3 at every level. 2D FE profiles + ys_wl_fields.npz.
Residual = log10(S*/S) (stress giving the observed life as median / applied). AIC = 2 nll + 2 p on the fitted set."""
import math, json, csv, sys, numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
L0 = 0.4
def W(a, r): return (a + r) / np.sqrt(r * (2 * a + r))          # Westergaard amplification at distance r ahead of a crack of half-length a
class DS: pass
def load_es():
    from sde_tcd import notch, R, S, lnN, ro, isplain, PROF, kprof, rows
    from es2020 import methods as M
    mt = json.load(open("/mnt/project-files/review/fig1_source_match.json")); hit = np.zeros(len(S))
    for i in np.where(isplain)[0]:
        hit[i] = [m for m in mt if m["R"] == R[i] and abs(math.log(m["N"]) - lnN[i]) < 2e-3 and abs(m["s"] * 42.9 / S[i] - 1) < 2e-3][0]["hit"]
    keep = ~(isplain & np.isin(R, (-1, 0)) & (hit > 0.3))         # drop other-lab and ambiguous plain points
    d = DS(); d.name = "ES"; nm = {"plain": "plain", "sharp": "V", "intermediate": "U1", "blunt": "U3"}
    d.g = np.array([nm[x] for x in notch[keep]]); d.R, d.S, d.lnN, d.ro = R[keep], S[keep], lnN[keep], ro[keep]
    d.dv = np.zeros(keep.sum()); d.phi = np.ones(keep.sum()); d.lev = np.zeros(keep.sum(), int)
    inv = {v: k for k, v in nm.items()}
    d.b = 2.997                                                   # ligament half-width (net 6 mm): fields are clipped there, never mirrored
    d.K = lambda g, x: np.interp(np.clip(np.asarray(x, float), 0.0, d.b), PROF[inv[g]].d, PROF[inv[g]].s)
    d.vol = {g: (np.asarray(M.fields()[inv[g]]["volume"]["w"]), np.asarray(M.fields()[inv[g]]["volume"]["s1"])) for g in ("V", "U1", "U3")}
    d.vol["plain"] = None                                          # plain geometry not reported: effective volume is a free constant
    d.area = {g: M.fields()[inv[g]]["area"] for g in ("V", "U1", "U3")}
    d.fit = np.isin(d.g, ("plain", "V")); return d
def load_ys():
    Y = "/mnt/project-files/yaren-susmel-2025/"
    rd = lambda f: [r for r in csv.DictReader(l for l in open(Y + f) if not l.startswith("#"))]
    DV = {("plain", int(r["infill_pct"]), int(r["raster_deg"])): (0.0 if r["dv_mm"] == "-" else float(r["dv_mm"])) for r in rd("table1_plain.csv")}
    for r in rd("table2_notched.csv"): DV[({"V": "V", "U": "U"}[r["notch"]] + ("" if r["notch"] == "V" else r["root_radius_mm"]), int(r["infill_pct"]), 0)] = 0.0 if r["dv_mm"] == "-" else float(r["dv_mm"])
    nm = {"plain": "plain", "V_r0.15": "V", "U_r1": "U1", "U_r3": "U3"}; S = rd("specimens_ijf2025.csv")
    d = DS(); d.name = "YS"; d.g = np.array([nm[r["geometry"]] for r in S]); d.R = np.array([float(r["R"]) for r in S]); d.S = np.array([float(r["sigma_max_MPa"]) for r in S])
    d.lnN = np.log([float(r["N"]) for r in S]); d.ro = np.array([r["runout"] == "True" for r in S]); inf = np.array([int(r["infill_pct"]) for r in S])
    d.dv = np.array([DV[(g, i, int(r["raster_deg"]))] for g, i, r in zip(d.g, inf, S)]); d.phi = inf / 100.0; d.lev = np.array([{100: 0, 80: 1, 60: 2, 40: 3, 20: 4}[i] for i in inf])
    P = json.load(open(Y + "profiles.json")); pr = {nm[k]: (np.array(v[-1]["r"]), np.array(v[-1]["K"])) for k, v in P.items()}
    d.b = 6.0                                                     # ligament half-width (net 12 mm)
    d.K = lambda g, x: np.interp(np.clip(np.asarray(x, float), 0.0, d.b), *pr[g])
    z = np.load("/mnt/project-files/review/defects/ys_wl_fields.npz"); d.vol = {nm[k]: (z[k + "_w"], z[k + "_s"]) for k in nm}
    d.area = None; d.fit = np.isin(d.g, ("plain", "V")); return d

_GL_X, _GL_W = np.polynomial.legendre.leggauss(128)
_GL_TH, _GL_WT = (_GL_X + 1) * math.pi / 4, _GL_W / 2         # theta in (0, pi/2), weights summing to 1
def wf_mean(d, g, a):                                              # edge/embedded-crack weight average of the uncracked field over the crack [0, a]
    """(2/pi) int_0^{pi/2} K(a sin th) dth, 128-point Gauss-Legendre (the old 24-point midpoint rule was 2-3 % low for V)."""
    return d.K(g, np.outer(np.atleast_1d(a), np.sin(_GL_TH))) @ _GL_WT
FINITE_WIDTH = False   # 2026-11-16: the Irwin factor costs ~34 AIC on the YS fitted set (fixed_field.json); off by default, cap a + L/2 < b kept
def Y_fw(a, b):
    """Finite-ligament factor of two symmetric cracks of depth a growing towards each other across a ligament of half-width b
    (Irwin tangent form; infinite as a -> b, i.e. net-section collapse). The crack is inadmissible at a >= b."""
    x = 0.5 * math.pi * a / b
    return math.sqrt(math.tan(x) / x) if x > 1e-9 else 1.0
def per_group(d, fn):
    F = np.ones(len(d.S))
    for g in ("plain", "V", "U1", "U3"):
        for lv in np.unique(d.lev):
            m = (d.g == g) & (d.lev == lv)
            if m.any():
                for dv in np.unique(d.dv[m]):
                    mm = m & (d.dv == dv); F[mm] = fn(g, dv, d.phi[mm][0], lv)
    return F
def pm(d, g, depth): return 1.0 if g == "plain" and d.name == "ES" else (float(d.K(g, depth)) if g != "plain" else 1.0)
def wl(d, g, m, lnVp):
    if d.vol[g] is None: return math.exp(lnVp / m)
    w, s = d.vol[g]; return float(np.sum(w * s ** m)) ** (1 / m)
def k_area(d, g, L):
    if g == "plain": return 1.0
    a = d.area[g]; r, wth, s = np.asarray(a["r"]), np.asarray(a["wth"]), np.asarray(a["s1"]); q = s @ wth; l = min(L, r[-1])
    rr = np.concatenate([r[r < l], [l]]); qq = np.interp(rr, r, q); dd = np.diff(rr); w = np.zeros_like(rr); w[:-1] += dd / 2; w[1:] += dd / 2
    return 4 / (math.pi * l * l) * float(np.sum(w * rr * qq))
# model = (extra parameter names, starts, bounds, F builder). uni = uniform in-fill knock-down phi^-n (YS only, parameter n).
def models(d):
    ys = d.name == "YS"; M = {}
    aeq = (lambda p, dv: p["a0"] + p["kap"] * dv) if ys else (lambda p, dv: p["a0"])
    crk = (["a0", "kap"], [[0.5, 3.0], [0.05, 8.0], [1.5, 1.0]], dict(a0=(0, 5.5), kap=(0, 30))) if ys else (["a0"], [[0.3], [1.2], [2.2]], dict(a0=(0, 2.9)))
    M["net stress"] = ([], [[]], {}, lambda p: per_group(d, lambda g, dv, ph, lv: 1.0))
    M["TCD point, no defect"] = ([], [[]], {}, lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2)))
    if ys: M["free plain factor per in-fill (reference)"] = (["b0", "b1", "b2", "b3", "b4"], [[0, .3, .3, .5, .5], [.5, .8, .8, 1, 1]], {k: (-2, 3) for k in ("b0", "b1", "b2", "b3", "b4")},
                                              lambda p: per_group(d, lambda g, dv, ph, lv: math.exp(p["b%d" % lv]) if g == "plain" else pm(d, g, L0 / 2)))
    else: M["free plain factor (reference)"] = (["b"], [[0.0], [0.7]], dict(b=(-2, 3)), lambda p: per_group(d, lambda g, dv, ph, lv: math.exp(p["b"]) if g == "plain" else pm(d, g, L0 / 2)))
    if ys:
        M["homogenised crack LM, a = kap dv (paper)"] = (["kap"], [[3.0], [9.0]], dict(kap=(0, 60)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * math.sqrt((p["kap"] * dv + L0) / L0)))
        M["homogenised crack PM, a = kap dv (paper)"] = (["kap"], [[3.0], [9.0]], dict(kap=(0, 60)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * W(p["kap"] * dv, L0 / 2)))
        def fw(a):                                                 # centre crack in the 12 mm net width, secant finite-width correction
            a = min(a, 5.7); return a / math.cos(math.pi * a / 12.0)
        M["homogenised crack, finite width"] = (["kap"], [[3.0], [6.0]], dict(kap=(0, 60)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * math.sqrt((fw(p["kap"] * dv) + L0) / L0)))
        M["homogenised crack, nonlinear a = kap dv^q"] = (["kap", "q"], [[3.0, 1.0], [6.0, 0.5]], dict(kap=(0, 60), q=(0.05, 3)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * math.sqrt((p["kap"] * dv ** p["q"] + L0) / L0)))
        M["Murakami sqrt-area, uniform (dv/d0)^(1/6)"] = (["d0"], [[0.05], [0.2]], dict(d0=(1e-3, 2)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * max(1.0, (dv / p["d0"])) ** (1 / 6)))
        M["porosity Basquin, uniform phi^-n"] = (["n"], [[1.0], [0.5]], dict(n=(0, 5)), lambda p: per_group(d, lambda g, dv, ph, lv: pm(d, g, L0 / 2) * ph ** -p["n"]))
    uni = (lambda p, ph: ph ** -p["n"]) if ys else (lambda p, ph: 1.0)
    un, us, ub = (["n"], [0.7], dict(n=(0, 5))) if ys else ([], [], {})
    def crack_local(p):
        def f(g, dv, ph, lv):
            a = aeq(p, dv); r = L0 / 2
            if a + r >= d.b: return math.inf                       # crack (plus process zone) cannot leave the ligament
            Wf = W(a, r) * (Y_fw(a, d.b) if FINITE_WIDTH else 1.0)
            if g == "plain": return uni(p, ph) * Wf
            return uni(p, ph) * (float(d.K(g, a + r)) + (Wf - 1) * float(wf_mean(d, g, a)[0]))
        return per_group(d, f)
    M["crack superposed on local field (my version)"] = (crk[0] + un, [s + us for s in crk[1]], {**crk[2], **ub}, crack_local)
    lq = (["l0", "kl"], [[0.4, 1.0], [1.5, 0.2], [0.4, 4.0]], dict(l0=(0.02, 11), kl=(0, 20))) if ys else (["l0"], [[0.6], [1.5], [3.0]], dict(l0=(0.02, 5.9)))
    leff = (lambda p, dv: p["l0"] + p["kl"] * dv) if ys else (lambda p, dv: p["l0"])
    M["TCD point, critical distance grows with void (L free)"] = (lq[0] + un, [s + us for s in lq[1]], {**lq[2], **ub}, lambda p: per_group(d, lambda g, dv, ph, lv: uni(p, ph) * pm(d, g, leff(p, dv) / 2)))
    M["TCD line, critical distance grows with void (L free)"] = (lq[0] + un, [s + us for s in lq[1]], {**lq[2], **ub},
        lambda p: per_group(d, lambda g, dv, ph, lv: uni(p, ph) * (1.0 if g == "plain" else float(np.mean(d.K(g, np.linspace(0, 2 * leff(p, dv), 200)))))))
    if not ys: M["TCD area (L free)"] = (["l0"], [[0.6], [1.5], [3.0]], dict(l0=(0.05, 3)), lambda p: per_group(d, lambda g, dv, ph, lv: k_area(d, g, p["l0"])))
    vp = ([], [], {}) if ys else (["lnVp"], [7.0], dict(lnVp=(-5, 40)))
    M["weakest link, volume (m free)"] = (["m"] + vp[0] + un, [[5.0] + vp[1] + us, [10.0] + vp[1] + us, [3.0] + vp[1] + us], {**dict(m=(1.5, 60)), **vp[2], **ub},
        lambda p: per_group(d, lambda g, dv, ph, lv: uni(p, ph) * wl(d, g, p["m"], p.get("lnVp", 0))))
    M["weakest link, volume, m tied to scatter"] = (vp[0] + un, [vp[1] + us], {**vp[2], **ub}, "strict")
    if ys:
        M["weakest link, m falls with void (m0/(1+c dv))"] = (["m", "c"] + un, [[6.0, 0.5] + us, [10.0, 2.0] + us], {**dict(m=(1.5, 60), c=(0, 30)), **ub},
            lambda p: per_group(d, lambda g, dv, ph, lv: uni(p, ph) * wl(d, g, max(1.5, p["m"] / (1 + p["c"] * dv)), 0)))
    return M

def run(d, name, spec, verbose=True):
    ex, starts, bnd, build = spec; base = [math.log(60 if d.name == "ES" else 20), 5.0, 0.45, math.log(1.3)]
    def unpack(x): return dict(lnsD=x[0], k=x[1], gam=x[2], lnsZ=x[3], **dict(zip(ex, x[4:])))
    def lnc(p):
        if build == "strict":
            m = max(1.5, 1.2825 * p["k"] / math.exp(p["lnsZ"]))          # Weibull modulus of strength from the life scatter (Gumbel sd pi/sqrt6 = 1.2825/m)
            F = per_group(d, lambda g, dv, ph, lv: (ph ** -p["n"] if "n" in p else 1.0) * wl(d, g, m, p.get("lnVp", 0)))
        else: F = build(p)
        return p["k"] * (p["lnsD"] - np.log(d.S * F) - p["gam"] * np.log((1 - d.R) / 2))
    def nll(x):
        p = unpack(x)
        if not (1 < p["k"] < 40 and 0 <= p["gam"] <= 1 and -4 < p["lnsZ"] < 1.5): return 1e12
        for k, (lo, hi) in bnd.items():
            if not lo <= p[k] <= hi: return 1e12
        c = lnc(p)
        if not np.all(np.isfinite(c[d.fit])): return 1e12
        z = (d.lnN - c) / math.exp(p["lnsZ"])
        ll = np.where(d.ro, norm.logsf(z), norm.logpdf(z) - p["lnsZ"])
        return -float(ll[d.fit].sum())
    best = None
    for s in starts:
        for b in (base, [base[0] + 0.7, 4.0, 0.3, math.log(1.0)]):
            o = minimize(nll, np.array(b + list(s)), method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-5, fatol=1e-7))
            o = minimize(nll, o.x, method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-6, fatol=1e-9))
            if best is None or o.fun < best.fun: best = o
    p = unpack(best.x); res = (lnc(p) - d.lnN) / (p["k"] * math.log(10))
    def st(m):
        m = m & ~d.ro
        if m.sum() < 3: return None
        x = d.lnN[m] / math.log(10); r = res[m]; sl = float(np.polyfit(x, r, 1)[0])
        return dict(n=int(m.sum()), bias=float(r.mean()), rms=float(np.sqrt((r ** 2).mean())), slope=sl, tilted=bool(abs(sl * (x.max() - x.min())) > 0.05))
    out = dict(model=name, data=d.name, p=len(best.x), nll=float(best.fun), aic=float(2 * best.fun + 2 * len(best.x)), par={k: float(v) for k, v in p.items()})
    for g in ("plain", "V", "U1", "U3"): out[g] = st(d.g == g)
    out["fit"] = st(d.fit); out["held"] = st(~d.fit)
    if d.name == "YS":
        out["held_by_level"] = {int(100 - 20 * lv): st(~d.fit & (d.lev == lv)) for lv in range(5)}; out["plain_by_level"] = {int(100 - 20 * lv): st((d.g == "plain") & (d.lev == lv)) for lv in range(5)}
    # implied K_f = F(notch)/F(plain) at the fitted constants (100 % level)
    if build != "strict":
        F = build(p); kf = {}
        for g in ("V", "U1", "U3"):
            a, b = F[(d.g == g) & (d.lev == 0)], F[(d.g == "plain") & (d.lev == 0)]
            kf[g] = float(a[0] / b[0])
        out["Kf"] = kf
    if verbose:
        f = lambda s: "%+.3f/%.3f/%+.3f%s" % (s["bias"], s["rms"], s["slope"], "T" if s["tilted"] else "")
        print("%s | %-52s p %d AIC %7.1f | FIT plain %s V %s | HELD U1 %s U3 %s all %.3f | Kf %s | %s" % (d.name, name, out["p"], out["aic"], f(out["plain"]), f(out["V"]), f(out["U1"]), f(out["U3"]), out["held"]["rms"],
              {k: round(v, 2) for k, v in out.get("Kf", {}).items()}, {k: round(v, 3) for k, v in p.items()}), flush=True)
    return out
if __name__ == "__main__":
    which = sys.argv[1:] or ["ES", "YS"]; allres = []
    for w in which:
        d = load_es() if w == "ES" else load_ys()
        print(w, "n", len(d.S), "fit", int(d.fit.sum()), "held", int((~d.fit).sum()), "runouts", int(d.ro.sum()), flush=True)
        for name, spec in models(d).items(): allres.append(run(d, name, spec))
        json.dump(allres, open("/mnt/project-files/review/defects/rank_%s.json" % w, "w"), indent=1); allres = []
