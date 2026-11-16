"""Independent refit (audit2). Own field interpolation, quadrature (exact piecewise-linear weight-function mean), likelihood, optimiser."""
import sys, math, json, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
for p_ in ('/home/user/wt_uxm/python', '/mnt/project-files/review', '/mnt/project-files/review/defects', '/mnt/project-files/yaren-susmel-2025'): sys.path.insert(0, p_)
L0 = 0.4; r0 = L0 / 2; LN10 = math.log(10)
C = '/home/user/igabem-elasticity-/cache/ys2025/'; Y = '/mnt/project-files/yaren-susmel-2025/'
class D: pass
# ---------- exact (2/pi) int_0^a K(x)/sqrt(a^2-x^2) dx for piecewise-linear K, clamped beyond the table ----------
def wfm(xs, ys, a):
    br = np.unique(np.r_[0.0, xs[(xs > 0) & (xs < a)], a]); x0, x1 = br[:-1], br[1:]
    k0, k1 = np.interp(x0, xs, ys), np.interp(x1, xs, ys); be = (k1 - k0) / (x1 - x0); al = k0 - be * x0
    t = al * (np.arcsin(np.minimum(x1 / a, 1)) - np.arcsin(x0 / a)) - be * (np.sqrt(np.maximum(a * a - x1 * x1, 0)) - np.sqrt(a * a - x0 * x0))
    return 2 / math.pi * t.sum()
def Wst(a, r): return (a + r) / math.sqrt(r * (2 * a + r))
def Yfw(a, b): x = 0.5 * math.pi * a / b; return math.sqrt(math.tan(x) / x) if x > 1e-12 else 1.0
# ---------- YS loader (own, pandas) ----------
def load_ys():
    S_ = pd.read_csv(Y + 'specimens_ijf2025.csv', comment='#'); t1 = pd.read_csv(Y + 'table1_plain.csv', comment='#'); t2 = pd.read_csv(Y + 'table2_notched.csv', comment='#')
    dv = {}
    for _, r in t1.iterrows(): dv[('plain', r.infill_pct, r.raster_deg)] = 0.0 if r.dv_mm == '-' else float(r.dv_mm)
    for _, r in t2.iterrows():
        g = 'V' if r.notch == 'V' else ('U1' if float(r.root_radius_mm) == 1 else 'U3'); dv[(g, r.infill_pct, 0)] = 0.0 if r.dv_mm == '-' else float(r.dv_mm)
    GM = {'plain': 'plain', 'V_r0.15': 'V', 'U_r1': 'U1', 'U_r3': 'U3'}
    d = D(); d.name = 'YS'; d.g = S_.geometry.map(GM).values; d.R = S_.R.values.astype(float); d.S = S_.sigma_max_MPa.values.astype(float)
    d.lnN = np.log(S_.N.values.astype(float)); d.ro = S_.runout.astype(str).values == 'True'; d.inf = S_.infill_pct.values
    d.dv = np.array([dv[(g, i, rr)] for g, i, rr in zip(d.g, d.inf, S_.raster_deg.values)]); d.phi = d.inf / 100.0
    d.fit = np.isin(d.g, ['plain', 'V']); d.b = 6.0
    d.fld = {}
    for k, g in GM.items():
        z = np.load(C + f'profile_{k}_h0.npz'); assert abs(z['z'][-1] - 2.5) < 1e-12; d.fld[g] = (z['d'].astype(float), z['s1'][-1].astype(float))
    t = np.load(C + 'triax_bisector.npz'); HX = {}
    for k, g in GM.items():
        assert t[k + '_z'][0] == 2.5; HX[g] = max(float(t[k + '_h'][0][0]) - 1 / 3, 0)
    d.Dh = np.array([HX[g] for g in d.g]); d.comp = np.maximum(-d.R, 0); d.bmax = 1 / d.Dh.max()
    return d
def load_es(pooled=False):
    import rank, sde_tcd as s
    if pooled:
        import reconcile_pool as rp; e = rp.pooled()
    else: e = rank.load_es()
    d = D(); d.name = 'ESpool' if pooled else 'ES'
    for k in ('g', 'R', 'S', 'lnN', 'ro', 'fit'): setattr(d, k, np.asarray(getattr(e, k)))
    d.dv = np.zeros(len(d.S)); d.phi = np.ones(len(d.S)); d.b = 2.997
    inv = {'V': 'sharp', 'U1': 'intermediate', 'U3': 'blunt'}
    d.fld = {g: (np.asarray(s.PROF[k].d, float), np.asarray(s.PROF[k].s, float)) for g, k in inv.items()}
    return d
def f_sent(al): al = min(al, 0.9); return 1.12 - 0.231 * al + 10.55 * al ** 2 - 21.72 * al ** 3 + 30.39 * al ** 4
# ---------- models: return (names, bounds, Fbuilder) ----------
def groups(d):
    keys = sorted(set(zip(d.g, d.dv, d.phi))); return [(k, np.where((d.g == k[0]) & (d.dv == k[1]) & (d.phi == k[2]))[0]) for k in keys]
def ys_crack(d, triax=False, fw=False):
    G = groups(d); dvmax = d.dv.max()
    names = ['lnsD', 'k', 'gam', 'lnsZ', 'a0', 'u', 'n'] + (['b'] if triax else [])
    B = dict(lnsD=(-5, 15), k=(1, 40), gam=(0, 1), lnsZ=(-4, 1.5), a0=(0, 5.5), u=(0, 1), n=(0, 5), b=(0, d.bmax))
    # u parametrises kap in [0, min(30, (b - r - a0)/dvmax)) so a_max + r < b is a bound (approached, never reached: u<=1-1e-9)
    def kap_of(p): return p['u'] * min(30.0, (d.b - r0 - p['a0']) / dvmax) * (1 - 1e-9)
    def F(p):
        kap = kap_of(p); out = np.empty(len(d.S))
        for (g, dv, ph), ii in G:
            a = p['a0'] + kap * dv
            if a + r0 >= d.b: out[ii] = np.inf; continue
            Wf = Wst(a, r0) * (Yfw(a, d.b) if fw else 1.0); xs, ys = d.fld[g]
            out[ii] = ph ** -p['n'] * (np.interp(min(a + r0, d.b), xs, ys) + (Wf - 1) * (wfm(xs, ys, a) if a > 0 else np.interp(0, xs, ys)))
        if triax: out = out * (1 + p['b'] * d.Dh * d.comp)
        return out
    return names, B, F, kap_of
def es_recon(d, Wp=3.0, Wn=6.0):
    G = groups(d); names = ['lnsD', 'k', 'gam', 'lnsZ', 'a0']; B = dict(lnsD=(-5, 15), k=(1, 40), gam=(0, 1), lnsZ=(-4, 1.5), a0=(1e-3, 2.9))
    def F(p):
        a = p['a0']; out = np.empty(len(d.S))
        for (g, dv, ph), ii in G:
            out[ii] = f_sent(a / Wp) / 1.12 if g == 'plain' else wfm(*d.fld[g], a) * f_sent(a / Wn) / 1.12
        return out
    return names, B, F, None
def make_nll(d, names, F):
    lw = np.log((1 - d.R) / 2)
    def lnc(p): return p['k'] * (p['lnsD'] - np.log(d.S * F(p)) - p['gam'] * lw)
    def nll(x):
        p = dict(zip(names, x)); c = lnc(p)
        if not np.all(np.isfinite(c[d.fit])): return 1e10
        z = (d.lnN - c) / math.exp(p['lnsZ']); ll = np.where(d.ro, norm.logsf(z), norm.logpdf(z) - p['lnsZ'])
        return -float(ll[d.fit].sum())
    return lnc, nll
def fit(d, names, B, F, nstart=30, seed=1):
    lnc, nll = make_nll(d, names, F); rng = np.random.default_rng(seed)
    lo = np.array([B[n][0] for n in names]); hi = np.array([B[n][1] for n in names]); hi_eff = hi.copy()
    box = dict(lnsD=(3, 6), k=(3, 8), gam=(0.05, 0.8), lnsZ=(-0.5, 0.4), a0=(0, 3), u=(0.05, 0.95), n=(0, 1), b=(0, 0.8 * B.get('b', (0, 1))[1]))
    slo = np.array([max(box[n][0], lo[i]) for i, n in enumerate(names)]); shi = np.array([min(box[n][1], hi[i]) for i, n in enumerate(names)])
    if 'a0' in names and d.name.startswith('ES'): i = names.index('a0'); slo[i], shi[i] = 0.01, 2.85
    bnds = list(zip(lo, hi)); res = []
    for s in range(nstart):
        x0 = slo + rng.random(len(names)) * (shi - slo)
        o1 = minimize(nll, x0, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=3000, ftol=1e-13, gtol=1e-8))
        o2 = minimize(nll, o1.x, method='Powell', bounds=bnds, options=dict(maxiter=40000, xtol=1e-9, ftol=1e-13))
        o3 = minimize(nll, o2.x, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=3000, ftol=1e-15, gtol=1e-10))
        res.append(min([o1, o2, o3], key=lambda q: q.fun))
    res.sort(key=lambda q: q.fun); best = res[0]
    # projected gradient at the optimum (central differences)
    x = best.x; gr = np.zeros(len(x))
    for i in range(len(x)):
        h = 1e-6 * max(1, abs(x[i])); xp, xm = x.copy(), x.copy(); xp[i] = min(x[i] + h, hi[i]); xm[i] = max(x[i] - h, lo[i]); gr[i] = (nll(xp) - nll(xm)) / (xp[i] - xm[i])
    fv = np.array([q.fun for q in res])
    return dict(x=x, nll=best.fun, aic=2 * best.fun + 2 * len(x), grad=gr, n_within_0p01=int((fv - fv[0] < 0.01).sum()), nstart=nstart,
                spread=[float(np.quantile(fv, q)) for q in (0, .25, .5)]), lnc, nll
def scores(d, p, lnc):
    lp = lnc(p) / LN10; le = d.lnN / LN10; eN = le - lp; sZ = math.exp(p['lnsZ']) / LN10; hw = norm.ppf(0.9) * sZ; m = ~d.ro; out = {}
    eS = -eN / p['k']    # log10(S*/S): S* = stress whose median life equals the observed life (rank's residual convention)
    for nm, mk in (('fit', d.fit & m), ('held', ~d.fit & m)):
        out[nm] = dict(n=int(mk.sum()), rms_S=float(np.sqrt(np.mean(eS[mk] ** 2))), bias_S=float(eS[mk].mean()), rms_N=float(np.sqrt(np.mean(eN[mk] ** 2))), bias_N=float(eN[mk].mean()),
                       slope_Nexp=float(np.polyfit(le[mk], eN[mk], 1)[0]), slope_Npred=float(np.polyfit(lp[mk], eN[mk], 1)[0]), cov=float(100 * np.mean(np.abs(eN[mk]) <= hw)))
    return out
def run(tag, d, spec, nstart):
    names, B, F, kap_of = spec; r, lnc, nll = fit(d, names, B, F, nstart)
    p = dict(zip(names, r['x'])); par = dict(p)
    if kap_of: par['kap'] = kap_of(p)
    amax = par['a0'] + par.get('kap', 0) * d.dv.max(); amax_fit = par['a0'] + par.get('kap', 0) * d.dv[d.fit].max()
    onb = [n for n in names if (abs(p[n] - B[n][0]) < 1e-4 * (1 + abs(B[n][0])) or abs(p[n] - B[n][1]) < 1e-4 * (1 + abs(B[n][1])))]
    out = dict(tag=tag, aic=r['aic'], nll=r['nll'], par=par, a_max=amax, a_max_fitset=amax_fit, a_max_plus_r_over_b=(amax + r0) / d.b, on_bound=onb,
               grad=dict(zip(names, r['grad'].tolist())), n_within_0p01=r['n_within_0p01'], nstart=nstart, spread=r['spread'], **scores(d, p, lnc))
    print(json.dumps(out, default=float), flush=True); return out
if __name__ == '__main__':
    job = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if job == 'ys_base': d = load_ys(); o = run(job, d, ys_crack(d), ns)
    if job == 'ys_triax': d = load_ys(); o = run(job, d, ys_crack(d, triax=True), ns)
    if job == 'ys_base_fw': d = load_ys(); o = run(job, d, ys_crack(d, fw=True), ns)
    if job == 'ys_triax_fw': d = load_ys(); o = run(job, d, ys_crack(d, triax=True, fw=True), ns)
    if job == 'es_own': d = load_es(); o = run(job, d, es_recon(d), ns)
    if job == 'es_pool': d = load_es(True); o = run(job, d, es_recon(d), ns)
    json.dump(o, open(f'/tmp/claude-0/audit2/res_{job}.json', 'w'), indent=1, default=float)
