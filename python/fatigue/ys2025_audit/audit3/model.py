"""Independent re-implementation of the YS 3D-baseline and saturating-defect fatigue models (audit3)."""
import csv, math, numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
Y = '/mnt/project-files/yaren-susmel-2025/'; C = '/home/user/igabem-elasticity-/cache/ys2025/'
L = 0.4; r0 = L / 2; B = 6.0
GN = {'plain': 'plain', 'V_r0.15': 'V', 'U_r1': 'U1', 'U_r3': 'U3'}
def rows(f): return list(csv.DictReader(l for l in open(Y + f) if not l.startswith('#')))
DV = {}
for r in rows('table1_plain.csv'): DV[('plain', int(r['infill_pct']), int(r['raster_deg']))] = 0.0 if r['dv_mm'] == '-' else float(r['dv_mm'])
for r in rows('table2_notched.csv'):
    g = 'V' if r['notch'] == 'V' else ('U1' if r['root_radius_mm'] == '1' else 'U3')
    DV[(g, int(r['infill_pct']), 0)] = 0.0 if r['dv_mm'] == '-' else float(r['dv_mm'])
S_ = rows('specimens_ijf2025.csv')
g = np.array([GN[r['geometry']] for r in S_]); R = np.array([float(r['R']) for r in S_]); S = np.array([float(r['sigma_max_MPa']) for r in S_])
lnN = np.log([float(r['N']) for r in S_]); ro = np.array([r['runout'] == 'True' for r in S_]); inf = np.array([int(r['infill_pct']) for r in S_])
ras = np.array([int(r['raster_deg']) for r in S_]); dv = np.array([DV[(gg, i, ra if gg == 'plain' else 0)] for gg, i, ra in zip(g, inf, ras)])
phi = inf / 100.0; fit = np.isin(g, ('plain', 'V'))
PR = {}
for k, v in GN.items():
    z = np.load(C + f'profile_{k}_h0.npz'); PR[v] = (z['d'], z['s1'][-1])
def K(gg, x): return np.interp(np.clip(np.asarray(x, float), 0, B), *PR[gg])
gx, gw = np.polynomial.legendre.leggauss(400); TH = (gx + 1) * math.pi / 4; WT = gw / 2
def wfm(gg, a): return float(K(gg, a * np.sin(TH)) @ WT)
def W(a, r): return (a + r) / math.sqrt(r * (2 * a + r))
# unique (geometry, infill, dv) cells
cells = sorted(set(zip(g, inf, dv))); cid = np.array([cells.index(c) for c in zip(g, inf, dv)])
def amap(model, p, d):
    if model == 'base': return p['a0'] + p['kap'] * d
    ac = p['acap']; return p['a0'] + ac * (1 - math.exp(-p['kap'] * d / ac))
def Fcells(model, p, mask=None):
    use = set(cid[mask]) if mask is not None else set(range(len(cells)))
    Fc = np.full(len(cells), np.nan)
    for j in use:
        gg, i, d = cells[j]; a = amap(model, p, d)
        if a + r0 >= B: Fc[j] = np.inf; continue
        Fc[j] = (i / 100) ** -p['n'] * (float(K(gg, a + r0)) + (W(a, r0) - 1) * wfm(gg, a))
    return Fc
def Frow(model, p): return Fcells(model, p)[cid]
GLOB = ['lnsD', 'k', 'gam', 'lnsZ']; GB = dict(lnsD=(1, 8), k=(1, 40), gam=(0, 1), lnsZ=(-4, 1.5))
def names(model, acap_fixed):
    ex = ['a0', 'kap', 'n'] + (['acap'] if (model == 'sat' and acap_fixed is None) else [])
    return GLOB + ex
def bounds(model, acap_fixed):
    b = dict(GB, a0=(0, 5.5), n=(0, 5), kap=(0, 30) if model == 'base' else (0, 60))
    if model == 'sat' and acap_fixed is None: b['acap'] = (0.05, 5.8)
    return [b[n_] for n_ in names(model, acap_fixed)]
def unpack(model, acap_fixed, x):
    p = dict(zip(names(model, acap_fixed), x))
    if model == 'sat' and acap_fixed is not None: p['acap'] = acap_fixed
    return p
def lnc(model, p, F): return p['k'] * (p['lnsD'] - np.log(S * F) - p['gam'] * np.log((1 - R) / 2))
def nll_rows(model, p, mask):
    F = Fcells(model, p, mask)[cid]; c = lnc(model, p, F); s = math.exp(p['lnsZ'])
    z = (lnN - c) / s; ll = np.where(ro, norm.logsf(z), norm.logpdf(z) - p['lnsZ'])
    return -ll, c
def make_nll(model, acap_fixed, mask):
    bd = bounds(model, acap_fixed)
    def f(x):
        for xi, (lo, hi) in zip(x, bd):
            if not lo <= xi <= hi: return 1e12
        p = unpack(model, acap_fixed, x); l, c = nll_rows(model, p, mask)
        if not np.all(np.isfinite(c[mask])): return 1e12
        return float(l[mask].sum())
    return f, bd
def fitmodel(model, acap_fixed=None, mask=fit, nstart=60, seed=0, x0s=()):
    f, bd = make_nll(model, acap_fixed, mask); rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bd]); hi = np.array([b[1] for b in bd])
    # sensible box for random starts
    slo = lo.copy(); shi = hi.copy(); nm = names(model, acap_fixed)
    box = dict(lnsD=(2.5, 5.5), k=(3, 8), gam=(0.2, 0.7), lnsZ=(-0.6, 0.3), a0=(0, 2), kap=(0, 12), n=(0, 1.5), acap=(0.3, 5.5))
    for i, n_ in enumerate(nm): slo[i], shi[i] = box[n_]
    starts = [slo + rng.random(len(lo)) * (shi - slo) for _ in range(nstart)] + [np.array(x) for x in x0s]
    res = []
    for x0 in starts:
        if f(x0) >= 1e11: continue
        o = minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8, adaptive=True))
        res.append(o)
    res.sort(key=lambda o: o.fun); best = None; sols = []
    for o in res[:8]:
        o2 = minimize(f, o.x, method='L-BFGS-B', bounds=[(a + 1e-12, b - 1e-12) for a, b in bd], options=dict(ftol=1e-14, gtol=1e-9, maxiter=2000))
        o3 = minimize(f, o2.x if o2.fun <= o.fun else o.x, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-8, fatol=1e-11, adaptive=True))
        sols.append(o3)
        if best is None or o3.fun < best.fun: best = o3
    spread = [round(o.fun, 3) for o in res[:10]]
    p = unpack(model, acap_fixed, best.x)
    return dict(p=p, nll=best.fun, kpar=len(best.x), aic=2 * best.fun + 2 * len(best.x), x=best.x, spread=spread, bounds=bd, names=nm, f=f)
def stats(model, p, sel=None):
    F = Frow(model, p); se = S * F * ((1 - R) / 2) ** p['gam']
    ep = p['k'] * (p['lnsD'] - np.log(se)) / math.log(10); eN = lnN / math.log(10) - ep; eS = -eN / p['k']
    sZ = math.exp(p['lnsZ']) / math.log(10); m = ~ro; out = {}
    for nm_, msk in (('fit', fit & m), ('held', ~fit & m)):
        x, r = lnN[msk] / math.log(10), eN[msk]
        out[nm_] = dict(n=int(msk.sum()), rms_N=float(np.sqrt(np.mean(r ** 2))), bias_N=float(r.mean()), rms_S=float(np.sqrt(np.mean(eS[msk] ** 2))), bias_S=float(eS[msk].mean()),
                        tilt_exp=float(np.polyfit(x, r, 1)[0]), tilt_pred=float(np.polyfit(ep[msk], r, 1)[0]),
                        cov=float(100 * np.mean((r >= norm.ppf(.1) * sZ) & (r <= norm.ppf(.9) * sZ))))
    u1, u3 = (g == 'U1') & m, (g == 'U3') & m
    out['U1mU3'] = float(eN[u1].mean() - eN[u3].mean())
    out['U1mU3_inf'] = {l: float(eN[u1 & (inf == l)].mean() - eN[u3 & (inf == l)].mean()) for l in (100, 80, 60, 40, 20)}
    out['eN'] = eN; out['F'] = F
    return out
