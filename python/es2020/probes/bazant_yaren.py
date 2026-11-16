import sys, math, csv, json, time, numpy as np, emcee
sys.path.insert(0, '/home/user/igabem-elasticity-/python')
from es2020 import sde, sde_bazant_probe as B
from scipy.optimize import minimize
from scipy.special import expit
D = '/mnt/project-files/yaren-susmel-2025/'
R_ = [x for x in csv.DictReader(l for l in open(D + 'specimens_ijf2025.csv') if not l.startswith('#'))]
P = json.load(open(D + 'profiles.json'))
prof = {"plain": None}
for k in ("V_r0.15", "U_r1", "U_r3"):
    p = P[k][-1]; r, K = np.array(p['r']), np.array(p['K'])
    prof[k] = (lambda r, K: (lambda x: np.interp(np.where(x <= 6, x, 12 - np.minimum(x, 12)), r, K)))(r, K)
W = {"plain": 5.0, "V_r0.15": 12.0, "U_r1": 12.0, "U_r3": 12.0}   # crack_path rule: plain uniform -> thickness 5 < width 12; notches -> opposite root
INF = (100, 80, 60, 40, 20)
models = {}
for i in INF:
    rows = [dict(notch=x['geometry'], R=float(x['R']), s_max=float(x['sigma_max_MPa']), N=float(x['N']), runout=x['runout'] == 'True')
            for x in R_ if int(x['infill_pct']) == i]
    models[i] = B.ParisChain(rows, prof, W)
def lp(z):
    tot = 0.0
    for j, i in enumerate(INF):
        th = np.r_[z[:3], z[5 + j], z[3:5]]
        v = models[i].loglike(th)
        if not np.isfinite(v): return -np.inf
        tot += v
    return tot
best = None
for a0 in (0.1, 0.5):
    x0 = np.r_[math.log(30), 5.0, 0.5, math.log(0.3), -3.0, [math.log(a0)] * 5]
    o = minimize(lambda z: -lp(z), x0, method='Nelder-Mead', options=dict(maxiter=20000, maxfev=20000, xatol=1e-6, fatol=1e-8))
    if best is None or o.fun < best.fun: best = o
print('MAP nll', round(best.fun, 2), flush=True)
nd = 10; nw = 30; rng = np.random.default_rng(0)
s = emcee.EnsembleSampler(nw, nd, lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
s.run_mcmc(best.x + 1e-4 * rng.standard_normal((nw, nd)), 3000, progress=False)
ch = s.get_chain()[1500:].reshape(-1, nd); np.savez('chain_ys.npz', chain=ch)
q = np.percentile(ch, [5, 50, 95], axis=0)
print('sigma_D', np.round(np.exp(q[:, 0]), 2), 'n', np.round(q[:, 1], 2), 'gamma', np.round(q[:, 2], 2), 's', np.round(np.exp(q[:, 3]), 3), 'Pgr', np.round(expit(q[:, 4]), 4))
dv = {100: '-', 80: '0.20-0.30', 60: '0.38-0.40', 40: '0.73', 20: '1.57'}
for j, i in enumerate(INF):
    print(f'  infill {i:3d}%  a0 {np.exp(q[1, 5 + j]):.3f} mm ({np.exp(q[0, 5 + j]):.3f}-{np.exp(q[2, 5 + j]):.3f})   d_v {dv[i]}')
# per-infill fit quality (in-sample, every geometry)
draws = ch[rng.choice(len(ch), 80, replace=False)]
for j, i in enumerate(INF):
    m = models[i]; dr = np.array([np.r_[d[:3], d[5 + j], d[3:5]] for d in draws])
    r = sde.predictive_residual(m, dr); f = ~m.ro
    out = []
    for g in ("plain", "V_r0.15", "U_r1", "U_r3"):
        mk = f & (m.notch == g); d = r['dlogS'][mk]
        out.append(f"{g} {math.sqrt(np.mean(d**2)):.3f}/{np.mean(d):+.3f}")
    print(f'  infill {i:3d}% rms/bias:', '  '.join(out), f' cover {np.mean(r["inside"][f])*100:.0f}%')
