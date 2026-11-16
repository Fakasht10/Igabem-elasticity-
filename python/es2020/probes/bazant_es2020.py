import sys, math, time, numpy as np, emcee
sys.path.insert(0, '/home/user/igabem-elasticity-/python')
from es2020 import sde, sde_bazant_probe as B, methods as M
from scipy.optimize import minimize
tp = float(sys.argv[1])
rows = sde.dataset()
prof = {n: (None if n == "plain" else (lambda p: (lambda x: p(sde._refl(x))))(M._prof(n))) for n in ("plain","sharp","intermediate","blunt")}
W = {n: sde.crack_path(n, tp)[3] for n in prof}
fit = B.ParisChain([r for r in rows if r["set"] != "held"], prof, W)
lp = fit.loglike
best = None
for a0 in (0.05, 0.3, 1.0):
  for n0 in (4.0, 6.0):
    x0 = np.array([math.log(50), n0, 0.45, math.log(a0), math.log(0.3), -3.0])
    o = minimize(lambda z: -lp(z), x0, method='Nelder-Mead', options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))
    if best is None or o.fun < best.fun: best = o
print('MAP nll', round(best.fun, 2), np.round(best.x, 3), flush=True)
nw = 24; rng = np.random.default_rng(0)
p0 = np.clip(best.x + 1e-3 * (B.HI - B.LO) * rng.standard_normal((nw, 6)), B.LO + 1e-9, B.HI - 1e-9)
s = emcee.EnsembleSampler(nw, 6, lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
t = time.time(); s.run_mcmc(p0, 3000, progress=False)
ch = s.get_chain()[1500:].reshape(-1, 6); np.savez(f'chain_bz_t{tp}.npz', chain=ch)
q = np.percentile(ch, [5, 50, 95], axis=0)
from scipy.special import expit
tr = [math.exp, float, float, math.exp, math.exp, expit]
print('time', round(time.time()-t), 'post', {n: tuple(round(tr[i](v), 4) for v in q[:, i]) for i, n in enumerate(B.NAMES)})
zg = __import__('scipy.stats').stats.norm.ppf(expit(q[1,5])); print('tail modulus m (strength)', round(__import__('scipy.stats').stats.norm.pdf(zg)/(math.exp(q[1,4])*expit(q[1,5])),2))
draws = ch[rng.choice(len(ch), 120, replace=False)]
m = B.ParisChain(rows, prof, W)
r = sde.predictive_residual(m, draws)
sets = np.array([x['set'] for x in rows]); logN = m.lnN / math.log(10); fail = ~m.ro
for name, mask in [('plain', sets == 'plain'), ('V', m.notch == 'sharp'), ('U r1', m.notch == 'intermediate'), ('U r3', m.notch == 'blunt'), ('HELD', sets == 'held')]:
    sc = sde.score(r, mask & fail, logN)
    print(f"  {name:6s} n {sc['n']:3d} rms {sc['rms']:.3f} bias {sc['bias']:+.3f} slope {sc['slope']:+.3f}{' TILTED' if sc['tilted'] else ''} cover {sc['coverage']*100:.0f}% below P10 {np.mean(r['F'][mask & fail] < 0.1)*100:.0f}%")
for Rr in (-1.0, 0.0):
    mk = (sets == 'held') & fail & (m.R == Rr); print(f'   held R {Rr:+.0f} bias {np.mean(r["dlogS"][mk]):+.3f}')
