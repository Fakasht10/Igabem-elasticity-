import sys, math, time, numpy as np, emcee
sys.path.insert(0, '/home/user/igabem-elasticity-/python')
from es2020 import sde
from scipy.optimize import minimize
L = float(sys.argv[1]); free = sys.argv[2] == 'free'; tag = f"wt_L{L}_{sys.argv[2]}"
rows = sde.dataset()
fit = sde.WalkerTCD([r for r in rows if r['set'] != 'held'], L, free)
idx = [0, 1, 2, 3, 4] + ([5] if free else [])
full = lambda z: np.r_[z, 0.0] if not free else z
lp = lambda z: fit.loglike(full(z))
best = None
for x0 in ([math.log(60), 6, 0.4, math.log(1.3), math.log(1e8), 0.3], [math.log(100), 5, 0.55, math.log(1.3), math.log(1e6), 0.0]):
    o = minimize(lambda z: -lp(z), np.array(x0)[idx], method='Nelder-Mead', options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
    if best is None or o.fun < best.fun: best = o
print(tag, 'MAP nll', round(best.fun, 2), np.round(best.x, 3), flush=True)
nw = 32; rng = np.random.default_rng(0)
lo, hi = fit.LO[idx], fit.HI[idx]
p0 = np.clip(best.x + 1e-3 * (hi - lo) * rng.standard_normal((nw, len(idx))), lo + 1e-9, hi - 1e-9)
s = emcee.EnsembleSampler(nw, len(idx), lp, moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
t = time.time(); s.run_mcmc(p0, 5000, progress=False)
ch = s.get_chain()[2500:].reshape(-1, len(idx)); chf = np.array([full(z) for z in ch])
np.savez(f'./chain_{tag}.npz', chain=chf, tau=s.get_autocorr_time(tol=0))
draws = chf[rng.choice(len(chf), 150, replace=False)]
m = sde.WalkerTCD(rows, L, free)
r = sde.predictive_residual(m, draws)
np.savez(f'./resid_{tag}.npz', **r)
sets = np.array([q['set'] for q in rows]); logN = m.lnN / math.log(10); fail = ~m.ro
q = np.percentile(chf, [5, 50, 95], axis=0)
print(tag, 'time', round(time.time() - t), 'tau', np.round(s.get_autocorr_time(tol=0)), 'post 5/50/95', {n: tuple(np.round(q[:, i], 3)) for i, n in enumerate(sde.WalkerTCD.NAMES)})
for name, mask in [('plain', sets == 'plain'), ('V', m.notch == 'sharp'), ('U r1', m.notch == 'intermediate'), ('U r3', m.notch == 'blunt'), ('HELD', sets == 'held')]:
    sc = sde.score(r, mask & fail, logN)
    lowP10 = np.mean(r['F'][mask & fail] < 0.1)
    print(f"  {name:6s} n {sc['n']:3d} rms {sc['rms']:.3f} bias {sc['bias']:+.3f} slope {sc['slope']:+.3f}{' TILTED' if sc['tilted'] else ''} cover {sc['coverage']*100:.0f}% below P10 {lowP10*100:.0f}%")
for Rr in (-1.0, 0.0):
    mk = (sets == 'held') & fail & (m.R == Rr); print(f'   held R {Rr:+.0f} bias {np.mean(r["dlogS"][mk]):+.3f}')
