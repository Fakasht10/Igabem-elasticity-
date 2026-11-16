import sys, json, math, time, numpy as np
sys.path.insert(0, '/tmp/claude-0/audit3'); import model as M
OUT = {}
for key, model, ac in (('base', 'base', None), ('sat21', 'sat', 2.1), ('satfit', 'sat', None)):
    t = time.time(); r = M.fitmodel(model, ac, nstart=80, seed=1)
    p = r['p']; s = M.stats(model, p)
    nb = [(n_, round(x, 4)) for n_, x, (lo, hi) in zip(r['names'], r['x'], r['bounds']) if min(x - lo, hi - x) < 1e-3 * max(1, abs(hi - lo))]
    amax = M.amap(model, p, M.dv.max()); amaxf = M.amap(model, p, M.dv[M.fit].max())
    # numerical Hessian SEs
    f = r['f']; x = r['x']; h = 1e-4 * np.maximum(1, np.abs(x)); n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.eye(n)[i] * h[i]; ej = np.eye(n)[j] * h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    try: se = dict(zip(r['names'], np.sqrt(np.diag(np.linalg.inv(H)))))
    except Exception: se = None
    OUT[key] = dict(aic=r['aic'], nll=r['nll'], par=p, se={k: float(v) for k, v in se.items()} if se else None, on_bound=nb, a_max=amax, a_max_fit=amaxf, wall_gap=M.B - M.r0 - amax,
                    spread=r['spread'], fit=s['fit'], held=s['held'], U1mU3=s['U1mU3'], U1mU3_inf=s['U1mU3_inf'], sec=time.time() - t)
    print(key, json.dumps({k: v for k, v in OUT[key].items()}, default=float, indent=None)[:2500], flush=True)
json.dump(OUT, open('/tmp/claude-0/audit3/refit.json', 'w'), indent=1, default=float)
