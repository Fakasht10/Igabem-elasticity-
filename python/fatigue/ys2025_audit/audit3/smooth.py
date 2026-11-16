import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/audit3'); import model as M
from scipy.interpolate import PchipInterpolator
P = {g: PchipInterpolator(*M.PR[g], extrapolate=True) for g in M.PR}
M.K = lambda g, x: P[g](np.clip(np.asarray(x, float), 0, M.B))
R0 = json.load(open('refit.json'))
for key, model, ac in (('base', 'base', None), ('sat21', 'sat', 2.1), ('satfit', 'sat', None)):
    x0 = [R0[key]['par'][k] for k in M.names(model, ac)]
    r = M.fitmodel(model, ac, nstart=20, seed=5, x0s=[x0]); s = M.stats(model, r['p'])
    print(key, 'AIC %.2f' % r['aic'], {k: round(v, 4) for k, v in r['p'].items()}, 'held rms %.4f' % s['held']['rms_N'], flush=True)
