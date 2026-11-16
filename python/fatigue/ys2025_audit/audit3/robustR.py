import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/audit3'); import model as M
R0 = json.load(open('refit.json'))
for Rv in (-1, 0.1):
    mk = M.fit & (M.R != Rv); out = []
    for key, model, ac in (('base', 'base', None), ('sat21', 'sat', 2.1)):
        gm = R0[key]['par']['gam']; M.GB['gam'] = (gm - 1e-9, gm + 1e-9)   # gamma not identifiable from one R: fixed at its full-fit value
        x0 = [R0[key]['par'][k] for k in M.names(model, ac)]
        r = M.fitmodel(model, ac, mask=mk, nstart=0, x0s=[x0]); F = M.Frow(model, r['p']); ok = ~M.fit & ~M.ro & np.isfinite(F)
        e = M.stats.__globals__  # noqa
        se = M.S * F * ((1 - M.R) / 2) ** r['p']['gam']; eN = M.lnN / np.log(10) - r['p']['k'] * (r['p']['lnsD'] - np.log(se)) / np.log(10)
        out.append((r['aic'], float(np.sqrt(np.mean(eN[ok] ** 2))), int((~M.fit & ~M.ro & ~np.isfinite(F)).sum())))
    print('drop R %g (gam fixed): dAIC %+.1f  held rms base %.3f sat21 %.3f  (held rows past wall: base %d sat %d)' % (Rv, out[1][0] - out[0][0], out[0][1], out[1][1], out[0][2], out[1][2]))
