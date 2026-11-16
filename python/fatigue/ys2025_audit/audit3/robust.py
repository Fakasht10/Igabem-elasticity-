import sys, json, math, numpy as np
sys.path.insert(0, '/tmp/claude-0/audit3'); import model as M
R0 = json.load(open('/tmp/claude-0/audit3/refit.json'))
pb, ps = R0['base']['par'], R0['sat21']['par']
lb, _ = M.nll_rows('base', pb, np.ones(len(M.S), bool)); ls, _ = M.nll_rows('sat', ps, np.ones(len(M.S), bool))
eb, es = M.stats('base', pb)['eN'], M.stats('sat', ps)['eN']; m = ~M.ro
print('fit-set nll gain (base - sat21), total %.2f' % (lb - ls)[M.fit].sum())
for lev in (100, 80, 60, 40, 20):
    k = M.fit & (M.inf == lev); print('  infill %3d n %3d dNLL %+.2f' % (lev, k.sum(), (lb - ls)[k].sum()))
for Rv in (-1, 0.1):
    k = M.fit & (M.R == Rv); print('  R %4.1f n %3d dNLL %+.2f' % (Rv, k.sum(), (lb - ls)[k].sum()))
for gg in ('plain', 'V'):
    k = M.fit & (M.g == gg); print('  %s dNLL %+.2f' % (gg, (lb - ls)[k].sum()))
rms = lambda e, k: (math.sqrt(np.mean(e[k] ** 2)), e[k].mean(), int(k.sum()))
print('held-out failures, rms/bias log10 N: base vs sat21')
for lab, k in [('all', ~M.fit & m)] + [('infill %d' % l, ~M.fit & m & (M.inf == l)) for l in (100, 80, 60, 40, 20)] + [('R %g' % r, ~M.fit & m & (M.R == r)) for r in (-1, 0.1)] + [(gg, m & (M.g == gg)) for gg in ('U1', 'U3')]:
    a, b = rms(eb, k), rms(es, k); print('  %-10s n %3d  base %.3f/%+.3f  sat21 %.3f/%+.3f  d_rms %+.3f' % (lab, a[2], a[0], a[1], b[0], b[1], b[0] - a[0]))
print('held rms, infill x R')
for l in (100, 80, 60, 40, 20):
    print('  ', l, ['R%g: %.3f->%.3f' % (r, rms(eb, ~M.fit & m & (M.inf == l) & (M.R == r))[0], rms(es, ~M.fit & m & (M.inf == l) & (M.R == r))[0]) for r in (-1, 0.1)])
# leave-one-level-out / one-R-out refit: does dAIC survive without each subset?
print('drop-one-subset refits (fit set only): dAIC = AIC_sat21 - AIC_base, and held rms')
for lab, drop in [('drop infill %d' % l, M.inf == l) for l in (100, 80, 60, 40, 20)] + [('drop R %g' % r, M.R == r) for r in (-1, 0.1)]:
    mk = M.fit & ~drop
    rb = M.fitmodel('base', None, mask=mk, nstart=20, seed=3, x0s=[list(pb.values())[:7]]); rs = M.fitmodel('sat', 2.1, mask=mk, nstart=20, seed=3, x0s=[[ps[k] for k in M.names('sat', 2.1)]])
    hb = M.stats('base', rb['p']); hs = M.stats('sat', rs['p'])
    print('  %-16s dAIC %+.1f   held rms base %.3f sat21 %.3f' % (lab, rs['aic'] - rb['aic'], hb['held']['rms_N'], hs['held']['rms_N']), flush=True)
