import sys, json, math, numpy as np
sys.path.insert(0, '/tmp/claude-0/audit3'); import model as M
prof = []
for ac in (0.3, 0.6, 1.0, 1.4, 1.8, 2.1, 2.4, 2.8, 3.3, 4.0, 4.8, 5.8, 20.0, 100.0):
    r = M.fitmodel('sat', ac, nstart=25, seed=2); p = r['p']
    prof.append(dict(acap=ac, nll=r['nll'], a0=p['a0'], kap=p['kap'], n=p['n'], wall_gap=M.B - M.r0 - M.amap('sat', p, M.dv.max()), held_rms=M.stats('sat', p)['held']['rms_N']))
    print(json.dumps(prof[-1]), flush=True)
json.dump(prof, open('/tmp/claude-0/audit3/profile.json', 'w'), indent=1)
