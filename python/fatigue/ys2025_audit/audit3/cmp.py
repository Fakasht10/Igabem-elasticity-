import sys, json, math, numpy as np
sys.path[:0]=['/home/user/wt_uxm/python','/mnt/project-files/review','/mnt/project-files/review/defects','/mnt/project-files/yaren-susmel-2025','/tmp/claude-0/audit3']
import model as M, rank as RK, fixed_field as FF
src = open('/mnt/project-files/yaren-susmel-2025/satur.py').read().split('OUT = {}')[0]   # spec() only; do NOT run the driver (it rewrites satur.json)
ns = {}; exec(src, ns)
d = RK.load_ys()
print('data match', (d.g==M.g).all(), np.allclose(d.S,M.S), np.allclose(d.lnN,M.lnN), (d.ro==M.ro).all(), np.allclose(d.dv,M.dv), (d.fit==M.fit).all(), 'n',len(M.S),'fit',M.fit.sum(),'held',(~M.fit).sum())
SJ = json.load(open('/mnt/project-files/yaren-susmel-2025/satur.json')); BJ = json.load(open('/mnt/project-files/yaren-susmel-2025/fixed_field.json'))['YS | 3D mid-plane + plain field (baseline)']
for nm, ac in (('saturating, a_cap = 2.1 mm a priori (T/2 - shell)', 2.1), ('saturating, a_cap fitted (one admissible constant)', None)):
    d = RK.load_ys(); sp = ns['spec'](d, ac); p = SJ[nm]['par']
    print(nm, 'names', sp[0], 'bounds', sp[2], 'nstarts', len(sp[1]))
    F1 = sp[3](p); pp = dict(p); pp.setdefault('acap', ac); F2 = M.Frow('sat', pp)
    print('  max rel F diff', np.max(np.abs(F1/F2-1)))
    s1 = FF.stats(d, p, F1); s2 = M.stats('sat', pp)
    for k in ('fit','held'): print('  ', k, {q: (round(s1[k][q],4), round(s2[k][q],4)) for q in s1[k]})
d = RK.load_ys(); sp = FF.ys3d(d, False); p = BJ['par']; F1 = sp[3](p); F2 = M.Frow('base', p); print('baseline max rel F diff', np.max(np.abs(F1/F2-1)), 'bounds', sp[2])
# wall leak test: F evaluated on held-out cells inside the objective?
d = RK.load_ys(); sp = ns['spec'](d, 2.1); p = dict(SJ['saturating, a_cap = 2.1 mm a priori (T/2 - shell)']['par'])
for a0 in (3.65, 3.68, 3.70):
    p['a0']=a0; F = sp[3](p); print('a0',a0,'inf rows fit',int(np.isinf(F[d.fit]).sum()),'held',int(np.isinf(F[~d.fit]).sum()))
# small/large dv limits
p = dict(a0=0.4, kap=4.458, acap=2.1)
for x in (1e-6, 1e-3, 10, 1e3): print('dv',x,'a',M.amap('sat',p,x), 'lin', 0.4+4.458*x)
