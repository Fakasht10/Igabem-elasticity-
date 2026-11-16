import sys, numpy as np, json
for p_ in ('/home/user/wt_uxm/python', '/mnt/project-files/review', '/mnt/project-files/review/defects'): sys.path.insert(0, p_)
C='/home/user/igabem-elasticity-/cache/ys2025/'
for k in ('plain','V_r0.15','U_r1','U_r3'):
    z=np.load(C+f'profile_{k}_h0.npz'); print(k, list(z.keys()), z['d'].shape, z['s1'].shape, 'd range',z['d'].min(),z['d'].max(),'z',z['z'], 'K0',z['s1'][-1][0],'K end',z['s1'][-1][-1])
t=np.load(C+'triax_bisector.npz'); print(list(t.keys()))
for k in ('plain','V_r0.15','U_r1','U_r3'): print(k, t[k+'_h'].shape, t[k+'_h'][0][0], t[k+'_z'])
P=json.load(open('/mnt/project-files/yaren-susmel-2025/profiles.json'))
for k,v in P.items(): print('2D',k,len(v), min(v[-1]['r']), max(v[-1]['r']), v[-1]['K'][0], v[-1]['K'][-1], [x for x in v[-1] if x not in ('r','K')])
import rank as RK
d=RK.load_ys(); print('YS n',len(d.S),'fit',d.fit.sum(),'ro',d.ro.sum())
for g in ('plain','V','U1','U3'):
    m=d.g==g; print(g, m.sum(), 'dv uniq', np.unique(d.dv[m]), 'ro', d.ro[m].sum())
print('max dv fit', d.dv[d.fit].max(), 'max dv all', d.dv.max())
import sde_tcd as s
for k,v in s.PROF.items(): print('ES PROF',k, v.d.min(), v.d.max(), v.s[0], v.s[-1], len(v.d))
e=RK.load_es(); print('ES n',len(e.S),'fit',e.fit.sum(),'ro',e.ro.sum(), {g:int((e.g==g).sum()) for g in ('plain','V','U1','U3')})
import reconcile_pool as rp; p=rp.pooled(); print('pool n',len(p.S),'fit',p.fit.sum(), {g:int((p.g==g).sum()) for g in ('plain','V','U1','U3')})
