import sys, json, math, numpy as np, os
sys.path.insert(0,'/tmp/claude-0/audit'); import fit as M
os.chdir('/tmp/claude-0/audit')
sys.path.insert(0,'/mnt/project-files/yaren-susmel-2025'); sys.path.insert(0,'/mnt/project-files/review/defects')
import kchi as K, rank as RK
from scipy.stats import norm
d=K.d
# row alignment
assert np.allclose(d.S,M.S) and np.allclose(d.lnN,M.lnN) and (d.ro==M.ro).all() and (d.g==M.g).all() and np.allclose(d.dv,M.DV) and (d.fit==M.fit).all()
print('data identical; dv identical')
def their_nll(spec,p):
    ex,st,bd,build=spec; F=build(p)
    c=p['k']*(p['lnsD']-np.log(d.S*F)-p['gam']*np.log((1-d.R)/2)); z=(d.lnN-c)/math.exp(p['lnsZ'])
    ll=np.where(d.ro,norm.logsf(z),norm.logpdf(z)-p['lnsZ']); return -ll[d.fit].sum()
J=json.load(open('/mnt/project-files/yaren-susmel-2025/kchi.json'))['variants']; Rf=json.load(open('/tmp/claude-0/audit/refit.json'))
for lab,(kc,rm),mv in [('BASELINE',(False,False),'baseline'),('+ R-1 triaxiality term',(False,True),'beta'),('+ k(chi)',(True,False),'kchi')]:
    spec=K.make(kc,rm); ps=J[lab]['par']; pm=Rf[mv]['par']
    n_ms,B,f,nll=M.make(mv)
    print(lab,'| stored par: their nll %.4f, my nll %.4f, stored-AIC-implied nll %.4f'%(their_nll(spec,ps),nll(np.array([ps[k] for k in n_ms])),(J[lab]['aic']-2*len(ps))/2),
          '| my par: their nll %.4f my nll %.4f'%(their_nll(spec,pm),Rf[mv]['nll']))
    # F comparison
    Ft=spec[3](pm); Fm=M.Fbase(pm['a0'],pm['kap'],pm['n'])
    print('   max rel F diff (their full build vs my base F)', np.abs(Ft/ (Fm if not (kc or rm) else Ft) -1).max() if not (kc or rm) else '')
p=Rf['baseline']['par']; Ft=K.make(False,False)[3](p); Fm=M.Fbase(p['a0'],p['kap'],p['n']); print('baseline F max rel diff',np.abs(Ft/Fm-1).max())
