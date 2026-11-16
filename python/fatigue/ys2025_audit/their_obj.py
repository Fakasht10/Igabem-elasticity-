# Minimise THEIR objective exactly (their F with 24-pt rule, their bounds/penalties) with gradient-based + Powell from several starts,
# to separate optimiser non-convergence from my quadrature change.
import sys, json, math, numpy as np
sys.path.insert(0,'/mnt/project-files/yaren-susmel-2025'); sys.path.insert(0,'/mnt/project-files/review/defects')
import kchi as K, rank as RK
from scipy.optimize import minimize
from scipy.stats import norm
d=K.d; Y='/mnt/project-files/yaren-susmel-2025/'
Gk=K.Gk
def gbuild(p):
    F=K.make(False,False)[3]({k:v for k,v in p.items() if k!='g1'}); return F*K.w**(p['g1']*Gk)
specs={'baseline':(K.make(False,False),'kchi.json','BASELINE'),'beta':(K.make(False,True),'kchi.json','+ R-1 triaxiality term'),
       'kchi':(K.make(True,False),'kchi.json','+ k(chi)'),'gchi':((K.make(False,False)[0]+['g1'],None,dict(K.make(False,False)[2],g1=(-1,1)),gbuild),'gamma_chi.json','gamma(chi_root), l = L0')}
which=sys.argv[1]; spec,jf,lab=specs[which]; ex,_,bd,build=spec
names=['lnsD','k','gam','lnsZ']+ex
def nll(x):   # verbatim logic of rank.run.nll
    p=dict(zip(names,x))
    if not (1<p['k']<40 and 0<=p['gam']<=1 and -4<p['lnsZ']<1.5): return 1e12
    for k,(lo,hi) in bd.items():
        if not lo<=p[k]<=hi: return 1e12
    F=build(p); c=p['k']*(p['lnsD']-np.log(d.S*F)-p['gam']*np.log((1-d.R)/2))
    if not np.all(np.isfinite(c)): return 1e12
    z=(d.lnN-c)/math.exp(p['lnsZ']); ll=np.where(d.ro,norm.logsf(z),norm.logpdf(z)-p['lnsZ']); return -float(ll[d.fit].sum())
J=json.load(open(Y+jf))['variants'][lab]; ps=J['par']; x_st=np.array([ps[n] for n in names])
mine=json.load(open(f'/tmp/claude-0/audit/res_{which}_con.json'))['par']
if 'u' in mine: mine['kap']=json.load(open(f'/tmp/claude-0/audit/res_{which}_con.json'))['kap']
x_my=np.array([mine[n] for n in names])
B=[(-5,15),(1.0001,39.999),(0,1),(-3.999,1.499)]+[bd[n] for n in ex]
print(which,'stored nll %.4f'%nll(x_st),'my-opt (GL256 fit) on their objective %.4f'%nll(x_my))
for nm,x0 in (('from stored',x_st),('from mine',x_my)):
    o=minimize(nll,x0,method='L-BFGS-B',bounds=B,options=dict(ftol=1e-14,gtol=1e-9)); o2=minimize(nll,o.x,method='Powell',bounds=B,options=dict(xtol=1e-8,ftol=1e-12,maxiter=50000))
    o3=minimize(nll,o2.x,method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-8,fatol=1e-11))
    print('  ',nm,'L-BFGS-B %.4f -> Powell %.4f -> NM %.4f'%(o.fun,o2.fun,o3.fun),{n:round(float(v),4) for n,v in zip(names,o3.x)},'amax %.2f'%(o3.x[names.index('a0')]+o3.x[names.index('kap')]*1.66),flush=True)
