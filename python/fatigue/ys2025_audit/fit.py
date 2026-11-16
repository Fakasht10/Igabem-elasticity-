import numpy as np, pandas as pd, math, json, sys
from scipy.optimize import minimize
from scipy.stats import norm
sys.path.insert(0,'/tmp/claude-0/audit'); from fields import field, chi, triax, L0
Y='/mnt/project-files/yaren-susmel-2025/'
D=pd.read_csv(Y+'specimens_ijf2025.csv',comment='#')
t1=pd.read_csv(Y+'table1_plain.csv',comment='#'); t2=pd.read_csv(Y+'table2_notched.csv',comment='#')
dv={}
for _,r in t1.iterrows(): dv[('plain',r.infill_pct,r.raster_deg)]=0.0 if r.dv_mm=='-' else float(r.dv_mm)
for _,r in t2.iterrows():
    g='V' if r.notch=='V' else ('U1' if float(r.root_radius_mm)==1 else 'U3')
    v=0.0 if r.dv_mm=='-' else float(r.dv_mm)
    if (g,r.infill_pct,0) in dv: assert dv[(g,r.infill_pct,0)]==v
    dv[(g,r.infill_pct,0)]=v
GM={'plain':'plain','V_r0.15':'V','U_r1':'U1','U_r3':'U3'}
g=D.geometry.map(GM).values; R=D.R.values; S=D.sigma_max_MPa.values; lnN=np.log(D.N.values.astype(float)); ro=D.runout.values.astype(bool)
inf=D.infill_pct.values; DV=np.array([dv[(gg,i,rr)] for gg,i,rr in zip(g,inf,D.raster_deg.values)]); phi=inf/100.
fit=np.isin(g,['plain','V']); w=(1-R)/2; comp=np.maximum(-R,0)
FLD={k:field(k) for k in ['plain','V','U1','U3']}
def Kf(k,x):
    d,K=FLD[k]; x=np.abs(((np.asarray(x,float)+6.0)%12.0)-6.0); return np.interp(x,d,K)
# weight average (2/pi) int_0^{pi/2} K(a sin th) dth with Gauss-Legendre (independent of their 24-pt midpoint rule)
xg,wg=np.polynomial.legendre.leggauss(256); TH=(xg+1)*math.pi/4; WT=wg*math.pi/4*2/math.pi
def wf(k,a): return float(np.sum(WT*Kf(k,a*np.sin(TH))))
def Wst(a,r): return (a+r)/math.sqrt(r*(2*a+r))
keys=sorted(set(zip(g,DV,phi)))
idx={kk:np.where((g==kk[0])&(DV==kk[1])&(phi==kk[2]))[0] for kk in keys}
def Fbase(a0,kap,n,Lc=L0):
    F=np.empty(len(S)); r=Lc/2
    for (k,v,p),ii in idx.items():
        a=a0+kap*v; F[ii]=p**-n*(Kf(k,a+r)+(Wst(a,r)-1)*wf(k,a))
    return F
gx=lambda x:x/(1+x)
CHI={k:chi(k) for k in FLD}; Gc=np.array([gx(CHI[k]*L0) for k in g])
HX={k:max(triax(k)[1][0]-1/3,0) for k in FLD}; Dh=np.array([HX[k] for k in g]); BMAX=1/Dh.max()
# variants: names, start values, bounds, function returning (lnc) given dict p
BASE=['lnsD','k','gam','lnsZ','a0','kap','n']
BB=dict(lnsD=(-5,15),k=(1,40),gam=(0,1),lnsZ=(-4,1.5),a0=(0,5.5),kap=(0,30),n=(0,5))
DVMAX=DV.max(); AMAX=6.0
def lnc_fn(var,constrained=False):
    def f(p):
        if constrained:
            p=dict(p); p['kap']=p['u']*(AMAX-p.get('Lc',L0)/2-p['a0'])/DVMAX
        F=Fbase(p['a0'],p['kap'],p['n'],p.get('Lc',L0))
        lnsX=np.log(S*F)+p['gam']*np.log(w)
        if var=='gchi': lnsX=lnsX+p['g1']*Gc*np.log(w)
        if var=='beta': lnsX=lnsX+np.log1p(p['b']*Dh*comp)
        kk=p['k']*(1+p['c']*Gc) if var=='kchi' else p['k']
        return kk*(p['lnsD']-lnsX)
    return f
VARS={'baseline':([],{}),'gchi':(['g1'],dict(g1=(-1,1))),'beta':(['b'],dict(b=(0,BMAX))),'kchi':(['c'],dict(c=(-0.75,3))),'Lfree':(['Lc'],dict(Lc=(0.02,6)))}
def make(var,bounds_over=None,constrained=False):
    ex,eb=VARS[var]; names=BASE+ex; B={**BB,**eb,**(bounds_over or {})}
    if constrained: names=[('u' if n=='kap' else n) for n in names]; B['u']=(0,1)
    f=lnc_fn(var,constrained)
    def nll(x,mask=fit):
        p=dict(zip(names,x)); c=f(p)
        if not np.all(np.isfinite(c)): return 1e10
        s=math.exp(p['lnsZ']); z=(lnN-c)/s
        ll=np.where(ro,norm.logsf(z),norm.logpdf(z)-p['lnsZ'])
        return -ll[mask].sum()
    return names,B,f,nll
def fitvar(var,bounds_over=None,nstart=40,seed=0,verbose=False,constrained=False):
    names,B,f,nll=make(var,bounds_over,constrained); rng=np.random.default_rng(seed)
    lo=np.array([B[n][0] for n in names]); hi=np.array([B[n][1] for n in names])
    slo=lo.copy(); shi=hi.copy()
    # sensible start box
    box=dict(lnsD=(3,5.5),k=(3,8),gam=(0.05,0.8),lnsZ=(-0.5,0.3),a0=(0,3),kap=(0,20),u=(0,1),n=(-0.5,1),g1=(-0.5,0.5),b=(0,BMAX*0.8),c=(-0.5,1),Lc=(0.05,3))
    for i,n in enumerate(names): slo[i],shi[i]=max(box[n][0],lo[i]),min(box[n][1],hi[i])
    res=[]
    for s in range(nstart):
        x0=slo+rng.random(len(names))*(shi-slo)
        o=minimize(nll,x0,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=5000,ftol=1e-13,gtol=1e-8))
        o2=minimize(nll,o.x,method='Powell',bounds=list(zip(lo,hi)),options=dict(maxiter=50000,xtol=1e-8,ftol=1e-12))
        o3=minimize(nll,o2.x,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(maxiter=5000,ftol=1e-14,gtol=1e-9))
        best=min([o,o2,o3],key=lambda q:q.fun); res.append((best.fun,best.x,o.fun))
    res.sort(key=lambda t:t[0])
    fx=res[0]; p=dict(zip(names,fx[1]))
    onb=[n for n,v in p.items() if (abs(v-B[n][0])<1e-4*(1+abs(B[n][0])) or abs(v-B[n][1])<1e-4*(1+abs(B[n][1])))]
    spread=[r[0] for r in res]
    pp=dict(p)
    if constrained: pp['kap']=p['u']*(AMAX-p.get('Lc',L0)/2-p['a0'])/DVMAX
    return dict(var=var,constrained=constrained,amax_fit=float(pp['a0']+pp['kap']*DV[fit].max()),amax_all=float(pp['a0']+pp['kap']*DVMAX),kap=float(pp['kap']),nll=fx[0],aic=2*fx[0]+2*len(names),par=p,on_bound=onb,nll_starts=dict(best=spread[0],q10=float(np.quantile(spread,.1)),median=float(np.median(spread)),
                n_within_0p01=int(sum(1 for v in spread if v-spread[0]<0.01)),lbfgs_only_best=float(min(r[2] for r in res)),n=len(spread)),f=f,nll_fn=nll,names=names)
def stats(R_):
    p=R_['par']; f=R_['f']; lp=f(p)/math.log(10); le=lnN/math.log(10); e=le-lp; sZ=math.exp(p['lnsZ'])
    hw=norm.ppf(0.9)*sZ/math.log(10)   # P10-P90 half-width in log10 N (for k(chi), lnN scatter is common)
    m=~ro
    def sc(mm):
        mm=mm&m; r=e[mm]
        return dict(n=int(mm.sum()),rms=float(np.sqrt(np.mean(r**2))),bias=float(r.mean()),slope_vs_Nexp=float(np.polyfit(le[mm],r,1)[0]),
                    slope_vs_Npred=float(np.polyfit(lp[mm],r,1)[0]),cov=float(100*np.mean(np.abs(r)<=hw)))
    out=dict(fit=sc(fit),held=sc(~fit))
    u1=m&(g=='U1'); u3=m&(g=='U3')
    gap=dict(all=float(e[u1].mean()-e[u3].mean()))
    for lv in (100,80,60,40,20): gap[lv]=float(e[u1&(inf==lv)].mean()-e[u3&(inf==lv)].mean())
    for RR in (-1,0.1): gap['R%g'%RR]=float(e[u1&(R==RR)].mean()-e[u3&(R==RR)].mean())
    out['U1_minus_U3']=gap; out['e']=e
    return out
