import sys, json, numpy as np, time; sys.path.insert(0,'/tmp/claude-0/audit'); import fit as M
from scipy.optimize import minimize
v=sys.argv[1]; tag=sys.argv[2]
bo=dict(n=(-5,5),kap=(0,60),a0=(0,20),gam=(-1,2),k=(1,40),c=(-0.95,10),g1=(-3,3),b=(-20,20),Lc=(0.005,12)) if tag=='wide' else None
names,B,f,nll=M.make(v,bo); lo=np.array([B[n][0] for n in names]); hi=np.array([B[n][1] for n in names])
box=dict(lnsD=(4.2,4.8),k=(4.5,5.5),gam=(0.2,0.45),lnsZ=(-0.25,-0.1),a0=(0.5,2.0),kap=(10,15),n=(0,0.3),g1=(-0.3,0.5),b=(0.5,3),c=(-0.3,0.1),Lc=(0.1,1.0))
rng=np.random.default_rng(1); best=None; vals=[]
for s in range(24):
    x0=np.array([rng.uniform(*box[n]) for n in names]); x0=np.clip(x0,lo,hi)
    o=minimize(nll,x0,method='Powell',bounds=list(zip(lo,hi)),options=dict(maxiter=50000,xtol=1e-8,ftol=1e-12))
    o=minimize(nll,o.x,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(ftol=1e-14,gtol=1e-9))
    o=minimize(nll,o.x,method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-7,fatol=1e-10))
    vals.append(o.fun)
    if best is None or o.fun<best.fun: best=o
p=dict(zip(names,best.x)); r=dict(var=v,nll=best.fun,aic=2*best.fun+2*len(names),par=p,f=f); st=M.stats(r)
out=dict(aic=r['aic'],nll=r['nll'],par=p,kap=p['kap'],amax_all=p['a0']+p['kap']*M.DVMAX,vals=sorted(vals)[:6],**{k:x for k,x in st.items() if k!='e'})
json.dump(out,open(f'/tmp/claude-0/audit/res_{v}_mode_{tag}.json','w'),indent=1,default=float)
print(v,tag,'AIC %.3f'%r['aic'],{k:round(float(x),4) for k,x in p.items()},'amax %.2f'%out['amax_all'],[round(x,3) for x in out['vals']],flush=True)
