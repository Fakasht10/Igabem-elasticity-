import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
import numpy as np, sys, time
from es2020 import paper, collapse as C, methods as M
src=sys.argv[1]
if src=='figs57': paper.fig8_specimens=paper.specimens
t=time.time(); out=C.run(); _,rows=C.field_rank(out); print("run s",time.time()-t)
specs=paper.fig8_specimens()
lab={k:[(s.p,s.R) for s in specs if s.notch==k and not s.runout] for k in paper.NOTCHES}
names=['net','weibull_mN','grad_paperL','lsa_point','gradient','neuber_nsr'] 
names=[n for n in names if n in out]; print([n for n in out])
def H(name): 
    r=np.concatenate([out[name]['R'][k][1] for k in ('intermediate','blunt')]); return r
ser=np.array([f"{k}{p}{R}" for k in ('intermediate','blunt') for (p,R) in lab[k]]); us=np.unique(ser)
Rr=np.array([R for k in ('intermediate','blunt') for (p,R) in lab[k]])
rng=np.random.default_rng(1)
print(src,"held-out n",len(ser),"series",len(us))
base=H('net')
for n in out:
    r=H(n); 
    d=[];d2=[]
    for _ in range(2000):
        i=rng.integers(0,len(r),len(r)); d.append(np.sqrt(np.mean(r[i]**2))-np.sqrt(np.mean(base[i]**2)))
        pick=rng.choice(us,len(us)); j=np.concatenate([np.where(ser==u)[0] for u in pick]); d2.append(np.sqrt(np.mean(r[j]**2))-np.sqrt(np.mean(base[j]**2)))
    # rms after removing R mean offset (what a Walker term would absorb) 
    rr=r.copy(); 
    for R in (-1,0): rr[Rr==R]-=r[Rr==R].mean()-r.mean()
    sm=np.array([r[ser==u].mean() for u in us]); 
    within=np.sqrt(np.mean(np.concatenate([r[ser==u]-r[ser==u].mean() for u in us])**2))
    print("%-16s rms %.3f  diff vs net %+.3f  90%%CI specimen [%+.3f,%+.3f] series-cluster [%+.3f,%+.3f] | rms R-offset removed %.3f | within-series %.3f, sd of series means %.3f"%(n,np.sqrt(np.mean(r**2)),np.sqrt(np.mean(r**2))-np.sqrt(np.mean(base**2)),*np.percentile(d,[5,95]),*np.percentile(d2,[5,95]),np.sqrt(np.mean(rr**2)),within,sm.std()))
