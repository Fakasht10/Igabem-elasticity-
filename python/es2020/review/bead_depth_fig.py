import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from es2020 import paper
js=json.load(open('es2020/data/igabem_bisector_measures.json')); prof={}
def find(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in paper.NOTCHES and isinstance(v,dict) and 'd' in v: prof[k]=v
            else: find(v)
find(js)
S=[x for x in paper.fig8_specimens() if not x.runout]
nt=np.array([x.notch for x in S]); N=np.array([x.N for x in S]); R=np.array([x.R for x in S])
x=np.log10(N); w=np.log10((1-R)/2); y=np.log10([x_.s_max for x_ in S])
P=paper.fig1_plain(); Rp,Np,sp=P.T; sp=sp*paper.UTS; kp=Np<1.8e6
ds=np.geomspace(0.02,3,80)
def lk(d): return np.array([np.log10(np.interp(d,prof[n]['d'],prof[n]['s1'])) for n in nt])
A=np.c_[np.ones(len(x)),x,w]
def fit(tr):
    best=None
    for d in ds:
        c=np.linalg.lstsq(A[tr],(y+lk(d))[tr],rcond=None)[0]; r=(y+lk(d))-A@c
        if best is None or (r[tr]**2).sum()<best[0]: best=((r[tr]**2).sum(),d,c,r)
    return best[1:]
d,c,r=fit(np.ones(len(x),bool)); g=-c[2]
C={'sharp':'#1b1b1b','intermediate':'#1f5fa8','blunt':'#c0392b'}; MK={'sharp':'^','intermediate':'o','blunt':'D'}
plt.rcParams.update({'font.size':9,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
a=ax[0]; NN=np.geomspace(3e2,2e6,50)
a.scatter(Np[kp],sp[kp]*((1-Rp[kp])/2)**g,s=10,c='0.72',lw=0,label=f'plain, Fig. 1 pool of three source papers (specimen, vector-extracted), n = {kp.sum()}')
Ap=np.c_[np.ones(kp.sum()),np.log10(Np[kp])]; cp=np.linalg.lstsq(Ap,np.log10(sp[kp]*((1-Rp[kp])/2)**g),rcond=None)[0]
a.plot(NN,10**(cp[0]+cp[1]*np.log10(NN)),'--',c='0.45',lw=1.2,label=f'plain pool, least-squares line (derived), k = {-1/cp[1]:.1f}')
for k in paper.NOTCHES:
    m=nt==k; a.scatter(N[m],10**(y[m]+lk(d)[m]+g*w[m]),s=22,marker=MK[k],facecolors='none',edgecolors=C[k],lw=0.9,label=f'{paper.LABEL[k]} (specimen, Fig. 8 extraction x IgBEM field), n = {m.sum()}')
a.plot(NN,10**(c[0]+c[1]*np.log10(NN)),'-',c='k',lw=1.4,label=f'one line through all notched (fitted), k = {-1/c[1]:.1f}, rms {r.std():.3f}')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('cycles to failure N'); a.set_ylabel(r'$\sigma_1$(depth %.2f mm) $\times$ $[(1-R)/2]^{%.2f}$  (MPa)'%(d,g))
a.set_title(f'(a) Field-only collapse without the plain band: depth {d:.2f} mm, Walker {g:.2f} (all fitted, 4 constants)',fontsize=9)
a.legend(fontsize=7,frameon=False,loc='lower left'); a.set_ylim(3,80)
b=ax[1]
for k in paper.NOTCHES:
    dk,ck,rk=fit(nt!=k); m=nt==k; h=rk[m]
    b.scatter(N[m],h,s=22,marker=MK[k],facecolors='none',edgecolors=C[k],lw=0.9,label=f'{paper.LABEL[k]} HELD OUT (fit on other two: depth {dk:.2f} mm): bias {h.mean():+.3f}, rms {np.sqrt((h**2).mean()):.3f}')
    p=np.polyfit(x[m],h,1); b.plot(N[m][np.argsort(N[m])],np.polyval(p,np.sort(x[m])),c=C[k],lw=1)
b.axhline(0,c='k',lw=0.6); b.axhspan(-0.139,0.139,color='0.9',zorder=0,label='current best field-only held-out rms (0.139), for scale')
b.set_xscale('log'); b.set_xlabel('cycles to failure N'); b.set_ylabel('held-out residual, log10 stress'); b.set_ylim(-0.35,0.35)
b.set_title('(b) Leave one notch out (specimen residuals; lines = residual trend, derived)',fontsize=9); b.legend(fontsize=7,frameon=False,loc='lower left')
fig.text(0.5,0.005,'Departure from the paper: Walker effective stress instead of sigma_max; plain band not used for calibration. Stored IgBEM bisector field (thickness mean), no re-solve. Run-outs (8) not shown.',ha='center',fontsize=7)
fig.tight_layout(rect=(0,0.02,1,1)); 
for e in ('png','pdf'): fig.savefig(HERE + f'fig_review_bead_depth.{e}',dpi=200)
