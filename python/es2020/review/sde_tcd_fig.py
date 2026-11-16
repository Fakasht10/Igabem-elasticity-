import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
import math, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sde_tcd as T
from sde_tcd import V, notch, R, lnN, ro
A=json.load(open(HERE + "sde_tcd_results.json")); B=json.load(open(HERE + "sde_tcd_more_results.json"))
def xof(v,par): return np.array([ (math.log(par[n]) if n.startswith("ln") else par[n]) for n in v.free])
cases=[("(a) SDE thread model, refit: gradient support + Neuber/Masing loop\nplain tied, 6 fitted constants",V("grad","ro"),A[1]),
       ("(b) TCD point stress at L/2, L = 0.4 mm fixed (shell bead), no plasticity,\nWalker damage, plain level free; 5 fitted constants",V("tcd","wb","free",fixed=dict(lnL=math.log(0.4))),B[0])]
plt.rcParams.update({"font.size":9,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
fig,axs=plt.subplots(1,2,figsize=(12.5,4.4),sharey=True)
C={"intermediate":"#1f5fa8","blunt":"#c0392b"}; LAB={"intermediate":"U r1","blunt":"U r3"}
for ax,(title,v,rec) in zip(axs,cases):
    d=v.resid(xof(v,rec["par"]))
    for k in ("intermediate","blunt"):
        for Rr,mk in ((-1.0,"o"),(0.0,"s")):
            m=(notch==k)&(R==Rr)&~ro
            ax.scatter(np.exp(lnN[m]),d[m],s=20,marker=mk,facecolors=C[k] if Rr==0 else "none",edgecolors=C[k],lw=0.9,alpha=0.85,label=f"{LAB[k]}, R = {Rr:g} (held-out specimen, Figs 5-7 stress)")
        m=(notch==k)&~ro; x=lnN[m]/math.log(10); p=np.polyfit(x,d[m],1); xs=np.sort(x)
        ax.plot(10**xs,np.polyval(p,xs),c=C[k],lw=1.2,label=f"{LAB[k]} trend (derived): bias {d[m].mean():+.3f}, rms {np.sqrt(np.mean(d[m]**2)):.3f}, {p[0]:+.3f}/decade")
    ax.axhline(0,c="k",lw=0.6); ax.set_xscale("log"); ax.set_xlabel("cycles to failure N"); ax.set_ylim(-0.45,0.35)
    ax.set_title(f"{title}; held-out rms {rec['held']['rms']:.3f}",fontsize=9); ax.legend(fontsize=6.8,frameon=False,loc="lower left",ncol=1)
axs[0].set_ylabel("held-out residual log10(S* / S_obs)\n(negative: model conservative)")
fig.text(0.5,0.005,"Both fitted on plain (Fig. 1 pool) + sharp V only; U r1 and U r3 never used. Maximum-likelihood point estimates, frozen log-normal scatter. Stored IgBEM bisector fields, no re-solve. Departure from the paper: R enters the damage law.",ha="center",fontsize=7)
fig.tight_layout(rect=(0,0.02,1,1))
for e in ("png","pdf"): fig.savefig(HERE + f"fig_sde_tcd_heldout.{e}",dpi=200)
