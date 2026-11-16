# recompute stored statistics at the STORED parameters, with their F (24-pt rule) and with my F (GL-256), using my scoring
import sys, json, math, numpy as np, os
sys.path.insert(0,'/tmp/claude-0/audit'); import fit as M
sys.path.insert(0,'/mnt/project-files/yaren-susmel-2025'); sys.path.insert(0,'/mnt/project-files/review/defects')
import kchi as K
from scipy.stats import norm
d=K.d; Y='/mnt/project-files/yaren-susmel-2025/'
def sc(lp,sZ):
    le=d.lnN/math.log(10); e=le-lp; hw=norm.ppf(.9)*sZ/math.log(10); m=~d.ro
    def s(mm):
        mm=mm&m; r=e[mm]; return dict(rms=r.std()*0+np.sqrt(np.mean(r**2)),bias=r.mean(),tilt=np.polyfit(le[mm],r,1)[0],tilt_pred=np.polyfit(lp[mm],r,1)[0],cov=100*np.mean(np.abs(r)<=hw))
    u1,u3=m&(d.g=='U1'),m&(d.g=='U3'); return s(d.fit),s(~d.fit),e[u1].mean()-e[u3].mean()
fmt=lambda s:'rms %.3f bias %+.3f tilt %+.3f tiltp %+.3f cov %.1f'%(s['rms'],s['bias'],s['tilt'],s['tilt_pred'],s['cov'])
J=json.load(open(Y+'kchi.json'))['variants']
for lab,(kc,rm) in [('BASELINE',(False,False)),('+ R-1 triaxiality term',(False,True)),('+ k(chi)',(True,False)),('+ k(chi) + R-1 term',(True,True))]:
    p=J[lab]['par']; F=K.make(kc,rm)[3](p); lp=p['k']*(p['lnsD']-np.log(d.S*F*K.w**p['gam']))/math.log(10)
    f,h,gap=sc(lp,math.exp(p['lnsZ'])); print('kchi',lab,'\n  mine@stored FIT',fmt(f),'| HELD',fmt(h),'gap %+.3f'%gap)
    print('  stored        FIT',fmt(J[lab]['fit']),'| HELD',fmt(J[lab]['held']),'gap %+.3f'%J[lab]['U1_minus_U3']['all'])
# gamma_chi variants: rebuild gchi with their F
G=json.load(open(Y+'gamma_chi.json'))['variants']
for lab in G:
    p=G[lab]; gm=p['gamma']; par=p['par']
    F=K.make(False,False)[3](par); gam=np.array([gm[x] for x in d.g])
    lp=par['k']*(par['lnsD']-np.log(d.S*F*K.w**gam))/math.log(10)
    f,h,gap=sc(lp,math.exp(par['lnsZ'])); print('gchi',lab,'\n  mine@stored FIT',fmt(f),'| HELD',fmt(h),'gap %+.3f'%gap)
    print('  stored        FIT rms %.3f bias %+.3f tilt %+.3f cov %.1f | HELD rms %.3f bias %+.3f tilt %+.3f cov %.1f'%tuple([p['fit'][k] for k in ('rms','bias','tilt','cov')]+[p['held'][k] for k in ('rms','bias','tilt','cov')]))
Rm=json.load(open(Y+'rminus.json'))
for lab,v in Rm['variants'].items():
    par=v['par']; F=K.make(False,False)[3]({k:x for k,x in par.items() if k!='b'}); beta=np.array([v['beta'][x] for x in d.g])
    lp=par['k']*(par['lnsD']-np.log(d.S*F*(1+beta*K.comp)*K.w**par['gam']))/math.log(10)
    f,h,gap=sc(lp,math.exp(par['lnsZ'])); print('rminus',lab,'\n  mine@stored FIT',fmt(f),'| HELD',fmt(h),'gap %+.3f'%gap)
    print('  stored        FIT rms %.3f bias %+.3f tilt %+.3f cov %.1f | HELD rms %.3f bias %+.3f tilt %+.3f cov %.1f'%tuple([v['fit'][k] for k in ('rms','bias','tilt','cov')]+[v['held'][k] for k in ('rms','bias','tilt','cov')]))
