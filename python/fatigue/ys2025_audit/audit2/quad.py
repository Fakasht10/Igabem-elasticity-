import sys, math, numpy as np
for p_ in ('/home/user/wt_uxm/python', '/mnt/project-files/review', '/mnt/project-files/review/defects'): sys.path.insert(0, p_)
import rank as RK
from scipy.integrate import quad
C='/home/user/igabem-elasticity-/cache/ys2025/'
def ref_exact(xs, ys, a):
    # exact (2/pi) int_0^a K(x)/sqrt(a^2-x^2) dx for piecewise-linear K (np.interp clamped)
    br=np.unique(np.r_[0.0, xs[(xs>0)&(xs<a)], a]); tot=0.0
    for x0,x1 in zip(br[:-1],br[1:]):
        k0,k1=np.interp([x0,x1],xs,ys); be=(k1-k0)/(x1-x0); al=k0-be*x0
        tot+=al*(math.asin(min(x1/a,1))-math.asin(x0/a)) - be*(math.sqrt(max(a*a-x1*x1,0))-math.sqrt(a*a-x0*x0))
    return 2/math.pi*tot
def ref(K, a):
    # exact for piecewise-linear K: integrate in x with weight 1/sqrt(a^2-x^2), breakpoints at nodes, using theta substitution per segment
    f=lambda th: float(K(a*math.sin(th)))
    return 2/math.pi*quad(f,0,math.pi/2,limit=2000,epsabs=1e-13,epsrel=1e-12)[0]
def getXS(d,g,ds):
    import json, sde_tcd as s
    if ds=='ES': inv={'V':'sharp','U1':'intermediate','U3':'blunt'}; return s.PROF[inv[g]].d, s.PROF[inv[g]].s
    if ds=='YS3D': z=np.load(C+'profile_%s_h0.npz'%{'plain':'plain','V':'V_r0.15','U1':'U_r1','U3':'U_r3'}[g]); return z['d'], z['s1'][-1]
    P=json.load(open('/mnt/project-files/yaren-susmel-2025/profiles.json')); v=P[{'V':'V_r0.15','U1':'U_r1','U3':'U_r3'}[g]][-1]; return v['r'],v['K']
def dense(K,a,n=200000):
    th=(np.arange(n)+0.5)/n*math.pi/2; return float(np.mean(K(a*np.sin(th))))
def mid24(K,a):
    th=(np.arange(24)+0.5)/24*math.pi/2; return float(np.mean(K(a*np.sin(th))))
for ds in ('YS2D','YS3D','ES'):
    if ds=='ES': d=RK.load_es(); gs=('V','U1','U3')
    else:
        d=RK.load_ys(); gs=('V','U1','U3')
        if ds=='YS3D':
            NM={'plain':'plain','V_r0.15':'V','U_r1':'U1','U_r3':'U3'}
            fld={NM[k]:(np.load(C+f'profile_{k}_h0.npz')['d'],np.load(C+f'profile_{k}_h0.npz')['s1'][-1]) for k in NM}
            d.K=lambda g,x: np.interp(np.clip(np.asarray(x,float),0,d.b),*fld[g]); gs=('plain','V','U1','U3')
    worst={}
    XS={g:(np.asarray(xx,float),np.asarray(yy,float)) for g in gs for xx,yy in [getXS(d,g,ds)]}
    for g in gs:
        K=lambda x,g=g: d.K(g,x)
        E=[]; E24=[]
        for a in np.r_[np.linspace(0.005,0.1,8),np.linspace(0.1,d.b-0.2,40)]:
            r=ref_exact(XS[g][0],XS[g][1],a); dn=dense(K,a); gl=float(RK.wf_mean(d,g,a)[0])
            E.append((abs(gl/r-1),a)); E24.append(mid24(K,a)/r-1); assert abs(dn/r-1)<1e-6,(dn,r)
        print(ds,g,'GL128 max rel err %.2e at a=%.3f'%max(E),' old mid24 range %.3f%% .. %.3f%%'%(100*min(E24),100*max(E24)))
