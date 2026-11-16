import numpy as np
C='/home/user/igabem-elasticity-/cache/ys2025/'
GEO={'plain':'plain','V':'V_r0.15','U1':'U_r1','U3':'U_r3'}
L0=0.4
def field(g):
    z=np.load(C+f'profile_{GEO[g]}_h0.npz'); assert abs(z['z'][-1]-2.5)<1e-12
    return z['d'].astype(float), z['s1'][-1].astype(float)
def chi(g, dmax=0.02, deg=2, use_fit_c0=False):
    d,K=field(g); m=d<dmax; c=np.polyfit(d[m],K[m],deg)
    return -c[-2]/(c[-1] if use_fit_c0 else K[0])
def triax(g):
    t=np.load(C+'triax_bisector.npz'); k=GEO[g]; assert t[k+'_z'][0]==2.5
    S=t[k+'_sig'][0]   # (nd,3,3)
    sm=np.trace(S,axis1=1,axis2=2)/3; dev=S-sm[:,None,None]*np.eye(3)
    vm=np.sqrt(1.5*np.sum(dev*dev,axis=(1,2)))
    return t[k+'_d'], sm/vm, t[k+'_h'][0], t[k+'_s1'][0]
if __name__=='__main__':
    for g in GEO:
        d,K=field(g); m=d<0.02
        print(g,'n(d<0.02)',m.sum(),'K0',K[0],'K(0.001)',K[1],
              'chi(/K0)',chi(g),'chi(/c0)',chi(g,use_fit_c0=True),'chi lin d<0.02',chi(g,deg=1),'chi quad d<0.05',chi(g,0.05),'chi quad d in(0,.02) excl root',
              -np.polyfit(d[1:][d[1:]<0.02],K[1:][d[1:]<0.02],2)[1]/K[0])
        dd,h,hs,s1=triax(g)
        print('   h_root mine',h[0],'stored',hs[0],'max|diff|',np.abs(h-hs).max(),' s1 match profile', np.abs(s1-K).max(), 'h_root-1/3',h[0]-1/3)
        hz=dd<=L0/2; print('   hL-1/3', np.trapezoid(h[hz],dd[hz])/(L0/2)-1/3, 'last d in zone', dd[hz][-1])
