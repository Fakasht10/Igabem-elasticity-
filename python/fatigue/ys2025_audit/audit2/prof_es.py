import sys; sys.path.insert(0,'/tmp/claude-0/audit2'); from fit2 import *
for pooled in (False, True):
    d=load_es(pooled); names,B,F,_=es_recon(d); lnc,nll=make_nll(d,names,F)
    for a0 in (0.3,0.6,0.8,0.9,0.95,1.0,1.1,1.3,1.6,2.0,2.5):
        Fa=lambda p: F({**p,'a0':a0}); n2=names[:4]; _,nl=make_nll(d,n2,lambda p: F({**p,'a0':a0}))
        best=min((minimize(nl,np.array(x0),method='Nelder-Mead',options=dict(maxiter=20000,xatol=1e-8,fatol=1e-10)) for x0 in ([5,5,0.45,0.2],[4.5,6,0.3,0.0])),key=lambda o:o.fun)
        Fp=F(dict(lnsD=0,k=5,gam=.5,lnsZ=0,a0=a0)); kfV=Fp[d.g=='V'][0]/Fp[d.g=='plain'][0]; kU1=Fp[d.g=='U1'][0]/Fp[d.g=='plain'][0]
        print(d.name,'a0 %.2f nll %.4f  F_V/F_plain %.3f F_U1/F_plain %.3f'%(a0,best.fun,kfV,kU1))
