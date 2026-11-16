import sys; sys.path.insert(0,'/tmp/claude-0/audit2'); from fit2 import *
d=load_ys()
for triax in (False, True):
    names,B,F,kap_of=ys_crack(d,triax=triax); lnc,nll=make_nll(d,names,F)
    r=json.load(open('res_ys_%s.json'%('triax' if triax else 'base'))); x=np.array([r['par'][n] for n in names])
    ia=names.index('a0'); print('triax',triax,'nll',nll(x),' a0 scan:',[round(nll(np.r_[x[:ia],x[ia]+h,x[ia+1:]])-nll(x),5) for h in (-1e-3,-1e-4,1e-4,1e-3)])
    iu=names.index('u'); lo=np.array([B[n][0] for n in names]); hi=np.array([B[n][1] for n in names])
    for u in (0.8,0.9,0.95,0.99,1.0):
        idx=[i for i in range(len(names)) if i!=iu]
        f=lambda y: nll(np.insert(y,iu,u)); o=minimize(f,np.delete(x,iu),method='L-BFGS-B',bounds=[(lo[i],hi[i]) for i in idx]); o=minimize(f,o.x,method='Powell',bounds=[(lo[i],hi[i]) for i in idx])
        p=dict(zip(names,np.insert(o.x,iu,u))); print('   u %.2f amax %.3f nll %.4f'%(u,p['a0']+kap_of(p)*d.dv.max(),o.fun), {k:round(v,3) for k,v in p.items()})
