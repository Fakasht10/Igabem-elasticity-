import sys, json, numpy as np, time; sys.path.insert(0,'/tmp/claude-0/audit'); from fit import *
v=sys.argv[1]; ns=int(sys.argv[2]); tag=sys.argv[3]
bo={'wide':dict(n=(-5,5),kap=(0,60),a0=(0,20),gam=(-1,2),k=(1,40),c=(-0.95,10),g1=(-3,3),b=(-20,20),Lc=(0.005,12))}.get(tag.split('_')[0],None)
con='con' in tag
t=time.time(); r=fitvar(v,bounds_over=bo,nstart=ns,constrained=con); st=stats(r)
out=dict(aic=r['aic'],nll=r['nll'],par=r['par'],kap=r['kap'],amax_fit=r['amax_fit'],amax_all=r['amax_all'],on_bound=r['on_bound'],starts=r['nll_starts'],**{k:x for k,x in st.items() if k!='e'})
json.dump(out,open(f'/tmp/claude-0/audit/res_{v}_{tag}.json','w'),indent=1,default=float)
np.save(f'/tmp/claude-0/audit/e_{v}_{tag}.npy',st['e'])
print(v,tag,'%.1fs'%(time.time()-t),'AIC %.3f'%r['aic'],{k:round(float(x),4) for k,x in r['par'].items()},'onb',r['on_bound'],'amax',round(r['amax_all'],2),r['nll_starts'],flush=True)
