import sys, json, numpy as np; sys.path.insert(0,'/tmp/claude-0/audit'); from fit import *
print('n',len(S),'fit',fit.sum(),'fit failures',(fit&~ro).sum(),'held failures',(~fit&~ro).sum(),'runouts fit/held',(fit&ro).sum(),(~fit&ro).sum())
print('CHI',CHI,'G',{k:gx(v*L0) for k,v in CHI.items()},'Dh',HX,'BMAX',BMAX)
OUT={}
for v in ['baseline','gchi','beta','kchi','Lfree']:
    r=fitvar(v,nstart=int(sys.argv[1]) if len(sys.argv)>1 else 30); st=stats(r)
    OUT[v]=dict(aic=r['aic'],nll=r['nll'],par=r['par'],on_bound=r['on_bound'],starts=r['nll_starts'],**{k:x for k,x in st.items() if k!='e'})
    print(v, json.dumps({k:x for k,x in OUT[v].items()},default=float,indent=None),flush=True)
json.dump(OUT,open('/tmp/claude-0/audit/refit.json','w'),indent=1,default=float)
