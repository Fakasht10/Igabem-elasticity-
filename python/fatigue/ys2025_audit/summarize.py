import json,glob,os
rows=[('baseline','std'),('baseline','con'),('gchi','std'),('gchi','con'),('beta','std'),('beta','con'),('beta','mode_wide'),('kchi','std'),('kchi','con'),('kchi','mode_wide'),('Lfree','std')]
for v,t in rows:
    f=f'res_{v}_{t}.json'
    if not os.path.exists(f): continue
    r=json.load(open(f)); F,H,G=r['fit'],r['held'],r['U1_minus_U3']
    par={k:round(x,3) for k,x in r['par'].items()}
    print(f"{v:8s} {t:9s} AIC {r['aic']:.2f} nll {r['nll']:.3f} kap {r['kap']:.2f} amax {r['amax_all']:.2f} par {par}")
    print(f"   FIT rms {F['rms']:.3f} bias {F['bias']:+.3f} sl_exp {F['slope_vs_Nexp']:+.3f} sl_pred {F['slope_vs_Npred']:+.3f} cov {F['cov']:.1f} | HELD rms {H['rms']:.3f} bias {H['bias']:+.3f} sl_exp {H['slope_vs_Nexp']:+.3f} sl_pred {H['slope_vs_Npred']:+.3f} cov {H['cov']:.1f}")
    print("   U1-U3 all %+.3f | 100 %+.3f 80 %+.3f 60 %+.3f 40 %+.3f 20 %+.3f | R-1 %+.3f R0.1 %+.3f"%tuple(G[k] for k in ['all','100','80','60','40','20','R-1','R0.1']))
