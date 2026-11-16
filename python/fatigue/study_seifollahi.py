"""Seifollahi & Kabir 2026 (YZ PLA, laser-cut double edge notch and hole R2, R 0.05) at N = 2e4: Weibull against m,
the m and TCD lengths that reproduce each notch, cross-calibration edge <-> hole, and TCD with the Ezeh-Susmel
L(N) = 16.4 N^-0.242 mm (another PLA, printer and lab: a blind cross-dataset check).

    cd python && python3 -m fatigue.study_seifollahi        (fields cached by fatigue.transfer; < 5 s)
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import brentq
from fatigue.specimens import Specimen
from fatigue.fields import field
from fatigue import criteria as C
d=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'seifollahi_kabir.json'))); ps=d['plain_specimen']; V=ps['W']*ps['T']*50
F={k:field(Specimen.from_dict(v),h=1.5,log=lambda *a:None) for k,v in d['specimens'].items()}
s0=d['plain'][0]['sigma_A50']
meas={r['notch']:r['sigma_nA50'] for r in d['notched']}
print('Weibull vs m:'); 
for m in (5,7.5,10,15,20,30,119):
    print('  m %5.1f '%m+'  '.join('%s %.2f (%+.0f%%)'%(k,C.weibull(F[k],s0,m,V),100*(C.weibull(F[k],s0,m,V)/meas[k]-1)) for k in F))
for k in F:
    mfit=brentq(lambda m: C.weibull(F[k],s0,m,V)-meas[k],1.0,200)
    x,s=C.line_profile(F[k]); t=s0/meas[k]
    j=int(np.argmax(s<t)); Lpm=2*np.interp(t,s[j::-1],x[j::-1])
    Llm=brentq(lambda L: C.tcd_line(F[k],s0,L)-meas[k],1e-3,x[-1]*0.99)
    print(k,'Kt %.2f Kf %.3f  m that matches %.2f   TCD L that matches: PM %.2f mm, LM %.2f mm'%(float(F[k]['Kt_mean']),s0/meas[k],mfit,Lpm,Llm))
# cross: L from edge -> predict hole and vice versa
for a,b in (('edge','hole'),('hole','edge')):
    x,s=C.line_profile(F[a]); t=s0/meas[a]; j=int(np.argmax(s<t)); L=2*np.interp(t,s[j::-1],x[j::-1])
    p=C.tcd_point(F[b],s0,L); 
    m=brentq(lambda m: C.weibull(F[a],s0,m,V)-meas[a],1.0,200); pw=C.weibull(F[b],s0,m,V)
    print(f'calibrate on {a}, predict {b}: TCD PM {p:.2f} ({100*(p/meas[b]-1):+.0f}%), Weibull m {m:.2f}: {pw:.2f} ({100*(pw/meas[b]-1):+.0f}%)')
L = 16.4 * 2e4 ** -0.242
print(f"Ezeh-Susmel L(2e4) = {L:.2f} mm")
for k in F:
    for n, v in (("PM", C.tcd_point(F[k], s0, L)), ("LM", C.tcd_line(F[k], s0, L))):
        print(f"  {k} TCD {n}: {v:.2f} ({100 * (v / meas[k] - 1):+.0f}%)")
