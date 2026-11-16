import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Fig. 1 plain pool split by source. Other-lab points = 2020 Fig. 1 points (R -1, 0) that sit on a marker of Fig. 3 in
Ezeh & Susmel, Mat Design Process Comm 2019 e45 (Letcher R=-1 + Afrose R=0 only); see fig1_source_match.json (hit = marker coverage).
Refit as plain_split.py. Stored fields only."""
import math, json, numpy as np
from sde_tcd import V, fit, report, notch, R, S, lnN, ro, isplain, rows
mt = json.load(open(HERE + "fig1_source_match.json"))
hit = np.full(len(S), np.nan)
for i in np.where(isplain)[0]:
    c = [m for m in mt if m["R"] == R[i] and abs(math.log(m["N"]) - lnN[i]) < 2e-3 and abs(m["s"] * 42.9 / S[i] - 1) < 2e-3]
    if c: hit[i] = c[0]["hit"]
print("plain rows", int(isplain.sum()), "matched to Fig.1 list", int(np.isfinite(hit).sum()))
RR = np.isin(R, (-1, 0))
own = isplain & (~RR | (hit <= 0.3)); ownw = isplain & (~RR | (hit <= 0.8)); oth = isplain & RR & (hit > 0.8)
subs = {"own lab strict": own, "own lab + ambiguous": ownw, "other labs (Letcher+Afrose)": oth, "own lab, R -1 & 0 only": isplain & RR & (hit <= 0.3)}
for k, m in subs.items(): print(k, int(m.sum()), {r: int((m & (R == r)).sum()) for r in (-1, -0.5, 0, 0.3)}, "runouts", int((m & ro).sum()), "N %.0f-%.0f" % (math.exp(lnN[m].min()), math.exp(lnN[m].max())))
wb0 = [math.log(60), 6.0, 0.4, math.log(1.4)]; res = []
for k, m in subs.items():
    mask = m | (notch == "sharp")
    for nm, v, st in [("TCD L0.4 tied", V("tcd", "wb", "tied", fixed=dict(lnL=math.log(0.4))), [wb0, [math.log(30), 4.0, 0.3, math.log(1.4)]]),
                      ("TCD L0.4 free", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.4))), [wb0 + [0.0], wb0 + [0.7], [math.log(160), 5.3, 0.44, math.log(1.3), 0.7]]),
                      ("net free", V("peak", "wb", "free"), [wb0 + [0.1], [math.log(30), 5.0, 0.4, math.log(1.4), 0.2], [math.log(90), 5.0, 0.4, math.log(1.3), -0.2]])]:
        x, f = fit(v, [np.array(s) for s in st], mask); r = report(f"{k} | {nm}", v, x, f); r["subset"] = k
        d = v.resid(x); r["sub_plain"] = dict(n=int((m & ~ro).sum()), bias=float(d[m & ~ro].mean()), rms=float(np.sqrt((d[m & ~ro] ** 2).mean())))
        print("    fitted plain subset bias/rms %+.3f/%.3f" % (r["sub_plain"]["bias"], r["sub_plain"]["rms"])); res.append(r)
json.dump(res, open(HERE + "plain_split2_results.json", "w"), indent=1)
