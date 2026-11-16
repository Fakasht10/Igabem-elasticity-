import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Fig. 1 plain pool split by load ratio (the only source proxy the PDF carries). Stored fields only.
Refit: sigma_1 at L/2 (L=0.4 fixed) or net stress, Walker-Basquin, frozen lognormal; fit = plain subset + sharp V; hold out U r1, U r3."""
import math, json, numpy as np
from sde_tcd import V, fit, report, notch, R, S, lnN, ro, isplain
wb0 = [math.log(60), 6.0, 0.4, math.log(1.4)]
subs = {"all 195": isplain, "R -0.5 & 0.3 (authors only)": isplain & np.isin(R, (-0.5, 0.3)), "R -1 only": isplain & (R == -1), "R 0 only": isplain & (R == 0),
        "R -1 & 0": isplain & np.isin(R, (-1, 0)), "all, N>=1e4": isplain & (lnN >= math.log(1e4))}
for k, m in subs.items(): print(k, int(m.sum()), "runouts", int((m & ro).sum()), "N range %.0f-%.0f" % (math.exp(lnN[m].min()), math.exp(lnN[m].max())), "s/UTS %.2f-%.2f" % (S[m].min()/42.9, S[m].max()/42.9))
res = []
for k, m in subs.items():
    mask = m | (notch == "sharp")
    for nm, v, st in [("TCD L0.4 tied", V("tcd", "wb", "tied", fixed=dict(lnL=math.log(0.4))), [wb0, [math.log(30), 4.0, 0.3, math.log(1.4)]]),
                      ("TCD L0.4 free", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.4))), [wb0 + [0.0], wb0 + [0.7], [math.log(160), 5.3, 0.44, math.log(1.3), 0.7]]),
                      ("net tied", V("peak", "wb", "tied"), [wb0, [math.log(30), 5.0, 0.4, math.log(1.4)]]),
                      ("net free", V("peak", "wb", "free"), [wb0 + [0.0], [math.log(30), 5.0, 0.4, math.log(1.4), 0.2]])]:
        x, f = fit(v, [np.array(s) for s in st], mask); r = report(f"{k} | {nm}", v, x, f); r["subset"] = k; res.append(r)
json.dump(res, open(HERE + "plain_split_results.json", "w"), indent=1)
