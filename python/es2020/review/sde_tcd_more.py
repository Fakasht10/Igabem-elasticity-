import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Follow-up variants: plain batch constant with a bead-fixed L, and a separate Walker exponent for the plain gauge."""
import math, json, numpy as np
import sde_tcd as T
from sde_tcd import V, fit, report, notch, R, isplain

class V2(V):
    """Walker-Basquin with gamma_plain separate from gamma (notch)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.names = self.names + ["gamp"]; self.free = [n for n in self.names if n not in self.fixed]
    def ok(self, p): return super().ok(p) and 0 <= p["gamp"] <= 1
    def lnc_at(self, p, se, Rr, lc):
        g = np.where(lc, p["gamp"], p["gam"])
        return p["k"] * (p["lnsD"] - np.log(se) - g * np.log((1 - Rr) / 2))

fitmask = np.isin(notch, ("plain", "sharp")); res = []
wb = lambda sD, k, g, sZ: [math.log(sD), k, g, math.log(sZ)]
starts8 = [wb(sD, k, g, 1.3) + [b] for sD in (60, 90, 130) for k in (4.5, 6.0) for g in (0.4, 0.55) for b in (0.0, 0.2, 0.4)]
todo = [
 ("8 TCD L=0.4 fixed, plain free, WB", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.4))), starts8),
 ("8b same, L=0.2 fixed", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.2))), starts8),
 ("8c same, L=0.8 fixed", V("tcd", "wb", "free", fixed=dict(lnL=math.log(0.8))), starts8),
 ("10 TCD const L, WB, gamma split", V2("tcd", "wb"), [wb(70, 5.5, 0.55, 1.4) + [math.log(L), 0.35] for L in (0.6, 1.5)]),
 ("11 TCD L=0.4, plain free, WB, gamma split", V2("tcd", "wb", "free", fixed=dict(lnL=math.log(0.4))), [s + [0.35] for s in starts8[::3]]),
 ("12 TCD L free, plain free, WB, gamma split", V2("tcd", "wb", "free"), [wb(90, 5, 0.55, 1.3) + [math.log(L), b, 0.35] for L in (0.3, 0.6, 1.5) for b in (0.0, 0.3)]),
]
for name, v, st in todo:
    x, f = fit(v, [np.array(s) for s in st], fitmask); res.append(report(name, v, x, f))
json.dump(res, open(HERE + "sde_tcd_more_results.json", "w"), indent=1)
