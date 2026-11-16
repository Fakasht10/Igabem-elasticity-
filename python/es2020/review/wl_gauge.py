import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Weakest link over volume on Ezeh-Susmel with the plain gauge volume FIXED (6 x 3 mm section x assumed gauge length), m tied to scatter.
No free plain constant: 4 constants (sD, k, gam, sZ). Fit own-lab plain + V, hold out U."""
import sys, math, json, numpy as np
sys.path.insert(0, HERE); import rank
from rank import per_group, run, wl
d = rank.load_es(); res = []
strict = lambda p: max(1.5, 1.2825 * p["k"] / math.exp(p["lnsZ"]))
for Lg in (10, 20, 33, 40, 60, 100):
    V = 6.0 * 3.0 * Lg
    def build(p, V=V):
        m = strict(p); return per_group(d, lambda g, dv, ph, lv: V ** (1 / m) if g == "plain" else wl(d, g, m, 0))
    r = run(d, "weakest link, plain gauge %d mm fixed (V %.0f mm3)" % (Lg, V), ([], [[]], {}, build)); r["m"] = strict(r["par"]); print("   m =", round(r["m"], 2)); res.append(r)
json.dump(res, open(HERE + "wl_gauge.json", "w"), indent=1)
