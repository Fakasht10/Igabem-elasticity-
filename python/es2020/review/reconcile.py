import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""One basis for the two leading geometry-free rows on Ezeh-Susmel: lognormal Walker-Basquin, ML, fit plain + V, hold out U (Figs 5-7 stresses).
(a) SDE thread's automatic-path finite-width crack: F = sigma_K(a) F_SENT(a/W)/1.12, W = 3 mm plain (thickness), W = 6 mm notches (ligament), a free.
(b) weakest link over volume, m tied to scatter, plain gauge 33 mm fixed (ASTM D638 Type IV assumption), no extra constant.
Each on own-lab plain (125) and on the pooled 195 plain points."""
import sys, math, json, numpy as np
sys.path.insert(0, HERE); import rank
from rank import per_group, run, wl, wf_mean
from es2020 import sde
strict = lambda p: max(1.5, 1.2825 * p["k"] / math.exp(p["lnsZ"]))
def pooled():
    import sde_tcd as s
    d = rank.load_es(); nm = {"plain": "plain", "sharp": "V", "intermediate": "U1", "blunt": "U3"}
    d.g = np.array([nm[x] for x in s.notch]); d.R, d.S, d.lnN, d.ro = s.R, s.S, s.lnN, s.ro
    n = len(d.S); d.dv = np.zeros(n); d.phi = np.ones(n); d.lev = np.zeros(n, int); d.fit = np.isin(d.g, ("plain", "V")); return d
res = []
for tag, d in (("own-lab 125", rank.load_es()), ("pooled 195", pooled())):
    def crackfw(p, d=d, Wp=3.0, Wn=6.0):
        a = p["a0"]
        return per_group(d, lambda g, dv, ph, lv: float(sde.f_sent(a / Wp)) / 1.12 if g == "plain" else float(wf_mean(d, g, a)[0]) * float(sde.f_sent(a / Wn)) / 1.12)
    def crackfw_same(p, d=d): return crackfw(p, d, 6.0, 6.0)
    def wl33(p, d=d):
        m = strict(p); return per_group(d, lambda g, dv, ph, lv: 594.0 ** (1 / m) if g == "plain" else wl(d, g, m, 0))
    for name, spec in (("crack, finite width, plain through thickness W 3 (SDE thread rule)", (["a0"], [[0.3], [1.0], [1.8], [2.4]], dict(a0=(1e-3, 2.9)), crackfw)),
                       ("crack, finite width, same W 6 for plain and notch", (["a0"], [[0.3], [1.0], [1.8], [2.4]], dict(a0=(1e-3, 2.9)), crackfw_same)),
                       ("weakest link, 33 mm gauge fixed", ([], [[]], {}, wl33))):
        r = run(d, tag + " | " + name, spec); res.append(r)
json.dump(res, open(HERE + "reconcile.json", "w"), indent=1)
