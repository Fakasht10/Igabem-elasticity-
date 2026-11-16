import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Combined candidate: weakest link over the crack-amplified, defect-averaged field.
F = phi^-n * W(a, L/2) * ( sum_V w * sbar_a^m )^(1/m), sbar_a = sigma_1 averaged over a disc of radius a (the defect size), W = Westergaard
amplification of a crack of half-length a read at L/2 (L = 0.4 mm), m tied to the life scatter (m = 1.2825 k / sZ) or free.
YS: a = a0 + kap dv. ES: a fixed (plain + V identify one constant only, and the plain volume is already free: HIDDEN FITTED CONSTANT lnVp)."""
import sys, math, json, numpy as np
sys.path.insert(0, HERE); import rank
from rank import W, L0, per_group, run
T = dict(np.load(HERE + "combo_tables.npz"))
def lnsum(ds, g, a, m):
    A = T[ds + "_A"]; a = min(max(a, 0.0), A[-1]); i = min(np.searchsorted(A, a, side="right") - 1, len(A) - 2); t = (a - A[i]) / (A[i + 1] - A[i])
    far = float(np.sum(T[f"{ds}_{g}_far_w"] * T[f"{ds}_{g}_far_s"] ** m)); r0 = float(np.sum(T[f"{ds}_{g}_w0"] * T[f"{ds}_{g}_s0"] ** m))
    v = [math.log(max(far - r0 + float(np.sum(T[f"{ds}_{g}_w{j}"] * T[f"{ds}_{g}_s{j}"] ** m)), 1e-12)) for j in (i, i + 1)]
    return (1 - t) * v[0] + t * v[1]
def builder(d, afun, mfun):
    def build(p):
        m = mfun(p)
        def f(g, dv, ph, lv):
            a = afun(p, dv); u = ph ** -p["n"] if "n" in p else 1.0
            if g == "plain" and d.name == "ES": return W(a, L0 / 2) * math.exp(p["lnVp"] / m)
            return u * W(a, L0 / 2) * math.exp(lnsum(d.name, g, a, m) / m)
        return per_group(d, f)
    return build
strict = lambda p: max(1.5, 1.2825 * p["k"] / math.exp(p["lnsZ"]))
res = []
for w in (sys.argv[1:] or ["ES", "YS"]):
    d = rank.load_es() if w == "ES" else rank.load_ys()
    if w == "ES":
        for a in (0.0, 0.2, 0.41, 0.8, 1.2):
            res.append(run(d, "combined, a = %.2f fixed, m tied, plain volume free" % a, (["lnVp"], [[5.5], [7.0]], dict(lnVp=(-5, 40)), builder(d, lambda p, dv, a=a: a, strict))))
    else:
        res.append(run(d, "combined, a = a0 + kap dv, m tied", (["a0", "kap", "n"], [[0.4, 2.5, 0.3], [0.1, 1.0, 0.6], [1.0, 0.5, 0.5]], dict(a0=(0, 6), kap=(0, 6), n=(0, 5)), builder(d, lambda p, dv: p["a0"] + p["kap"] * dv, strict))))
        res.append(run(d, "combined, a = a0 + kap dv, m free", (["a0", "kap", "n", "m"], [[0.4, 2.5, 0.3, 5.0], [0.1, 1.0, 0.6, 8.0]], dict(a0=(0, 6), kap=(0, 6), n=(0, 5), m=(1.5, 60)), builder(d, lambda p, dv: p["a0"] + p["kap"] * dv, lambda p: p["m"]))))
        res.append(run(d, "combined, a = kap dv (no intrinsic), m tied", (["kap", "n"], [[2.5, 0.3], [1.0, 0.6]], dict(kap=(0, 6), n=(0, 5)), builder(d, lambda p, dv: p["kap"] * dv, strict))))
    json.dump(res, open(HERE + "combo_%s.json" % w, "w"), indent=1); res = []
