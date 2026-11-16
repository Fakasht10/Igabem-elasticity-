import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Lumped self-heating, ONE law for every specimen (plain and notched).
Thin plate cooled on both faces (lumped through the thickness, Biot ~ 0.15): k t lap(T) - 2 h T + q t = 0
  => Delta T(x0) = (1 / 2 pi k) * int q(x) K0(|x - x0| / ell) dA,   ell = sqrt(k t / 2 h)   (cooling length)
  q = f pi tan(delta) sigma_a(x)^2 / E   (viscoelastic loss, amplitude^2)
Uniform field: Delta T = q ell^2 / k = q t / (2 h). So with H = Delta T / Delta T_uniform(net amplitude):
  H(x0) = (1 / 2 pi ell^2) * int sigma_hat(x)^2 K0(|x - x0|/ell) dA        (sigma_hat = field per unit net stress)
plain gauge: H = 1.  Hot spot on a free (adiabatic) edge: image source, factor 2 over the material half-plane.
Damage rate x exp(beta Delta T)  =>  ln c -> ln c - theta * sigma_a,net^2 * H,  theta = beta f pi tan(delta) t / (2 h E) >= 0.
Latent thermal constants: theta (>= 0) and ell (> 0). sigma_hat = sigma_1 from the stored polar grid (r <= 5.9 mm), far field
(gross section, sigma_hat = 6/25) added analytically."""
import math, json, sys, numpy as np
from scipy.special import k0, k1
import sde_tcd as T
from sde_tcd import V, fit, report, notch, R, S, isplain
from es2020 import methods as M, paper

def H(n, ell):
    a = M.fields()[n]["area"]; r = np.array(a["r"]); th = np.array(a["th"]); wth = np.array(a["wth"]); s1 = np.array(a["s1"])   # (r, th)
    if s1.shape != (len(r), len(th)): s1 = s1.T
    f = (np.maximum(s1, 0) ** 2 * wth[None, :]).sum(1) * 2          # both sides of the bisector: integral over the material half-plane angle
    g = f * k0(r / ell) * r
    near = np.trapezoid(g, r) + g[0] * r[0] / 2
    Rm = r[-1]; far = (6 / 25) ** 2 * math.pi * ell ** 2 * (Rm / ell) * k1(Rm / ell)   # int_Rm^inf K0(r/ell) r dr = ell^2 (Rm/ell) K1
    return 2 * (near + far) / (2 * math.pi * ell ** 2)

class VH(V):
    def __init__(self, ell, *a, **k):
        super().__init__(*a, **k); self.ell = ell
        self.Hn = np.ones(len(S))
        for n in paper.NOTCHES: self.Hn[notch == n] = H(n, ell)
        self.names = self.names + ["lnth"]; self.free = [n for n in self.names if n not in self.fixed]
    def ok(self, p): return super().ok(p) and -14 < p["lnth"] < 0
    def lnc(self, p, s=None):
        s0 = S if s is None else s
        return super().lnc(p, s) - math.exp(p["lnth"]) * (s0 * (1 - R) / 2) ** 2 * self.Hn

if __name__ == "__main__":
    for ell in (2.5, 3.9, 6.0):
        print("ell %.1f mm: H plain 1.00, " % ell + ", ".join(f"{paper.LABEL[n]} {H(n, ell):.2f}" for n in paper.NOTCHES), flush=True)
    fitmask = np.isin(notch, ("plain", "sharp")); res = []
    wb = lambda sD, k, g, sZ: [math.log(sD), k, g, math.log(sZ)]
    L4 = dict(lnL=math.log(0.4)); ths = (math.log(1e-4), math.log(1.5e-3), math.log(1e-2))
    for ell in (3.9, 2.5, 6.0):
        todo = [(f"T1 lumped heating all, plain tied, ell {ell}", VH(ell, "tcd", "wb", fixed=L4), [wb(sD, k, 0.45, 1.4) + [t] for sD in (80, 160) for k in (4.5, 6) for t in ths])]
        if ell == 3.9:
            todo.append((f"T2 lumped heating all + plain lot free, ell {ell}", VH(ell, "tcd", "wb", "free", fixed=L4), [wb(sD, 5.3, 0.45, 1.4) + [b, t] for sD in (100, 160) for b in (0.0, 0.4, 0.7) for t in ths]))
            todo.append((f"T3 lumped heating all, L free, plain tied, ell {ell}", VH(ell, "tcd", "wb"), [wb(sD, 5.3, 0.45, 1.4) + [math.log(L), t] for sD in (70, 120) for L in (0.4, 1.0, 1.5) for t in ths]))
        for name, v, st in todo:
            x, f = fit(v, [np.array(s) for s in st], fitmask); r = report(name, v, x, f); th = math.exp(v.par(x)["lnth"])
            print("    theta %.2e /MPa^2: plain life factor at sa 7 / 15 / 25 MPa: %.2f / %.2f / %.2f" % (th, *[math.exp(-th * a * a) for a in (7, 15, 25)]), flush=True)
            r["theta"] = th; r["ell"] = ell; res.append(r)
    json.dump(res, open(HERE + "sde_tcd_heat2_results.json", "w"), indent=1)
