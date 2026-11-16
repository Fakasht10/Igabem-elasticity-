"""Cross-dataset prediction: fit each criterion's parameters on one PLA data set, predict every test of the other.

    python3 -m fatigue.cross [--out results/fatigue]

Each data set's own plain S-N data are always used (they are what a designer has); only the notch criterion's
parameters cross over. Plain and notched share R within each data set, so the Walker term cancels and needs no gamma.
    Seifollahi & Kabir: life of each of the 14 notched specimens from the smooth Basquin curve (sn_transfer.life).
    Ezeh & Susmel:      life of each of the 149 failed notched specimens (vector-extracted from the 2020 Figs 5-7) from the
                        plain curve at the same R pooled over print angle (2019 Table 1 series fits).
Score: bias and rms of log10(N_pred / N), and the share within a factor 2 in life. Fit: least squares of the same.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from . import criteria as C
from .master import SPEC, factor
from .sn_transfer import basquin, life
from .support_fit import ezeh_setup
from .fields import field
from .specimens import Specimen
from .transfer import HERE, OUT

NAMES = ["kt", "hotspot", "tcd_point", "tcd_line", "weibull", "sweibull"]


class Ezeh:
    """Every notched specimen of Ezeh & Susmel 2020 (vector-extracted from Figs 5-7, run-outs left out); its life is
    predicted from the plain curve at the same R pooled over print angle (2019 Table 1: geometric-mean sigma_A50 at 2e6,
    mean k), N = 2e6 (f_eff s / s0_R)^(-k_R)."""
    name = "Ezeh-Susmel"

    def __init__(self, gauge=30.0, blunt=True):
        self.E, self.V, self.F, self.s0, _ = ezeh_setup(gauge)
        self.k = {R: float(np.mean([q["k"] for q in self.E["plain"] if q["R"] == R])) for R in self.s0}
        sp = json.load(open(os.path.join(HERE, "data", "ezeh_susmel_2020_specimens.json")))
        self.spec = [(q["notch"], q["R"], t["sigma_na"], t["N"]) for q in sp["series"] for t in q["specimens"]
                     if not t["runout"] and (blunt or q["notch"] != "blunt")]

    def errors(self, crit, th):
        out = []
        fs = {k: factor(crit, f, th, self.V) / factor(crit, None, th, self.V) for k, f in self.F.items()}
        for notch, R, s, N in self.spec:
            f = fs[notch]
            out.append(math.log10(self.E["N_ref"] * (f * s / self.s0[R]) ** (-self.k[R]) / N))
        return np.array(out)


class Seif:
    name = "Seifollahi"

    def __init__(self):
        self.spec = None
        S = json.load(open(os.path.join(HERE, "data", "seifollahi_kabir.json")))
        ps = S["plain_specimen"]
        self.V = ps["W"] * ps["T"] * 50.0
        self.raw = S["raw_lives"]
        self.A, self.B = basquin(self.raw["smooth"])
        self.F = {k: field(Specimen.from_dict(v), h=1.5, log=lambda *_: None) for k, v in S["specimens"].items()}

    def errors(self, crit, th):
        out = []
        for k, f in self.F.items():
            w = factor(crit, f, th, self.V) / factor(crit, None, th, self.V)
            out += [math.log10(life(lambda N: w, s, self.A, self.B) / n) for s, *ns in self.raw[k] for n in ns]
        return np.array(out)


def fit(ds, crit):
    names, th0 = SPEC[crit]
    if not names:
        return []
    lo = {"m": 2.0, "l": 1e-3, "L": 1e-2}
    hi = {"m": 1e3, "l": 20.0, "L": 10.0}
    clip = lambda z: [min(max(math.exp(v), lo[n]), hi[n]) for v, n in zip(z, names)]   # physical ranges
    obj = lambda z: float(np.mean(ds.errors(crit, clip(z)) ** 2))
    best = min((minimize(obj, np.log(th0) + math.log(s), method="Nelder-Mead",
                         options=dict(xatol=1e-3, fatol=1e-7, maxiter=400)) for s in (0.3, 1.0, 3.0)),
               key=lambda o: o.fun)
    return clip(best.x)


def floor(ds):
    """In-sample floor: a free constant notch factor per notch geometry (no criterion)."""
    kf = {}
    for k in ds.F:
        o = minimize(lambda z: float(np.mean(_free(ds, k, math.exp(z[0])) ** 2)), [0.1], method="Nelder-Mead")
        kf[k] = math.exp(o.x[0])
    return np.concatenate([_free(ds, k, v) for k, v in kf.items()]), kf


def _free(ds, k, w):
    if isinstance(ds, Ezeh):
        return np.array([math.log10(ds.E["N_ref"] * (w * s / ds.s0[R]) ** (-ds.k[R]) / N)
                         for n, R, s, N in ds.spec if n == k])
    return np.array([math.log10(life(lambda N: w, s, ds.A, ds.B) / n) for s, *ns in ds.raw[k] for n in ns])


def stats(e):
    return f"{e.mean():+.2f} / {np.sqrt(np.mean(e ** 2)):.2f}, {int((abs(e) < math.log10(2)).sum())}/{len(e)}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    sets = [Ezeh(), Seif()]
    lines = ["# Cross-dataset prediction: criterion fitted on one data set, every test of the other predicted", "",
             "Cells: bias / rms log10(N_pred/N), count within x2 in life.", "",
             "| criterion | fitted on | parameters | on the fit set (in-sample) | on the other set (predicted) |",
             "|---|---|---|---|---|"]
    res = {}
    for crit in NAMES:
        for i, src in enumerate(sets):
            tgt = sets[1 - i]
            th = fit(src, crit)
            par = ", ".join(f"{p} {v:.3g}" for p, v in zip(SPEC[crit][0], th)) or "-"
            ein, eout = src.errors(crit, th), tgt.errors(crit, th)
            res[(crit, src.name)] = dict(th=th, insample=ein.tolist(), predicted=eout.tolist())
            lines.append(f"| {crit} | {src.name} | {par} | {stats(ein)} | {stats(eout)} |")
    lines += ["", "In-sample floor, a free constant notch factor per notch geometry (no criterion can beat it):"]
    for ds in sets:
        e, kf = floor(ds)
        lines.append(f"- {ds.name}: {stats(e)}; Kf " + ", ".join(f"{k} {v:.3f}" for k, v in kf.items()))
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "cross_dataset.md"), "w").write("\n".join(lines) + "\n")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(os.path.join(a.out, "cross_dataset.json"), "w"),
              indent=1)
    print("\n".join(lines))
    return res


if __name__ == "__main__":
    main()
