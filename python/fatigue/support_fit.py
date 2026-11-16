"""Fit the gradient-supported Weibull support length l on Ezeh & Susmel PLA and score its transfer to Seifollahi & Kabir.

    python3 -m fatigue.support_fit [--m 7.5 20 40 60 80 119] [--gauge-ezeh 30] [--out results/fatigue]

Supported Weibull (Maede et al.): sigma_1 is divided by n = 1 + sqrt(l chi), chi = |grad sigma_1| / sigma_1, inside the
weakest-link integral. For each fixed m, l is fitted by least squares in ln(predicted / measured) on all 18 Ezeh-Susmel
notched series (plain s0 pooled over print angle at the same R). The Ezeh-Susmel plain gauge length is not given in
the paper, so the plain volume uses an ASSUMED gauge (default 30 mm, as transfer.py's middle value).

Provenance of m: Ezeh-Susmel's own pooled scatter gives m ~7.5. Larger m values are not identified by Ezeh-Susmel
(its fit error is flat for m >= 20); m 119 is Seifollahi & Kabir's own smooth scatter (T 1.028), so a transfer at that m
uses one number from the target data set.
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize_scalar

from . import criteria as C
from .fields import field
from .sn_transfer import basquin, life
from .specimens import Specimen
from .transfer import HERE, OUT


def ezeh_setup(gauge):
    E = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))
    ps = E["plain_specimen"]
    V = ps["W"] * ps["T"] * gauge
    F = {k: field(Specimen.from_dict(v), h=1.5, log=lambda *_: None) for k, v in E["specimens"].items()}
    byR = {}
    for q in E["plain"]:
        byR.setdefault(q["R"], []).append(q)
    s0 = {R: math.exp(np.mean([math.log(q["sigma_A50"]) for q in v])) for R, v in byR.items()}
    m_own = {R: C.m_pooled([(q["T"], q["n"]) for q in v]) for R, v in byR.items()}
    return E, V, F, s0, m_own


def ezeh_errors(E, V, F, s0, m, l):
    """ln(predicted / measured) notched strength, every Ezeh-Susmel series."""
    return np.array([math.log(C.weibull_support(F[r["notch"]], s0[r["R"]], m, V, l) / r["sigma_nA50"])
                     for r in E["notched"]])


def fit_l(E, V, F, s0, m):
    obj = lambda ll: float(np.sqrt(np.mean(ezeh_errors(E, V, F, s0, m, math.exp(ll)) ** 2)))
    o = minimize_scalar(obj, bounds=(math.log(1e-3), math.log(20.0)), method="bounded")
    return math.exp(o.x), o.fun


def seifollahi_errors(m, l, gauge=50.0):
    """log10(predicted / measured) life of every Seifollahi notched specimen."""
    S = json.load(open(os.path.join(HERE, "data", "seifollahi_kabir.json")))
    ps = S["plain_specimen"]
    V = ps["W"] * ps["T"] * gauge
    A, B = basquin(S["raw_lives"]["smooth"])
    e = []
    for k, v in S["specimens"].items():
        w = 1.0 / C.weibull_support(field(Specimen.from_dict(v), h=1.5, log=lambda *_: None), 1.0, m, V, l)
        e += [math.log10(life(lambda N: w, s, A, B) / n) for s, *ns in S["raw_lives"][k] for n in ns]
    return np.array(e)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=float, nargs="*", default=[7.5, 12, 20, 40, 60, 80, 119])
    ap.add_argument("--gauge-ezeh", type=float, default=30.0, help="ASSUMED Ezeh-Susmel plain gauge length (mm)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    E, V, F, s0, m_own = ezeh_setup(a.gauge_ezeh)
    lines = [f"# Supported Weibull: l fitted on Ezeh-Susmel (18 series, assumed plain gauge {a.gauge_ezeh:g} mm) per fixed m,"
             " transferred to Seifollahi & Kabir (14 lives)", "",
             f"Ezeh-Susmel's own pooled m: {', '.join(f'R {R:g}: {m:.2f}' for R, m in m_own.items())}", "",
             "| m | source of m | l (mm) | Ezeh-Susmel rms ln | Seifollahi bias / rms log10 | within x2 |",
             "|---|---|---|---|---|---|"]
    for m in a.m:
        l, rms = fit_l(E, V, F, s0, m)
        e = seifollahi_errors(m, l)
        src = ("Ezeh-Susmel scatter" if abs(m - 7.5) < 0.3 else
               "Seifollahi smooth scatter (T 1.028)" if abs(m - 119) < 1 else "chosen")
        lines.append(f"| {m:g} | {src} | {l:.3f} | {rms:.3f} | {e.mean():+.2f} / {np.sqrt(np.mean(e ** 2)):.2f} | "
                     f"{int((abs(e) < math.log10(2)).sum())}/{len(e)} |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "support_fit.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
