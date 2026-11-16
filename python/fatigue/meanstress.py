"""Mean stress at a notch root, and the stress-gradient factor Y of Shi, Wei & Mao (Chin. J. Aeronaut. 39(7) 104033,
2026), on the cached fields of a dataset.

    python3 -m fatigue.meanstress [--data ...] [--gauge 30] [--out results/fatigue]

1. Mean-stress transfer of the R = 0 notched series from R = -1 data only (plain R = -1 strength s_r1 and the plain
   Walker exponent gamma per print angle):
     matched   plain R = 0 strength directly (no mean-stress model; the reference of fatigue.transfer)
     walker    Walker on the effective stress of the criterion, elastic: s_max^(1 - gamma) s_a^gamma = s_r1
     relaxed   the same, with the root's elastic maximum capped at sigma_cap (default the plain UTS of that print
               angle; no fit): after the first cycles the root mean stress drops to sigma_cap - a_root (not below
               0), and the effective stress's mean is scaled by the same fraction (it sits in the same root zone)
   for the Weibull (gauge L_g, m from the R = -1 plain scatter) and TCD point criteria. Both are linear in the load,
   so a criterion is an effective concentration F = s0 / criterion(f, s0).
2. Y = integral over [0, 1] of sigma / sigma_max along the bisector, normalised by the curve length L. L ends at the
   symmetry plane (rule 1: then Y ~ 1/Kt by force balance on the net section), where d sigma / dx = 0 (rule 2), or
   where sigma drops to the plain fatigue strength s0 (rule 3, load dependent). Leave-one-notch-out fit of
   sigma_nA / s0 = Y^a shows whether Y orders the notches better than Kt.
"""
import argparse
import json
import math
import os

import numpy as np

from . import criteria as C
from .fields import field
from .specimens import Specimen
from .transfer import HERE, OUT, plain_lookup


def relaxed_strength(F, Kt, R, s_r1, gamma, cap=None):
    """Notched nominal amplitude at which the Walker stress of the effective amplitude F s_n equals s_r1. cap: root
    maximum stress limit (None: elastic)."""
    def walker(sn):
        a_eff, a_root = F * sn, Kt * sn
        k = (1 + R) / (1 - R)                                  # mean / amplitude, elastic
        frac = 1.0
        if cap is not None and k > 0:
            m_root = max(0.0, min(k * a_root, cap - a_root))
            frac = m_root / (k * a_root)
        return (frac * k * a_eff + a_eff) ** (1 - gamma) * a_eff ** gamma
    lo, hi = 1e-6, s_r1 / F * 4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if walker(mid) < s_r1 else (lo, mid)
    return 0.5 * (lo + hi)


def gradient_factor(f, s_nom=None, s0=None):
    """Y (Shi et al.) on the thickness-mean bisector profile: rule 1/2 ends (symmetry plane or zero slope), or with
    s_nom and s0 the rule 3 end where s_nom sigma(x) = s0. Returns (Y, L)."""
    d, s = C.line_profile(f)
    g = np.diff(s)
    j = int(np.argmax(g >= 0)) if (g >= 0).any() else len(s) - 1
    if s_nom is not None and s_nom * s[j] < s0 <= s_nom * s[0]:
        j = int(np.argmax(s_nom * s < s0))
        x = np.interp(s0 / s_nom, s[j::-1], d[j::-1])           # crossing, s decreasing
        m = d < x
        dd, ss = np.r_[d[m], x], np.r_[s[m], s0 / s_nom]
    else:
        dd, ss = d[: j + 1], s[: j + 1]
    L = dd[-1]
    return float(np.trapezoid(ss / ss[0], dd) / L), float(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "ezeh_susmel.json"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--gauge", type=float, default=30.0)
    ap.add_argument("--cap", default="UTS", help="root maximum stress limit: UTS (per print angle) or a value in MPa")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    specs = {k: Specimen.from_dict(v) for k, v in data["specimens"].items()}
    fields = {k: field(s, h=a.h, log=lambda *_: None) for k, s in specs.items()}
    plain = plain_lookup(data)
    ps = data["plain_specimen"]
    tcd = data.get("tcd_reference")
    Lm = tcd["A_mm"] * data["N_ref"] ** tcd["b"] if tcd else None
    gam = {p: C.walker_fit([(r["R"], r["sigma_A50"]) for r in data["plain"] if r["p"] == p])[0]
           for p in {r["p"] for r in data["plain"]}}
    rows = []
    for r in sorted(data["notched"], key=lambda r: (r["notch"], r["p"], r["R"])):
        f = fields[r["notch"]]
        Kt = float(np.asarray(f["s1_line"])[:, 0].mean())
        p1, p0 = plain[(r["p"], -1)], plain.get((r["p"], r["R"]))
        s_r1, m1 = p1["sigma_A50"], C.m_from_T(p1["T"])
        cap = p1.get("UTS") if a.cap == "UTS" else float(a.cap)
        row = dict(notch=r["notch"], p=r["p"], R=r["R"], measured=r["sigma_nA50"], Kt=Kt, gamma=gam[r["p"]], cap=cap,
                   root_max_el=Kt * r["sigma_nA50"] * 2 / (1 - r["R"]))
        crit = {"weibull": lambda s0: C.weibull(f, s0, m1, ps["W"] * ps["T"] * a.gauge)}
        if Lm:
            crit["tcd_point"] = lambda s0: C.tcd_point(f, s0, Lm)
        for name, fn in crit.items():
            F = 1.0 / fn(1.0)
            if p0 is not None and name == "weibull":
                row[f"{name}_matched"] = C.weibull(f, p0["sigma_A50"], C.m_from_T(p0["T"]), ps["W"] * ps["T"] * a.gauge)
            elif p0 is not None:
                row[f"{name}_matched"] = fn(p0["sigma_A50"])
            row[f"{name}_walker"] = relaxed_strength(F, Kt, r["R"], s_r1, gam[r["p"]])
            row[f"{name}_relaxed"] = relaxed_strength(F, Kt, r["R"], s_r1, gam[r["p"]], cap)
        s0 = p0["sigma_A50"] if p0 else s_r1
        row["Y12"], row["L12"] = gradient_factor(f)
        row["Y3"], row["L3"] = gradient_factor(f, r["sigma_nA50"], s0)
        row["s0"] = s0
        rows.append(row)
    # leave-one-notch-out: sigma_nA / s0 = Y^a (rule 1/2 Y ~ 1/Kt; rule 3 Y at the measured load)
    notches = sorted({r["notch"] for r in rows})
    for yk in ("Y12", "Y3"):
        for n in notches:
            tr = [r for r in rows if r["notch"] != n]
            x, y = np.log([r[yk] for r in tr]), np.log([r["measured"] / r["s0"] for r in tr])
            A = float(x @ y / (x @ x))
            for r in rows:
                if r["notch"] == n:
                    r[f"loo_{yk}"] = r["s0"] * r[yk] ** A          # rule 3: Y taken at the measured load (optimistic)
                    r[f"a_{yk}"] = A
    keys = [k for k in ("weibull_matched", "weibull_walker", "weibull_relaxed", "tcd_point_matched",
                        "tcd_point_walker", "tcd_point_relaxed", "loo_Y12", "loo_Y3") if k in rows[0]]
    lines = ["| notch | p | R | measured | Kt | root max (elastic) / cap | Y (rule 1) · Kt | Y (rule 3) | "
             + " | ".join(keys) + " |", "| " + " | ".join(["---"] * (8 + len(keys))) + " |"]
    for r in rows:
        lines.append(f"| {r['notch']} | {r['p']} | {r['R']:g} | {r['measured']:.1f} | {r['Kt']:.2f} | "
                     f"{r['root_max_el'] / r['cap']:.2f} | {r['Y12'] * r['Kt']:.3f} | {r['Y3']:.3f} | "
                     + " | ".join(f"{r[k]:.1f} ({100 * (r[k] / r['measured'] - 1):+.0f}%)" for k in keys) + " |")
    lines.append("")
    for subset, sel in (("R = 0", lambda r: r["R"] == 0), ("all", lambda r: True)):
        for k in keys:
            e = np.array([r[k] / r["measured"] - 1 for r in rows if sel(r)])
            lines.append(f"- {subset:5s} {k:20s} mean {100 * e.mean():+6.1f}%  rms log "
                         f"{np.sqrt(np.mean(np.log1p(e) ** 2)):.3f}  within 20%: {(abs(e) < 0.2).sum()}/{len(e)}")
    os.makedirs(a.out, exist_ok=True)
    tag = f"meanstress_{data['id']}_h{a.h:g}"
    open(os.path.join(a.out, tag + ".md"), "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print("\n".join(lines))
    return rows


if __name__ == "__main__":
    main()
