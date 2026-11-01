"""Calibrate the bead cell on coupon moduli and predict a multi-angle laminate (Agotnes et al. 2026, Materials &
Design 266, 116204: Tough PLA, 100 % line infill, layer 0.1 mm, line 0.4 mm, ISO 1926 coupons).

Data (their Table 1, 2, 5; strain from the crosshead, so only ratios to E1 are used, see MACHINE below)
  fit        E2 / E1   the 90 deg coupon                                  706.44 / 988.41
             E45 / E1  the [+-45] coupon: Ex45 = 717.46 MPa, recovered exactly from their G12 = 257.97 MPa
                       through their Mallick formula with nu12 = 0.36 (so no assumption on nu12 enters here)
  predict    Ex / E1   the [0, +-30, +-60, 0, 90]_R coupon                  831.44 / 988.41
Model: the 0/0 bead cell (python/rve/beads.bead_00, w 0.4, h 0.1, notch 15 deg) with a weld spring of stiffness
kappa = K w / E (rve.strength.weld_args) on every weld, its response C(kappa) from the response library; layers are
that lamina turned by the raster angle and stacked (rve.laminate.stack_C). Unknowns: kappa (lateral welds) and the
lateral bond height b_lat at an assumed interlayer bond width b_int (0.7 w and 0.9 w: the paper gives no section).
The interlayer weld stiffness does not enter any in-plane modulus (measured here on the 2D response grid: 0.1 %
over three decades), so it is not identified by these data.

The [+-45] coupon is modelled two ways (MODES): "stack", the 0/0 laminae turned +-45 and stacked (what the paper's
CLT assumes), and "cell", the 0/90 bead cell turned 45 deg (the crossing contacts the coupon really has). The cell is
stiffer at equal porosity (-3 % at a perfect bond, -8 % at kappa 17, -37 % at kappa 1 for the stack; "crossing" in
the output): a layer is one bead high, so the crossing beads of the next layer bridge its welds, which stacked
laminae (no scale separation through the thickness) miss. The multi-angle laminate is predicted with stacked
laminae in both modes (no periodic bead cell for 30 / 60 deg crossings yet), so the "cell" fit gives a lower estimate.

MACHINE: crosshead strain adds the machine compliance in series, 1/E_meas = 1/E + c, the same c for every coupon
of one geometry, which biases every ratio toward 1. The fit is repeated with c taking a fraction f of the measured
0 deg compliance (f = 0: the paper's assumption).

usage (from python/): python3 -m rve.clt_check [--work /tmp/rve_clt] [--out rve/data/results]
  runtime (4 cores): 3 min 10 s to build the 9 geometries (0/0 and 0/90 responses) and one 2D (lateral x
  interlayer) grid; 11 s from the response library.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve import strength  # noqa: E402
from rve.figures import INK, INK2, S1, S2, S3, plt, save  # noqa: E402
from rve.homog import engineering, rotate_z  # noqa: E402
from rve.laminate import inplane, lamina_from_engineering, layup, stack_C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NU = 0.36
W, H = 0.4, 0.1
E1M, E2M, G12M, LAMM, LAM_CLT = 988.41, 706.44, 257.97, 831.44, 819.09
SD = {"0": 51.599, "45": 196.060, "lam": 107.045}   # MPa, test scatter (Table 5); 90 deg: n/a
LAYUP = "[0, +-30, +-60, 0, 90]"
BINT = (0.7, 0.9)                       # b_int / w
BLAT = (0.2, 0.35, 0.5, 0.65, 0.8)      # b_lat / h
FRAC = (0.0, 0.2, 0.4)                  # machine share of the measured 0 deg compliance
K_LO, K_HI = 0.02, 1e4


def e45_measured():
    """Ex of the +-45 coupon from the paper's G12 (their eq.: 1/Ex45 = (1/E1 + 1/E2 + 1/G12 - 2 nu12 / E1) / 4)."""
    return 4.0 / (1 / E1M + 1 / E2M + 1 / G12M - 2 * NU / E1M)


def data(f):
    """Ratios to E1 after removing a machine compliance c = f / E1M from every measured modulus."""
    c = f / E1M
    true = lambda Em: 1.0 / (1.0 / Em - c)  # noqa: E731
    e1 = true(E1M)
    return dict(E2=true(E2M) / e1, E45=true(e45_measured()) / e1, lam=true(LAMM) / e1, E1=e1)


def geom(bi, bl):
    return dict(w=W, h=H, b_int=bi * W, b_lat=bl * H, alpha=15.0, wc=math.sqrt(0.5), L=H)


def response(bi, bl, work, two=False, kind="00"):
    os.makedirs(work, exist_ok=True)
    return strength.cell_response(kind, geom(bi, bl), 1, 2, work, "clt%s_%g_%g" % (kind, bi, bl), NU, two=two)


def ratios(C0, C9=None):
    """Coupon ratios to E1 of the 0/0 lamina C0: E2 (90 deg coupon), E45 ([+-45] coupon: the 0/0 laminae stacked,
    or the 0/90 bead cell C9 turned 45 deg when given, the crossing geometry the coupon has), lam (the laminate,
    stacked laminae)."""
    E1 = inplane(C0)["Ex"]
    e45 = inplane(stack_C(C0, [45, -45]) if C9 is None else rotate_z(C9, 45))["Ex"]
    return dict(E2=inplane(C0)["Ey"] / E1, E45=e45 / E1, lam=inplane(stack_C(C0, layup(LAYUP)))["Ex"] / E1)


def model(r, k, mode):
    r0, r9 = r
    return ratios(strength.response_eval(r0, "C", k), None if mode == "stack" else strength.response_eval(r9, "C", k))


def fit(resps, target, mode):
    """kappa on every b_lat line matching E2/E1, then b_lat where E45/E1 matches (polynomial in b_lat through the
    lines, degree <= 3); kappa and the predictions at that b_lat from the same polynomials."""
    rows = []
    for bl, r in resps.items():
        f = lambda lk: model(r, math.exp(lk), mode)["E2"] - target["E2"]  # noqa: E731
        lo, hi = math.log(K_LO), math.log(K_HI)
        if f(lo) * f(hi) > 0:
            continue    # E2/E1 out of reach at this bond height (perfect bond already softer)
        k = math.exp(brentq(f, lo, hi, xtol=1e-10))
        rows.append(dict(b_lat=bl, kappa=k, logk=math.log(k), **model(r, k, mode)))
    if len(rows) < 2:
        return dict(rows=rows, ok=False)
    bl = np.array([q["b_lat"] for q in rows])
    deg = min(3, len(rows) - 1)
    poly = lambda key: np.polyfit(bl, [q[key] for q in rows], deg)  # noqa: E731
    pr = poly("E45") - np.r_[np.zeros(deg), target["E45"]]
    roots = [x.real for x in np.roots(pr) if abs(x.imag) < 1e-9 and bl.min() - 1e-9 <= x.real <= bl.max() + 1e-9]
    if not roots:
        return dict(rows=rows, ok=False)
    b = float(min(roots, key=lambda x: abs(np.polyval(pr, x))))
    at = {key: float(np.polyval(poly(key), b)) for key in ("logk", "E2", "E45", "lam")}
    return dict(rows=rows, ok=True, b_lat=b, kappa=math.exp(at.pop("logk")), **at)


MODES = ("stack", "cell")


def run(work):
    resps = {bi: {} for bi in BINT}
    for bi in BINT:
        for bl in BLAT:
            try:
                resps[bi][bl] = (response(bi, bl, work), response(bi, bl, work, kind="090"))
            except ValueError:   # no convex bead section for this pair of bond widths at the 15 deg notch
                pass
    out = dict(data={str(f): data(f) for f in FRAC}, layup=LAYUP,
               fits={m: {str(f): {str(bi): fit(resps[bi], data(f), m) for bi in BINT} for f in FRAC} for m in MODES})
    out["perfect"] = {str(bi): {str(bl): model(r, None, "cell") for bl, r in resps[bi].items()} for bi in BINT}
    # verification: the laminate layer on the paper's own CLT inputs (their Table 1)
    Cp = lamina_from_engineering(E1M, E2M, G12M, NU)
    out["clt_inputs"] = dict(ours=inplane(stack_C(Cp, layup(LAYUP))), paper=dict(Ex=819.09, Ey=777.04, Gxy=287.58,
                                                                                    nuxy=0.352))
    # identifiability: interlayer kappa at the f = 0, b_int 0.9 stack fit (nearest grid b_lat), 2D response grid
    f0 = out["fits"]["stack"]["0.0"]["0.9"]
    bl = min(resps[0.9], key=lambda x: abs(x - f0["b_lat"]))
    k = f0["kappa"]
    r2 = response(0.9, bl, work, two=True)
    rows = []
    for ki in (k / 30, k / 3, k, 3 * k, 30 * k):
        C = strength.response_eval2(r2, k, ki)
        e = engineering(C)
        rows.append(dict(kappa_int=ki, E3=e["E3"] / e["E1"], G13=e["G13"] / e["E1"], **ratios(C)))
    out["kappa_int"] = dict(b_lat=bl, kappa_lat=k, rows=rows)
    # stacking approximation against the crossing cell, same geometry, over kappa
    r0, r9 = resps[0.9][bl]
    gap = []
    for kk in (None, 1000, 100, 30, k, 5, 1):
        C0, C9 = strength.response_eval(r0, "C", kk), strength.response_eval(r9, "C", kk)
        E1 = inplane(C0)["Ex"]
        gap.append(dict(kappa=kk, pm45_stack=inplane(stack_C(C0, [45, -45]))["Ex"] / E1,
                        pm45_cell=inplane(rotate_z(C9, 45))["Ex"] / E1,
                        x090_stack=inplane(stack_C(C0, [0, 90]))["Ex"] / E1, x090_cell=inplane(C9)["Ex"] / E1))
    out["crossing"] = dict(b_lat=bl, b_int=0.9, porosity00=r0["porosity"], porosity090=r9["porosity"], rows=gap)
    return out, resps


def figure(res, resps, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.55))
    fig.subplots_adjust(wspace=0.5)
    k = np.geomspace(1, 1000, 48)
    d0 = res["data"]["0.0"]
    bls = sorted(resps[0.9])
    shades = dict(zip(bls, ["#a9cbef", "#6fa8e6", S1, "#1d559a", "#0f2f57"][:len(bls)]))
    ax = axs[0]
    for bl in bls:
        ax.semilogx(k, [model(resps[0.9][bl], kk, "cell")["E2"] for kk in k], color=shades[bl], lw=1.2,
                    label="%.2f" % bl)
    ax.axhline(d0["E2"], color=S2, lw=1.2)
    ax.text(900, d0["E2"] - 0.035, "measured", color=S2, fontsize=7, ha="right")
    ax.set_ylabel("$E_2/E_1$  (90$^\\circ$ coupon)")
    ax.legend(title="$b_{lat}/h$", fontsize=6.5, title_fontsize=7, loc="lower right")
    ax = axs[1]
    bl = res["crossing"]["b_lat"]
    for mode, ls, lab in (("cell", "-", "0/90 cell at 45$^\\circ$"), ("stack", "--", "stacked laminae")):
        ax.semilogx(k, [model(resps[0.9][bl], kk, mode)["E45"] for kk in k], ls, color=S1, lw=1.3, label=lab)
    sd = SD["45"] / E1M
    ax.axhline(d0["E45"], color=S2, lw=1.2)
    ax.axhspan(d0["E45"] - sd, d0["E45"] + sd, color=S2, alpha=0.12, lw=0)
    ax.text(1.1, d0["E45"] + sd - 0.04, "measured $\\pm$1 SD", color=S2, fontsize=7)
    ax.set_ylabel("$E_{45}/E_1$  ([$\\pm$45] coupon)")
    ax.legend(fontsize=6.5, loc="lower right", title="$b_{lat}/h$ = %g, $b_{int}/w$ = 0.9" % bl, title_fontsize=6.5,
              alignment="left")
    for a_ in axs[:2]:
        a_.set_xlabel("lateral weld stiffness  $\\kappa = K w / E$")
        a_.set_ylim(0.3, 1.0)
        a_.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax = axs[2]
    for x, f in enumerate(FRAC):
        d = res["data"][str(f)]
        ax.errorbar([x - 0.18], [d["lam"]], yerr=[SD["lam"] / LAMM * d["lam"]], fmt="s", color=S2, ms=4, capsize=2, lw=1)
        for j, (mode, mfc) in enumerate((("stack", "white"), ("cell", S1))):
            z = res["fits"][mode][str(f)]["0.9"]
            if z["ok"]:
                ax.plot([x + 0.18 * j], [z["lam"]], "o", color=S1, mfc=mfc, ms=4.5)
    ax.plot([], [], "s", color=S2, ms=4, label="measured")
    ax.plot([], [], "o", color=S1, mfc="white", ms=4.5, label="fit, stacked laminae")
    ax.plot([], [], "o", color=S1, ms=4.5, label="fit, 0/90 cell")
    ax.axhline(LAM_CLT / E1M, color=INK2, lw=0.8, ls="--")
    ax.text(0.5, LAM_CLT / E1M + 0.008, "paper CLT", color=INK2, fontsize=6.5, ha="center")
    ax.set_xticks(np.arange(len(FRAC)), ["f = %g" % f for f in FRAC])
    ax.set_xlim(-0.5, len(FRAC) - 0.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("laminate $E_x/E_1$")
    ax.set_xlabel("machine share of $1/E_1$")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(0.65, 1.1)
    for a_, t in zip(axs, "abc"):
        a_.set_title("(%s)" % t, loc="left", fontsize=8.5, color=INK)
    save(fig, out, "fig_clt_check")


def table(res):
    L = ["| [+-45] model | f | b_int/w | b_lat/h | kappa | E2/E1 | E45/E1 | laminate Ex/E1 measured | predicted | error |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in MODES:
        for f in FRAC:
            d = res["data"][str(f)]
            for bi in BINT:
                z = res["fits"][mode][str(f)][str(bi)]
                head = "| %s | %g | %g |" % ("stacked laminae" if mode == "stack" else "0/90 bead cell", f, bi)
                if not z["ok"]:
                    L.append(head + " no fit | | %.3f | %.3f | %.3f | | |" % (d["E2"], d["E45"], d["lam"]))
                else:
                    L.append(head + " %.3f | %.1f | %.3f | %.3f | %.3f | %.3f | %+.1f %% |" % (
                        z["b_lat"], z["kappa"], d["E2"], d["E45"], d["lam"], z["lam"], 100 * (z["lam"] / d["lam"] - 1)))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/rve_clt")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results"))
    a = ap.parse_args(argv)
    res, resps = run(a.work)
    with open(os.path.join(a.out, "clt_check.json"), "w") as f:
        json.dump(res, f, indent=1)
    figure(res, resps, a.out)
    print(table(res))
    c = res["clt_inputs"]
    print("laminate layer on the paper's CLT inputs: Ex %.2f Ey %.2f Gxy %.2f nuxy %.3f (paper 819.09 777.04 287.58 0.352)"
          % (c["ours"]["Ex"], c["ours"]["Ey"], c["ours"]["Gxy"], c["ours"]["nuxy"]))
    ki = res["kappa_int"]
    print("interlayer kappa at kappa_lat %.1f, b_lat/h %g:" % (ki["kappa_lat"], ki["b_lat"]))
    for r in ki["rows"]:
        print("  kappa_int %-8.3g E2/E1 %.4f E45/E1 %.4f lam %.4f | E3/E1 %.3f G13/E1 %.3f" % (
            r["kappa_int"], r["E2"], r["E45"], r["lam"], r["E3"], r["G13"]))
    c = res["crossing"]
    print("crossing vs stacked laminae, b_int/w 0.9, b_lat/h %g (porosity %.4f / %.4f):" % (c["b_lat"], c["porosity00"],
                                                                                         c["porosity090"]))
    for r in c["rows"]:
        print("  kappa %-6s [+-45] stack %.4f cell %.4f (%+.1f %%)   [0/90] stack %.4f cell %.4f (%+.1f %%)" % (
            "inf" if r["kappa"] is None else "%.3g" % r["kappa"], r["pm45_stack"], r["pm45_cell"],
            100 * (r["pm45_stack"] / r["pm45_cell"] - 1), r["x090_stack"], r["x090_cell"],
            100 * (r["x090_stack"] / r["x090_cell"] - 1)))


if __name__ == "__main__":
    main()
