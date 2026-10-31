"""Stiffness of a printed cell against the weld stiffness (bead-boundary BEM, app/bead_cell).

Every weld (in-layer lateral bond, interlayer bond) carries a linear traction-separation law t = K [[u]], the
thin-interphase form: Kn = K, Kt = K (1 - 2 nu) / (2 (1 - nu)) (an isotropic interphase layer of thickness d and
the bead's moduli: Kn = (lambda + 2 mu) / d, Kt = mu / d). The dimensionless weld stiffness is kappa = K w / E
(w the filament pitch); kappa -> infinity is the perfect bond of the single-region cell.

Checks drawn with the curves: the series-spring estimate of the transverse moduli of the 0/0 cell,
  1/E2 = 1/E2(inf) + h / (b_lat K w),   1/E3 = 1/E3(inf) + w / (b_int K h)
(the weld opening of a bond of height b_lat on a pitch w, under the cell-average stress; exact as the bonds
become rigid-body links, i.e. for kappa small compared with the bead's own stiffness, and an upper bound of the
compliance increase otherwise).

usage (from python/): python3 -m rve.welds [--work /tmp/rve_welds] [--out ../docs/figures] [--only 00|090]
  runtime: 0/0 ~2 s per point, 0/90 ~60 s per point (grade 2, 12.8k unknowns, dense LU)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve import beads  # noqa: E402
from rve.figures import INK, INK2, S1, S2, S3, plt, save  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BEAD_CELL = os.environ.get("IGABEM_BEAD_CELL", os.path.join(HERE, "..", "..", "build", "bead_cell"))
GEO = dict(w=0.6, h=0.3, b_int=0.36, b_lat=0.10, alpha=15.0)
E, NU = 2200.0, 0.35
KAPPA_00 = (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 1e4)
KAPPA_090 = (0.3, 1, 3, 10, 30, 100, 1000)


def cells(work, refine=1, grade=2):
    g = GEO
    b00 = os.path.join(work, "bead00.dat")
    if not os.path.exists(b00):
        beads.write_dat(b00, beads.bead_00(g["w"], g["h"], g["b_int"], g["b_lat"], g["alpha"], refine=refine,
                                           grade=grade), "0/0 bead")
    b090 = [os.path.join(work, "b090_%s.dat" % t) for t in "AB"]
    if not all(map(os.path.exists, b090)):
        A, B = beads.bead_090(g["w"], g["h"], g["b_int"], g["b_lat"], g["alpha"], refine=refine, grade=grade)
        for f, P in zip(b090, (A, B)):
            beads.write_dat(f, P, "0/90 bead")
    return b00, b090


def run(kind, kappa, work, files):
    """One bead_cell solve, cached as <work>/<kind>_k<kappa>.json (kappa None: perfect welds)."""
    out = os.path.join(work, "%s_k%s.json" % (kind, "inf" if kappa is None else "%g" % kappa))
    if not os.path.exists(out):
        cmd = [BEAD_CELL] + files + ["--E", str(E), "--nu", str(NU), "--ng", "12", "--cache",
                                     os.path.join(work, "cache"), "--json", out]
        cmd += ["--axes", "x"] if kind == "00" else ["--copy", "1:0:90", "--axes", "x,y"]
        if kappa is not None:
            Kn = kappa * E / GEO["w"]
            Kt = Kn * (1 - 2 * NU) / (2 * (1 - NU))
            law = "%.10g:%.10g" % (Kn, Kt)
            axes = "yz" if kind == "00" else "xyz"
            for a in axes:
                cmd += ["--weld", "%s:%s" % (a, law)]
            if kind == "090":
                cmd += ["--weld-int", "z:" + law]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(out) as f:
        return json.load(f)


def sweep(kind, kappas, work, files):
    perfect = run(kind, None, work, files)
    return perfect, [run(kind, k, work, files) for k in kappas]


def figure(res, out):
    fig, axs = plt.subplots(1, len(res), figsize=(3.3 * len(res), 2.8), squeeze=False)
    for ax, (kind, (kappas, perfect, rows)) in zip(axs[0], res.items()):
        k = np.array(kappas, float)
        for name, col, mk in (("E1", INK2, "s"), ("E2", S1, "o"), ("E3", S2, "^"), ("G23", S3, "d")):
            if kind == "090" and name == "E2":
                continue  # = E1 by symmetry
            r = np.array([q["eng"][name] for q in rows]) / perfect["eng"][name]
            ax.semilogx(k, r, "-" + mk, color=col, ms=3.5, lw=1.4, label="%s / %s$^\\infty$" % (name, name))
        if kind == "00":
            g, Kw = GEO, k * E / GEO["w"]
            for name, extra, col in (("E2", g["h"] / (g["b_lat"] * g["w"]), S1), ("E3", g["w"] / (g["b_int"] * g["h"]), S2)):
                E0 = perfect["eng"][name]
                ax.semilogx(k, 1 / (1 + E0 * extra / Kw), "--", color=col, lw=0.9)
            ax.text(0.04, 0.55, "dashed: series springs", color=INK2, fontsize=7, transform=ax.transAxes)
        ax.axhline(1.0, color=INK2, lw=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("weld stiffness  $\\kappa = K w / E$")
        ax.set_title("%s cell" % ("0/0" if kind == "00" else "0/90"), fontsize=9, color=INK)
        ax.legend(fontsize=7, loc="lower right")
    axs[0][0].set_ylabel("modulus / perfect bond")
    save(fig, out, "fig_weld_stiffness")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/rve_welds")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results"))
    ap.add_argument("--only", choices=["00", "090"], default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    b00, b090 = cells(a.work)
    res = {}
    if a.only in (None, "00"):
        perfect, rows = sweep("00", KAPPA_00, a.work, [b00])
        res["00"] = (KAPPA_00, perfect, rows)
    if a.only in (None, "090"):
        perfect, rows = sweep("090", KAPPA_090, a.work, b090)
        res["090"] = (KAPPA_090, perfect, rows)
    table = {kind: dict(kappa=list(k), perfect=p["eng"], rows=[dict(q["eng"], hill_mandel=q["hill_mandel"]) for q in r])
             for kind, (k, p, r) in res.items()}
    with open(os.path.join(a.out, "weld_stiffness.json"), "w") as f:
        json.dump(dict(geometry=GEO, E=E, nu=NU, law="Kn = kappa E / w, Kt = Kn (1 - 2 nu) / (2 (1 - nu))",
                       sweeps=table), f, indent=1)
    figure(res, a.out)
    for kind, (k, p, r) in res.items():
        print(kind, "perfect", {n: round(v, 1) for n, v in p["eng"].items() if n[0] in "EG"})
        for kk, q in zip(k, r):
            print("  kappa %-7g" % kk, " ".join("%s %.3f" % (n, q["eng"][n] / p["eng"][n]) for n in ("E1", "E2", "E3", "G23")),
                  " HM %.0e" % q["hill_mandel"])


if __name__ == "__main__":
    main()
