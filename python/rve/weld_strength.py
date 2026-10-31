"""Bond strength of imperfectly bonded prints: FFM (rve.strength) against the weld stiffness kappa = K w / E.

Every weld carries the linear spring of rve.strength.weld_args, in both FFM parts: the stress part reads the weld
tractions of the intact bead cell (bounded at the bond tips for a finite spring: the notch singularity is relieved),
the energy part the compliance of the same cell with the bond narrowed by 2D. kappa -> infinity is the perfect bond.
The condensed bead operators do not depend on kappa (the weld law enters only the weld system), so after the first
kappa every cell is a cached solve.

Reference lines drawn with the curves: the net-section stress T / lig (the uniform-traction limit of a soft weld,
where the stress part no longer sees the notch) for the families that alone carry the load.

usage (from python/): python3 -m rve.weld_strength [--work /tmp/rve_wstr] [--out rve/data/results] [--only 00|090]
  runtime (4 cores, 9 kappas + perfect bond): 0/0 ~15 s; 0/90 ~6.5 min cold (25 new bead geometries to assemble, one per
  debond depth, ~15 s each; 4.4 GB of cache), ~2.7 min with the operators cached. Each cell is solved once for all
  kappas (strength --batch: one LU + a Woodbury update per kappa, ~0.85 s per extra kappa against 3.7 s alone).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve import strength  # noqa: E402
from rve.figures import INK, INK2, S1, S2, S3, plt, save  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA = (0.3, 1, 2, 3, 5, 10, 30, 100, 1000)
T, GC = 30.3, 0.05   # PLA weld strength (MPa) and toughness (N/mm) of the FFM runs in README
SHOW = {"00": (("Y (22)", "Y  (in-layer)", S1, "o"), ("Z (33)", "Z  (interlayer)", S2, "^"), ("S23", "S23", S3, "d")),
        "090": (("X (11)", "X = Y", S1, "o"), ("Z (33)", "Z  (interlayer)", S2, "^"), ("S13", "S13 = S23", S3, "d"))}


def run(kind, kappa, work):
    out = os.path.join(work, "%s_k%s.json" % (kind, "inf" if kappa is None else "%g" % kappa))
    if not os.path.exists(out):
        args = [kind, out, "--T", str(T), "--Gc", str(GC), "--grade", "2", "--stress", "weld",
                "--work", os.path.join(work, kind), "--batch"] + ["%g" % k for k in KAPPA] + ["inf"]
        if kappa is not None:
            args += ["--kappa", str(kappa)]
        strength.main(args)
    with open(out) as f:
        return json.load(f)


def table(kind, work):
    rows = [run(kind, k, work) for k in KAPPA]
    perfect = run(kind, None, work)
    key = "T=%g" % T
    get = lambda r: r["results"][key]["uniaxial"]  # noqa: E731
    return dict(kappa=list(KAPPA), perfect=get(perfect), rows=[get(r) for r in rows],
                C_perfect=perfect["C"], C=[r["C"] for r in rows])


def figure(res, out):
    fig, axs = plt.subplots(1, len(res), figsize=(3.3 * len(res), 2.9), squeeze=False)
    for ax, (kind, tb) in zip(axs[0], res.items()):
        k = np.array(tb["kappa"], float)
        for name, label, col, mk in SHOW[kind]:
            s0 = tb["perfect"][name]["strength"]
            s = np.array([r[name]["strength"] for r in tb["rows"]])
            ax.semilogx(k, s / s0, "-" + mk, color=col, ms=3.5, lw=1.4, label=label)
        if kind == "00":   # soft-weld limits: net-section stress over the perfect-bond strength
            g = GEO00
            for name, lig, col in (("Y (22)", g["h"] / g["b_lat"], S1), ("Z (33)", g["w"] / g["b_int"], S2)):
                ax.axhline(T / lig / tb["perfect"][name]["strength"], color=col, ls="--", lw=0.8)
            ax.text(0.03, 0.95, "dashed: net section T / lig (soft-weld limit)", color=INK2, fontsize=7,
                    transform=ax.transAxes, va="top")
        else:   # X leaves the scale: soft lateral welds hand sigma_11 to the A filaments (bulk failure not modelled)
            ax.set_ylim(0.8, 1.6)
            x0 = tb["perfect"]["X (11)"]["strength"]
            hi = [(kk, r["X (11)"]["strength"] / x0) for kk, r in zip(k, tb["rows"]) if r["X (11)"]["strength"] / x0 > 1.6]
            if hi:
                ax.annotate("X: %s" % ", ".join("%.1fx at %g" % (v, kk) for kk, v in hi), xy=(hi[-1][0], 1.6),
                            xytext=(0.03, 0.95), textcoords="axes fraction", va="top", fontsize=7, color=S1,
                            arrowprops=dict(arrowstyle="->", color=S1, lw=0.7))
        ax.axhline(1.0, color=INK2, lw=0.6)
        ax.set_xlabel("weld stiffness  $\\kappa = K w / E$")
        ax.set_title("%s cell, T = %g MPa, G$_c$ = %g N/mm" % ("0/0" if kind == "00" else "0/90", T, GC),
                     fontsize=8.5, color=INK)
        ax.legend(fontsize=7, loc="lower right" if kind == "00" else "center right", frameon=False)
    axs[0][0].set_ylabel("FFM strength / perfect bond")
    save(fig, out, "fig_weld_strength")


GEO00 = dict(w=0.6, h=0.3, b_int=0.36, b_lat=0.10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/rve_wstr")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "results"))
    ap.add_argument("--only", choices=["00", "090"], default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    res = {kind: table(kind, a.work) for kind in ("00", "090") if a.only in (None, kind)}
    with open(os.path.join(a.out, "weld_strength.json"), "w") as f:
        json.dump(dict(T=T, Gc=GC, law="Kn = kappa E / w, Kt = Kn (1 - 2 nu) / (2 (1 - nu))", sweeps=res), f, indent=1)
    figure(res, a.out)
    for kind, tb in res.items():
        names = [s[0] for s in SHOW[kind]]
        print(kind, "perfect", {n: round(tb["perfect"][n]["strength"], 2) for n in names})
        for kk, r in zip(tb["kappa"], tb["rows"]):
            print("  kappa %-6g" % kk, "  ".join("%s %.2f (%s, D %.3f)" % (n, r[n]["strength"], r[n]["family"], r[n]["debond"])
                                                for n in names))


if __name__ == "__main__":
    main()
