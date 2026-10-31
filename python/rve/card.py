"""Material card block from the bead-cell results: homogenized stiffness and bond strengths in the ply frame.

Input: the JSON of python/rve/strength.py for a 0/0 cell (one filament direction: the ply frame e1 = filament,
e2 = in-plane transverse, e3 = build direction, exactly the frame of python/failure/aniso.py). Output: an "rve"
block (C in the ply frame, engineering constants, porosity, geometry, provenance) and the two bond modes of the
orientation card,

    T  filament bond (lateral welds):  S0_MPa = Y,  shear_ratio = Y / S12
    L  layer bond (interlayer welds):  S0_MPa = Z,  shear_ratio = Z / S13

with the 0/0 cell's Y (22), Z (33), S12, S13 (S23 reported). The F (filament) mode is bulk strength, not a bond:
it stays from the bulk card and is never guessed here.

The RVE strengths are deterministic (the weld strength T and toughness Gc in, one cell). On an orientation card the
S0_MPa are Weibull scale values; use the ratios S0_L / S0_T and the shear ratios as the transferable prediction and
keep the absolute scale from coupon tests (one Z test fixes the interlayer weld, as the scope doc says).

usage (from python/): python3 -m rve.card str00.json out.json [--T 30.3]
"""
import argparse
import json


def card(res, T=None):
    """Card block from a strength.py result dict (0/0 cell) at weld strength T (default: the first one run)."""
    if res["kind"] != "00":
        raise ValueError("the card's ply frame is the 0/0 cell (one filament direction)")
    key = "T=%g" % T if T is not None else next(iter(res["results"]))
    u = res["results"][key]["uniaxial"]
    s = {k.split()[0]: v["strength"] for k, v in u.items()}
    Y, Z, S12, S13 = s["Y"], s["Z"], s["S12"], s["S13"]
    with_c = res["C"]
    return {
        "rve": {
            "cell": "0/0 bead cell (python/rve/beads.py), periodic BC",
            "geometry_mm": res["geometry"],
            "porosity": res["porosity"],
            "E_MPa": res["E"], "nu": res["nu"],
            "C_ply_MPa": with_c,
            "weld": {"T_MPa": float(key[2:]), "Gc_N_per_mm": res["Gc"]},
            "strength_MPa": {k: v for k, v in s.items()},
            "controlling": {k.split()[0]: v["family"] for k, v in u.items()},
            "ratio_L_T": Z / Y,
            "method": "finite fracture mechanics (coupled stress + energy) at the bond tips, Williams tail",
        },
        "modes": {
            "T": {"S0_MPa": Y, "shear_ratio": Y / S12},
            "L": {"S0_MPa": Z, "shear_ratio": Z / S13},
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strength_json")
    ap.add_argument("out")
    ap.add_argument("--T", type=float, default=None)
    a = ap.parse_args(argv)
    with open(a.strength_json) as f:
        c = card(json.load(f), a.T)
    with open(a.out, "w") as f:
        json.dump(c, f, indent=1)
    m = c["modes"]
    print("T: S0 %.2f MPa, shear ratio %.3f;  L: S0 %.2f MPa, shear ratio %.3f;  S0_L / S0_T = %.3f"
          % (m["T"]["S0_MPa"], m["T"]["shear_ratio"], m["L"]["S0_MPa"], m["L"]["shear_ratio"], c["rve"]["ratio_L_T"]))


if __name__ == "__main__":
    main()
