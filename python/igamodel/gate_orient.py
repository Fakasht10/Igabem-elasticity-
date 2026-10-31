"""Orientation gate: a weibull node with an orientation card must give failure/pla_orient.py's hazard.

The PLA US quarter is built as a tree (pla_orient's IgBEM and TET10 settings) with one weibull node carrying the
card pla-as2018-orient, the flat print and the 0/90 layup about the load axis y. Its per-mode log hazards must
equal those of the cached field pla_orient solved (same mesh, same solve, same gradient).

usage (from python/; after python3 -m failure.pla_orient --fields-only):
  python3 -m igamodel.gate_orient [--work W] [--rtol 1e-9]
"""
import argparse
import math
import os

import numpy as np

from failure import aniso as A
from failure import pla_orient as PO
from failure.validate_cicero import ROOT
from failure.weibull_parity import GEOM, T, WN, case_for

from . import Model, merge_case
from .run import assess

CARD = "pla-as2018-orient"


def tree(family, work, h):
    rho = GEOM[family][0]
    c = A.load_card(CARD)
    m = Model()
    m.apply({"op": "add", "parent": "root", "kind": "cad",
             "props": {"step": os.path.join(work, f"pla_{family}_quarter.step"), "h": h, "curvature": 12}})
    merge_case(m, case_for(family))
    m.apply({"op": "add", "parent": "root", "kind": "material", "props": {"E": c["E_MPa"], "nu": c["nu"]}})
    m.apply({"op": "add", "parent": "root", "kind": "weibull",
             "props": {"label": "flat 0/90", "orientation_card": CARD, "build_dir": [0, 0, 1], "layup": "0",
                       "ref_dir": [0, 1, 0], "probability": 0.01, "mesh_h": 1.0, "mesh_curvature": 16,
                       "focus_point": [WN / 2, 0.0, 0.5 * T], "focus_radius": 2 * rho, "focus_h": rho / 5}})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="US")
    ap.add_argument("--work", default=os.path.join(ROOT, "cache", "failure"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--rtol", type=float, default=1e-9)
    a = ap.parse_args()
    row = assess(tree(a.family, a.work, a.h), ROOT, a.work)[0]
    f = PO.field(a.family, a.h, a.work)
    _, lh = PO.log_h1(f, 0.0, A.load_card(CARD)["weibull"])
    worst = max(abs(row["log_hazard"][k] - lh[k]) / abs(lh[k]) for k in lh)
    print(f"{a.family}: tree " + ", ".join(f"{k} {row['log_hazard'][k]:.10f}" for k in lh)
          + "; direct " + ", ".join(f"{k} {lh[k]:.10f}" for k in lh)
          + f"; governing {row['governing_mode']}, shares " + ", ".join(f"{k} {v:.3f}" for k, v in row["mode_shares"].items())
          + f"; factor mean {row['factor_mean']:.2f}, at 1 % {row['factor_at_probability']:.2f} "
            f"(physical scatter only {row['factor_at_probability_physical']:.2f})")
    ok = worst <= a.rtol
    print(f"gate {'passed' if ok else 'FAILED'}: worst relative difference {worst:.1e} (rtol {a.rtol:g})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
