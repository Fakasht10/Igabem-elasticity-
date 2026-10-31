"""Weibull gate: log J through the model tree must equal python/failure/weibull_parity.py's.

Each PLA quarter specimen (UB, US) is built as a tree by checked operations: the cad node with weibull_parity's
IgBEM settings, its symmetry and grip case, the card's material, and two weibull nodes, which share one solve
and one integration mesh:
  * "grid": m and S0 set, with m on the parity grid nearest the card's m. Its log J must equal the parity run's
    value at that grid point. This is the bitwise check.
  * "card": the card itself. This is what a user gets: the exact J at the card's m. weibull_parity interpolates
    log J over its m grid, and the difference between the two is reported.

usage (from python/):
  python3 -m failure.weibull_parity --work W --out D.json
  python3 -m igamodel.gate_weibull D.json --work W [--rtol 1e-12]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from failure import materials
from failure.validate_cicero import ROOT
from failure.weibull_parity import GEOM, T, WN, case_for, quarter_step

from . import Model, merge_case
from .run import assess


def tree(family, card, m_grid, work, h):
    rho = GEOM[family][0]
    step = os.path.join(work, f"pla_{family}_quarter.step")
    if not os.path.exists(step):
        quarter_step(family, step)
    c = materials.card(card)
    mesh = {"mesh_h": 1.0, "mesh_curvature": 16, "focus_point": [WN / 2, 0.0, 0.5 * T], "focus_radius": rho,
            "focus_h": rho / 5}
    m = Model()
    m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": step, "h": h, "curvature": 12}})
    merge_case(m, case_for(family))
    m.apply({"op": "add", "parent": "root", "kind": "material", "props": {"E": c["E_MPa"], "nu": c["nu"]}})
    m.apply({"op": "add", "parent": "root", "kind": "weibull",
             "props": {"label": "grid", "m": m_grid, "S0": c["weibull"]["S0_MPa"], **mesh}})
    m.apply({"op": "add", "parent": "root", "kind": "weibull",
             "props": {"label": "card", "card": card, "probability": 0.1, **mesh}})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direct", help="weibull_parity result file (same machine)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--h", type=float, default=1.5, help="as weibull_parity --h")
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--out")
    a = ap.parse_args()
    direct = json.load(open(a.direct))["rows"]
    worst, out = 0.0, []
    for d in direct:
        card, m_card = d["card"], d["m_card"]
        grid = np.asarray(d["m"])
        k = int(np.argmin(abs(grid - m_card)))
        rows = assess(tree(d["family"], card, float(grid[k]), a.work, a.h), ROOT, a.work)
        g, c = (next(r for r in rows if r["weibull"] == w) for w in ("grid", "card"))
        e = abs(g["log_J"] - d["logJ_igabem"][k]) / abs(d["logJ_igabem"][k])
        worst = max(worst, e)
        interp = float(np.interp(m_card, grid, d["logJ_igabem"]))
        print(f"{d['family']}: log J at m = {grid[k]:.4f}: tree {g['log_J']:.12f}, parity {d['logJ_igabem'][k]:.12f} "
              f"(rel diff {e:.1e}); at the card's m = {m_card:.4f}: exact {c['log_J']:.6f}, parity interpolated "
              f"{interp:.6f}, mean load x{math.exp(-(c['log_J'] - interp) / m_card):.5f}; "
              f"characteristic {c['factor_characteristic']:.2f} N, mean {c['factor_mean']:.2f} N, "
              f"10 % {c['factor_at_probability']:.2f} N ({c['unknowns']} unknowns, {c['gauss_points']} Gauss points)",
              flush=True)
        out.append({"family": d["family"], "grid": g, "card": c, "parity_logJ_at_grid": d["logJ_igabem"][k],
                    "parity_logJ_interp_at_card_m": interp, "rel_diff": e})
    if a.out:
        json.dump({"rtol": a.rtol, "worst": worst, "rows": out}, open(a.out, "w"), indent=1)
    ok = worst <= a.rtol
    print(f"gate {'passed' if ok else 'FAILED'}: worst relative difference {worst:.1e} (rtol {a.rtol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
