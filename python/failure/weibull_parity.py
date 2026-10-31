"""Weibull card parity: log J(m) of the PLA calibration specimens UB and US on IgBEM against the qualified NGSolve
reductions the frozen PLA cards were fitted on (nurb-fea-tcd calibration/data/qualified_weibull_reductions.json,
TET4 h = 0.18 mm).

Specimen convention (nurb-fea-tcd calibration/same_mesh_3d.py, from Kwon's run_test.py): the x/y quarter of the
notched plate (gross width 25, net 15, U-notch root rho as a quadratic Bezier, half-length L/2), extruded through a
unit reference thickness z in [0, 1] with both faces free, symmetry on x = 0 and y = 0, and 1 N on the quarter's
grip face (traction 1 / 12.5 MPa). J = sum_q w_q max(sigma_1, 0)^m / V0 over the quarter, V0 = 1 mm^3.

What it says about the cards: a card's S0 was fitted with the TET4 J, so on IgBEM the predicted mean load moves by
exp(-(log J_igabem - log J_tet4) / m) at the card's m; that ratio is the parity number. E, nu and m come from the card
(default pla-raster-0-weibull); with the grip loaded by traction the stress, and so J, does not depend on E.

usage: python3 -m failure.weibull_parity [--card pla-raster-0-weibull] [--families UB US] [--h 1.0] [--out results/failure/weibull_parity.json]
"""
import argparse
import json
import math
import os
import time

import numpy as np

from . import materials
from . import weibull as W
from .fields import tet10_quadrature
from .igabem_backend import ROOT, Solve, fea_mesh, igabem_geometry

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration")
GEOM = {"UB": (3.0, 96.0, 0.0), "UI": (1.0, 92.0, 0.0), "US": (0.5, 91.0, 0.0),     # rho, specimen length, notch
        "OB": (3.0, 75.0, 135.0), "OI": (1.0, 75.0, 135.0), "OS": (0.5, 75.0, 135.0)}  # opening (nominal_geometry)
WG, WN, T = 25.0, 15.0, 1.0


def quarter_step(family, path):
    """The quarter specimen as a STEP (OpenCASCADE through Gmsh), the source's a-b-q-c-d-e-f outline
    (same_mesh_3d.source_curve_controls): a quadratic Bezier root for the U notches, the interpolating spline through
    the same three points for the 135 deg V notches, as the source builds them."""
    import gmsh
    rho, L, alpha = GEOM[family]
    al = math.radians(alpha)
    xe, ye = WN / 2 + rho * (1 - math.sin(al / 2)), rho * math.cos(al / 2)
    a, b, q, c = (0.0, 0.0), (WN / 2, 0.0), (WN / 2, rho * math.tan(math.pi / 4 - al / 4)), (xe, ye)
    d, e, f = (WG / 2, ye + (WG / 2 - xe) * math.tan(al / 2)), (WG / 2, L / 2), (0.0, L / 2)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(family)
    o = gmsh.model.occ
    P = [o.addPoint(*p, 0) for p in (a, b, q, c, d, e, f)]
    curve = o.addBSpline([P[1], P[2], P[3]], degree=2) if alpha == 0 else o.addSpline([P[1], P[2], P[3]])
    lines = [o.addLine(P[0], P[1]), curve, o.addLine(P[3], P[4]), o.addLine(P[4], P[5]), o.addLine(P[5], P[6]),
             o.addLine(P[6], P[0])]
    o.extrude([(2, o.addPlaneSurface([o.addCurveLoop(lines)]))], 0, 0, T)
    o.synchronize()
    gmsh.write(path)
    gmsh.finalize()


def case_for(family):
    rho, L, _ = GEOM[family]
    plane = lambda p, n: {"plane": {"point": p, "normal": n, "tol": 1e-6}}
    return {"supports": [{"label": "sym x", "select": plane([0, 0, 0], [1, 0, 0]), "fix": ["ux"], "snap": True},
                         {"label": "sym y", "select": plane([0, 0, 0], [0, 1, 0]), "fix": ["uy"], "snap": True}],
            "cases": [{"name": "pull", "loads": [{"label": "grip", "select": plane([0, L / 2, 0], [0, 1, 0]),
                                                  "traction": [0, 1.0 / (WG / 2 * T), 0]}]}]}


def run(family, h, work, E, nu, log=print):
    rho, L, _ = GEOM[family]
    step = os.path.join(work, f"pla_{family}_quarter.step")
    os.makedirs(work, exist_ok=True)
    if not os.path.exists(step):
        quarter_step(family, step)
    t0 = time.perf_counter()
    dat, info = igabem_geometry(step, work, h=h, curvature=12, degree=2, log=log)
    root = (WN / 2, 0.0, 0.5 * T)
    mesh = fea_mesh(step, h=1.0, curvature=16, focus=[(*root, rho, rho / 5)])
    with Solve(dat, case_for(family), E, nu, workdir=os.path.join(work, f"run_{family}"), log=log) as s:
        P, w, _ = tet10_quadrature(mesh.points, mesh.cells, np.zeros((len(mesh.points), 1)))
        t1 = time.perf_counter()
        sig, wind = s.stress(P)
        t_eval = time.perf_counter() - t1
        report = s.report
    ref = json.load(open(os.path.join(DATA, "qualified_weibull_reductions.json")))[family]["reductions"]
    m = np.asarray(ref["m_grid"])
    logJ = np.array([W.log_stress_power_integral(sig, w, mm, 1.0) for mm in m])
    return {"family": family, "rho": rho, "unknowns": int(next(l for l in report if "unknowns" in l)
                                                         .split(" unknowns")[0].split()[-1]),
            "patches": info["patches"], "gauss_points": int(len(P)), "volume_mm3": float(w.sum()),
            "points_outside": int((wind < 0.5).sum()), "m": m.tolist(),
            "logJ_igabem": logJ.tolist(), "logJ_ngsolve_tet4": ref["logJ"],
            "s1_max": float(np.linalg.eigvalsh(sig)[:, -1].max()), "t_total_s": round(time.perf_counter() - t0, 1),
            "t_eval_s": round(t_eval, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default="pla-raster-0-weibull")
    ap.add_argument("--families", nargs="*", default=["UB", "US"])
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--work", default=os.path.join(ROOT, "cache", "failure"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "failure", "weibull_parity.json"))
    a = ap.parse_args()
    card = materials.card(a.card)
    m_card = card["weibull"]["m"]
    print(f"card {a.card}: E {card['E_MPa']:g}, nu {card['nu']:g}, m {m_card:.4f}, S0 {card['weibull']['S0_MPa']:.1f} MPa",
          flush=True)
    rows = []
    for fam in a.families:
        r = run(fam, a.h, a.work, card["E_MPa"], card["nu"])
        r["card"] = a.card
        lj_i = float(np.interp(m_card, r["m"], r["logJ_igabem"]))
        lj_t = float(np.interp(m_card, r["m"], r["logJ_ngsolve_tet4"]))
        r["m_card"] = m_card
        r["dlogJ_at_card_m"] = lj_i - lj_t
        r["mean_load_ratio_at_card_m"] = math.exp(-(lj_i - lj_t) / m_card)
        print(f"{fam}: log J at m={m_card:.3f}: IgBEM {lj_i:.4f}, NGSolve TET4 {lj_t:.4f} -> predicted mean load "
              f"x{r['mean_load_ratio_at_card_m']:.4f} on IgBEM ({r['unknowns']} unknowns, {r['gauss_points']} Gauss "
              f"points, {r['t_total_s']} s)", flush=True)
        rows.append(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
