"""TCD point method on IgBEM against Cicero et al. (2020): 3D-printed ABS SENB, U-notch radii 0.25-2 mm.

The same test as nurb-fea-tcd's bench/validate_cicero_2020.py (TET10 + JAX-FEM), run on IgBEM through the vendored
failure layer: SENB 44 x 10 x 4 mm, span 40, a0 = 5 mm, pads 1 mm high and 2 mm wide (parts/senb_rho*.py, STEP in
geo/step/), left pad pinned, right pad held in y and z, 1 N down on the top pad. L from the paper's calibration
(Table 3), sigma_0 = K_mat / sqrt(pi L). With sigma_0 from the paper's K_mat the TET10 run over-predicts by a uniform
~52 % (K_mat carries plastic work); so sigma_0 is also calibrated on one radius and the other three are predictions.

Source: Cicero, Martinez-Mata, Alonso-Estebanez, Castanon-Jano, Arroyo, Materials 13(21) 4919, 2020 (open access).

usage: python3 -m failure.validate_cicero [--orientation 0/90] [--radii 0.25 0.5 1 2] [--calibrate-on 1]
                                          [--h 3] [--out results/failure/cicero.json]
"""
import argparse
import json
import math
import os
import time

import numpy as np

from .assess import assess
from .igabem_backend import ROOT, Solve, fea_mesh, feakit_face_map, fields, igabem_geometry

ORIENTATIONS = {   # Table 1 (E), Table 3 (calibrated L), Table 2 cracked reference (K_mat)
    "0/90": {"E": 2241.0, "nu": 0.35, "sigma_u": 51.7, "K_mat": 3.54, "L_pm": 0.44},
    "30/-60": {"E": 2329.0, "nu": 0.35, "sigma_u": 59.3, "K_mat": 3.98, "L_pm": 0.28},
    "45/-45": {"E": 2388.0, "nu": 0.35, "sigma_u": 60.8, "K_mat": 4.12, "L_pm": 0.32},
}
MEASURED_P_CRIT = {   # Table 2, mean critical load (N)
    "0/90": {0.25: 89.70, 0.5: 95.91, 1.0: 88.23, 2.0: 102.15},
    "30/-60": {0.25: 100.61, 0.5: 100.79, 1.0: 108.93, 2.0: 111.76},
    "45/-45": {0.25: 104.85, 0.5: 105.74, 1.0: 108.16, 2.0: 113.83},
}
STEP = {0.25: "senb_rho025.step", 0.5: "senb_rho05.step", 1.0: "senb_rho1.step", 2.0: "senb_rho2.step"}
ROOT_PT = (22.0, 5.0, 0.0)          # mid-span, notch root, mid-thickness


def case_for(step):
    """Supports and the load by CAD face, located by face centre (pads: left x 1-3, right x 41-43, top x 21-23)."""
    m = feakit_face_map(step, [(2.0, -1.0, 0.0), (42.0, -1.0, 0.0), (22.0, 11.0, 0.0)])
    return {"supports": [{"label": "left pad", "face_id": m[0], "fix": ["ux", "uy", "uz"]},
                         {"label": "right pad", "face_id": m[1], "fix": ["uy", "uz"]}],
            "cases": [{"name": "bend", "loads": [{"label": "top pad", "face_id": m[2], "force_N": 1.0,
                                                  "direction": [0, -1, 0]}]}]}


def run_one(rho, name, h, workdir, log):
    props = ORIENTATIONS[name]
    step = os.path.join(ROOT, "geo", "step", STEP[rho])
    L = props["L_pm"]
    K = props["K_mat"] * math.sqrt(1000.0)
    sigma_0 = K / math.sqrt(math.pi * L)
    t0 = time.perf_counter()
    dat, ginfo = igabem_geometry(step, workdir, h=h, curvature=6, degree=2, log=log)
    t_geo = time.perf_counter() - t0
    mesh = fea_mesh(step, h=2.5, curvature=16, focus=[(*ROOT_PT, 1.5 * L, min(0.15, rho / 6))])
    with Solve(dat, case_for(step), props["E"], props["nu"], workdir=os.path.join(workdir, f"run_{rho}"), log=log) as run:
        t1 = time.perf_counter()
        res = fields(run, mesh, props["E"], props["nu"], exact_edges=False, log=log)
        t_fields = time.perf_counter() - t1
        a = assess(run, mesh, res, ROOT_PT, props["E"], props["nu"], K, sigma_0, applied_N=1.0, rho=rho)
        t_solve = run.seconds
    sp = a["stress_pm"]
    reg = sp["regression"] or {}
    series = reg.get("series") or {}
    return {"rho": rho, "L": L, "sigma_0": sigma_0, "unknowns": int(next(l for l in run.report if "unknowns" in l)
                                                                     .split(" unknowns")[0].split()[-1]),
            "patches": ginfo["patches"], "tet10_nodes": int(len(mesh.points)),
            "origin": a["origin"], "ray": a["ray"], "path_mode": a["path"]["mode"],
            "root_radius_measured": a["path"].get("root_radius_mm"),
            "s1_L2_per_N": sp["probed"], "predicted_N": sp["failure_load_N"],
            "energy_pm_predicted_N": a["energy_pm"]["failure_load_N"],
            "cp_series_r2": series.get("r2"), "cp_series_terms": series.get("n_terms"),
            "measured_N": MEASURED_P_CRIT[name][rho],
            "t_geometry_s": round(t_geo, 1), "t_solve_s": round(t_solve, 1), "t_fields_s": round(t_fields, 1),
            "line_r": a["line"]["distances"].tolist(), "line_s1": a["line"]["s1"].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orientation", default="0/90", choices=list(ORIENTATIONS))
    ap.add_argument("--radii", type=float, nargs="*", default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--calibrate-on", type=float, default=1.0)
    ap.add_argument("--h", type=float, default=3.0, help="IgBEM knot span (mm); notch arcs follow the curvature rule")
    ap.add_argument("--work", default=os.path.join(ROOT, "cache", "failure"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "failure", "cicero.json"))
    a = ap.parse_args()
    props = ORIENTATIONS[a.orientation]
    rows = []
    for rho in a.radii:
        r = run_one(rho, a.orientation, a.h, a.work, log=print)
        r["error_pct"] = 100 * (r["predicted_N"] / r["measured_N"] - 1)
        print(f"rho={rho:<5} s1(L/2)={r['s1_L2_per_N']:.5f} MPa/N  predicted={r['predicted_N']:7.1f} N  "
              f"measured={r['measured_N']:6.2f} N  err={r['error_pct']:+6.1f}%  ({r['unknowns']} unknowns, "
              f"{r['t_geometry_s']}+{r['t_solve_s']}+{r['t_fields_s']} s)", flush=True)
        rows.append(r)
    ref = next((r for r in rows if r["rho"] == a.calibrate_on), None)
    if ref is not None:
        scale = ref["measured_N"] / ref["predicted_N"]
        for r in rows:
            r["predicted_cal_N"] = r["predicted_N"] * scale
            r["error_cal_pct"] = 100 * (r["predicted_cal_N"] / r["measured_N"] - 1)
        print(f"sigma_0 calibrated on rho={a.calibrate_on}: x{scale:.3f} -> "
              + ", ".join(f"rho {r['rho']}: {r['error_cal_pct']:+.1f}%" for r in rows))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"orientation": a.orientation, "props": props, "calibrate_on": a.calibrate_on, "rows": rows}, f,
                  indent=1)


if __name__ == "__main__":
    main()
