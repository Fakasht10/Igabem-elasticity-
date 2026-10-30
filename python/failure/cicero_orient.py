"""Gate G6: one per-mode TCD card for the three Cicero ABS layups (0/90, 30/-60, 45/-45; SENB, four root radii).

The SENB is force-loaded and simply supported, so its stress does not depend on E (all layups have nu 0.35): one
IgBEM solve per radius serves every layup. The full stress tensor along each root bisector (the path of
validate_cicero, results/failure/cicero.json) is cached in <work>/cicero_line_<rho>.npz.

Card: modes F (filament) and T (filament bond, shear ratio Y/S12) with inherent strengths X, Y and one critical
distance L for both (K_k = sigma_0k sqrt(pi L)); prints are flat, plies [theta, theta - 90] about z from the beam
axis x. Fitted to the 12 mean critical loads (log least squares); scored in-sample and by leave-one-layup-out
against the per-layup isotropic TCD (paper K_mat and L; and sigma_0 recalibrated per layup on rho = 1).

usage: python3 -m failure.cicero_orient      (lines ~1 min per radius, once; fit seconds)
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from . import aniso as A
from .igabem_backend import ROOT, Solve, igabem_geometry
from .validate_cicero import MEASURED_P_CRIT, ORIENTATIONS, STEP, case_for

LAYUPS = {"0/90": (0.0, -90.0), "30/-60": (30.0, -60.0), "45/-45": (45.0, -45.0)}
RADII = (0.25, 0.5, 1.0, 2.0)


def line(rho, work, h=3.0, r_far=5.0, log=print):
    path = os.path.join(work, f"cicero_line_{rho:g}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    row = next(r for r in json.load(open(os.path.join(ROOT, "results", "failure", "cicero.json")))["rows"]
               if r["rho"] == rho)
    origin, ray = np.asarray(row["origin"]), np.asarray(row["ray"]) / np.linalg.norm(row["ray"])
    step = os.path.join(ROOT, "geo", "step", STEP[rho])
    dat, _ = igabem_geometry(step, work, h=h, curvature=6, degree=2, log=log)
    hh = 0.1 * rho                                                       # assess.exact_line's sampling
    d = np.concatenate([[0.0], np.geomspace(min(3e-3, 0.03 * hh), hh, 24), np.geomspace(hh, r_far, 136)[1:]])
    with Solve(dat, case_for(step), 2241.0, 0.35, workdir=os.path.join(work, f"run_cic_{rho:g}"), log=log) as run:
        sig, _ = run.stress(origin + d[:, None] * ray, d <= 1e-12)
    np.savez(path, d=d, sig=sig)
    return {"d": d, "sig": sig}


def card_from(x):
    X, Y, r, L = x
    k = math.sqrt(math.pi * L)
    return {"modes": {"F": {"K": X * k, "sigma_0": X}, "T": {"K": Y * k, "sigma_0": Y, "shear_ratio": r}}}


def predict(lines, x, layups):
    card, out = card_from(x), {}
    for name in layups:
        pr = A.Print(build_dir=(0, 0, 1), angles_deg=LAYUPS[name], ref_dir=(1, 0, 0))
        for rho in RADII:
            res, gov = A.tcd_point(lines[rho]["sig"], lines[rho]["d"], card, pr)
            out[(name, rho)] = (1.0 / res[gov]["ratio"], gov)
    return out


def fit(lines, layups):
    keys = [(n, r) for n in layups for r in RADII]
    obs = np.log([MEASURED_P_CRIT[n][r] for n, r in keys])

    def loss(z):
        p = predict(lines, np.exp(z), layups)
        return float(np.sum((obs - np.log([p[k][0] for k in keys])) ** 2))

    best = None
    for x0 in ([60.0, 50.0, 1.4, 0.4], [80.0, 45.0, 1.0, 0.3], [55.0, 55.0, 2.0, 0.5]):
        r = minimize(loss, np.log(x0), method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 6000})
        best = r if best is None or r.fun < best.fun else best
    return np.exp(best.x)


def iso_reference(lines):
    """Per-layup isotropic stress point method: the paper's K_mat and L, and sigma_0 recalibrated on rho = 1."""
    out = {}
    for name, p in ORIENTATIONS.items():
        L, K = p["L_pm"], p["K_mat"] * math.sqrt(1000.0)
        s0 = K / math.sqrt(math.pi * L)
        raw = {rho: A.tcd_point(lines[rho]["sig"], lines[rho]["d"], {"modes": {"P": {"K": K, "sigma_0": s0}}})[0]["P"]
               for rho in RADII}
        P = {rho: 1.0 / raw[rho]["ratio"] for rho in RADII}
        scale = MEASURED_P_CRIT[name][1.0] / P[1.0]
        out[name] = {rho: {"paper_N": P[rho], "cal_rho1_N": P[rho] * scale} for rho in RADII}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.path.join(ROOT, "cache", "failure"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "failure", "cicero_orient.json"))
    a = ap.parse_args()
    lines = {rho: line(rho, a.work) for rho in RADII}
    names = list(LAYUPS)
    x = fit(lines, names)
    insample = predict(lines, x, names)
    loo = {}
    for n in names:
        loo.update({k: v for k, v in predict(lines, fit(lines, [m for m in names if m != n]), [n]).items()})
    iso = iso_reference(lines)
    rows = []
    for n in names:
        for rho in RADII:
            meas = MEASURED_P_CRIT[n][rho]
            rows.append({"layup": n, "rho": rho, "measured_N": meas, "modes_N": insample[(n, rho)][0],
                         "governing": insample[(n, rho)][1], "modes_loo_N": loo[(n, rho)][0],
                         "iso_paper_N": iso[n][rho]["paper_N"], "iso_cal_rho1_N": iso[n][rho]["cal_rho1_N"]})
    err = lambda key: [100 * (r[key] / r["measured_N"] - 1) for r in rows]
    summary = {key: {"min_pct": min(err(key)), "max_pct": max(err(key)),
                     "rms_pct": float(np.sqrt(np.mean(np.square(err(key)))))}
               for key in ("modes_N", "modes_loo_N", "iso_paper_N", "iso_cal_rho1_N")}
    params = dict(zip(["X_MPa", "Y_MPa", "Y_over_S12", "L_mm"], x.tolist()))
    print("card:", json.dumps({k: round(v, 4) for k, v in params.items()}))
    for r in rows:
        print(f"{r['layup']:7s} rho {r['rho']:<4} measured {r['measured_N']:6.1f}  modes {r['modes_N']:6.1f} "
              f"({r['governing']})  LOO {r['modes_loo_N']:6.1f}  iso paper {r['iso_paper_N']:6.1f}  "
              f"iso cal {r['iso_cal_rho1_N']:6.1f}")
    for k, v in summary.items():
        print(f"{k:15s} error {v['min_pct']:+6.1f} .. {v['max_pct']:+6.1f} %   rms {v['rms_pct']:.1f} %")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"params": params, "rows": rows, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
