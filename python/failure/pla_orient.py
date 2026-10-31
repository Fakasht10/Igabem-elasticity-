"""Gate G3/G4: one orientation-aware PLA card for all raster angles (Ahmed & Susmel 2018, 54 notched tension rows).

The six tension families (UB UI US OB OI OS) are solved once each on IgBEM, as the unit-thickness quarter at 1 N
(weibull_parity's convention); the raster angle does not change the isotropic stress, so the same field serves all
three angles. Stress and its gradient at the TET10 Gauss points are cached in <work>/pla_field_<family>.npz.

Prints are flat (build direction z = specimen thickness); a raster label theta is the cross-ply layup [theta,
theta - 90] about z from the load axis y (Cicero's and, presumably, Susmel's labels: to confirm in the paper).
Mean failure load of a specimen of thickness t (nurb-fea-tcd calibration/reductions.py): with the unit-quarter
hazard H1 = sum_k J_k / S0_k^m,  mean P = 2 t Gamma(1 + 1/m) / (4 t H1)^(1/m).

Models fitted to the group means (log least squares), each scored by the campaign protocol (calibrate on UB + US,
hold out UI OB OI OS) and by leave-one-family-out:
    iso-per-angle   mode P, m and S0 per angle (today's three cards; 6 parameters)
    modes           modes F + T, one m (4 parameters: m, S0_F, S0_T, Y/S12)
    modes+support   modes F + T with gradient support at the scatter modulus m_s (4: l, S0_F, S0_T, Y/S12)

usage: python3 -m failure.pla_orient [--fields-only] [--h 1.5]      (fields ~1 min per family, once; fit ~1 min)
"""
import argparse
import json
import math
import os
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

from . import aniso as A
from . import prob
from .fields import tet10_gradient, tet10_quadrature
from .igabem_backend import ROOT, Solve, fea_mesh, igabem_geometry
from .weibull_parity import GEOM, T, WN, case_for, quarter_step

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "calibration", "pla_notched_tension.json")
FAMILY = {"B04": "UB", "I04": "UI", "S04": "US", "OR3": "OB", "OR1": "OI", "OR0": "OS"}
ANGLES = (0.0, 30.0, 45.0)
CAL = ("UB", "US")
E_PLA, NU_PLA = 3235.0, 0.35         # traction-loaded quarter: the stress does not depend on E


def rows():
    out = []
    for r in json.load(open(DATA)):
        out.append({"id": r["id"], "family": FAMILY[r["id"][:3]], "angle": float(r["angle_deg"]),
                    "t": float(r["thickness_mm"]), "P": float(r["failure_load_N"])})
    return out


# ------------------------------------------------------------------------------------------- fields (once)
def field(family, h, work, hmat=False, log=print):
    path = os.path.join(work, f"pla_field_{family}_h{h:g}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    rho = GEOM[family][0]
    step = os.path.join(work, f"pla_{family}_quarter.step")
    os.makedirs(work, exist_ok=True)
    if not os.path.exists(step):
        quarter_step(family, step)
    t0 = time.perf_counter()
    dat, _ = igabem_geometry(step, work, h=h, curvature=12, degree=2, log=log)
    root = (WN / 2, 0.0, 0.5 * T)
    mesh = fea_mesh(step, h=1.0, curvature=16, focus=[(*root, 2 * rho, rho / 5)])
    pts, cells = np.asarray(mesh.points, float), np.asarray(mesh.cells)
    on = np.zeros(len(pts), bool)
    on[np.unique(mesh.surface_tri6)] = True
    case = dict(case_for(family), **({"solver": {"backend": "hmat", "eps": 1e-5}} if hmat else {}))
    with Solve(dat, case, E_PLA, NU_PLA, workdir=os.path.join(work, f"run_{family}"), log=log) as s:
        sig_n, wind = s.stress(pts, on)
        bad = ~on & (wind < 0.5)
        if bad.any():
            sig_n[bad], _ = s.stress(pts[bad], True)
        P, w, _ = tet10_quadrature(pts, cells, np.zeros((len(pts), 1)))
        sig, _ = s.stress(P)
    grad = tet10_gradient(pts, cells, sig_n)                            # (nq, 3, 3, 3)
    out = {"P": P, "w": w, "sig": sig, "grad": grad, "seconds": time.perf_counter() - t0}
    np.savez(path, **out)
    log(f"{family}: {len(P)} Gauss points, volume {w.sum():.2f} mm^3, {out['seconds']:.0f} s")
    return out


# ------------------------------------------------------------------------------------------- the models
def print_for(angle):
    return A.Print(build_dir=(0, 0, 1), angles_deg=(angle, angle - 90.0), ref_dir=(0, 1, 0))


def log_h1(f, angle, card):
    """log of the unit-quarter hazard sum_k J_k / S0_k^m (ply components cached on the field per angle)."""
    pr = print_for(angle)
    comps = f.setdefault("comps", {})
    if angle not in comps and set(card["modes"]) - {"P"}:
        R = A.ply_frames(pr.build_dir, pr.plies()[0], pr.ref_dir)[0]
        comps[angle] = A.components(f["sig"], R, f["grad"])
    if "s_p" not in f:
        f["s_p"] = A.mode_stress("P", f["sig"])
    lh = A.mode_hazards(f["sig"], f["w"], card, pr, f["grad"], comps=comps.get(angle), s_p=f["s_p"])
    return A._logsumexp(list(lh.values())), lh


def mean_load(lh1, m, t):
    return 2 * t * math.gamma(1 + 1 / m) * math.exp(-(math.log(4 * t) + lh1) / m)


MODELS = {
    # name: (parameter names, x0, card builder(x, angle) -> card)
    "iso-per-angle": (["m0", "S0_0", "m30", "S0_30", "m45", "S0_45"], [4.1, 350, 4.1, 320, 5.2, 220],
                      lambda x, a: {"m": x[2 * ANGLES.index(a)], "modes": {"P": {"S0_MPa": x[2 * ANGLES.index(a) + 1]}}}),
    "modes": (["m", "S0_F", "S0_T", "Y_over_S12"], [4.5, 320, 300, 1.0],
              lambda x, a: {"m": x[0], "modes": {"F": {"S0_MPa": x[1]}, "T": {"S0_MPa": x[2], "shear_ratio": x[3]}}}),
}


def support_model(m_s):
    return (["l_mm", "S0_F", "S0_T", "Y_over_S12"], [0.3, 70, 60, 1.0],
            lambda x, a: {"m": m_s, "modes": {"F": {"S0_MPa": x[1], "support_mm": x[0]},
                                              "T": {"S0_MPa": x[2], "shear_ratio": x[3], "support_mm": x[0]}}})


def predict(fields, data, build, x):
    cache, out = {}, []
    for r in data:
        key = (r["family"], r["angle"])
        if key not in cache:
            card = build(x, r["angle"])
            cache[key] = (log_h1(fields[r["family"]], r["angle"], card)[0], card["m"])
        lh1, m = cache[key]
        out.append(mean_load(lh1, m, r["t"]))
    return np.array(out)


def fit(fields, data, model, train, starts=None):
    names, x0, build = model
    groups = defaultdict(list)
    for i, r in enumerate(data):
        if r["family"] in train:
            groups[(r["family"], r["angle"])].append(i)
    obs = np.array([r["P"] for r in data])
    pos = lambda x: np.all(np.asarray(x) > 0)

    def loss(z):
        x = np.exp(z)
        if not pos(x):
            return 1e9
        p = predict(fields, data, build, x)
        return float(sum((math.log(obs[ix].mean()) - math.log(p[ix].mean())) ** 2 for ix in groups.values()))

    best = None
    for start in starts if starts is not None else (np.log(x0), np.log(x0) + 0.3, np.log(x0) - 0.3):
        r = minimize(loss, start, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    x = np.exp(best.x)
    return dict(zip(names, x.tolist())), x, float(best.fun)


def score(fields, data, model, x):
    p = predict(fields, data, model[2], x)
    obs = np.array([r["P"] for r in data])
    err = 100 * (p / obs - 1)
    held = np.array([r["family"] not in CAL for r in data])
    return p, {"held_out_RMSE_percent": float(np.sqrt(np.mean(err[held] ** 2))),
               "all_RMSE_percent": float(np.sqrt(np.mean(err ** 2)))}


def bands(res):
    """G4: random-effects model error from the leave-one-family-out residuals, and the share of specimens inside the
    predicted 90 % band (5-95 %) with physical scatter only and with scatter plus model error."""
    obs = np.array([r["P"] for r in res["rows"]])
    fam = np.array([r["family"] for r in res["rows"]])
    for e in res["models"].values():
        if "lofo_predicted_N" not in e:
            continue
        p = np.array(e["lofo_predicted_N"])
        b, tau, off = prob.family_effects(np.log(obs / p), fam, res["within_log_std"])
        e["lofo_bias"], e["lofo_tau"], e["lofo_family_offsets"] = b, tau, off
        for key, sc in (("scatter_only", prob.Scatter(res["m_s"])), ("scatter_model", prob.Scatter(res["m_s"], b, tau))):
            inside = [sc.load_at(0.05, q) <= o <= sc.load_at(0.95, q) for o, q in zip(obs, p)]
            e[f"coverage90_{key}"] = float(np.mean(inside))
            e[f"p01_over_pred_{key}"] = sc.load_at(0.01, 1.0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--work", default=os.path.join(ROOT, "cache", "failure"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "failure", "pla_orient.json"))
    ap.add_argument("--fields-only", action="store_true")
    ap.add_argument("--hmat", action="store_true", help="hierarchical solver for the solves still to run (> ~15k unknowns)")
    ap.add_argument("--no-lofo", action="store_true")
    a = ap.parse_args()
    fields = {f: field(f, a.h, a.work, a.hmat) for f in GEOM}
    if a.fields_only:
        return
    data = rows()
    groups = defaultdict(list)
    for r in data:
        groups[(r["family"], r["angle"])].append(r["P"])
    m_s, s_within, dof = prob.scatter_modulus(groups.values())
    print(f"replicate scatter: log-std {s_within:.4f} ({dof} dof) -> m_s {m_s:.1f}", flush=True)
    models = dict(MODELS, **{"modes+support": support_model(m_s)})
    res = {"m_s": m_s, "within_log_std": s_within, "dof": dof, "models": {}}
    fams = sorted(GEOM)
    for name, model in models.items():
        t0 = time.perf_counter()
        params, x, loss = fit(fields, data, model, CAL)
        p, sc = score(fields, data, model, x)
        lr = np.log(np.array([r["P"] for r in data]) / p)
        held = np.array([r["family"] not in CAL for r in data])
        b, tau, offs = prob.family_effects(lr[held], np.array([r["family"] for r in data])[held], s_within)
        entry = {"params": params, "loss": loss, **sc, "bias": b, "tau": tau, "family_offsets": offs,
                 "predicted_N": p.tolist()}
        if not a.no_lofo:
            lofo = np.empty(len(data))
            for f in fams:
                _, xf, _ = fit(fields, data, model, [g for g in fams if g != f], starts=[np.log(x)])
                ix = [i for i, r in enumerate(data) if r["family"] == f]
                lofo[ix] = predict(fields, [data[i] for i in ix], model[2], xf)
            e = 100 * (lofo / np.array([r["P"] for r in data]) - 1)
            entry["lofo_RMSE_percent"] = float(np.sqrt(np.mean(e ** 2)))
            entry["lofo_predicted_N"] = lofo.tolist()
        res["models"][name] = entry
        print(f"{name:15s} held-out RMSE {sc['held_out_RMSE_percent']:5.2f} %  all {sc['all_RMSE_percent']:5.2f} %"
              + (f"  LOFO {entry['lofo_RMSE_percent']:5.2f} %" if "lofo_RMSE_percent" in entry else "")
              + f"  bias {b:+.3f} tau {tau:.3f}  {json.dumps({k: round(v, 4) for k, v in params.items()})}"
              f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    res["rows"] = data
    bands(res)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
