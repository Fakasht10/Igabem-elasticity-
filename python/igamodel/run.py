"""Run a model's assessments: geometry from the cad node if the case has none, one
solve for every load case, then the failure layer's Assess chain
(python/failure/assess.py) at each assessment's point for each load case.

    rows = run.assess(model, base="cases/", workdir="cache/failure")

A weibull node gives the weakest-link (stress volume law) load factors of the whole
part instead: characteristic (63.2 %) and mean, optionally at a failure probability.

Each assessment row gives the load factor at failure: the multiple of the load case, as
applied, at which the point method (s1 at L/2 = sigma_0) or the energy point
method (W at R0 = W_c) is met. Linear elasticity, so one solve gives every
factor. A load case of one force_N load also gets the failure load in N.

An assessment with a submodel runs on that box of the part instead (igamodel.submodel):
cut from the STEP, meshed at its own h, its cut faces driven by the global solve. Its
probe line must stay inside the box.

Defaults reproduce python/failure/validate_cicero.py, the phase-1 gate
(python/igamodel/gate_cicero.py): IgBEM h 3, curvature 6, degree 2; TET10 h 2.5,
curvature 16, focus sphere 1.5 L with element size min(0.15, rho / 6).
"""
from __future__ import annotations

import contextlib
import os
import time

import numpy as np

from . import backends
from .casefile import resolve
from .model import ModelError

MESH_DEFAULTS = {"mesh_h": 2.5, "mesh_curvature": 16}
SUB_DEFAULTS = {"h": 0.5, "curvature": 6, "degree": 2}


def strength(a, material):
    """(K_Ic, sigma_0, source) of an assessment node. A card must agree with the material node's E and nu:
    the solve uses the material node, so a card that disagrees would mix two materials."""
    p = a.props
    if "card" in p:
        from failure import materials
        c = materials.card(p["card"])
        for k, ck in (("E", "E_MPa"), ("nu", "nu")):
            if abs(c[ck] - material[k]) > 1e-12 * abs(c[ck]):
                raise ModelError(f"{a.tag}: card {p['card']} has {k} = {c[ck]:g}, the material node {material[k]:g}")
        K, s = c["tcd"]["K_Ic_MPa_sqrt_mm"], p.get("sigma_uts", c["tcd"]["sigma_uts_MPa"])
        return K, s, f"card {p['card']}" + (", sigma_uts set" if "sigma_uts" in p else "")
    if "K_Ic" not in p or "sigma_uts" not in p:
        raise ModelError(f"{a.tag}: an assessment needs a card, or K_Ic and sigma_uts")
    return p["K_Ic"], p["sigma_uts"], "K_Ic, sigma_uts set"


def _force_N(model, case):
    lc = next((n for n in model.of_kind("loadcase") if n.props.get("name") == case), None)
    loads = lc.children if lc else []
    if len(loads) == 1 and set(loads[0].props) - {"label", "direction"} == {"force_N"}:
        return loads[0].props["force_N"]
    return None


def weibull_params(n, material):
    """(m, S0, V0, source) of a weibull node; a card must agree with the material node, as in strength()."""
    p = n.props
    if "card" in p:
        from failure import materials
        c = materials.card(p["card"])
        for k, ck in (("E", "E_MPa"), ("nu", "nu")):
            if abs(c[ck] - material[k]) > 1e-12 * abs(c[ck]):
                raise ModelError(f"{n.tag}: card {p['card']} has {k} = {c[ck]:g}, the material node {material[k]:g}")
        w = c.get("weibull") or {}
        if "m" not in w or "S0_MPa" not in w:
            raise ModelError(f"{n.tag}: card {p['card']} has no Weibull m and S0")
        return w["m"], w["S0_MPa"], p.get("V0", w.get("V0_mm3", 1.0)), f"card {p['card']}"
    if "m" not in p or "S0" not in p:
        raise ModelError(f"{n.tag}: a weibull check needs a card, or m and S0")
    return p["m"], p["S0"], p.get("V0", 1.0), "m, S0 set"


def _weibull_rows(model, n, params, run, mesh, t_geo, log):
    import math
    from failure import weibull as W
    from failure.fields import tet10_quadrature
    m, S0, V0, src = params
    P, w, _ = tet10_quadrature(mesh.points, mesh.cells, np.zeros((len(mesh.points), 1)))
    rows = []
    for case in n.props.get("cases") or run.cases:
        t = time.perf_counter()
        sig, wind = run.view(case).stress(P)
        logJ = W.log_stress_power_integral(sig, w, m, V0)
        eta = W.characteristic_failure_load_from_log_j(logJ, S0, m, 1.0)
        F, q = _force_N(model, case), n.props.get("probability")
        row = {"weibull": n.props.get("label", n.tag), "case": case, "params": src, "m": m, "S0": S0, "V0": V0,
               "log_J": logJ, "factor_characteristic": eta, "factor_mean": eta * math.gamma(1.0 + 1.0 / m),
               **({"probability": q, "factor_at_probability": eta * (-math.log1p(-q)) ** (1.0 / m)} if q else {}),
               "gauss_points": int(len(P)), "volume_mm3": float(w.sum()), "points_outside": int((wind < 0.5).sum()),
               "unknowns": _unknowns(run.report), "t_geometry_s": round(t_geo, 1), "t_solve_s": round(run.seconds, 1),
               "t_assess_s": round(time.perf_counter() - t, 1)}
        if F:
            for k in ("characteristic", "mean") + (("at_probability",) if q else ()):
                row[f"failure_load_{k}_N"] = F * row[f"factor_{k}"]
        if row["points_outside"]:
            log(f"{n.tag}: {row['points_outside']} Gauss points lie outside the spline surface")
        rows.append(row)
    return rows


def _region_faces(model):
    """CAD faces the supports and loads act on; a submodel box must not reach them."""
    faces = []
    for n in model.of_kind("support") + model.of_kind("load"):
        r = resolve(model, n.region) or {}
        if "select" in r or "patches" in r:
            raise ModelError(f"{n.tag} acts on a selection or patches; with a submodel, put supports and loads on CAD "
                             "faces so the box can be checked against them")
        faces += ([r["face_id"]] if "face_id" in r else []) + list(r.get("face_ids", []))
    return faces


def _inside(points, lo, hi, margin):
    P = np.asarray(points, float)
    return bool(np.all((P >= np.asarray(lo) + margin) & (P <= np.asarray(hi) - margin)))


def assess(model, base, workdir, backend="igabem", log=print) -> list[dict]:
    from failure.assess import assess as assess_point
    from failure.fracture import L_tcd, PLANE_STRAIN
    from failure.igabem_backend import fea_mesh, fields
    from .submodel import Submodel, check_box

    b = backends.get(backend)
    backends.check(b, model)
    items, wbs = model.of_kind("assessment"), model.of_kind("weibull")
    if not items and not wbs:
        raise ModelError("the model has no assessment or weibull check")
    if not model.of_kind("cad"):
        raise ModelError("an assessment meshes the part's CAD (TET10); the model has no cad")
    step = backends._abs(model.of_kind("cad")[0].props["step"], base)
    mat = backends._one(model, "material").props
    subs = {n.props.get("label", n.tag): n for n in model.of_kind("submodel")}
    plans = []
    for a in items:                                  # every refusal before any work
        if "point" not in a.props:
            raise ModelError(f"{a.tag}: an assessment needs a point")
        K, s, src = strength(a, mat)
        sub = a.props.get("submodel")
        if sub is not None:
            if sub not in subs:
                raise ModelError(f"{a.tag}: no submodel {sub!r}; there are {sorted(subs)}")
            n = subs[sub].props
            if "lo" not in n or "hi" not in n:
                raise ModelError(f"submodel {sub!r} needs lo and hi")
            check_box(step, n["lo"], n["hi"], a.props["point"], _region_faces(model),
                      1e-6 * float(np.linalg.norm(np.subtract(n["hi"], n["lo"]))))
        plans.append((a, K, s, src, sub))
    wplans = []
    for n in wbs:
        q = n.props.get("probability")
        if q is not None and not 0.0 < q < 1.0:
            raise ModelError(f"{n.tag}: probability is strictly between 0 and 1")
        f = [k for k in ("focus_point", "focus_radius", "focus_h") if k in n.props]
        if f and len(f) < 3:
            raise ModelError(f"{n.tag}: a refinement sphere needs focus_point, focus_radius and focus_h")
        wplans.append((n, weibull_params(n, mat)))
    t0 = time.perf_counter()
    if "geometry" not in model.root.props:
        b.geometry(model, base, workdir, log=log)
    t_geo = time.perf_counter() - t0
    meshes, rows = {}, []
    with contextlib.ExitStack() as stack:
        glob = stack.enter_context(b.open(model, base, os.path.join(workdir, "run"), log=log))
        runs = {None: (glob, step, None)}
        for a, K, s, src, sub in plans:
            if sub not in runs:
                n = {**SUB_DEFAULTS, **subs[sub].props}
                sm = Submodel.build(step, n["lo"], n["hi"], os.path.join(workdir, "sub"), h=n["h"],
                                    curvature=n["curvature"], degree=n["degree"], log=log)
                r = stack.enter_context(sm.open(glob, mat["E"], mat["nu"], os.path.join(workdir, f"sub_{sub}"),
                                                log=log))
                runs[sub] = (r, sm.step, (n["lo"], n["hi"]))
            run, part, box = runs[sub]
            p = {**MESH_DEFAULTS, **a.props}
            L, rho = L_tcd(K, s), p.get("rho")
            focus = (*p["point"], p.get("focus_radius", 1.5 * L), p.get("focus_h", min(0.15, rho / 6) if rho else 0.15))
            key = (part, p["mesh_h"], p["mesh_curvature"], focus)
            if key not in meshes:
                meshes[key] = fea_mesh(part, h=p["mesh_h"], curvature=p["mesh_curvature"], focus=[focus])
            mesh = meshes[key]
            for case in p.get("cases") or run.cases:
                view = run.view(case)
                t = time.perf_counter()
                res = fields(view, mesh, mat["E"], mat["nu"], exact_edges=False, log=log)
                r = assess_point(view, mesh, res, p["point"], mat["E"], mat["nu"], K, s, applied_N=1.0,
                                 state=p.get("state", PLANE_STRAIN), rho=rho)
                if box and not _inside(r["line"]["points"], *box, margin=0.25 * r["L_mm"]):
                    raise ModelError(f"{a.tag}: the probe line (to {r['line']['distances'].max():.3g} mm) leaves "
                                     f"submodel {sub!r}; enlarge the box along {np.round(r['ray'], 3).tolist()}")
                sp, ep = r["stress_pm"], r["energy_pm"]
                series = (sp["regression"] or {}).get("series") or {}
                F = _force_N(model, case)
                rows.append({
                    "assessment": a.props.get("label", a.tag), "case": case, "strength": src, "submodel": sub,
                    "K_Ic": K, "sigma_0": s, "L_mm": r["L_mm"], "R0_mm": r["R0_mm"],
                    "s1_L2_per_case": sp["probed"], "factor_pm": sp["failure_load_N"],
                    "factor_energy_pm": ep["failure_load_N"],
                    **({"failure_load_N": F * sp["failure_load_N"],
                        "failure_load_energy_pm_N": F * ep["failure_load_N"]} if F else {}),
                    "origin": r["origin"], "ray": r["ray"], "path_mode": r["path"]["mode"],
                    "root_radius_measured": r["path"].get("root_radius_mm"),
                    "cp_series_r2": series.get("r2"), "tet10_nodes": int(len(mesh.points)),
                    "unknowns": _unknowns(run.report),
                    "t_geometry_s": round(t_geo, 1), "t_solve_s": round(glob.seconds, 1),
                    **({"t_submodel_solve_s": round(run.seconds, 1)} if sub else {}),
                    "t_assess_s": round(time.perf_counter() - t, 1)})
        for n, params in wplans:
            p = {**MESH_DEFAULTS, **n.props}
            focus = ((*p["focus_point"], p["focus_radius"], p["focus_h"]),) if "focus_point" in p else ()
            key = (step, p["mesh_h"], p["mesh_curvature"], focus)
            if key not in meshes:
                meshes[key] = fea_mesh(step, h=p["mesh_h"], curvature=p["mesh_curvature"], focus=list(focus))
            rows += _weibull_rows(model, n, params, glob, meshes[key], t_geo, log)
    return rows


def _unknowns(report):
    line = next((ln for ln in report if " unknowns" in ln), None)
    return int(line.split(" unknowns")[0].split()[-1]) if line else None
