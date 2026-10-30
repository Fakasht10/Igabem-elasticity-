"""Run a model's assessments: geometry from the cad node if the case has none, one
solve for every load case, then the failure layer's Assess chain
(python/failure/assess.py) at each assessment's point for each load case.

    rows = run.assess(model, base="cases/", workdir="cache/failure")

Each row gives the load factor at failure: the multiple of the load case, as
applied, at which the point method (s1 at L/2 = sigma_0) or the energy point
method (W at R0 = W_c) is met. Linear elasticity, so one solve gives every
factor. A load case of one force_N load also gets the failure load in N.

Defaults reproduce python/failure/validate_cicero.py, the phase-1 gate
(python/igamodel/gate_cicero.py): IgBEM h 3, curvature 6, degree 2; TET10 h 2.5,
curvature 16, focus sphere 1.5 L with element size min(0.15, rho / 6).
"""
from __future__ import annotations

import os
import time

from . import backends
from .model import ModelError

MESH_DEFAULTS = {"mesh_h": 2.5, "mesh_curvature": 16}


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


def assess(model, base, workdir, backend="igabem", log=print) -> list[dict]:
    from failure.assess import assess as assess_point
    from failure.fracture import L_tcd, PLANE_STRAIN
    from failure.igabem_backend import fea_mesh, fields

    b = backends.get(backend)
    backends.check(b, model)
    items = model.of_kind("assessment")
    if not items:
        raise ModelError("the model has no assessment")
    if not model.of_kind("cad"):
        raise ModelError("an assessment meshes the part's CAD (TET10); the model has no cad")
    step = backends._abs(model.of_kind("cad")[0].props["step"], base)
    mat = backends._one(model, "material").props
    plans = []
    for a in items:                                  # every refusal before any work
        if "point" not in a.props:
            raise ModelError(f"{a.tag}: an assessment needs a point")
        K, s, src = strength(a, mat)
        plans.append((a, K, s, src))
    t0 = time.perf_counter()
    if "geometry" not in model.root.props:
        b.geometry(model, base, workdir, log=log)
    t_geo = time.perf_counter() - t0
    meshes, rows = {}, []
    with b.open(model, base, os.path.join(workdir, "run"), log=log) as run:
        for a, K, s, src in plans:
            p = {**MESH_DEFAULTS, **a.props}
            L, rho = L_tcd(K, s), p.get("rho")
            focus = (*p["point"], p.get("focus_radius", 1.5 * L), p.get("focus_h", min(0.15, rho / 6) if rho else 0.15))
            key = (p["mesh_h"], p["mesh_curvature"], focus)
            if key not in meshes:
                meshes[key] = fea_mesh(step, h=p["mesh_h"], curvature=p["mesh_curvature"], focus=[focus])
            mesh = meshes[key]
            for case in p.get("cases") or run.cases:
                view = run.view(case)
                t = time.perf_counter()
                res = fields(view, mesh, mat["E"], mat["nu"], exact_edges=False, log=log)
                r = assess_point(view, mesh, res, p["point"], mat["E"], mat["nu"], K, s, applied_N=1.0,
                                 state=p.get("state", PLANE_STRAIN), rho=rho)
                sp, ep = r["stress_pm"], r["energy_pm"]
                series = (sp["regression"] or {}).get("series") or {}
                F = _force_N(model, case)
                rows.append({
                    "assessment": a.props.get("label", a.tag), "case": case, "strength": src,
                    "K_Ic": K, "sigma_0": s, "L_mm": r["L_mm"], "R0_mm": r["R0_mm"],
                    "s1_L2_per_case": sp["probed"], "factor_pm": sp["failure_load_N"],
                    "factor_energy_pm": ep["failure_load_N"],
                    **({"failure_load_N": F * sp["failure_load_N"],
                        "failure_load_energy_pm_N": F * ep["failure_load_N"]} if F else {}),
                    "origin": r["origin"], "ray": r["ray"], "path_mode": r["path"]["mode"],
                    "root_radius_measured": r["path"].get("root_radius_mm"),
                    "cp_series_r2": series.get("r2"), "tet10_nodes": int(len(mesh.points)),
                    "t_geometry_s": round(t_geo, 1), "t_solve_s": round(run.seconds, 1),
                    "t_assess_s": round(time.perf_counter() - t, 1)})
    return rows
