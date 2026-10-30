"""Point-method failure assessment of one feature on an IgBEM solution: nurb-fea-tcd's Assess chain for its BEM solver
(viewer.py Assess with CACHED_SOLVER == "bem", at commit 553dd2a), on the vendored fracture.py, without the viewer.

  1. origin: the picked point snaps to the nearest surface node, then to the s1 peak within L/2 (fracture.snap_to_peak);
  2. path: MATS (fracture.mats_path, rays averaged over L); in mode I (the normal kept) the origin moves to the field
     peak round the root ARC and the ray to the arc's normal there (fracture.notch_apex); a sharp re-entrant edge
     within L/2 takes the edge and the bisector of its faces (fracture.reentrant_edge_at);
  3. exact line: stress from the solve itself along the ray (wall stress at r = 0), densely near the root (the
     viewer's _bem_line); stress PM reads s1 at L/2, Energy PM reads W at R0_PM;
  4. regression: fracture.path_regression on the exact line (Creager-Paris series with rho held at the measured root
     radius; Williams' exponent on a sharp edge);
  5. failure loads: linear elasticity, P = applied / ratio for both methods.
"""
import numpy as np

from . import fracture as F
from .igabem_backend import sed


def exact_line(run, origin, ray, r_far, rho, E, nu):
    """s1 and SED along the ray from the solve (viewer._bem_line's sampling: 24 points to rho/10, 136 beyond)."""
    v = np.asarray(ray, float) / np.linalg.norm(ray)
    h = 0.1 * float(rho)
    d = np.concatenate([[0.0], np.geomspace(min(3e-3, 0.03 * h), h, 24),
                        np.geomspace(h, max(float(r_far), 2.0 * h), 136)[1:]])
    P = np.asarray(origin, float) + d[:, None] * v
    sig, wind = run.stress(P, d <= 1e-12)
    return {"distances": d, "points": P, "s1": np.linalg.eigvalsh(sig)[:, -1], "sed": sed(sig, E, nu), "winding": wind}


def path_at(mesh, res, point, K_Ic, sigma_uts):
    """(origin, ray, info): where the extraction path starts and where it goes (steps 1-2 above)."""
    L = F.L_tcd(K_Ic, sigma_uts)
    edge = F.reentrant_edge_at(mesh, np.asarray(point, float), 0.5 * L)
    if edge is not None:
        return np.asarray(edge["origin"]), np.asarray(edge["bisector"]), {"mode": "sharp_edge", "edge": edge,
                                                                          "williams": F.williams_exponent(edge["opening_deg"])}
    s1 = res["s1_nodal"]
    _n0, c0, _m = F.snap_to_node(mesh, point)
    nid, c, _m, _ = F.snap_to_peak(mesh, s1, c0, radius=0.5 * L)
    mats = F.mats_path(mesh, res["sig6_nodal"], int(nid), L)
    info = {"mode": "mats", "node": int(nid), "mats": {k: mats[k] for k in ("angle_from_normal_deg", "normal_preferred",
                                                                              "front_anisotropy") if k in mats}}
    origin, ray = np.asarray(mats["origin"], float), np.asarray(mats["ray"], float)
    rr = F.root_radius_at_node(mesh, int(nid), mats["front_tangent"], mats["d_mm"])
    if rr is not None:
        info["root_radius_mm"] = float(rr[0])
        if mats.get("normal_preferred"):
            apex = F.notch_apex(mesh, s1, int(nid), mats["front_tangent"], rr[0], mats["surface_normal"])
            if apex is not None:
                origin, ray = np.asarray(apex["origin"], float), np.asarray(apex["ray"], float)
                info["mode"] = "apex"
    return origin, ray / np.linalg.norm(ray), info


def assess(run, mesh, res, point, E, nu, K_Ic, sigma_uts, applied_N=1.0, state=F.PLANE_STRAIN, rho=None):
    """Stress PM and Energy PM at the feature near `point`: ratios, failure loads, the exact line and its fits."""
    L = F.L_tcd(K_Ic, sigma_uts)
    R0 = F.R0_energy_pm(K_Ic, sigma_uts, nu, state)
    W_c = F.critical_sed(sigma_uts, E)
    origin, ray, info = path_at(mesh, res, point, K_Ic, sigma_uts)
    rho = rho or info.get("root_radius_mm") or L
    r_far = 2.5 * max(L, R0)
    line = exact_line(run, origin, ray, r_far, rho, E, nu)
    d = line["distances"]
    s1_L2 = float(np.interp(0.5 * L, d, line["s1"]))
    w_R0 = float(np.interp(R0, d, line["sed"]))
    out = {"origin": origin.tolist(), "ray": ray.tolist(), "path": info, "L_mm": L, "R0_mm": R0, "W_c": W_c,
           "line": line}
    for name, energy, probed, dist in (("stress_pm", False, s1_L2, 0.5 * L), ("energy_pm", True, w_R0, R0)):
        ratio = (np.sqrt(probed / W_c) if energy else probed / sigma_uts)
        P, _ = F.failure_load_from_probe(probed, applied_N, sigma_uts=sigma_uts, W_c=W_c, energy=energy)
        reg = F.path_regression(line["points"], line["sed"] if energy else line["s1"], origin, ray, dist,
                                r_max=r_far, window=(0.0, r_far / dist), near_max=2.0 * L,
                                tube=max(0.5 * dist, 0.15 * r_far / 2.5),
                                series_rho=None if info["mode"] == "sharp_edge" else rho, squared=energy,
                                williams=info.get("williams"))
        out[name] = {"distance_mm": dist, "probed": probed, "ratio": float(ratio), "failure_load_N": P,
                     "regression": reg}
    return out
