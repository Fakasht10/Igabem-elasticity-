# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Deterministic mesh diagnosis (Sutura 0.9.8 ``defects.detect``,
``topology.mesh_topology``/``mesh_geometric_measures`` and the exact SI
classifier, merged into one report)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as G
from . import intersect as X
from . import topology as T
from .mesh import Mesh


def duplicate_groups(F):
    """Group faces by vertex set.  Returns (same_dup_extra, reversed_pairs,
    group label per face, orientation parity per face)."""
    m = len(F)
    if m == 0:
        return 0, 0, np.zeros(0, np.int64), np.zeros(0, bool)
    r = np.argmin(F, axis=1)
    idx = (np.arange(3)[None, :] + r[:, None]) % 3
    R = np.take_along_axis(F, idx, axis=1)
    parity = R[:, 1] > R[:, 2]                 # orientation class within a vertex set
    key = np.sort(F, axis=1)
    _, lab, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    lab = lab.reshape(-1)
    dup = cnt[lab] > 1
    same_extra = 0
    rev = 0
    if dup.any():
        g = lab[dup]
        p = parity[dup]
        n_pos = np.bincount(g, weights=p, minlength=len(cnt))
        n_all = np.bincount(g, minlength=len(cnt))
        n_neg = n_all - n_pos
        both = (n_pos > 0) & (n_neg > 0)
        rev = int(np.minimum(n_pos, n_neg)[both].sum())
        same_extra = int((np.maximum(n_pos - 1, 0) + np.maximum(n_neg - 1, 0)).sum())
    return same_extra, rev, lab, parity


def near_coincident_pairs(V, tol):
    if len(V) < 2 or tol <= 0:
        return np.zeros((0, 2), np.int64)
    return cKDTree(V).query_pairs(tol, output_type="ndarray")


def component_table(mesh: Mesh, et, nm_verts, degen):
    F = mesh.F
    m = len(F)
    pairs, _ = T.face_adjacency(F, et, manifold_only=False)
    ncomp, lab = T.components(m, pairs)
    if m == 0:
        return [], lab
    he_cnt = et.count[et.he_edge].reshape(m, 3)
    face_bnd = (he_cnt == 1).any(1)
    face_nme = (he_cnt > 2).any(1)
    nmv = np.zeros(len(mesh.V), bool)
    nmv[nm_verts] = True
    face_nmv = nmv[F].any(1)
    _, same = T.orientation_parity(F, et)
    fp, _ = T.face_adjacency(F, et, manifold_only=True)
    face_incons = np.zeros(m, bool)
    if len(fp):
        face_incons[fp[same].reshape(-1)] = True
    area = G.tri_areas(mesh.V, F)
    ref = 0.5 * (mesh.V.min(0) + mesh.V.max(0))
    P = mesh.V[F] - ref
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    out = []
    for c in range(ncomp):
        s = lab == c
        closed = not face_bnd[s].any()
        manifold = not (face_nme[s].any() or face_nmv[s].any())
        oriented = not face_incons[s].any()
        v = float(vol[s].sum())
        a = float(area[s].sum())
        out.append({
            "faces": int(s.sum()), "area": a, "signed_volume": v,
            "closed": bool(closed), "manifold": bool(manifold), "consistently_oriented": bool(oriented),
            "degenerate_faces": int(degen[s].sum()),
            "healthy": bool(closed and manifold and oriented and not degen[s].any() and abs(v) > 0),
        })
    return out, lab


def diagnose(mesh: Mesh, *, weld_tol_abs=0.0, needle_aspect=1.0e3, with_si=True,
             loop_cap=50, engine="auto") -> dict:
    V, F = mesh.V, mesh.F
    m, n = len(F), len(V)
    d = {"faces": int(m), "vertices": int(n)}
    if m == 0:
        d.update(empty=True, watertight=False)
        return d
    # topology is measured on the duplicate-collapsed face set (one face per
    # vertex set); duplicates are reported once, as duplicates
    _, first = np.unique(np.sort(F, axis=1), axis=0, return_index=True)
    first = np.sort(first)
    Fu = F[first]
    mu = Mesh(V, Fu, mesh.src[first], mesh.attr[first], mesh.solid[first])
    et = T.edge_table(Fu, n)
    used = np.zeros(n, bool)
    used[F.reshape(-1)] = True
    idx_deg = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    degen = X.degenerate_mask(V, F, engine=engine) | idx_deg
    same_extra, rev, _, _ = duplicate_groups(F)
    nm_verts, _ = T.nonmanifold_vertices(Fu, et, n)
    loops, cloops = T.boundary_loops(Fu, et)
    _, same = T.orientation_parity(Fu, et)
    comps, lab = component_table(mu, et, nm_verts, degen[first])
    area = G.tri_areas(V, F)
    diag = G.bbox_diagonal(V)

    def loop_info(L):
        P = V[L]
        ext = P.max(0) - P.min(0)
        diam = float(np.max(np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2))) if len(L) <= 2000 \
            else float(np.linalg.norm(ext))
        return {"vertices": int(len(L)), "diameter": diam}
    linfo = sorted((loop_info(L) for L in loops), key=lambda x: (-x["diameter"], -x["vertices"]))
    d.update({
        "unreferenced_vertices": int((~used).sum()),
        "duplicate_vertices_exact": 0,  # bit-identical coordinates are merged at intake by construction
        "near_coincident_vertex_pairs": int(len(near_coincident_pairs(V[used], weld_tol_abs))) if weld_tol_abs > 0 else None,
        "duplicate_faces": int(same_extra),
        "reversed_duplicate_faces": int(rev),
        "degenerate_faces": int(degen.sum()),
        "index_degenerate_faces": int(idx_deg.sum()),
        "zero_area_faces": int((degen & ~idx_deg).sum()),
        "needle_faces": int(G.needle_mask(V, F, needle_aspect).sum()),
        "edges": int(len(et.count)),
        "boundary_edges": int((et.count == 1).sum()),
        "boundary_loops": int(len(loops)),
        "complex_boundary_loops": int(len(cloops)),
        "boundary_loop_list": linfo[:loop_cap],
        "nonmanifold_edges": int((et.count > 2).sum()),
        "nonmanifold_vertices": int(len(nm_verts)),
        "inconsistent_orientation_edges": int(same.sum()),
        "components": int(len(comps)),
        "healthy_components": int(sum(c["healthy"] for c in comps)),
        "component_list": sorted(comps, key=lambda c: -c["area"])[:loop_cap],
        "surface_area": float(area.sum()),
        "signed_volume": G.signed_volume(V, F),
        "abs_volume": float(sum(abs(c["signed_volume"]) for c in comps)),
        "bbox_min": V[used].min(0).tolist(),
        "bbox_max": V[used].max(0).tolist(),
        "bbox_diagonal": diag,
    })
    if with_si:
        si = X.self_intersections(V, F, engine=engine)
        d["self_intersections"] = X.si_counts(si)
    d["watertight"] = bool(d["boundary_edges"] == 0 and d["nonmanifold_edges"] == 0
                           and d["nonmanifold_vertices"] == 0 and d["inconsistent_orientation_edges"] == 0)
    d["outward"] = bool(all(c["signed_volume"] != 0 for c in comps)) and _outward(mu, comps, lab)
    return d


def _outward(mesh, comps, lab):
    """Every closed component has the sign its nesting depth requires."""
    from .ops import nesting_depth
    closed = [i for i, c in enumerate(comps) if c["closed"] and c["consistently_oriented"]]
    if not closed:
        return False
    depth = nesting_depth(mesh, lab, closed)
    for i in closed:
        want = 1.0 if depth[i] % 2 == 0 else -1.0
        if np.sign(comps[i]["signed_volume"]) != want:
            return False
    return True


def defect_vector(d):
    """Quantities that a repair must never increase."""
    si = d.get("self_intersections", {})
    return {
        "boundary_edges": d.get("boundary_edges", 0),
        "boundary_loops": d.get("boundary_loops", 0) + d.get("complex_boundary_loops", 0),
        "nonmanifold_edges": d.get("nonmanifold_edges", 0),
        "nonmanifold_vertices": d.get("nonmanifold_vertices", 0),
        "si_proper": si.get("proper", 0),
        "si_coplanar_overlap": si.get("coplanar_overlap", 0),
        "si_touching": si.get("touching", 0),
    }
