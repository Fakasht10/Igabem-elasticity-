# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Gated repair driver, validation and JSON reports.

Every operation produces a candidate that is adopted only if all adoption
gates hold (Sutura 0.9.8 ``pipeline/adopt_guard.py`` ``check_candidate`` and
the AGENTS.md "shape, not bytes" rule, extended with distance, volume,
opening, component and provenance gates).  If the final mesh is not a valid
structural surface the ORIGINAL input bytes are written and the report says
FAIL; nothing is reconstructed or guessed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np

from . import __version__
from . import geometry as G
from . import ops as O
from . import stlio
from . import intersect as X
from . import topology as T
from .diagnose import defect_vector, diagnose
from .mesh import canonical, from_soup, geometry_hash

OP_ORDER = ("remove_degenerate", "remove_duplicates", "tolerance_weld", "remove_nm_fins",
            "reorient", "remove_debris", "fill_holes")

DEFAULTS = {
    "max_hole_diameter": 0.0,         # model units; 0 disables hole filling
    "weld_tolerance": 1.0e-6,         # fraction of the bounding-box diagonal
    "debris_area_threshold": 0.0,     # model units^2; 0 disables debris removal
    "max_surface_deviation": None,    # model units; None -> 1e-5 x bbox diagonal
    "max_volume_change": 1.0e-4,      # fraction of |volume|
    "max_hole_vertices": 100,
    "needle_aspect": 1.0e3,
}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not isinstance(v, np.ndarray) or v.size < 64}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def policy_hash(thresholds: dict) -> str:
    blob = json.dumps({"version": __version__, "op_order": OP_ORDER, "thresholds": thresholds},
                      sort_keys=True, default=float)
    return hashlib.sha256(blob.encode()).hexdigest()


def report_hash(rep: dict) -> str:
    """Hash of the report without wall-clock fields."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not k.startswith("runtime") and k not in ("report_hash", "paths")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(strip(rep), sort_keys=True, default=float).encode()).hexdigest()


# --------------------------------------------------------------------- context
class _Ctx:
    def __init__(self, mesh0, diag0, thr, protected):
        self.mesh0 = mesh0
        self.thr = thr
        self.protected = protected
        self.index0 = G.SurfaceIndex(mesh0.V, mesh0.F)
        self.diag = diag0["bbox_diagonal"]
        # input facet coordinates by src, for unchanged-face detection
        self.src_coords = {}
        P = mesh0.V[mesh0.F]
        self.P0 = P
        self.src0 = mesh0.src
        self.src_pos = np.full(int(mesh0.src.max()) + 1 if mesh0.nf else 0, -1, np.int64)
        self.src_pos[mesh0.src] = np.arange(mesh0.nf)
        # healthy input components (by src) must survive
        et = T.edge_table(mesh0.F, len(mesh0.V))
        from .diagnose import component_table
        from . import intersect as X
        nm, _ = T.nonmanifold_vertices(mesh0.F, et, len(mesh0.V))
        deg = X.degenerate_mask(mesh0.V, mesh0.F)
        comps, lab = component_table(mesh0, et, nm, deg)
        self.healthy_src = [mesh0.src[lab == i] for i, c in enumerate(comps) if c["healthy"]]
        # intentional openings: every loop above the fill limit, and every complex loop
        loops, cloops = T.boundary_loops(mesh0.F, et)
        keep_edges = set()
        self.openings = 0
        for L in loops + cloops:
            Pl = mesh0.V[L]
            diam = float(np.max(np.linalg.norm(Pl[:, None] - Pl[None], axis=2))) if len(L) <= 2000 else np.inf
            if not any(L is c for c in cloops) and diam > thr["max_hole_diameter"]:
                self.openings += 1
                for i in range(len(L)):
                    keep_edges.add(_ekey(mesh0.V[L[i]], mesh0.V[L[(i + 1) % len(L)]]))
        self.opening_edges = keep_edges
        self.vref = max(abs(diag0.get("abs_volume", 0.0)), 1e-300)
        self.cum_dvol = 0.0


def _ekey(p, q):
    a, b = tuple(p.tolist()), tuple(q.tolist())
    return (a, b) if a <= b else (b, a)


def _tri_key(P):
    return tuple(sorted(tuple(p) for p in np.asarray(P).tolist()))


def _boundary_ekeys(mesh):
    et = T.edge_table(mesh.F, len(mesh.V))
    E = et.edges[et.count == 1]
    return {_ekey(mesh.V[a], mesh.V[b]) for a, b in E.tolist()}


def _changed_faces(ctx, cand):
    """Input faces not present unchanged in cand; cand faces not identical to
    their input face (new faces included)."""
    pos = np.where(cand.src >= 0, ctx.src_pos[np.maximum(cand.src, 0)], -1)
    P = cand.V[cand.F]
    same = np.zeros(cand.nf, bool)
    ok = pos >= 0
    if ok.any():
        P0 = ctx.P0[pos[ok]]
        # identical up to cyclic rotation or reversal (surface unchanged)
        Pc = P[ok]
        eq = np.zeros(ok.sum(), bool)
        for perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)):
            eq |= (Pc[:, perm] == P0).all(axis=(1, 2))
        same[np.flatnonzero(ok)[eq]] = True
    present = np.zeros(ctx.mesh0.nf, bool)
    present[pos[same]] = True
    return np.flatnonzero(~present), np.flatnonzero(~same)


def _deviation(ctx, cand, exempt_src):
    gone, changed = _changed_faces(ctx, cand)
    if exempt_src is not None and len(exempt_src):
        gone = gone[~np.isin(ctx.src0[gone], exempt_src)]
    A0 = float(G.tri_areas(ctx.mesh0.V, ctx.mesh0.F).sum())
    Ac = float(G.tri_areas(cand.V, cand.F).sum())
    idx_c = G.SurfaceIndex(cand.V, cand.F)
    d_in, w_in, h_in = G.one_sided(ctx.mesh0.V, ctx.mesh0.F, idx_c, gone)
    existing = changed[cand.src[changed] >= 0]
    new = changed[cand.src[changed] < 0]
    d_out, w_out, h_out = G.one_sided(cand.V, cand.F, ctx.index0, existing)
    d_new, w_new, h_new = G.one_sided(cand.V, cand.F, ctx.index0, new)
    return {
        "input_to_output": G.weighted_stats(d_in, w_in, A0),
        "output_to_input_existing": G.weighted_stats(d_out, w_out, Ac),
        "output_to_input_new_faces": G.weighted_stats(d_new, w_new, Ac),
        "displacement_max": float(max(d_in.max() if len(d_in) else 0.0, d_out.max() if len(d_out) else 0.0)),
        "sample_spacing": float(max(h_in, h_out, h_new)),
        "sampled_input_faces": int(len(gone)), "sampled_output_faces": int(len(changed)),
    }


def _local_dvol(before, cand):
    """Signed-volume change of the changed region, measured about the
    centroid of the changed faces' vertices (exact for closed->closed, the
    patch-vs-fan volume for hole fills, 0 for coplanar duplicates)."""
    Pb, Pc = before.V[before.F], cand.V[cand.F]

    def canon_rows(P):
        """Oriented triangle key invariant to cyclic rotation."""
        best = P.reshape(len(P), -1)
        for r in (1, 2):
            R = np.roll(P, -r, axis=1).reshape(len(P), -1)
            best = np.where(_lexless(R, best)[:, None], R, best)
        return best
    kb, kc = canon_rows(Pb), canon_rows(Pc)
    from collections import Counter
    cb = Counter(map(bytes, np.ascontiguousarray(kb)))
    cc = Counter(map(bytes, np.ascontiguousarray(kc)))
    rem = cb - cc
    add = cc - cb
    def tris(cnt):
        out = []
        for k, n in cnt.items():
            t = np.frombuffer(k, dtype=np.float64).reshape(3, 3)
            out += [t] * n
        return np.asarray(out).reshape(-1, 3, 3)
    R, A = tris(rem), tris(add)
    allp = np.concatenate([R.reshape(-1, 3), A.reshape(-1, 3)])
    if len(allp) == 0:
        return 0.0
    ref = allp.mean(0)

    def vol(P):
        if len(P) == 0:
            return 0.0
        Q = P - ref
        return float(np.einsum("ij,ij->i", Q[:, 0], np.cross(Q[:, 1], Q[:, 2])).sum() / 6.0)
    return vol(A) - vol(R)


def _lexless(a, b):
    """Row-wise lexicographic a < b."""
    neq = a != b
    first = np.argmax(neq, axis=1)
    rows = np.arange(len(a))
    return neq.any(1) & (a[rows, first] < b[rows, first])


def _gates(ctx, cur, cur_d, cand, cand_d, info, op):
    reasons = []
    dv_c, dv_n = defect_vector(cur_d), defect_vector(cand_d)
    for k in dv_c:
        if dv_n[k] > dv_c[k]:
            reasons.append("%s increased %d -> %d" % (k, dv_c[k], dv_n[k]))
    for k in ("degenerate_faces", "inconsistent_orientation_edges"):
        if cand_d[k] > cur_d[k]:
            reasons.append("%s increased %d -> %d" % (k, cur_d[k], cand_d[k]))
    if not np.isfinite(cand.V).all():
        reasons.append("non-finite coordinates")
    have = set(cand.src.tolist())
    for hs in ctx.healthy_src:
        if not set(hs.tolist()) <= have:
            reasons.append("a healthy connected component lost faces")
            break
    if ctx.protected is not None and len(ctx.protected):
        missing = np.setdiff1d(ctx.protected, cand.src)
        if len(missing):
            # an exact (or reversed) duplicate of a surviving face loses no surface
            have_keys = {_tri_key(t) for t in cand.V[cand.F]}
            lost = [s_ for s_ in missing.tolist() if _tri_key(ctx.P0[ctx.src_pos[s_]]) not in have_keys]
            if lost:
                reasons.append("%d protected faces removed" % len(lost))
    if ctx.opening_edges:
        b = _boundary_ekeys(cand)
        gone = ctx.opening_edges - b
        ex = info.get("exempt_src")
        if gone and ex is not None and len(ex):
            # openings on explicitly removed debris components do not count
            exk = set()
            for P in ctx.P0[ctx.src_pos[np.asarray(ex, np.int64)]]:
                for i in range(3):
                    exk.add(_ekey(P[i], P[(i + 1) % 3]))
            gone = gone - exk
        lost = len(gone)
        if lost:
            reasons.append("an intentional opening (> max_hole_diameter) was modified (%d edges)" % lost)
    dev = _deviation(ctx, cand, info.get("exempt_src"))
    if dev["displacement_max"] > ctx.thr["max_surface_deviation"]:
        reasons.append("surface displacement %.3g > %.3g" % (dev["displacement_max"], ctx.thr["max_surface_deviation"]))
    dvol = _local_dvol(cur, cand)
    if op in ("remove_debris", "reorient"):
        # debris: explicit user threshold, reported separately; reorient: the
        # point set is unchanged, only the sign convention of the faces
        dvol = 0.0
    cum = ctx.cum_dvol + abs(dvol)
    if cum > ctx.thr["max_volume_change"] * ctx.vref:
        reasons.append("cumulative |dV| %.3g > %.3g x |V|" % (cum, ctx.thr["max_volume_change"]))
    return reasons, dev, dvol


def _status(d, ctx):
    """PASS criteria for a structural surface; openings above the limit allowed."""
    si = d.get("self_intersections", {})
    problems = []
    for k in ("nonmanifold_edges", "nonmanifold_vertices", "degenerate_faces", "duplicate_faces",
              "reversed_duplicate_faces", "inconsistent_orientation_edges", "complex_boundary_loops"):
        if d.get(k, 0):
            problems.append("%s=%d" % (k, d[k]))
    for k in ("proper", "coplanar_overlap", "touching"):
        if si.get(k, 0):
            problems.append("self_intersections.%s=%d" % (k, si[k]))
    small = [L for L in d.get("boundary_loop_list", []) if L["diameter"] <= ctx.thr["max_hole_diameter"]]
    if small:
        problems.append("%d boundary loops at or below max_hole_diameter remain" % len(small))
    if d.get("boundary_loops", 0) > ctx.openings:
        problems.append("boundary loops (%d) exceed the intentional openings of the input (%d)"
                        % (d["boundary_loops"], ctx.openings))
    if d.get("boundary_edges", 0) == 0 and not d.get("outward", False):
        problems.append("closed components not outward-oriented by nesting depth")
    return problems


def _resolve(thr, diag0):
    t = dict(DEFAULTS)
    t.update({k: v for k, v in thr.items() if v is not None})
    if t["max_surface_deviation"] is None:
        t["max_surface_deviation"] = 1.0e-5 * diag0
    t["weld_tolerance_abs"] = t["weld_tolerance"] * diag0
    return t


def _protected_ids(protected, soup):
    """dict name -> ids, list of ids, or None.  Returns (ids array, regions dict)."""
    if protected is None:
        return np.zeros(0, np.int64), {}
    if isinstance(protected, dict):
        regions = {str(k): np.asarray(sorted(set(int(i) for i in v)), np.int64) for k, v in protected.items()}
    else:
        regions = {"protected": np.asarray(sorted(set(int(i) for i in protected)), np.int64)}
    ids = np.unique(np.concatenate(list(regions.values()))) if regions else np.zeros(0, np.int64)
    if len(ids) and (ids.min() < 0 or ids.max() >= len(soup.tri)):
        raise ValueError("protected face id out of range 0..%d" % (len(soup.tri) - 1))
    return ids, regions


def regions_from_tags(soup, attr=True, solids=True):
    """Named regions from binary attribute words (geomio face ids) or ASCII solids."""
    reg = {}
    if attr and soup.fmt == "binary":
        for v in np.unique(soup.attr):
            if v != 0:
                reg["attr:%d" % v] = np.flatnonzero(soup.attr == v)
    if solids and soup.fmt == "ascii" and len(soup.solid_names) > 1:
        for i, nm in enumerate(soup.solid_names):
            reg["solid:%s" % (nm or i)] = np.flatnonzero(soup.solid == i)
    return reg


def _provenance(mesh0, cur, regions, removed_by):
    src = cur.src
    stats = {
        "input_faces": int(mesh0.nf), "output_faces": int(cur.nf),
        "output_from_input": int((src >= 0).sum()), "output_new": int((src < 0).sum()),
        "input_kept": int(len(np.unique(src[src >= 0]))),
        "input_removed": int(mesh0.nf - len(np.unique(src[src >= 0]))),
        "input_removed_by_operation": {k: int(len(v)) for k, v in removed_by.items()},
        "output_faces_split_from_one_input": int((np.bincount(src[src >= 0]) > 1).sum()) if (src >= 0).any() else 0,
    }
    A = G.tri_areas(cur.V, cur.F)
    A0 = G.tri_areas(mesh0.V, mesh0.F)
    rr = {}
    for name, ids in regions.items():
        kept = np.isin(src, ids)
        rr[name] = {"input_faces": int(len(ids)), "output_faces": int(kept.sum()),
                    "input_faces_recovered": int(len(np.intersect1d(ids, src))),
                    "input_area": float(A0[np.isin(mesh0.src, ids)].sum()), "output_area": float(A[kept].sum())}
    stats["regions"] = rr
    return stats


# --------------------------------------------------------------------- API
def validate_stl(input_path, json_path=None, *, weld_tolerance=DEFAULTS["weld_tolerance"],
                 engine="auto", allow_open=False):
    t0 = time.perf_counter()
    rep = {"tool": "stl_repair_core", "version": __version__, "command": "validate",
           "paths": {"input": str(input_path)}}
    try:
        soup = stlio.read_stl(input_path)
    except (stlio.StlError, OSError) as exc:
        rep.update(status="FAIL", error=str(exc), runtime_s=time.perf_counter() - t0)
        rep["report_hash"] = report_hash(rep)
        _dump(rep, json_path)
        return rep
    mesh = canonical(from_soup(soup))
    diag0 = G.bbox_diagonal(mesh.V)
    d = diagnose(mesh, weld_tol_abs=weld_tolerance * diag0, engine=engine)
    rl = _reload_check(mesh, soup, d, engine)
    problems = []
    for k in ("nonmanifold_edges", "nonmanifold_vertices", "degenerate_faces", "duplicate_faces",
              "reversed_duplicate_faces", "inconsistent_orientation_edges", "complex_boundary_loops"):
        if d.get(k, 0):
            problems.append("%s=%d" % (k, d[k]))
    for k in ("proper", "coplanar_overlap", "touching"):
        if d["self_intersections"].get(k, 0):
            problems.append("self_intersections.%s=%d" % (k, d["self_intersections"][k]))
    if d["boundary_edges"] and not allow_open:
        problems.append("boundary_edges=%d" % d["boundary_edges"])
    if d["boundary_edges"] == 0 and not d["outward"]:
        problems.append("not outward-oriented")
    if not rl["watertight_after_reload"] and not allow_open:
        problems.append("not watertight after save/reload")
    rep.update({
        "status": "PASS" if not problems else "FAIL", "problems": problems,
        "input": {"sha256": soup.sha256, "bytes": soup.nbytes, "format": soup.fmt, "facets": int(len(soup.tri)),
                  "geometry_hash": geometry_hash(mesh), "units": stlio.detect_units(soup.header, soup.tri),
                  "warnings": soup.warnings},
        "diagnosis": d, "reload": rl, "thresholds": {"weld_tolerance": weld_tolerance},
        "policy_hash": policy_hash({"weld_tolerance": weld_tolerance, "allow_open": allow_open}),
        "runtime_s": time.perf_counter() - t0,
    })
    rep = _jsonable(rep)
    rep["report_hash"] = report_hash(rep)
    _dump(rep, json_path)
    return rep


def _reload_check(mesh, soup, d, engine, path=None):
    """Save in the input format (or reload ``path``), re-read, compare."""
    tmpdir = None
    if path is None:
        tmpdir = tempfile.mkdtemp(prefix="stlrc_")
        path = os.path.join(tmpdir, "reload.stl")
        stlio.write_stl(path, mesh.V, mesh.F, fmt=soup.fmt, attr=mesh.attr, header=soup.header,
                        solid_of_face=mesh.solid, solid_names=soup.solid_names)
    try:
        s2 = stlio.read_stl(path)
        m2 = canonical(from_soup(s2))
        d2 = diagnose(m2, with_si=False, engine=engine)
        out = {"geometry_hash_equal": geometry_hash(m2) == geometry_hash(mesh),
               "watertight_after_reload": bool(d2["watertight"]),
               "boundary_edges": d2["boundary_edges"], "nonmanifold_edges": d2["nonmanifold_edges"],
               "nonmanifold_vertices": d2["nonmanifold_vertices"], "faces": d2["faces"]}
        if soup.fmt == "ascii" and len(mesh.V):
            V32 = mesh.V.astype(np.float32).astype(np.float64)
            m3 = canonical(from_soup(stlio.Soup(V32[mesh.F], mesh.attr, mesh.solid, [], "binary", b"", "", 0)))
            d3 = diagnose(m3, with_si=False, engine=engine)
            out["float32_binary_roundtrip_watertight"] = bool(d3["watertight"])
        return out
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _dump(rep, json_path):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(rep, f, indent=1, sort_keys=True, default=float)


def repair_stl(input_path, output_path, max_hole_diameter=DEFAULTS["max_hole_diameter"],
               weld_tolerance=DEFAULTS["weld_tolerance"], debris_area_threshold=DEFAULTS["debris_area_threshold"],
               max_surface_deviation=DEFAULTS["max_surface_deviation"],
               max_volume_change=DEFAULTS["max_volume_change"], *, protected_faces=None,
               protect_tags=False, json_path=None, engine="auto",
               max_hole_vertices=DEFAULTS["max_hole_vertices"], write_provenance=True, require_watertight=False):
    """Conservative repair.  Returns the report dict (also written to
    ``json_path`` when given).  On FAIL the original input bytes are written
    to ``output_path``."""
    t0 = time.perf_counter()
    timing = {}
    rep = {"tool": "stl_repair_core", "version": __version__, "command": "repair",
           "paths": {"input": str(input_path), "output": str(output_path)}}
    try:
        soup = stlio.read_stl(input_path)
    except (stlio.StlError, OSError) as exc:
        rep.update(status="FAIL", error=str(exc), output_written=False, runtime_s=time.perf_counter() - t0)
        rep["report_hash"] = report_hash(rep)
        _dump(rep, json_path)
        return rep
    pid, regions = _protected_ids(protected_faces, soup)
    if protect_tags:
        tag_regions = regions_from_tags(soup)
        regions.update(tag_regions)
        if tag_regions:
            pid = np.unique(np.concatenate([pid] + list(tag_regions.values())))
    mesh0 = canonical(from_soup(soup))
    early = None
    if mesh0.nf == 0:
        early = "empty mesh: no facets"
    else:
        from . import intersect as X
        if X.degenerate_mask(mesh0.V, mesh0.F, engine=engine).all():
            early = "all faces are degenerate (zero area); nothing to repair"
    if early:
        rep.update(status="FAIL", error=early, output_written=False,
                   input={"sha256": soup.sha256, "bytes": soup.nbytes, "format": soup.fmt, "facets": int(len(soup.tri))},
                   runtime_s=time.perf_counter() - t0)
        rep["report_hash"] = report_hash(rep)
        _dump(rep, json_path)
        return rep
    diag0 = G.bbox_diagonal(mesh0.V)
    thr = _resolve({"max_hole_diameter": max_hole_diameter, "weld_tolerance": weld_tolerance,
                    "debris_area_threshold": debris_area_threshold, "max_surface_deviation": max_surface_deviation,
                    "max_volume_change": max_volume_change, "max_hole_vertices": max_hole_vertices,
                    "require_watertight": bool(require_watertight)}, diag0)
    ts = time.perf_counter()
    d0 = diagnose(mesh0, weld_tol_abs=thr["weld_tolerance_abs"], engine=engine)
    timing["diagnose_input"] = time.perf_counter() - ts
    ctx = _Ctx(mesh0, d0, thr, pid)
    cur, cur_d = mesh0, d0
    adopted, rejected, skipped = [], [], []
    removed_by = {}

    def attempt(name, res, extra=None):
        nonlocal cur, cur_d
        if res is None:
            skipped.append(name)
            return False
        cand, info = res
        ts_ = time.perf_counter()
        if "new_face_corners" in info:
            # add-only operation: pairs among unchanged faces are unchanged, so
            # the global SI count is the old count plus the pairs touching the
            # new faces (exactly equivalent to a full rescan)
            cand_d = diagnose(cand, weld_tol_abs=thr["weld_tolerance_abs"], engine=engine, with_si=False)
            keys = {_tri_key(P) for P in info["new_face_corners"]}
            newf = [i for i in np.flatnonzero(cand.src < 0).tolist() if _tri_key(cand.V[cand.F[i]]) in keys]
            loc = X.si_counts(X.self_intersections(cand.V, cand.F, subset=newf, engine=engine))
            si = dict(cur_d["self_intersections"])
            for k in ("adjacent_contact", "touching", "coplanar_overlap", "proper"):
                si[k] = si[k] + loc[k]
            si["local_recheck_pairs"] = loc["candidate_pairs"]
            for k in ("touching_faces", "coplanar_overlap_faces", "proper_faces"):
                si[k] = si[k] + loc[k]
            cand_d["self_intersections"] = si
        else:
            cand_d = diagnose(cand, weld_tol_abs=thr["weld_tolerance_abs"], engine=engine)
        reasons, dev, dvol = _gates(ctx, cur, cur_d, cand, cand_d, info, name.split(":")[0])
        rec = {"operation": name, "info": {k: v for k, v in info.items()
                                           if k not in ("removed_src", "exempt_src", "new_face_corners")},
               "defects_before": defect_vector(cur_d), "defects_after": defect_vector(cand_d),
               "faces_before": cur.nf, "faces_after": cand.nf, "deviation": dev, "local_dvolume": dvol,
               "runtime_s": time.perf_counter() - ts_}
        if extra:
            rec.update(extra)
        if reasons:
            rec["rejected_because"] = reasons
            rejected.append(rec)
            return False
        ctx.cum_dvol += abs(dvol)
        adopted.append(rec)
        if "removed_src" in info and len(info["removed_src"]):
            removed_by.setdefault(name.split(":")[0], []).extend(np.asarray(info["removed_src"]).tolist())
        cur, cur_d = cand, cand_d
        return True

    for name in OP_ORDER:
        ts = time.perf_counter()
        if name == "remove_degenerate":
            attempt(name, O.remove_degenerate(cur, pid, engine=engine))
        elif name == "remove_duplicates":
            attempt(name, O.remove_duplicates(cur, pid))
        elif name == "tolerance_weld":
            attempt(name, O.tolerance_weld(cur, thr["weld_tolerance_abs"], pid))
        elif name == "remove_nm_fins":
            attempt(name, O.remove_nm_fins(cur, pid))
        elif name == "reorient":
            si = cur_d.get("self_intersections", {})
            attempt(name, O.reorient(cur, component_flips=(si.get("proper", 0) + si.get("coplanar_overlap", 0)) == 0))
        elif name == "remove_debris":
            attempt(name, O.remove_debris(cur, thr["debris_area_threshold"], pid))
        elif name == "fill_holes":
            if thr["max_hole_diameter"] > 0:
                n_try = 0
                tried = set()
                while True:
                    et = T.edge_table(cur.F, len(cur.V))
                    loops, _ = T.boundary_loops(cur.F, et)
                    todo = []
                    for L in loops:
                        P = cur.V[L]
                        diam = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=2))) if len(L) <= 2000 else np.inf
                        key = tuple(sorted(map(tuple, P.tolist())))
                        if diam <= thr["max_hole_diameter"] and key not in tried:
                            todo.append((diam, key, L))
                    if not todo:
                        break
                    todo.sort(key=lambda x: (x[0], x[1]))
                    diam, key, L = todo[0]
                    tried.add(key)
                    n_try += 1
                    res, why = O.fill_hole(cur, L, max_vertices=thr["max_hole_vertices"])
                    if res is None:
                        rejected.append({"operation": "fill_hole", "loop_vertices": int(len(L)),
                                         "loop_diameter": diam, "rejected_because": [why]})
                        continue
                    attempt("fill_hole", res, {"loop_diameter": diam})
                if any(o["operation"] == "fill_hole" for o in adopted):
                    # global confirmation of the local rechecks
                    full = diagnose(cur, weld_tol_abs=thr["weld_tolerance_abs"], engine=engine)
                    a, b = full["self_intersections"], cur_d["self_intersections"]
                    if any(a[k] != b[k] for k in ("touching", "coplanar_overlap", "proper")):
                        raise RuntimeError("local SI recheck disagrees with global rescan: %s vs %s" % (a, b))
                    cur_d = full
            else:
                skipped.append("fill_holes (max_hole_diameter = 0)")
        timing[name] = time.perf_counter() - ts

    problems = _status(cur_d, ctx)
    if require_watertight and not cur_d.get("watertight", False):
        problems.append("output is not watertight (require_watertight=True)")
    out_fmt = soup.fmt
    ts = time.perf_counter()
    status = "PASS" if not problems else "FAIL"
    if status == "PASS" and not adopted:
        shutil.copyfile(input_path, output_path)          # byte pass-through
        written = "input bytes (no operation adopted)"
    elif status == "PASS":
        stlio.write_stl(output_path, cur.V, cur.F, fmt=out_fmt, attr=cur.attr, header=soup.header,
                        solid_of_face=cur.solid, solid_names=soup.solid_names)
        written = "repaired mesh"
    else:
        shutil.copyfile(input_path, output_path)
        written = "input bytes (FAIL: original mesh returned)"
    final = cur if status == "PASS" else mesh0
    rl = _reload_check(final, soup, None, engine, path=output_path)
    if not rl["geometry_hash_equal"]:
        status = "FAIL"
        problems.append("reloaded output differs from the adopted mesh")
    timing["write_reload"] = time.perf_counter() - ts
    final_d = cur_d if status == "PASS" else d0
    prov = _provenance(mesh0, final, regions, removed_by if status == "PASS" else {})
    if status == "PASS" and write_provenance:
        pp = str(output_path) + ".provenance.npz"
        np.savez_compressed(pp, output_face_to_input_facet=final.src, attr=final.attr,
                            input_facets=np.int64(mesh0.nf))
        rep["paths"]["provenance"] = pp
    rep.update({
        "status": status, "problems": problems, "output_written": written,
        "thresholds": {k: thr[k] for k in sorted(thr)}, "policy_hash": policy_hash(thr),
        "input": {"sha256": soup.sha256, "bytes": soup.nbytes, "format": soup.fmt, "facets": int(len(soup.tri)),
                  "geometry_hash": geometry_hash(mesh0), "units": stlio.detect_units(soup.header, soup.tri),
                  "warnings": soup.warnings},
        "output": {"sha256": stlio.file_sha256(output_path), "format": out_fmt,
                   "geometry_hash": geometry_hash(final), "watertight": bool(final_d["watertight"]),
                   "intentional_openings": int(ctx.openings)},
        "before": d0, "after": final_d, "attempted_best_after": cur_d if status == "FAIL" else None,
        "adopted_operations": adopted, "rejected_operations": rejected, "skipped_operations": skipped,
        "cumulative_abs_dvolume": ctx.cum_dvol,
        "deviation_final": _deviation(ctx, final, np.asarray(removed_by.get("remove_debris", []), np.int64))
        if status == "PASS" else None,
        "provenance": prov, "reload": rl,
        "runtime_breakdown_s": timing, "runtime_s": time.perf_counter() - t0,
    })
    rep = _jsonable(rep)
    rep["report_hash"] = report_hash(rep)
    _dump(rep, json_path)
    return rep
