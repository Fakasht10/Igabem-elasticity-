"""Gated STL intake: stl_repair_core (top-level ../../stl_repair_core, PolyForm Noncommercial 1.0.0, derived from
Sutura 0.9.8) checks and conservatively repairs the file before geomio reads it.

The two licences stay separate.  This module only calls the optional package at runtime; nothing from it is copied
here.  Use is limited to noncommercial research (see stl_repair_core/LICENSE).

    m, info = read_stl_gated("part.stl", max_hole_diameter=0.5, report="part.repair.json")

PASS returns the repaired mesh with the CAD face ids kept.  Faces added by hole filling take the id of the
best-aligned tagged neighbour (the patch belongs to the face it closes).  FAIL raises StlGateError, which carries
the full report, so a broken part never reaches the solver silently."""
import os
import tempfile

import numpy as np

from .mesh import TriMesh
from .stl import FACEID_MARK, StlError, read_stl


class StlGateError(StlError):
    def __init__(self, msg, report):
        super().__init__(msg)
        self.report = report


def _core():
    try:
        import stl_repair_core
    except ImportError as e:
        raise StlError("gated STL intake needs stl_repair_core: pip install ./stl_repair_core") from e
    return stl_repair_core


def _inherit_ids(m, new):
    """new: bool per triangle with no source facet.  Each takes the id of the edge neighbour (tagged, not new) whose
    normal is closest to its own, repeated until patches of several triangles are filled from their rim inwards."""
    fid, N = m.fid.copy(), m.normals()
    T = m.T
    e = np.sort(np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 1).reshape(-1, 2), 1)
    tri = np.repeat(np.arange(len(T)), 3)
    key = e[:, 0] * (len(m.V) + 1) + e[:, 1]
    o = np.argsort(key, kind="stable")
    k, t = key[o], tri[o]
    pair = np.flatnonzero(k[1:] == k[:-1])
    a, b = np.r_[t[pair], t[pair + 1]], np.r_[t[pair + 1], t[pair]]   # directed neighbour list a -> b
    todo = new.copy()
    while todo.any():
        s = todo[a] & ~todo[b] & (fid[b] != 0)
        if not s.any():
            break
        aa, bb = a[s], b[s]
        score = np.einsum("ij,ij->i", N[aa], N[bb])
        best = np.full(len(T), -np.inf)
        np.maximum.at(best, aa, score)
        win = score == best[aa]
        fid[aa[win]] = fid[bb[win]]
        todo[np.unique(aa)] = False
    return fid


def read_stl_gated(path, report=None, keep=None, **thresholds):
    """-> (TriMesh, info).  thresholds go to stl_repair_core.repair_stl (max_hole_diameter, weld_tolerance,
    debris_area_threshold, max_surface_deviation, max_volume_change, require_watertight, protected_faces).
    report: path for the JSON report; keep: path for the repaired STL (default: a temporary file)."""
    core = _core()
    with open(path, "rb") as f:
        head = f.read(80)
    tagged = head.startswith(FACEID_MARK) or head.lstrip().startswith(b"solid face_")   # files geomio wrote
    thresholds.setdefault("require_watertight", True)
    with tempfile.TemporaryDirectory() as tmp:
        out = keep or os.path.join(tmp, "gated.stl")
        rep = core.repair_stl(path, out, json_path=report, protect_tags=tagged, **thresholds)
        if rep["status"] != "PASS":
            raise StlGateError("%s: stl_repair_core FAIL: %s" % (path, "; ".join(rep["problems"][:6])), rep)
        m, info = read_stl(out, repair=False, weld_tol=0.0)
        prov = rep.get("paths", {}).get("provenance")
        src = np.load(prov)["output_face_to_input_facet"] if prov else np.zeros(len(m.T), np.int64)
    if len(src) != len(m.T):
        raise StlError("provenance does not match the repaired mesh (%d vs %d faces)" % (len(src), len(m.T)))
    new = src < 0
    if new.any() and m.fid.any():
        m = TriMesh(m.V, m.T, _inherit_ids(m, new))
    info.update(facets=rep["input"]["facets"], watertight=bool(rep["after"]["watertight"]), gate=dict(
        status=rep["status"], adopted=[o["operation"] for o in rep["adopted_operations"]],
        rejected=[o["operation"] for o in rep["rejected_operations"]], new_faces=int(new.sum()),
        input_sha256=rep["input"]["sha256"], policy_hash=rep["policy_hash"], report=report))
    return m, info
