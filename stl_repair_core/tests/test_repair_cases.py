"""Repair matrix on synthetic fixtures, verified with independent oracles
(manifold3d, trimesh, PyMeshLab) on the reloaded output file."""
import json

import numpy as np
import pytest

import fixtures as fx
import oracles
from stl_repair_core import repair_stl, validate_stl
from stl_repair_core.stlio import file_sha256

DIAG = np.sqrt(3.0)

# name: (generator, kwargs, expected status, expected adopted ops (subset), expected closed volume or None)
CASES = {
    "clean_cube": (lambda: fx.box(n=2), {}, "PASS", [], 1.0),
    "missing_triangle": (fx.cube_missing_triangle, {"max_hole_diameter": 1.5}, "PASS", ["fill_hole"], 1.0),
    "missing_triangle_limit": (fx.cube_missing_triangle, {"max_hole_diameter": 0.3}, "PASS", [], None),
    "reversed_face": (fx.cube_reversed_face, {}, "PASS", ["reorient"], 1.0),
    "duplicates": (fx.cube_duplicates, {}, "PASS", ["remove_duplicates"], 1.0),
    "zero_area_cap": (fx.cube_zero_area_cap, {}, "PASS", ["remove_degenerate"], 1.0),
    "needle": (fx.cube_needle, {}, "PASS", [], 1.0),
    "nonmanifold_edge": (fx.two_cubes_sharing_edge, {}, "FAIL", [], None),
    "pinched_vertex": (fx.two_cubes_sharing_vertex, {}, "FAIL", [], None),
    "fin_default": (fx.cube_with_fin, {}, "FAIL", [], None),
    "fin_tolerant": (fx.cube_with_fin, {"max_surface_deviation": 0.05}, "PASS", ["remove_nm_fins"], 1.0),
    "proper_intersection": (fx.overlapping_cubes, {}, "FAIL", [], None),
    "poking_triangle": (fx.poking_triangle, {"max_hole_diameter": 1.0}, "FAIL", [], None),
    "coplanar_overlap": (fx.coplanar_cubes, {}, "FAIL", [], None),
    "multi_components": (fx.multi_components, {}, "PASS", [], 1.0 + 8.0 - 1.0),
    "debris_removed": (fx.cube_with_debris, {"debris_area_threshold": 1e-3}, "PASS", ["remove_debris"], 1.0),
    "debris_kept": (fx.cube_with_debris, {}, "PASS", [], None),          # open sheet = intentional opening
    "debris_kept_strict": (fx.cube_with_debris, {"require_watertight": True}, "FAIL", [], None),
    "thin_wall": (lambda: fx.thin_wall(1e-3), {"max_hole_diameter": 0.5}, "PASS", ["fill_hole"], float(np.float32(1e-3))),
}


def _run(tmp_path, name):
    gen, kw, *_ = CASES[name]
    V, F = gen()
    p = fx.soup_write(tmp_path / ("%s.stl" % name), V, F)
    out = tmp_path / ("%s_out.stl" % name)
    js = tmp_path / ("%s.json" % name)
    rep = repair_stl(p, out, json_path=js, **kw)
    return p, out, rep, json.loads(js.read_text())


@pytest.mark.parametrize("name", sorted(CASES))
def test_case(tmp_path, name):
    gen, kw, status, ops, vol = CASES[name]
    p, out, rep, rep_file = _run(tmp_path, name)
    assert rep_file["report_hash"] == rep["report_hash"]
    assert rep["status"] == status, rep["problems"]
    adopted = [o["operation"] for o in rep["adopted_operations"]]
    for op in ops:
        assert op in adopted, adopted
    if not ops and status == "PASS":
        # nothing to do -> byte-identical pass-through
        assert file_sha256(out) == file_sha256(p)
    if status == "FAIL":
        # the original mesh is returned, never a guess
        assert file_sha256(out) == file_sha256(p)
        return
    before, after = rep["before"], rep["after"]
    # zero unintended component loss
    removed_debris = sum(len(o["info"].get("removed_components", [])) for o in rep["adopted_operations"])
    assert after["components"] == before["components"] - removed_debris
    assert after["healthy_components"] >= before["healthy_components"]
    # no added self-intersections
    for k in ("proper", "coplanar_overlap", "touching"):
        assert after["self_intersections"][k] <= before["self_intersections"][k]
    # bounded bidirectional deviation of the existing surface
    dev = rep["deviation_final"]
    assert dev["displacement_max"] <= rep["thresholds"]["max_surface_deviation"] + 1e-15
    # independent reload + verification
    tm = oracles.trimesh_check(out)
    ml = oracles.meshlab_check(out)
    assert ml["self_intersecting_faces"] == 0
    if vol is not None:
        mf = oracles.manifold_check(out)
        assert mf["ok"], mf
        assert tm["watertight"] and tm["winding_consistent"]
        assert ml["boundary_edges"] == 0 and ml["non_two_manifold_edges"] == 0 and ml["non_two_manifold_vertices"] == 0
        assert abs(mf["volume"] - vol) <= 1e-6 * max(vol, 1e-3)
        assert abs(tm["volume"] - vol) <= 1e-6 * max(vol, 1e-3)
        assert abs(after["signed_volume"] - vol) <= 1e-9 * max(vol, 1e-3)
        # area conservation: input area minus removed debris/fin plus hole patches
        assert abs(after["surface_area"] - mf["area"]) <= 1e-6 * mf["area"]


def test_intentional_opening_preserved(tmp_path):
    p, out, rep, _ = _run(tmp_path, "missing_triangle_limit")
    assert rep["after"]["boundary_loops"] == 1 and rep["output"]["intentional_openings"] == 1
    assert oracles.meshlab_check(out)["boundary_edges"] == 3


def test_deterministic_hashes(tmp_path):
    h = []
    for k in range(3):
        d = tmp_path / str(k)
        d.mkdir()
        _, out, rep, _ = _run(d, "missing_triangle")
        h.append((rep["report_hash"], rep["output"]["geometry_hash"], rep["output"]["sha256"], rep["policy_hash"]))
    assert h[0] == h[1] == h[2]


def test_validate_reports(tmp_path):
    V, F = fx.cube_missing_triangle()
    p = fx.soup_write(tmp_path / "m.stl", V, F)
    r = validate_stl(p)
    assert r["status"] == "FAIL" and r["diagnosis"]["boundary_loops"] == 1
    assert r["reload"]["geometry_hash_equal"]
    V, F = fx.box(n=3)
    p = fx.soup_write(tmp_path / "c.stl", V, F)
    r = validate_stl(p)
    assert r["status"] == "PASS" and r["diagnosis"]["watertight"] and r["reload"]["watertight_after_reload"]
    assert abs(r["diagnosis"]["signed_volume"] - 1.0) < 1e-12
    assert abs(r["diagnosis"]["surface_area"] - 6.0) < 1e-12
    assert abs(r["diagnosis"]["bbox_diagonal"] - DIAG) < 1e-6


def test_diagnosis_counts(tmp_path):
    from stl_repair_core.diagnose import diagnose
    from stl_repair_core.mesh import canonical, from_soup
    from stl_repair_core.stlio import Soup

    def d(V, F):
        P = np.asarray(V, float)[np.asarray(F)]
        n = len(P)
        return diagnose(canonical(from_soup(Soup(P, np.zeros(n, np.uint16), np.zeros(n, np.int32), [], "binary", b"", "", 0))))
    r = d(*fx.cube_duplicates())
    assert r["duplicate_faces"] == 1 and r["reversed_duplicate_faces"] == 1
    r = d(*fx.two_cubes_sharing_edge())
    assert r["nonmanifold_edges"] == 1 and r["components"] == 1
    r = d(*fx.two_cubes_sharing_vertex())
    assert r["nonmanifold_vertices"] == 1 and r["nonmanifold_edges"] == 0
    r = d(*fx.cube_zero_area_cap())
    assert r["zero_area_faces"] == 1
    r = d(*fx.cube_needle())
    assert r["needle_faces"] >= 1 and r["degenerate_faces"] == 0 and r["watertight"]
    r = d(*fx.cube_reversed_face())
    assert r["inconsistent_orientation_edges"] == 3
    r = d(*fx.overlapping_cubes())
    assert r["self_intersections"]["proper"] > 0 and r["components"] == 2
    r = d(*fx.coplanar_cubes())
    assert r["self_intersections"]["coplanar_overlap"] > 0
    r = d(*fx.cube_with_debris())
    assert r["components"] == 2 and r["healthy_components"] == 1
