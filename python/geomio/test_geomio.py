"""Regression tests of geomio: STL/3MF round trips keep face ids, the reader catches what materializr found silent
STL readers get wrong, repair closes and orients broken meshes, and STEP parts become watertight face-tagged meshes
that the cut-cell quadrature integrates to the mesh volume.

    python3 -m pytest python/geomio -q        (STEP tests need pip gmsh + libglu1-mesa; cut-cell test needs numba)
"""
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import geomio as gio  # noqa: E402
from geomio.stl import FACEID_MARK  # noqa: E402

STEP = os.path.join(HERE, "..", "..", "geo", "step")


def box(lo=(0, 0, 0), hi=(2, 1, 0.5)):
    """12 outward triangles, face id 1..6 per side."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    Q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    T = [t for a, b, c, d in Q for t in ((a, b, c), (a, c, d))]
    return gio.TriMesh(V, T, np.repeat(np.arange(1, 7), 2))


def cylinder(r=1.0, z0=0.0, z1=2.0, k=48):
    th = 2 * np.pi * np.arange(k) / k
    ring = np.c_[r * np.cos(th), r * np.sin(th)]
    V = np.vstack([np.c_[ring, np.full(k, z0)], np.c_[ring, np.full(k, z1)], [[0, 0, z0], [0, 0, z1]]])
    T, F = [], []
    for i in range(k):
        j = (i + 1) % k
        T += [(i, j, k + j), (i, k + j, k + i)]; F += [1, 1]
        T += [(2 * k, j, i)]; F += [2]
        T += [(2 * k + 1, k + i, k + j)]; F += [3]
    return gio.TriMesh(V, T, F)


def soup(m):
    """per-facet vertex copies, as an STL stores them."""
    return gio.TriMesh(m.V[m.T].reshape(-1, 3), np.arange(3 * len(m.T)).reshape(-1, 3), m.fid)


def same_surface(a, b, tol=1e-6):
    """same triangles (as point sets, any vertex order) and the same face id on each."""
    def key(m):
        P = np.round(m.V[m.T] / tol).astype(np.int64)
        P = np.sort(P.view([("x", np.int64), ("y", np.int64), ("z", np.int64)]).reshape(-1, 3), axis=1)
        return sorted(zip(map(bytes, P), m.fid.tolist()))
    return key(a) == key(b)


# ---------------------------------------------------------------- STL format
@pytest.mark.parametrize("binary", [True, False])
def test_stl_roundtrip_keeps_face_ids(tmp_path, binary):
    m = box()
    p = str(tmp_path / "box.stl")
    gio.write_stl(p, m, binary=binary)
    r, info = gio.read_stl(p)
    assert info["format"] == ("binary" if binary else "ascii")
    assert info["watertight"] and len(r.V) == 8 and len(r.T) == 12
    assert sorted(set(r.fid.tolist())) == [1, 2, 3, 4, 5, 6] and same_surface(m, r)
    assert r.volume() == pytest.approx(1.0, rel=1e-12)


def test_binary_starting_with_solid_is_binary(tmp_path):
    """the header of a binary file may begin with 'solid' (several writers do): size decides."""
    m = box()
    p = str(tmp_path / "b.stl")
    gio.write_stl(p, m)
    data = bytearray(open(p, "rb").read())
    data[:80] = b"solid exported by some CAD".ljust(80)
    open(p, "wb").write(bytes(data))
    r, info = gio.read_stl(p, faceid="attr")
    assert info["format"] == "binary" and len(r.T) == 12 and set(r.fid.tolist()) == set(range(1, 7))
    r, _ = gio.read_stl(p)               # foreign header: attribute words are not trusted as ids
    assert set(r.fid.tolist()) == {0}


def test_ascii_blank_lines_and_upper_case(tmp_path):
    """assimp writes a blank line between facets; some writers use upper case.  The 26988-facet model that came back
    as one triangle in materializr's OCCT reader is the case being guarded."""
    p = str(tmp_path / "a.stl")
    gio.write_stl(p, box(), binary=False)
    txt = open(p).read().replace("endfacet\n", "ENDFACET\n\n\n").replace("vertex", "VERTEX")
    open(p, "w").write(txt)
    r, info = gio.read_stl(p)
    assert info["format"] == "ascii" and info["facets"] == 12 and info["watertight"]


def test_truncated_files_raise(tmp_path):
    p = str(tmp_path / "t.stl")
    gio.write_stl(p, box())
    data = open(p, "rb").read()
    open(p, "wb").write(data[:-75])          # 1.5 facets short
    with pytest.raises(gio.StlError, match="truncated"):
        gio.read_stl(p)
    r, _ = gio.read_stl(p, strict=False, repair=False)
    assert len(r.T) == 10
    gio.write_stl(p, box(), binary=False)
    txt = open(p).read()
    open(p, "w").write(txt[:txt.rfind("vertex")])
    with pytest.raises(gio.StlError):
        gio.read_stl(p)


def test_face_id_range(tmp_path):
    m = box()
    m.fid[0] = 70000
    with pytest.raises(gio.StlError, match="16-bit"):
        gio.write_stl(str(tmp_path / "x.stl"), m)


# ---------------------------------------------------------------- repair
def test_repair_flipped_and_inside_out():
    rng = np.random.default_rng(0)
    m = soup(box())
    flip = rng.random(len(m.T)) < 0.5
    m.T[flip] = m.T[flip][:, [0, 2, 1]]
    r, info = gio.repair(m)
    assert info["watertight"] and info["components"] == 1 and r.volume() == pytest.approx(1.0, rel=1e-12)
    inv = box()
    inv.T = inv.T[:, [0, 2, 1]]
    r, info = gio.repair(inv)
    assert info["inverted_components"] == 1 and r.volume() == pytest.approx(1.0, rel=1e-12)


def test_hollow_box_cavity_points_inward():
    """two shells: per-component nesting keeps the cavity wall reversed (a global flip rule would get this wrong
    whenever the outer shell is already right and the inner one is not)."""
    outer, inner = box((0, 0, 0), (4, 4, 4)), box((1, 1, 1), (2, 2, 2))
    inner.fid += 6
    # both shells wound outward: the cavity wall is wrong until repair reverses it
    m = gio.TriMesh(np.vstack([outer.V, inner.V]), np.vstack([outer.T, inner.T + 8]), np.r_[outer.fid, inner.fid])
    r, info = gio.repair(m)
    assert info["components"] == 2 and info["watertight"]
    assert r.volume() == pytest.approx(64 - 1, rel=1e-12)
    w = gio.winding_number(r.V, r.T, np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [5, 5, 5]]))
    assert np.allclose(w, [1, 0, 0], atol=1e-9)


def test_weld_tolerance_and_fold_guard():
    """vertices off by float32 round-off weld; two shells 1e-9 apart (touching bodies) must not fuse into
    edges with four triangles."""
    m = soup(box())
    m.V += np.random.default_rng(1).normal(0, 1e-9, m.V.shape)
    r, info = gio.repair(m, weld_tol=1e-7)
    assert info["watertight"] and len(r.V) == 8
    a, b = box((0, 0, 0), (1, 1, 1)), box((1 + 1e-9, 0, 0), (2, 1, 1))
    two = gio.TriMesh(np.vstack([a.V, b.V]), np.vstack([a.T, b.T + 8]), np.r_[a.fid, b.fid + 6])
    w = gio.weld(two, tol=1e-6)
    rep = gio.edge_report(w)
    assert rep["nonmanifold"] == 0


def test_open_mesh_reported():
    m = box()
    m = gio.TriMesh(m.V, m.T[:-2], m.fid[:-2])          # one side missing
    r, info = gio.repair(m)
    assert not info["watertight"] and info["boundary"] == 4


def test_segment_faces():
    s = gio.segment_faces(box(), 30.0)
    assert len(s.faces()) == 6
    c = gio.segment_faces(cylinder(), 30.0)                                  # 7.5 deg facets: wall is one face
    assert len(c.faces()) == 3
    assert c.fid[0] == 1                                                     # largest face (the wall) first
    assert len(gio.segment_faces(cylinder(), 5.0).faces()) == 48 + 2


# ---------------------------------------------------------------- 3MF
def test_3mf_roundtrip(tmp_path):
    m = cylinder()
    p = str(tmp_path / "c.3mf")
    gio.write_3mf(p, m)
    r = gio.read_3mf(p)
    assert same_surface(m, r) and np.array_equal(r.T, m.T)
    r2, info = gio.load(p)
    assert info["watertight"] and r2.volume() == pytest.approx(m.volume(), rel=1e-7)


# ---------------------------------------------------------------- CAD
def _gmsh():
    return pytest.importorskip("gmsh")


@pytest.mark.parametrize("name,h,tol", [("bracket.step", None, 2e-3), ("occt_screw.step", None, 2e-2)])
def test_step_watertight_face_tagged(tmp_path, name, h, tol):
    gmsh = _gmsh()
    path = os.path.join(STEP, name)
    t0 = time.perf_counter()
    m, info = gio.cad_to_mesh(path, h=h, curvature=16)
    dt = time.perf_counter() - t0
    assert info["watertight"] and info["components"] == info["volumes"] == 1
    assert info["vol_rel_err"] < tol and m.volume() > 0
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(path); gmsh.model.occ.synchronize()
    tags = sorted(t for _, t in gmsh.model.getEntities(2))
    gmsh.finalize()
    assert sorted(set(m.fid.tolist())) == tags                    # every CAD face, numbered as step_auto sees it
    assert dt < 30                                                # bolt-sized parts: well inside the 1 min budget
    p = str(tmp_path / "part.stl")
    gio.write_stl(p, m)
    r, ri = gio.read_stl(p)
    assert ri["watertight"] and np.array_equal(np.bincount(r.fid), np.bincount(m.fid))


def test_step_through_cut_cells():
    """STEP -> mesh -> cut3d: cut-cell volumes sum to the mesh volume (exact for a polyhedron), every surface point
    carries a CAD face id."""
    _gmsh()
    pytest.importorskip("numba")
    pytest.importorskip("scipy")
    sys.path.insert(0, os.path.join(HERE, "..", "immersed"))
    import cut_quadrature as cq
    m, info = gio.cad_to_mesh(os.path.join(STEP, "bracket.step"), curvature=12)
    lo, hi = m.V.min(0), m.V.max(0)
    h = (hi - lo).max() / 24
    n = np.ceil((hi - lo) / h).astype(int) + 3
    g = cq.cut_cells(m.V, m.T, lo - 1.37 * h, h, n, fid=m.fid)
    assert g["check"]["vol_sum"] == pytest.approx(m.volume(), rel=1e-10)
    s = cq.surface_rule(g)
    assert s["orphan"] == 0 and set(np.unique(s["face"]).tolist()) == set(m.fid.tolist())
    for f in np.unique(m.fid)[:5]:                                # per-face area: surface rule vs. mesh
        assert s["W"][s["face"] == f].sum() == pytest.approx(m.areas()[m.fid == f].sum(), rel=1e-10)
