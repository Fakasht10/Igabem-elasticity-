"""STL intake: formats, rejection rules, exact coordinates, provenance, units."""
import struct

import numpy as np
import pytest

import fixtures as fx
from stl_repair_core import stlio
from stl_repair_core.mesh import canonical, from_soup, geometry_hash


def _bin(path, tris, count=None, attr=0, header=b"t"):
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris) if count is None else count))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            for p in t:
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", attr))


def test_binary_and_ascii_same_geometry(tmp_path):
    V, F = fx.box(n=2)
    fx.soup_write(tmp_path / "b.stl", V, F, "binary")
    fx.soup_write(tmp_path / "a.stl", V, F, "ascii")
    sb, sa = stlio.read_stl(tmp_path / "b.stl"), stlio.read_stl(tmp_path / "a.stl")
    assert sb.fmt == "binary" and sa.fmt == "ascii"
    assert np.array_equal(sb.tri, sa.tri)
    assert geometry_hash(canonical(from_soup(sb))) == geometry_hash(canonical(from_soup(sa)))


def test_coordinates_bit_exact_and_not_rescaled(tmp_path):
    V, F = fx.box((0, 0, 0), (0.1, 0.2, 0.3))     # would look like metres/inches to a heuristic
    fx.soup_write(tmp_path / "s.stl", V, F)
    s = stlio.read_stl(tmp_path / "s.stl")
    assert np.array_equal(s.tri, V[F])
    u = stlio.detect_units(s.header, s.tri)
    assert u["rescaled"] is False and u["heuristic"]


def test_ascii_float64_kept(tmp_path):
    p = tmp_path / "a.stl"
    p.write_text("solid x\nfacet normal 0 0 1\nouter loop\nvertex 0.1 0.2 0.3\nvertex 1 0 0\n"
                 "vertex 0 1 0\nendloop\nendfacet\nendsolid x\n")
    s = stlio.read_stl(p)
    assert s.tri[0, 0, 0] == 0.1 and s.tri[0, 0, 2] == 0.3   # float64, not float32-rounded


def test_units_from_header(tmp_path):
    V, F = fx.box()
    stlio.write_stl(tmp_path / "u.stl", V, F, header=b"exported UNITS=inch")
    s = stlio.read_stl(tmp_path / "u.stl")
    assert stlio.detect_units(s.header, s.tri)["declared"] == "inch"


@pytest.mark.parametrize("case", ["truncated", "count_too_large", "trailing", "nan", "inf", "short", "garbage"])
def test_rejections(tmp_path, case):
    V, F = fx.box()
    tris = V[F].tolist()
    p = tmp_path / ("%s.stl" % case)
    if case == "truncated":
        _bin(p, tris)
        p.write_bytes(p.read_bytes()[:300])
    elif case == "count_too_large":
        _bin(p, tris, count=1000)
    elif case == "trailing":
        _bin(p, tris)
        p.write_bytes(p.read_bytes() + b"\0" * 17)
    elif case == "nan":
        tris[3][1][0] = float("nan")
        _bin(p, tris)
    elif case == "inf":
        tris[3][1][0] = float("inf")
        _bin(p, tris)
    elif case == "short":
        p.write_bytes(b"\0" * 20)
    elif case == "garbage":
        p.write_text("v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nf 1 2 3\n")
    with pytest.raises(stlio.StlError):
        stlio.read_stl(p)


@pytest.mark.parametrize("text", [
    "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nendloop\nendfacet\nendsolid x\n",
    "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nvertex 1 1 0\nendloop\nendfacet\nendsolid\n",
    "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 nan\nendloop\nendfacet\nendsolid\n",
    "solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\n",
])
def test_ascii_rejections(tmp_path, text):
    p = tmp_path / "bad.stl"
    p.write_text(text)
    with pytest.raises(stlio.StlError):
        stlio.read_stl(p)


def test_safety_limits(tmp_path):
    V, F = fx.box(n=4)
    fx.soup_write(tmp_path / "s.stl", V, F)
    with pytest.raises(stlio.StlError):
        stlio.read_stl(tmp_path / "s.stl", max_bytes=1000)
    with pytest.raises(stlio.StlError):
        stlio.read_stl(tmp_path / "s.stl", max_triangles=10)
    with pytest.raises(stlio.StlError):
        stlio.read_stl(tmp_path / "s.stl", max_abs_coord=0.5)


def test_provenance_and_attributes(tmp_path):
    V, F = fx.box(n=2)
    attr = np.arange(len(F), dtype=np.uint16) + 100
    stlio.write_stl(tmp_path / "s.stl", V, F, attr=attr)
    s = stlio.read_stl(tmp_path / "s.stl")
    m = canonical(from_soup(s))
    assert sorted(m.src.tolist()) == list(range(len(F)))
    assert np.array_equal(m.attr, attr[m.src])          # attribute travels with its facet
    P, Q = m.V[m.F], s.tri[m.src]                        # every face equals its input facet (up to rotation)
    assert all(any(np.array_equal(np.roll(P[i], r, axis=0), Q[i]) for r in range(3)) for i in range(len(P)))


def test_negative_zero_folded(tmp_path):
    V, F = fx.box()
    V = V - 0.0
    V[V == 0] = -0.0
    fx.soup_write(tmp_path / "z.stl", V, F)
    m = from_soup(stlio.read_stl(tmp_path / "z.stl"))
    assert len(m.V) == 8 and not np.signbit(m.V).any()


def test_ascii_multi_solid_regions(tmp_path):
    V, F = fx.box()
    solid = np.repeat([0, 1], [6, 6])
    stlio.write_stl(tmp_path / "m.stl", V, F, fmt="ascii", solid_of_face=solid, solid_names=["walls", "lid"])
    s = stlio.read_stl(tmp_path / "m.stl")
    assert s.solid_names == ["walls", "lid"] and np.array_equal(s.solid, solid)
