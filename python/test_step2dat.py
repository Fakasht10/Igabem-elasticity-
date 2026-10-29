"""Regression test of the STEP -> .dat pipeline: the plate with a hole written as STEP by
OpenCASCADE, split by the diagonal planes and untrimmed, must give 16 conforming patches
that enclose the CAD volume to round-off and match make_geometries.plate_hole.

    python3 -m pytest python/test_step2dat.py -q     (needs cadquery-ocp; ~3 s)
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("OCP")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_geometries as mg  # noqa: E402
import step2dat as sd  # noqa: E402
from make_plate_step import plate_hole_solid  # noqa: E402

DIAGONALS = [((1, -1, 0), (0, 0, 0)), ((1, 1, 0), (0, 0, 0))]


def test_plate_hole_roundtrip(tmp_path):
    step = str(tmp_path / "plate.step")
    sd.write_step(plate_hole_solid(0.2), step)
    solid = sd.read_step(step)
    boundary = sd.split(solid, DIAGONALS)
    assert sd.face_census(solid) == (7, 5)
    assert sd.face_census(boundary) == (16, 16)
    r = sd.convert(solid, boundary, str(tmp_path / "plate.dat"))
    assert r["patches"] == 16 and r["shared_edges"] == 32 and r["open_edges"] == 0
    assert r["conformity_gap"] < 1e-10 and r["max_dev"] < 1e-10 and r["vol_rel_err"] < 1e-12
    # same surface as the code-built plate: every patch matches one of make_geometries'
    # up to orientation-preserving reparametrisation, checked through corner points and area
    ref = mg.plate_hole(0.2)
    got = mg.read_dat(str(tmp_path / "plate.dat"))

    def key(P):
        X = P.euclid()
        c = sorted(tuple(np.round(x, 9) + 0.0) for x in (X[0, 0], X[0, -1], X[-1, 0], X[-1, -1]))
        x, xu, xv = sd.evaluate(P, 0.5, 0.5)
        return tuple(c), tuple(0.0 + np.round(np.cross(xu, xv) / np.linalg.norm(np.cross(xu, xv)), 9))

    assert sorted(map(key, got)) == sorted(map(key, ref))


def test_refuses_non_quad_faces(tmp_path):
    with pytest.raises(ValueError):
        sd.to_patches(plate_hole_solid(0.2))
