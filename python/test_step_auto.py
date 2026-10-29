"""Regression tests of the automatic STEP -> .dat pipeline (python/step_auto.py), one per
geometry: every part must come out watertight (all patch sides matched exactly, none open),
outward-oriented (positive volume equal to the CAD volume) and close to its CAD faces.

    python3 -m pytest python/test_step_auto.py -q     (needs cadquery-ocp and gmsh; ~1 min)
"""
import os
import sys

import pytest

pytest.importorskip("OCP")
pytest.importorskip("gmsh")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step2dat as sd  # noqa: E402
import step_auto as sa  # noqa: E402
from make_bracket_step import bracket_solid  # noqa: E402
from make_plate_step import plate_hole_solid  # noqa: E402

SCREW = os.path.join(HERE, "..", "geo", "step", "occt_screw.step")


def _watertight(r, vol_tol, dev_tol):
    assert r["fallback_blocks"] == 0
    assert r["conformity_gap"] == 0.0 and r["open_edges"] == 0
    assert 2 * r["shared_edges"] == 4 * r["patches"]
    assert r["vol_rel_err"] < vol_tol and r["max_dev"] < dev_tol
    assert r["min_quad_quality"] > 0.1 and r["min_jacobian_ratio"] > 0.02


def test_plate_hole(tmp_path):
    step = str(tmp_path / "plate.step")
    sd.write_step(plate_hole_solid(0.2), step)
    r = sa.convert(step, str(tmp_path / "plate.dat"), h=0.4, degree=3)
    _watertight(r, 2e-5, 2e-4)


def test_bracket(tmp_path):
    step = str(tmp_path / "bracket.step")
    sd.write_step(bracket_solid(), step)
    r = sa.convert(step, str(tmp_path / "bracket.dat"), h=0.5, degree=3, curvature=6)
    _watertight(r, 2e-3, 1e-3)


def test_occt_screw(tmp_path):
    r = sa.convert(SCREW, str(tmp_path / "screw.dat"), h=6.0, degree=3)
    _watertight(r, 1e-4, 5e-3)  # part is ~40 mm long
