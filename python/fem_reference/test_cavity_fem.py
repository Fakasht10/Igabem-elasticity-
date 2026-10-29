"""Regression tests for the IGA-FEM reference (pytest, ~1 min).

  python -m pytest python/fem_reference -q
"""

import numpy as np
import pytest

import cavity_fem as cf
from octant_geometry import A, L, jacobian_range, octant_patches


def test_geometry_exact_and_valid():
    for vol in octant_patches():
        lo, hi = jacobian_range(vol)
        assert lo > 0 and hi > 0


@pytest.fixture(scope="module")
def mesh():
    return cf.Mesh(4, 2, 1.5)


def test_patch_test(mesh):
    row, _ = mesh.solve("patch")
    assert row["rel_L2_err"] < 1e-9
    assert abs(row["szz_eq"] - 2.5) < 1e-8


def test_goodier_and_tension_regression(mesh):
    g, _ = mesh.solve("goodier")
    t, _ = mesh.solve("tension")
    # values at p = 4, n = 2 (1,638 dofs); converged values in results/fem_cavity.csv
    assert abs(g["szz_eq"] - 2.03741864) < 1e-6
    assert abs(g["rel_L2_err"] - 2.423e-4) < 1e-6
    assert abs(t["szz_eq"] - 2.28822511) < 1e-6


def test_matches_jx_solve():
    assert abs(cf.check_jaxiga("tension", 4, 1, 1.5) - 2.31759465) < 1e-7
