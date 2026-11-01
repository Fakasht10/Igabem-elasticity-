"""Results must not depend on facet order, corner rotation or vertex order."""
import numpy as np
import pytest

import fixtures as fx
from stl_repair_core import repair_stl

CASES = {
    "missing": (fx.cube_missing_triangle, {"max_hole_diameter": 1.5}),
    "dups": (fx.cube_duplicates, {}),
    "reversed": (fx.cube_reversed_face, {}),
    "cap": (fx.cube_zero_area_cap, {}),
    "debris": (fx.cube_with_debris, {"debris_area_threshold": 1e-3}),
    "multi": (fx.multi_components, {}),
    "overlap": (fx.overlapping_cubes, {}),
    "thin": (lambda: fx.thin_wall(1e-3), {"max_hole_diameter": 0.5}),
}


def _sig(rep):
    return (rep["status"], rep["output"]["geometry_hash"], [o["operation"] for o in rep["adopted_operations"]],
            {k: v for k, v in rep["after"].items() if k in ("faces", "boundary_loops", "components", "watertight")},
            rep["before"]["self_intersections"]["proper"], rep["before"]["self_intersections"]["coplanar_overlap"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_permutation_invariance(tmp_path, name):
    gen, kw = CASES[name]
    V, F = gen()
    p = fx.soup_write(tmp_path / "base.stl", V, F)
    base = _sig(repair_stl(p, tmp_path / "base_out.stl", **kw))
    for seed in range(4):
        V2, F2, _ = fx.random_permutation(V, F, seed)
        q = fx.soup_write(tmp_path / ("p%d.stl" % seed), V2, F2)
        got = _sig(repair_stl(q, tmp_path / ("p%d_out.stl" % seed), **kw))
        assert got == base, (seed, got, base)
