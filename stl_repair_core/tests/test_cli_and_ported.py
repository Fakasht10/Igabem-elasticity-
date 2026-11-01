"""CLI contract and Sutura 0.9.8 tests ported to this package.

Ported (adapted to the conservative contract, see docs/MAPPING.md):
  tests/test_adversarial.py     malformed inputs exit non-zero with an error
  tests/make_broken_stl.py      the broken-cube smoke mesh
  tests/test_si_definition.py   SI fixture: proper crossing vs coplanar channel
  tests/test_pinched_vertices.py bowtie diagnosis (STL cannot carry the split)
  tests/test_order_independence.py  NM-region / topology order independence
"""
import json
import struct
import subprocess
import sys

import numpy as np
import pytest

import fixtures as fx
from stl_repair_core import intersect as X
from stl_repair_core import repair_stl, validate_stl

CUBE = [
    ((-1, -1, -1), (1, -1, -1), (1, 1, -1)), ((-1, -1, -1), (1, 1, -1), (-1, 1, -1)),
    ((-1, -1, 1), (1, -1, 1), (1, 1, 1)), ((-1, -1, 1), (1, 1, 1), (-1, 1, 1)),
    ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1)), ((-1, -1, -1), (-1, 1, 1), (-1, 1, -1)),
    ((1, -1, -1), (1, -1, 1), (1, 1, 1)), ((1, -1, -1), (1, 1, 1), (1, 1, -1)),
    ((-1, 1, -1), (1, 1, -1), (1, 1, 1)), ((-1, 1, -1), (1, 1, 1), (-1, 1, 1)),
    ((-1, -1, -1), (1, -1, -1), (1, -1, 1)), ((-1, -1, -1), (1, -1, 1), (-1, -1, 1)),
]


def write_stl(path, tris, count=None, header=b"adversarial"):
    with open(path, "wb") as f:
        f.write(header.ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris) if count is None else count))
        for a, b, c in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            f.write(struct.pack("<3f", *a))
            f.write(struct.pack("<3f", *b))
            f.write(struct.pack("<3f", *c))
            f.write(struct.pack("<H", 0))


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "stl_repair_core.cli", *map(str, args)],
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout


def adversarial(tmp):
    cases = {}
    p = tmp / "truncated.stl"
    write_stl(p, CUBE)
    p.write_bytes(p.read_bytes()[:len(p.read_bytes()) // 2])
    cases["truncated"] = p
    p = tmp / "wrongcount.stl"
    write_stl(p, CUBE, count=1000)
    cases["wrongcount"] = p
    p = tmp / "open_square.stl"
    write_stl(p, [((0, 0, 0), (1, 0, 0), (1, 1, 0)), ((0, 0, 0), (1, 1, 0), (0, 1, 0))])
    cases["open_square"] = p
    p = tmp / "all_zero_area.stl"
    write_stl(p, [((0, 0, 0), (1, 0, 0), (2, 0, 0)), ((0, 0, 0), (0, 1, 0), (0, 2, 0))])
    cases["all_zero_area"] = p
    p = tmp / "nan_inf.stl"
    write_stl(p, [((-1, -1, -1), (float("nan"), -1, -1), (1, 1, -1)),
                  ((1, -1, -1), (1, -1, 1), (float("inf"),) * 3)] + CUBE[2:])
    cases["nan_inf"] = p
    p = tmp / "empty.stl"
    write_stl(p, [])
    cases["empty"] = p
    p = tmp / "obj_as_stl.stl"
    p.write_text("v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nf 1 2 3\n")
    cases["wrong_ext"] = p
    return cases


@pytest.mark.parametrize("case", ["truncated", "wrongcount", "open_square", "all_zero_area", "nan_inf", "empty", "wrong_ext"])
def test_adversarial_rejected(tmp_path, case):
    p = adversarial(tmp_path)[case]
    js = tmp_path / "r.json"
    code, _ = cli("repair", p, "-o", tmp_path / "o.stl", "--json", js, "--max-hole-diameter", "0.1",
                  "--require-watertight")
    rep = json.loads(js.read_text())
    assert code == 1 and rep["status"] == "FAIL"
    assert rep.get("error") or rep.get("problems")
    if case in ("truncated", "wrongcount", "nan_inf", "wrong_ext", "empty", "all_zero_area"):
        assert rep["output_written"] is False


def broken_cube(path):
    """Sutura tests/make_broken_stl.py: hole, inverted face, duplicate, fin, crossing triangle."""
    Vd = {"lb": (-1, -1, -1), "rb": (1, -1, -1), "rt": (1, 1, -1), "lt": (-1, 1, -1),
          "lT": (-1, -1, 1), "rT": (1, -1, 1), "RT": (1, 1, 1), "LT": (-1, 1, 1)}
    t = lambda a, b, c: (Vd[a], Vd[b], Vd[c])  # noqa: E731
    faces = [t("lb", "rb", "rt"), t("lb", "rt", "lt"), t("lT", "LT", "RT"), t("lT", "RT", "rT"),
             t("lb", "lT", "rT"), t("lb", "rT", "rb"), t("rb", "rT", "RT"), t("rb", "RT", "rt"),
             t("rt", "RT", "LT"), t("rt", "LT", "lt"), t("lt", "LT", "lT"), t("lt", "lT", "lb")]
    broken = faces[:2] + faces[4:]
    broken[2] = (broken[2][2], broken[2][1], broken[2][0])
    broken += [broken[4]]
    broken += [((0, 0, 0), (1, -1, 1), (-1, -1, 1))]
    broken += [((2, 0, 0), (0, 2, 0), (0, 0, 2))]
    write_stl(path, broken, header=b"broken mesh for repair test")


def test_broken_cube_smoke(tmp_path):
    p = tmp_path / "broken.stl"
    broken_cube(p)
    v = validate_stl(p)
    d = v["diagnosis"]
    assert v["status"] == "FAIL"
    assert d["boundary_edges"] > 0 and d["duplicate_faces"] == 1
    assert d["inconsistent_orientation_edges"] > 0 and d["self_intersections"]["proper"] > 0
    r = repair_stl(p, tmp_path / "o.stl", max_hole_diameter=3.0, max_surface_deviation=2.0)
    # the crossing triangle cannot be removed conservatively -> original returned
    assert r["status"] == "FAIL"
    adopted = [o["operation"] for o in r["adopted_operations"]]
    assert "remove_duplicates" in adopted
    assert r["after"] == r["before"]


def test_si_definition_fixture():
    v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 0.5, -1.0], [1.0, 0.5, 1.0],
                  [0.5, 1.0, 1.0], [1.4, 0.3, 0.0], [1.6, 0.3, 0.0], [1.4, 0.5, 0.0]])
    t = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    si = X.self_intersections(v, t)
    proper_faces = sorted(set(np.unique(si["proper"]).tolist()))
    assert proper_faces == [0, 1]                       # Sutura: exact mask [T, T, F]
    assert si["coplanar_overlap"].tolist() == [[0, 2]]  # coplanar kept out of the SI channel


def test_bowtie_diagnosed_not_split(tmp_path):
    TET = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    BV = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    BT = TET + [(0, 5, 4), (0, 4, 6), (0, 6, 5), (4, 5, 6)]
    V, F = fx.f32(BV), np.asarray(BT)
    p = fx.soup_write(tmp_path / "bowtie.stl", V, F)
    v = validate_stl(p)
    assert v["diagnosis"]["nonmanifold_vertices"] == 1 and v["diagnosis"]["nonmanifold_edges"] == 0
    r = repair_stl(p, tmp_path / "o.stl")
    # Sutura splits the vertex in memory; an STL re-welds it on reload, so the
    # conservative contract reports FAIL instead of claiming a manifold result
    assert r["status"] == "FAIL" and "nonmanifold_vertices=1" in r["problems"]


def test_order_independent_diagnosis(tmp_path):
    from stl_repair_core.diagnose import diagnose
    from stl_repair_core.mesh import Mesh, canonical
    V, F = fx.merge(fx.two_cubes_sharing_vertex(), fx.box((5, 5, 5), (6, 6, 6)))
    keys = ("nonmanifold_vertices", "nonmanifold_edges", "components", "boundary_edges", "signed_volume")
    base = None
    for seed in range(5):
        V2, F2, _ = fx.random_permutation(V, F, seed)
        d = diagnose(canonical(Mesh(V2, F2, np.arange(len(F2)), np.zeros(len(F2), np.uint16), np.zeros(len(F2), np.int32))))
        got = tuple(d[k] for k in keys)
        base = base or got
        assert got == base


def test_cli_validate_and_repair(tmp_path):
    V, F = fx.cube_missing_triangle()
    p = fx.soup_write(tmp_path / "m.stl", V, F)
    code, out = cli("validate", p, "--json", tmp_path / "v.json")
    v = json.loads((tmp_path / "v.json").read_text())
    assert code == 1 and v["status"] == "FAIL"
    code, out = cli("repair", p, "-o", tmp_path / "r.stl", "--json", tmp_path / "r.json", "--max-hole-diameter", "1.5")
    r = json.loads((tmp_path / "r.json").read_text())
    assert code == 0 and r["status"] == "PASS"
    for key in ("before", "after", "adopted_operations", "rejected_operations", "thresholds", "input", "output",
                "runtime_s", "policy_hash", "provenance", "report_hash", "status"):
        assert key in r, key
    assert r["input"]["sha256"] and r["output"]["sha256"] and r["input"]["sha256"] != r["output"]["sha256"]
    code, _ = cli("validate", tmp_path / "r.stl", "--json", tmp_path / "v2.json")
    assert code == 0 and json.loads((tmp_path / "v2.json").read_text())["status"] == "PASS"
