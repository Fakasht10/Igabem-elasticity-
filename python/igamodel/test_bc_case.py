"""bc_case features the tree writes. A support driven by a displacement field (the submodel's cut faces):
the field of uniform tension prescribed on every face of a cube gives the uniform stress inside, and a field
missing an interpolation point is refused. Named surfaces: a case written with names gives the same results
as the same case with the regions written out. Needs bc_case (IGABEM_BC_CASE, ctest model_bc_case); < 1 s.

usage: IGABEM_BC_CASE=<build>/bc_case python3 -m unittest igamodel.test_bc_case      (from python/)
"""
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
BC_CASE = os.environ.get("IGABEM_BC_CASE", str(ROOT / "build" / "bc_case"))
E, NU, S = 1000.0, 0.3, 2.0


@unittest.skipUnless(os.path.exists(BC_CASE), "bc_case not built")
class FieldSupport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        case = {"geometry": str(ROOT / "geo" / "cube_p2.dat"), "cache": "none", "material": {"E": E, "nu": NU},
                "supports": [{"label": "all", "patches": list(range(6)), "fix": ["ux", "uy", "uz"]}],
                "cases": [{"name": "t", "fields": {"all": "u.txt"}, "loads": []}]}
        (self.d / "case.json").write_text(json.dumps(case))
        self.run_bc("--greville", "all", "g.txt")
        X = np.loadtxt(self.d / "g.txt")
        self.rows = np.column_stack([X, S / E * X[:, 0], -NU * S / E * X[:, 1], -NU * S / E * X[:, 2]])

    def tearDown(self):
        self.tmp.cleanup()

    def run_bc(self, *args, check=True):
        return subprocess.run([BC_CASE, "case.json", *args], cwd=self.d, capture_output=True, text=True, check=check)

    def test_uniform_tension_is_reproduced_inside(self):
        np.savetxt(self.d / "u.txt", self.rows, fmt="%.17g")
        (self.d / "p.txt").write_text("0.5 0.5 0.5\n0.1 0.2 0.3\n0.9 0.5 0.1\n")
        self.run_bc("--points", "p.txt", "--points-out", "out")
        s = np.loadtxt(self.d / "out_t.txt", ndmin=2)[:, 3:9]
        np.testing.assert_allclose(s, np.tile([S, 0, 0, 0, 0, 0], (3, 1)), atol=5e-5 * S)

    def test_a_missing_point_is_refused(self):
        keep = ~np.all(np.isclose(self.rows[:, :3], [0.5, 0.5, 0.0]), axis=1)
        self.assertEqual(int((~keep).sum()), 1)
        np.savetxt(self.d / "u.txt", self.rows[keep], fmt="%.17g")
        r = self.run_bc(check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("has no value at (0.5, 0.5, 0)", r.stderr)


@unittest.skipUnless(os.path.exists(BC_CASE), "bc_case not built")
class NamedSurfaces(unittest.TestCase):
    def test_a_named_case_solves_as_the_written_out_one(self):
        from . import from_case, rename, write_case
        plain = {"geometry": str(ROOT / "geo" / "bar_L4_p3_n8.dat"), "cache": "none",
                 "material": {"E": 1000.0, "nu": 0.3},
                 "supports": [{"label": "end", "select": {"plane": {"point": [0, 0, 0], "normal": [1, 0, 0]}},
                               "fix": ["ux", "uy", "uz"]}],
                 "cases": [{"name": "pull", "loads": [{"label": "tip", "select": {"plane": {
                     "point": [4, 0, 0], "normal": [1, 0, 0]}}, "force_N": 10.0, "direction": [1, 0, 0]}]}]}
        named = json.loads(json.dumps(plain))
        named["names"] = {"root": {"select": plain["supports"][0]["select"]},
                          "tip": {"select": plain["cases"][0]["loads"][0]["select"]}}
        named["supports"][0] = {"label": "end", "name": "root", "fix": ["ux", "uy", "uz"]}
        named["cases"][0]["loads"][0] = {"label": "tip", "name": "tip", "force_N": 10.0, "direction": [1, 0, 0]}
        m = from_case(named)
        m.apply(rename(m, "root", "clamp"))
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(tmp)
            (d / "plain.json").write_text(json.dumps(plain))
            write_case(m, d / "named.json")
            out = [subprocess.run([BC_CASE, f"{k}.json", "--csv", f"{k}.csv"], cwd=d, capture_output=True,
                                  text=True, check=True) for k in ("plain", "named")]
            self.assertEqual((d / "plain.csv").read_bytes(), (d / "named.csv").read_bytes())
            self.assertIn("support end", out[1].stdout)


if __name__ == "__main__":
    unittest.main()
