"""Unit tests of python/rve: bead cells (watertight, conforming, periodic copies, porosity), conforming graded
refinement, stiffness rotation, notch eigenvalues, and (with the rve_cell binary: IGABEM_RVE_CELL or build/) the
bond-tip tractions against exact equilibrium. The C++ homogenization has its own ctests (rve_*)."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve.beads import cell_00, cell_090, section_porosity  # noqa: E402
from rve.homog import engineering, isotropic_projection, rotate_z  # noqa: E402
from spline_checks import conformity_gap, spline_volume  # noqa: E402

G = dict(w=0.6, h=0.3, b_int=0.36, b_lat=0.10)


def _faces(P, axis, c, tol=1e-12):
    out = []
    for Q in P:
        X = Q.Pw[..., :3] / Q.Pw[..., 3:4]
        if np.all(np.abs(X[..., axis] - c) < tol):
            out.append(np.round(np.delete(X.reshape(-1, 3), axis, 1), 10))
    return sorted(map(lambda a: sorted(map(tuple, a)), out))


class TestBeads(unittest.TestCase):
    def check_cell(self, P, box):
        gap, shared, lonely = conformity_gap(P)
        self.assertLess(gap, 1e-12)
        self.assertEqual(lonely, 0)
        for a in range(3):
            self.assertEqual(_faces(P, a, 0.0), _faces(P, a, box[a]), "face %d not a translated copy" % a)
        return spline_volume(P)

    def test_00(self):
        for alpha in (0.0, 15.0):
            P = cell_00(**G, alpha=alpha, refine=1, grade=1)
            V = self.check_cell(P, (G["h"], G["w"], G["h"]))
            phi = section_porosity(*G.values(), alpha, math.sqrt(0.5))
            self.assertAlmostEqual(1 - V / (G["h"] * G["w"] * G["h"]), phi, places=5)

    def test_090(self):
        P = cell_090(**G, alpha=15.0, refine=0, grade=1, b_lat_b=0.08)
        V = self.check_cell(P, (G["w"], G["w"], 2 * G["h"]))
        self.assertGreater(V, 0)
        P = cell_090(**G, alpha=15.0, refine=0)
        phi = section_porosity(*G.values(), 15.0, math.sqrt(0.5))
        self.assertAlmostEqual(1 - spline_volume(P) / (2 * G["w"] ** 2 * G["h"]), phi, places=5)

    def test_notch_angle_limit(self):
        with self.assertRaises(ValueError):
            cell_00(**G, alpha=80.0, refine=0)


class TestHomog(unittest.TestCase):
    def test_rotation(self):
        E, nu = 2200.0, 0.35
        lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[range(3), range(3)] += 2 * mu
        C[range(3, 6), range(3, 6)] = mu
        self.assertLess(np.abs(rotate_z(C, 33.0) - C).max(), 1e-9)
        self.assertAlmostEqual(isotropic_projection(C)[2], E, places=8)
        # tetragonal C: 90 deg rotation swaps 1 and 2, 45 deg mixes G12 into E1
        T = C.copy()
        T[2, 2] *= 0.6
        T[3, 3] = T[4, 4] = 0.7 * mu
        R = rotate_z(T, 90.0)
        self.assertLess(np.abs(R - T).max(), 1e-9)
        e0, e45 = engineering(T), engineering(rotate_z(T, 45.0))
        self.assertAlmostEqual(e45["E3"], e0["E3"], places=8)
        self.assertAlmostEqual(e45["G13"], e0["G13"], places=8)


class TestStrength(unittest.TestCase):
    def test_williams(self):
        from rve.strength import williams
        for v in williams(0.0):
            self.assertAlmostEqual(v, 0.5, places=8)            # crack
        l1, l2, l3 = williams(45.0)                            # 270 deg solid: 0.5445, 0.9085 (Williams 1952)
        self.assertAlmostEqual(l1, 0.544484, places=5)
        self.assertAlmostEqual(l2, 0.908529, places=5)
        self.assertAlmostEqual(l3, 2.0 / 3.0, places=10)

    def test_ligament_equilibrium(self):
        """0/0 cell: the tip-corrected mean traction over each whole bond equals the net-section value."""
        import tempfile
        from rve import strength as st
        if not os.path.exists(st.RVE_CELL):
            self.skipTest("rve_cell not built")
        with tempfile.TemporaryDirectory() as d:
            fam = st.main(["00", os.path.join(d, "s.json"), "--work", d, "--check-only"])
        self.assertLess(abs(fam["lat"]["check"] - 1), 0.03)
        self.assertLess(abs(fam["int"]["check"] - 1), 0.01)


@unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid.json")),
                     "no precomputed grid")
class TestApi(unittest.TestCase):
    """rve.api against the cells it interpolates, its symmetries, and the page's JavaScript engine."""

    def test_node_exact(self):
        """At a grid node the card returns the computed cell (0/0, nu on the grid)."""
        import json
        from rve import api
        g = api._grid()[0]
        i, j, k = 2, 1, 1
        h, bi, be = g["axes"]["h"][i], g["axes"]["b_int"][j], g["axes"]["beta"][k]
        r = api.printed_card({"E_MPa": 1000.0, "nu": 0.35, "sigma_uts_MPa": 40.0}, layup="aligned", w=1.0, h=h,
                             b_int=bi, b_lat=be * h)
        C = 1000.0 * np.array(g["c00"]["0.35"][i][j][k])
        self.assertLess(np.abs(np.array(r["C_ply_MPa"]) - C).max() / np.abs(C).max(), 1e-12)
        self.assertAlmostEqual(r["porosity"], g["porosity"][i][j][k], places=4)
        json.dumps(r)                                          # JSON-ready for the MCP tool

    def test_cross_ply_symmetry_and_strength(self):
        from rve import api
        m = {"E_MPa": 2200.0, "nu": 0.35, "sigma_uts_MPa": 30.0}
        a = api.printed_card(m, layup="cross", raster_deg=0, w=0.6, h=0.3, b_int=0.36, b_lat=0.1,
                             weld_strength_MPa=15.26)
        b = api.printed_card(m, layup="cross", raster_deg=90, w=0.6, h=0.3, b_int=0.36, b_lat=0.1,
                             weld_strength_MPa=15.26)
        self.assertAlmostEqual(a["engineering"]["E1"], a["engineering"]["E2"], delta=1e-9 * a["engineering"]["E1"])
        self.assertAlmostEqual(a["engineering"]["E1"], b["engineering"]["E1"], delta=1e-9 * a["engineering"]["E1"])
        self.assertAlmostEqual(a["strength"]["Z"], 5.28, places=2)            # the FFM run (README)
        al = api.printed_card(m, layup="aligned", w=0.6, h=0.3, b_int=0.36, b_lat=0.1, weld_strength_MPa=15.26)
        self.assertAlmostEqual(al["strength"]["Z"], 8.98, places=2)
        with self.assertRaises(ValueError):
            api.printed_card(m, layup="cross", w=0.6, h=0.6, b_int=0.36, b_lat=0.1)   # h/w = 1 outside

    def test_from_porosity(self):
        from rve import api
        m = "ABS-M30-iso"
        r = api.printed_card(m, layup="aligned", w=1.0, h=0.5, b_int=0.6, b_lat=0.15)
        q = api.printed_card(m, layup="aligned", w=1.0, h=0.5, b_lat=0.15, porosity=r["porosity"],
                             bond="from_porosity")
        self.assertAlmostEqual(q["geometry"]["b_int"], 0.6, places=3)
        self.assertTrue(any("porosity" in f for f in q["flags"]))

    def test_page_engine(self):
        """The interactive page's JavaScript engine gives the same card as rve.api."""
        import json
        import shutil
        import subprocess
        import tempfile
        from rve import api, card_page
        if shutil.which("node") is None:
            self.skipTest("node not installed")
        with open(card_page.TEMPLATE) as f:
            html = f.read()
        js = html.split("<script>")[1].split("// ---------------------------------------------------------------- UI")[0]
        js = js.replace("/*__DATA__*/null", json.dumps(card_page.payload()))
        cases = [("cross", 30.0, 0.45, 0.2, 0.27, 0.06), ("aligned", -20.0, 0.5, 0.3, 0.2, 0.1)]
        js += "\nconst out=[];\n"
        for lay, th, w, h, bi, bl in cases:
            js += ("{const r=compute({mat:'PA12',layup:'%s',raster:%g,w:%g,h:%g,bi:%g,bl:%g,eta:0.7,gc:0});"
                   "out.push({C:r.C,phi:r.phi,Z:r.st.Z});}\n" % (lay, th, w, h, bi, bl))
        js += "console.log(JSON.stringify(out));\n"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
        got = json.loads(subprocess.run(["node", f.name], capture_output=True, text=True, check=True).stdout)
        os.remove(f.name)
        for (lay, th, w, h, bi, bl), g in zip(cases, got):
            r = api.printed_card("PA12", layup=lay, raster_deg=th, w=w, h=h, b_int=bi, b_lat=bl, weld_efficiency=0.7)
            C = np.array(r["C_MPa"])
            self.assertLess(np.abs(np.array(g["C"]) - C).max() / np.abs(C).max(), 2e-5)   # page data: 6 digits
            self.assertAlmostEqual(g["Z"], r["strength"]["Z"], places=6)
            self.assertAlmostEqual(g["phi"], r["porosity"], delta=2e-3)                 # cubic vs trilinear


if __name__ == "__main__":
    unittest.main()


class TestBeadCell(unittest.TestCase):
    """Bead-boundary cells (app/bead_cell, src/bead_cell.hpp): one region per bead, welds as spring laws."""

    @classmethod
    def setUpClass(cls):
        from rve import welds
        from rve.strength import RVE_CELL
        cls.rve_cell, cls.bead_cell = RVE_CELL, welds.BEAD_CELL
        if not (os.path.exists(RVE_CELL) and os.path.exists(welds.BEAD_CELL)):
            raise unittest.SkipTest("rve_cell / bead_cell not built")

    def _run(self, exe, args, d, name):
        import json
        import subprocess
        out = os.path.join(d, name + ".json")
        subprocess.run([exe] + args + ["--json", out], check=True, stdout=subprocess.DEVNULL)
        with open(out) as f:
            r = json.load(f)
        return np.array(r["bc"]["pbc"]["C"] if "bc" in r else r["C"]), r

    def test_00_matches_single_region_and_series_springs(self):
        """Perfect welds: the C of rve_cell to discretization error (5e-3: the collocation PBC system is consistent
        only to discretization error, and the condensed system borders the translations on the weld dofs only, so the
        two gauges differ by 2e-3 here at grade 1, less on finer meshes); spring welds: E2, E3 within 3 % of the
        series-spring estimate."""
        import tempfile
        from rve.beads import bead_00
        from make_geometries import write_dat
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "b.dat")
            write_dat(f, bead_00(G["w"], G["h"], G["b_int"], G["b_lat"], refine=1, grade=1), "test")
            C1, _ = self._run(self.rve_cell, [f, "--bc", "pbc", "--ng", "12"], d, "single")
            C2, r = self._run(self.bead_cell, [f, "--ng", "12", "--axes", "x"], d, "bead")
            self.assertLess(np.abs(C1 - C2).max() / np.abs(C1).max(), 5e-3)
            E0 = r["eng"]
            K = 10 * 2200 / G["w"]
            law = "%g:%g" % (K, K)
            _, s = self._run(self.bead_cell, [f, "--ng", "12", "--axes", "x", "--weld", "y:" + law, "--weld", "z:" + law],
                             d, "spring")
            self.assertLess(s["hill_mandel"], 1e-3)
            self.assertAlmostEqual(s["eng"]["E1"] / E0["E1"], 1.0, places=2)
            for name, extra in (("E2", G["h"] / (G["b_lat"] * G["w"])), ("E3", G["w"] / (G["b_int"] * G["h"]))):
                est = 1 / (1 / E0[name] + extra / K)
                self.assertLess(abs(s["eng"][name] / est - 1), 0.03, name)

    def test_090_rotated_operator(self):
        """0/90: bead B's operator rotated from A's equals B assembled, and C is x-y symmetric."""
        import tempfile
        from rve.beads import bead_090
        from make_geometries import write_dat
        with tempfile.TemporaryDirectory() as d:
            files = [os.path.join(d, t + ".dat") for t in "AB"]
            for f, P in zip(files, bead_090(G["w"], G["h"], G["b_int"], G["b_lat"], refine=1, grade=0)):
                write_dat(f, P, "test")
            Cr, r = self._run(self.bead_cell, files + ["--copy", "1:0:90", "--axes", "x,y"], d, "copy")
            Ca, _ = self._run(self.bead_cell, files + ["--axes", "x,y"], d, "asm")
            self.assertLess(np.abs(Cr - Ca).max() / np.abs(Ca).max(), 1e-6)
            self.assertAlmostEqual(Cr[0, 0] / Cr[1, 1], 1.0, places=6)
            self.assertAlmostEqual(Cr[4, 4] / Cr[3, 3], 1.0, places=3)  # 2e-4 at grade 0: discretization
            self.assertLess(r["hill_mandel"], 1e-8)

    def test_00_debond_matches_narrowed_bond(self):
        """A debond (weld patches unpaired, beads.py debonds= C0 cuts) is consistent (Hill-Mandel) and stiffens like
        the bond narrowed by the same depth from both tips (E2 within 1 %); a full debond leaves no E2."""
        import tempfile
        from rve.beads import bead_00
        from make_geometries import write_dat
        with tempfile.TemporaryDirectory() as d:
            f, n = os.path.join(d, "c.dat"), os.path.join(d, "n.dat")
            write_dat(f, bead_00(G["w"], G["h"], G["b_int"], G["b_lat"], refine=1, grade=1, debonds=(0.4,)), "test")
            write_dat(n, bead_00(G["w"], G["h"], G["b_int"], 0.6 * G["b_lat"], refine=1, grade=1), "test")
            lim = 0.6 * G["b_lat"] / 2
            _, c = self._run(self.bead_cell, [f, "--axes", "x", "--crack", "P:y:-1:z=0:%g" % lim], d, "crack")
            _, r = self._run(self.bead_cell, [n, "--axes", "x"], d, "narrow")
            self.assertLess(c["hill_mandel"], 1e-10)
            self.assertLess(abs(c["eng"]["E2"] / r["eng"]["E2"] - 1), 0.01)
            Cz, _ = self._run(self.bead_cell, [f, "--axes", "x", "--crack", "P:y:-1:z=0:0"], d, "full")
            self.assertLess(abs(Cz[1, 1]), 1e-6 * r["eng"]["E2"])   # no lateral weld: no sigma_22
