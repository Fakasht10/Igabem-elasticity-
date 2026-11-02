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
class TestLaminate(unittest.TestCase):
    def test_layup(self):
        from rve.laminate import layup
        self.assertEqual(layup("[0, +-30, ±60, 0, 90]_R"), [0, 30, -30, 60, -60, 0, 90])
        self.assertEqual(layup("[0, 90]_s"), [0, 90, 90, 0])
        self.assertEqual(layup("[45, -45]_2"), [45, -45, 45, -45])

    def test_stack(self):
        """One angle: the lamina turned; the paper's own CLT inputs (Agotnes 2026 Table 1) give their laminate
        moduli within 0.25 % (the classical A-matrix: Ex 817.84 vs their 819.09); an aligned 3D stack keeps C."""
        from rve.laminate import inplane, lamina_from_engineering, layup, stack_C
        C = lamina_from_engineering(988.41, 706.44, 257.97, 0.36)
        self.assertLess(np.abs(stack_C(C, [30]) - rotate_z(C, 30)).max() / np.abs(C).max(), 1e-12)
        self.assertLess(np.abs(stack_C(C, [0, 0, 0]) - C).max() / np.abs(C).max(), 1e-12)
        e = inplane(stack_C(C, layup("[0, +-30, +-60, 0, 90]")))
        for k, v in (("Ex", 819.09), ("Ey", 777.04), ("Gxy", 287.58), ("nuxy", 0.352)):
            self.assertLess(abs(e[k] / v - 1), 2.5e-3, k)
        # +-45 laminate of an orthotropic ply: Gxy and Ex from the plane-stress closed form
        Q = np.linalg.inv(np.linalg.inv(C)[np.ix_([0, 1, 5], [0, 1, 5])])
        U = (Q[0, 0] + Q[1, 1] - 2 * Q[0, 1]) / 4
        self.assertAlmostEqual(inplane(stack_C(C, [45, -45]))["Gxy"], U, places=6)


class TestOnset(unittest.TestCase):
    """rve.onset on a small 0/90 bead cell (perfect bond, net-section means)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from rve import onset, strength
        cls.lib, strength.RESPONSES = strength.RESPONSES, None
        cls.tmp = tempfile.TemporaryDirectory()
        cls.g = dict(G, alpha=15.0, wc=math.sqrt(0.5))
        cls.wm = onset.weld_map("090", cls.g, 0.35, grade=1, work=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        from rve import strength
        strength.RESPONSES = cls.lib
        cls.tmp.cleanup()

    def test_equilibrium_and_symmetry(self):
        """The interlayer squares alone carry s33: mean traction = cell section / bond area exactly; latA under s22
        equals latB under s11 (B = A turned 90 deg)."""
        from rve import onset
        for f, (got, lig) in onset.lig_check(self.wm, "090", self.g).items():
            self.assertAlmostEqual(got / lig, 1.0, places=6, msg=f)
        self.assertAlmostEqual(self.wm["latA"]["N"][1] / self.wm["latB"]["N"][0], 1.0, places=4)

    def test_frame_and_scaling(self):
        """A stress given in a frame turned with the print gives the same weld tractions; onset scales linearly."""
        from rve import onset
        rng = np.random.default_rng(1)
        A = rng.normal(size=(5, 3, 3))
        sig = A + np.swapaxes(A, 1, 2)
        R = onset.cell_frame(30.0)
        ref = onset.weld_tractions(onset.cell_stress(sig, np.eye(3)), self.wm)
        rot = onset.weld_tractions(onset.cell_stress(np.einsum("ia,jb,nab->nij", R, R, sig), R), self.wm)
        for f in ref:
            for a, b in zip(ref[f], rot[f]):
                self.assertLess(np.abs(a - b).max(), 1e-12)
        d = np.linspace(0, 2, 5)
        f1 = onset.onset(sig, d, self.wm, R, 50.0, 1.5, 0.4, 60.0)[0]
        f2 = onset.onset(2 * sig, d, self.wm, R, 50.0, 1.5, 0.4, 60.0)[0]
        self.assertAlmostEqual(f1 / f2, 2.0, places=12)


class TestWindow(unittest.TestCase):
    """app/bead_window on grade-0 pieces: the corrector drive reproduces the periodic cell, GMRES the direct solve."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        from rve import window
        if not os.path.exists(window.BEAD_WINDOW):
            raise unittest.SkipTest("bead_window not built")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.files = window.pieces(window.BASE, 1, 0, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_corrector_drive_is_the_periodic_cell(self):
        """2 x 2 cells, cut faces driven by E x + chi E: the weld net tractions of a corner cell (both lateral welds
        inside) = the periodic cell's to the mesh level (2 % at grade 0; 0.3 % at grade 2), with springs too."""
        from rve import window
        ks = [None, 10.0]
        refs = window.cell_tractions(self.files, ks, self.tmp.name)
        Ts, _ = window.run_window(self.files, (2, 2), ks, "corrector", self.tmp.name)
        for k, T, Tr in zip(ks, Ts, refs):
            for f, e in window.error(window.window_means(T, 2, 0, 0), window.cell_means(Tr)).items():
                self.assertLess(e, 2.5, msg="%s kappa %s" % (f, k))

    def test_gmres_matches_direct(self):
        """2 x 1 cells, affine drive: GMRES with the cell block preconditioner = UMFPACK."""
        import subprocess
        from rve import window
        out = []
        for extra in ([], ["--direct"]):
            t = os.path.join(self.tmp.name, "t%d.txt" % len(out))
            subprocess.run([window.BEAD_WINDOW] + self.files + ["--cells", "2:1", "--cache", os.path.join(self.tmp.name, "cache"),
                                                                "--tractions", t] + extra, check=True, stdout=subprocess.DEVNULL)
            out.append(np.loadtxt(t)[:, 10:])
        self.assertLess(np.abs(out[0] - out[1]).max() / np.abs(out[1]).max(), 1e-7)


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
        if "runs" in r:
            return None, r
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

    def test_00_weld_scales_match_separate_solves(self):
        """--scales (one LU of the perfect-bond system + a Woodbury update per weld compliance scale) gives the C of
        separate solves with the scaled laws, and scale 0 the perfect bond (1e-10)."""
        import tempfile
        from rve.beads import bead_00
        from make_geometries import write_dat
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "b.dat")
            write_dat(f, bead_00(G["w"], G["h"], G["b_int"], G["b_lat"], refine=1, grade=1), "test")
            law = lambda K: ["--weld", "y:%.12g:%.12g" % (K, K / 3), "--weld", "z:%.12g:%.12g" % (K, K / 3)]  # noqa: E731
            K1 = 2200 / G["w"]
            _, r = self._run(self.bead_cell, [f, "--axes", "x", "--scales", "0,0.1,1"] + law(K1), d, "scales")
            Cp, _ = self._run(self.bead_cell, [f, "--axes", "x"], d, "perfect")
            C10, _ = self._run(self.bead_cell, [f, "--axes", "x"] + law(10 * K1), d, "k10")
            C1, _ = self._run(self.bead_cell, [f, "--axes", "x"] + law(K1), d, "k1")
            for run, C in zip(r["runs"], (Cp, C10, C1)):
                self.assertLess(np.abs(np.array(run["C"]) - C).max() / np.abs(C).max(), 1e-10)

    def test_00_response_matches_direct_solve(self):
        """The stored response (C(kappa) / E at RESPONSE_S, rational fit in 1 / kappa) gives the compliance of the direct
        spring-weld solve at kappas off the nodes (1e-5) and the perfect bond, for any E."""
        import tempfile
        from rve import strength
        g = dict(G, alpha=15.0, wc=math.sqrt(0.5), L=G["h"])
        lib, strength.RESPONSES = strength.RESPONSES, None   # keep test responses out of the shared library
        self.addCleanup(setattr, strength, "RESPONSES", lib)
        with tempfile.TemporaryDirectory() as d:
            r = strength.cell_response("00", g, 1, 1, d, "cell", 0.35)
            for kappa in (None, 0.7, 7.0, 300.0):
                S = strength.bead_compliance("00", g, 1, 1, d, "direct", 1300.0, 0.35, kappa)
                C = 1300.0 * strength.response_eval(r, "C", kappa)
                self.assertLess(np.abs(np.linalg.inv(S) - C).max() / np.abs(C).max(), 1e-5, kappa)

    def test_00_two_kappa_response(self):
        """Separate lateral and interlayer weld stiffnesses: the per-axis multi-scale solve (bead_cell --scales
        sx:sy:sz, one LU) equals the direct solve with the two laws; the 2D response (C2 grid, nested rational fits)
        gives it off the nodes to 1e-5, and the in-plane moduli do not move with the interlayer stiffness."""
        import json
        import subprocess
        import tempfile
        from rve import strength
        from rve.laminate import inplane
        g = dict(G, alpha=15.0, wc=math.sqrt(0.5), L=G["h"])
        lib, strength.RESPONSES = strength.RESPONSES, None
        self.addCleanup(setattr, strength, "RESPONSES", lib)
        nu = 0.35

        def law(k):
            Kn = k / G["w"]
            return "%.12g:%.12g" % (Kn, Kn * (1 - 2 * nu) / (2 * (1 - nu)))
        with tempfile.TemporaryDirectory() as d:
            r = strength.cell_response("00", g, 1, 1, d, "cell", nu, two=True)
            base = [strength.BEAD_CELL, os.path.join(d, "cell.dat"), "--axes", "x", "--E", "1", "--nu", str(nu),
                    "--ng", "12", "--cache", os.path.join(d, "cache")]
            out = os.path.join(d, "o.json")
            subprocess.run(base + ["--weld", "y:" + law(1), "--weld", "z:" + law(1), "--scales", "0:0.25:2,0",
                                   "--json", out], check=True, stdout=subprocess.DEVNULL)
            def rd():
                with open(out) as fh:
                    return json.load(fh)
            W = np.array(rd()["runs"][0]["C"])
            subprocess.run(base + ["--weld", "y:" + law(4), "--weld", "z:" + law(0.5), "--json", out], check=True,
                           stdout=subprocess.DEVNULL)
            D = np.array(rd()["C"])
            self.assertLess(np.abs(W - D).max() / np.abs(D).max(), 1e-10)
            C = strength.response_eval2(r, 4.0, 0.5)
            self.assertLess(np.abs(C - D).max() / np.abs(D).max(), 1e-5)
            soft, stiff = (inplane(strength.response_eval2(r, 4.0, ki)) for ki in (0.1, 100.0))
            self.assertLess(abs(soft["Ey"] / stiff["Ey"] - 1), 2e-3)
            self.assertGreater(engineering(strength.response_eval2(r, 4.0, 100.0))["E3"] /
                               engineering(strength.response_eval2(r, 4.0, 0.1))["E3"], 5)

    def test_00_weld_tractions(self):
        """Weld tractions (bead_cell --tractions, FFM stress part): the force on each weld per unit macro stress is
        the net section exactly (the weld is the cell's only load path across its plane), perfect or spring; a soft
        weld carries a near-uniform traction (edge mean within 5 % of the net section); the D -> 0 energy of a spring
        weld, G(0) = 1/2 t.C_w.t, agrees with the first narrowed cell (D = 0.05 half) within 10 %."""
        import tempfile
        from rve import strength
        from rve.beads import bead_00
        from make_geometries import write_dat
        g = dict(G, alpha=15.0, wc=math.sqrt(0.5), L=G["h"])
        fams = strength.families("00", G["w"], G["h"], G["b_int"], G["b_lat"], G["h"])
        lib, strength.OPLIB = strength.OPLIB, None   # keep test operators out of the shared library
        self.addCleanup(setattr, strength, "OPLIB", lib)
        with tempfile.TemporaryDirectory() as d:
            for kappa in (None, 0.3):
                S, T = strength.bead_compliance("00", g, 1, 1, d, "k%s" % kappa, 2200.0, 0.35, kappa, True)
                wm = strength.weld_means(T, S, "00", fams, lambda F: [0.05 * F["half"], F["half"]])
                for name, F in fams.items():
                    tn, ts, tedge = wm[name]
                    j = F["cuts"][0]
                    self.assertAlmostEqual(tn[-1, j] / (F["lig"]), 1.0, places=3)
                    if kappa is not None:
                        self.assertLess(abs(tedge[j] @ F["normal"] / F["lig"] - 1), 0.05)
                if kappa is not None:
                    F = fams["lat"]
                    tedge = wm["lat"][2]
                    Q = tedge @ strength.spring_compliance(F["normal"], kappa, 2200.0, 0.35, G["w"]) @ tedge.T
                    D = 0.05 * F["half"]
                    Sn = strength.bead_compliance("00", dict(g, **F["change"](D)), 1, 1, d, "n", 2200.0, 0.35, kappa)
                    V = G["w"] * G["h"] * G["h"]
                    ratio = (Sn - S)[1, 1] * V / F["area"](D) / Q[1, 1]
                    self.assertLess(abs(ratio - 1), 0.1)
