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


if __name__ == "__main__":
    unittest.main()
