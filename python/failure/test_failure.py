"""Fast checks of the vendored failure layer (no solve): cards and their calibration hashes, the TCD lengths, the
Creager-Paris fit, the Weibull integral, and the TET10 quadrature volume.

usage: python3 -m unittest failure.test_failure      (from python/; ~1 s)
"""
import math
import unittest

import numpy as np

from . import fracture as F
from . import materials as M
from . import weibull as W
from .fields import tet10_quadrature


class Cards(unittest.TestCase):
    def test_cards_load_and_weibull_hashes_pass(self):
        self.assertGreaterEqual(len(M.all_cards()), 50)
        self.assertTrue(M.weibull_cards(), "no Weibull card passed its calibration hash")

    def test_tcd_length(self):
        self.assertAlmostEqual(F.L_tcd(82.2, 23.0), (82.2 / 23.0) ** 2 / math.pi, places=12)
        self.assertAlmostEqual(F.L_tcd(82.2, 23.0), 4.066, places=3)


class Regression(unittest.TestCase):
    def test_creager_paris_recovers_K_and_rho(self):
        r = np.linspace(0.02, 1.0, 60)
        fit = F.creager_paris_fit(r, F.creager_paris(r, 120.0, 0.5))
        self.assertAlmostEqual(fit["K_I"], 120.0, delta=0.5)
        self.assertAlmostEqual(fit["rho"], 0.5, delta=0.01)
        self.assertGreater(fit["r2"], 0.9999)


class Weibull(unittest.TestCase):
    def test_uniform_field(self):
        sig = np.zeros((10, 3, 3))
        sig[:, 0, 0] = 50.0
        w = np.full(10, 0.3)
        self.assertAlmostEqual(W.log_stress_power_integral(sig, w, 5.0, 1.0), math.log(3.0 * 50.0 ** 5), places=10)
        sig[:, 0, 0] = -50.0
        self.assertEqual(W.stress_power_integral(sig, w, 5.0, 1.0), 0.0)


class Quadrature(unittest.TestCase):
    def test_tet10_volume(self):
        x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float)
        mid = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
        pts = np.vstack([x] + [0.5 * (x[i] + x[j]) for i, j in mid])
        P, w, _ = tet10_quadrature(pts, np.arange(10)[None, :], np.zeros((10, 1)))
        self.assertAlmostEqual(w.sum(), 4.0, places=12)


if __name__ == "__main__":
    unittest.main()
