"""Gates G1 and G2 of the anisotropic failure layer plus its kernels (no solve).

usage: python3 -m unittest failure.test_aniso      (from python/; ~1 s)
"""
import math
import unittest

import numpy as np

from . import aniso as A
from . import prob
from . import weibull as W
from .fields import tet10_gradient, tet10_quadrature


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = math.cos(a), math.sin(a)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def _random_field(n, seed=0):
    g = np.random.default_rng(seed)
    a = g.normal(size=(n, 3, 3)) * 40.0
    return 0.5 * (a + np.swapaxes(a, 1, 2)), g.uniform(0.01, 0.2, n)


class Frames(unittest.TestCase):
    def test_orthonormal_right_handed(self):
        R = A.ply_frames(A.fibonacci_hemisphere(50), [0, 30, -45, 90])
        I = np.einsum("dpia,dpib->dpab", R, R)
        self.assertLess(np.abs(I - np.eye(3)).max(), 1e-12)
        self.assertLess(np.abs(np.linalg.det(R) - 1.0).max(), 1e-12)

    def test_flat_print_axes(self):
        R = A.ply_frames([0, 0, 1], [0, 90])[0]
        np.testing.assert_allclose(R[0], np.eye(3), atol=1e-15)          # 0 deg: filament along x
        np.testing.assert_allclose(R[1][:, 0], [0, 1, 0], atol=1e-15)    # 90 deg: filament along y

    def test_reference_parallel_to_build_direction(self):
        R = A.ply_frames([[0, 1, 0], [1, 0, 0]], [0.0], ref_dir=(0, 1, 0))
        self.assertLess(np.abs(np.einsum("dpia,dpib->dpab", R, R) - np.eye(3)).max(), 1e-12)

    def test_hemisphere(self):
        d = A.fibonacci_hemisphere(400)
        self.assertTrue(np.allclose(np.linalg.norm(d, axis=1), 1.0) and (d[:, 2] >= 0).all())


class G1IsotropicReduction(unittest.TestCase):
    """Mode P alone with the card's S0 is the vendored weibull.py law, bit for bit."""

    def test_bit_identical_log_j(self):
        sig, w = _random_field(5000)
        for m in (4.131588397578871, 10.0, 46.0):
            ref = W.log_stress_power_integral(sig, w, m, 1.0)
            lh = A.mode_hazards(sig, w, {"m": m, "modes": {"P": {"S0_MPa": 1.0}}})
            self.assertEqual(lh["P"], ref)
            S0 = 354.76819409548693
            eta = A.characteristic_load(A.mode_hazards(sig, w, {"m": m, "modes": {"P": {"S0_MPa": S0}}}), m, 1.0)
            self.assertAlmostEqual(eta / W.characteristic_failure_load_from_log_j(ref, S0, m, 1.0), 1.0, places=13)

    def test_compressive_field_has_no_hazard(self):
        sig = -np.eye(3)[None].repeat(4, 0) * 10.0
        lh = A.mode_hazards(sig, np.ones(4), {"m": 5.0, "modes": {"P": {"S0_MPa": 1.0}, "F": {"S0_MPa": 1.0}}},
                            A.Print())
        self.assertTrue(all(v == -math.inf for v in lh.values()))


class G2OffAxisBar(unittest.TestCase):
    """Uniform uniaxial stress at theta to the layer plane, unidirectional ply along its in-plane projection."""
    card = {"m": 8.0, "modes": {"F": {"S0_MPa": 60.0, "shear_ratio": 0.7}, "T": {"S0_MPa": 40.0, "shear_ratio": 1.3},
                                "L": {"S0_MPa": 30.0, "shear_ratio": 1.6}}}

    def test_matches_closed_form(self):
        pr = A.Print(build_dir=(0, 0, 1), angles_deg=(0.0,))
        for th in (0.0, 10.0, 30.0, 45.0, 60.0, 89.0, 90.0):
            u = _rot(1, -th) @ np.array([1.0, 0, 0])                     # load axis tilted out of the xy plane
            sig = np.outer(u, u)[None].repeat(3, 0)
            eta = A.characteristic_load(A.mode_hazards(sig, np.full(3, 1.0 / 3), self.card, pr), self.card["m"])
            ref = float(A.off_axis_strength(th, self.card))
            self.assertAlmostEqual(eta / ref, 1.0, places=12, msg=f"theta {th}")

    def test_limits(self):
        z = float(A.off_axis_strength(90.0, self.card))
        self.assertAlmostEqual(z, 30.0, places=10)                       # pure interlayer tension: Z
        x = float(A.off_axis_strength(0.0, self.card))
        self.assertAlmostEqual(x, 60.0, places=10)                       # along the filament: X

    def test_shares_and_probability(self):
        pr = A.Print(angles_deg=(0.0,))
        sig = np.diag([10.0, 0, 10.0])[None]
        lh = A.mode_hazards(sig, np.ones(1), self.card, pr)
        sh = A.mode_shares(lh)
        self.assertAlmostEqual(sum(sh.values()), 1.0, places=14)
        self.assertGreater(sh["L"], sh["F"])                             # Z (30) is weaker than X (60)
        eta = A.characteristic_load(lh, 8.0)
        self.assertAlmostEqual(A.failure_probability(eta, lh, 8.0), 1 - math.exp(-1), places=14)


class Rotation(unittest.TestCase):
    def test_rotating_part_and_print_together_changes_nothing(self):
        sig, w = _random_field(2000, seed=3)
        card = G2OffAxisBar.card
        pr = A.Print(build_dir=(0.3, -0.2, 0.9), angles_deg=(30.0, -60.0))
        Q = _rot(0, 25) @ _rot(2, 40)
        a = A.mode_hazards(sig, w, card, pr)
        b = A.mode_hazards(np.einsum("ia,nab,jb->nij", Q, sig, Q), w, card,
                           A.Print(build_dir=tuple(Q @ np.array(pr.build_dir)), angles_deg=pr.angles_deg,
                                   ref_dir=tuple(Q @ np.array(pr.ref_dir))))
        for k in a:
            self.assertAlmostEqual(a[k], b[k], places=9)

    def test_sweep_isotropic_is_flat(self):
        sig, w = _random_field(500, seed=4)
        eta, gov = A.sweep(sig, w, {"m": 5.0, "modes": {"P": {"S0_MPa": 50.0}}}, A.Print(),
                           A.fibonacci_hemisphere(20))
        self.assertLess(np.ptp(eta) / eta.mean(), 1e-14)
        self.assertEqual(set(gov), {"P"})


class Gradients(unittest.TestCase):
    def test_tet10_gradient_exact_on_quadratic(self):
        x = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]], float) + 0.1
        mid = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
        pts = np.vstack([x] + [0.5 * (x[i] + x[j]) for i, j in mid])
        f = lambda p: p[:, 0] ** 2 + 3 * p[:, 1] * p[:, 2] - p[:, 2]
        df = lambda p: np.column_stack([2 * p[:, 0], 3 * p[:, 2], 3 * p[:, 1] - 1])
        P, _, _ = tet10_quadrature(pts, np.arange(10)[None], f(pts))
        np.testing.assert_allclose(tet10_gradient(pts, np.arange(10)[None], f(pts)), df(P), atol=1e-12)

    def test_mode_gradient_matches_finite_difference(self):
        g = np.random.default_rng(7)
        s0, ds = _random_field(1, seed=8)[0][0], g.normal(size=(3, 3, 3)) * 5
        ds = 0.5 * (ds + np.swapaxes(ds, 0, 1))                          # d sigma_ij / d x_l, symmetric in ij
        s0 = s0 + np.diag([80.0, 60.0, 70.0])                            # tensile normals: every mode active
        R = A.ply_frames([0.2, 0.1, 1.0], [20.0])[0]
        h = 1e-6
        for k, r in (("P", 1.0), ("F", 0.0), ("F", 1.9), ("T", 1.4), ("L", 0.8)):
            fd = np.array([(A.mode_stress(k, (s0 + h * ds[:, :, l])[None], R, r)
                            - A.mode_stress(k, (s0 - h * ds[:, :, l])[None], R, r)).ravel()[0] / (2 * h)
                           for l in range(3)])
            an = A.mode_gradient(k, s0[None], ds[None], R if k != "P" else None, r).ravel()[0]
            self.assertAlmostEqual(an, np.linalg.norm(fd), delta=1e-6 * max(1.0, an), msg=k)

    def test_support_bounds_crack_tip(self):
        # sigma = K / sqrt(2 pi r): chi = 1/(2r), sigma/n -> K / sqrt(pi l) as r -> 0 (finite at the tip)
        r = np.logspace(-9, -1, 50)
        K, l = 30.0, 0.2
        s = K / np.sqrt(2 * np.pi * r)
        eff = s / A.support_factor(s, s / (2 * r), l)
        self.assertAlmostEqual(eff[0], K / math.sqrt(math.pi * l), delta=1e-3 * eff[0])
        self.assertTrue(np.all(np.diff(eff) < 0))


class Probability(unittest.TestCase):
    def test_physical_scatter_is_weibull(self):
        sc = prob.Scatter(m_s=46.0)
        g = math.gamma(1 + 1 / 46.0)
        for F in (2500.0, 3000.0, 3200.0):
            self.assertAlmostEqual(sc.cdf(F, 3100.0)[0], 1 - math.exp(-(F * g / 3100.0) ** 46.0), places=13)
        self.assertAlmostEqual(sc.load_at(0.01, 3100.0) / 3100.0, (-math.log(0.99)) ** (1 / 46.0) / g, places=10)

    def test_model_error_widens_band(self):
        a, b = prob.Scatter(46.0), prob.Scatter(46.0, tau=0.08)
        self.assertLess(b.load_at(0.01, 1.0), a.load_at(0.01, 1.0))
        self.assertAlmostEqual(b.cdf(b.load_at(0.3, 1.0), 1.0)[0], 0.3, places=9)

    def test_scatter_modulus_recovers_m(self):
        g = np.random.default_rng(1)
        groups = [3000.0 * g.weibull(40.0, 3) for _ in range(4000)]
        m_s, s, dof = prob.scatter_modulus(groups)
        self.assertEqual(dof, 8000)
        self.assertAlmostEqual(m_s / 40.0, 1.0, delta=0.02)

    def test_family_effects(self):
        fam = np.repeat(["a", "b", "c", "d"], 3)
        res = np.repeat([0.1, -0.1, 0.05, -0.05], 3)
        b, tau, off = prob.family_effects(res, fam, 0.0)
        self.assertAlmostEqual(b, 0.0, places=14)
        self.assertAlmostEqual(tau, float(np.std([0.1, -0.1, 0.05, -0.05], ddof=1)), places=14)
        self.assertAlmostEqual(off["a"], 0.1, places=14)


class TcdPoint(unittest.TestCase):
    def test_isotropic_mode_is_the_stress_point_method(self):
        d = np.linspace(0, 2, 201)
        sig = np.zeros((len(d), 3, 3))
        sig[:, 0, 0] = 50.0 / np.sqrt(1 + d)
        K, s0 = 120.0, 40.0
        out, gov = A.tcd_point(sig, d, {"modes": {"P": {"K": K, "sigma_0": s0}}})
        L = (K / s0) ** 2 / math.pi
        self.assertEqual(gov, "P")
        self.assertAlmostEqual(out["P"]["ratio"], float(np.interp(L / 2, d, sig[:, 0, 0])) / s0, places=12)

    def test_weak_bond_governs_transverse_load(self):
        d = np.linspace(0, 2, 201)
        sig = np.zeros((len(d), 3, 3))
        sig[:, 0, 0] = 30.0
        card = {"modes": {"F": {"K": 150.0, "sigma_0": 60.0}, "T": {"K": 80.0, "sigma_0": 40.0}}}
        out, gov = A.tcd_point(sig, d, card, A.Print(angles_deg=(90.0,)))   # filament along y, load along x
        self.assertEqual(gov, "T")
        self.assertAlmostEqual(out["T"]["ratio"], 0.75, places=12)
        self.assertLess(out["F"]["ratio"], 1e-15)


class G5PetgOffAxis(unittest.TestCase):
    """In-plane off-axis check on literature data (out-of-plane build-tilt data was not found): Popa, Galatanu &
    Marsavina, Frattura ed Integrita Strutturale 19(73) 153-165, 2025, PETG, uncontoured, Table 2 and text:
    X 55.21, Y 48.89, S12 26.83 (0 deg shear) MPa; a +-45 deg laminate measured 48.62 MPa (Tsai-Hill: 48.9)."""

    def test_pm45_laminate_strength(self):
        X, Y, S12, measured = 55.21, 48.89, 26.83, 48.62
        R = A.ply_frames([0, 0, 1], [45.0, -45.0])[0]
        sig = np.diag([1.0, 0.0, 0.0])[None]                             # unit uniaxial stress along x
        c, _ = A.components(sig, R)
        ratio = max(float(A.mode_from_components("F", c, X / S12)[p, 0]) / X for p in range(2))
        ratio = max(ratio, max(float(A.mode_from_components("T", c, Y / S12)[p, 0]) / Y for p in range(2)))
        self.assertLess(abs(1.0 / ratio / measured - 1.0), 0.05)        # first-ply mode strength 47.05 MPa, -3.2 %


class Cards(unittest.TestCase):
    def test_cards_load_and_domain_is_explicit(self):
        pla = A.load_card("pla-as2018-orient")
        self.assertEqual(set(pla["weibull"]["modes"]), {"F", "T"})          # L is off: no data
        demo = A.load_card("pla-as2018-orient-demo-L")
        self.assertIn("L", demo["weibull"]["modes"])
        self.assertIn("DEMONSTRATION", demo["name"])

    def test_g8_abs_card_is_flagged_out_of_calibration(self):
        abs_ = A.load_card("abs-cicero2020-orient")
        self.assertNotIn("weibull", abs_)                                    # TCD only
        self.assertIn("Ng-Susmel", abs_["domain"])

    def test_layup(self):
        self.assertEqual(A.parse_layup("30"), (30.0, -60.0))
        self.assertEqual(A.parse_layup("45/-45"), (45.0, -45.0))


if __name__ == "__main__":
    unittest.main()
