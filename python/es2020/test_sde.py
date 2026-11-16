"""Unit tests of the latent-damage SDE kernel (es2020.sde): python -m pytest python/es2020/test_sde.py -q"""
import math

import numpy as np
import pytest

from . import sde

E, K, N = 3479.0, 80.0, 0.25


def th(E=E, K=K, n=N, sD=60.0, s=2.0, sZ=0.5, tau=1e4, L=0.5):
    return np.array([math.log(E), math.log(K), n, math.log(sD), s, math.log(sZ), math.log(tau), math.log(L)])


def test_neuber_root_and_uniqueness():
    se = np.array([5.0, 20.0, 60.0])
    smax, smin = sde.loop(se, -1.0, E, K, N)
    eps = smax / E + (smax / K) ** (1 / N)
    assert np.allclose(smax * eps, se ** 2 / E, rtol=1e-10)
    assert np.allclose(smin, -smax, rtol=1e-9)                  # R = -1, Masing: symmetric loop
    assert np.all(smax < se)                                     # plasticity relaxes the elastic peak


def test_elastic_limit():
    smax, smin = sde.loop(np.array([10.0]), 0.0, E, 1e4, 0.2)
    assert abs(smax[0] - 10.0) < 1e-6 and abs(smin[0]) < 1e-6


def test_loop_integral_converged():
    smax, smin = sde.loop(np.array([30.0]), -1.0, E, K, N)
    a = sde.phi(smax, smin, K, N, 60.0, 2.0)
    x, w = sde.GL_X, sde.GL_W
    sde.GL_X, sde.GL_W = np.polynomial.legendre.leggauss(64)
    try:
        b = sde.phi(smax, smin, K, N, 60.0, 2.0)
    finally:
        sde.GL_X, sde.GL_W = x, w
    assert abs(a[0] / b[0] - 1) < 1e-4


def test_dissipation_sign_is_second_law():
    se = np.array([30.0])
    assert sde.dissipation_per_cycle(se, -1.0, th(n=0.3))[0] > 0
    assert abs(sde.dissipation_per_cycle(se, -1.0, th(n=1.0))[0]) < 1e-12   # reversible limit
    assert not sde.admissible(th(n=1.2))


def test_plastic_slope():
    """Plastic-dominated, R = -1, stiff elastic part: c ~ sigma^-(2s + 1/n')."""
    t = th(E=1e5, K=10.0, n=0.25, s=1.5)
    se = np.array([40.0, 80.0])
    # load control (uniform gauge) so the local stress is the applied stress
    c = sde.life_det(se, np.array([-1.0, -1.0]), t, load_control=np.array([True, True]))
    k = -math.log(c[1] / c[0]) / math.log(2)
    assert abs(k - (2 * 1.5 + 1 / 0.25)) < 0.05


def test_table_limits():
    q = sde.quantiles(np.array([0.8, 0.8]), np.array([1e-4, 1e4]))
    from scipy.stats import norm
    z = norm.ppf(sde.table()["qlev"])
    assert np.allclose(q[0], 0.32 + 0.8 * z)                     # frozen: log-normal, mean +sigma^2/2
    assert q[1, 49] < 0.05 and q[1, 89] - q[1, 9] < 0.1          # averaging: scatter vanishes
    # table edge matches the analytic limits
    T = sde.table()
    i = np.argmin(abs(T["sig"] - 0.8))
    lo = T["Q"][i, 0]
    assert abs(np.median(lo) - T["sig"][i] ** 2 / 2) < 0.05
    assert abs((lo[89] - lo[9]) / (2 * 1.2816) - T["sig"][i]) < 0.05


def test_likelihood_runs_and_respects_admissibility():
    m = sde.Model(sde.dataset(), "point")
    assert np.isfinite(m.loglike(th()))
    assert m.loglike(th(n=1.3)) == -np.inf


def test_field_direction_is_bisector_and_chi_matches():
    """Geometry-free input: the steepest-descent direction read from the field cloud equals the notch bisector
    (x axis) within 2 degrees, and chi matches the bisector-profile fit within 4 %, on all three coupon notches."""
    from . import methods as M
    for n in ("sharp", "intermediate", "blunt"):
        xy, s = sde.notch_field_points(n)
        peak, e, chi, h = sde.hot_spot_auto(xy, s)
        assert abs(math.degrees(math.atan2(e[1], e[0]))) < 2.0
        assert abs(chi / M.chi(n) - 1) < 0.04
        assert abs(peak / M._prof(n).s[0] - 1) < 0.002
        # window sensitivity: halving c changes chi by < 4 %
        assert abs(sde.hot_spot_auto(xy, s, c=0.1)[2] / chi - 1) < 0.04


def test_crack_path_rule_is_automatic():
    """One path rule: notches follow the field's steepest descent (bisector) to the opposite root (W = net width 6 mm);
    the uniform plain gauge has no descent direction, so the rule takes the shortest free path (the thickness)."""
    for n in ("sharp", "intermediate", "blunt"):
        lab, x0, e, W = sde.crack_path(n)
        assert lab.startswith("steepest") and abs(W - 6.0) < 1e-9
    for t in (3.0, 5.0):
        lab, _, _, W = sde.crack_path("plain", t)
        assert "thickness" in lab and W == t
    assert sde.crack_path("plain", 8.0)[3] == 6.0                 # thicker than wide: the width wins
    assert abs(sde.ray_to_free_surface([(0, 6), (0, 3)], np.array([1.0, 1.5]), np.array([1.0, 0.0])) - 5.0) < 1e-12
