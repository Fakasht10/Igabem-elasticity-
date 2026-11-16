"""Gates for the Ezeh & Susmel 2020 reproduction. No solver needed: IgBEM profiles are stored in data/.

    python -m pytest python/es2020 -q
"""
import collections

import numpy as np
import pytest

from es2020 import fig10, paper, point


def _fit(specs):
    """Paper's regression: log N on log sigma, failures only; returns k and sigma at 2e6."""
    S = np.array([(s.s_a, s.N) for s in specs if not s.runout])
    b, a = np.polyfit(np.log10(S[:, 0]), np.log10(S[:, 1]), 1)
    return -b, 10 ** ((np.log10(paper.N_REF) - a) / b), len(S)


def test_figs5to7_reproduce_table1():
    g = collections.defaultdict(list)
    for s in paper.specimens():
        g[(s.notch, s.p, int(s.R))].append(s)
    assert len(g) == 18 and sum(map(len, g.values())) == 157
    for key, (n, k, sA) in paper.TABLE1.items():
        kk, s50, nf = _fit(g[key])
        assert len(g[key]) == n, key
        assert abs(kk - k) < 0.06 and abs(s50 - sA) < 0.06, (key, kk, s50)


def test_fig8_counts_and_curves():
    f8 = paper.fig8_specimens()
    c = collections.Counter((s.notch, s.p, int(s.R)) for s in f8)
    assert c == {key: v[0] for key, v in paper.TABLE1.items()}
    printed = {"sharp": (3.2, 3.9), "intermediate": (3.4, 4.6), "blunt": (5.1, 9.0)}   # Fig. 8a-c, sigma_n,max
    for notch, (k, s) in printed.items():
        S = [paper.Specimen(x.notch, x.p, -1.0, x.s_max, x.N, x.runout) for x in f8 if x.notch == notch]
        kk, s50, _ = _fit(S)                                     # R = -1 trick: s_a = s_max
        assert abs(kk - k) < 0.06 and abs(s50 - s) < 0.06, (notch, kk, s50)


def test_fig10_markers_match_specimens():
    M = paper.fig10_markers()
    assert collections.Counter((m[0], m[3]) for m in M) == {("sharp", False): 47, ("intermediate", False): 48,
                                                            ("intermediate", True): 4, ("blunt", False): 54,
                                                            ("blunt", True): 4}


def test_profiles_physical():
    P = point.profiles()
    for k, kt in (("sharp", 6.41), ("intermediate", 2.54), ("blunt", 1.59)):
        assert P[k](0) == pytest.approx(kt, abs=0.01)
        assert np.all(np.diff(P[k].s[:50]) < 0)                  # decays from the root


def test_calibration_reproduces_eq6():
    A, B, _, _ = point.calibrate(point.profiles()["sharp"])     # PLAIN + SHARP_FIG8A, Fig. 9e
    assert A == pytest.approx(paper.LM_A, rel=0.03) and B == pytest.approx(paper.LM_B, abs=0.003)


def test_fig10_point_method_marker_by_marker():
    C = fig10.compare()
    r = np.concatenate([np.log10(C[k][:, 1] / C[k][:, 2]) for k in paper.NOTCHES])
    dn = np.concatenate([np.log10(C[k][:, 0] / C[k][:, 3]) for k in paper.NOTCHES])
    assert len(r) == 157 and np.abs(dn).max() < 0.005            # paired by N (within 1 %)
    assert np.abs(r).max() < np.log10(1.03) and np.sqrt(np.mean(r ** 2)) < np.log10(1.015)
