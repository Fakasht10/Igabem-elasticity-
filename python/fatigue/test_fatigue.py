"""Unit tests of the fatigue transfer harness (fast, no solve) plus an opt-in solve gate.

    python3 -m pytest fatigue/test_fatigue.py            (fast)
    IGABEM_FATIGUE_SOLVE=1 python3 -m pytest fatigue/test_fatigue.py    (+ blunt-notch Kt on IgBEM, ~1.5 min)
"""
import json
import math
import os

import numpy as np
import pytest

from fatigue import criteria as C
from fatigue.specimens import Specimen, outline
from fatigue.transfer import plain_lookup, predict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "ezeh_susmel.json")))


def _closed(segs):
    ends = [(s[1], s[-1]) for s in segs]
    return all(np.allclose(ends[i][1], ends[(i + 1) % len(ends)][0]) for i in range(len(ends)))


def test_outlines_close_and_widths():
    for v in DATA["specimens"].values():
        s = Specimen.from_dict(v)
        assert _closed(outline(s)) and s.net_width == 6.0 and s.gross_width == 25.0
    h = Specimen(shape="hole", T=3.2, rho=2.0, W=13.0, L=60.0)
    assert _closed(outline(h)) and math.isclose(h.net_width, 9.0) and h.root[1] == 1.0


def _uniform(n=50):
    d = np.linspace(0, 3, n)
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 1.5, 400)
    return {"d": d, "s1_line": np.ones((5, n)), "s1": np.ones(400), "w": w, "g1": np.zeros(400)}


def test_uniform_field_transfers_unchanged():
    f, s0 = _uniform(), 7.0
    V = 4 * f["w"].sum()
    for got in (C.kt(f, s0), C.tcd_point(f, s0, 0.5), C.tcd_line(f, s0, 0.5), C.weibull(f, s0, 5.0, V),
                C.weibull_support(f, s0, 5.0, V, 0.3)):
        assert math.isclose(got, s0, rel_tol=1e-9)


def test_tcd_on_a_linear_profile():
    d = np.linspace(0, 3, 301)
    f = {"d": d, "s1_line": np.tile(3.0 - d, (5, 1))}
    assert math.isclose(C.tcd_point(f, 1.0, 1.0), 1 / 2.5, rel_tol=1e-9)
    assert math.isclose(C.tcd_line(f, 1.0, 0.5), 1 / 2.5, rel_tol=1e-6)      # mean of 3 - d over [0, 1]


def test_weibull_volume_scaling():
    f, s0, m = _uniform(), 5.0, 8.0
    V = 4 * f["w"].sum()
    assert math.isclose(C.weibull(f, s0, m, 2 ** m * V), 2 * s0, rel_tol=1e-9)


def test_m_from_T():
    s = math.log(1.5) / (2 * C.Z90)
    assert math.isclose(C.m_from_T(1.5), C.GUMBEL_STD / s, rel_tol=1e-12)


def test_every_notched_series_has_its_plain_series():
    pl = plain_lookup(DATA)
    assert all((r["p"], r["R"]) in pl for r in DATA["notched"])
    f = {k: _uniform() for k in DATA["specimens"]}
    rows = predict(DATA, f, [30.0], log=lambda *_: None)
    assert len(rows) == len(DATA["notched"]) and all(math.isclose(r["kt"], r["s0"]) for r in rows)


@pytest.mark.skipif(not os.environ.get("IGABEM_FATIGUE_SOLVE"), reason="set IGABEM_FATIGUE_SOLVE=1 to solve")
def test_blunt_notch_Kt_on_igabem(tmp_path):
    from fatigue.fields import field
    f = field(Specimen.from_dict(DATA["specimens"]["blunt"]), h=1.5, work=str(tmp_path), volume=False)
    assert abs(f["Kt_mid"] / 1.621 - 1) < 0.01           # regression value, designed blocks, h 1.5 (2026-11-15)


def _outline_area(s, n=4000):
    pts = []
    for seg in outline(s):
        if seg[0] == "line":
            pts.append(seg[1])
            continue
        p, c, q = (np.asarray(x) for x in seg[1:])
        a0, a1 = math.atan2(*(p - c)[::-1]), math.atan2(*(q - c)[::-1])
        pts += [c + s.rho * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(a0, a1, n)[:-1]]
    x, y = np.array(pts).T
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def test_blocks_exact_closed_and_fast():
    """Designed root layout: every patch edge shared (closed, conforming), top face area = outline area, < 1 s."""
    import time
    from grade_dat import link
    from spline_checks import evaluate
    from fatigue import blocks
    g, w = np.polynomial.legendre.leggauss(4)
    g, w = 0.5 * (g + 1), 0.5 * w
    for v in DATA["specimens"].values():
        s = Specimen.from_dict(v)
        t = time.perf_counter()
        P, names, info = blocks.build(s)
        assert time.perf_counter() - t < 1.0
        _, shared = link(P)
        assert all((k, d, e) in shared for k in range(len(P)) for d in (0, 1) for e in (0, 1))
        area = 0.0
        for Q, nm in zip(P, names):
            if not nm.endswith(".top"):
                continue
            U, V = np.unique(Q.U), np.unique(Q.V)
            for i in range(len(U) - 1):
                for j in range(len(V) - 1):
                    du, dv = U[i + 1] - U[i], V[j + 1] - V[j]
                    for a, wa in zip(g, w):
                        for b, wb in zip(g, w):
                            _, xu, xv = evaluate(Q, U[i] + a * du, V[j] + b * dv)
                            area += wa * wb * du * dv * np.linalg.norm(np.cross(xu, xv))
        assert math.isclose(area, _outline_area(s), rel_tol=1e-6)
