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
    for s in [Specimen.from_dict(v) for v in DATA["specimens"].values()] + [Specimen("hole", T=3.2, rho=2.0, W=13.0,
                                                                                        L=60.0),
                                                                               Specimen("edge", T=3.2, rho=2.0, WG=13.0,
                                                                                        WN=9.0, L=60.0)]:
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


def test_printed_edge_distance_and_frames():
    from fatigue import printed as Pr
    s = Specimen.from_dict(DATA["specimens"]["blunt"])              # root arc centre (WN/2 + rho, 0), radius 3
    xy = np.array([[2.9, 0.0], [3.0 + 3 - 3 * math.cos(0.3), 3 * math.sin(0.3) * 0.9], [1.0, 0.0], [12.4, 30.0]])
    d, t = Pr.edge_distance(s, xy)
    assert d[0] == pytest.approx(0.1) and abs(t[0] @ [1, 0]) < 1e-12        # root: tangent along the load axis
    assert d[2] == pytest.approx(2.0)                                       # symmetry lines are not free edges
    assert d[3] == pytest.approx(0.1) and abs(t[3] @ [0, 1]) == pytest.approx(1.0)
    # uniaxial unit stress along y in a ply at phi: F = cos^2, T = sqrt(sin^4 + r^2 sin^2 cos^2)
    sig = Pr.tensor(np.array([[0, 1.0, 0, 0, 0, 0]]))
    F, N, X = Pr.mode_stresses(sig, Pr._frame_from_dir(Pr.ply_dir(30.0)[None]))
    assert F[0] == pytest.approx(math.cos(math.radians(30)) ** 2) and N[0] == pytest.approx(0.25)
    assert abs(X[0]) == pytest.approx(math.sin(math.radians(30)) * math.cos(math.radians(30)))


def test_printed_plain_hazard_is_the_uniform_limit():
    from fatigue import printed as Pr
    m, V = 30.0, 540.0
    # contour only: J = V / S_F^m; one 0-deg ply set and no contour: F mode at S_F, T mode zero
    assert Pr.plain_log_hazard((0.0,), m, 20.0, 9.0, 1.0, V, 1.0) == pytest.approx(math.log(V) - m * math.log(20.0))
    assert Pr.plain_log_hazard((0.0,), m, 20.0, 9.0, 1.0, V, 0.0) == pytest.approx(math.log(V) - m * math.log(20.0))


def test_sn_life_and_supported_weibull_limits():
    """life() inverts Basquin; supported Weibull with l -> 0 is plain Weibull and lowers the effective stress for l > 0."""
    import math
    from . import criteria as C
    from .fields import field
    from .sn_transfer import life
    from .specimens import Specimen
    A, B = 68.13, -0.161
    assert abs(life(lambda N: 1.0, A * 2e4 ** B, A, B) / 2e4 - 1) < 1e-9
    assert abs(life(lambda N: 1.2, A * 2e4 ** B / 1.2, A, B) / 2e4 - 1) < 1e-9
    f = field(Specimen.from_dict(dict(shape="hole", W=13.0, T=3.2, rho=2.0, L=60.0)), h=1.5, log=lambda *_: None)
    V = 13 * 3.2 * 50
    for m in (7.5, 40.0):
        w0, ws = C.weibull(f, 1.0, m, V), C.weibull_support(f, 1.0, m, V, 1e-9)
        assert abs(ws / w0 - 1) < 1e-4
        assert C.weibull_support(f, 1.0, m, V, 0.7) > w0


def test_root_chi_near_siebel_and_supported_kt_limit():
    """chi_0 of a semicircular hole is near Siebel's 2/rho and stable between meshes; supported_kt(l=0) is Kt."""
    from . import criteria as C
    from .fields import field
    from .specimens import Specimen
    f = field(Specimen.from_dict(dict(shape="hole", W=13.0, T=3.2, rho=2.0, L=60.0)), h=1.5, log=lambda *_: None)
    chi = C.root_chi(f)
    assert 0.5 < chi < 1.3                                        # 2/rho = 1.0; finite width raises it
    assert abs(C.supported_kt(f, 1.0, 0.0) * float(C.line_profile(f)[1][0]) - 1) < 1e-12


def test_master_factors_plain_limits():
    """Master collapse: a plain bar has f_eff 1 (net, kt, hotspot) and (V/V_ref)^(1/m) for Weibull."""
    from .master import V_REF, factor
    assert factor("net", None, [], 500.0) == 1.0 and factor("hotspot", None, [0.9], 500.0) == 1.0
    assert abs(factor("weibull", None, [8.0], V_REF) - 1.0) < 1e-12
    assert abs(factor("weibull", None, [8.0], 2 * V_REF) - 2 ** 0.125) < 1e-12


def test_mhsv_reduces_to_point_method_when_volumes_match():
    from fatigue import hsv
    pl = dict(sigma_A50=5.0, k=6.0, T=1.4, UTS=42.0)
    W, T, Lg, Lf = 6.0, 3.0, 30.0, 0.4
    S90 = 2 * (W + T) * Lg * Lf / hsv.L_CD                  # V_N = V_S
    g = {"n": dict(smax=3.0, S90=S90, V90=W * T * Lg, seff=1.7, prof=lambda x: 1.7)}
    a = hsv.predict("tcd_pm", g, pl, "n", 4.0, Lf, Lg)[0]
    assert abs(hsv.predict("mhsv", g, pl, "n", 4.0, Lf, Lg)[0] / a - 1) < 1e-12
    b = hsv.predict("hsv", g, pl, "n", 4.0, Lf, Lg)[0]
    assert abs(b - 2e6 * (3.0 * 4.0 / 5.0) ** -6.0) / b < 1e-12


def test_hsv_fix_surface_weights_and_size_term():
    """Trapezoid surface weights integrate the exact arc / side areas; e = 0 is the point method; eq. 10 in the life
    domain equals the stress factor with e = 1/(k k_SN) on a power-law plain curve; Weibull MLE recovers k."""
    from fatigue import hsv, hsv_fix
    th, z, r, rho = np.linspace(0, 1.2, 31), np.linspace(0, 3, 21), np.linspace(0.5, 1.0, 17), 0.5
    s = dict(th=th, z=z, r=r, dA_arc=rho * np.outer(np.gradient(th), np.gradient(z)),
             s1_arc=np.ones((31, 21)), s1_side=np.zeros((31, 17)))
    a, b = hsv.surface_weights(dict(s, dA_side=np.outer(np.gradient(th), r * np.gradient(r))))
    assert abs(a.sum() - rho * 1.2 * 3) < 1e-12 and abs(b.sum() - 1.2 / 2 * (1.0 - 0.25)) < 1e-3
    n = type("N", (), {"pm": lambda self, L: 1.7})()
    assert hsv_fix.feff(n, 2.0, 0.0, 0.49, 100.0) == 1.7
    k, ksn, ratio, sn = 0.9, 5.1, 300.0, 1.3                  # N_N = N_S(sigma_eff) ratio^(1/k), N_S = (S/s0)^-ksn
    NN = (1.7 * sn) ** -ksn * ratio ** (1 / k)
    f = hsv_fix.feff(n, 1.0, 1 / (k * ksn), 1.0, ratio)       # S_S L_FCI / (S_N L_CD) = ratio
    assert abs((f * sn) ** -ksn / NN - 1) < 1e-12
    x = np.random.default_rng(1).weibull(1.5, 4000)
    assert abs(hsv_fix.k_weibull_mle(x) - 1.5) < 0.06


def test_calib_stats_flags_tilted_fit():
    """A residual that is centred (bias 0) but trends with life must be flagged; a flat one must not."""
    import numpy as np
    from fatigue.calib import stats
    N = np.geomspace(1e2, 1e6, 40)
    tilt = stats(-0.09 * (np.log10(N) - 4), N)
    flat = stats(0.01 * np.sin(np.arange(40)), N)
    assert abs(tilt["bias"]) < 1e-12 and tilt["tilted"] and abs(tilt["slope"] + 0.09) < 1e-9
    assert not flat["tilted"]
