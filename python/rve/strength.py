"""Bond strength of bead cells by finite fracture mechanics: the coupled stress and energy criterion
(Leguillon 2002, Eur. J. Mech. A 21, 61-72) at the bond tips, from linear periodic-cell solves only.

A bond (a flat weld between two beads: lateral in a layer, or interlayer) debonds from its tips inward. A finite
debond of length D at every tip of a family of bonds happens under the macro stress s d (d: unit direction, s: load
factor) when both hold:

  stress   the mean traction over D ahead of the tips reaches the weld strength:
           max(<t_n>_D / T_n, |t_s|_D / T_s) * s >= 1        (the max-nominal-stress law of Biswas 2019, eq. 3.9)
  energy   the energy released by the debond reaches the weld toughness:
           s^2 V d.(S(D) - S(0)).d / (2 A(D)) >= Gc          (S: cell compliance, A: debonded area per cell)

so s_f = min_D max(s_stress(D), s_energy(D)). Both parts are linear-elastic: the tractions come from the six unit
macro strain solutions of the intact cell (interior stress on the bond planes, app/rve_cell --points), the energy
from the compliance of the same cell with the bond narrowed by 2D (a debond advancing from both tips is modelled
as the notch moving inward at the same notch angle). Every load direction d then costs nothing: the whole strength
envelope comes from one intact-cell solve plus one compliance per debond length and family.

Limits: a short cohesive length (brittle weld, l_ch = E Gc / T^2 << bond) gives the Griffith-like energy-controlled
strength; a long one (the thesis's l_ch ~ 2.6 mm >> 0.1-0.36 mm bonds) gives the net-section stress limit at
D -> half the bond.

Imperfect welds (--kappa): a linear spring on every weld (weld_args); the stress part then reads the weld tractions
of the intact bead cell (weld_means: bounded at the tips, plain strip means) and the energy part the spring-welded
narrowed cells, with the weak-interface limit G(0) = 1/2 t.C_w.t at D = 0 (python/rve/weld_strength.py sweeps kappa).

Cells: python/rve/beads.py. Families: 0/0 "lat" (in-layer, normal y), "int" (interlayer, normal z); 0/90 "latA"
(A layer along x, normal y), "latB" (B layer along y, normal x), "int" (b x b squares, normal z).

usage (from python/): python3 -m rve.strength 00|090 out.json [--w .6 --h .3 --bint .36 --blat .1 --alpha 15]
                       [--T 30.3 --Gc 1.09] [--refine 1 --grade 4] [--work dir]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from make_geometries import write_dat  # noqa: E402
from rve.beads import cell_00, cell_090, tip_marker  # noqa: E402
from rve.homog import rotate_z  # noqa: E402
from spline_checks import evaluate  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RVE_CELL = os.environ.get("IGABEM_RVE_CELL", os.path.join(ROOT, "build", "rve_cell"))
BEAD_CELL = os.environ.get("IGABEM_BEAD_CELL", os.path.join(ROOT, "build", "bead_cell"))
# What is kept between runs is each cell's RESPONSE, not its operator (cell_response): the stiffness C(kappa) / E
# (and the FFM weld-traction means of the intact cell) at RESPONSE_S, a few KB per cell, keyed by the bead files,
# nu and quadrature. It serves every E, raster angle, layup and kappa. Default: the persistent project folder when it
# exists (survives sessions), else <work>/responses. The bead operators (bead_cell --cache, 235 MB per 0/90 bead)
# are only a local scratch cache for building responses of new geometry.
_PROJECT_RESP = "/mnt/project-files/igabem-rve/responses"
RESPONSES = os.environ.get("IGABEM_RESPONSES") or (
    _PROJECT_RESP if os.path.isdir(os.path.dirname(_PROJECT_RESP)) else None)
OPLIB = os.environ.get("IGABEM_OPLIB")
# weld compliance scale s = 1 / kappa (laws at kappa = 1): 0 (perfect bond) and 16 nodes over kappa 0.01 - 1e4. The
# response is rational in s (A(s) = A0 + s U E^T); AAA through these nodes reproduces the direct solve to ~1e-6.
RESPONSE_S = (0.0,) + tuple(float(x) for x in np.geomspace(1e-4, 1e2, 16))
RESPONSE_VERSION = 1
FRACTIONS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.9)   # debond length / half bond width (compliance runs)
# D = half bond: the whole ligament. Its mean traction is the net-section stress (an equilibrium check); if the
# family is the only load path for some macro stress components (CUTS), separating it releases unbounded energy
# under load control, so the energy criterion holds there at any load.


# ------------------------------------------------------------------ bond-plane traction profiles
RHO = np.geomspace(0.02, 0.995, 28)   # distance from the tip / half bond width
FIT_H = (1.0, 3.0)   # asymptote fit window at the tips, in units of the smallest tip element (h_tip)
FIT_MAX = 0.3        # ... capped at this fraction of the half bond


def williams(alpha_deg):
    """Notch-root eigenvalues (lambda_I, lambda_II, lambda_III) of a V-notch of opening 2 alpha (solid angle
    2 beta = 2 pi - 2 alpha): stresses ~ r^(lambda - 1) (Williams 1952; mode III pi / (2 beta)). Crack: 0.5 each."""
    from scipy.optimize import brentq
    b2 = 2 * math.pi - 2 * math.radians(alpha_deg)
    out = []
    for sgn in (1.0, -1.0):
        f = lambda l: math.sin(b2 * l) + sgn * l * math.sin(b2)  # noqa: E731
        ls = np.linspace(0.3003, 0.9997, 701)
        v = [f(l) for l in ls]
        root = next(brentq(f, ls[i], ls[i + 1]) for i in range(700) if v[i] * v[i + 1] < 0)
        out.append(root)
    return out[0], out[1], math.pi / b2


def line_profiles(tip, inward, axis, length, nax=1):
    """Stations (tip, inward, weight) along a straight tip line tip + a axis, a in [0, length] (one station at
    mid-length when the field does not vary along the line)."""
    if nax == 1:
        a, wa = np.array([0.5]), np.array([1.0])
    else:
        xa, wa = leggauss(nax)
        a, wa = (xa + 1) / 2, wa / 2
    return [(np.asarray(tip, float) + ai * length * np.asarray(axis, float), np.asarray(inward, float), wai)
            for ai, wai in zip(a, wa)]


def square_profiles(b, z, w, nst=6):
    """Stations on the four edges of the b x b bond square centred on the cell corner (its quarters at the four
    corners of [0, w]^2): inward normals of the edges, stations along each half edge."""
    xa, wa = leggauss(nst)
    t, wt = (xa + 1) / 2, wa / 2
    out = []
    for cx, sx in ((0.0, 1.0), (w, -1.0)):
        for cy, sy in ((0.0, 1.0), (w, -1.0)):
            for ti, wi in zip(t, wt):
                out.append((np.array([cx + sx * b / 2, cy + sy * ti * b / 2, z]), np.array([-sx, 0, 0.0]), wi))
                out.append((np.array([cx + sx * ti * b / 2, cy + sy * b / 2, z]), np.array([0, -sy, 0.0]), wi))
    return out


def families(kind, w, h, b_int, b_lat, L):
    """Per family: bond normal, the macro stress components it alone carries (cuts) and its net-section factor lig
    (cell section / bond area: the exact mean traction of the whole ligament per unit macro traction), debonded area
    per cell A(D), geometry change for D, the half bond width, the strip weight g(r) of the debond front at depth r
    g(r) = 1 - gk r (gk = 0 for a straight front, 2/b for a b x b square shrinking from its edges) and profile stations (tip, inward, weight)."""
    ex, ey, ez = np.eye(3)
    if kind == "00":
        return {
            "lat": dict(normal=ey, cuts=(1, 3, 5), lig=h / b_lat, area=lambda D: 2 * D * L, gk=0.0,
                        change=lambda D: {"b_lat": b_lat - 2 * D}, half=b_lat / 2,
                        stations=line_profiles([0, w / 2, b_lat / 2], -ez, ex, L)
                        + line_profiles([0, w / 2, h - b_lat / 2], ez, ex, L)),
            "int": dict(normal=ez, cuts=(2, 3, 4), lig=w / b_int, area=lambda D: 2 * D * L, gk=0.0,
                        change=lambda D: {"b_int": b_int - 2 * D}, half=b_int / 2,
                        stations=line_profiles([0, b_int / 2, h / 2], -ey, ex, L)
                        + line_profiles([0, w - b_int / 2, h / 2], ey, ex, L)),
        }
    return {
        "latA": dict(normal=ey, cuts=(), lig=None, area=lambda D: 2 * D * w, gk=0.0,
                     change=lambda D: {"b_lat": b_lat - 2 * D, "b_lat_b": b_lat}, half=b_lat / 2,
                     stations=line_profiles([0, w / 2, b_lat / 2], -ez, ex, w, 6)
                     + line_profiles([0, w / 2, 2 * h - b_lat / 2], ez, ex, w, 6)),
        "latB": dict(normal=ex, cuts=(), lig=None, area=lambda D: 2 * D * w, gk=0.0,
                     change=lambda D: {"b_lat_b": b_lat - 2 * D}, half=b_lat / 2,
                     stations=line_profiles([w / 2, 0, h - b_lat / 2], ez, ey, w, 6)
                     + line_profiles([w / 2, 0, h + b_lat / 2], -ez, ey, w, 6)),
        "int": dict(normal=ez, cuts=(2, 3, 4), lig=(w / b_int) ** 2, gk=2 / b_int,
                    area=lambda D: 2 * (b_int ** 2 - (b_int - 2 * D) ** 2),
                    change=lambda D: {"b_int": b_int - 2 * D}, half=b_int / 2,
                    stations=[st for z in (h / 2, 3 * h / 2) for st in square_profiles(b_int, z, w)]),
    }


# ------------------------------------------------------------------ cell runs
def build(kind, g, refine, grade, path):
    if kind == "00":
        P = cell_00(g["w"], g["h"], g["b_int"], g["b_lat"], g["alpha"], g["wc"], g["L"], refine, grade)
    else:
        P = cell_090(g["w"], g["h"], g["b_int"], g["b_lat"], g["alpha"], g["wc"], refine, grade, g.get("b_lat_b"))
    tmp = path + ".new"
    write_dat(tmp, P, "bead cell %s" % kind)
    if os.path.exists(path) and open(tmp, "rb").read() == open(path, "rb").read():
        os.remove(tmp)                 # unchanged cell: keep its mtime so its cached solve is reused
    else:
        os.replace(tmp, path)
    return P


def tip_element(patches, lines):
    """Smallest element length leaving the tip lines [(point, axis)]: distance from a tip corner of a patch to the
    first knot line along the other parameter direction. The interior stress is trusted from about one such element
    from the root on (FIT_H)."""
    on = tip_marker(lines, 1e-7)
    hmin = np.inf
    for P in patches:
        ku, kv = np.unique(P.U), np.unique(P.V)
        for u, u1 in ((ku[0], ku[1]), (ku[-1], ku[-2])):
            for v, v1 in ((kv[0], kv[1]), (kv[-1], kv[-2])):
                x = evaluate(P, u, v)[0]
                if not on(x):
                    continue
                for y in (evaluate(P, u1, v)[0], evaluate(P, u, v1)[0]):
                    if not on(y):
                        hmin = min(hmin, float(np.linalg.norm(y - x)))
    return hmin


def run_cell(dat, out_json, E, nu, points=None):
    """PBC solve of a cell (skipped when out_json is newer than the cell and its points, from the same E, nu)."""
    deps = [dat] + ([points] if points else [])
    if os.path.exists(out_json) and all(os.path.getmtime(out_json) > os.path.getmtime(f) for f in deps):
        try:
            with open(out_json) as f:
                r = json.load(f)
            if r.get("E") == E and r.get("nu") == nu and (points is None or "points" in r):
                return r
        except ValueError:
            pass
    cmd = [RVE_CELL, dat, "--bc", "pbc", "--ng", "12", "--E", str(E), "--nu", str(nu), "--json", out_json]
    if points is not None:
        cmd += ["--points", points]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(out_json) as f:
        return json.load(f)


def weld_args(kind, kappa, E, nu, w):
    """bead_cell options for a linear spring on every weld: Kn = kappa E / w (w the filament pitch), Kt = Kn (1 - 2 nu)
    / (2 (1 - nu)) (a thin isotropic interphase of the bead's moduli); kappa None: perfect bond."""
    if kappa is None:
        return []
    Kn = kappa * E / w
    law = "%.10g:%.10g" % (Kn, Kn * (1 - 2 * nu) / (2 * (1 - nu)))
    args = []
    for a in ("yz" if kind == "00" else "xyz"):
        args += ["--weld", "%s:%s" % (a, law)]
    return args + (["--weld-int", "z:" + law] if kind == "090" else [])


def _write_if_changed(path, P, title):
    """write_dat, keeping the file (and its mtime) when the content is unchanged: cached results stay valid."""
    write_dat(path + ".new", P, title)
    if os.path.exists(path) and open(path + ".new", "rb").read() == open(path, "rb").read():
        os.remove(path + ".new")
    else:
        os.replace(path + ".new", path)


def _ktag(kappa):
    return "" if kappa is None else "_k%g" % kappa


_GEOMETRY = {}


def _bead_patches(kind, g, refine, grade):
    """Bead patches of a cell (memoized: a kappa sweep revisits every cell once per kappa)."""
    from rve.beads import bead_00, bead_090
    key = (kind, tuple(sorted(g.items())), refine, grade)
    if key not in _GEOMETRY:
        if kind == "00":
            P = [bead_00(g["w"], g["h"], g["b_int"], g["b_lat"], g["alpha"], g["wc"], g["L"], refine, grade)]
        else:
            bb = g.get("b_lat_b") or g["b_lat"]
            mk = lambda bl: bead_090(g["w"], g["h"], g["b_int"], bl, g["alpha"], g["wc"], refine, grade)  # noqa: E731
            A = mk(g["b_lat"])
            P = [A[0], A[1] if bb == g["b_lat"] else mk(bb)[1]]
        _GEOMETRY[key] = P
    return _GEOMETRY[key]


def bead_compliance(kind, g, refine, grade, work, tag, E, nu, kappa=None, tractions=False, batch=None):
    """Compliance of the bead-boundary cell (app/bead_cell: one region per bead, condensed onto the welds, cached)
    for the geometry g, welds perfect (kappa None) or springs of stiffness kappa (weld_args). 0/90: B is A turned
    90 deg (one operator) unless its lateral bond differs (b_lat_b). tractions: also return the weld tractions
    (BeadCell::weldTractions).

    batch: every kappa of a sweep (None: perfect bond). The first call solves the cell for all of them in one
    bead_cell run (--scales: one LU of the perfect-bond system, then a Woodbury update per kappa) and stores each
    result as <tag>_k<kappa>.json; later calls for the same cell read them."""
    title = "bead cell"   # fixed: the operator cache hashes the whole file
    patches = _bead_patches(kind, g, refine, grade)
    if kind == "00":
        files = [os.path.join(work, tag + ".dat")]
        cmd = [BEAD_CELL] + files + ["--axes", "x"]
    else:
        files = [os.path.join(work, "%s_%s.dat" % (tag, t)) for t in "AB"]
        same = (g.get("b_lat_b") or g["b_lat"]) == g["b_lat"]
        cmd = [BEAD_CELL] + files + ["--axes", "x,y"] + (["--copy", "1:0:90"] if same else [])
    for f, P in zip(files, patches):
        _write_if_changed(f, P, title)
    out = lambda k: os.path.join(work, tag + _ktag(k) + ".json")  # noqa: E731
    tfile = lambda k: os.path.join(work, tag + _ktag(k) + "_tractions.txt")  # noqa: E731

    def done(k):
        if not os.path.exists(out(k)) or (tractions and not os.path.exists(tfile(k))):
            return False
        t = min(os.path.getmtime(out(k)), os.path.getmtime(tfile(k)) if tractions else np.inf)
        with open(out(k)) as fh:
            r = json.load(fh)
        return all(t > os.path.getmtime(f) for f in files) and r.get("E") == E and r.get("nu") == nu

    if not done(kappa):
        ks = [k for k in (batch or [kappa]) if not done(k)]
        if kappa not in ks:
            ks.append(kappa)
        springs = any(k is not None for k in ks)
        cmd += ["--E", str(E), "--nu", str(nu), "--ng", "12", "--cache", OPLIB or os.path.join(work, "cache"),
                "--json", out(kappa) + ".all"] + (["--tractions", tfile(kappa) + ".all"] if tractions else [])
        if springs:   # laws at kappa = 1, each kappa a compliance scale 1 / kappa (perfect: 0)
            cmd += weld_args(kind, 1.0, E, nu, g["w"]) + ["--scales", ",".join("0" if k is None else "%.17g" % (1 / k) for k in ks)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        with open(out(kappa) + ".all") as fh:
            r = json.load(fh)
        runs = r["runs"] if "runs" in r else [r]
        for q, (k, rk) in enumerate(zip(ks, runs)):
            if tractions:
                os.replace(tfile(kappa) + ".all" + ("." + str(q) if len(ks) > 1 else ""), tfile(k))
            with open(out(k), "w") as fh:
                json.dump(dict(rk, kappa=k), fh)
        os.remove(out(kappa) + ".all")
    with open(out(kappa)) as fh:
        S = np.linalg.inv(np.array(json.load(fh)["C"]))
    return (S, np.loadtxt(tfile(kappa))) if tractions else S


_FITS = {}


def _aaa(s_nodes, y):
    """Rational interpolant in s through the response nodes, entry by entry (scipy AAA); constant entries exact."""
    from scipy.interpolate import AAA
    y = np.asarray(y, float)
    flat = y.reshape(len(s_nodes), -1)
    scale = max(np.abs(flat).max(), 1e-300)
    fits = []
    for c in range(flat.shape[1]):
        v = flat[:, c]
        if np.ptp(v) <= 1e-14 * scale:
            fits.append(float(v[0]))
        else:
            fits.append(AAA(np.asarray(s_nodes), v, rtol=1e-13))
    shape = y.shape[1:]
    return lambda s: np.array([f if isinstance(f, float) else float(np.real(f(s))) for f in fits]).reshape(shape)


def response_eval(resp, field, kappa, values=None):
    """Response field (or the given node values of one) at weld stiffness kappa (None: perfect bond); the rational
    fits are memoized per response, field."""
    s = 0.0 if kappa is None else 1.0 / kappa
    if s > resp["s"][-1] * (1 + 1e-9):
        raise ValueError("kappa %g below the response range (%g)" % (kappa, 1 / resp["s"][-1]))
    key = (resp["key"], field)
    if key not in _FITS:
        _FITS[key] = _aaa(resp["s"], resp[field] if values is None else values)
    return _FITS[key](s)


def cell_response(kind, g, refine, grade, work, tag, nu, means=None):
    """Response of the bead-boundary cell for the geometry g against the weld stiffness, from the response library
    (built on a miss: one bead_cell run at E = 1 over RESPONSE_S, laws Kn = kappa E / w, Kt from nu as weld_args):
      C        (nodes, 6, 6) stiffness / E;  porosity, box
      means    {mkey: {family: [tn, ts, tedge] per node}} FFM weld-traction means per unit macro stress (MPa / MPa:
               independent of E) when means = (mkey, fn) is given, fn(T, S) -> weld_means(...) dict
    Responses do not depend on E, raster angle or layup; the bead operators are not kept beyond the local cache."""
    import hashlib
    title = "bead cell"   # fixed: the operator cache and the response key hash the whole file
    patches = _bead_patches(kind, g, refine, grade)
    if kind == "00":
        files = [os.path.join(work, tag + ".dat")]
        cmd = [BEAD_CELL] + files + ["--axes", "x"]
    else:
        files = [os.path.join(work, "%s_%s.dat" % (tag, t)) for t in "AB"]
        same = (g.get("b_lat_b") or g["b_lat"]) == g["b_lat"]
        cmd = [BEAD_CELL] + files + ["--axes", "x,y"] + (["--copy", "1:0:90"] if same else [])
    for f, P in zip(files, patches):
        _write_if_changed(f, P, title)
    h = hashlib.sha1()
    for f in files:
        with open(f, "rb") as fh:
            h.update(fh.read())
    h.update(repr((kind, round(nu, 12), 12, cmd[len(files) + 1:], RESPONSE_S, RESPONSE_VERSION)).encode())
    lib = RESPONSES or os.path.join(work, "responses")
    os.makedirs(lib, exist_ok=True)
    path = os.path.join(lib, h.hexdigest() + ".json")
    resp = None
    if os.path.exists(path):
        with open(path) as fh:
            resp = json.load(fh)
        if means is None or means[0] in resp.get("means", {}):
            return dict(resp, key=h.hexdigest())
    out, tfile = os.path.join(work, tag + "_resp.json"), os.path.join(work, tag + "_resp_t")
    cmd += ["--E", "1", "--nu", str(nu), "--ng", "12", "--cache", OPLIB or os.path.join(work, "cache"),
            "--json", out] + weld_args(kind, 1.0, 1.0, nu, g["w"])
    cmd += ["--scales", ",".join("%.17g" % x for x in RESPONSE_S)] + (["--tractions", tfile] if means else [])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    with open(out) as fh:
        runs = json.load(fh)["runs"]
    if resp is None:
        resp = dict(kind=kind, nu=nu, s=list(RESPONSE_S), C=[r["C"] for r in runs], porosity=runs[0]["porosity"],
                    box=runs[0]["box"], files=[os.path.basename(f) for f in files], means={})
    if means:
        mk, fn = means
        per = [fn(np.loadtxt(tfile + ".%d" % q), np.linalg.inv(np.array(r["C"]))) for q, r in enumerate(runs)]
        resp["means"][mk] = {fam: [[np.asarray(p[fam][c]).tolist() for p in per] for c in range(3)] for fam in per[0]}
        for q in range(len(runs)):
            os.remove(tfile + ".%d" % q)
    with open(path + ".tmp%d" % os.getpid(), "w") as fh:
        json.dump(resp, fh)
    os.replace(path + ".tmp%d" % os.getpid(), path)
    return dict(resp, key=h.hexdigest())


# Weld of each family in the bead cell (src/bead_cell.hpp pairs): normal axis, internal flags allowed.
WELDS = {"00": {"lat": (1, (0,)), "int": (2, (0,))},
         "090": {"latA": (1, (0,)), "latB": (0, (0,)), "int": (2, (0, 1))}}


def weld_means(T, S, kind, fams, Ds_of):
    """FFM stress part from the weld tractions of the intact bead cell (T: BeadCell::weldTractions rows, S: its
    compliance): the mean traction over the strip within D of the bond edges, tn (nD, 6) and ts (nD, 6, 3), per unit
    macro stress. With spring welds the tractions are bounded at the bond tips (no notch singularity), so the mean
    is the plain area mean over the Gauss points, D from a cumulative integral sorted by the distance to the edge
    r = half - max_c |x_c - centre_c| over the in-plane coordinates c across the bond. D = half: the whole bond
    (equilibrium: the net-section traction exactly)."""
    ax, it, x, w, nrm = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2:5], T[:, 5], T[:, 6:9]
    t = np.einsum("kj,pkc->pjc", S, T[:, 9:].reshape(-1, 6, 3))        # per unit macro stress
    box = x.max(0) - x.min(0)
    out = {}
    for name, F in fams.items():
        a, ints = WELDS[kind][name]
        m = (ax == a) & np.isin(it, ints)
        e = np.asarray(F["normal"], float)
        te = t[m] * np.sign(nrm[m] @ e)[:, None, None]                  # traction on the face with normal e
        tn = te @ e                                                      # (p, 6)
        tsv = te - tn[..., None] * e
        xm = x[m]
        across = [c for c in range(3) if c != a and np.ptp(xm[:, c]) < 0.99 * box[c]]
        lo, hi = xm[:, across].min(0), xm[:, across].max(0)
        half = 0.5 * float(np.max(hi - lo))
        if abs(half - F["half"]) > 0.02 * F["half"]:
            raise RuntimeError("weld %s: half width %.4g, expected %.4g" % (name, half, F["half"]))
        r = F["half"] - np.max(np.abs(xm[:, across] - 0.5 * (lo + hi)), axis=1)
        o = np.argsort(r)
        cw = np.concatenate([[0.0], np.cumsum(w[m][o])])
        rr = np.concatenate([[0.0], np.maximum(r[o], 0.0)])
        cn = np.vstack([np.zeros(6), np.cumsum(w[m][o, None] * tn[o], 0)])
        cs = np.concatenate([np.zeros((1, 6, 3)), np.cumsum(w[m][o, None, None] * tsv[o], 0)])
        Ds = np.asarray(Ds_of(F))
        A = np.interp(Ds, rr, cw)
        tnD = np.stack([np.interp(Ds, rr, cn[:, j]) for j in range(6)], 1) / A[:, None]
        tsD = np.stack([np.stack([np.interp(Ds, rr, cs[:, j, c]) for c in range(3)], 1) for j in range(6)], 1) / A[:, None, None]
        edge = r < EDGE * F["half"]                                      # traction at the bond edge (D -> 0)
        tedge = (w[m][edge, None, None] * te[edge]).sum(0) / w[m][edge].sum()
        out[name] = (tnD, tsD, tedge)
    return out


EDGE = 0.02   # edge traction: mean over the strip r < EDGE half


def spring_compliance(normal, kappa, E, nu, w):
    """3 x 3 compliance of the weld spring (rve.strength.weld_args) on a bond of this normal."""
    Kn = kappa * E / w
    Kt = Kn * (1 - 2 * nu) / (2 * (1 - nu))
    nn = np.outer(normal, normal)
    return nn / Kn + (np.eye(3) - nn) / Kt


# ------------------------------------------------------------------ criterion
def unit_stress_fields(res):
    """Point stresses (n, 6) for the six unit macro stresses, from the six unit macro strain solutions."""
    C = np.array(res["bc"]["pbc"]["C"])
    Se = np.array(res["points"]["stress"])           # (6 cases, n, 6)
    S = np.linalg.inv(C)
    return np.einsum("kj,kpi->jpi", S, Se), S        # field j: macro stress e_j


VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def tensor(v):
    T = np.zeros(v.shape[:-1] + (3, 3))
    for a, (i, k) in enumerate(VOIGT):
        T[..., i, k] = T[..., k, i] = v[..., a]
    return T


def profile_points(F):
    r = RHO * F["half"]
    return [np.asarray(p) + r[:, None] * np.asarray(q) for p, q, _ in F["stations"]]


def _gint(lo, hi, lam, gk):
    """int_lo^hi r^(lam - 1) (1 - gk r) dr."""
    return (hi ** lam - lo ** lam) / lam - gk * (hi ** (lam + 1) - lo ** (lam + 1)) / (lam + 1)


def tip_means(fields, idx, F, lams, Ds, h_tip):
    """Mean traction over a debond of depth D from the tips, per unit macro stress component j: tn (nD, 6),
    ts (nD, 6, 3); plus the station-averaged local profiles (6, nr, 3: normal, inward, along the tip) for plots.

    Each station's profile is split into the tip-local components (normal: mode I, inward shear: mode II, shear
    along the tip line: mode III). Close to the notch root the boundary-element interior stress loses accuracy
    (points within about one element h_tip of the singular root), so below r0 = FIT_H[0] h_tip the profile is
    replaced by the Williams asymptote K r^(lambda - 1) + c fitted on [FIT_H] h_tip (capped at FIT_MAX half); it is
    integrated in closed form."""
    n = F["normal"]
    r = RHO * F["half"]
    r1 = min(FIT_H[1] * h_tip, FIT_MAX * F["half"])
    r0 = max(r1 * FIT_H[0] / FIT_H[1], r[0])
    fit = (r >= r0 * 0.999) & (r <= r1 * 1.001)
    gk = F["gk"]
    wsum = sum(c for _, _, c in F["stations"])
    tn, ts, prof = np.zeros((len(Ds), 6)), np.zeros((len(Ds), 6, 3)), np.zeros((6, len(r), 3))
    for (p, q, c), ix in zip(F["stations"], idx):
        q = np.asarray(q, float)
        a = np.cross(n, q)
        B = np.stack([n, q, a])                                     # local frame rows
        for j in range(6):
            t = tensor(fields[j][ix]) @ n                           # (nr, 3)
            loc = t @ B.T                                           # (nr, 3) local components
            prof[j] += c / wsum * loc
            for m in range(3):
                lam = lams[m]
                A = np.stack([r[fit] ** (lam - 1), np.ones(fit.sum())], 1)
                K, c0 = np.linalg.lstsq(A, loc[fit, m], rcond=None)[0]
                for i, D in enumerate(Ds):
                    hi = min(D, r0)
                    I = K * _gint(0.0, hi, lam, gk) + c0 * (hi - gk * hi ** 2 / 2)
                    if D > r0:
                        rf = np.linspace(r0, D, 200)
                        y = np.interp(rf, r, loc[:, m]) * (1 - gk * rf)
                        I += np.trapezoid(y, rf)
                    mean = I / (D - gk * D ** 2 / 2)
                    vec = mean * B[m]
                    tn[i, j] += c / wsum * (vec @ n)
                    ts[i, j] += c / wsum * (vec - (vec @ n) * n)
    return tn, ts, prof


def ligament(F):
    """Exact mean traction of the whole ligament (net section) for the six unit macro stresses, when the bond plane
    cuts the cell through this bond only: lig * sigma n."""
    n = F["normal"]
    tn, ts = np.zeros(6), np.zeros((6, 3))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        t = F["lig"] * tensor(e) @ n
        tn[j], ts[j] = t @ n, t - (t @ n) * n
    return tn, ts


def strength(d, fam_data, T_n, T_s, Gc, V):
    """Load factor at failure in direction d (unit macro stress, Voigt) and the controlling family / debond."""
    best = (np.inf, None, None)
    for name, F in fam_data.items():
        Ds = np.array(F["D"])
        s_st, s_en = [], []
        for i, D in enumerate(Ds):
            tn = F["tn"][i] @ d
            ts = np.linalg.norm(F["ts"][i].T @ d)
            f = max(max(tn, 0.0) / T_n, ts / T_s)
            s_st.append(1 / f if f > 0 else np.inf)
            g = V * 0.5 * d @ (F["S"][i] - F["S0"]) @ d / F["A"][i]
            s_en.append(math.sqrt(Gc / g) if g > 0 else np.inf)
        s_st, s_en = np.array(s_st), np.array(s_en)
        if F.get("lig") is not None and any(abs(d[c]) > 1e-12 for c in F["cuts"]):   # whole ligament: energy unbounded, stress = net section
            tn = F["lig"]["tn"] @ d
            ts = np.linalg.norm(F["lig"]["ts"].T @ d)
            f = max(max(tn, 0.0) / T_n, ts / T_s)
            Ds = np.append(Ds, F["lig"]["D"])
            s_st = np.append(s_st, 1 / f if f > 0 else np.inf)
            s_en = np.append(s_en, 1e-30)
        # interpolate both on a fine grid in D (log-linear in s), take min over D of the max
        Df = np.linspace(Ds[0], Ds[-1], 400)
        lg = lambda y: np.interp(Df, Ds, np.log(np.minimum(y, 1e30)))  # noqa: E731
        s = np.exp(np.maximum(lg(s_st), lg(s_en)))
        i = int(np.argmin(s))
        if s[i] < best[0]:
            best = (float(s[i]), name, float(Df[i]))
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["00", "090"])
    ap.add_argument("out")
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--bint", type=float, default=0.36)
    ap.add_argument("--blat", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=15.0)
    ap.add_argument("--wc", type=float, default=math.sqrt(0.5))
    ap.add_argument("--E", type=float, default=2200.0)
    ap.add_argument("--nu", type=float, default=0.35)
    ap.add_argument("--T", type=float, nargs="+", default=[30.3], help="weld strength(s) (normal = shear)")
    ap.add_argument("--Gc", type=float, default=1.09)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--grade", type=int, default=4)
    ap.add_argument("--work", default="/tmp/rve_strength")
    ap.add_argument("--grade-energy", type=int, default=None,
                    help="tip grading of the compliance cells (default: --grade); the energy release rate is a global "
                         "quantity and converges on coarser meshes than the tip tractions")
    ap.add_argument("--fractions", type=float, nargs="+", default=list(FRACTIONS),
                    help="debond depths of the compliance cells, as fractions of the half bond")
    ap.add_argument("--check-only", action="store_true", help="intact cell only: profiles and the ligament check")
    ap.add_argument("--cells", choices=["single", "bead"], default="single",
                    help="compliance cells: single-region (rve_cell) or bead-boundary (bead_cell: condensed beads, "
                         "B = A turned; 3-6x faster for 0/90)")
    ap.add_argument("--kappa", type=float, default=None,
                    help="weld stiffness kappa = K w / E of a linear spring on every weld (rve.strength.weld_args; "
                         "default: perfect bond). Needs the bead cells; both FFM parts then see the springs")
    ap.add_argument("--stress", choices=["points", "weld"], default=None,
                    help="stress part from the interior stress of the intact single-region cell (points, Williams tip "
                         "fit; perfect bond only) or the weld tractions of the intact bead cell (weld; default with --kappa)")
    a = ap.parse_args(argv)
    if a.stress is None:
        a.stress = "weld" if a.kappa is not None else "points"
    if a.kappa is not None or a.stress == "weld":
        if a.stress == "points":
            raise SystemExit("--kappa: the stress part needs the weld tractions (--stress weld)")
        a.cells = "bead"
    os.makedirs(a.work, exist_ok=True)
    g = dict(w=a.w, h=a.h, b_int=a.bint, b_lat=a.blat, alpha=a.alpha, wc=a.wc, L=a.h)
    fams = families(a.kind, a.w, a.h, a.bint, a.blat, g["L"])
    lams = williams(a.alpha)
    # intact cell: one solve, the stress at every profile point of every family
    pts, index, k = [], {}, 0
    for name, F in fams.items():
        for P in profile_points(F):
            index.setdefault(name, []).append(np.arange(k, k + len(P)))
            pts.append(P)
            k += len(P)
    pfile = os.path.join(a.work, "points.txt")
    np.savetxt(pfile + ".new", np.vstack(pts))
    if os.path.exists(pfile) and open(pfile + ".new").read() == open(pfile).read():
        os.remove(pfile + ".new")
    else:
        os.replace(pfile + ".new", pfile)
    ge = a.grade if a.grade_energy is None else a.grade_energy
    h_tip = {}
    if a.stress == "points":
        base_dat = os.path.join(a.work, "intact.dat")
        patches = build(a.kind, g, a.refine, a.grade, base_dat)
        h_tip = {n: tip_element(patches, [(st[0], np.cross(F["normal"], st[1])) for st in F["stations"]])
                 for n, F in fams.items()}
        res0 = run_cell(base_dat, os.path.join(a.work, "intact.json"), a.E, a.nu, pfile)
        fields, S0 = unit_stress_fields(res0)
        info = dict(box=res0["box"], porosity=res0["porosity"], C=res0["bc"]["pbc"]["C"])
        print("tip elements %s mm; Williams exponents lambda I/II/III = %.4f %.4f %.4f (alpha %g deg)" % ({n: round(v, 5) for n, v in h_tip.items()}, *lams, a.alpha))
    S0e = None if a.stress == "weld" else S0
    if a.cells == "bead" and not (a.check_only and a.stress == "points"):
        Ds_of = lambda F: [fr * F["half"] for fr in a.fractions] + [F["half"]]  # noqa: E731
        mkey = "fractions=" + ",".join("%g" % fr for fr in a.fractions)
        r0 = cell_response(a.kind, g, a.refine, ge, a.work, "bead_intact", a.nu,
                           (mkey, lambda T, S: weld_means(T, S, a.kind, fams, Ds_of)))
        C0 = a.E * response_eval(r0, "C", a.kappa)
        S0e = np.linalg.inv(C0)
        if a.stress == "weld":
            info = dict(box=r0["box"], porosity=r0["porosity"], C=C0.tolist())
            wm = {}
            for fam, vals in r0["means"][mkey].items():
                wm[fam] = tuple(response_eval(r0, (mkey, fam, c), a.kappa, vals[c]) for c in range(3))
    elif ge != a.grade and not a.check_only:   # reference compliance on the energy mesh
        dat = os.path.join(a.work, "intact_e.dat")
        build(a.kind, g, a.refine, ge, dat)
        S0e = np.linalg.inv(np.array(run_cell(dat, dat.replace(".dat", ".json"), a.E, a.nu)["bc"]["pbc"]["C"]))
    V = float(np.prod(info["box"]))
    fam_data = {}
    for name, F in fams.items():
        Ds = [fr * F["half"] for fr in a.fractions]
        if a.stress == "points":
            tn, ts, prof = tip_means(fields, index[name], F, lams, Ds + [F["half"]], h_tip[name])
        else:
            (tn, ts, tedge), prof = wm[name], None
        fd = dict(D=[], tn=[], ts=[], S=[], A=[], S0=S0e, cuts=F["cuts"], profile=prof, check=None)
        if F["lig"] is not None:
            tl, sl = ligament(F)
            fd["lig"] = dict(D=F["half"], tn=tl, ts=sl)
            j = F["cuts"][0]   # normal cut component: integrated profile vs equilibrium
            fd["check"] = float(tn[-1, j] / tl[j])
            print("  %s: ligament mean traction from the profiles / equilibrium = %.4f (%.4f vs %.4f)"
                  % (name, fd["check"], tn[-1, j], tl[j]))
        fam_data[name] = fd
        if a.check_only:
            continue
        if a.kappa is not None:
            # D -> 0 on a spring weld: the stress part is the edge traction and the energy released per area is the
            # spring energy there, G(0) = 1/2 t.C_w.t (weak-interface limit), entered as dS/dA = Q / V
            e = np.asarray(F["normal"], float)
            Q = tedge @ spring_compliance(e, a.kappa, a.E, a.nu, a.w) @ tedge.T
            fd["D"].append(0.0)
            fd["tn"].append(tedge @ e)
            fd["ts"].append(tedge - np.outer(tedge @ e, e))
            fd["S"].append(S0e + Q / V)
            fd["A"].append(1.0)
        for i, D in enumerate(Ds):
            gi = dict(g, **F["change"](D))
            if a.cells == "bead" and a.kind == "090" and name == "latA":
                # the 0/90 cell turned 90 deg about z (and shifted a layer) is itself, with the A and B layers
                # swapped: a latA debond is the latB debond of the same depth turned 90 deg (8e-5 apart when both
                # are solved), so it costs no cell of its own
                gB = dict(g, **fams["latB"]["change"](D))
                SB = np.linalg.inv(a.E * response_eval(cell_response(a.kind, gB, a.refine, ge, a.work,
                                                                     "bead_latB_%d" % i, a.nu), "C", a.kappa))
                Si = np.linalg.inv(rotate_z(np.linalg.inv(SB), 90.0))
            elif a.cells == "bead":
                Si = np.linalg.inv(a.E * response_eval(cell_response(a.kind, gi, a.refine, ge, a.work,
                                                                     "bead_%s_%d" % (name, i), a.nu), "C", a.kappa))
            else:
                dat = os.path.join(a.work, "%s_%d.dat" % (name, i))
                build(a.kind, gi, a.refine, ge, dat)
                Si = np.linalg.inv(np.array(run_cell(dat, dat.replace(".dat", ".json"), a.E, a.nu)["bc"]["pbc"]["C"]))
            fd["D"].append(D)
            fd["tn"].append(tn[i])
            fd["ts"].append(ts[i])
            fd["S"].append(Si)
            fd["A"].append(F["area"](D))
    if a.check_only:
        return fam_data
    out = dict(kind=a.kind, geometry=g, E=a.E, nu=a.nu, Gc=a.Gc, kappa=a.kappa, stress=a.stress,
               porosity=info["porosity"], C=info["C"], l_ch=[a.E * a.Gc / T ** 2 for T in a.T], results={})
    names = ["X (11)", "Y (22)", "Z (33)", "S23", "S13", "S12"]
    for T in a.T:
        rows = {}
        for j in range(6):
            d = np.zeros(6)
            d[j] = 1.0
            s, fam, D = strength(d, fam_data, T, T, a.Gc, V)
            if s > 1e3 * T:   # no bond carries this stress: filament (bulk) failure, not a bond mode
                s, fam, D = None, "filament", None
            rows[names[j]] = dict(strength=s, family=fam, debond=D)
        # sigma_11 - sigma_33 and sigma_22 - sigma_33 envelopes
        env = {}
        for (i, j) in ((0, 2), (1, 2)):
            pts_env = []
            for th in np.linspace(0, 2 * np.pi, 73)[:-1]:
                d = np.zeros(6)
                d[i], d[j] = math.cos(th), math.sin(th)
                s, fam, D = strength(d, fam_data, T, T, a.Gc, V)
                pts_env.append([s * math.cos(th), s * math.sin(th), fam])
            env["%d%d" % (i + 1, j + 1)] = pts_env
        out["results"]["T=%g" % T] = dict(uniaxial=rows, envelope=env)
        print("T = %g MPa, Gc = %g N/mm, l_ch = %.3g mm" % (T, a.Gc, a.E * a.Gc / T ** 2))
        for k2, v in rows.items():
            if v["strength"] is None:
                print("  %-7s      --       (no bond loaded: filament strength)" % k2)
            else:
                print("  %-7s %8.2f MPa  (%s, debond %.4f mm)" % (k2, v["strength"], v["family"], v["debond"]))
    # traction / compliance tables for plots
    out["williams"] = list(lams)
    out["h_tip"] = h_tip
    out["rho"] = RHO.tolist()
    out["families"] = {n: dict(D=f["D"], tn=[t.tolist() for t in f["tn"]], ts=[t.tolist() for t in f["ts"]],
                               dS=[(s - S0e).tolist() for s in f["S"]], A=f["A"], cuts=list(f["cuts"]),
                               half=fams[n]["half"],
                               profile=None if f["profile"] is None else f["profile"].tolist(), ligament_check=f["check"],
                               ligament=None if "lig" not in f else
                               dict(D=f["lig"]["D"], tn=f["lig"]["tn"].tolist(), ts=f["lig"]["ts"].tolist()))
                       for n, f in fam_data.items()}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
