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
from spline_checks import evaluate  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RVE_CELL = os.environ.get("IGABEM_RVE_CELL", os.path.join(ROOT, "build", "rve_cell"))
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
    a = ap.parse_args(argv)
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
    base_dat = os.path.join(a.work, "intact.dat")
    patches = build(a.kind, g, a.refine, a.grade, base_dat)
    h_tip = {n: tip_element(patches, [(st[0], np.cross(F["normal"], st[1])) for st in F["stations"]])
             for n, F in fams.items()}
    res0 = run_cell(base_dat, os.path.join(a.work, "intact.json"), a.E, a.nu, pfile)
    fields, S0 = unit_stress_fields(res0)
    V = float(np.prod(res0["box"]))
    ge = a.grade if a.grade_energy is None else a.grade_energy
    S0e = S0
    if ge != a.grade and not a.check_only:   # reference compliance on the energy mesh
        dat = os.path.join(a.work, "intact_e.dat")
        build(a.kind, g, a.refine, ge, dat)
        S0e = np.linalg.inv(np.array(run_cell(dat, dat.replace(".dat", ".json"), a.E, a.nu)["bc"]["pbc"]["C"]))
    print("tip elements %s mm; Williams exponents lambda I/II/III = %.4f %.4f %.4f (alpha %g deg)" % ({n: round(v, 5) for n, v in h_tip.items()}, *lams, a.alpha))
    fam_data = {}
    for name, F in fams.items():
        Ds = [fr * F["half"] for fr in a.fractions]
        tn, ts, prof = tip_means(fields, index[name], F, lams, Ds + [F["half"]], h_tip[name])
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
        for i, D in enumerate(Ds):
            gi = dict(g, **F["change"](D))
            dat = os.path.join(a.work, "%s_%d.dat" % (name, i))
            build(a.kind, gi, a.refine, ge, dat)
            r = run_cell(dat, dat.replace(".dat", ".json"), a.E, a.nu)
            fd["D"].append(D)
            fd["tn"].append(tn[i])
            fd["ts"].append(ts[i])
            fd["S"].append(np.linalg.inv(np.array(r["bc"]["pbc"]["C"])))
            fd["A"].append(F["area"](D))
    if a.check_only:
        return fam_data
    out = dict(kind=a.kind, geometry=g, E=a.E, nu=a.nu, Gc=a.Gc, porosity=res0["porosity"],
               C=res0["bc"]["pbc"]["C"], l_ch=[a.E * a.Gc / T ** 2 for T in a.T], results={})
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
                               half=fams[n]["half"], profile=f["profile"].tolist(), ligament_check=f["check"],
                               ligament=None if "lig" not in f else
                               dict(D=f["lig"]["D"], tn=f["lig"]["tn"].tolist(), ts=f["lig"]["ts"].tolist()))
                       for n, f in fam_data.items()}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
