"""Printed-material card from the precomputed bead-cell grid (rve/data/grid.json): milliseconds per query.

    from rve.api import printed_card
    r = printed_card("ABS-M30-iso", layup="cross", raster_deg=45, h=0.2, w=0.45, b_int=0.3, b_lat=0.08)

Inputs
  material   a registry id (failure/data/generic_material_registry.json: E, nu, sigma_uts) or a dict with E_MPa,
             nu and optionally sigma_uts_MPa
  layup      "aligned" (every layer along raster_deg, the 0/0 cell) or "cross" (raster_deg / raster_deg - 90,
             the 0/90 cell); raster_deg is the angle of the first layer from the part x axis about the build axis
  geometry   filament pitch w, layer height h (mm) and the bond widths: b_int (interlayer weld width) and b_lat
             (height of the weld to the neighbour in the layer). Porosity is an OUTPUT. Give porosity instead of
             b_int only with bond="from_porosity": b_int is then solved at the given b_lat / h, and the answer is
             flagged, because many bead shapes share one porosity and differ by up to ~40 % in E3 and G13
  weld       weld_strength_MPa (T; default: sigma_uts * weld_efficiency, efficiency default 1 and flagged
             assumed) and Gc_N_per_mm (optional: only checks the strength regime)

Outputs (dict, JSON-ready)
  C_MPa        6x6 stiffness in the part frame (x, y in plane, z build), Voigt 11 22 33 23 13 12, engineering shear
  C_ply_MPa    the same in the layup frame (raster_deg = 0)
  engineering  E1 E2 E3 G23 G13 G12 nu12 nu13 nu23 in the part frame; E_theta: in-plane modulus every 5 deg
  porosity
  strength     bond strengths in the ply frame (MPa): Y (in-layer bond), Z (interlayer), S12, S13, S23, and
               X = filament (sigma_uts) when known. Net-section values, valid when the weld's cohesive length
               E Gc / T^2 is much longer than the bond (every polymer weld measured so far; the FFM run of
               rve.strength reproduces them to 2-4 %); outside that regime they are flagged as upper bounds
  card         {"rve": ..., "modes": {"F", "T", "L"}} for failure/aniso.py
  accuracy     interpolation error estimate (linear vs cubic) per constant, the mesh correction applied, flags

The grid spans h / w 0.3-0.75, b_int / w 0.3-0.9, b_lat / h 0.15-0.7 (notch angle 5 deg); queries outside it are
refused, not extrapolated.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from rve.beads import section_porosity
from rve.homog import engineering, rotate_z

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "data", "grid.json")
REGISTRY = os.path.join(HERE, "..", "failure", "data", "generic_material_registry.json")
FFM_FACTOR = {"aligned": 8.98 / 9.16, "cross": 5.28 / 5.49}   # FFM / net section, Table 8 cell (README)


@lru_cache(maxsize=1)
def _grid():
    with open(GRID) as f:
        g = json.load(f)
    ax = tuple(np.array(g["axes"][k], float) for k in ("h", "b_int", "beta"))
    interp = {}
    for kind in ("linear", "cubic"):
        c00 = {float(nu): RegularGridInterpolator(ax, np.array(C), method=kind) for nu, C in g["c00"].items()}
        c090 = RegularGridInterpolator(ax, np.array(g["c090_g1"]), method=kind)
        interp[kind] = (c00, c090)
    # grade-1 -> grade-2 mesh correction of the 0/90 cell, trilinear over the box corners
    corr = g.get("c090_corr")
    corr_i = None
    if corr:
        cax = tuple(np.array(sorted(set(p[i] for p in corr["points"]))) for i in range(3))
        R = np.ones((2, 2, 2, 6, 6))
        for p, r in zip(corr["points"], corr["ratio"]):
            R[tuple(int(np.searchsorted(cax[i], p[i])) for i in range(3))] = r
        corr_i = RegularGridInterpolator(cax, R)
    por = RegularGridInterpolator(ax, np.array(g["porosity"]), method="cubic")
    return g, ax, interp, corr_i, por


def material(m):
    if isinstance(m, dict):
        return dict(id=m.get("id", "custom"), E_MPa=float(m["E_MPa"]), nu=float(m["nu"]),
                    sigma_uts_MPa=m.get("sigma_uts_MPa"))
    with open(REGISTRY) as f:
        reg = {r["id"]: r for r in json.load(f)}
    if m not in reg:
        raise ValueError("unknown material %r; one of %s" % (m, sorted(reg)))
    r = reg[m]
    if not r.get("E_MPa"):
        raise ValueError("material %r has no stiffness in the registry" % m)
    return dict(id=m, E_MPa=r["E_MPa"], nu=r["nu"], sigma_uts_MPa=(r.get("tcd") or {}).get("sigma_uts_MPa"))


def _c_over_e(layup, x, nu, kind):
    g, ax, interp, corr_i, _ = _grid()
    c00, c090 = interp[kind]
    nus = sorted(c00)
    if not nus[0] - 1e-9 <= nu <= nus[-1] + 1e-9:
        raise ValueError("nu %.3f outside the grid's %.2f-%.2f" % (nu, nus[0], nus[-1]))
    # piecewise-linear in nu on the 0/0 cells
    i = min(int(np.searchsorted(nus, nu, side="right")) - 1, len(nus) - 2)
    t = (nu - nus[i]) / (nus[i + 1] - nus[i])
    a00 = (1 - t) * c00[nus[i]]([x])[0] + t * c00[nus[i + 1]]([x])[0]
    if layup == "aligned":
        return a00
    a = c090([x])[0]
    if corr_i is not None:
        a = a * corr_i([x])[0]
    # nu dependence of the 0/90 cell (computed at 0.35) taken from the 0/0 cell, entry by entry
    b00 = c00[0.35]([x])[0] if 0.35 in c00 else a00
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.abs(b00) > 1e-9, a00 / b00, 1.0)
    return a * f


def _sym_avg(C, layup):
    """A cross-ply cell rotated by 90 deg is the same layup: average out the small mesh asymmetry."""
    return 0.5 * (C + rotate_z(C, 90.0)) if layup == "cross" else C


def porosity_of(h, b_int, beta):
    """Exact pore fraction of the bead section (w = 1); the same for the 0/0 and the 0/90 cell."""
    g = _grid()[0]
    return float(section_porosity(1.0, h, b_int, beta * h, g["alpha"], g["wc"], n=2000))


def printed_card(material_id, layup="cross", raster_deg=0.0, w=None, h=None, b_int=None, b_lat=None,
                 porosity=None, bond="given", weld_strength_MPa=None, weld_efficiency=None, Gc_N_per_mm=None):
    if layup not in ("aligned", "cross"):
        raise ValueError("layup is 'aligned' or 'cross'")
    mat = material(material_id)
    g, ax, *_ = _grid()
    flags = []
    if w is None or h is None or b_lat is None:
        raise ValueError("give the filament pitch w, the layer height h and the lateral bond height b_lat (mm)")
    hn, beta = h / w, b_lat / h
    if bond == "from_porosity":
        if porosity is None:
            raise ValueError("bond='from_porosity' needs porosity")
        f = lambda b: porosity_of(hn, b, beta) - porosity  # noqa: E731
        lo, hi = ax[1][0], ax[1][-1]
        if f(lo) * f(hi) > 0:
            raise ValueError("porosity %.3f not reachable at h/w %.3f, b_lat/h %.3f in the grid" % (porosity, hn, beta))
        b_int = brentq(f, lo, hi) * w
        flags.append("b_int solved from porosity: out-of-plane stiffness and Z depend on the bond width, which "
                     "porosity does not fix; measure it for design values")
    elif b_int is None:
        raise ValueError("give b_int (interlayer bond width, mm), or porosity with bond='from_porosity'")
    bn = b_int / w
    x = (hn, bn, beta)
    for v, a, name in zip(x, ax, ("h/w", "b_int/w", "b_lat/h")):
        if not a[0] - 1e-9 <= v <= a[-1] + 1e-9:
            raise ValueError("%s = %.3f outside the grid %.2f-%.2f" % (name, v, a[0], a[-1]))
    E, nu = mat["E_MPa"], mat["nu"]
    Cp = _sym_avg(E * _c_over_e(layup, x, nu, "linear"), layup)
    Cc = _sym_avg(E * _c_over_e(layup, x, nu, "cubic"), layup)
    if layup == "cross" and abs(nu - 0.35) > 0.005:
        flags.append("0/90 cell computed at nu 0.35; nu %.2f applied through the 0/0 cell's nu dependence" % nu)
    C = rotate_z(Cp, raster_deg)
    e, ec = engineering(C), engineering(rotate_z(Cc, raster_deg))
    err = {k: abs(e[k] - ec[k]) / abs(e[k]) for k in e}
    th = np.arange(0, 181, 5)
    S = np.linalg.inv(C)
    E_theta = [1.0 / (np.array([math.cos(math.radians(t)) ** 2, math.sin(math.radians(t)) ** 2, 0, 0, 0,
                                math.cos(math.radians(t)) * math.sin(math.radians(t))])
                      @ S @ np.array([math.cos(math.radians(t)) ** 2, math.sin(math.radians(t)) ** 2, 0, 0, 0,
                                      math.cos(math.radians(t)) * math.sin(math.radians(t))])) for t in th]
    phi = porosity_of(hn, bn, beta)

    # bond strengths (net section, ply frame)
    if weld_strength_MPa is None:
        if mat["sigma_uts_MPa"] is None:
            raise ValueError("give weld_strength_MPa (no sigma_uts for this material)")
        eta = 1.0 if weld_efficiency is None else weld_efficiency
        if weld_efficiency is None:
            flags.append("weld strength assumed = sigma_uts (efficiency 1); real welds are 0.3-0.9 of it")
        T = eta * mat["sigma_uts_MPa"]
    else:
        T = weld_strength_MPa
    k = FFM_FACTOR[layup]
    f_int = bn ** 2 if layup == "cross" else bn         # interlayer bond area / cell section
    Y, Z = k * T * beta, k * T * f_int
    st = dict(Y=Y if layup == "aligned" else None, Z=Z, S13=Z, S23=min(Z, Y) if layup == "aligned" else Z,
              S12=Y if layup == "aligned" else None, X=mat["sigma_uts_MPa"], T_weld=T)
    if layup == "cross":
        flags.append("cross-ply in-plane strength is filament-controlled (the FFM run gives bond strengths of "
                     "~3 T in plane); use the F mode per ply")
    regime = None
    if Gc_N_per_mm is not None:
        l_ch = E * Gc_N_per_mm / T ** 2
        half = 0.5 * min(b_int, b_lat)
        regime = dict(l_ch_mm=l_ch, half_bond_mm=half, net_section_valid=l_ch > 5 * half)
        if l_ch <= 5 * half:
            flags.append("brittle weld (l_ch %.3g mm vs half bond %.3g mm): net-section strengths are upper "
                         "bounds; run python3 -m rve.strength for the FFM values" % (l_ch, half))
    else:
        flags.append("no Gc given: net-section regime assumed (l_ch >> bond)")
    Yt = T * beta * FFM_FACTOR["aligned"]                # T mode always from the aligned in-layer bond
    modes = {"T": {"S0_MPa": Yt, "shear_ratio": 1.0}, "L": {"S0_MPa": Z, "shear_ratio": 1.0}}
    if mat["sigma_uts_MPa"]:
        modes["F"] = {"S0_MPa": mat["sigma_uts_MPa"]}
    card = {"rve": {"material": mat["id"], "layup": layup, "raster_deg": raster_deg,
                    "geometry_mm": dict(w=w, h=h, b_int=b_int, b_lat=b_lat), "porosity": phi,
                    "C_ply_MPa": Cp.tolist(), "weld_T_MPa": T, "source": "rve.api grid (net-section bond strength)"},
            "modes": modes}
    return dict(material=mat, layup=layup, raster_deg=raster_deg, porosity=phi,
                geometry=dict(w=w, h=h, b_int=b_int, b_lat=b_lat, h_over_w=hn, b_int_over_w=bn, b_lat_over_h=beta),
                C_MPa=C.tolist(), C_ply_MPa=Cp.tolist(), engineering=e,
                E_theta=dict(theta_deg=th.tolist(), E_MPa=E_theta), strength=st, regime=regime, card=card,
                accuracy=dict(interpolation_rel=err, max_interpolation_rel=max(err.values()),
                              mesh="0/0 refine 1 grade 2; 0/90 grade 1 with the grade-2 corner correction"),
                flags=flags)
