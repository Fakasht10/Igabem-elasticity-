"""Weld onset at notches: the part stress mapped onto the bead welds of the periodic cell, then TCD along the bisector.

Baseline (first-order, one-way) localization. At every point of an extraction line the part solution gives the macro
stress sigma (homogenized, any frame). Turned into the cell frame of the print (e1 along the A filaments, e3 the build
direction), the cell's stored response gives the mean traction on every bond family by superposition of its six
unit-stress solutions (rve.strength.weld_means, kept in the response library): no cell solve per point, a 6-vector
dot product per family. Families of the 0/90 cell: latA (in-layer bonds of the A layer, normal e2), latB (of the B
layer, normal e1), int (interlayer squares, normal e3); of the 0/0 cell: lat, int.

    tn_f = N_f . sigma_cell,   ts_f = |S_f sigma_cell|              (mean over the bond strip of depth D)

D = the whole bond (default, "net") gives the net-section traction, exact by equilibrium and free of the bond-tip
singularity; a fraction of the half bond gives the FFM stress-part strip mean. The equivalent weld traction
t_f = sqrt(<tn_f>^2 + (r ts_f)^2) (r = T_n / T_s) against the weld strength T_n is a mode in the sense of
failure.aniso (same algebra, so the TCD point / line method applies per mode with one length L). The filament mode F
(ply stress s11 against X) stays as in failure.aniso.

What the cell adds over the ply modes: the load sharing between the crossing layers and the welds (each weld family
sees the whole macro tensor, including s33 and the out-of-plane shears at a notch root in a thick part), and the
geometric concentration lig of every bond from the bead section. The weld shear/normal ratio is still a material
parameter (r), the concentrations are not.

Limits (first order): uniform macro stress over the cell. At a notch root the stress changes by ~ w / rho across one
bead; the second-order correction is the subject of rve.gradient.
"""
import math
import os

import numpy as np

from rve import strength

FAMILIES = {"090": ("latA", "latB", "int"), "00": ("lat", "int")}


def weld_map(kind, g, nu, kappa=None, D="net", refine=1, grade=2, work="/tmp/rve_onset",
             fractions=strength.FRACTIONS):
    """Per family: N (6,) and S (3, 6), mean normal / shear traction on the bond per unit macro stress (cell frame,
    Voigt s11 s22 s33 s23 s13 s12), at weld stiffness kappa (None: perfect bond) and strip depth D ("net": the whole
    bond, else a fraction of the half bond that is one of `fractions`). From the response library (built on a miss:
    one bead_cell run over the weld-stiffness nodes)."""
    os.makedirs(work, exist_ok=True)
    fams = strength.families(kind, g["w"], g["h"], g["b_int"], g["b_lat"], g.get("L", g["h"]))
    Ds_of = lambda F: [fr * F["half"] for fr in fractions] + [F["half"]]  # noqa: E731
    mkey = "fractions=" + ",".join("%g" % fr for fr in fractions)
    r0 = strength.cell_response(kind, dict(g, L=g.get("L", g["h"])), refine, grade, work, "bead_intact", nu,
                                (mkey, lambda T, S: strength.weld_means(T, S, kind, fams, Ds_of)))
    iD = len(fractions) if D == "net" else list(fractions).index(D)
    out = {}
    for fam, vals in r0["means"][mkey].items():
        tn = strength.response_eval(r0, (mkey, fam, 0), kappa, vals[0])     # (nD, 6)
        ts = strength.response_eval(r0, (mkey, fam, 1), kappa, vals[1])     # (nD, 6, 3)
        out[fam] = dict(N=np.asarray(tn)[iD], S=np.asarray(ts)[iD].T, normal=np.asarray(fams[fam]["normal"], float))
    return out


def cell_stress(sig, R):
    """Voigt macro stress (n, 6) in the cell frame from global tensors sig (n, 3, 3); R columns = cell axes."""
    s = np.einsum("ia,jb,nij->nab", R, R, np.asarray(sig, float))
    return np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2], s[:, 1, 2], s[:, 0, 2], s[:, 0, 1]], axis=1)


def weld_tractions(sigV, wmap):
    """{family: (tn (n,), ts (n,))}: mean normal and shear-magnitude bond traction per point."""
    return {f: (sigV @ m["N"], np.linalg.norm(sigV @ m["S"].T, axis=1)) for f, m in wmap.items()}


def equivalent(tn, ts, r):
    """Equivalent weld traction sqrt(<tn>^2 + (r ts)^2) (tension-only normal part, as failure.aniso's modes)."""
    return np.sqrt(np.maximum(tn, 0.0) ** 2 + (r * ts) ** 2)


def cell_frame(theta_deg, build_dir=(0.0, 0.0, 1.0), ref_dir=(1.0, 0.0, 0.0)):
    """Cell axes (3, 3) for a print whose A layer runs at theta from ref_dir about build_dir (failure.aniso frames)."""
    from failure.aniso import ply_frames
    return ply_frames(build_dir, [theta_deg], ref_dir)[0, 0]


def tcd_profile(values, distances, L, method="point"):
    """TCD value of one mode along the line: the point method probes L / 2, the line method averages 0 .. 2L."""
    if method == "point":
        return float(np.interp(0.5 * L, distances, values))
    u = np.linspace(0.0, 2.0 * L, 400)
    return float(np.trapezoid(np.interp(u, distances, values), u)) / (2.0 * L)


def onset(sig_line, distances, wmap, R, T, r, L, X=None, method="point"):
    """Failure-load factor per unit applied load of a print along a bisector line: max over the weld families
    (equivalent traction / T) and, with X, the filament mode of both layers (ply s11 / X). Returns (factor, mode)
    with factor = 1 / max ratio, and the per-mode ratios."""
    sigV = cell_stress(sig_line, R)
    ratios = {}
    for f, (tn, ts) in weld_tractions(sigV, wmap).items():
        ratios[f] = tcd_profile(equivalent(tn, ts, r), distances, L, method) / T
    if X is not None:
        R90 = np.column_stack([R[:, 1], -R[:, 0], R[:, 2]])          # B layer: e1 along the B filaments
        for name, Rk in (("F_A", R), ("F_B", R90)):
            s11 = cell_stress(sig_line, Rk)[:, 0]
            ratios[name] = tcd_profile(np.maximum(s11, 0.0), distances, L, method) / X
    gov = max(ratios, key=ratios.get)
    return 1.0 / ratios[gov], gov, ratios


def lig_check(wmap, kind, g):
    """Net-section factors of the families that alone carry a macro component (equilibrium check): tn per unit of
    that component equals cell section / bond area (rve.strength.families lig)."""
    fams = strength.families(kind, g["w"], g["h"], g["b_int"], g["b_lat"], g.get("L", g["h"]))
    out = {}
    for f, F in fams.items():
        if F["lig"] is not None and F["cuts"]:
            j = F["cuts"][0]
            out[f] = (float(wmap[f]["N"][j]), float(F["lig"]))
    return out


if __name__ == "__main__":
    g = dict(w=0.6, h=0.3, b_int=0.36, b_lat=0.10, alpha=15.0, wc=math.sqrt(0.5))
    wm = weld_map("090", g, 0.35)
    for f, m in wm.items():
        print(f, "N", np.round(m["N"], 3), "|S|", np.round(np.linalg.norm(m["S"], axis=0), 3))
    print("lig check", lig_check(wm, "090", g))
