"""Print-resolved weakest link for FDM fatigue: hazards summed over print regions and bead-scale failure modes.

    python3 -m fatigue.printed [--data ...] [--gauge 30] [--contour 0.4] [--out results/fatigue]

A printed plate is two regions:
    contour  the perimeter bead(s), t_c deep from every lateral free edge (notch included), filament along the edge
    infill   rasters at the layup angles of the data set (Ezeh-Susmel: +-45 deg to the plate axes), rotated in plane by
             the manufacturing angle p; top/bottom skins print the same rasters and count as infill
Each region has the ply frame e1 = filament, e2 = in-plane transverse, e3 = build direction z, and the tension-only
Hashin-form modes of failure.aniso (same definitions):
    F  filament       <s11>
    T  filament bond  sqrt(<s22>^2 + r^2 s12^2)
The layer-bond mode (s33, interlayer shear) is off: there is no build-tilt data to give it a strength.

    P_f = 1 - exp(-H),  H = sum_regions sum_k int (sigma_k / S_k)^m dV          (one exponent m per load ratio)

Everything is linear in the load, so a specimen's median strength (H = ln 2) is s = (ln 2 / J)^(1/m) with J the
unit-load hazard, and a notched prediction is s_n = s0 (J_plain / J_notch)^(1/m) for any plain series s0.

The quarter model's stresses hold in all four quadrants, but a raster angle phi mirrors to -phi in the two quadrants
reflected about one axis: J_infill = 2 (J(S) + J(-S)) for the ply set S. The contour frame follows the edge, so it
mirrors with the geometry (4 J). The plain coupon is uniform: contour volume share 2 t_c / W (the two side walls).

Calibration, per load ratio, on the plain coupons only (one per angle p): S_T and the shear ratio r; S_F is not
identifiable from plain coupons (the infill T mode governs them) and is set from the static PLA card's S_F / S_T
(failure/data/orientation_cards.json, pla-as2018-orient) for the blind run, or fitted on two notches and scored on
the third (leave one notch out).
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.optimize import minimize

from failure.aniso import load_card

from . import criteria as C
from .fields import field
from .specimens import Specimen, outline
from .transfer import HERE, OUT

LN2 = math.log(2.0)
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def tensor(v):
    """(n, 6) Voigt (xx, yy, zz, yz, xz, xy) -> (n, 3, 3)."""
    v = np.asarray(v, float)
    t = np.empty((len(v), 3, 3))
    for k, (i, j) in enumerate(_VOIGT):
        t[:, i, j] = t[:, j, i] = v[:, k]
    return t


# ------------------------------------------------------------------------------------------- geometry
def _free(seg, s):
    """A lateral free edge: not on a symmetry line (x = 0, y = 0) nor the grip (y = L/2)."""
    pts = [seg[1], seg[-1]]
    for axis, val in ((0, 0.0), (1, 0.0), (1, s.L / 2)):
        if all(abs(p[axis] - val) < 1e-9 for p in pts):
            return False
    return True


def edge_distance(s, xy):
    """Distance (n,) from points xy (n, 2) to the nearest lateral free edge, and the unit edge tangent (n, 2) there."""
    best = np.full(len(xy), np.inf)
    tan = np.zeros((len(xy), 2))
    for seg in outline(s):
        if not _free(seg, s):
            continue
        if seg[0] == "line":
            p, q = np.array(seg[1]), np.array(seg[2])
            u = q - p
            t = np.clip((xy - p) @ u / (u @ u), 0.0, 1.0)
            dist = np.linalg.norm(xy - (p + t[:, None] * u), axis=1)
            tg = np.broadcast_to(u / np.linalg.norm(u), xy.shape)
        else:
            p, c, q = (np.array(v) for v in seg[1:])
            r = np.linalg.norm(p - c)
            a0, a1 = (math.atan2(*(v - c)[::-1]) for v in (p, q))
            lo, hi = min(a0, a1), max(a0, a1)
            a = np.clip(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]), lo, hi)
            near = c + r * np.c_[np.cos(a), np.sin(a)]
            dist = np.linalg.norm(xy - near, axis=1)
            tg = np.c_[-np.sin(a), np.cos(a)]
        k = dist < best
        best[k], tan[k] = dist[k], tg[k]
    return best, tan


def _frame_from_dir(e1xy):
    """Ply frames (n, 3, 3), rows e1 (filament), e2 (in-plane transverse), e3 = z."""
    n = len(e1xy)
    R = np.zeros((n, 3, 3))
    R[:, 0, :2] = e1xy
    R[:, 1, :2] = np.c_[-e1xy[:, 1], e1xy[:, 0]]
    R[:, 2, 2] = 1.0
    return R


def ply_dir(phi_deg):
    """Filament direction for a raster angle phi measured from the load axis y towards +x."""
    a = math.radians(phi_deg)
    return np.array([math.sin(a), math.cos(a)])


# ------------------------------------------------------------------------------------------- modes and hazards
def mode_stresses(sig, R):
    """F and T equivalent stresses (n,) in frames R (n, 3, 3) or one frame (3, 3); T without the shear term and the
    shear component separately, so the shear ratio is a cheap parameter: T = sqrt(n22^2 + r^2 s12^2)."""
    R = np.broadcast_to(R, (len(sig), 3, 3))
    loc = np.einsum("nai,nij,nbj->nab", R, sig, R)
    return np.maximum(loc[:, 0, 0], 0.0), np.maximum(loc[:, 1, 1], 0.0), loc[:, 0, 1]


def _logsum(logs):
    logs = np.asarray(logs, float)
    a = logs.max()
    return a + math.log(np.exp(logs - a).sum()) if np.isfinite(a) else -math.inf


def _log_int(s, w, m):
    k = s > 0
    return _logsum(m * np.log(s[k]) + np.log(w[k])) if k.any() else -math.inf


class Notched:
    """Unit-load ply-frame stresses of one cached notch field, per region and raster angle (cheap re-scoring)."""

    def __init__(self, spec, f, t_c):
        sig, w, P = tensor(f["sig"]), np.asarray(f["w"], float), np.asarray(f["P"], float)
        dist, tan = edge_distance(spec, P[:, :2])
        self.contour = dist < t_c
        self.w = w
        c = self.contour
        self.parts = {"contour": (mode_stresses(sig[c], _frame_from_dir(tan[c])), w[c], 4.0)}
        self.sig_in, self.w_in = sig[~c], w[~c]
        self._ply = {}

    def ply(self, phi):
        key = round(phi % 180.0, 9)
        if key not in self._ply:
            R = _frame_from_dir(np.broadcast_to(ply_dir(phi), (len(self.sig_in), 2)))
            self._ply[key] = mode_stresses(self.sig_in, R)
        return self._ply[key]

    def log_hazards(self, plies, m, S_F, S_T, r):
        """{(region, mode): log J} at unit nominal net stress; infill plies share its volume equally and the mirrored
        set -S covers the other two quadrants."""
        out = {}
        (F, N, X), w, mult = self.parts["contour"]
        out[("contour", "F")] = math.log(mult) + _log_int(F, w, m) - m * math.log(S_F)
        out[("contour", "T")] = math.log(mult) + _log_int(np.sqrt(N ** 2 + (r * X) ** 2), w, m) - m * math.log(S_T)
        for mode in ("F", "T"):
            terms = []
            for phi in [*plies, *(-p for p in plies)]:
                F, N, X = self.ply(phi)
                s = F if mode == "F" else np.sqrt(N ** 2 + (r * X) ** 2)
                terms.append(math.log(2.0 / len(plies)) + _log_int(s, self.w_in, m))
            out[("infill", mode)] = _logsum(terms) - m * math.log(S_F if mode == "F" else S_T)
        return out


def plain_log_hazard(plies, m, S_F, S_T, r, V, f_c):
    """log J of a plain coupon of volume V at unit stress along y (contour share f_c, filament along the load)."""
    terms = [math.log(f_c * V) - m * math.log(S_F)] if f_c > 0 else []
    for phi in ([*plies, *(-p for p in plies)] if f_c < 1 else []):
        a = math.radians(phi)
        F, N, X = math.cos(a) ** 2, math.sin(a) ** 2, math.sin(a) * math.cos(a)
        share = (1 - f_c) * V / (2 * len(plies))
        terms += [math.log(share) + m * math.log(F / S_F) if F > 0 else -math.inf,
                  math.log(share) + m * math.log(math.hypot(N, r * X) / S_T) if math.hypot(N, r * X) > 0 else -math.inf]
    return _logsum(terms)


# ------------------------------------------------------------------------------------------- data set
def layup(data, p):
    """Infill raster angles from the load axis for manufacturing angle p (in-plane rotation of the plate)."""
    base = data.get("infill_deg", (45.0, -45.0))
    return tuple(b - p for b in base)


def calibrate_plain(data, R, m, V, f_c, ratio_F):
    """S_T and r from the plain series at load ratio R (all angles); S_F = ratio_F S_T. Returns (S_T, r, rms log)."""
    rows = [r for r in data["plain"] if r["R"] == R]

    def resid(th):
        S_T, r = math.exp(th[0]), math.exp(th[1])
        return [((LN2 / math.exp(plain_log_hazard(layup(data, q["p"]), m, ratio_F * S_T, S_T, r, V, f_c))) ** (1 / m)
                 / q["sigma_A50"]) for q in rows]

    obj = lambda th: float(np.mean(np.log(resid(th)) ** 2))
    best = min((minimize(obj, [math.log(s), math.log(r0)], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12})
                for s in (5.0, 20.0) for r0 in (0.3, 1.0, 3.0)), key=lambda o: o.fun)
    return math.exp(best.x[0]), math.exp(best.x[1]), math.sqrt(best.fun)


def predict(data, notches, gauge, t_c, ratio_F, Rs=None):
    """Median notched nominal amplitude per notched row, for one S_F / S_T ratio. Returns (rows, plain fits)."""
    ps = data["plain_specimen"]
    V, f_c = ps["W"] * ps["T"] * gauge, min(1.0, 2 * t_c / ps["W"])
    Rs = Rs or sorted({r["R"] for r in data["notched"]})
    fits, out = {}, []
    for R in Rs:
        ser = [r for r in data["plain"] if r["R"] == R]
        m = C.m_pooled([(r["T"], r["n"]) for r in ser])
        S_T, r, rms = calibrate_plain(data, R, m, V, f_c, ratio_F)
        fits[R] = dict(m=m, S_T=S_T, S_F=ratio_F * S_T, r=r, plain_rms_log=rms)
        for q in (q for q in data["notched"] if q["R"] == R):
            lh = notches[q["notch"]].log_hazards(layup(data, q["p"]), m, ratio_F * S_T, S_T, r)
            J = _logsum(list(lh.values()))
            share = {f"{a}-{b}": math.exp(v - J) for (a, b), v in lh.items()}
            out.append(dict(notch=q["notch"], p=q["p"], R=R, measured=q["sigma_nA50"],
                            predicted=(LN2 / math.exp(J)) ** (1 / m), shares=share))
    return out, fits


def score(rows, key="predicted", sel=lambda r: True):
    e = np.array([r[key] / r["measured"] - 1 for r in rows if sel(r)])
    return dict(n=len(e), mean=float(e.mean()), rms_log=float(np.sqrt(np.mean(np.log1p(e) ** 2))),
                within20=int((abs(e) < 0.2).sum()))


def fit_ratio(data, notches, gauge, t_c, rows_sel):
    """S_F / S_T minimising the log error on the selected notched rows (1-D, bounded search over a grid + golden)."""
    def err(lr):
        rows, _ = predict(data, notches, gauge, t_c, math.exp(lr))
        return score(rows, sel=rows_sel)["rms_log"]
    grid = np.linspace(math.log(0.5), math.log(8.0), 23)
    k = int(np.argmin([err(g) for g in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    from scipy.optimize import minimize_scalar
    return math.exp(minimize_scalar(err, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3}).x)


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"blunt": "U r3", "intermediate": "U r1", "sharp": "V r0.15"}
    marks = {"blunt": "o", "intermediate": "s", "sharp": "^"}
    cols = {0: "#1f77b4", 30: "#d62728", 45: "#2ca02c"}
    fig, axs = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), sharey=True)
    for ax, (title, rows) in zip(axs, results.items()):
        for r in rows:
            ax.scatter(r["measured"], r["predicted"], marker=marks[r["notch"]], edgecolors=cols[r["p"]],
                       facecolors="none" if r["R"] == 0 else cols[r["p"]], s=46, lw=1.3)
        lim = [1.5, 14]
        ax.plot(lim, lim, "k-", lw=0.8)
        for f in (0.8, 1.2):
            ax.plot(lim, [f * v for v in lim], "k:", lw=0.7)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="measured $\\sigma_{nA,50}$ (MPa)")
        sc = score(rows)
        ax.set_title(f"{title}\nmean {100 * sc['mean']:+.0f}%, rms log {sc['rms_log']:.2f}, "
                     f"{sc['within20']}/{sc['n']} in ±20%", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator([2, 3, 4, 6, 8, 10, 14]))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
    axs[0].set_ylabel("predicted (MPa)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=marks[k], mec="k", mfc="none", label=v) for k, v in names.items()]
    h += [Line2D([], [], ls="", marker="o", c=c, label=f"p = {p}°") for p, c in cols.items()]
    h += [Line2D([], [], ls="", marker="o", mec="k", mfc="k", label="R = −1"),
          Line2D([], [], ls="", marker="o", mec="k", mfc="none", label="R = 0")]
    axs[-1].legend(handles=h, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)
    # which region and mode carries the hazard (blind run, R = 0), per notch and angle
    rows = [r for r in next(iter(results.values())) if r["R"] == 0]
    keys = ["contour-F", "contour-T", "infill-F", "infill-T"]
    kc = {"contour-F": "#08519c", "contour-T": "#6baed6", "infill-F": "#a50f15", "infill-T": "#fb6a4a"}
    rows.sort(key=lambda r: (list(names).index(r["notch"]), r["p"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for k in keys:
        v = np.array([r["shares"].get(k, 0.0) for r in rows])
        ax.bar(x, v, bottom=bottom, color=kc[k], label=k, width=0.75)
        bottom += v
    ax.set_xticks(x, [f"{names[r['notch']]}\np = {r['p']}°" for r in rows], fontsize=7.5)
    ax.set(ylabel="share of the hazard H", ylim=(0, 1), title="Where the failure probability sits (blind, R = 0)")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}_shares.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "ezeh_susmel.json"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--gauge", type=float, default=30.0, help="assumed plain gauge length (mm)")
    ap.add_argument("--contour", type=float, default=None, help="contour depth t_c (mm); default the data set's shell")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    t_c = a.contour if a.contour is not None else data.get("shell_mm", 0.4)
    specs = {k: Specimen.from_dict(v) for k, v in data["specimens"].items()}
    notches = {k: Notched(s, field(s, h=a.h, log=lambda *_: None), t_c) for k, s in specs.items()}
    card = load_card("pla-as2018-orient")["weibull"]["modes"]
    ratio0 = card["F"]["S0_MPa"] / card["T"]["S0_MPa"]
    blind, fits = predict(data, notches, a.gauge, t_c, ratio0)
    lines = [f"# Print-resolved Weibull, {data['id']}: contour {t_c:g} mm + infill {data.get('infill_deg', (45, -45))}",
             "", f"Blind: S_F / S_T = {ratio0:.3f} (static PLA card), gauge {a.gauge:g} mm.", "",
             "| R | m (pooled plain) | S_T (MPa) | S_F (MPa) | shear ratio r | plain fit rms log |", "|---|---|---|---|---|---|"]
    lines += [f"| {R:g} | {v['m']:.1f} | {v['S_T']:.2f} | {v['S_F']:.2f} | {v['r']:.3f} | {v['plain_rms_log']:.3f} |"
              for R, v in fits.items()]
    # leave one notch out on S_F / S_T, and the in-sample fit on all 18
    loo, ratios = [], {}
    for n in specs:
        ratios[n] = fit_ratio(data, notches, a.gauge, t_c, lambda r, n=n: r["notch"] != n)
        rows, _ = predict(data, notches, a.gauge, t_c, ratios[n])
        loo += [r for r in rows if r["notch"] == n]
    r_all = fit_ratio(data, notches, a.gauge, t_c, lambda r: True)
    fitted, _ = predict(data, notches, a.gauge, t_c, r_all)
    lines += ["", "| notch | p | R | measured | blind | LOO (S_F/S_T from the other two notches) | blind hazard shares |",
              "|---|---|---|---|---|---|---|"]
    lo = {(r["notch"], r["p"], r["R"]): r for r in loo}
    for r in sorted(blind, key=lambda r: (r["notch"], r["p"], r["R"])):
        l = lo[(r["notch"], r["p"], r["R"])]
        sh = ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(r["shares"].items(), key=lambda kv: -kv[1]) if v > 0.01)
        lines.append(f"| {r['notch']} | {r['p']} | {r['R']:g} | {r['measured']:.1f} | {r['predicted']:.2f} "
                     f"({100 * (r['predicted'] / r['measured'] - 1):+.0f}%) | {l['predicted']:.2f} "
                     f"({100 * (l['predicted'] / l['measured'] - 1):+.0f}%) | {sh} |")
    lines.append("")
    for name, rows in (("blind", blind), ("leave one notch out", loo), (f"fitted on all 18 (S_F/S_T {r_all:.2f})", fitted)):
        s = score(rows)
        lines.append(f"- {name}: mean {100 * s['mean']:+.1f}%, rms log {s['rms_log']:.3f}, within 20% {s['within20']}/{s['n']}")
    lines.append("- LOO ratios S_F/S_T: " + ", ".join(f"held out {n}: {v:.2f}" for n, v in ratios.items()))
    os.makedirs(a.out, exist_ok=True)
    tag = f"printed_{data['id']}_h{a.h:g}"
    open(os.path.join(a.out, tag + ".md"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(t_c=t_c, ratio_blind=ratio0, fits={str(k): v for k, v in fits.items()}, blind=blind, loo=loo,
                   loo_ratios=ratios, fitted_ratio=r_all), open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    figure({"blind (S_F/S_T from static card)": blind, "leave one notch out": loo}, os.path.join(a.out, "fig_" + tag))
    print("\n".join(lines))
    return blind, loo


if __name__ == "__main__":
    main()
