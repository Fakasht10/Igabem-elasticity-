"""Blind transfer: plain-specimen fatigue strength -> notched specimens, every series matched on print angle p and
load ratio R (so no mean-stress model enters), every criterion of fatigue.criteria, on cached IgBEM fields.

    python3 -m fatigue.transfer [--data fatigue/data/ezeh_susmel.json] [--h 1.5] [--gauge 15 30 50] [--out DIR]

Criteria and what each one is allowed to know:
    kt               nothing but the elastic field (Kf = Kt)                                   blind
    weibull          m from the plain series' own scatter ratio T; plain gauge volume W T L_g  blind (L_g assumed)
    weibullpool      as weibull, m pooled over every plain series (sample-size weighted)      blind (L_g assumed)
    tcd_point/line   L from the dataset's own TCD calibration (tcd_reference), if it has one   fitted on notches
Runtime: one IgBEM solve per notch geometry (~1.5 min each at h 1.5, cached in cache/fatigue), criteria in < 1 s.
"""
import argparse
import json
import math
import os

import numpy as np

from failure.igabem_backend import ROOT

from . import criteria as C
from .fields import field
from .specimens import Specimen

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "fatigue")


def plain_lookup(data):
    return {(r["p"], r["R"]): r for r in data["plain"]}


def walker_plain(data, base_R=-1.0):
    """Plain table as if only base_R had been tested: gamma per angle from the full plain table, then every other R
    mapped from the base_R series by Walker (its scatter ratio T kept). Returns (plain rows, {p: fit})."""
    fits, rows = {}, []
    for p in sorted({r["p"] for r in data["plain"]}):
        ser = [r for r in data["plain"] if r["p"] == p]
        g, s_ar, rms = C.walker_fit([(r["R"], r["sigma_A50"]) for r in ser])
        base = next(r for r in ser if r["R"] == base_R)
        s_r1 = base["sigma_A50"] * (2 / (1 - base_R)) ** (1 - g)        # base series -> fully reversed
        fits[p] = {"gamma": g, "s_ar_fit": s_ar, "rms_log": rms}
        rows += [dict(base, R=R, sigma_A50=C.walker_amplitude(s_r1, R, g), walker_from=base_R)
                 for R in sorted({r["R"] for r in ser})]
    return rows, fits


def predict(data, fields, gauges, log=print):
    plain = plain_lookup(data)
    ps = data["plain_specimen"]
    tcd = data.get("tcd_reference")
    L = tcd["A_mm"] * data["N_ref"] ** tcd["b"] if tcd else None
    m_pool = C.m_pooled([(r["T"], r["n"]) for r in data["plain"]])
    rows = []
    for r in data["notched"]:
        pr = plain.get((r["p"], r["R"]))
        if pr is None:
            log(f"no plain series for p {r['p']} R {r['R']}: skipped")
            continue
        f, s0 = fields[r["notch"]], pr["sigma_A50"]
        m = C.m_from_T(pr["T"])
        row = dict(r, s0=s0, m=m, Kt=float(np.asarray(f["s1_line"])[:, 0].mean()))
        row["kt"] = C.kt(f, s0)
        for g in gauges:
            row[f"weibull_Lg{g:g}"] = C.weibull(f, s0, m, ps["W"] * ps["T"] * g)
            row[f"weibullpool_Lg{g:g}"] = C.weibull(f, s0, m_pool, ps["W"] * ps["T"] * g)
        row["m_pool"] = m_pool
        if L:
            row["L_mm"] = L
            row["tcd_point"] = C.tcd_point(f, s0, L)
            row["tcd_line"] = C.tcd_line(f, s0, L)
        rows.append(row)
    return rows


def scores(rows, keys):
    out = {}
    for k in keys:
        e = np.log(np.array([r[k] / r["sigma_nA50"] for r in rows]))
        out[k] = {"mean_log_ratio": float(e.mean()), "rms_log_ratio": float(np.sqrt((e ** 2).mean())),
                  "max_abs_err_pct": float(100 * np.max(np.abs(np.expm1(e)))),
                  "within_20pct": int(np.sum(np.abs(np.expm1(e)) <= 0.2)), "n": int(len(e))}
    return out


def table_md(rows, keys):
    head = "| notch | p | R | plain s0 | Kt | m | measured | " + " | ".join(keys) + " |"
    lines = [head, "|" + " --- |" * (7 + len(keys))]
    for r in rows:
        lines.append(f"| {r['notch']} | {r['p']} | {r['R']:g} | {r['s0']:.1f} | {r['Kt']:.2f} | {r['m']:.1f} | "
                     f"{r['sigma_nA50']:.1f} | " + " | ".join(f"{r[k]:.1f} ({100 * (r[k] / r['sigma_nA50'] - 1):+.0f}%)"
                                                         for k in keys) + " |")
    return "\n".join(lines)


def figure(rows, keys, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm"})
    marker = {"blunt": "o", "intermediate": "s", "sharp": "^"}
    colour = {0: "#1f77b4", 30: "#d62728", 45: "#2ca02c"}
    fig, axs = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.0), sharey=True)
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    lo, hi, TICKS = 0.7, 20.0, [1, 2, 5, 10, 20]
    for ax, k in zip(np.atleast_1d(axs), keys):
        x = np.geomspace(lo, hi, 2)
        ax.fill_between(x, x / 1.2, x * 1.2, color="0.9", lw=0, label=r"$\pm 20\%$")
        ax.plot(x, x, "k-", lw=0.8)
        for r in rows:
            ax.scatter(r["sigma_nA50"], r[k], marker=marker[r["notch"]], s=26, edgecolor=colour[r["p"]],
                       facecolor=colour[r["p"]] if r["R"] == -1 else "white", lw=1.0, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(TICKS)); axis.set_major_formatter(FormatStrFormatter("%g"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"measured $\sigma_{nA,50\%}$ (MPa)")
        ax.set_title(labels[k], fontsize=9)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    np.atleast_1d(axs)[0].set_ylabel(r"predicted $\sigma_{nA,50\%}$ (MPa)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=marker[n], mfc="0.4", mec="0.4", label=n) for n in marker]
    h += [Line2D([], [], ls="", marker="o", mfc=c, mec=c, label=f"p = {p}°") for p, c in colour.items()]
    h += [Line2D([], [], ls="", marker="o", mfc="white", mec="k", label="R = 0 (open)")]
    fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "ezeh_susmel.json"))
    ap.add_argument("--h", type=float, default=1.5)
    ap.add_argument("--gauge", type=float, nargs="*", default=[15.0, 30.0, 60.0],
                    help="assumed plain gauge lengths (mm) for the Weibull volume")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--walker", type=float, metavar="R", help="calibrate on the plain series at this R only and map "
                    "to the other R by a Walker exponent fitted per angle on the full plain table")
    a = ap.parse_args(argv)
    data = json.load(open(a.data))
    fields = {k: field(Specimen.from_dict(v), h=a.h) for k, v in data["specimens"].items()}
    walker = None
    if a.walker is not None:
        data["plain"], walker = walker_plain(data, a.walker)
        for p, w in walker.items():
            print(f"Walker p {p}: gamma {w['gamma']:.3f}, rms log residual of the plain fit {w['rms_log']:.3f}")
    rows = predict(data, fields, a.gauge)
    keys = ["kt"] + [f"weibull_Lg{g:g}" for g in a.gauge] + [f"weibullpool_Lg{g:g}" for g in a.gauge] + (["tcd_point", "tcd_line"] if data.get("tcd_reference") else [])
    sc = scores(rows, keys)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{data['id']}_h{a.h:g}" + (f"_walker{a.walker:g}" if a.walker is not None else "")
    geo = {k: {"Kt_mean": float(f["Kt_mean"]), "Kt_mid": float(f["Kt_mid"]), "unknowns": int(f["unknowns"])}
           for k, f in fields.items()}
    json.dump({"data": a.data, "h": a.h, "walker": walker, "geometry": geo, "scores": sc, "rows": rows},
              open(os.path.join(a.out, f"transfer_{tag}.json"), "w"), indent=1)
    md = table_md(rows, keys)
    open(os.path.join(a.out, f"transfer_{tag}.md"), "w").write(md + "\n")
    print(md)
    for k, v in sc.items():
        print(f"{k:16s} mean {100 * math.expm1(v['mean_log_ratio']):+6.1f}%  rms log {v['rms_log_ratio']:.3f}  "
              f"max |err| {v['max_abs_err_pct']:5.1f}%  within 20%: {v['within_20pct']}/{v['n']}")
    g = a.gauge[len(a.gauge) // 2]
    mid, pool = f"weibull_Lg{g:g}", f"weibullpool_Lg{g:g}"
    labels = {"kt": r"$K_f = K_t$ (blind)", mid: f"Weibull, m per series\n($L_g$ = {g:g} mm, blind)",
              pool: f"Weibull, pooled m = {rows[0]['m_pool']:.1f}\n($L_g$ = {g:g} mm, blind)",
              "tcd_point": "TCD point, authors' $L_M$\n(L fitted on notches)"}
    figure(rows, [k for k in ("kt", mid, pool, "tcd_point") if k in keys], labels,
           os.path.join(a.out, f"fig_transfer_{tag}.png"))


if __name__ == "__main__":
    main()
