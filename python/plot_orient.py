"""Figures of the print-orientation failure layer (python/failure/aniso.py, prob.py).

  results/fig_orient_validation.{pdf,png}: (a) PLA one card for all rasters, leave-one-family-out prediction vs
      measured, with the 90 % band; (b) measured / predicted on Weibull paper against the scatter models;
      (c) Cicero ABS, one per-mode TCD card for three layups vs per-layup isotropic TCD; (d) off-axis strength of
      a +-theta PETG laminate (Popa et al. 2025): mode model vs Tsai-Hill vs measured.
  results/fig_orient_map.{pdf,png}: mean failure load of the PLA US specimen against build direction (equal-area
      hemisphere) for the demonstration layer-bond card, relative to the flat print, governing mode marked.
  results/failure/orient_US_{flat,upright}.vtu: Gauss-point hazard density per mode (ParaView).

usage (from python/; after failure.pla_orient and failure.cicero_orient): python3 plot_orient.py [--dirs 600]
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from failure import aniso as A  # noqa: E402
from failure import pla_orient as PO  # noqa: E402
from failure import prob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # dataviz reference palette, fixed order
FAMS = ["UB", "UI", "US", "OB", "OI", "OS"]
MARK = dict(zip(FAMS, "osD^vP"))


def style(ax, title):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def validation(out):
    r = json.load(open(os.path.join(RES, "failure", "pla_orient.json")))
    e = r["models"]["tcd-line-modes"]
    obs = np.array([x["P"] for x in r["rows"]]) / 1000
    pred = np.array(e["lofo_predicted_N"]) / 1000
    fam = np.array([x["family"] for x in r["rows"]])
    sc = prob.Scatter(r["m_s"], e["lofo_bias"], e["lofo_tau"])
    phys = prob.Scatter(r["m_s"])
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.4))

    a = ax[0, 0]
    lo, hi = 2.3, 4.0
    t = np.linspace(lo, hi, 2)
    a.plot(t, t, color=MUTED, lw=1)
    a.fill_between(t, t * sc.load_at(0.05, 1.0), t * sc.load_at(0.95, 1.0), color=GRID, alpha=0.6, lw=0,
                   label="90 % band (scatter + model)")
    for i, f in enumerate(FAMS):
        k = fam == f
        a.scatter(pred[k], obs[k], s=22, marker=MARK[f], color=C[i], edgecolor="white", lw=0.6, label=f, zorder=3)
    a.set_xlim(lo, hi), a.set_ylim(lo, hi)
    a.set_xlabel("predicted mean failure load (kN), family left out", fontsize=8)
    a.set_ylabel("measured failure load (kN)", fontsize=8)
    a.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper left")
    style(a, f"(a) PLA, one TCD card (line, F+T) for 0/30/45: LOFO {e['lofo_RMSE_percent']:.1f} %")

    b = ax[0, 1]
    z = np.sort(np.log(obs / pred))
    n = len(z)
    pp = (np.arange(1, n + 1) - 0.3) / (n + 0.4)                          # Benard median ranks
    b.scatter(z, np.log(-np.log1p(-pp)), s=10, color=INK, label="measured / predicted", zorder=3)
    x = np.linspace(-0.35, 0.25, 200)
    for s_, lab, col in ((phys, f"physical scatter only (m_s = {r['m_s']:.1f})", C[0]),
                         (sc, f"+ model error (tau = {e['lofo_tau']:.3f})", C[1])):
        p = s_.cdf(np.exp(x), 1.0)
        b.plot(x, np.log(-np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))), color=col, lw=2, label=lab)
    b.set_ylim(-5.5, 2)
    b.set_xlabel("ln(measured / predicted)", fontsize=8)
    b.set_ylabel("ln(-ln(1 - P))", fontsize=8)
    b.legend(fontsize=6.5, frameon=False, loc="upper left")
    style(b, f"(b) 90 % band covers {100 * e['coverage90_scatter_model']:.0f} % "
             f"(scatter only {100 * e['coverage90_scatter_only']:.0f} %)")

    c = ax[1, 0]
    cic = json.load(open(os.path.join(RES, "failure", "cicero_orient.json")))
    rows = cic["rows"]
    labels = [f"{x['layup']}\n{x['rho']:g}" for x in rows]
    xs = np.arange(len(rows))
    err = lambda k: [100 * (x[k] / x["measured_N"] - 1) for x in rows]
    for j, (k, lab, col, mk) in enumerate((("modes_N", "one per-mode card", C[0], "o"),
                                           ("modes_loo_N", "same, layup left out", C[1], "s"),
                                           ("iso_cal_rho1_N", "isotropic, sigma_0 per layup on rho = 1", C[2], "D"))):
        c.plot(xs + (j - 1) * 0.22, err(k), mk, color=col, ms=5, mec="white", mew=0.6, label=lab, ls="none")
    c.axhline(0, color=MUTED, lw=1)
    for v in (3.5, 7.5):
        c.axvline(v, color=GRID, lw=1)
    c.set_xticks(xs, [f"{x['rho']:g}" for x in rows], fontsize=7)
    for j, name in enumerate(dict.fromkeys(x["layup"] for x in rows)):
        c.text(4 * j + 1.5, 1.02, name, transform=c.get_xaxis_transform(), ha="center", fontsize=7, color=MUTED)
    c.set_ylabel("predicted / measured - 1 (%)", fontsize=8)
    c.set_xlabel("notch root radius (mm), by layup", fontsize=8)
    c.legend(fontsize=6.5, frameon=False, loc="lower left")
    s = cic["summary"]
    style(c, f"(c) ABS SENB, TCD: one card {s['modes_N']['rms_pct']:.1f} % rms, LOO {s['modes_loo_N']['rms_pct']:.1f} %\n")

    d = ax[1, 1]
    X, Y, S12 = 55.21, 48.89, 26.83
    th = np.linspace(0, 90, 181)
    mode = []
    for t_ in th:
        R = A.ply_frames([0, 0, 1], [t_, -t_])[0]
        cc, _ = A.components(np.diag([1.0, 0, 0])[None], R)
        mode.append(1 / max(np.max(A.mode_from_components("F", cc, X / S12)) / X,
                            np.max(A.mode_from_components("T", cc, Y / S12)) / Y))
    tr = np.radians(th)
    tsai = (np.cos(tr) ** 4 / X ** 2 + (1 / S12 ** 2 - 1 / X ** 2) * np.sin(tr) ** 2 * np.cos(tr) ** 2
            + np.sin(tr) ** 4 / Y ** 2) ** -0.5
    d.plot(th, mode, color=C[0], lw=2, label="modes F + T, Hashin shear terms (first ply)")
    d.plot(th, tsai, color=C[1], lw=2, ls="--", label="Tsai-Hill (paper's inputs)")
    d.scatter([0, 45, 90], [X, 48.62, Y], s=30, color=INK, zorder=3, label="measured (Popa 2025, PETG)")
    d.set_xlabel("raster angle to the load, +-theta (deg)", fontsize=8)
    d.set_ylabel("tensile strength (MPa)", fontsize=8)
    d.set_ylim(30, 60)
    d.legend(fontsize=6.5, frameon=False, loc="lower left")
    style(d, "(d) PETG, 45 deg: modes -3.2 %, Tsai-Hill +0.6 %")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)


def orientation_map(out, n_dirs, h=1.5):
    f = PO.field("US", h, os.path.join(ROOT, "cache", "failure"))
    card = A.load_card("pla-as2018-orient-demo-L")["weibull"]
    pr = A.Print(angles_deg=(0.0, -90.0), ref_dir=(0, 1, 0))
    dirs = A.fibonacci_hemisphere(n_dirs)
    special = {"flat (d = z)": (0, 0, 1), "on edge (d = x)": (1, 0, 0), "upright (d = y, load)": (0, 1, 0)}
    cache = os.path.join(RES, "failure", f"orient_map_US_{n_dirs}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        eta, gov = z["eta"], list(z["gov"])
    else:
        eta, gov = A.sweep(f["sig"], f["w"], card, pr, np.vstack([dirs, list(special.values())]), f["grad"])
        np.savez(cache, eta=eta, gov=np.array(gov))
    ref = eta[-3]
    rel = eta[:n_dirs] / ref
    k = np.sqrt(2.0 / (1.0 + dirs[:, 2]))                               # Lambert azimuthal equal-area about +z
    x, y = k * dirs[:, 0], k * dirs[:, 1]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    tc = ax.tricontourf(x, y, rel, levels=np.linspace(np.floor(rel.min() * 20) / 20, np.ceil(rel.max() * 20) / 20, 13),
                        cmap="Blues")
    Lgov = np.array([g == "L" for g in gov[:n_dirs]], float)
    ax.tricontour(x, y, Lgov, levels=[0.5], colors=INK, linewidths=1.0)
    ax.text(0.0, -1.62, f"layer bond (L) governs {100 * Lgov.mean():.0f} % of directions (outside the line)",
            ha="center", fontsize=7, color=INK)
    ax.add_patch(plt.Circle((0, 0), math.sqrt(2), fill=False, color=MUTED, lw=1))
    offs = {"flat (d = z)": (8, -16), "on edge (d = x)": (-10, 14), "upright (d = y, load)": (8, 2)}
    for (lab, v), e_ in zip(special.items(), eta[-3:]):
        v = np.asarray(v, float)
        kk = math.sqrt(2.0 / (1.0 + v[2]))
        ax.plot(kk * v[0], kk * v[1], "o", color=INK, ms=5, mec="white")
        ax.annotate(f"{lab}: {e_ / ref:.2f}", (kk * v[0], kk * v[1]), textcoords="offset points", xytext=offs[lab],
                    fontsize=7, color=INK, ha="right" if v[0] > 0.5 else "left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(tc, ax=ax, shrink=0.8, ticks=np.arange(0.5, 1.11, 0.1))
    cb.set_label("mean failure load / flat print", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("PLA US notched plate, 0/90 layup, by build direction (equal-area)\n"
                 "DEMO card: layer-bond strength assumed (Z = 0.55 Y)\n", fontsize=8, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    return {lab: float(e_ / ref) for lab, e_ in zip(special, eta[-3:])}, f, card, pr


def vtu(f, card, pr):
    import meshio
    for name, d in (("flat", (0, 0, 1)), ("upright", (0, 1, 0))):
        p = A.Print(build_dir=d, angles_deg=pr.angles_deg, ref_dir=pr.ref_dir)
        R = A.ply_frames(p.build_dir, p.plies()[0], p.ref_dir)[0]
        c, g = A.components(f["sig"], R, f["grad"])
        data, m = {}, card["m"]
        for k, q in card["modes"].items():
            s = A.mode_from_components(k, c, q.get("shear_ratio", 1.0))
            s = s / A.support_factor(s, A.mode_gradient_from_components(k, c, g, q.get("shear_ratio", 1.0)),
                                     q["support_mm"])
            dens = p.plies()[1] @ (s / q["S0_MPa"]) ** m                 # hazard density per mm^3 at 1 N
            data[f"log10_hazard_{k}"] = np.log10(np.maximum(dens, 1e-300))
        data["governing"] = np.argmax(np.vstack([data[f"log10_hazard_{k}"] for k in card["modes"]]), axis=0).astype(float)
        n = len(f["P"])
        meshio.write(os.path.join(RES, "failure", f"orient_US_{name}.vtu"),
                     meshio.Mesh(f["P"], [("vertex", np.arange(n)[:, None])], point_data=data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", type=int, default=1500)
    a = ap.parse_args()
    validation(os.path.join(RES, "fig_orient_validation"))
    rel, f, card, pr = orientation_map(os.path.join(RES, "fig_orient_map"), a.dirs)
    vtu(f, card, pr)
    print("orientation map, relative to the flat print:", {k: round(v, 3) for k, v in rel.items()})


if __name__ == "__main__":
    main()
