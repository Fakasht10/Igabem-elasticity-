"""Canned probe plots from the CSVs written by src/probes.hpp (plate_test --fields).

Library functions (load, cylindrical components, one-panel plot) plus two CLIs:

  plot_probes.py line <probe.csv> --comp s_xx,s_yy [--x s|x|y|z|r|theta] [--frame cart|cyl]
                 [--scale 1] [--xlabel ...] --out fig.png
      any probe, any components (s_ij stress, e_ij strain, u_i displacement,
      von_mises; cylindrical frame about the z axis gives s_rr, s_tt, s_rt, s_rz,
      s_tz, ...); the exact field is overlaid when the CSV has it

  plot_probes.py plate <prefix> --a 0.2 --t 0.2 --out fig.png
      the four plate panels: ligament, around the hole, 45-degree bisector,
      up the hole wall
"""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
M = ["o", "s", "^", "D"]
SYM = ["xx", "yy", "zz", "xy", "xz", "yz"]
IJ = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
TEX = {"rr": "rr", "tt": r"\theta\theta", "rt": r"r\theta", "rz": "rz", "tz": r"\theta z", "zz": "zz",
       "xx": "xx", "yy": "yy", "xy": "xy", "xz": "xz", "yz": "yz"}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {k: [float(r[k]) if r[k] != "" else math.nan for r in rows] for k in rows[0]}
    d["r"] = [math.hypot(x, y) for x, y in zip(d["x"], d["y"])]
    d["theta"] = [math.degrees(math.atan2(y, x)) for x, y in zip(d["x"], d["y"])]
    return d


def tensor(d, i, prefix="s_", suffix=""):
    out = [[0.0] * 3 for _ in range(3)]
    for s, (a, b) in IJ.items():
        out[a][b] = out[b][a] = d[prefix + s + suffix][i]
    return out


def add_cylindrical(d):
    """Adds s_rr, s_tt, s_rt, s_rz, s_tz (and _ex, e_*) about the z axis."""
    for prefix, suffix in (("s_", ""), ("s_", "_ex"), ("e_", "")):
        if prefix + "xx" + suffix not in d:
            continue
        cols = {k: [] for k in ("rr", "tt", "rt", "rz", "tz")}
        for i in range(len(d["x"])):
            T = tensor(d, i, prefix, suffix)
            th = math.atan2(d["y"][i], d["x"][i])
            er, et = (math.cos(th), math.sin(th), 0.0), (-math.sin(th), math.cos(th), 0.0)
            ez = (0.0, 0.0, 1.0)
            q = lambda a, b: sum(a[p] * T[p][k] * b[k] for p in range(3) for k in range(3))
            cols["rr"].append(q(er, er))
            cols["tt"].append(q(et, et))
            cols["rt"].append(q(er, et))
            cols["rz"].append(q(er, ez))
            cols["tz"].append(q(et, ez))
        for k, v in cols.items():
            d[prefix + k + suffix] = v
    return d


def label(comp):
    if comp == "von_mises":
        return r"$\sigma_{\mathrm{vM}}$"
    kind, ij = comp.split("_", 1)
    sym = {"s": r"\sigma", "e": r"\varepsilon", "u": "u"}[kind]
    return "$%s_{%s}$" % (sym, TEX.get(ij, ij))


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def panel(ax, d, comps, x="s", xscale=1.0, scale=1.0, every=4):
    """BEM as lines with sparse markers, exact field as thin dashed black lines."""
    X = [v / xscale for v in d[x]]
    for i, c in enumerate(comps):
        ax.plot(X, [v / scale for v in d[c]], color=C[i % 4], lw=1.5, marker=M[i % 4], ms=4,
                markevery=every, label=label(c))
        if c + "_ex" in d:
            ax.plot(X, [v / scale for v in d[c + "_ex"]], color="0.15", lw=0.8, ls="--",
                    label="exact" if i == 0 else None)
    ax.legend(frameon=False, fontsize=8)


def plate(prefix, a, t, out, title=""):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 6.6), constrained_layout=True)
    S = r"$\,/\,S$"

    d = load(prefix + "_ligament.csv")
    panel(axs[0, 0], d, ["s_xx", "s_yy", "s_zz"], x="y", xscale=a)
    axs[0, 0].set_xlabel(r"$y/a$ (ligament $x = 0$, $z = 0$)")
    axs[0, 0].set_ylabel("stress" + S)
    style(axs[0, 0], "(a) ligament through the solid")

    ax = axs[0, 1]
    for i, (name, lab) in enumerate((("mid", "$z = 0$"), ("face", "$z = t/2$"))):
        d = add_cylindrical(load(prefix + "_hole_%s.csv" % name))
        ax.plot(d["theta"], d["s_tt"], color=C[i], lw=1.5, marker=M[i], ms=4, markevery=6,
                label=r"$\sigma_{\theta\theta}$, " + lab)
        if "s_tt_ex" in d and i == 0:
            ax.plot(d["theta"], d["s_tt_ex"], color="0.15", lw=0.8, ls="--", label="exact")
    ax.set_xlabel(r"$\theta$ [deg] around the hole")
    ax.set_ylabel(r"$\sigma_{\theta\theta}$" + S)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.legend(frameon=False, fontsize=8)
    style(ax, "(b) hoop stress on the hole wall")

    d = add_cylindrical(load(prefix + "_bisector.csv"))
    panel(axs[1, 0], d, ["s_rr", "s_tt", "s_rt"], x="r", xscale=a)
    axs[1, 0].set_xlabel(r"$r/a$ (bisector $\theta = 45^\circ$, $z = 0$)")
    axs[1, 0].set_ylabel("stress" + S)
    style(axs[1, 0], "(c) bisector, cylindrical components")

    ax = axs[1, 1]
    d = load(prefix + "_hole_z.csv")
    zt = [z / t for z in d["z"]]
    for i, c in enumerate(["s_xx", "s_zz"]):
        ax.plot(zt, d[c], color=C[i], lw=1.5, marker=M[i], ms=4, markevery=4, label=label(c))
        if c + "_ex" in d:
            ax.plot(zt, d[c + "_ex"], color="0.15", lw=0.8, ls="--", label="exact" if i == 0 else None)
    ax.set_xlabel(r"$z/t$ (hole wall at $\theta = 90^\circ$)")
    ax.set_ylabel("stress" + S)
    ax.set_xlim(-0.5, 0.5)
    ax.legend(frameon=False, fontsize=8)
    style(ax, "(d) through the thickness at the hole edge")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=200)
    if out.endswith(".png"):
        fig.savefig(out[:-4] + ".pdf")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("line")
    pl.add_argument("csv")
    pl.add_argument("--comp", required=True)
    pl.add_argument("--x", default="s")
    pl.add_argument("--frame", default="cart", choices=["cart", "cyl"])
    pl.add_argument("--scale", type=float, default=1.0)
    pl.add_argument("--xlabel", default=None)
    pl.add_argument("--title", default="")
    pl.add_argument("--out", required=True)
    pp = sub.add_parser("plate")
    pp.add_argument("prefix")
    pp.add_argument("--a", type=float, default=0.2)
    pp.add_argument("--t", type=float, default=0.2)
    pp.add_argument("--title", default="")
    pp.add_argument("--out", required=True)
    o = ap.parse_args()
    if o.cmd == "plate":
        plate(o.prefix, o.a, o.t, o.out, o.title)
        return
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    d = load(o.csv)
    if o.frame == "cyl":
        add_cylindrical(d)
    fig, ax = plt.subplots(figsize=(4.4, 3.3), constrained_layout=True)
    panel(ax, d, o.comp.split(","), x=o.x, scale=o.scale)
    ax.set_xlabel(o.xlabel or o.x)
    style(ax, o.title or os.path.basename(o.csv))
    fig.savefig(o.out, dpi=200)


if __name__ == "__main__":
    main()
