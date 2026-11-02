"""Bead window (app/bead_window, src/bead_window.hpp): bead-resolved patch of a print with driven cut faces.

A window is nx x ny cells of the 0/90 bead pieces (python/rve/beads.py bead090; every piece a rigid copy of one
condensed operator), periodic in z (the build direction of a thick part), cut in x and y. The cut faces carry a
prescribed displacement:
  strain     u = E x                    kinematic uniform BC (what a homogenized part solution gives as is)
  corrector  u = E x + chi(x) E         chi the periodic cell's fluctuation at the same dof: the consistent drive
                                        (first-order two-scale field), the window then reproduces the periodic cell
Weld laws as rve.strength.weld_args (Kn = kappa E / w); in the window the lateral welds are internal pairs.

Gate G2 (check): the weld-family net tractions of the centre cell against the periodic bead cell, per unit macro
strain, for kappa 1-1000 and both drives; per-cell errors show the cut-face boundary layer.

usage (from python/): python3 -m rve.window [--cells 3] [--refine 1 --grade 0] [--out dir]
  runtime: ~3 min (3 x 3 cells, 30k unknowns, 4 kappas x 2 drives; operator cached after the first run)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve import strength  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
BEAD_WINDOW = os.environ.get("IGABEM_BEAD_WINDOW", os.path.join(ROOT, "build", "bead_window"))
BASE = dict(w=0.6, h=0.3, b_int=0.36, b_lat=0.10, alpha=15.0, wc=math.sqrt(0.5))
KAPPAS = (1.0, 10.0, 100.0, 1000.0)
FAMILIES = ("latA", "latB", "int")


def pieces(g, refine, grade, work):
    """bead090 A and B files of geometry g (unchanged files keep their mtime: cached operators stay valid)."""
    os.makedirs(work, exist_ok=True)
    files = [os.path.join(work, "piece_r%dg%d_%s.dat" % (refine, grade, t)) for t in "AB"]
    for f, P in zip(files, strength._bead_patches("090", dict(g, L=g["h"]), refine, grade)):
        strength._write_if_changed(f, P, "bead cell")
    return files


def law_args(kappa, E, nu, w, window):
    """Spring laws at kappa (None: perfect). Window: lateral welds internal (x: B, y: A), interlayer internal A-B and
    periodic B-A; cell: rve.strength.weld_args."""
    if kappa is None:
        return []
    if not window:
        return strength.weld_args("090", kappa, E, nu, w)
    Kn = kappa * E / w
    law = "%.10g:%.10g" % (Kn, Kn * (1 - 2 * nu) / (2 * (1 - nu)))
    return ["--weld-int", "x:" + law, "--weld-int", "y:" + law, "--weld-int", "z:" + law, "--weld", "z:" + law]


def run_window(files, cells, kappas, drive, work, E=2200.0, nu=0.35, w=0.6, tag="win"):
    """Tractions (BeadWindow::tractions rows) per kappa: one bead_window run, one factorization per kappa."""
    tfile = os.path.join(work, "%s_%s_%dx%d.txt" % (tag, drive, cells[0], cells[1]))
    springs = any(k is not None for k in kappas)
    cmd = [BEAD_WINDOW] + files + ["--cells", "%d:%d" % cells, "--pitch", str(w), "--E", str(E), "--nu", str(nu),
                                   "--drive", drive, "--cache", os.path.join(work, "cache"), "--tractions", tfile,
                                   "--json", tfile + ".json"]
    if springs:
        cmd += law_args(1.0, E, nu, w, True) + ["--scales", ",".join("0" if k is None else "%.17g" % (1 / k) for k in kappas)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    names = [tfile + ".%d" % q for q in range(len(kappas))] if len(kappas) > 1 else [tfile]
    with open(tfile + ".json") as f:
        info = json.load(f)
    return [np.loadtxt(n) for n in names], info


def cell_tractions(files, kappas, work, E=2200.0, nu=0.35, w=0.6):
    """Periodic bead cell tractions (BeadCell::weldTractions rows) per kappa."""
    tfile = os.path.join(work, "cell_tractions.txt")
    cmd = [strength.BEAD_CELL] + files + ["--axes", "x,y", "--copy", "1:0:90", "--E", str(E), "--nu", str(nu),
                                          "--cache", os.path.join(work, "cache"), "--tractions", tfile]
    if any(k is not None for k in kappas):
        cmd += law_args(1.0, E, nu, w, False) + ["--scales", ",".join("0" if k is None else "%.17g" % (1 / k) for k in kappas)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    names = [tfile + ".%d" % q for q in range(len(kappas))] if len(kappas) > 1 else [tfile]
    return [np.loadtxt(n) for n in names]


def _means(W, t, masks):
    return {k: np.einsum("p,pjc->jc", W[m], t[m]) / W[m].sum() if m.any() else np.full((6, 3), np.nan)
            for k, m in masks.items()}   # nan: that face of the cell is a cut


def cell_means(T):
    """Net traction (6 unit strains x 3) per family of the periodic cell, on the + face (normal +e_axis)."""
    ax, it = T[:, 0].astype(int), T[:, 1].astype(int)
    return _means(T[:, 5], T[:, 9:].reshape(-1, 6, 3), {"latA": (ax == 1) & (it == 0), "latB": (ax == 0) & (it == 0),
                                                       "int": ax == 2})


def window_means(T, ny, i, j):
    """The same for window cell (i, j): its A piece's lateral + face (y), its B piece's (x), its two squares."""
    kind, ax, reg = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2].astype(int)
    A, B = 2 * (i * ny + j), 2 * (i * ny + j) + 1
    return _means(T[:, 6], T[:, 10:].reshape(-1, 6, 3),
                  {"latA": (kind == 1) & (ax == 1) & (reg == A), "latB": (kind == 1) & (ax == 0) & (reg == B),
                   "int": (ax == 2) & (((kind == 1) & (reg == A)) | ((kind == 0) & (reg == B)))})


def error(win, ref):
    """Per family: max over the 6 x 3 net tractions of |window - cell| / max |cell| (percent)."""
    return {k: float(100 * np.abs(win[k] - ref[k]).max() / np.abs(ref[k]).max()) for k in ref}


def check(cells=3, refine=1, grade=0, work="/tmp/rve_window", kappas=KAPPAS):
    """Gate G2: per kappa and drive, the error of every window cell that has both lateral + faces inside (i < nx - 1,
    j < ny - 1 for latA / latB; every cell for int) against the periodic cell."""
    files = pieces(BASE, refine, grade, work)
    t0 = time.time()
    refs = cell_tractions(files, kappas, work)
    out = dict(cells=[cells, cells], refine=refine, grade=grade, kappas=list(kappas), drives={})
    for drive in ("strain", "corrector"):
        Ts, info = run_window(files, (cells, cells), kappas, drive, work)
        rows = []
        for k, T, Tr in zip(kappas, Ts, refs):
            ref = cell_means(Tr)
            per = [[error(window_means(T, cells, i, j), ref) for j in range(cells)] for i in range(cells)]
            rows.append(dict(kappa=k, centre=per[cells // 2][cells // 2], cells=per))
        out["drives"][drive] = dict(rows=rows, runs=info["runs"])
    out["t_total"] = time.time() - t0
    return out


def figure(res, out):
    from rve.figures import INK, INK2, S1, S2, S3, plt, save
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    fig.subplots_adjust(wspace=0.35)
    ax = axs[0]
    ks = res["kappas"]
    cols = dict(zip(FAMILIES, (S1, S2, S3)))
    for drive, ls, mk in (("strain", "-", "o"), ("corrector", "--", "s")):
        for f in FAMILIES:
            if f == "latB":
                continue  # = latA by the 90 deg symmetry of the 0/90 cell
            e = [r["centre"][f] for r in res["drives"][drive]["rows"]]
            ax.loglog(ks, e, ls, marker=mk, color=cols[f], ms=4, lw=1.2,
                      label="%s, %s" % ("lateral" if f == "latA" else "interlayer", "$u = Ex$" if drive == "strain" else "$u = Ex + \\chi E$"))
    ax.axhline(2.0, color=INK2, lw=0.8, ls=":")
    ax.text(ks[0], 2.1, "gate 2 %", color=INK2, fontsize=7, va="bottom")
    ax.set_xlabel("weld stiffness  $\\kappa = K_n w / E$")
    ax.set_ylabel("centre-cell weld traction error (%)")
    ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    ax = axs[1]
    n = res["cells"][0]
    row = res["drives"]["strain"]["rows"][0]
    E = np.array([[row["cells"][i][j]["int"] for j in range(n)] for i in range(n)])
    im = ax.imshow(E.T, origin="lower", cmap="Blues", vmin=0, extent=[-0.5, n - 0.5, -0.5, n - 0.5])
    for i in range(n):
        for j in range(n):
            ax.text(i, j, "%.1f" % E[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if E[i, j] > 0.6 * E.max() else INK)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(False)
    ax.set_xlabel("cell i (x, along the A filaments)")
    ax.set_ylabel("cell j (y)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="interlayer error (%)")
    ax.set_title("$u = Ex$, $\\kappa$ = %g" % row["kappa"], fontsize=8, color=INK2)
    for a_, t in zip(axs, "ab"):
        a_.set_title("(%s)" % t, loc="left", fontsize=8.5, color=INK)
    save(fig, out, "fig_window_g2")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--grade", type=int, default=0)
    ap.add_argument("--work", default="/tmp/rve_window")
    ap.add_argument("--out", default=None, help="directory for window_g2.json and fig_window_g2.*")
    a = ap.parse_args(argv)
    res = check(a.cells, a.refine, a.grade, a.work)
    for drive, d in res["drives"].items():
        for r in d["rows"]:
            print("%-9s kappa %-6g centre: %s" % (drive, r["kappa"], "  ".join("%s %.2f%%" % kv for kv in r["centre"].items())))
        for run in d["runs"]:
            print("   %d unknowns, %d GMRES iterations, factor %.1f s, solve %.1f s" % (
                run["unknowns"], run["iterations"], run["t_factor"], run["t_solve"]))
    print("total %.0f s" % res["t_total"])
    if a.out:
        with open(os.path.join(a.out, "window_g2.json"), "w") as f:
            json.dump(res, f, indent=1)
        figure(res, a.out)
    return res


if __name__ == "__main__":
    main()
