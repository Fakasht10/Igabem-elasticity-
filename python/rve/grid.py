"""Precomputed bead-cell stiffness on a grid of dimensionless geometries (the response surface behind rve.api).

Geometry, in units of the filament pitch w = 1: layer height h, interlayer bond width b_int, lateral bond height
b_lat = beta h. Notch half-angle 5 deg (a near-cusp neck, as printed beads are; at 5 deg every grid point is a
valid convex bead, at 15 deg 14 of the 100 are not). The stiffness of a cell is linear in E at fixed nu, so the grid
stores C / E for a few nu.

  0/0   all grid points, nu in NU_00, refine 1 grade 2 (0.2 % from refine 3 on the Table 8 cell), ~3.5 s each
  0/90  all grid points, nu = 0.35, refine 1 grade 1 (~28 s each), plus the eight box corners at grade 2: the
        grade-1 to grade-2 ratio of every C entry (E3, G13 move ~3 %) is interpolated over the box and applied

Each point is cached in the work dir (rerunning skips finished points): the cell .dat, its C .json and, for the
0.35 cells, the six boundary solutions .sol (rve_cell --save-solutions), so interior fields (strain concentration,
data for a deep material network) need no re-solve. grid.json also carries the descriptor vector xi of every point.
Output: rve/data/grid.json.

usage (from python/): python3 -m rve.grid [--work /tmp/rve_grid] [--only 00|090|corr]
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve.beads import section_porosity  # noqa: E402
from rve.strength import RVE_CELL, build  # noqa: E402

H = (0.3, 0.4, 0.5, 0.6, 0.75)
B_INT = (0.3, 0.45, 0.6, 0.75, 0.9)
BETA = (0.15, 0.3, 0.5, 0.7)
NU_00 = (0.30, 0.35, 0.40)
NU_090 = 0.35
ALPHA = 5.0
WC = math.sqrt(0.5)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "grid.json")


def geom(h, bi, beta):
    return dict(w=1.0, h=h, b_int=bi, b_lat=beta * h, alpha=ALPHA, wc=WC, L=h)


def tag(kind, h, bi, beta, nu, grade):
    return "%s_h%g_b%g_l%g_nu%g_g%d" % (kind, h, bi, beta, nu, grade)


def point(kind, h, bi, beta, nu, grade, work, save=False):
    """PBC solve of one grid cell (E = 1), cached in work: <tag>.dat (the cell), <tag>.json (C), and with save
    <tag>.sol, the six unit-strain boundary solutions (rve_cell --save-solutions) for later interior fields."""
    t = tag(kind, h, bi, beta, nu, grade)
    dat, js, sol = (os.path.join(work, t + e) for e in (".dat", ".json", ".sol"))
    done = os.path.exists(js) and os.path.getsize(js) > 0 and (not save or os.path.exists(sol))
    if not done:
        build(kind, geom(h, bi, beta), 1, grade, dat)
        cmd = [RVE_CELL, dat, "--bc", "pbc", "--ng", "12", "--E", "1", "--nu", str(nu), "--json", js]
        subprocess.run(cmd + (["--save-solutions", sol] if save else []), check=True, stdout=subprocess.DEVNULL)
    with open(js) as f:
        r = json.load(f)
    return np.array(r["bc"]["pbc"]["C"]), r["porosity"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/tmp/rve_grid")
    ap.add_argument("--only", choices=["00", "090", "corr"], default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    pts = list(itertools.product(H, B_INT, BETA))
    shape = (len(H), len(B_INT), len(BETA))
    t0 = time.time()
    out = dict(axes=dict(h=H, b_int=B_INT, beta=BETA), alpha=ALPHA, wc=WC, units="C / E (w = 1)",
               convention="voigt_eng_shear (11 22 33 23 13 12)",
               xi=dict(names=["w", "h", "b_int", "b_lat", "alpha_deg", "porosity"],
                       units=["mm (= 1)", "w", "w", "w", "deg", "-"],
                       order="C-order over axes h, b_int, beta (b_lat = beta h); arrays [ih][ib][il]",
                       values=[[1.0, h, bi, be * h, ALPHA, section_porosity(1.0, h, bi, be * h, ALPHA, WC)]
                               for h, bi, be in pts]),
               solutions=dict(files="<tag>.dat / <tag>.sol, tag = %s_h%g_b%g_l%g_nu%g_g%d (kind 00 or 090)",
                              saved="0/0 at nu 0.35 grade 2; 0/90 at nu 0.35 grade 1 (and the grade-2 corners)",
                              read="app/rve_cell <tag>.dat --load-solutions <tag>.sol --points p.txt --json out.json",
                              location="/mnt/project-files/igabem-rve/grid_solutions/"),
               porosity=np.array([section_porosity(1.0, h, bi, be * h, ALPHA, WC) for h, bi, be in pts])
               .reshape(shape).tolist())
    if a.only in (None, "00"):
        C00 = {}
        for nu in NU_00:
            C00[nu] = np.array([point("00", h, bi, be, nu, 2, a.work, save=nu == NU_090)[0]
                             for h, bi, be in pts]).reshape(shape + (6, 6))
            print("0/0 nu %.2f done, %.0f s" % (nu, time.time() - t0), flush=True)
        out["c00"] = {"%.2f" % nu: C.tolist() for nu, C in C00.items()}
    if a.only in (None, "090"):
        C1 = np.array([point("090", h, bi, be, NU_090, 1, a.work, save=True)[0]
                       for h, bi, be in pts]).reshape(shape + (6, 6))
        print("0/90 grade 1 done, %.0f s" % (time.time() - t0), flush=True)
        out["c090_g1"] = C1.tolist()
    if a.only in (None, "corr"):
        corners = list(itertools.product((H[0], H[-1]), (B_INT[0], B_INT[-1]), (BETA[0], BETA[-1])))
        ratio = []
        for h, bi, be in corners:
            c1 = point("090", h, bi, be, NU_090, 1, a.work, save=True)[0]
            c2 = point("090", h, bi, be, NU_090, 2, a.work, save=True)[0]
            ratio.append(np.where(np.abs(c1) > 1e-9 * np.abs(c1).max(), c2 / np.where(c1 == 0, 1, c1), 1.0))
        out["c090_corr"] = dict(points=corners, ratio=np.array(ratio).tolist())
        print("0/90 grade-2 corners done, %.0f s" % (time.time() - t0), flush=True)
    old = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            old = json.load(f)
    old.update(out)
    with open(OUT, "w") as f:
        json.dump(old, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
