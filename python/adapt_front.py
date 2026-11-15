"""Adaptive front grading driven by the hypersingular residual (crack_prism --indicator).

Loop on the edge-cracked prism with the r^3/2 term (prism_a3): solve, take the TBIE residual indicator
eta_e^2 = h_e int_e |r|^2 of every lower-face element, sum it per band of distance from the front, mark bands by
Doerfler (the smallest set holding THETA of the total), split each marked band (r0, r1) by a knot line at
r0 + (r1 - r0) / 2 (at r1 q for the band touching the front: geometric toward the singularity), regenerate the mesh
(make_crack.py --front-lines) and repeat until mean |K_I - 1| <= TOL or MAX_IT steps. All front-adjacent patches
(crack faces, the split z faces) get the same lines.

usage: adapt_front.py [--k 2] [--theta 0.5] [--q 0.3] [--tol 1e-3] [--max-it 10] [--out results/xigabem/prism_a3/adapt]
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")


def bands(path):
    B = collections.defaultdict(float)
    with open(path) as f:
        for r in csv.DictReader(f):
            B[(round(float(r["r0"]), 10), round(float(r["r1"]), 10))] += float(r["eta2"])
    return B


def mark(B, theta):
    tot = sum(B.values())
    acc, out = 0.0, []
    for k, v in sorted(B.items(), key=lambda kv: -kv[1]):
        out.append(k)
        acc += v
        if acc >= theta * tot:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--max-it", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "xigabem", "prism_a3", "adapt"))
    ap.add_argument("--geo", default=os.path.join(ROOT, "geo", "crack"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    geo = a.geo
    lines, hist = [], []
    for it in range(a.max_it + 1):
        t0 = time.time()
        args = [sys.executable, os.path.join(ROOT, "python", "make_crack.py"), "prism", "--lateral", "split",
                "--n", str(a.k), "--suffix", "_a%d" % it, "--out", geo]
        if lines:
            args += ["--front-lines", ",".join("%.12g" % d for d in sorted(lines))]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
        dat = os.path.join(geo, "prism_split_k%d_a%d.dat" % (a.k, it))
        tag = os.path.join(a.out, "it%d" % it)
        r = subprocess.run([os.path.join(BUILD, "crack_prism"), dat, "--a3", "1", "--indicator", "--out", tag],
                           capture_output=True, text=True, timeout=a.timeout)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        B = bands(tag + "_indicator.csv")
        h = {"it": it, "dof": d["unknowns"], "KI_err": d["KI_err"], "eta": d["eta"], "energy_err": d["energy_err"],
             "effectivity": d["effectivity"], "lines": sorted(lines), "t": time.time() - t0}
        hist.append(h)
        print("it %2d  dof %5d  K_I err %.3e  eta %.3e  eff %.2f  %.1f s" % (it, h["dof"], h["KI_err"], h["eta"],
                                                                         h["effectivity"], h["t"]))
        if d["KI_err"] <= a.tol:
            break
        for r0, r1 in mark(B, a.theta):
            lines.append(r1 * a.q if r0 <= 1e-12 else 0.5 * (r0 + r1))
    with open(os.path.join(a.out, "adapt.json"), "w") as f:
        json.dump({"k": a.k, "theta": a.theta, "q": a.q, "tol": a.tol, "history": hist}, f, indent=1)
    return 0 if hist[-1]["KI_err"] <= a.tol else 1


if __name__ == "__main__":
    sys.exit(main())
