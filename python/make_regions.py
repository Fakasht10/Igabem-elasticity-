"""Two-region geometries (host shell + enclosed inclusion) for src/multi_region.hpp.

Each body is a pair of .dat files in geo/regions/:
  <name>_host.dat   outer surface (outward) + interface patches transposed in u and v,
                    so their normals point out of the host, into the inclusion
  <name>_core.dat   the same interface patches, oriented outward from the inclusion

  sph2_a0.5_r{k}    sphere of radius 1 around a concentric sphere of radius a = 0.5
                    (Bembel's exact rational quartic six-patch sphere, both refined k
                    times): two-layer Lame sphere and Goodier/Eshelby inclusion
  coupon_w{w}_n{n}  printed coupon [0, 4] x [0, 1] x [0, 1], solid walls of thickness w on
                    all six sides around a box core; cubic B-splines, n elements per unit
                    length on the outer box, the same per-axis breaks (shifted) on the core

usage: python3 python/make_regions.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_boxes import box  # noqa: E402
from make_geometries import Patch, read_dat, refine_uniform, reversed_patch, write_dat  # noqa: E402

GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo")
OUT = os.path.join(GEO, "regions")


def scaled(P, s, c=(0.0, 0.0, 0.0)):
    Q = Patch(P.p, P.q, P.U, P.V, P.Pw.copy())
    Q.set_euclid(np.asarray(c) + s * Q.euclid())  # affine: weights unchanged
    return Q


def write_pair(name, outer, core):
    write_dat(os.path.join(OUT, name + "_host.dat"), list(outer) + [reversed_patch(P) for P in core], name + "_host")
    write_dat(os.path.join(OUT, name + "_core.dat"), list(core), name + "_core")
    print("wrote geo/regions/%s_{host,core}.dat (%d + %d patches)" % (name, len(outer), len(core)))


def main():
    os.makedirs(OUT, exist_ok=True)
    base = read_dat(os.path.join(GEO, "bembel_sphere.dat"))
    for k in range(4):
        S = [Patch(P.p, P.q, P.U, P.V, P.Pw.copy()) for P in base]
        for _ in range(k):
            S = [refine_uniform(P) for P in S]
        write_pair("sph2_a0.5_r%d" % k, S, [scaled(P, 0.5) for P in S])
    L, W, H = 4.0, 1.0, 1.0
    for w in (0.15,):
        for n in (2, 4, 6):
            br = [np.linspace(0, d, int(round(n * d)) + 1) for d in (L, W, H)]
            outer = box([0, 0, 0], [L, W, H], br, 3)
            core = box([w, w, w], [L - w, W - w, H - w], br, 3)
            write_pair("coupon_w%g_n%d" % (w, n), outer, core)


if __name__ == "__main__":
    main()
