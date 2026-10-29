"""Axis-aligned boxes as 6-patch B-spline surfaces with graded knot vectors, for the
local-load validation cases (app/bc_validate).

Each axis has one knot vector, shared by every face that spans that axis, so the six
patches are conforming. Control points sit at the Greville abscissae, so the
parametrisation is the identity (x = u): the knots are the element edges.

  box_hs_q_p{p}_r{k}.dat   quarter block [0, 4] x [0, 4] x [-4, 0] (load radius a = 1 at
                           the origin of the top face z = 0): uniform elements h = a / 2^(k+2)
                           over [0, 1.25 a] in x and y (the load circle cuts them anywhere: a
                           tensor-product mesh cannot follow it), then growing by 1.3 to the
                           far faces; in z graded from h at the top, growth 1.3
  bar_p{p}_n{n}.dat        bar [0, L] x [-1/2, 1/2]^2, L = 4: n elements along x, n/4 across
                           (pure bending / spring / cantilever cases; L = 10 for the cantilever)
  plate_t0.2_{uniform,graded}.dat
                           thin plate [0, 3] x [-1/2, 1/2] x [-0.1, 0.1] (the bracket's arm):
                           24 uniform elements along x, or 12 graded from h = 0.025 at the
                           clamp x = 0 (growth 1.4); cases/plate_clamped_*.json

usage: python3 python/make_boxes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import Patch, write_dat  # noqa: E402

GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geo")


def knots(breaks, p):
    """Open knot vector of degree p on the given breakpoints (normalised to [0, 1])."""
    b = np.asarray(breaks, float)
    b = (b - b[0]) / (b[-1] - b[0])
    return np.concatenate([[0.0] * p, b, [1.0] * p])


def greville(U, p):
    n = len(U) - p - 1
    return np.array([np.mean(U[i + 1:i + p + 1]) for i in range(n)])


def uniform_then_graded(length, h, core, growth=1.3):
    """Uniform elements h on [0, core], then geometric growth to length."""
    n = int(round(core / h))
    b = list(np.linspace(0.0, n * h, n + 1))
    rest = graded(length - b[-1], h * growth, growth)
    return np.concatenate([b, b[-1] + rest[1:]])


def graded(length, h0, growth=1.35):
    """Breakpoints on [0, length] starting with h0 at 0, sizes growing geometrically,
    the last one absorbed so the breaks end exactly at length."""
    b, h = [0.0], h0
    while b[-1] + h < length - 0.5 * h:
        b.append(b[-1] + h)
        h *= growth
    b.append(length)
    return np.array(b)


def box(lo, hi, breaks, p):
    """6 conforming patches of [lo, hi] with per-axis breakpoints (physical coordinates)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    K = [knots(breaks[a], p) for a in range(3)]
    G = [lo[a] + (hi[a] - lo[a]) * greville(K[a], p) for a in range(3)]
    out = []
    # (fixed axis, side, axis along u, axis along v) with e_u x e_v = outward normal
    for ax, side in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
        a, b = [k for k in range(3) if k != ax]
        # e_a x e_b = +e_ax for (a, b) cyclic after ax; flip when the outward normal is -e_ax
        cyc = (a, b) == ((ax + 1) % 3, (ax + 2) % 3)
        outward_pos = side == 1
        if cyc != outward_pos:
            a, b = b, a
        X = np.zeros((len(G[b]), len(G[a]), 4))
        X[..., ax] = hi[ax] if side else lo[ax]
        X[..., a] = G[a][None, :]
        X[..., b] = G[b][:, None]
        X[..., 3] = 1.0
        out.append(Patch(p, p, K[a], K[b], X))
    return out


def main():
    os.makedirs(GEO, exist_ok=True)
    made = []
    for p in (3,):
        for k in range(3):
            h = 0.25 / 2 ** k
            br = uniform_then_graded(4.0, h, 1.25)
            bz = graded(4.0, h, 1.3)
            P = box([0, 0, -4], [4, 4, 0], [br, br, 4.0 - bz[::-1]], p)
            f = os.path.join(GEO, "box_hs_q_p%d_r%d.dat" % (p, k))
            write_dat(f, P, os.path.basename(f))
            made.append((f, len(br) - 1))
    for p, n, L in [(3, 8, 4.0), (3, 20, 10.0), (3, 40, 10.0)]:
        m = max(2, n * 1 // int(L))
        P = box([0, -0.5, -0.5], [L, 0.5, 0.5],
                [np.linspace(0, L, n + 1), np.linspace(-0.5, 0.5, m + 1), np.linspace(-0.5, 0.5, m + 1)], p)
        f = os.path.join(GEO, "bar_L%g_p%d_n%d.dat" % (L, p, n))
        write_dat(f, P, os.path.basename(f))
        made.append((f, n))
    for tag, bx in [("uniform", np.linspace(0, 3, 25)), ("graded", graded(3.0, 0.025, 1.4))]:
        P = box([0, -0.5, -0.1], [3, 0.5, 0.1], [bx, np.linspace(-0.5, 0.5, 9), np.linspace(-0.1, 0.1, 3)], 3)
        f = os.path.join(GEO, "plate_t0.2_%s.dat" % tag)
        write_dat(f, P, os.path.basename(f))
        made.append((f, len(bx) - 1))
    for f, n in made:
        print("wrote", os.path.relpath(f), "(%d elements along the first axis)" % n)


if __name__ == "__main__":
    main()
