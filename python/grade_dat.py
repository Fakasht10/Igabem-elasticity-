"""Grade a conforming multipatch surface towards chosen faces by knot insertion (geometry exact).

Why: at a clamped support in bending the transverse shear is carried in the singular corner zone
of the clamp, over about one wall thickness. The faces next to the clamp need elements of a
fraction of the thickness there, or the clamp reaction (and the equilibrium of the solution)
is off by 10-20 % on thin walls (README, "Local supports and loads").

How:
  1. every (patch, parametric direction) is a node; two nodes are linked when the patches share
     a boundary row of control points running in those directions (forward or reversed). Linked
     nodes must get the same knots, or the mesh stops being conforming, so knots are chosen per
     linked class, in a class parameter s in [0, 1] (orientation tracked per node);
  2. a patch whose end edge (u = 0 / u = 1, or v) coincides with an edge of a target patch asks
     for graded knots in that direction: physical sizes h0, h0 g, h0 g^2, ... from that end
     (arc length along the mid iso-line) until they reach the original first span;
  3. with --inner, the target patches are graded too, towards their edges shared with the given
     faces (for a clamp in bending: the faces carrying the bending stress, so the clamp face is
     graded through the wall thickness, where its traction is singular at both edges);
  4. all knots of a class are inserted in every member (Boehm insertion, rational exact).

The patch order is unchanged, so <stem>.faces (python/face_ids.py) is copied to the output.

usage: python3 python/grade_dat.py in.dat out.dat --faces 6[,7] [--patches i,j]
                                   [--inner 1,13] [--h0 0.025] [--growth 1.4]
"""
import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import insert_knot_curve, read_dat, write_dat  # noqa: E402
from spline_checks import evaluate  # noqa: E402


def read_faces(path):
    with open(path) as f:
        toks = [t for l in f if not l.startswith("#") for t in l.split()]
    n = int(toks[0])
    return [int(t) for t in toks[1:1 + n]]


def edges(P):
    """The four boundary rows as (direction they run in, end of the other direction, points)."""
    X = P.euclid()
    return [(0, 0, X[0]), (0, 1, X[-1]), (1, 0, X[:, 0]), (1, 1, X[:, -1])]


class Classes:
    """Union-find over (patch, direction) with an orientation bit (s -> 1 - s)."""

    def __init__(self, n):
        self.parent = {(k, d): (k, d) for k in range(n) for d in (0, 1)}
        self.flip = {key: False for key in self.parent}

    def find(self, a):
        if self.parent[a] == a:
            return a, False
        r, f = self.find(self.parent[a])
        self.parent[a], self.flip[a] = r, self.flip[a] ^ f
        return r, self.flip[a]

    def union(self, a, b, rev):
        (ra, fa), (rb, fb) = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb], self.flip[rb] = ra, fa ^ fb ^ rev


def match(A, B, tol):
    if A.shape != B.shape:
        return None
    if np.abs(A - B).max() < tol:
        return False
    if np.abs(A - B[::-1]).max() < tol:
        return True
    return None


def knot_range(P, d):
    K = P.U if d == 0 else P.V
    return K[0], K[-1], K


def mid_line(P, d, n=400):
    """Cumulative arc length along direction d at the middle of the other direction."""
    a, b, _ = knot_range(P, d)
    c0, c1, _ = knot_range(P, 1 - d)
    t = np.linspace(a, b, n)
    X = np.array([evaluate(P, x, 0.5 * (c0 + c1))[0] if d == 0 else evaluate(P, 0.5 * (c0 + c1), x)[0]
                  for x in t])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    return t, s


def graded_params(P, d, end, h0, g):
    """Parameters of graded knots from `end` of direction d: sizes h0, h0 g, ... (arc length on
    the mid iso-line), as long as they are finer than the knot span they fall in."""
    a, b, K = knot_range(P, d)
    t, s = mid_line(P, d)
    if end == 1:  # measure from the far end
        t, s = t[::-1], s[-1] - s[::-1]
    sk = np.sort(np.interp(np.unique(K), t[::-1] if end else t, s[::-1] if end else s))
    out, x, h = [], 0.0, h0
    while True:
        j = min(np.searchsorted(sk, x + h, side="right"), len(sk) - 1)
        span = sk[j] - sk[j - 1]
        if h * g >= span or x + h >= sk[-1] - 1e-9:
            break
        x += h
        if np.min(np.abs(sk - x)) > 0.5 * h:
            out.append(x)
        h *= g
    return [float(np.interp(v, s, t)) for v in out]  # s increases from `end`


def insert(P, d, params):
    deg = P.p if d == 0 else P.q
    K = P.U if d == 0 else P.V
    for u in sorted(params):
        if np.min(np.abs(K - u)) < 1e-12 * (K[-1] - K[0]):
            continue
        if d == 0:
            rows = [insert_knot_curve(deg, K, P.Pw[j], u) for j in range(P.Pw.shape[0])]
            K = rows[0][0]
            P.Pw = np.stack([r[1] for r in rows], axis=0)
        else:
            cols = [insert_knot_curve(deg, K, P.Pw[:, i], u) for i in range(P.Pw.shape[1])]
            K = cols[0][0]
            P.Pw = np.stack([c[1] for c in cols], axis=1)
    if d == 0:
        P.U = K
    else:
        P.V = K


def grade(patches, targets, h0=0.025, growth=1.4, split=0, inner=(), tol=None):
    if tol is None:
        X = np.vstack([P.euclid().reshape(-1, 3) for P in patches])
        tol = 1e-8 * np.linalg.norm(X.max(0) - X.min(0))
    E = [edges(P) for P in patches]
    cls = Classes(len(patches))
    # 1. link (patch, direction) nodes through shared rows (bucketed by rounded end points)
    key = lambda x: tuple(np.round(x / (100 * tol)).astype(np.int64))  # noqa: E731
    bucket = {}
    for k in range(len(patches)):
        for d, side, R in E[k]:
            bucket.setdefault(frozenset((key(R[0]), key(R[-1]))), []).append((k, d, side, R))
    shared = {}  # (patch, d, side) -> list of (other patch) sharing that row
    for rows in bucket.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (ka, da, sa, Ra), (kb, db, sb, Rb) = rows[i], rows[j]
                rev = match(Ra, Rb, tol)
                if rev is None:
                    continue
                cls.union((ka, da), (kb, db), rev)
                shared.setdefault((ka, da, sa), []).append(kb)
                shared.setdefault((kb, db, sb), []).append(ka)
    tset = set(targets)
    # 2. requests: rows of direction 1 - d at side `end` are the ends of direction d
    want = {}
    for k, P in enumerate(patches):
        for d in (0, 1):
            for end in (0, 1):
                nb = shared.get((k, 1 - d, end), [])
                iset = set(inner)
                if k in tset:  # a target patch: towards its edges on the `inner` faces
                    if not any(o in iset for o in nb):
                        continue
                elif not any(o in tset for o in nb):
                    continue
                a, b, _ = knot_range(P, d)
                root, f = cls.find((k, d))
                for u in graded_params(P, d, end, h0, growth):
                    s = (u - a) / (b - a)
                    want.setdefault(root, []).append(1 - s if f else s)
    # target patches themselves: each knot span split into split + 1 (both directions)
    for k in tset if split else ():
        for d in (0, 1):
            a, b, K = knot_range(patches[k], d)
            root, f = cls.find((k, d))
            uk = (np.unique(K) - a) / (b - a)
            for s0, s1 in zip(uk[:-1], uk[1:]):
                for i in range(1, split + 1):
                    s = s0 + (s1 - s0) * i / (split + 1)
                    want.setdefault(root, []).append(1 - s if f else s)
    # 3. insert every class's knots in all its members
    added = 0
    for k, P in enumerate(patches):
        for d in (0, 1):
            root, f = cls.find((k, d))
            if root not in want:
                continue
            a, b, _ = knot_range(P, d)
            S = np.unique(np.round(np.array(want[root]), 12))
            params = [a + (b - a) * (1 - s if f else s) for s in S]
            n0 = len(P.U if d == 0 else P.V)
            insert(P, d, params)
            added += len(P.U if d == 0 else P.V) - n0
    return patches, added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--faces", default="", help="CAD face ids to grade towards (needs <stem>.faces)")
    ap.add_argument("--patches", default="", help="patch indices to grade towards")
    ap.add_argument("--inner", default="", help="CAD face ids: grade the targets towards their edges on these")
    ap.add_argument("--h0", type=float, default=0.025, help="first element size at the target")
    ap.add_argument("--growth", type=float, default=1.4)
    ap.add_argument("--split", type=int, default=0, help="also split every knot span of the targets into split + 1")
    o = ap.parse_args(argv)
    P = read_dat(o.inp)
    fin = os.path.splitext(o.inp)[0] + ".faces"
    targets = [int(x) for x in o.patches.split(",") if x]
    if o.faces:
        ids = read_faces(fin)
        want = {int(x) for x in o.faces.split(",")}
        targets += [k for k, f in enumerate(ids) if f in want]
    if not targets:
        ap.error("no target patches (--faces needs %s)" % fin)
    inner = []
    if o.inner:
        ids = read_faces(fin)
        want = {int(x) for x in o.inner.split(",")}
        inner = [k for k, f in enumerate(ids) if f in want]
    n0 = sum(p.Pw.shape[0] * p.Pw.shape[1] for p in P)
    P, added = grade(P, targets, o.h0, o.growth, o.split, inner)
    write_dat(o.out, P, os.path.basename(o.out))
    if os.path.exists(fin):
        shutil.copyfile(fin, os.path.splitext(o.out)[0] + ".faces")
    n1 = sum(p.Pw.shape[0] * p.Pw.shape[1] for p in P)
    print("graded %d patches towards %d target patch(es): %d knots inserted, control points %d -> %d; wrote %s"
          % (len(P), len(targets), added, n0, n1, o.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
