"""Light spline utilities shared by the STEP importers (no CAD kernel): evaluation of
.dat patches, watertightness and enclosed-volume checks."""

from collections import defaultdict

import numpy as np
from numpy.polynomial.legendre import leggauss


def _basis(p, U, u):
    """B-spline basis values and first derivatives of all n functions at u."""
    n = len(U) - p - 1
    span = min(max(np.searchsorted(U, u, side="right") - 1, p), n - 1)
    N = np.zeros((p + 1, len(U) - 1))
    N[0, span] = 1.0
    for k in range(1, p + 1):
        for i in range(len(U) - k - 1):
            a = (u - U[i]) / (U[i + k] - U[i]) * N[k - 1, i] if U[i + k] > U[i] else 0.0
            b = (U[i + k + 1] - u) / (U[i + k + 1] - U[i + 1]) * N[k - 1, i + 1] \
                if U[i + k + 1] > U[i + 1] else 0.0
            N[k, i] = a + b
    dN = np.zeros(n)
    for i in range(n):
        if U[i + p] > U[i]:
            dN[i] += p / (U[i + p] - U[i]) * N[p - 1, i]
        if U[i + p + 1] > U[i + 1]:
            dN[i] -= p / (U[i + p + 1] - U[i + 1]) * N[p - 1, i + 1]
    return N[p, :n], dN


def evaluate(P, u, v):
    """Point x and tangents x_u, x_v of patch P at (u, v)."""
    Nu, dNu = _basis(P.p, P.U, u)
    Nv, dNv = _basis(P.q, P.V, v)
    A = np.einsum("j,i,jik->k", Nv, Nu, P.Pw)
    Au = np.einsum("j,i,jik->k", Nv, dNu, P.Pw)
    Av = np.einsum("j,i,jik->k", dNv, Nu, P.Pw)
    x = A[:3] / A[3]
    return x, (Au[:3] - x * Au[3]) / A[3], (Av[:3] - x * Av[3]) / A[3]


def conformity_gap(patches, tol=1e-7):
    """Max mismatch of control points and (relative) weights between boundary rows that
    share both end points; returns (gap, #shared edges, #rows without a partner).
    Rows are bucketed by their rounded end points, so the check is linear in the rows."""
    rows = []
    for k, P in enumerate(patches):
        for r in (P.Pw[0], P.Pw[-1], P.Pw[:, 0], P.Pw[:, -1]):
            rows.append((k, np.hstack([r[:, :3] / r[:, 3:], r[:, 3:] / r[0, 3]])))
    scale = 1.0 / (10 * tol)
    key = lambda x: tuple(np.round(x * scale).astype(np.int64))  # noqa: E731
    bucket = defaultdict(list)
    for i, (k, r) in enumerate(rows):
        bucket[frozenset((key(r[0, :3]), key(r[-1, :3])))].append(i)
    gap, matched, lonely = 0.0, 0, 0
    for i, (k, r) in enumerate(rows):
        partner = None
        for j in bucket[frozenset((key(r[0, :3]), key(r[-1, :3])))]:
            l, s = rows[j]
            if l == k:
                continue
            for t in (s, s[::-1]):
                if np.linalg.norm(t[0, :3] - r[0, :3]) < tol and np.linalg.norm(t[-1, :3] - r[-1, :3]) < tol:
                    partner = t * np.array([1, 1, 1, 1 / t[0, 3]])
            if partner is not None:
                break
        if partner is None:
            lonely += 1
        elif partner.shape != r.shape:
            gap = np.inf
        else:
            gap, matched = max(gap, np.abs(partner - r).max()), matched + 1
    return gap, matched // 2, lonely


def spline_volume(patches, n=10):
    """(1/3) int x . n dA over all patches, tensor Gauss on every knot span."""
    g, gw = leggauss(n)
    vol = 0.0
    for P in patches:
        ku, kv = np.unique(P.U), np.unique(P.V)
        for a, b in zip(ku[:-1], ku[1:]):
            for c, d in zip(kv[:-1], kv[1:]):
                for xi, wi in zip(g, gw):
                    for eta, wj in zip(g, gw):
                        x, xu, xv = evaluate(P, a + (b - a) * (xi + 1) / 2, c + (d - c) * (eta + 1) / 2)
                        vol += wi * wj * (b - a) * (d - c) / 4 * np.dot(x, np.cross(xu, xv)) / 3
    return vol
