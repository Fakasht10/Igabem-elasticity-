# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Combinatorial topology on (n,3) vertex / (m,3) face arrays.

Vectorised re-implementations of the Sutura 0.9.8 numpy oracles in
``sutura/topology.py`` (``_edge_table_numpy``, ``_mesh_topology_numpy``,
``_re_orient_faces_coherently_numpy``, ``_repair_non_manifold_vertices_numpy``)
and ``sutura/defects.py`` (``_boundary_edges``, ``_boundary_loops``).  Every
routine is deterministic and depends only on the index arrays it is given.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class EdgeTable:
    edges: np.ndarray    # (E,2) sorted vertex pairs
    count: np.ndarray    # (E,) number of incident faces
    he_edge: np.ndarray  # (3m,) half-edge -> edge id; half-edge h = 3 f + k, k: (F[f,k] -> F[f,k+1])
    fwd: np.ndarray      # (3m,) True when the half-edge runs low -> high vertex


def half_edges(F):
    a = F.reshape(-1)
    b = F[:, [1, 2, 0]].reshape(-1)
    return a, b


def edge_table(F, nv=None) -> EdgeTable:
    F = np.asarray(F, np.int64)
    if len(F) == 0:
        z = np.zeros(0, np.int64)
        return EdgeTable(np.zeros((0, 2), np.int64), z, z, np.zeros(0, bool))
    nv = int(F.max()) + 1 if nv is None else nv
    a, b = half_edges(F)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * nv + hi
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return EdgeTable(np.stack([uk // nv, uk % nv], 1), cnt, inv.reshape(-1), a < b)


def face_adjacency(F, et: EdgeTable, manifold_only=True):
    """Pairs of faces sharing an edge (only count==2 edges when manifold_only)."""
    m = len(F)
    if m == 0:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    he_face = np.repeat(np.arange(m), 3)
    order = np.argsort(et.he_edge, kind="stable")
    e_sorted = et.he_edge[order]
    starts = np.flatnonzero(np.r_[True, e_sorted[1:] != e_sorted[:-1]])
    cnt = np.diff(np.r_[starts, len(e_sorted)])
    if manifold_only:
        sel = starts[cnt == 2]
        pairs = np.stack([he_face[order[sel]], he_face[order[sel + 1]]], 1)
        return pairs, order[np.stack([sel, sel + 1], 1)]
    pairs, hes = [], []
    for c in np.unique(cnt):
        if c < 2:
            continue
        s = starts[cnt == c]
        for i in range(c):
            for j in range(i + 1, c):
                pairs.append(np.stack([he_face[order[s + i]], he_face[order[s + j]]], 1))
                hes.append(np.stack([order[s + i], order[s + j]], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    return np.concatenate(pairs), np.concatenate(hes)


def components(m, pairs):
    """Connected components of faces under the given adjacency (labels sorted
    by smallest face index, so labels are canonical for a canonical mesh)."""
    if m == 0:
        return 0, np.zeros(0, np.int64)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m)) if len(pairs) \
        else coo_matrix((m, m))
    n, lab = connected_components(g, directed=False)
    first = np.full(n, m, np.int64)
    np.minimum.at(first, lab, np.arange(m))
    rank = np.empty(n, np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(n)
    return n, rank[lab]


def boundary_loops(F, et: EdgeTable):
    """Directed boundary half-edges chained into loops.

    Returns (loops, complex_loops): each loop is an int array of vertex ids in
    the direction of the existing faces' half-edges.  A loop passing through a
    vertex with more than one outgoing boundary half-edge is ambiguous and is
    returned in ``complex_loops`` (still traced, but never auto-filled).
    """
    if len(F) == 0:
        return [], []
    a, b = half_edges(F)
    bh = np.flatnonzero(et.count[et.he_edge] == 1)
    if len(bh) == 0:
        return [], []
    src, dst = a[bh], b[bh]
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    out_cnt = {}
    for s in src.tolist():
        out_cnt[s] = out_cnt.get(s, 0) + 1
    nxt = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        nxt.setdefault(s, []).append(d)
    used = set()
    loops, complex_loops = [], []
    for s0, d0 in zip(src.tolist(), dst.tolist()):
        if (s0, d0) in used:
            continue
        loop = [s0]
        used.add((s0, d0))
        cur, ok = d0, out_cnt[s0] == 1
        guard = 0
        while cur != s0 and guard <= len(src):
            guard += 1
            cand = [d for d in nxt.get(cur, []) if (cur, d) not in used]
            if not cand:
                ok = False
                break
            if out_cnt[cur] != 1:
                ok = False
            loop.append(cur)
            used.add((cur, cand[0]))
            cur = cand[0]
        (loops if ok and cur == s0 else complex_loops).append(np.asarray(loop, np.int64))
    return loops, complex_loops


def nonmanifold_vertices(F, et: EdgeTable, nv):
    """Vertices whose incident faces form more than one edge-connected fan.

    Vertices on non-manifold edges are reported separately (Sutura convention).
    Returns (nm_vertex_ids, fan_label_per_corner).
    """
    m = len(F)
    if m == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs, hes = face_adjacency(F, et, manifold_only=True)
    # link the corners of both endpoints across every manifold edge
    a, b = half_edges(F)
    links = []
    for col in (0, 1):
        h1, h2 = hes[:, 0], hes[:, 1]
        # vertex shared: endpoint 'a' of h1 is either a or b of h2
        v1 = a[h1] if col == 0 else b[h1]
        c1 = h1 if col == 0 else 3 * (h1 // 3) + (h1 % 3 + 1) % 3
        c2 = np.where(a[h2] == v1, h2, 3 * (h2 // 3) + (h2 % 3 + 1) % 3)
        links.append(np.stack([c1, c2], 1))
    links = np.concatenate(links) if links else np.zeros((0, 2), np.int64)
    n, lab = components(3 * m, links)
    vert = F.reshape(-1)
    key = np.unique(vert * (n + 1) + lab)
    fans = np.bincount(key // (n + 1), minlength=nv)
    on_nm_edge = np.zeros(nv, bool)
    on_nm_edge[et.edges[et.count > 2].reshape(-1)] = True
    nm = np.flatnonzero((fans > 1) & ~on_nm_edge)
    return nm, lab


def orientation_parity(F, et: EdgeTable):
    """Inconsistent manifold edges (both faces traverse the edge the same way)."""
    pairs, hes = face_adjacency(F, et, manifold_only=True)
    if len(pairs) == 0:
        return pairs, np.zeros(0, bool)
    same = et.fwd[hes[:, 0]] == et.fwd[hes[:, 1]]
    return pairs, same


def coherent_flips(m, pairs, same):
    """BFS parity propagation over manifold edges.

    Returns (flip mask, orientable flag per component label, labels).  The
    root of each component is its smallest face index, so the result depends
    only on the (canonical) face order.
    """
    adj = [[] for _ in range(m)]
    for (f, g), s in zip(pairs.tolist(), same.tolist()):
        adj[f].append((g, s))
        adj[g].append((f, s))
    flip = np.zeros(m, bool)
    seen = np.zeros(m, bool)
    ncomp, lab = components(m, pairs)
    orientable = np.ones(ncomp, bool)
    for root in range(m):
        if seen[root]:
            continue
        seen[root] = True
        q = deque([root])
        while q:
            f = q.popleft()
            for g, s in adj[f]:
                want = flip[f] ^ s
                if not seen[g]:
                    seen[g] = True
                    flip[g] = want
                    q.append(g)
                elif flip[g] != want:
                    orientable[lab[f]] = False
    return flip, orientable, lab
