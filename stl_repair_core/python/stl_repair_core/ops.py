# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Conservative repair operations.

Each operation maps a canonical :class:`Mesh` to a candidate mesh plus an
info dict, or returns ``None`` when it does not apply.  Operations never move
a vertex to a new coordinate (the tolerance weld snaps to an *existing*
coordinate), never add a vertex, and never remove a protected face.  Adoption
is decided by :mod:`stl_repair_core.repair` gates, not here.

Sutura 0.9.8 counterparts (``sutura/topology.py`` numpy oracles,
``rust/sutura-geom/src/hole.rs``):
  remove_duplicates      <- _remove_duplicate_faces_numpy (+ reversed-pair rule)
  remove_degenerate      <- _remove_null_faces_numpy(safe=True) (+ cap edge-flip)
  tolerance_weld         <- _remove_duplicate_vertices_numpy (exact) generalised
  remove_nm_fins         <- _repair_non_manifold_edges_numpy, restricted
  reorient               <- _re_orient_faces_coherently_numpy (+ outward by nesting)
  fill_hole              <- hole.rs close_holes (Liepa-style DP, diameter bound)
  remove_debris          <- _remove_connected_components_numpy, health-gated
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import geometry as G
from . import intersect as X
from . import topology as T
from .diagnose import component_table, duplicate_groups
from .mesh import Mesh, canonical


def _prot_mask(mesh, protected):
    if protected is None or len(protected) == 0:
        return np.zeros(mesh.nf, bool)
    return np.isin(mesh.src, protected)


# ------------------------------------------------------------- duplicates
def remove_duplicates(mesh: Mesh, protected=None):
    F = mesh.F
    same_extra, rev, lab, parity = duplicate_groups(F)
    if same_extra == 0 and rev == 0:
        return None
    m = len(F)
    prot = _prot_mask(mesh, protected)
    et = T.edge_table(F, len(mesh.V))
    a, b = T.half_edges(F)
    # orientation support: for every half-edge, count opposite half-edges of other faces
    nv = len(mesh.V)
    dkey = a * nv + b
    rkey = b * nv + a
    uk, cnt = np.unique(dkey, return_counts=True)
    keep = np.ones(m, bool)
    removed_reason = {}
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    starts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]])
    ends = np.r_[starts[1:], len(ls)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        g = order[s:e]
        classes = {}
        for f in g.tolist():
            classes.setdefault(bool(parity[f]), []).append(f)
        if len(classes) == 1:
            grp = next(iter(classes.values()))
            pk = [f for f in grp if prot[f]]
            k = pk[0] if pk else grp[0]
            for f in grp:
                if f != k:
                    keep[f] = False
                    removed_reason[f] = "duplicate"
            continue
        # reversed duplicates
        edges_of = et.he_edge.reshape(m, 3)[g]
        isolated = bool((et.count[edges_of] == len(g)).all())
        if isolated and not prot[g].any():
            keep[g] = False
            for f in g.tolist():
                removed_reason[f] = "reversed_duplicate_sliver"
            continue

        def support(f):
            hk = rkey[3 * f:3 * f + 3]
            pos = np.searchsorted(uk, hk)
            pos = np.minimum(pos, len(uk) - 1)
            return int(np.where(uk[pos] == hk, cnt[pos], 0).sum())
        best = None
        for cls in sorted(classes):
            fs = classes[cls]
            score = (any(prot[f] for f in fs), support(fs[0]), -min(fs))
            if best is None or score > best[0]:
                best = (score, cls)
        kc = best[1]
        grp = classes[kc]
        pk = [f for f in grp if prot[f]]
        k = pk[0] if pk else grp[0]
        for f in g.tolist():
            if f != k:
                keep[f] = False
                removed_reason[f] = "reversed_duplicate" if parity[f] != parity[k] else "duplicate"
    if keep.all():
        return None
    cand = canonical(mesh.take(keep))
    reasons = {}
    for f, r in removed_reason.items():
        reasons[r] = reasons.get(r, 0) + 1
    return cand, {"removed": reasons, "removed_src": mesh.src[~keep]}


# ------------------------------------------------------------- degenerate
def _between(u, v, w):
    """v lies on segment uw (points known to be collinear)."""
    return bool(np.all(np.minimum(u, w) <= v) and np.all(v <= np.maximum(u, w)))


def remove_degenerate(mesh: Mesh, protected=None, engine="auto", max_rounds=8):
    """Remove zero-area faces whose removal opens no edge; resolve load-bearing
    collinear caps by splitting the neighbour across the long edge (an edge
    flip with zero surface displacement)."""
    V = mesh.V
    cur = mesh
    flips = removed = 0
    removed_src = []
    for _ in range(max_rounds):
        F = cur.F
        idx_deg = (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
        deg = X.degenerate_mask(V, F, engine=engine) | idx_deg
        if not deg.any():
            break
        prot = _prot_mask(cur, protected)
        et = T.edge_table(F, len(V))
        ecnt = et.count[et.he_edge].reshape(-1, 3)
        removable = deg & ((ecnt != 2).all(1) | idx_deg) & ~prot
        changed = False
        F2 = F.copy()
        keep = ~removable
        if removable.any():
            removed += int(removable.sum())
            removed_src.extend(cur.src[removable].tolist())
            changed = True
        # cap flips on the remaining degenerate faces
        a, b = T.half_edges(F)
        nv = len(V)
        he_lookup = {}
        for h, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
            he_lookup.setdefault((x, y), []).append(h)
        edge_set = set(map(tuple, et.edges.tolist()))
        touched = np.zeros(len(F), bool)
        new_faces, new_src, new_attr, new_solid = [], [], [], []
        for f in np.flatnonzero(deg & keep & ~idx_deg):
            if touched[f]:
                continue
            tri = F[f].tolist()
            done = False
            for k in range(3):
                u, w, v = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]   # long edge u->w, v between
                if not _between(V[u], V[v], V[w]):
                    continue
                nb = [h for h in he_lookup.get((w, u), []) if h // 3 != f]
                if len(nb) != 1 or len(he_lookup.get((u, w), [])) != 1:
                    continue
                g = nb[0] // 3
                if touched[g] or prot[g]:
                    continue
                gt = F[g].tolist()
                x = [q for q in gt if q not in (u, w)]
                if len(x) != 1:
                    continue
                x = x[0]
                if (min(v, x), max(v, x)) in edge_set:
                    continue
                t1, t2 = [w, v, x], [v, u, x]   # g = (w,u,x) split at v
                cand_deg = X.degenerate_mask(V, np.asarray([t1, t2]), engine=engine)
                if cand_deg.any():
                    continue
                touched[f] = touched[g] = True
                keep[f] = keep[g] = False
                for t in (t1, t2):
                    new_faces.append(t)
                    new_src.append(cur.src[g])
                    new_attr.append(cur.attr[g])
                    new_solid.append(cur.solid[g])
                edge_set.add((min(v, x), max(v, x)))
                removed_src.append(int(cur.src[f]))
                flips += 1
                done = changed = True
                break
            if done:
                continue
        if not changed:
            break
        nf = np.asarray(new_faces, np.int64).reshape(-1, 3)
        cur = canonical(Mesh(V, np.concatenate([F2[keep], nf]),
                             np.concatenate([cur.src[keep], np.asarray(new_src, np.int64)]),
                             np.concatenate([cur.attr[keep], np.asarray(new_attr, np.uint16)]),
                             np.concatenate([cur.solid[keep], np.asarray(new_solid, np.int32)])))
        V = cur.V
    if removed == 0 and flips == 0:
        return None
    return cur, {"removed_faces": removed, "cap_flips": flips, "removed_src": np.asarray(removed_src, np.int64),
                 "split_src_note": "cap flips keep the neighbour's src on both halves"}


# ------------------------------------------------------------- weld
def tolerance_weld(mesh: Mesh, tol_abs, protected=None):
    """Snap clusters of distinct vertices closer than ``tol_abs`` to the
    lexicographically smallest member (an existing coordinate).  Clusters
    whose diameter exceeds 2 tol (chains) or that would collapse a protected
    face are skipped."""
    if tol_abs <= 0 or mesh.nf == 0:
        return None
    V, F = mesh.V, mesh.F
    used = np.unique(F)
    pr = cKDTree(V[used]).query_pairs(tol_abs, output_type="ndarray")
    if len(pr) == 0:
        return None
    pr = used[pr]
    n = len(V)
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    multi = np.flatnonzero(sizes > 1)
    target = np.arange(n)
    prot = _prot_mask(mesh, protected)
    skipped = 0
    merged = 0
    order = np.lexsort((V[:, 2], V[:, 1], V[:, 0]))
    first = np.full(len(sizes), -1)
    for v in order[::-1]:
        first[lab[v]] = v
    members = {}
    for v in np.flatnonzero(sizes[lab] > 1):
        members.setdefault(lab[v], []).append(v)
    for c in multi:
        mem = np.asarray(members[c])
        P = V[mem]
        diam = float(np.max(np.linalg.norm(P[:, None] - P[None], axis=2))) if len(mem) <= 64 else np.inf
        if diam > 2 * tol_abs:
            skipped += 1
            continue
        target[mem] = first[c]
        merged += len(mem) - 1
    if merged == 0:
        return None
    F2 = target[F]
    collapsed = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
    if (collapsed & prot).any():
        # undo clusters touching collapsed protected faces
        bad_v = np.unique(F[collapsed & prot])
        bad_c = np.unique(lab[bad_v])
        undo = np.isin(lab, bad_c)
        target[undo] = np.arange(n)[undo]
        merged -= int(undo.sum() - len(bad_c))
        F2 = target[F]
        collapsed = (F2[:, 0] == F2[:, 1]) | (F2[:, 1] == F2[:, 2]) | (F2[:, 0] == F2[:, 2])
        skipped += len(bad_c)
    cand = canonical(Mesh(V, F2[~collapsed], mesh.src[~collapsed], mesh.attr[~collapsed], mesh.solid[~collapsed]))
    info = {"merged_vertices": int(merged), "skipped_chain_clusters": int(skipped),
            "collapsed_faces": int(collapsed.sum()), "removed_src": mesh.src[collapsed]}
    dd = remove_duplicates(cand, protected)
    if dd is not None:
        cand, di = dd
        info["post_weld_duplicates"] = di["removed"]
        info["removed_src"] = np.concatenate([info["removed_src"], di["removed_src"]])
    return cand, info


# ------------------------------------------------------------- non-manifold fins
def remove_nm_fins(mesh: Mesh, protected=None):
    """On every non-manifold edge, delete faces whose other two edges are
    boundary or non-manifold, smallest area first, while the edge stays
    non-manifold.  Never creates a boundary edge."""
    F, V = mesh.F, mesh.V
    et = T.edge_table(F, len(V))
    if not (et.count > 2).any():
        return None
    prot = _prot_mask(mesh, protected)
    E = et.he_edge.reshape(-1, 3)
    cnt = et.count.copy()
    cand = np.flatnonzero((cnt[E] > 2).any(1) & ~prot)
    area = G.tri_areas(V, F)
    cand = cand[np.lexsort((cand, area[cand]))]
    removed = np.zeros(len(F), bool)
    for f in cand.tolist():
        e = E[f]
        if not (cnt[e] > 2).any():
            continue
        if (cnt[e] == 2).any():
            continue
        removed[f] = True
        cnt[e] -= 1
    if not removed.any():
        return None
    return canonical(mesh.take(~removed)), {"removed_faces": int(removed.sum()), "removed_src": mesh.src[removed]}


# ------------------------------------------------------------- orientation
def _solid_angle_winding(p, P):
    a, b, c = P[:, 0] - p, P[:, 1] - p, P[:, 2] - p
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    num = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la \
        + np.einsum("ij,ij->i", c, a) * lb
    return float(np.arctan2(num, den).sum() * 2.0 / (4 * np.pi))


def nesting_depth(mesh: Mesh, lab, comp_ids):
    """Number of other closed components containing each listed component."""
    V, F = mesh.V, mesh.F
    depth = {c: 0 for c in comp_ids}
    if len(comp_ids) < 2:
        return depth
    info = {}
    for c in comp_ids:
        fs = np.flatnonzero(lab == c)
        P = V[F[fs]]
        vid = np.unique(F[fs])
        o = np.lexsort((V[vid, 2], V[vid, 1], V[vid, 0]))
        info[c] = (P, P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0), V[vid[o[0]]])
    for c in comp_ids:
        p = info[c][3]
        for d in comp_ids:
            if d == c:
                continue
            P, lo, hi, _ = info[d]
            if (p < lo).any() or (p > hi).any():
                continue
            if abs(_solid_angle_winding(p, P)) > 0.5:
                depth[c] += 1
    return depth


def reorient(mesh: Mesh, component_flips=True):
    F = mesh.F
    m = len(F)
    if m == 0:
        return None
    et = T.edge_table(F, len(mesh.V))
    pairs, same = T.orientation_parity(F, et)
    flip, orientable, lab = T.coherent_flips(m, pairs, same)
    F2 = F.copy()
    F2[flip] = F2[flip][:, [0, 2, 1]]
    # outward / majority per edge-connected (manifold) patch
    area = G.tri_areas(mesh.V, F)
    tmp = Mesh(mesh.V, F2, mesh.src, mesh.attr, mesh.solid)
    et2 = T.edge_table(F2, len(mesh.V))
    nm, _ = T.nonmanifold_vertices(F2, et2, len(mesh.V))
    deg = np.zeros(m, bool)
    comps, clab = component_table(tmp, et2, nm, deg)
    closed = [i for i, c in enumerate(comps) if c["closed"] and c["consistently_oriented"]]
    depth = nesting_depth(tmp, clab, closed)
    inv = np.zeros(m, bool)
    for i, c in enumerate(comps):
        s = clab == i
        if not c["consistently_oriented"]:
            continue
        if i in depth and component_flips:
            want = 1.0 if depth[i] % 2 == 0 else -1.0
            if np.sign(c["signed_volume"]) == -want:
                inv |= s
        elif i not in depth:
            # open patch: keep the majority (by area) of the input orientation
            if area[s & flip].sum() > area[s & ~flip].sum():
                inv |= s
    F2[inv] = F2[inv][:, [0, 2, 1]]
    changed = (F2 != F).any(1)
    if not changed.any():
        return None
    return canonical(Mesh(mesh.V, F2, mesh.src, mesh.attr, mesh.solid)), {
        "flipped_faces": int(changed.sum()), "non_orientable_components": int((~orientable).sum())}


# ------------------------------------------------------------- holes
def _normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    return n / ln if ln > 0 else None, 0.5 * ln


def fill_hole(mesh: Mesh, loop, max_vertices=150):
    """Liepa (2003) minimum-weight triangulation of one boundary loop with
    weight (max dihedral angle, total area), lexicographic.  The loop order
    follows the existing faces' half-edges, so patch faces are written in
    reverse loop order to match orientation.  No vertex is added."""
    V, F = mesh.V, mesh.F
    L = np.asarray(loop, np.int64)
    n = len(L)
    if n < 3:
        return None, "loop shorter than 3"
    if n > max_vertices:
        return None, "loop has %d vertices > %d" % (n, max_vertices)
    r = int(np.argmin(L))
    L = np.roll(L, -r)
    P = V[L]
    scale = float(np.linalg.norm(P.max(0) - P.min(0)))
    et = T.edge_table(F, len(V))
    nv = len(V)
    ekeys = set((et.edges[:, 0] * nv + et.edges[:, 1]).tolist())
    # existing face across each loop edge (L[i] -> L[i+1] is a half-edge of that face)
    a, b = T.half_edges(F)
    hk = a * nv + b
    order = np.argsort(hk)
    sk = hk[order]
    nb_normal = []
    for i in range(n):
        key = L[i] * nv + L[(i + 1) % n]
        pos = np.searchsorted(sk, key)
        f = order[pos] // 3
        nb_normal.append(_normal(*V[F[f]])[0])

    def edge_exists(i, k):
        u, w = L[i], L[k]
        return (min(u, w) * nv + max(u, w)) in ekeys

    INF = (np.inf, np.inf)
    W = {}
    O = {}
    tri_n = {}
    for i in range(n - 1):
        W[(i, i + 1)] = (0.0, 0.0)
    for i in range(n - 2):
        k = i + 2
        tri_n[(i, i + 1, k)] = None
    amin = 1e-12 * scale * scale

    def tri_normal(i, m_, k):
        key = (i, m_, k)
        if key not in tri_n or tri_n[key] is None:
            nn, ar = _normal(P[k], P[m_], P[i])   # reversed loop order
            tri_n[key] = (nn, ar)
        return tri_n[key]

    def ang(n1, n2):
        if n1 is None or n2 is None:
            return np.pi
        return float(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

    for span in range(2, n):
        for i in range(0, n - span):
            k = i + span
            if span < n - 1 and edge_exists(i, k):
                W[(i, k)] = INF
                continue
            best, bm = INF, -1
            for m_ in range(i + 1, k):
                wl, wr = W[(i, m_)], W[(m_, k)]
                if wl[0] == np.inf or wr[0] == np.inf:
                    continue
                nn, ar = tri_normal(i, m_, k)
                if nn is None or ar <= amin:
                    continue
                angs = [wl[0], wr[0]]
                for (x, y) in ((i, m_), (m_, k)):
                    if y == x + 1:
                        angs.append(ang(nn, nb_normal[x]))
                    else:
                        angs.append(ang(nn, tri_normal(x, O[(x, y)], y)[0]))
                if k == n - 1 and i == 0:
                    angs.append(ang(nn, nb_normal[n - 1]))
                w = (max(angs), wl[1] + wr[1] + ar)
                if w < best:
                    best, bm = w, m_
            W[(i, k)] = best
            O[(i, k)] = bm
    if W[(0, n - 1)][0] == np.inf:
        return None, "no valid triangulation (existing interior edges or collinear loop)"
    tris = []
    stack = [(0, n - 1)]
    while stack:
        i, k = stack.pop()
        if k - i < 2:
            continue
        m_ = O[(i, k)]
        tris.append((L[k], L[m_], L[i]))
        stack += [(i, m_), (m_, k)]
    nf = np.asarray(tris, np.int64)
    k = len(nf)
    cand = canonical(Mesh(V, np.concatenate([F, nf]), np.concatenate([mesh.src, -np.ones(k, np.int64)]),
                          np.concatenate([mesh.attr, np.zeros(k, np.uint16)]),
                          np.concatenate([mesh.solid, -np.ones(k, np.int32)])))
    return (cand, {"loop_vertices": int(n), "new_faces": int(k), "max_dihedral_rad": float(W[(0, n - 1)][0]),
                   "patch_area": float(W[(0, n - 1)][1]), "new_face_corners": V[nf]}), None


# ------------------------------------------------------------- debris
def remove_debris(mesh: Mesh, area_threshold, protected=None):
    if not area_threshold or area_threshold <= 0 or mesh.nf == 0:
        return None
    et = T.edge_table(mesh.F, len(mesh.V))
    nm, _ = T.nonmanifold_vertices(mesh.F, et, len(mesh.V))
    deg = X.degenerate_mask(mesh.V, mesh.F)
    comps, lab = component_table(mesh, et, nm, deg)
    prot = _prot_mask(mesh, protected)
    kill = np.zeros(mesh.nf, bool)
    removed, kept_small = [], 0
    for i, c in enumerate(comps):
        if c["area"] >= area_threshold:
            continue
        s = lab == i
        if c["healthy"] or prot[s].any():
            kept_small += 1
            continue
        kill |= s
        removed.append({"faces": c["faces"], "area": c["area"], "closed": c["closed"]})
    if not kill.any():
        return None
    return canonical(mesh.take(~kill)), {"removed_components": removed, "small_healthy_or_protected_kept": kept_small,
                                         "removed_src": mesh.src[kill], "exempt_src": mesh.src[kill]}
