# Vendored from nurb-fea-tcd (github.com/Fakasht10/nurb-fea-tcd, MIT, Aleksey Kashtelyan), feakit/bem_backend.py at
# commit 553dd2a: the solver-independent helpers that put a boundary-element solution onto the FEA mesh (TET10
# Gauss rule and interpolation, sharp-edge rings, display-only harmonic extension). Unchanged apart from this header.
"""Fields on the TET10 mesh: Gauss points and interpolation, sharp-edge rings, display displacement."""
import numpy as np


def harmonic_extension(n_nodes, cells, boundary, values):
    """Fill interior nodal values from boundary ones with a graph Laplacian (all node pairs of each cell).
    Display only: elasticity displacement is not harmonic, so no reported number may use the interior values."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    cells = np.asarray(cells)
    a, b = np.triu_indices(cells.shape[1], 1)
    i, j = cells[:, a].ravel(), cells[:, b].ravel()
    W = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n_nodes, n_nodes)).tocsr()
    W = W + W.T
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W
    is_b = np.zeros(n_nodes, bool)
    is_b[boundary] = True
    inner = np.nonzero(~is_b)[0]
    out = np.zeros((n_nodes, values.shape[1]))
    out[boundary] = values
    if len(inner):
        # Jacobi-preconditioned CG, not a direct factorization: the TET10 graph has heavy fill-in
        # (splu took ~400 s on a 134k-node SENB mesh). The values are display only, so 1e-6 is plenty.
        A = L[inner][:, inner].tocsr()
        rhs = -np.asarray(L[inner][:, boundary] @ values)
        M = sp.diags(1.0 / A.diagonal())
        x0 = np.broadcast_to(values.mean(axis=0), (len(inner), values.shape[1]))
        for k in range(values.shape[1]):
            out[inner, k], _ = spla.cg(A, rhs[:, k], x0=x0[:, k].copy(), rtol=1e-6, maxiter=2000, M=M)
    return out


# 4-point degree-2 tet rule (the one bemkit.mesh.volume_quadrature and JAX-FEM's TET10 use), barycentric (L1, L2, L3)
_TA, _TB = 0.5854101966249685, 0.1381966011250105
_TET_RULE = np.array([[_TB, _TB, _TB], [_TA, _TB, _TB], [_TB, _TA, _TB], [_TB, _TB, _TA]])
# Abaqus/meshio TET10 order (fea_gmsh.GMSH_TET10_TO_ABAQUS): 0-3 corners, 4=(0,1) 5=(1,2) 6=(0,2) 7=(0,3) 8=(1,3) 9=(2,3)
_TET10_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


def _tet10_shape(q):
    """TET10 shape functions N (nq, 10) and their reference gradients dN (nq, 10, 3) at points q = (L1, L2, L3)."""
    q = np.atleast_2d(q)
    L = np.column_stack([1.0 - q.sum(axis=1), q])                       # (nq, 4)
    dL = np.array([[-1.0, -1.0, -1.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # dL_i / d(L1, L2, L3)
    N = np.empty((len(q), 10))
    dN = np.empty((len(q), 10, 3))
    for i in range(4):
        N[:, i] = L[:, i] * (2 * L[:, i] - 1)
        dN[:, i] = (4 * L[:, i] - 1)[:, None] * dL[i]
    for k, (a, b) in enumerate(_TET10_EDGES):
        N[:, 4 + k] = 4 * L[:, a] * L[:, b]
        dN[:, 4 + k] = 4 * (L[:, b][:, None] * dL[a] + L[:, a][:, None] * dL[b])
    return N, dN


EDGE_KINK_DEG = 20.0          # two faces meeting at this angle or more are a sharp edge (nodal stress unreliable)


def tet10_quadrature(points, cells, nodal):
    """Gauss points, weights (mm^3) and the TET10 interpolation of a nodal field at them, on the FEA mesh itself.
    nodal is (n_nodes, ...); returns P (n_cells * 4, 3), w (n_cells * 4,), values (n_cells * 4, ...)."""
    cells = np.asarray(cells)
    if cells.shape[1] == 4:                                              # TET4 mesh: linear map and interpolation
        q = _TET_RULE
        N = np.column_stack([1.0 - q.sum(axis=1), q])
        dN = np.broadcast_to(np.array([[-1.0, -1.0, -1.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), (len(q), 4, 3))
    else:
        N, dN = _tet10_shape(_TET_RULE)
    X = np.asarray(points, float)[cells]                                 # (nc, 10, 3)
    J = np.einsum("qkr,ckx->cqxr", dN, X)                                # dx/dL per cell and point
    w = np.abs(np.linalg.det(J)) / 24.0                                  # (1/6 volume) x (1/4 per point)
    P = np.einsum("qk,ckx->cqx", N, X).reshape(-1, 3)
    V = np.tensordot(N, np.asarray(nodal)[cells], axes=([1], [1]))       # (4, nc, ...)
    V = np.moveaxis(V, 0, 1).reshape((-1,) + np.asarray(nodal).shape[1:])
    return P, w.ravel(), V


def tet10_gradient(points, cells, nodal):
    """Spatial gradient of a nodal field at the Gauss points of tet10_quadrature (same order): (n_cells * 4, ..., 3)."""
    _, dN = _tet10_shape(_TET_RULE)
    X = np.asarray(points, float)[np.asarray(cells)]                     # (nc, 10, 3)
    J = np.einsum("qkr,ckx->cqxr", dN, X)                                # dx/dL
    G = np.einsum("qkr,cqrx->cqkx", dN, np.linalg.inv(J))                # dN_k/dx
    V = np.asarray(nodal, float)[np.asarray(cells)]                      # (nc, 10, ...)
    g = np.einsum("cqkx,ck...->cq...x", G, V)
    return g.reshape((-1,) + g.shape[2:])

def kinked_edge_nodes(mesh, min_angle_deg=EDGE_KINK_DEG):
    """Surface nodes where two CAD faces meet at a kink of at least `min_angle_deg` (a sharp edge, convex or
    re-entrant). Faces that meet tangentially (a fillet running into a flat) are not a kink. Orientation-free: the
    test is on |n_a . n_b| of the faces' mean normals at the node, so a flipped triangle cannot hide an edge."""
    fids = getattr(mesh, "surface_face_ids", None)
    if fids is None or not len(mesh.surface_triangles):
        return np.zeros(0, int)
    P = np.asarray(mesh.points, float)
    tri = np.asarray(mesh.surface_triangles)[:, :3]
    n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    node, face = tri.ravel(), np.repeat(np.asarray(fids, int), 3)
    pairs, inv = np.unique(np.column_stack([node, face]), axis=0, return_inverse=True)
    acc = np.zeros((len(pairs), 3))
    nn = np.repeat(n, 3, axis=0)
    # align each triangle normal with the first one seen for its (node, face) so opposite windings do not cancel
    first = np.full(len(pairs), -1)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    flip = np.einsum("ij,ij->i", nn, nn[first[inv]]) < 0
    np.add.at(acc, inv.ravel(), np.where(flip[:, None], -nn, nn))
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
    cos_kink = np.cos(np.radians(min_angle_deg))
    out = []
    starts = np.flatnonzero(np.r_[True, pairs[1:, 0] != pairs[:-1, 0]])
    for a, b in zip(starts, np.r_[starts[1:], len(pairs)]):
        if b - a < 2:
            continue
        g = np.abs(acc[a:b] @ acc[a:b].T)
        if g.min() < cos_kink:
            out.append(pairs[a, 0])
    return np.asarray(out, int)


def edge_rings(mesh, rings=2):
    """Gauss-point mask (cell-major, as tet10_quadrature orders them) of the `rings` element layers round the
    kinked edges."""
    cells = np.asarray(mesh.cells)
    nodes = kinked_edge_nodes(mesh)
    hit = np.zeros(len(cells), bool)
    for _ in range(rings if len(nodes) else 0):
        hit = np.isin(cells, nodes).any(axis=1)
        nodes = np.unique(cells[hit])
    return np.repeat(hit, len(_TET_RULE))


def _sig6(s):
    """(n, 3, 3) -> (n, 6) in the FEA order xx yy zz xy yz zx."""
    return np.stack([s[:, 0, 0], s[:, 1, 1], s[:, 2, 2], s[:, 0, 1], s[:, 1, 2], s[:, 2, 0]], axis=1)
