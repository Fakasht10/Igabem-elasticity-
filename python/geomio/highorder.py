"""Second-order (curved) meshes from CAD: tri6 surfaces for collocation BEM and TET10 volumes for FEM, with the mid-edge
nodes on the exact CAD curves and faces and the CAD face id on every surface element.

Method, as in nurb-fea-tcd (github.com/Fakasht10/nurb-fea-tcd, MIT, Aleksey Kashtelyan; feakit/fea_gmsh.py and
bemkit/mesh.py at commit 553dd2a):
  1. Gmsh meshes the part at first order (same sizing as cad.py: h and `curvature` elements per 2 pi);
  2. gmsh.model.mesh.setOrder(2) with Mesh.SecondOrderLinear = 0 adds the mid-edge nodes ON the CAD entity the edge
     lies on: mid-parameter of the curve for an edge on a CAD curve, the surface point for an edge inside a face;
  3. Jacobian check (corners + centroid of every element).  A TET10 folds when its mid-nodes wrap a feature much
     smaller than the element; the folds are first given to Gmsh's optimisers (HighOrderElastic, then
     HighOrderFastCurving), and whatever is still folded is straightened: its mid-nodes go back to the chord
     midpoints (the TET4 under it is valid, so this always ends).  Straightened edges are counted and returned, since
     the surface there is a facet again.  For tri6 the same check uses n6 . n3 > 0 (curved normal against the flat
     triangle's).
Changes from nurb-fea-tcd: face ids are the Gmsh surface tags (cad.py, step_auto.py) instead of build123d faces
matched by centroid; the surface is oriented outward by mesh.orient on the corner triangles; the result carries its
own accuracy checks (mid-node distance to the CAD, curved volume against the CAD volume).

Node orders are Gmsh's: tri6 v1 v2 v3 m12 m23 m31 (bemkit.shape.tri6_shape); TET10 v0..v3 m01 m12 m20 m30 m32 m31.
Abaqus/meshio/JAX-FEM swap nodes 8 and 9 (TET10_GMSH_TO_ABAQUS); getting it wrong does not raise, it only distorts the
elements, so the tests check the TET10 volume.

Accuracy: a quadratic chord through the mid-parameter point is O(h^3) off a smooth surface (first order: O(h^2)), so
the enclosed volume converges at least as h^3, and as h^4 where the error cancels symmetrically (circles)."""
import time
from dataclasses import dataclass, field

import gmsh
import numpy as np

from .mesh import TriMesh, orient

TRI6_FLIP = [0, 2, 1, 5, 4, 3]
TET10_EDGES = ((0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1))
TET10_GMSH_TO_ABAQUS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
_TET_PROBE = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, .25, .25, .25]


@dataclass
class Mesh2:
    X: np.ndarray                     # (nn, 3) all nodes
    T6: np.ndarray                    # (nt, 6) boundary tri6, outward, Gmsh order
    fid: np.ndarray                   # (nt,) CAD face id (Gmsh surface tag)
    tet10: np.ndarray = None          # (ne, 10) Gmsh order, or None for a surface mesh
    names: dict = field(default_factory=dict)

    def corners(self):
        """the first-order surface on the same vertices (TriMesh), e.g. for STL export or cut cells."""
        used, T = np.unique(self.T6[:, :3], return_inverse=True)
        return TriMesh(self.X[used], T.reshape(-1, 3), self.fid, dict(self.names))


# ---------------------------------------------------------------- tri6 geometry (numpy)
def tri6_shape(P):
    u, v = P[:, 0], P[:, 1]
    w = 1 - u - v
    N = np.stack([w * (2 * w - 1), u * (2 * u - 1), v * (2 * v - 1), 4 * w * u, 4 * u * v, 4 * v * w], -1)
    Nu = np.stack([-(4 * w - 1), 4 * u - 1, 0 * u, 4 * (w - u), 4 * v, -4 * v], -1)
    Nv = np.stack([-(4 * w - 1), 0 * u, 4 * v - 1, -4 * u, 4 * u, 4 * (w - v)], -1)
    return N, Nu, Nv


def tri_rule(order=8):
    """Gmsh's Gauss rule on the reference triangle, exact for degree `order`: (P (q, 2), W (q,)), sum W = 1/2."""
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    try:
        P, W = gmsh.model.mesh.getIntegrationPoints(2, "Gauss%d" % order)
    finally:
        if own:
            gmsh.finalize()
    return np.asarray(P).reshape(-1, 3)[:, :2], np.asarray(W)


def tri6_moments(X, T6, rule=None):
    """per element: area and the divergence-theorem volume share (1/3) int x . n dS, plus n6 . n3 minimum over the
    rule points (> 0: the curved element is not folded against its chord triangle).  x . (x_u x x_v) is a degree-6
    polynomial on a tri6, so Gauss6 or higher gives the curved volume to round-off."""
    P, W = tri_rule() if rule is None else rule
    N, Nu, Nv = tri6_shape(P)
    E = X[T6]
    x = np.einsum("qk,ekc->eqc", N, E)
    n = np.cross(np.einsum("qk,ekc->eqc", Nu, E), np.einsum("qk,ekc->eqc", Nv, E))
    J = np.linalg.norm(n, axis=2)
    n3 = np.cross(E[:, 1] - E[:, 0], E[:, 2] - E[:, 0])
    cosang = np.einsum("eqc,ec->eq", n, n3) / np.maximum(J * np.linalg.norm(n3, axis=1)[:, None], 1e-300)
    return J @ W, np.einsum("eqc,eqc->eq", x, n) @ W / 3.0, cosang.min(1)


# ---------------------------------------------------------------- folds
def _tet_jacobians():
    _, det, _ = gmsh.model.mesh.getJacobians(11, _TET_PROBE)
    return np.asarray(det).reshape(-1, len(_TET_PROBE) // 3)


def _fix_tet_folds(optimise=True, max_pass=100):
    det = _tet_jacobians()
    bad = np.flatnonzero(det.min(1) <= 0)
    first, opt_used = len(bad), None
    if len(bad) and optimise:
        for method in ("HighOrderElastic", "HighOrderFastCurving"):
            try:
                gmsh.model.mesh.optimize(method, force=True)
            except Exception:  # noqa: BLE001 - an older Gmsh may lack the method
                continue
            opt_used = method
            det = _tet_jacobians()
            bad = np.flatnonzero(det.min(1) <= 0)
            if not len(bad):
                break
    straightened = 0
    for _ in range(max_pass):
        if not len(bad):
            break
        _, nodes = gmsh.model.mesh.getElementsByType(11)
        for row in np.asarray(nodes, np.int64).reshape(-1, 10)[bad]:
            for mid, (a, b) in zip(row[4:], TET10_EDGES):
                pa, pb = gmsh.model.mesh.getNode(int(row[a]))[0], gmsh.model.mesh.getNode(int(row[b]))[0]
                gmsh.model.mesh.setNode(int(mid), list(0.5 * (np.asarray(pa) + np.asarray(pb))), [])
                straightened += 1
        det = _tet_jacobians()
        bad = np.flatnonzero(det.min(1) <= 0)
    return dict(tet_folded=first, tet_optimiser=opt_used, tet_edges_straightened=straightened,
                tet_inverted=int(len(bad)), tet_min_jac_ratio=float((det.min(1) / det.max(1)).min()))


# ---------------------------------------------------------------- driver
def cad_to_mesh2(path, h=None, curvature=16, h_min=None, volume=False, optimise=True, check_cad=True, msh=None):
    """-> (Mesh2, info).  volume=True also builds TET10 (its boundary is the tri6 surface, same nodes).
    msh: optional path for Gmsh's own .msh output (order 2) before the model is closed."""
    t0 = time.perf_counter()
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("geomio2")
    info = {}
    try:
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        h = float(np.linalg.norm(bb[3:] - bb[:3])) / 20 if h is None else h
        for k, v in {"Mesh.Algorithm": 6, "Mesh.MeshSizeMax": h, "Mesh.MeshSizeMin": h_min or h / 50,
                     "Mesh.MeshSizeFromCurvature": curvature, "Mesh.MeshSizeFromPoints": 0,
                     "Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.SecondOrderLinear": 0}.items():
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(3 if volume else 2)
        gmsh.model.mesh.setOrder(2)
        if volume:
            info.update(_fix_tet_folds(optimise))
        t_mesh = time.perf_counter() - t0
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        idx = np.full(int(tags.max()) + 1, -1, np.int64)
        idx[tags.astype(np.int64)] = np.arange(len(tags))
        X = xyz.reshape(-1, 3)
        T6, F, names = [], [], {}
        for _, s in gmsh.model.getEntities(2):
            et, _, nodes = gmsh.model.mesh.getElements(2, s)
            for e, n in zip(et, nodes):
                if e != 9:
                    raise RuntimeError("surface %d has element type %d, expected 6-node triangles" % (s, e))
                T6.append(idx[n.astype(np.int64)].reshape(-1, 6))
                F.append(np.full(len(T6[-1]), s, np.int64))
            nm = gmsh.model.getEntityName(2, s)
            if nm:
                names[int(s)] = nm
        T6, F = np.vstack(T6), np.concatenate(F)
        if check_cad:   # distance of every surface mid-node to its CAD face (the projection actually achieved)
            dev = 0.0
            for s in np.unique(F):
                mid = np.unique(T6[F == s][:, 3:])
                c = np.asarray(gmsh.model.getClosestPoint(2, int(s), X[mid].ravel().tolist())[0]).reshape(-1, 3)
                dev = max(dev, float(np.linalg.norm(c - X[mid], axis=1).max()))
            info["mid_node_cad_dev"] = dev
        tet = None
        if volume:
            _, nodes = gmsh.model.mesh.getElementsByType(11)
            tet = idx[np.asarray(nodes, np.int64)].reshape(-1, 10)
            pts, wts = gmsh.model.mesh.getIntegrationPoints(11, "Gauss4")
            _, det, _ = gmsh.model.mesh.getJacobians(11, pts)
            info["vol_tet10"] = float((np.asarray(det).reshape(len(tet), -1) @ np.asarray(wts)).sum())
        vol_cad = sum(gmsh.model.occ.getMass(3, v) for _, v in gmsh.model.getEntities(3))
        if msh:
            gmsh.write(msh)
        rule = tri_rule(8)
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()

    # outward: orient the corner triangles, carry the flips over to the tri6 rows
    used, Tc = np.unique(T6[:, :3], return_inverse=True)
    Tc = Tc.reshape(-1, 3)
    m3, oinfo = orient(TriMesh(X[used], Tc, F))
    flip = ~np.all(m3.T == Tc, axis=1)
    T6[flip] = T6[flip][:, TRI6_FLIP]
    area, vol, cosmin = tri6_moments(X, T6, rule)
    V2 = float(vol.sum())
    info.update(faces=len(np.unique(F)), tri6=len(T6), tet10=0 if tet is None else len(tet), nodes=len(X), h=h,
                curvature=curvature, watertight=oinfo["boundary"] == 0 and oinfo["nonmanifold"] == 0
                and oinfo["misoriented"] == 0, tri6_folded=int((cosmin <= 0).sum()), tri6_min_cos=float(cosmin.min()),
                vol_cad=vol_cad, vol_tri6=V2, vol_tri3=m3.volume(),
                vol_rel_err=abs(V2 - vol_cad) / abs(vol_cad), vol_rel_err_tri3=abs(m3.volume() - vol_cad) / abs(vol_cad),
                face_area={int(s): float(area[F == s].sum()) for s in np.unique(F)},
                t_mesh=t_mesh, t_total=time.perf_counter() - t0)
    if tet is not None:
        info["vol_rel_err_tet10"] = abs(info["vol_tet10"] - vol_cad) / abs(vol_cad)
    return Mesh2(X, T6, F, tet, names), info


# ---------------------------------------------------------------- output
def write_vtu(path, m, cell_data=None):
    """ParaView file of the curved mesh: quadratic triangles (VTK 22) with face id, plus quadratic tets (VTK 24) when
    present.  VTK's tri6 order equals Gmsh's; its tet10 order equals Gmsh's except nodes 8 and 9 (as Abaqus)."""
    cells, types, fid = [m.T6], [np.full(len(m.T6), 22)], [m.fid]
    if m.tet10 is not None:
        cells.append(m.tet10[:, TET10_GMSH_TO_ABAQUS]); types.append(np.full(len(m.tet10), 24))
        fid.append(np.zeros(len(m.tet10), np.int64))
    conn = np.concatenate([c.ravel() for c in cells])
    off = np.cumsum(np.concatenate([np.full(len(c), c.shape[1]) for c in cells]))
    types, fid = np.concatenate(types), np.concatenate(fid)

    def arr(name, a, typ, nc=1):
        return ('<DataArray type="%s" Name="%s" NumberOfComponents="%d" format="ascii">%s</DataArray>'
                % (typ, name, nc, " ".join(map(repr if typ == "Float64" else str, a.ravel().tolist()))))
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
                '<UnstructuredGrid><Piece NumberOfPoints="%d" NumberOfCells="%d">' % (len(m.X), len(types)))
        f.write("<Points>%s</Points>" % arr("Points", m.X, "Float64", 3))
        f.write("<Cells>%s%s%s</Cells>" % (arr("connectivity", conn, "Int64"), arr("offsets", off, "Int64"),
                                           arr("types", types, "UInt8")))
        f.write("<CellData>%s" % arr("face_id", fid, "Int64"))
        for k, v in (cell_data or {}).items():
            f.write(arr(k, np.asarray(v, float), "Float64"))
        f.write("</CellData></Piece></UnstructuredGrid></VTKFile>\n")
