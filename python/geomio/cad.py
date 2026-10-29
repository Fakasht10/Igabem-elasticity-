"""STEP / IGES / BREP -> watertight face-tagged triangle mesh, by Gmsh's OpenCASCADE kernel (pip gmsh, no OCP).

Face ids are the Gmsh surface tags after gmsh.model.occ.importShapes of the same file, i.e. the numbering that
python/step_auto.py uses for its blocks (QuadMesh.surf) when it runs without --split, so a boundary condition picked by
CAD face lands on the same face in the IgBEM patches, in this mesh and in the cut-cell surface rule
(python/immersed/cut_quadrature.surface_rule returns the face id per point).  CAD face names (STEP labels) come back
in TriMesh.names when the file has them.

Watertight by construction: Gmsh meshes every CAD curve once and both neighbouring faces use those nodes, so there is
no per-face tessellation to weld afterwards.  (materializr's StlExport meshes faces independently with BRepMesh and
then welds at 1e-3 mm and fills pinholes; that repair is in mesh.py for STL files from elsewhere.)  The winding is then
made outward by mesh.orient.

Accuracy: element size h (MeshSizeMax) and `curvature` elements per 2 pi of curvature; the chord error on a radius R
is about R (1 - cos(pi / curvature)) ~ R pi^2 / (2 curvature^2) (4.8 % of R at 8, 1.2 % at 16), and the volume error
is of the same relative order on curved parts and zero on polyhedral ones."""
import time

import gmsh
import numpy as np

from .mesh import TriMesh, orient


def cad_to_mesh(path, h=None, curvature=16, h_min=None):
    """-> (TriMesh, info).  h defaults to 1/20 of the bounding-box diagonal."""
    t0 = time.perf_counter()
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("geomio")
    try:
        gmsh.model.occ.importShapes(path)
        gmsh.model.occ.synchronize()
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        diag = float(np.linalg.norm(bb[3:] - bb[:3]))
        h = diag / 20 if h is None else h
        for k, v in {"Mesh.Algorithm": 6, "Mesh.MeshSizeMax": h, "Mesh.MeshSizeMin": h_min or h / 50,
                     "Mesh.MeshSizeFromCurvature": curvature, "Mesh.MeshSizeFromPoints": 0,
                     "Mesh.MeshSizeExtendFromBoundary": 0}.items():
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(2)
        t_mesh = time.perf_counter() - t0
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        idx = np.full(int(tags.max()) + 1, -1, np.int64)
        idx[tags.astype(np.int64)] = np.arange(len(tags))
        T, F, names = [], [], {}
        for _, s in gmsh.model.getEntities(2):
            et, _, nodes = gmsh.model.mesh.getElements(2, s)
            for e, n in zip(et, nodes):
                if e != 2:
                    raise RuntimeError("surface %d meshed with element type %d, expected 3-node triangles" % (s, e))
                T.append(idx[n.astype(np.int64)].reshape(-1, 3))
                F.append(np.full(len(T[-1]), s, np.int64))
            nm = gmsh.model.getEntityName(2, s)
            if nm:
                names[int(s)] = nm
        vol_cad = sum(gmsh.model.occ.getMass(3, v) for _, v in gmsh.model.getEntities(3))
        nvol = len(gmsh.model.getEntities(3))
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()
    m = TriMesh(xyz.reshape(-1, 3), np.vstack(T), np.concatenate(F), names)
    used, Tn = np.unique(m.T, return_inverse=True)
    m = TriMesh(m.V[used], Tn.reshape(-1, 3), m.fid, names)
    m, info = orient(m)
    vol = m.volume()
    info.update(volumes=nvol, faces=len(np.unique(m.fid)), triangles=len(m.T), h=h, curvature=curvature,
                vol_cad=vol_cad, vol_mesh=vol, vol_rel_err=abs(vol - vol_cad) / abs(vol_cad) if vol_cad else None,
                watertight=info["boundary"] == 0 and info["nonmanifold"] == 0 and info["misoriented"] == 0,
                t_mesh=t_mesh, t_total=time.perf_counter() - t0)
    return m, info
