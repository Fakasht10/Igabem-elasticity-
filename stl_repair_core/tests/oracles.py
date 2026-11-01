# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Independent geometry oracles: none of these share code with the repair.

* manifold3d  (Manifold library): manifold status, genus, volume, area
* trimesh     (own STL reader, own merge): watertight, winding, volume, bodies
* PyMeshLab   (VCG): topological measures and self-intersecting faces
"""
import numpy as np


def manifold_check(path):
    import manifold3d as m3
    import trimesh
    tm = trimesh.load(path, force="mesh", process=True)
    mesh = m3.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32))
    man = m3.Manifold(mesh)
    st = man.status()
    ok = st == m3.Error.NoError
    return {"ok": bool(ok), "status": str(st), "volume": float(man.volume()) if ok else None,
            "area": float(man.surface_area()) if ok else None, "genus": int(man.genus()) if ok else None}


def trimesh_check(path):
    import trimesh
    tm = trimesh.load(path, force="mesh", process=True)
    return {"watertight": bool(tm.is_watertight), "winding_consistent": bool(tm.is_winding_consistent),
            "volume": float(tm.volume), "area": float(tm.area), "faces": int(len(tm.faces)),
            "bodies": int(len(tm.split(only_watertight=False)))}


def meshlab_check(path):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    topo = ms.get_topological_measures()
    ms.compute_selection_by_self_intersections_per_face()
    si = int(ms.current_mesh().selected_face_number())
    return {"boundary_edges": int(topo["boundary_edges"]), "non_two_manifold_edges": int(topo["non_two_manifold_edges"]),
            "non_two_manifold_vertices": int(topo["non_two_manifold_vertices"]), "holes": int(topo["number_holes"]),
            "connected_components": int(topo["connected_components_number"]), "self_intersecting_faces": si}
