"""CAD face id of every patch of a .dat surface, written next to it as <stem>.faces (read by
src/regions.hpp: loadFaceMap), so supports and loads can be put on CAD faces ("face_id" in a
bc_case JSON file).

Face ids are the Gmsh surface tags after gmsh.model.occ.importShapes of the STEP file, the
numbering python/geomio (cad_to_mesh, the tri6/TET10 meshes) and python/step_auto.py (without
--split) use, so a face picked on a geomio mesh or in a viewer is the same face here. Patches are
classified geometrically (5 points per patch projected onto every CAD face; the face with the
smallest worst distance wins), which also covers .dat files made with --split, whose patches lie
on pieces of the original faces, and files from step2dat.py.

The .faces file lists the face table too (id, kind, centre of mass, area), so an adapter can map
another face numbering (e.g. nurbTCD's build123d order, given as face centres) onto these ids.

usage: python3 python/face_ids.py in.step patches.dat [out.faces]
"""
import os
import sys

import gmsh
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import read_dat  # noqa: E402
from spline_checks import evaluate  # noqa: E402


def face_ids(step, dat):
    patches = read_dat(dat)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("faces")
        gmsh.model.occ.importShapes(step)
        gmsh.model.occ.synchronize()
        surfs = [s for _, s in gmsh.model.getEntities(2)]
        bb = np.array(gmsh.model.getBoundingBox(-1, -1))
        diag = float(np.linalg.norm(bb[3:] - bb[:3]))
        table = []
        for s in surfs:
            c = gmsh.model.occ.getCenterOfMass(2, s)
            table.append((s, gmsh.model.getType(2, s), c, gmsh.model.occ.getMass(2, s)))
        ids, worst = [], 0.0
        for P in patches:
            u0, u1, v0, v1 = P.U[0], P.U[-1], P.V[0], P.V[-1]
            X = np.array([evaluate(P, u0 + (u1 - u0) * a, v0 + (v1 - v0) * b)[0]
                          for a, b in [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]])
            best, bd = None, np.inf
            for s in surfs:
                Y = np.asarray(gmsh.model.getClosestPoint(2, s, X.ravel())[0]).reshape(-1, 3)
                d = np.linalg.norm(X - Y, axis=1).max()
                if d < bd:
                    best, bd = s, d
            ids.append(best)
            worst = max(worst, bd)
    finally:
        gmsh.finalize()
    return ids, table, worst / diag


def write_faces(path, ids, table, step):
    with open(path, "w") as f:
        f.write("# igabem face map: CAD face id (Gmsh surface tag of %s) per patch\n" % os.path.basename(step))
        f.write("# face table: id kind centre_x centre_y centre_z area\n")
        for s, kind, c, A in table:
            f.write("# face %d %s %.12g %.12g %.12g %.12g\n" % (s, kind, c[0], c[1], c[2], A))
        f.write("%d\n" % len(ids))
        for k in range(0, len(ids), 20):
            f.write(" ".join(str(i) for i in ids[k:k + 20]) + "\n")


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) < 2:
        print(__doc__)
        return 1
    step, dat = a[0], a[1]
    out = a[2] if len(a) > 2 else os.path.splitext(dat)[0] + ".faces"
    ids, table, rel = face_ids(step, dat)
    write_faces(out, ids, table, step)
    used = sorted(set(ids))
    print("%d patches on %d of %d CAD faces; worst patch-to-face distance %.2e of the size; wrote %s"
          % (len(ids), len(used), len(table), rel, out))
    for s, kind, c, A in table:
        n = ids.count(s)
        print("  face %3d %-10s centre (%9.4f %9.4f %9.4f) area %10.5f  %d patch(es)" % (s, kind, *c, A, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
