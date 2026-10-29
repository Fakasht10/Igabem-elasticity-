"""Write the plate with a circular through-hole as a STEP file from a CAD kernel (OpenCASCADE):
box [-W/2, W/2]^2 x [-t/2, t/2] minus a cylinder of radius a on the z axis. It is the CAD
counterpart of make_geometries.plate_hole and the end-to-end test of python/step2dat.py.

usage: python3 python/make_plate_step.py [t] [out.step]
"""
import sys

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from step2dat import write_step  # noqa: E402


def plate_hole_solid(t, a=0.2, W=1.0):
    box = BRepPrimAPI_MakeBox(gp_Pnt(-W / 2, -W / 2, -t / 2), W, W, t).Shape()
    # seam of the periodic cylinder on the diagonal x = y, where the quad layout cuts anyway
    ax = gp_Ax2(gp_Pnt(0, 0, -t), gp_Dir(0, 0, 1), gp_Dir(1, 1, 0))
    cyl = BRepPrimAPI_MakeCylinder(ax, a, 2 * t).Shape()
    return BRepAlgoAPI_Cut(box, cyl).Shape()


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    out = sys.argv[2] if len(sys.argv) > 2 else "geo/step/plate_hole_t%g.step" % t
    write_step(plate_hole_solid(t), out)
    print("wrote", out)
