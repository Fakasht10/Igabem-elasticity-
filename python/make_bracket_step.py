"""Write a bracket-like part as a STEP file from a CAD kernel (OpenCASCADE): a stepped flat
bar, width D = 1 for x in [-L1, 0] and d = 1/2 for x in [0, L2], thickness t, shoulder
fillets of radius r at x = 0, and a through-hole of diameter dh at (-L1/2, 0). Clamped at
x = -L1 and pulled at x = L2, it has two textbook stress raisers: the shoulder fillets
(Peterson, opposite shoulder fillets in tension) and the hole (Heywood, finite width).

usage: python3 python/make_bracket_step.py [out.step]
"""
import sys

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from step2dat import write_step  # noqa: E402

DIMS = dict(D=1.0, d=0.5, t=0.2, r=0.1, L1=1.5, L2=1.5, dh=0.3)


def bracket_solid(D=1.0, d=0.5, t=0.2, r=0.1, L1=1.5, L2=1.5, dh=0.3):
    wide = BRepPrimAPI_MakeBox(gp_Pnt(-L1, -D / 2, -t / 2), L1, D, t).Shape()
    narrow = BRepPrimAPI_MakeBox(gp_Pnt(0, -d / 2, -t / 2), L2, d, t).Shape()
    fuse = BRepAlgoAPI_Fuse(wide, narrow).Shape()
    u = ShapeUpgrade_UnifySameDomain(fuse, True, True, True)
    u.Build()
    body = u.Shape()
    # the two concave shoulder edges: parallel to z at (0, +-d/2)
    fil = BRepFilletAPI_MakeFillet(body)
    e = TopExp_Explorer(body, TopAbs_EDGE)
    while e.More():
        ed = TopoDS.Edge(e.Current())
        f, l = BRep_Tool.Range_s(ed)
        c = BRep_Tool.Curve_s(ed, 0.0, 0.0)
        a, b = c.Value(f), c.Value(l)
        if abs(a.X()) < 1e-9 and abs(b.X()) < 1e-9 and abs(abs(a.Y()) - d / 2) < 1e-9 \
                and abs(a.Y() - b.Y()) < 1e-9:
            fil.Add(r, ed)
        e.Next()
    body = fil.Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-L1 / 2, 0, -t), gp_Dir(0, 0, 1)), dh / 2, 2 * t).Shape()
    return BRepAlgoAPI_Cut(body, hole).Shape()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "geo/step/bracket.step"
    write_step(bracket_solid(), out)
    print("wrote", out)
