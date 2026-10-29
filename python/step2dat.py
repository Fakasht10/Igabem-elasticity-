"""STEP (or any OpenCASCADE shape) -> conforming multipatch NURBS in Bembel .dat format.

Pipeline (OpenCASCADE through the `cadquery-ocp` wheel: `pip install cadquery-ocp`):

  1. read the STEP file and, optionally, split its boundary shell with planes so that
     every face is bounded by one wire of exactly four edges (a quad layout). Faces with
     holes or another number of edges are reported and refused;
  2. bring every CAD edge curve to one degree and one knot vector, shared by all faces
     that use the edge and by the opposite edge of each of those faces (a fixed point
     over the edge chains of the quad layout), so neighbouring patches are conforming;
  3. untrim each face: the Coons patch of its four exact edge curves, built in
     homogeneous coordinates. The boundary is the CAD edge itself, so patches sharing an
     edge share its control points and weights (watertight). The interior is exact for
     planar and ruled faces and an approximation otherwise; its distance to the CAD face
     is measured and reported;
  4. orient every patch so x_u x x_v points out of the solid; optionally elevate the
     degree (of the edge curves, before step 3) and refine uniformly;
  5. check conformity and the enclosed volume (divergence theorem vs. OpenCASCADE) and
     write the .dat file.

usage: python3 python/step2dat.py in.step out.dat [--split nx,ny,nz,px,py,pz ...]
                                  [--degree p] [--refine k]
"""

import argparse
import os
import sys

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.collections import List_TopoDS_Shape
from OCP.Geom import Geom_TrimmedCurve
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.GeomConvert import GeomConvert
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_geometries import Patch, insert_knot_curve, refine_uniform, reversed_patch, write_dat  # noqa: E402
from spline_checks import _basis, conformity_gap, evaluate, spline_volume  # noqa: E402,F401


# ---------------------------------------------------------------- CAD access
def read_step(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != 1:
        raise RuntimeError("cannot read " + path)
    r.TransferRoots()
    return r.OneShape()


def write_step(shape, path):
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(path) != 1:
        raise RuntimeError("cannot write " + path)


def _explore(shape, kind):
    e = TopExp_Explorer(shape, kind)
    while e.More():
        yield e.Current()
        e.Next()


def faces(shape):
    return [TopoDS.Face(f) for f in _explore(shape, TopAbs_FACE)]


def split(shape, planes, size=1e3):
    """Split the boundary faces of `shape` by planes given as (normal, point). The shells,
    not the solid, are split, so no internal faces are created."""
    sp = BRepAlgoAPI_Splitter()
    args, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
    for s in _explore(shape, TopAbs_SHELL):
        args.Append(s)
    for n, p in planes:
        pl = gp_Pln(gp_Pnt(*p), gp_Dir(*n))
        tools.Append(BRepBuilderAPI_MakeFace(pl, -size, size, -size, size).Face())
    sp.SetArguments(args)
    sp.SetTools(tools)
    sp.Build()
    return sp.Shape()


def wire_edges(face):
    ex, out = BRepTools_WireExplorer(BRepTools.OuterWire_s(face), face), []
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def is_quad(face):
    return len(list(_explore(face, TopAbs_WIRE))) == 1 and len(wire_edges(face)) == 4


def face_census(shape):
    """(#faces, #faces bounded by one wire of four edges)."""
    F = faces(shape)
    return len(F), sum(is_quad(f) for f in F)


def cad_volume(shape):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return g.Mass()


# ---------------------------------------------------------------- edge curves
class Curve:
    """Homogeneous B-spline curve: degree p, knots U on [0, 1], points Pw (n, 4)."""

    def __init__(self, p, U, Pw):
        self.p, self.U, self.Pw = p, np.asarray(U, float), np.asarray(Pw, float)

    def reversed(self):
        return Curve(self.p, 1.0 - self.U[::-1], self.Pw[::-1].copy())


def _occ_curve(edge, degree):
    """The edge's 3D curve in its stored direction as a Curve on [0, 1], degree >= degree."""
    c = BRep_Tool.Curve_s(edge, 0.0, 0.0)
    f, l = BRep_Tool.Range_s(edge)
    b = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(c, f, l))
    if b.Degree() < degree:
        b.IncreaseDegree(degree)
    U = []
    for i in range(1, b.NbKnots() + 1):
        U += [b.Knot(i)] * b.Multiplicity(i)
    U = np.array(U)
    Pw = []
    for i in range(1, b.NbPoles() + 1):
        P, w = b.Pole(i), b.Weight(i)
        Pw.append((P.X() * w, P.Y() * w, P.Z() * w, w))
    return Curve(b.Degree(), (U - U[0]) / (U[-1] - U[0]), Pw)


def _interior(C):
    """Interior knots of C as {value: multiplicity}."""
    out = {}
    for u in np.round(C.U[C.p + 1:len(C.U) - C.p - 1], 12):
        out[u] = out.get(u, 0) + 1
    return out


def _flip(K, f):
    return {round(1.0 - u, 12): m for u, m in K.items()} if f else dict(K)


def _union(A, B):
    return {u: max(A.get(u, 0), B.get(u, 0)) for u in set(A) | set(B)}


class EdgeTable:
    """One Curve per CAD edge, made compatible across the quad layout: opposite edges of
    every face get the same degree and (mirrored) knots, so do all faces sharing an edge."""

    def __init__(self, quad_faces, degree=None):
        self.edges, self.faces = [], []  # faces[i] = [(edge id, curve runs against wire?)]
        for f in quad_faces:
            row = []
            for e in wire_edges(f):
                k = next((i for i, g in enumerate(self.edges) if g.IsSame(e)), None)
                if k is None:
                    k = len(self.edges)
                    self.edges.append(e)
                # BRep_Tool.Curve follows the stored curve; the wire uses it reversed if
                # the edge occurrence in this wire is REVERSED
                row.append((k, e.Orientation() == TopAbs_REVERSED))
            self.faces.append(row)
        base = [_occ_curve(e, 1) for e in self.edges]
        deg = [max(c.p, degree or 1) for c in base]
        knots = [_interior(c) for c in base]
        # fixed point over chains of opposite edges. In face coordinates the edges a = 0
        # and b = 2 both run in u: c0 along the wire, c2 against it (and likewise 1, 3 in v)
        for _ in range(1000):
            changed = False
            for row in self.faces:
                for a, b in ((0, 2), (1, 3)):
                    (ia, ra), (ib, rb) = row[a], row[b]
                    fa, fb = ra, not rb  # flip that maps stored parameter -> face parameter
                    S = _union(_flip(knots[ia], fa), _flip(knots[ib], fb))
                    d = max(deg[ia], deg[ib])
                    na, nb = _flip(S, fa), _flip(S, fb)
                    if (d, d, na, nb) != (deg[ia], deg[ib], knots[ia], knots[ib]):
                        deg[ia] = deg[ib] = d
                        knots[ia], knots[ib] = na, nb
                        changed = True
            if not changed:
                break
        self.curves = []
        for e, d, K in zip(self.edges, deg, knots):
            C = _occ_curve(e, d)
            have = _interior(C)
            for u, m in sorted(K.items()):
                for _ in range(m - have.get(u, 0)):
                    C.U, C.Pw = insert_knot_curve(C.p, C.U, C.Pw, u)
            self.curves.append(C)

    def face_curves(self, i):
        """The four curves of face i, each running along the wire."""
        return [self.curves[k].reversed() if r else self.curves[k] for k, r in self.faces[i]]


# ---------------------------------------------------------------- Coons patch
def _greville(p, U):
    return np.array([np.mean(U[i + 1:i + p + 1]) for i in range(len(U) - p - 1)])


def coons(c0, c1, c2, c3):
    """Homogeneous Coons patch of the loop c0: P00->P10, c1: P10->P11, c2: P11->P01,
    c3: P01->P00. Opposite curves must share degree and (mirrored) knots."""
    A, B = c0, c2.reversed()  # v = 0 and v = 1, running in u
    D, E = c3.reversed(), c1  # u = 0 and u = 1, running in v
    assert A.p == B.p and np.allclose(A.U, B.U) and D.p == E.p and np.allclose(D.U, E.U)
    gu, gv = _greville(A.p, A.U)[None, :, None], _greville(D.p, D.U)[:, None, None]
    P00, P10, P01, P11 = A.Pw[0], A.Pw[-1], B.Pw[0], B.Pw[-1]
    # linear blends are exact in any spline space through their Greville values
    Rv = (1 - gv) * A.Pw[None] + gv * B.Pw[None]
    Ru = (1 - gu) * D.Pw[:, None] + gu * E.Pw[:, None]
    Bl = (1 - gu) * (1 - gv) * P00 + gu * (1 - gv) * P10 + (1 - gu) * gv * P01 + gu * gv * P11
    return Patch(A.p, D.p, A.U, D.U, Rv + Ru - Bl)


# ---------------------------------------------------------------- evaluation
def _cad_normal(face, x):
    """Outward unit normal of the CAD face at the foot point of x."""
    proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*x), BRep_Tool.Surface_s(face))
    u, v = proj.LowerDistanceParameters()
    n = np.array(BRepLProp_SLProps(BRepAdaptor_Surface(face), u, v, 1, 1e-9).Normal().Coord())
    return -n if face.Orientation() == TopAbs_REVERSED else n


def deviation(P, face, m=9):
    """Max distance from m x m patch points to the CAD face's surface."""
    S, d = BRep_Tool.Surface_s(face), 0.0
    for u in np.linspace(0, 1, m):
        for v in np.linspace(0, 1, m):
            pj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*evaluate(P, u, v)[0]), S)
            if pj.NbPoints():
                d = max(d, pj.LowerDistance())
    return d


# ---------------------------------------------------------------- checks
# ---------------------------------------------------------------- driver
def to_patches(boundary, degree=None, refine=0):
    """Untrimmed, conforming, outward-oriented patches of every face of `boundary`,
    and the distance of each to its CAD face."""
    F = faces(boundary)
    bad = [i for i, f in enumerate(F) if not is_quad(f)]
    if bad:
        raise ValueError("%d of %d faces are not four-sided: split them first" % (len(bad), len(F)))
    T = EdgeTable(F, degree)
    out, devs = [], []
    for i, f in enumerate(F):
        P = coons(*T.face_curves(i))
        x, xu, xv = evaluate(P, 0.5, 0.5)
        if np.dot(np.cross(xu, xv), _cad_normal(f, x)) < 0:
            P = reversed_patch(P)
        devs.append(deviation(P, f))
        for _ in range(refine):
            refine_uniform(P)
        out.append(P)
    return out, np.array(devs)


def convert(solid, boundary, out, degree=None, refine=0, name="step"):
    """Write the patches of `boundary` (the solid's faces, possibly split) to `out`."""
    patches, dev = to_patches(boundary, degree, refine)
    gap, nshared, lonely = conformity_gap(patches)
    v_cad, v_spl = cad_volume(solid), spline_volume(patches)
    write_dat(out, patches, name)
    return dict(patches=len(patches), max_dev=dev.max(), conformity_gap=gap, shared_edges=nshared,
                open_edges=lonely, vol_cad=v_cad, vol_spline=v_spl, vol_rel_err=abs(v_spl - v_cad) / abs(v_cad))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step")
    ap.add_argument("out")
    ap.add_argument("--split", action="append", default=[], help="cutting plane nx,ny,nz,px,py,pz (repeatable)")
    ap.add_argument("--degree", type=int)
    ap.add_argument("--refine", type=int, default=0)
    a = ap.parse_args(argv)
    solid = shape = read_step(a.step)
    if a.split:
        pl = [tuple(map(float, s.split(","))) for s in a.split]
        shape = split(solid, [(x[:3], x[3:]) for x in pl])
    nf, nq = face_census(shape)
    print("faces %d, four-sided %d" % (nf, nq))
    if nq < nf:
        sys.exit("not a quad layout: add --split planes so every face has four edges")
    r = convert(solid, shape, a.out, a.degree, a.refine, os.path.basename(a.out))
    for k, v in r.items():
        print("%-15s %s" % (k, v))
    return r


if __name__ == "__main__":
    main()
