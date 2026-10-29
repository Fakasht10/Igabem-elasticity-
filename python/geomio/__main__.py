import argparse
import time

from . import load, save, segment_faces


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m geomio", description="convert CAD/mesh files, keeping face ids")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--h", type=float, default=None, help="CAD input: element size (default diag/20)")
    ap.add_argument("--curvature", type=int, default=16, help="CAD input: elements per 2 pi of curvature")
    ap.add_argument("--ascii", action="store_true", help="STL output: ASCII, one solid per face")
    ap.add_argument("--segment", type=float, default=None,
                    help="regrow face ids across edges smoother than this angle (degrees), e.g. for a scanned STL")
    ap.add_argument("--order", type=int, default=1, choices=(1, 2),
                    help="2: curved tri6 (+ TET10 with --volume), mid-edge nodes on the CAD; output .vtu or .msh")
    ap.add_argument("--volume", action="store_true", help="--order 2: also mesh the volume with TET10")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    if a.order == 2:
        from .highorder import cad_to_mesh2, write_vtu
        msh = a.out if a.out.lower().endswith(".msh") else None
        m, info = cad_to_mesh2(a.inp, h=a.h, curvature=a.curvature, volume=a.volume, msh=msh)
        if a.out.lower().endswith(".vtu"):
            write_vtu(a.out, m)
        elif not msh:
            save(a.out, m.corners())            # STL/3MF hold flat triangles only: the corner mesh
        info.pop("face_area")
        info["seconds"] = time.perf_counter() - t0
        for k, v in info.items():
            print("%-24s %s" % (k, v))
        return
    cad = a.inp.lower().rsplit(".", 1)[-1] in ("step", "stp", "iges", "igs", "brep")
    m, info = load(a.inp, **(dict(h=a.h, curvature=a.curvature) if cad else {}))
    if a.segment is not None:
        m = segment_faces(m, a.segment)
    save(a.out, m, **(dict(binary=not a.ascii) if a.out.lower().endswith(".stl") else {}))
    info.update(faces=len(m.faces()), triangles=len(m.T), volume=m.volume(), seconds=time.perf_counter() - t0)
    for k, v in info.items():
        print("%-20s %s" % (k, v))


if __name__ == "__main__":
    main()
