"""Geometry import/export for the IgBEM and cut-cell paths: face-tagged triangle meshes from STEP/IGES/BREP (Gmsh) and
STL/3MF files, repaired to closed outward surfaces, written back to STL/3MF with the CAD face id per triangle.

    from geomio import cad_to_mesh, read_stl, write_stl, write_3mf
    m, info = cad_to_mesh("geo/step/bracket.step", curvature=16)   # m.V, m.T, m.fid (Gmsh surface tag = CAD face)
    write_stl("bracket.stl", m)                                     # binary, face id in the attribute word
    m2, info = read_stl("bracket.stl")                              # welded, oriented outward, ids back

    python3 -m geomio in.step|in.stl|in.3mf out.stl|out.3mf [--h H] [--curvature N] [--ascii] [--segment DEG]

The STL/3MF handling borrows from materializr (github.com/materializr-cad/materializr, GPLv3, src/io/StlIO.cpp,
StlExport.cpp, ThreeMfExport.cpp, commit c5f9aa5); see the module headers for what was taken and what changed."""
from .mesh import TriMesh, edge_report, is_watertight, orient, repair, segment_faces, weld, winding_number  # noqa
from .stl import StlError, read_stl, write_stl  # noqa: F401
from .threemf import read_3mf, write_3mf  # noqa: F401


def cad_to_mesh(*a, **k):
    from .cad import cad_to_mesh as f   # gmsh is imported only when a CAD file is read
    return f(*a, **k)


def load(path, **k):
    """any supported file -> (TriMesh, info)."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("step", "stp", "iges", "igs", "brep"):
        return cad_to_mesh(path, **k)
    if ext == "stl":
        return read_stl(path, **k)
    if ext == "3mf":
        m, info = repair(read_3mf(path))
        return m, info
    raise ValueError("unsupported geometry file: " + path)


def save(path, m, **k):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "stl":
        return write_stl(path, m, **k)
    if ext == "3mf":
        return write_3mf(path, m, **k)
    raise ValueError("unsupported output file: " + path)
