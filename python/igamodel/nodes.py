"""The node types of an IgBEM model, declared once, mirroring the case file
that app/bc_case reads (src/case_io.hpp).

    case                  geometry, faces map, cache, notes
      material  (one)     E, nu
      solver    (one)     n_gauss, far_tol
      grade     (one)     mesh graded towards supports (python/grade_dat.py)
      support   (many)    fixed | symmetry | spring | rbe2, on a region
      loadcase  (many)    a named load case; all share one operator
        load    (many)    force/moment | pressure | traction | bearing | on an rbe2 master
      output    (one)     vtu, csv

Prop keys are the case file's keys, so nothing is renamed between the tree,
the file and the solver. The rules refuse conflicting settings (two support
behaviours at once); a node still being filled in (no behaviour yet) is
allowed in the tree and refused when the case file is written.
"""
from __future__ import annotations

from .model import ModelError, NodeType, Prop, register

SUPPORT_KINDS = ("fix", "dofs", "spring", "rbe2", "symmetry")
LOAD_KINDS = ("pressure", "traction", "bearing", "rbe2")


def _one_of(what, props, keys):
    given = [k for k in keys if k in props]
    if len(given) > 1:
        raise ModelError(f"a {what} is one of {list(keys)}; this one sets {given}")
    return given[0] if given else None


def support_rule(props, region):
    kind = _one_of("support", props, SUPPORT_KINDS)
    if kind == "symmetry" and region is not None:
        raise ModelError("a symmetry support is a plane x|y|z = at and takes no region")
    for k in ("value", "snap"):
        if k in props and kind in ("spring", "symmetry"):
            raise ModelError(f"{k} does not apply to a {kind} support")
    for k in ("at", "tol"):
        if k in props and kind not in (None, "symmetry"):
            raise ModelError(f"{k} applies to a symmetry support only")


def load_rule(props, region):
    kind = _one_of("load", props, LOAD_KINDS)
    force_like = [k for k in ("force_N", "direction", "moment", "about") if k in props]
    if force_like and kind in ("pressure", "traction", "bearing"):
        raise ModelError(f"a {kind} load takes no {force_like}")
    if kind == "rbe2":
        if region is not None:
            raise ModelError("a load on an rbe2 master names the support, not a region")
        if "about" in props:
            raise ModelError("a load on an rbe2 master acts about the master's ref; about does not apply")


register(NodeType("case", "Model", "case", children=("material", "solver", "grade", "support", "loadcase", "output"),
                  props=(
    Prop("geometry", "str", doc="the .dat surface (step_auto.py / geomio), relative to the case file"),
    Prop("faces", "str", doc="CAD face map; unset = next to the geometry (python/face_ids.py)"),
    Prop("cache", "str", doc="operator cache folder, or none"),
    Prop("notes", "str", doc="the comment block at the top of the case file"),
), doc="One part, its supports, and its load cases."))

register(NodeType("material", "Material", "mat", singleton=True, props=(
    Prop("E", "float", "MPa"), Prop("nu", "float")), doc="Isotropic linear elastic."))

register(NodeType("solver", "Solver", "solver", singleton=True, props=(
    Prop("n_gauss", "int", doc="Gauss points per direction"),
    Prop("far_tol", "float", doc="far-field quadrature tolerance")), doc="Quadrature settings."))

register(NodeType("grade", "Grading", "grade", singleton=True, props=(
    Prop("faces", "ints", doc="CAD faces to grade towards"),
    Prop("inner", "ints", doc="faces graded from both sides"),
    Prop("h0", "float", doc="first knot span"), Prop("growth", "float")),
    doc="Refine the surface towards the supports by knot insertion."))

register(NodeType("support", "Support", "sup", region=True, rule=support_rule, props=(
    Prop("label", "str"),
    Prop("fix", "names", choices=("ux", "uy", "uz"), doc="fixed displacement components"),
    Prop("dofs", "dofs", doc="the same as fix, written as {ux: true, ...}"),
    Prop("value", "vec3", "mm", doc="prescribed displacement of the fixed components; unset = 0"),
    Prop("spring", "object", choices=("k", "K_total", "k_n", "k_t"),
         doc="elastic support: k [kx, ky, kz] | K_total | k_n, k_t (per area)"),
    Prop("rbe2", "object", choices=("ref", "hold", "k"),
         doc="rigid coupling to a master at ref (unset = centroid): hold [ux..rz], k 6 or 36 numbers"),
    Prop("symmetry", "choice", choices=("x", "y", "z"), doc="symmetry plane normal"),
    Prop("at", "float", "mm", doc="symmetry plane position"),
    Prop("tol", "float", "mm", doc="symmetry plane tolerance"),
    Prop("snap", "bool", doc="take patches covered at least half"),
), doc="A boundary condition: fixed, symmetry, spring or rbe2."))

register(NodeType("loadcase", "Load case", "lc", children=("load",), props=(
    Prop("name", "str"),
    Prop("top", "bool", doc="written as the case file's top-level loads (one-case form)"),
), doc="A named load case. All load cases share one assembled operator."))

register(NodeType("load", "Load", "load", region=True, rule=load_rule, props=(
    Prop("label", "str"),
    Prop("force_N", "float", "N", doc="resultant force"),
    Prop("direction", "vec3", doc="unset = push into the face"),
    Prop("moment", "vec3", "N mm"),
    Prop("about", "vec3", "mm", doc="moment centre; unset = region centroid"),
    Prop("pressure", "float", "MPa"),
    Prop("traction", "vec3", "MPa"),
    Prop("bearing", "object", choices=("force_N", "direction"), doc="pin bearing load"),
    Prop("rbe2", "str", doc="label of the rbe2 support whose master takes force_N/direction/moment"),
), doc="A load: force and moment, pressure, traction, bearing, or on an rbe2 master."))

register(NodeType("output", "Output", "out", singleton=True, props=(
    Prop("vtu", "str"), Prop("csv", "str")), doc="Where bc_case writes results."))
