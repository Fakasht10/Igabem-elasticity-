"""The node types of an IgBEM model, declared once, mirroring the case file
that app/bc_case reads (src/case_io.hpp).

    case                  geometry, faces map, cache, notes
      material  (one)     E, nu
      solver    (one)     n_gauss, far_tol
      grade     (one)     mesh graded towards supports (python/grade_dat.py)
      selection (many)    a named surface: a region supports and loads refer to by {"name": ...}
      support   (many)    fixed | symmetry | spring | rbe2, on a region
      loadcase  (many)    a named load case; all share one operator
        load    (many)    force/moment | pressure | traction | bearing | on an rbe2 master
      output    (one)     vtu, csv
      cad       (one)     the STEP the geometry is made from (step_auto.py) and the TET10 mesh of assessments
      assessment (many)   TCD failure check of one feature point (python/failure/assess.py)
      submodel  (many)    a box of the part meshed finer, driven by the global solve (igamodel/submodel.py)

Prop keys are the case file's keys, so nothing is renamed between the tree,
the file and the solver. cad and assessment are not part of the case file
(to_case skips them); they live in the model file and drive igamodel.run.
The rules refuse conflicting settings (two support behaviours at once); a node still being filled in (no behaviour yet) is
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


def selection_rule(props, region):
    if region is not None and "name" in region:
        raise ModelError("a named surface is a region, not another name")


def assessment_rule(props, region):
    if "card" in props and "K_Ic" in props:
        raise ModelError("an assessment takes K_Ic from its card or from K_Ic, not both")


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


register(NodeType("case", "Model", "case", children=("material", "solver", "grade", "selection", "support", "loadcase", "output",
                                                     "cad", "assessment", "submodel"),
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

register(NodeType("selection", "Named surface", "sel", region=True, rule=selection_rule, props=(
    Prop("name", "str", doc="what supports and loads call it"),
), doc="A name for a region (CAD faces, patches or a geometric selection), so supports and loads can refer to "
       "it and survive a re-pick."))

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

register(NodeType("cad", "CAD", "cad", singleton=True, props=(
    Prop("step", "str", doc="the STEP file, relative to the base folder run.assess is given"),
    Prop("h", "float", "mm", doc="IgBEM knot span; unset = 3"),
    Prop("curvature", "int", doc="knot spans per radian on curved edges; unset = 6"),
    Prop("degree", "int", doc="spline degree; unset = 2"),
), doc="The part's CAD. Generating the geometry (step_auto.py) sets the case's geometry."))

register(NodeType("assessment", "Assessment", "tcd", rule=assessment_rule, props=(
    Prop("label", "str"),
    Prop("point", "vec3", "mm", doc="the feature: a notch root, a hot spot"),
    Prop("cases", "names", doc="load cases to assess; unset = all"),
    Prop("card", "str", doc="material card (failure/materials.py) giving K_Ic and sigma_uts"),
    Prop("K_Ic", "float", "MPa sqrt(mm)", doc="fracture toughness, without a card"),
    Prop("sigma_uts", "float", "MPa", doc="inherent strength sigma_0; with a card, overrides the card's"),
    Prop("rho", "float", "mm", doc="notch root radius if known (blunt-root floor, mesh focus)"),
    Prop("state", "choice", choices=("plane_strain", "plane_stress"), doc="for the energy radii; unset = plane strain"),
    Prop("mesh_h", "float", "mm", doc="TET10 size; unset = 2.5"),
    Prop("mesh_curvature", "int", doc="TET10 elements per radian; unset = 16"),
    Prop("focus_radius", "float", "mm", doc="refinement sphere round the point; unset = 1.5 L"),
    Prop("focus_h", "float", "mm", doc="element size in it; unset = min(0.15, rho / 6), 0.15 without rho"),
    Prop("submodel", "str", doc="label of the submodel to assess on; unset = the whole part"),
), doc="Theory of Critical Distances at one point: the point and energy point methods give the load factor at failure."))

def submodel_rule(props, region):
    lo, hi = props.get("lo"), props.get("hi")
    if lo is not None and hi is not None and any(b <= a for a, b in zip(lo, hi)):
        raise ModelError("a submodel box needs lo < hi in x, y and z")


register(NodeType("submodel", "Submodel", "sub", rule=submodel_rule, props=(
    Prop("label", "str"),
    Prop("lo", "vec3", "mm", doc="box corner"), Prop("hi", "vec3", "mm", doc="opposite corner"),
    Prop("h", "float", "mm", doc="IgBEM knot span in the box; unset = 0.5"),
    Prop("curvature", "int", doc="knot spans per radian on curved edges; unset = 6"),
    Prop("degree", "int", doc="unset = 2"),
), doc="The part inside a box, meshed at its own h, its cut faces driven by the global displacement."))
