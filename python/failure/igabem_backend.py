"""IgBEM behind nurb-fea-tcd's failure layer: the same result contract as feakit/bem_backend.py (bemkit), so the
vendored fracture.py (hot spots, notch apex and bisector, MATS, CP-series regression, TCD / Energy TCD / ASED) and
weibull.py run unchanged on an IgBEM solution.

    mesh   = fea_mesh(step, h=2.0, focus=[(x, y, z, r, hf)])          # TET10 + curved tri6 skin, CAD face ids
    geo    = igabem_geometry(step, workdir, h=3.0, focus=[...])          # STEP -> multipatch .dat + .faces (cached)
    run    = Solve(geo, case, E, nu)                                   # bc_case --serve: one LU, kept alive
    result = fields(run, mesh, probe_point=root, probe_radius=2.0)     # the dict fracture.py / the report consume
    sig    = run.stress(points, on_surface)                            # exact probes later, no new solve

What IgBEM supplies, as bemkit does in nurb-fea-tcd:
  * stress at the TET10 nodes: surface nodes by the boundary recovery at their projection onto the spline surface,
    interior nodes by the near-boundary-safe Somigliana evaluation (src/interior.hpp); exact BEM values, not
    recovered ones;
  * the Gauss-point cloud for the path regression and the Weibull field: the TET10 mesh's own 4-point rule, the
    stress interpolated from the exact nodal tensors and re-evaluated exactly within `probe_radius` of the feature
    and in the two element rings round sharp CAD edges (bemkit's default "nodal" recipe; the nodal tensor on a sharp
    edge is finite, so interpolation flattens the singular rise there);
  * displacement: on the surface only; inside, a harmonic extension for display, never used by a reported number.

Face ids are Gmsh surface tags throughout (geomio, step_auto, face_ids.py). Cases written for nurb-fea-tcd use
build123d face indices; `feakit_face_map` translates them by face centre.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from .fields import _sig6, edge_rings, tet10_quadrature

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
ROOT = os.path.dirname(PY)
BC_CASE = os.environ.get("IGABEM_BC_CASE", os.path.join(ROOT, "build", "bc_case"))
GMSH_TET10_TO_ABAQUS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
MIN_SPAN_JACOBIAN = 0.15      # step_auto check: below this an element's map nearly collapses (false stress spikes)


# ------------------------------------------------------------------------------------------- the FEA-side mesh
@dataclass
class FeaMesh:
    """The attributes fracture.py reads from feakit's GmshMesh, built from a geomio second-order mesh.
    cells are TET10 in Abaqus order (corners, then (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)), as feakit stores them;
    surface_triangles are the four corner sub-triangles of every tri6 (feakit's convention)."""
    points: np.ndarray
    cells: np.ndarray
    face_nodes: dict
    surface_triangles: np.ndarray
    surface_face_ids: np.ndarray
    surface_tri6: np.ndarray
    surface_tri6_face_ids: np.ndarray
    element_type: str = "TET10"
    info: dict = field(default_factory=dict)

    @property
    def n_nodes(self):
        return len(self.points)

    @property
    def corners(self):
        return self.cells[:, :4]


def fea_mesh(step, h, curvature=16, focus=()):
    """TET10 mesh of a STEP solid with mid-nodes on the CAD and refinement spheres `focus` = (x, y, z, r, hf)."""
    sys.path.insert(0, PY)
    from geomio.highorder import cad_to_mesh2
    m, info = cad_to_mesh2(step, h=h, curvature=curvature, volume=True, focus=list(focus))
    if info.get("tet_edges_straightened"):
        info["warning"] = f"{info['tet_edges_straightened']} TET10 edges straightened (folded mid-nodes)"
    T6, F = m.T6, m.fid
    c0, c1, c2, m01, m12, m20 = T6.T
    tri = np.vstack([np.column_stack(t) for t in ((c0, m01, m20), (m01, c1, m12), (m20, m12, c2), (m01, m12, m20))])
    tri = tri.reshape(4, -1, 3).transpose(1, 0, 2).reshape(-1, 3)
    face_nodes = {int(s): np.unique(T6[F == s]) for s in np.unique(F)}
    return FeaMesh(points=m.X, cells=m.tet10[:, GMSH_TET10_TO_ABAQUS], face_nodes=face_nodes,
                   surface_triangles=tri, surface_face_ids=np.repeat(F, 4), surface_tri6=T6,
                   surface_tri6_face_ids=F, info=info)


def feakit_face_map(step, centres):
    """{build123d face index: Gmsh surface tag} for a nurb-fea-tcd case (its <part>.step.faces.json "centres")."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("facemap")
    try:
        gmsh.model.occ.importShapes(step)
        gmsh.model.occ.synchronize()
        tags = [t for _, t in gmsh.model.getEntities(2)]
        com = np.array([gmsh.model.occ.getCenterOfMass(2, t) for t in tags])
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()
    out = {}
    for i, c in enumerate(np.asarray(centres, float)):
        d = np.linalg.norm(com - c, axis=1)
        out[i] = int(tags[int(np.argmin(d))])
    if len(set(out.values())) != len(out):
        raise ValueError("face centres do not map one to one onto the STEP's faces")
    return out


# ------------------------------------------------------------------------------------------- the IgBEM geometry
def igabem_geometry(step, workdir, h=3.0, curvature=6, degree=2, focus=(), retries=3, log=print):
    """STEP -> conforming multipatch .dat (python/step_auto.py) and its CAD face ids (.faces), cached in workdir by
    the STEP's bytes and the settings. A layout whose fitted elements nearly collapse (min span Jacobian ratio below
    MIN_SPAN_JACOBIAN) gives false stress spikes on the surface, so it is rejected and the focus radius shrunk."""
    os.makedirs(workdir, exist_ok=True)
    with open(step, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    focus = [tuple(map(float, x)) for x in focus]
    for attempt in range(retries):
        key = hashlib.sha256(json.dumps([digest, h, curvature, degree, focus]).encode()).hexdigest()[:12]
        dat = os.path.join(workdir, f"geo_{key}.dat")
        meta = dat + ".json"
        if not os.path.exists(meta):
            t = time.perf_counter()
            cmd = [sys.executable, os.path.join(PY, "step_auto.py"), step, dat, "--h", str(h), "--curvature",
                   str(curvature), "--degree", str(degree)] + sum((["--focus", ",".join(map(str, x))] for x in focus), [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError("step_auto failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
            checks = {k: v for k, v in (ln.split(None, 1) for ln in r.stdout.splitlines() if ln.strip())}
            subprocess.run([sys.executable, os.path.join(PY, "face_ids.py"), step, dat], check=True, capture_output=True)
            info = {"min_span_jacobian": float(checks["min_span_jacobian"]), "patches": int(checks["patches"]),
                    "exact_patches": int(checks["exact_patches"]), "vol_rel_err": float(checks["vol_rel_err"]),
                    "seconds": time.perf_counter() - t, "h": h, "curvature": curvature, "degree": degree, "focus": focus}
            with open(meta, "w") as f:
                json.dump(info, f)
        with open(meta) as f:
            info = json.load(f)
        if info["min_span_jacobian"] >= MIN_SPAN_JACOBIAN:
            return dat, info
        log(f"IgBEM layout rejected: an element map nearly collapses (span Jacobian ratio "
            f"{info['min_span_jacobian']:.3g} < {MIN_SPAN_JACOBIAN}); retrying with a smaller focus radius")
        if not focus:
            break
        focus = [(x, y, z, 0.6 * r, hf) for x, y, z, r, hf in focus]
    raise RuntimeError("no IgBEM layout passed the element-map check; change h or the focus sphere")


# ------------------------------------------------------------------------------------------- the solve
class Solve:
    """One IgBEM solve of a case kept alive in `bc_case --serve`: stress at any points without a new LU.
    `case` is the bc_case JSON body without "geometry" and "material" (supports, cases or loads, solver options)."""

    def __init__(self, geometry, case, E, nu, workdir=None, log=print):
        self.dir = workdir or tempfile.mkdtemp(prefix="igabem_")
        os.makedirs(self.dir, exist_ok=True)
        body = dict(case, geometry=os.path.abspath(geometry), material={"E": float(E), "nu": float(nu)})
        body.setdefault("cache", "none")
        self.case_path = os.path.join(self.dir, "case.json")
        with open(self.case_path, "w") as f:
            json.dump(body, f, indent=1)
        t = time.perf_counter()
        self.proc = subprocess.Popen([BC_CASE, self.case_path, "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.dir)
        self.report = self._until_ready()
        self.seconds = time.perf_counter() - t
        self.cases = [ln.split()[1] for ln in self.report if ln.startswith("case ")]
        self.n = 0
        log(f"IgBEM solved in {self.seconds:.1f} s: " + next((ln[2:] for ln in self.report if "unknowns" in ln), ""))

    def _until_ready(self):
        lines = []
        for ln in self.proc.stdout:
            ln = ln.rstrip("\n")
            if ln == "ready":
                return lines
            lines.append(ln)
        raise RuntimeError("bc_case ended before the solve finished:\n" + "\n".join(lines[-20:]))

    def stress(self, points, on_surface=None, case=None):
        """(n, 3, 3) stress tensors and (n,) winding numbers (1 inside, 0 outside, 0.5 surface)."""
        return self.stress_all(points, on_surface)[case or self.cases[0]]

    def stress_all(self, points, on_surface=None):
        """{load case: (stress, winding)}: one evaluation gives every load case (they share the operator)."""
        P = np.atleast_2d(np.asarray(points, float))
        on = np.zeros(len(P), bool) if on_surface is None else np.broadcast_to(np.asarray(on_surface, bool), len(P))
        self.n += 1
        src = os.path.join(self.dir, f"pts_{self.n}.txt")
        np.savetxt(src, np.column_stack([P, on.astype(float)]), fmt="%.17g")
        prefix = os.path.join(self.dir, f"out_{self.n}")
        self.proc.stdin.write(f"points {src} {prefix}\n")
        self.proc.stdin.flush()
        self.last_report = self._until_ready()
        out = {}
        for name in self.cases:
            d = np.loadtxt(f"{prefix}_{name}.txt", ndmin=2)
            s = np.empty((len(d), 3, 3))
            s[:, 0, 0], s[:, 1, 1], s[:, 2, 2] = d[:, 3], d[:, 4], d[:, 5]
            s[:, 0, 1] = s[:, 1, 0] = d[:, 6]
            s[:, 1, 2] = s[:, 2, 1] = d[:, 7]
            s[:, 2, 0] = s[:, 0, 2] = d[:, 8]
            out[name] = (s, d[:, 9])
        for path in [src] + [f"{prefix}_{c}.txt" for c in self.cases]:
            if os.path.exists(path):
                os.unlink(path)
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------------------------------- fields on the mesh
def principal(s):
    return np.linalg.eigvalsh(s)


def sed(s, E, nu):
    """Strain energy density (n,) of stress tensors (n, 3, 3), linear isotropic."""
    tr = np.trace(s, axis1=1, axis2=2)
    return (np.einsum("nij,nij->n", s, s) * (1 + nu) - nu * tr ** 2) / (2 * E)


def von_mises(s):
    d = s - np.trace(s, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    return np.sqrt(1.5 * np.einsum("nij,nij->n", d, d))


def fields(run, mesh, E, nu, probe_point=None, probe_radius=0.0, exact_edges=True, log=print):
    """The solve mapped onto the FEA mesh, in the keys of feakit's _solve_once / bem_backend.solve_bem that the
    failure layer reads: nodal fields (exact), the Gauss cloud near the feature (exact), the Weibull field.
    exact_edges=False skips the exact re-evaluation round sharp edges (only the Weibull field needs it)."""
    timings = {}
    pts = np.asarray(mesh.points, float)
    surf = np.unique(mesh.surface_tri6)
    t = time.perf_counter()
    on = np.zeros(len(pts), bool)
    on[surf] = True
    sig, wind = run.stress(pts, on)
    timings["nodes_s"] = time.perf_counter() - t
    outside = np.flatnonzero(~on & (wind < 0.5))
    if len(outside):
        log(f"{len(outside)} interior TET10 nodes lie outside the spline surface (fitted patches); "
            "they take the boundary recovery at their projection")
        sig[outside], _ = run.stress(pts[outside], True)
    s1, w, vm = principal(sig)[:, -1], sed(sig, E, nu), von_mises(sig)

    t = time.perf_counter()
    Pq, wq, sq = tet10_quadrature(pts, np.asarray(mesh.cells), sig)
    near = np.zeros(len(Pq), bool)
    if probe_point is not None and probe_radius > 0:
        near = np.linalg.norm(Pq - np.asarray(probe_point, float), axis=1) <= probe_radius
    ring = (edge_rings(mesh) if exact_edges else np.zeros(len(Pq), bool)) & ~near
    exact = near | ring
    if exact.any():
        sq[exact], _ = run.stress(Pq[exact])
    timings["gauss_s"] = time.perf_counter() - t
    s1q = principal(sq)[:, -1]
    return {
        "sig6_nodal": _sig6(sig), "s1_nodal": s1, "sed_nodal": w, "stress_nodal": vm,
        "quad_points": Pq[near], "quad_sed": sed(sq[near], E, nu), "quad_s1": s1q[near],
        "weibull_field": (np.maximum(s1q, 0.0), wq), "weibull_points": Pq,
        "weibull_active_volume_mm3": float(wq[s1q > 0].sum()), "weibull_full_volume_mm3": float(wq.sum()),
        "max_von_mises": float(vm.max()), "hotspot": [round(float(c), 2) for c in pts[int(np.argmax(vm))]],
        "n_nodes": int(len(pts)), "n_elems": int(len(mesh.cells)),
        "solve_info": {"solver": "igabem", "seconds": run.seconds, "report": run.report,
                       "quadrature_points": int(len(Pq)), "exact_quadrature_points": int(exact.sum()),
                       "edge_exact_points": int(ring.sum()), "nodes_outside_spline": int(len(outside)),
                       "interior_displacement": "not computed (display only in bemkit's backend)"},
        "timing": timings,
    }
