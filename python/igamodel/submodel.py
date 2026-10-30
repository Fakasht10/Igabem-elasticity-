"""Submodels: a box cut out of the part round a feature, meshed finer, its cut faces
driven by the global solve.

    sub = Submodel.build(step, lo, hi, workdir, h=0.5)       # STEP ∩ box -> multipatch (cached)
    with sub.open(global_run, E, nu) as run:                # one solve for every load case
        sig, wind = run.view("bend").stress(points)

Theory: the cut faces take the global displacement, the part's own faces inside the
box stay traction-free (no support or load may reach into the box). By Saint-Venant
the error the coarse global makes near the feature decays away from it, so the cut
faces need to be a few feature sizes off the feature. The displacement is evaluated
exactly, by the global solve's Somigliana identity (boundary value on the rim), at
the points bc_case interpolates the cut faces at (bc_case --greville): nothing is
interpolated between solves.

A box face lying in a face of the part would make the cut faces ambiguous; it is
refused, as is a box that misses the part or the feature point.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import numpy as np

from .model import ModelError

CUT = "cut"


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _faces_bbox(step):
    """{gmsh surface tag: (lo, hi)} of a STEP."""
    import gmsh
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("bbox")
    try:
        gmsh.model.occ.importShapes(step)
        gmsh.model.occ.synchronize()
        out = {}
        for _, t in gmsh.model.getEntities(2):
            b = np.array(gmsh.model.getBoundingBox(2, t))
            out[t] = (b[:3], b[3:])
        return out
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def cut_step(step, lo, hi, out):
    """STEP ∩ box -> out (a STEP), and the face tags of the result that lie on the box (the cut faces)."""
    import gmsh
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    own = not gmsh.isInitialized()
    if own:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cut")
    try:
        vols = gmsh.model.occ.importShapes(step)
        box = gmsh.model.occ.addBox(*lo, *(hi - lo))
        res, _ = gmsh.model.occ.intersect([v for v in vols if v[0] == 3], [(3, box)])
        gmsh.model.occ.synchronize()
        if not res:
            raise ModelError("the submodel box misses the part")
        gmsh.write(out)
    finally:
        gmsh.model.remove()
        if own:
            gmsh.finalize()


def cut_faces(step_cut, lo, hi, tol):
    """Faces of the cut solid lying in a box plane."""
    out = []
    for t, (a, b) in _faces_bbox(step_cut).items():
        for ax in range(3):
            if b[ax] - a[ax] < tol and min(abs(a[ax] - lo[ax]), abs(a[ax] - hi[ax])) < tol:
                out.append(t)
    return sorted(set(out))


def check_box(step, lo, hi, point, region_faces, tol):
    """Refuse a box that would make the submodel ill-posed."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.any(hi <= lo):
        raise ModelError("the submodel box needs lo < hi")
    if point is not None and np.any((np.asarray(point) <= lo) | (np.asarray(point) >= hi)):
        raise ModelError(f"the feature point {list(point)} is not inside the submodel box")
    faces = _faces_bbox(step)
    for t, (a, b) in faces.items():
        for ax in range(3):
            for c in (lo[ax], hi[ax]):
                inside = np.all(np.delete((b > lo) & (a < hi), ax))
                if b[ax] - a[ax] < tol and abs(a[ax] - c) < tol and inside:
                    raise ModelError(f"the box face at {'xyz'[ax]} = {c:g} lies in face {t} of the part; move it "
                                     "off the face (into the part or past it)")
    for t in region_faces:
        a, b = faces[t]
        if np.all((b > lo + tol) & (a < hi - tol)):
            raise ModelError(f"face {t} carries a support or load and reaches into the submodel box; the box must "
                             "hold traction-free faces only")


class Submodel:
    """A cut-out of the part: its multipatch geometry and the cut faces' CAD ids."""

    def __init__(self, step, dat, cut, info):
        self.step, self.dat, self.cut, self.info = step, dat, cut, info

    @classmethod
    def build(cls, step, lo, hi, workdir, h=0.5, curvature=12, degree=2, log=print):
        from failure.igabem_backend import igabem_geometry
        os.makedirs(workdir, exist_ok=True)
        lo, hi = [float(x) for x in lo], [float(x) for x in hi]
        key = hashlib.sha256(json.dumps([_sha(step), lo, hi]).encode()).hexdigest()[:12]
        sub_step = os.path.join(workdir, f"sub_{key}.step")
        if not os.path.exists(sub_step):         # written once: Gmsh stamps the time into a STEP
            cut_step(step, lo, hi, sub_step)
        tol = 1e-6 * float(np.linalg.norm(np.subtract(hi, lo)))
        cut = cut_faces(sub_step, lo, hi, tol)
        if not cut:
            raise ModelError("the submodel box cuts nothing off the part; it holds the whole part")
        dat, info = igabem_geometry(sub_step, workdir, h=h, curvature=curvature, degree=degree, log=log)
        return cls(sub_step, dat, cut, info)

    def case(self, cases):
        return {"supports": [{"label": CUT, "face_ids": self.cut, "fix": ["ux", "uy", "uz"]}],
                "cases": [{"name": c, "fields": {CUT: f"u_{c}.txt"}, "loads": []} for c in cases]}

    def open(self, global_run, E, nu, workdir, log=print):
        """Drive the cut faces from global_run (a backends.Run) and solve: a backends.Run of the submodel."""
        from failure.igabem_backend import BC_CASE, Solve
        from .backends import Run
        os.makedirs(workdir, exist_ok=True)
        cases = list(global_run.cases)
        body = dict(self.case(cases), geometry=os.path.abspath(self.dat), material={"E": float(E), "nu": float(nu)},
                    cache="none")
        path = os.path.join(workdir, "greville.json")
        with open(path, "w") as f:
            json.dump(body, f, indent=1)
        pts = os.path.join(workdir, "greville.txt")
        r = subprocess.run([BC_CASE, path, "--greville", CUT, pts], capture_output=True, text=True)
        if r.returncode:
            raise ModelError("bc_case --greville: " + (r.stderr.strip() or r.stdout.strip()))
        X = np.loadtxt(pts, ndmin=2)
        t = time.perf_counter()
        U = global_run.displacement_all(X)
        log(f"submodel: global displacement at {len(X)} cut-face points in {time.perf_counter() - t:.1f} s")
        for c in cases:
            u, w = U[c]
            if np.any(w < 0.25):
                raise ModelError(f"{int(np.sum(w < 0.25))} cut-face points lie outside the global geometry")
            np.savetxt(os.path.join(workdir, f"u_{c}.txt"), np.column_stack([X, u]), fmt="%.17g")
        return Run(Solve(self.dat, self.case(cases), E, nu, workdir=workdir, log=log))
