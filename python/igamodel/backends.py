"""Solver backends behind the model tree.

A backend declares what it can solve as capability words; `features(model)`
says what a model asks for in the same words, so a model a backend cannot
solve is refused before any geometry is made or any operator assembled.

    backend = backends.get("igabem")
    backend.geometry(model, base, work)          # cad -> the case's geometry (cached)
    with backend.open(model, base, work) as run: # assemble and factorise once
        sig, wind = run.view("bend").stress(points, on_surface)

`open` assembles one operator for all the model's load cases. A `Run` keeps it
alive: every point query returns the stress of every load case at once, and the
per-case views share that result, so a second load case costs no evaluation.

Imported on demand: it pulls in numpy and the failure layer, the tree does not.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .casefile import to_case
from .model import ModelError

SUPPORT_FEATURES = {"fix": "fixed", "dofs": "fixed", "spring": "spring", "rbe2": "rbe2", "symmetry": "symmetry"}
LOAD_FEATURES = {"pressure": "pressure", "traction": "traction", "bearing": "bearing", "rbe2": "rbe2_load",
                 "force_N": "force", "moment": "moment"}


def features(model) -> set[str]:
    """What the model asks of a solver."""
    f = set()
    for n in model.of_kind("support"):
        f.update(v for k, v in SUPPORT_FEATURES.items() if k in n.props)
        if "value" in n.props:
            f.add("prescribed_displacement")
    for n in model.of_kind("load"):
        f.update(v for k, v in LOAD_FEATURES.items() if k in n.props)
    if model.of_kind("grade"):
        f.add("graded_mesh")
    if model.of_kind("cad"):
        f.add("cad_geometry")
    if model.of_kind("assessment"):
        f.add("stress_at_points")
    if model.of_kind("submodel"):
        f.add("submodel")
    return f


def check(backend, model) -> None:
    missing = features(model) - backend.capabilities
    if missing:
        raise ModelError(f"the {backend.name} backend cannot solve {sorted(missing)}")


def _abs(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


class Run:
    """An assembled, factorised model. `solve` is any object with `cases`, `seconds`, `report`,
    `stress_all(points, on_surface) -> {case: (stress, winding)}` and `close()`."""

    def __init__(self, solve, keep=4):
        self.solve, self.keep, self._memo = solve, keep, {}
        self.cases, self.seconds, self.report = solve.cases, solve.seconds, solve.report

    def stress_all(self, points, on_surface=None):
        P = np.ascontiguousarray(np.atleast_2d(np.asarray(points, float)))
        on = np.zeros(len(P), bool) if on_surface is None else np.broadcast_to(np.asarray(on_surface, bool), len(P))
        key = hashlib.sha1(P.tobytes() + np.ascontiguousarray(on).tobytes()).digest()
        if key not in self._memo:
            if len(self._memo) >= self.keep:
                self._memo.pop(next(iter(self._memo)))
            self._memo[key] = self.solve.stress_all(P, on)
        return self._memo[key]

    def stress(self, points, on_surface=None, case=None):
        s, w = self.stress_all(points, on_surface)[case or self.cases[0]]
        return s.copy(), w.copy()       # callers write into the result

    def displacement_all(self, points, on_surface=None):
        return self.solve.displacement_all(points, on_surface)

    def view(self, case):
        if case not in self.cases:
            raise ModelError(f"no load case {case!r}; the solve has {self.cases}")
        return CaseView(self, case)

    def close(self):
        self.solve.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class CaseView:
    """One load case of a Run, with the `stress(points, on_surface)` / `seconds` / `report`
    interface the failure layer takes."""

    def __init__(self, run, case):
        self.run, self.case, self.seconds, self.report = run, case, run.seconds, run.report

    def stress(self, points, on_surface=None):
        return self.run.stress(points, on_surface, self.case)


class IgBEM:
    """app/bc_case: collocation IgBEM, one LU kept alive in `bc_case --serve`."""
    name = "igabem"
    capabilities = frozenset({"fixed", "prescribed_displacement", "spring", "rbe2", "symmetry", "pressure",
                              "traction", "bearing", "rbe2_load", "force", "moment", "graded_mesh", "cad_geometry",
                              "stress_at_points", "submodel"})
    CAD_DEFAULTS = {"h": 3.0, "curvature": 6, "degree": 2}

    def geometry(self, model, base, workdir, log=print):
        """The cad node's STEP -> multipatch .dat + .faces (python/step_auto.py, cached by the STEP's bytes and
        the settings); sets the case's geometry. Returns the layout checks."""
        from failure.igabem_backend import igabem_geometry
        cad = _one(model, "cad")
        o = {k: cad.props.get(k, v) for k, v in self.CAD_DEFAULTS.items()}
        dat, info = igabem_geometry(_abs(cad.props["step"], base), workdir, log=log, **o)
        model.apply({"op": "set", "tag": model.root.tag, "key": "geometry", "value": os.path.abspath(dat)})
        return info

    def open(self, model, base, workdir, log=print) -> Run:
        from failure.igabem_backend import Solve
        check(self, model)
        case = to_case(model)
        mat = case.pop("material", None)
        if not mat or "E" not in mat or "nu" not in mat:
            raise ModelError("the model has no material E and nu")
        geometry = _abs(case.pop("geometry"), base)
        case.pop("output", None)                  # results come back through the points queries
        for k in ("faces", "cache"):
            if k in case and case[k] != "none":
                case[k] = _abs(case[k], base)
        return Run(Solve(geometry, case, mat["E"], mat["nu"], workdir=workdir, log=log))


def _one(model, kind):
    n = model.of_kind(kind)
    if not n:
        raise ModelError(f"the model has no {kind}")
    return n[0]


BACKENDS = {b.name: b for b in (IgBEM(),)}


def get(name: str):
    if name not in BACKENDS:
        raise ModelError(f"no backend {name!r}; there are {sorted(BACKENDS)}")
    return BACKENDS[name]
