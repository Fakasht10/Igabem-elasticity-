"""Fatigue specimens as quarter models (symmetry x = 0 and y = 0, load axis y, thickness z in [0, T], faces free).

Shapes (lengths in mm):
    edge   double-edge notched plate: gross width WG, net width WN, notch root the circular arc of radius rho with
           flanks at +-alpha/2 (alpha = 0: U slot of half-width rho), total length L; root at (WN/2, 0), bisector -x
           (failure.weibull_parity.quarter_step generalised: same outline, any widths and thickness)
    hole   plate of width W with a central circular hole of radius rho, length L; root at (rho, 0), bisector +x
    plain  uniform bar of width W (gauge section): unit net stress everywhere, no solve

Every field is computed for a unit nominal NET stress (traction WN/WG on the grip of an edge plate, (W - 2 rho)/W
for a hole), so a criterion returns the net-section strength directly.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Specimen:
    shape: str
    T: float
    rho: float = 0.0
    alpha_deg: float = 0.0
    WG: float = 0.0
    WN: float = 0.0
    W: float = 0.0
    L: float = 0.0

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: float(v) if k != "shape" else v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def tag(self):
        if self.shape == "edge":
            return f"edge_WG{self.WG:g}_WN{self.WN:g}_T{self.T:g}_r{self.rho:g}_a{self.alpha_deg:g}_L{self.L:g}"
        if self.shape == "hole":
            return f"hole_W{self.W:g}_T{self.T:g}_r{self.rho:g}_L{self.L:g}"
        return f"plain_W{self.W:g}_T{self.T:g}"

    @property
    def net_width(self):
        return {"edge": self.WN, "hole": self.W - 2 * self.rho, "plain": self.W}[self.shape]

    @property
    def gross_width(self):
        return self.WG if self.shape == "edge" else self.W

    @property
    def root(self):
        """Root point at mid-thickness and the inward bisector direction (x component)."""
        return {"edge": ((self.WN / 2, 0.0, self.T / 2), -1.0),
                "hole": ((self.rho, 0.0, self.T / 2), +1.0)}[self.shape]

    @property
    def ligament(self):
        """Bisector length from the root to the symmetry plane (edge) or the free edge (hole)."""
        return self.WN / 2 if self.shape == "edge" else self.W / 2 - self.rho


def outline(s):
    """Closed quarter outline as [('line', p, q) | ('arc', p, centre, q)], counter-clockwise."""
    if s.shape == "edge":
        al = math.radians(s.alpha_deg)
        xe, ye = s.WN / 2 + s.rho * (1 - math.sin(al / 2)), s.rho * math.cos(al / 2)
        a, b, c = (0.0, 0.0), (s.WN / 2, 0.0), (xe, ye)
        d, e, f = (s.WG / 2, ye + (s.WG / 2 - xe) * math.tan(al / 2)), (s.WG / 2, s.L / 2), (0.0, s.L / 2)
        if d[1] >= e[1]:
            raise ValueError("notch flank leaves the specimen: increase L")
        return [("line", a, b), ("arc", b, (s.WN / 2 + s.rho, 0.0), c), ("line", c, d), ("line", d, e),
                ("line", e, f), ("line", f, a)]
    if s.shape == "hole":
        a, b, c, d, e = (s.rho, 0.0), (s.W / 2, 0.0), (s.W / 2, s.L / 2), (0.0, s.L / 2), (0.0, s.rho)
        return [("line", a, b), ("line", b, c), ("line", c, d), ("line", d, e), ("arc", e, (0.0, 0.0), a)]
    raise ValueError(f"no outline for shape {s.shape!r}")


def quarter_step(s, path):
    """The quarter as a STEP (OpenCASCADE through Gmsh), extruded through the thickness T."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(s.tag)
    o = gmsh.model.occ
    pts = {}

    def P(p):
        key = (round(p[0], 12), round(p[1], 12))
        if key not in pts:
            pts[key] = o.addPoint(p[0], p[1], 0)
        return pts[key]

    curves, centres = [], []
    for seg in outline(s):
        if seg[0] == "line":
            curves.append(o.addLine(P(seg[1]), P(seg[2])))
        else:
            c = o.addPoint(seg[2][0], seg[2][1], 0)
            centres.append(c)
            curves.append(o.addCircleArc(P(seg[1]), c, P(seg[3])))
    o.remove([(0, c) for c in centres])
    o.extrude([(2, o.addPlaneSurface([o.addCurveLoop(curves)]))], 0, 0, s.T)
    o.synchronize()
    gmsh.write(path)
    gmsh.finalize()


def case(s):
    """bc_case body: symmetry supports and the grip traction giving a unit nominal net stress."""
    plane = lambda p, n: {"plane": {"point": p, "normal": n, "tol": 1e-6}}
    t = s.net_width / s.gross_width
    return {"supports": [{"label": "sym x", "select": plane([0, 0, 0], [1, 0, 0]), "fix": ["ux"], "snap": True},
                         {"label": "sym y", "select": plane([0, 0, 0], [0, 1, 0]), "fix": ["uy"], "snap": True}],
            "cases": [{"name": "pull", "loads": [{"label": "grip", "select": plane([0, s.L / 2, 0], [0, 1, 0]),
                                                  "traction": [0, t, 0]}]}]}
