import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Volume quadrature of sigma_1/sigma_net for the four Yaren-Susmel specimens (weakest-link integrals).
Reuses the existing 2D plane-stress FE of ys/fe2d.py (quarter model, P2), adds the plain hourglass.
New post-processing quantity (not stored before); 2D FE, seconds. Output: ys_wl_fields.npz with w [mm^3, full specimen, t = 5 mm], s1."""
import sys, time, numpy as np
sys.path.insert(0, HERE + "ys/")
import fe2d
from skfem import Basis, ElementTriP2, ElementVector, asm, solve, condense, FacetBasis, LinearForm
from skfem.models.elasticity import linear_elasticity
W2, XR, T = fe2d.W2, fe2d.XR, 5.0
_b0 = fe2d.boundary
def boundary(kind, rho, nroot):
    if kind != "plain": return _b0(kind, rho, nroot)
    R, cx = 18.5, XR + 18.5; h = np.sqrt(R ** 2 - (cx - W2) ** 2)
    y = np.linspace(0, h, 60)[1:]
    return np.array([(0, 0), (XR, 0)] + [(cx - np.sqrt(R * R - t * t), t) for t in y] + [(W2, fe2d.H), (0, fe2d.H)])
fe2d.boundary = boundary
def run(kind, rho, h_root, coords=False):
    m = fe2d.mesh(kind, rho, h_root); E, nu = 1.0, 0.35
    lam, mu = E * nu / (1 - nu ** 2), E / (2 * (1 + nu))
    e = ElementVector(ElementTriP2()); ib = Basis(m, e)
    K = asm(linear_elasticity(lam, mu), ib)
    fb = FacetBasis(m, e, facets=m.facets_satisfying(lambda x: np.isclose(x[1], fe2d.H)))
    f = asm(LinearForm(lambda v, w: XR / W2 * v[1]), fb)
    D = np.concatenate([ib.get_dofs(lambda x: np.isclose(x[0], 0)).all("u^1"), ib.get_dofs(lambda x: np.isclose(x[1], 0) & (x[0] <= XR + 1e-9)).all("u^2")])
    u = solve(*condense(K, f, D=D)); gu = ib.interpolate(u).grad
    exx, eyy, exy = gu[0, 0], gu[1, 1], 0.5 * (gu[0, 1] + gu[1, 0])
    sxx = lam * (exx + eyy) + 2 * mu * exx; syy = lam * (exx + eyy) + 2 * mu * eyy; sxy = 2 * mu * exy
    s1 = 0.5 * (sxx + syy) + np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
    if coords:
        X = ib.global_coordinates().value
        return (4 * T * ib.dx).ravel(), np.maximum(s1, 0).ravel(), np.c_[X[0].ravel(), X[1].ravel()]
    return (4 * T * ib.dx).ravel(), np.maximum(s1, 0).ravel(), int(ib.N)
if __name__ == "__main__":
  out = {}
  for name, kind, rho, h in (("plain", "plain", 18.5, 0.15), ("V_r0.15", "V", 0.15, 0.15 / 40), ("U_r1", "U", 1.0, 1 / 40), ("U_r3", "U", 3.0, 3 / 40)):
      t = time.time(); w, s, n = run(kind, rho, h); out[name + "_w"], out[name + "_s"] = w, s
      print(name, "ndof", n, "V", round(w.sum(), 1), "max s1", round(s.max(), 3), "eff vol m=5,10,20:", [round(float((w * s ** mm).sum()), 2) for mm in (5, 10, 20)], round(time.time() - t, 1), "s", flush=True)
  np.savez_compressed(HERE + "ys_wl_fields.npz", **out)
