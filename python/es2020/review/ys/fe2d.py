"""2D plane-stress FE (P2 triangles, scikit-fem + Triangle) of the Yaren-Susmel double-edge notched strips, quarter model.
Like-for-like with the paper (2D ANSYS PLANE182, linear elastic, homogeneous, isotropic; V root 0.15 mm).
Geometry (Fig. 2): gross width 25, net 12, notch depth 6.5 per side, half length 42.5 (grip holes omitted, Saint-Venant).
Output: K(r) = sigma_yy(bisector, r) / sigma_net, r from the root to the centre line (r = 0..6 mm)."""
import numpy as np, triangle, json, sys, time
from skfem import MeshTri, Basis, ElementTriP2, ElementVector, asm, solve, condense, FacetBasis, LinearForm
from skfem.models.elasticity import linear_elasticity
W2, XR, H = 12.5, 6.0, 42.5      # half gross width, root x, half length

def boundary(kind, rho, nroot):
    """polyline of the quarter domain, counter-clockwise, root at (XR, 0)."""
    pts = [(0, 0), (XR, 0)]
    if kind == 'U':
        th = np.linspace(np.pi, np.pi / 2, nroot)[1:]
        pts += [(XR + rho + rho * np.cos(t), rho * np.sin(t)) for t in th]
        pts += [(W2, rho)]
    else:  # V: 35 deg opening, root circle radius rho tangent to flanks
        a = np.deg2rad(17.5)
        th = np.linspace(np.pi, np.pi / 2 + a, nroot)[1:]
        pts += [(XR + rho + rho * np.cos(t), rho * np.sin(t)) for t in th]
        x0, y0 = pts[-1]
        pts += [(W2, y0 + (W2 - x0) * np.tan(a))]
    pts += [(W2, H), (0, H)]
    return np.array(pts)

def mesh(kind, rho, h_root, h_far=0.4, grow=0.12):
    P = boundary(kind, rho, max(int(np.pi / 2 * rho / h_root), 8))
    seg = np.array([(i, (i + 1) % len(P)) for i in range(len(P))])
    T = triangle.triangulate(dict(vertices=P, segments=seg), 'pq30a%f' % (h_far ** 2 / 2))
    for _ in range(12):
        c = T['vertices'][T['triangles']].mean(1)
        d = np.hypot(c[:, 0] - XR, c[:, 1])
        h = np.minimum(h_far, h_root + grow * d)
        V = T['vertices'][T['triangles']]; a1 = V[:, 1] - V[:, 0]; a2 = V[:, 2] - V[:, 0]
        area = 0.5 * np.abs(a1[:, 0] * a2[:, 1] - a1[:, 1] * a2[:, 0])
        if np.all(area <= h ** 2 / 2 * 1.5): break
        T = triangle.triangulate(dict(vertices=T['vertices'], segments=T['segments'], triangles=T['triangles'],
                                      triangle_max_area=h ** 2 / 2), 'rpq30a')
    p, t = T['vertices'].T.copy(), T['triangles'].T.copy()
    a = (p[0, t[1]] - p[0, t[0]]) * (p[1, t[2]] - p[1, t[0]]) - (p[1, t[1]] - p[1, t[0]]) * (p[0, t[2]] - p[0, t[0]])
    t[[1, 2]] = np.where(a < 0, t[[2, 1]], t[[1, 2]])        # counter-clockwise elements
    return MeshTri(p, t)

def solve_K(kind, rho, h_root):
    m = mesh(kind, rho, h_root)
    E, nu = 1.0, 0.35
    lam, mu = E * nu / (1 - nu ** 2), E / (2 * (1 + nu))      # plane stress
    e = ElementVector(ElementTriP2()); ib = Basis(m, e)
    K = asm(linear_elasticity(lam, mu), ib)
    top = m.facets_satisfying(lambda x: np.isclose(x[1], H))
    fb = FacetBasis(m, e, facets=top)
    sg = XR / W2                                                 # gross stress giving sigma_net = 1
    f = asm(LinearForm(lambda v, w: sg * v[1]), fb)
    D = np.concatenate([ib.get_dofs(lambda x: np.isclose(x[0], 0)).all('u^1'),
                        ib.get_dofs(lambda x: np.isclose(x[1], 0) & (x[0] <= XR + 1e-9)).all('u^2')])
    u = solve(*condense(K, f, D=D))
    # sigma_yy, L2-projected to P2, probed on the bisector
    sb = Basis(m, ElementTriP2())
    gu = ib.interpolate(u).grad
    syy = lam * (gu[0, 0] + gu[1, 1]) + 2 * mu * gu[1, 1]
    s = sb.project(syy)
    X = sb.doflocs; on = np.isclose(X[1], 0) & (X[0] <= XR + 1e-9)       # P2 nodes on the bisector (ligament)
    o = np.argsort(-X[0, on]); r = XR - X[0, on][o]; Kr = s[on][o]
    return dict(r=r.tolist(), K=Kr.tolist(), ndof=int(ib.N), nel=int(m.t.shape[1]), h_root=h_root)

if __name__ == '__main__':
    out = {}
    for name, kind, rho in (('V_r0.15', 'V', 0.15), ('U_r1', 'U', 1.0), ('U_r3', 'U', 3.0)):
        out[name] = []
        for h in (rho / 20, rho / 40):
            t = time.time(); R = solve_K(kind, rho, h); R['sec'] = round(time.time() - t, 1); out[name].append(R)
            print(name, 'h_root', round(h, 4), 'ndof', R['ndof'], 'Kt', round(R['K'][0], 4), 'K(0.1)', round(np.interp(0.1, R['r'], R['K']), 4),
                  'K(1)', round(np.interp(1, R['r'], R['K']), 4), 'mean', round(np.trapezoid(R['K'], R['r']) / XR, 4), R['sec'], 's', flush=True)
    json.dump(out, open(sys.argv[1], 'w'))
