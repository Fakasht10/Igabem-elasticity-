"""Independent IGA-FEM reference (JAXIGA, Galerkin) for the cube with a spherical cavity.

Same problem as app/cavity_test.cpp: unit cube, cavity a = 0.2 at the centre, lambda = mu = 1
(E = 5/2, nu = 1/4), exact NURBS geometry (octant_geometry.py). The octant model carries
symmetry conditions u_n = 0 on x = 0, y = 0, z = 0, which also remove the rigid motions.

Cases (as in src/problems/cube_cavity.hpp):
  patch    u = alpha x, alpha = diag(-1/4, -1/4, 1); its traction on every free surface
  goodier  exact infinite-medium field for remote tension S = 1 along z, prescribed on the
           cube faces; the cavity is traction-free. SCF = (27 - 15 nu) / (2 (7 - 5 nu))
  tension  t = S e_z on z = 1/2, zero elsewhere (no exact solution)

Reported per run: dofs, sigma_zz at (a, 0, 0) (the BEM's szz_eq), relative L2 displacement
error (exact cases), and wall time for space, assembly, factorization and load+solve. K is
assembled and factored once per mesh and reused for every load case.

usage: python cavity_fem.py [--cases patch,goodier,tension] [--p 4,5] [--n 1,2,4,8]
                            [--grade 1.5] [--csv results/fem_cavity.csv]
                            [--profile-dir results] [--check-jaxiga]
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxiga as jx  # noqa: E402
from jaxiga.space.pointset import PointSet  # noqa: E402

import assembly as asm  # noqa: E402
from octant_geometry import A, L, octant_patches, refined  # noqa: E402

LAM, MU = 1.0, 1.0
NU = LAM / (2 * (LAM + MU))
E = MU * (3 * LAM + 2 * MU) / (LAM + MU)
S = 1.0
ALPHA = jnp.diag(jnp.array([-0.25, -0.25, 1.0]))
SCF_GOODIER = (27 - 15 * NU) / (2 * (7 - 5 * NU))
TARGET = np.array([A, 0.0, 0.0])


def stress_of_grad(G):
    eps = 0.5 * (G + G.T)
    return LAM * jnp.trace(eps) * jnp.eye(3) + 2 * MU * eps


# --- Goodier (1933) field, same potentials and constants as python/gen_cavity_exact.py ---
_cA = -S * A**3 * (5 * NU - 6) / (4 * MU * (5 * NU - 7))
_cB = -S * A**5 / (4 * MU * (5 * NU - 7))
_cC = 5 * S * A**3 / (4 * MU * (5 * NU - 7))


def _phi(x):
    return 1.0 / jnp.linalg.norm(x)


def _dz(f):
    return lambda x: jax.grad(f)(x)[2]


def goodier_u(x):
    u_inf = S / E * jnp.array([-NU * x[0], -NU * x[1], x[2]])
    u1 = x / jnp.linalg.norm(x) ** 3
    u2 = jax.grad(_dz(_dz(_phi)))(x)
    dphi = _dz(_phi)
    u3 = 4 * (1 - NU) * jnp.array([0.0, 0.0, dphi(x)]) - jax.grad(lambda y: y[2] * dphi(y))(x)
    return u_inf + _cA * u1 + _cB * u2 + _cC * u3


def exact_u(case, x):
    return ALPHA @ x if case == "patch" else goodier_u(x)


def exact_stress(case, x):
    G = ALPHA if case == "patch" else jax.jacfwd(goodier_u)(x)
    return stress_of_grad(G)


def build_problem(case, space, quadrature=None):
    """Galerkin problem on the octant. The default rule, p + 3 Gauss points per direction,
    integrates the rational stiffness well enough that the patch test holds to ~1e-9 or
    better (JAXIGA's default p + 1 leaves a 1e-5 quadrature error on this geometry)."""
    sym = [jx.DirichletBC(0.0, where=(lambda x, d=d: np.abs(x[:, d]) < 1e-12), component=d)
           for d in range(3)]
    if case == "tension":
        neu = [jx.Neumann(lambda x, n, p: jnp.array([0.0, 0.0, S]), where="face_z")]
    else:
        where = ["face_x", "face_y", "face_z"] + (["cavity"] if case == "patch" else [])
        neu = [jx.Neumann(lambda x, n, p: exact_stress(case, x) @ n, where=w) for w in where]
    q = quadrature if quadrature is not None else max(space.degree) + 3
    return jx.LinearElasticity(space, dirichlet=sym, neumann=neu, quadrature=q)


def _corner_pointset(space):
    """The element and reference corner whose physical image is (a, 0, 0)."""
    corners = np.array([[i, j, k] for k in (-1, 1) for j in (-1, 1) for i in (-1, 1)], float)
    ps = PointSet(np.arange(space.n_elems), jnp.asarray(corners))
    x = np.asarray(jx.evaluate(space, ps).x)
    d = np.linalg.norm(x - TARGET, axis=-1)
    e, q = np.unravel_index(np.argmin(d), d.shape)
    assert d[e, q] < 1e-12, d[e, q]
    return PointSet(np.array([e]), jnp.asarray(corners[q:q + 1]))


def _meridian_pointset(space, m=17):
    """Points of the cavity meridian y = 0 (w = -1 edges lying in the plane y = 0)."""
    s = np.linspace(-1, 1, m)
    refs = []
    for e1 in (-1.0, 1.0):
        refs.append(np.stack([s, np.full(m, e1), np.full(m, -1.0)], 1))
        refs.append(np.stack([np.full(m, e1), s, np.full(m, -1.0)], 1))
    ref = np.concatenate(refs)
    ps = PointSet(np.arange(space.n_elems), jnp.asarray(ref))
    x = np.asarray(jx.evaluate(space, ps).x)
    keep = (np.abs(x[..., 1]) < 1e-12) & (np.abs(np.linalg.norm(x, axis=-1) - A) < 1e-12)
    elems, qs = np.nonzero(keep)
    return [PointSet(np.array([e]), jnp.asarray(ref[q:q + 1])) for e, q in zip(elems, qs)]


def _stress(space, u, ps):
    G, x = asm.grad_at(space, u, ps)
    return np.asarray(jax.vmap(stress_of_grad)(jnp.asarray(G))), x


_vec_exact_u = {c: jax.jit(jax.vmap(lambda y, c=c: exact_u(c, y))) for c in ("patch", "goodier")}
_vec_exact_s = {c: jax.jit(jax.vmap(lambda y, c=c: exact_stress(c, y))) for c in ("patch", "goodier")}


def traction_fn(case):
    if case == "tension":
        return {"face_z": lambda x, n: np.tile([0.0, 0.0, S], (len(x), 1))}
    labels = ["face_x", "face_y", "face_z"] + (["cavity"] if case == "patch" else [])
    t = lambda x, n: np.einsum("kij,kj->ki", np.asarray(_vec_exact_s[case](jnp.asarray(x))), n)
    return {lab: t for lab in labels}


def l2_error(case, space, u, q):
    uh, x, w = asm.values_at(space, u, jx.gauss(space, q))
    ue = np.asarray(_vec_exact_u[case](jnp.asarray(x)))
    return np.sqrt(np.sum(w * np.sum((uh - ue) ** 2, 1)) / np.sum(w * np.sum(ue**2, 1)))


class Mesh:
    """One discretization: space, stiffness and its factorization, built once for all cases."""

    def __init__(self, p, n, grade, q=None):
        t0 = time.perf_counter()
        self.p, self.n, self.grade = p, n, grade
        self.q = q if q is not None else p + 3
        self.space = jx.FunctionSpace(refined(octant_patches(), p, n, grade), vec=3)
        t1 = time.perf_counter()
        K = asm.stiffness(self.space, self.q, LAM, MU)
        t2 = time.perf_counter()
        self.solver = asm.Solver(K, asm.symmetry_dofs(self.space))
        t3 = time.perf_counter()
        self.times = dict(space_s=t1 - t0, assemble_s=t2 - t1, factor_s=t3 - t2)
        self.corner = _corner_pointset(self.space)
        self.meridian = _meridian_pointset(self.space)

    def solve(self, case):
        t0 = time.perf_counter()
        f = sum(asm.traction_load(self.space, lab, t, self.q)
                for lab, t in traction_fn(case).items())
        u = self.solver.solve(f)
        t1 = time.perf_counter()
        sig, _ = _stress(self.space, u, self.corner)
        err = l2_error(case, self.space, u, self.q) if case != "tension" else float("nan")
        row = dict(case=case, p=self.p, n=self.n, grade=self.grade, ndof=self.space.n_dofs,
                   elements=self.space.n_elems, szz_eq=sig[0, 2, 2], rel_L2_err=err,
                   load_solve_s=t1 - t0, **self.times)
        return row, u

    def profile(self, path, case, u):
        rows = []
        for ps in self.meridian:
            sig, x = _stress(self.space, u, ps)
            sig, x = sig[0], x[0]
            th = np.arctan2(x[0], x[2])
            et, ep = np.array([np.cos(th), 0, -np.sin(th)]), np.array([0, 1.0, 0])
            r = [np.degrees(th), et @ sig @ et, ep @ sig @ ep]
            if case != "tension":
                se = np.asarray(_vec_exact_s[case](jnp.asarray(x[None])))[0]
                r += [et @ se @ et, ep @ se @ ep]
            else:
                r += [np.nan, np.nan]
            rows.append(r)
        rows = np.array(sorted(rows))
        _, idx = np.unique(np.round(rows[:, 0], 10), return_index=True)
        np.savetxt(path, rows[idx], delimiter=",", fmt="%.10g",
                   header="theta_deg,s_mer,s_circ,s_mer_exact,s_circ_exact", comments="")


def check_jaxiga(case, p, n, grade):
    """sigma_zz(a, 0, 0) from jx.solve on the same space, for cross-checking the assembler."""
    space = jx.FunctionSpace(refined(octant_patches(), p, n, grade), vec=3)
    sol = jx.solve(build_problem(case, space), params={"E": E, "nu": NU}, chunk=64)
    G = np.asarray(sol.grad(_corner_pointset(space))).reshape(3, 3)
    return float(stress_of_grad(jnp.asarray(G))[2, 2])


FIELDS = ["case", "p", "n", "grade", "ndof", "elements", "szz_eq", "rel_L2_err",
          "space_s", "assemble_s", "factor_s", "load_solve_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="patch,goodier,tension")
    ap.add_argument("--p", default="4,5")
    ap.add_argument("--n", default="1,2,4,8")
    ap.add_argument("--grade", type=float, default=1.5)
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "..", "results", "fem_cavity.csv"))
    ap.add_argument("--profile-dir", default=os.path.join(HERE, "..", "..", "results"))
    ap.add_argument("--check-jaxiga", action="store_true",
                    help="also solve with jx.solve and compare sigma_zz (slow)")
    args = ap.parse_args()
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        print("# exact Goodier SCF %.8f (nu = %.2f)" % (SCF_GOODIER, NU))
        print("%-8s %2s %3s %7s %13s %11s %7s %7s %7s %7s" % (
            "case", "p", "n", "ndof", "szz(a,0,0)", "rel_L2", "space", "assem", "factor", "solve"))
        for p in map(int, args.p.split(",")):
            for n in map(int, args.n.split(",")):
                mesh = Mesh(p, n, args.grade)
                for case in args.cases.split(","):
                    row, u = mesh.solve(case)
                    wr.writerow(row)
                    f.flush()
                    print("%-8s %2d %3d %7d %13.8f %11.3e %7.2f %7.2f %7.2f %7.2f" % (
                        case, p, n, row["ndof"], row["szz_eq"], row["rel_L2_err"],
                        row["space_s"], row["assemble_s"], row["factor_s"], row["load_solve_s"]),
                        flush=True)
                    mesh.profile(os.path.join(args.profile_dir, "fem_profile_%s_p%d_n%d.csv"
                                              % (case, p, n)), case, u)
                    if args.check_jaxiga:
                        ref = check_jaxiga(case, p, n, args.grade)
                        print("   jx.solve szz = %.12f, diff %.2e" % (ref, ref - row["szz_eq"]))


if __name__ == "__main__":
    main()
