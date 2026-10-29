"""Derive the exact solution for an elastic spherical inclusion (radius a,
shear modulus mu2, Poisson ratio nu2) perfectly bonded in an infinite matrix
(mu1, nu1) under remote uniaxial tension S along z (Goodier 1933), and write
it as C++ to src/inclusion_exact_gen.hpp.

Matrix: the remote field plus the three decaying Papkovich-Neuber solutions of
gen_cavity_exact.py (centre of dilatation, harmonic potential, Papkovich vector),

  u_out = u_inf + A x / r^3 + B grad d^2(1/r)/dz^2
          + C [4(1 - nu1) e_z d(1/r)/dz - grad(z d(1/r)/dz)],

inclusion: a uniform strain (Eshelby), u_in = diag(D, D, F) x, plus the regular
l = 2 cubic solution of Navier's equation with constant Q (it comes out zero,
which checks Eshelby's uniformity rather than assuming it). Continuity of u and
of sigma n on r = a at rational points gives linear equations for A, B, C, D,
F, Q. The script checks Navier's equation in both phases and both interface
conditions symbolically, prints the inclusion stress, and checks the cavity
limit mu2 -> 0 against Goodier's SCF (27 - 15 nu) / (2 (7 - 5 nu)).

usage: python3 python/gen_inclusion_exact.py  (needs sympy)
"""

import os

import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
a, S, mu1, mu2, nu1, nu2 = sp.symbols("a S mu1 mu2 nu1 nu2", positive=True)
A, B, C, D, F, Q = sp.symbols("A B C D F Q")
X = sp.Matrix([x, y, z])
r = sp.sqrt(x**2 + y**2 + z**2)


def grad(f):
    return sp.Matrix([sp.diff(f, v) for v in (x, y, z)])


def stress(u, mu, nu):
    lam = 2 * mu * nu / (1 - 2 * nu)
    G = u.jacobian(X)
    eps = (G + G.T) / 2
    return lam * eps.trace() * sp.eye(3) + 2 * mu * eps


def navier(sig):
    return [sp.simplify(sum(sp.diff(sig[i, j], v) for j, v in enumerate((x, y, z)))) for i in range(3)]


E1 = 2 * mu1 * (1 + nu1)
phi = 1 / r
uinf = S / E1 * sp.Matrix([-nu1 * x, -nu1 * y, z])
u1 = X / r**3
u2 = grad(sp.diff(phi, z, 2))
u3 = 4 * (1 - nu1) * sp.Matrix([0, 0, sp.diff(phi, z)]) - grad(z * sp.diff(phi, z))
uo = uinf + A * u1 + B * u2 + C * u3

# Regular l = 2 cubic: Papkovich vector B = e_z h, h = z (2 z^2 - 3 (x^2 + y^2)) / 2 (harmonic),
# u = 4(1 - nu2) B - grad(z h) satisfies Navier's equation for (mu2, nu2).
h = z * (2 * z**2 - 3 * (x**2 + y**2)) / 2
ucub = 4 * (1 - nu2) * sp.Matrix([0, 0, h]) - grad(z * h)
ui = sp.Matrix([D * x, D * y, F * z]) + Q * ucub

so = stress(uo, mu1, nu1)
si = stress(ui, mu2, nu2)
assert all(e == 0 for e in navier(si)), navier(si)

pts = [(sp.Rational(3, 5), sp.Rational(4, 5)), (sp.Rational(5, 13), sp.Rational(12, 13)),
       (sp.Rational(8, 17), sp.Rational(15, 17)), (sp.Rational(4, 5), sp.Rational(3, 5))]
eqs = []
for s_, c_ in pts:
    sub = {x: a * s_, y: 0, z: a * c_}
    n = sp.Matrix([s_, 0, c_])
    for i in (0, 2):
        eqs.append(sp.together((uo[i] - ui[i]).subs(sub)))
        eqs.append(sp.together(((so - si) * n)[i].subs(sub)))
sol = sp.solve(eqs, [A, B, C, D, F, Q], dict=True)
assert len(sol) == 1, sol
sol = {k: sp.factor(v) for k, v in sol[0].items()}
print("constants:")
for k, v in sol.items():
    print("  %s = %s" % (k, v))
assert sol[Q] == 0, "inclusion strain not uniform?"

ueo, uei = uo.subs(sol), ui.subs(sol)
seo, sei = stress(ueo, mu1, nu1), stress(uei, mu2, nu2)
assert all(e == 0 for e in navier(seo)), "matrix field not in equilibrium"
for p in [(sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)),
          (sp.Rational(2, 7), sp.Rational(3, 7), sp.Rational(6, 7))]:
    sub = {x: a * p[0], y: a * p[1], z: a * p[2]}
    n = sp.Matrix(p)
    res = [sp.simplify((ueo - uei)[i].subs(sub)) for i in range(3)]
    res += [sp.simplify(((seo - sei) * n)[i].subs(sub)) for i in range(3)]
    assert all(e == 0 for e in res), res

szz_in = sp.factor(sp.simplify(sei[2, 2] / S))
sxx_in = sp.factor(sp.simplify(sei[0, 0] / S))
print("inclusion stress / S: zz =", szz_in, "\n                      xx =", sxx_in)
scf = sp.factor(sp.simplify(seo[2, 2].subs({x: a, y: 0, z: 0}) / S))
cav = sp.simplify(sp.limit(scf, mu2, 0))
goodier = (27 - 15 * nu1) / (2 * (7 - 5 * nu1))
assert sp.simplify(cav - goodier) == 0, (cav, goodier)
print("matrix sigma_zz(a,0,0)/S =", scf)
print("  cavity limit mu2 -> 0:", cav, "(Goodier)")
num = {mu1: 1, nu1: sp.Rational(1, 4), mu2: sp.Rational(1, 10), nu2: sp.Rational(1, 4)}
print("  at mu2/mu1 = 1/10, nu = 1/4:", sp.nsimplify(scf.subs(num)), "=", float(scf.subs(num)))


def emit(name, u):
    G = u.jacobian(X)
    exprs = list(u) + [G[i, j] for i in range(3) for j in range(3)]
    reps, red = sp.cse(exprs)
    body = ["inline void %s(double x, double y, double z, double a, double S, double mu1, double nu1," % name,
            "                %sdouble mu2, double nu2, Eigen::Vector3d& u, Eigen::Matrix3d& G) {" % (" " * len(name)),
            "  (void)a; (void)mu1; (void)nu1; (void)mu2; (void)nu2;"]
    for s, e in reps:
        body.append("  const double %s = %s;" % (s, sp.ccode(e)))
    for i in range(3):
        body.append("  u(%d) = %s;" % (i, sp.ccode(red[i])))
    for k in range(9):
        body.append("  G(%d, %d) = %s;" % (k // 3, k % 3, sp.ccode(red[3 + k])))
    body.append("}")
    return "\n".join(body)


out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "inclusion_exact_gen.hpp")
with open(out, "w") as f:
    f.write("// Generated by python/gen_inclusion_exact.py. Do not edit.\n")
    f.write("// Elastic sphere (mu2, nu2) of radius a at the origin bonded in an infinite matrix\n")
    f.write("// (mu1, nu1), remote uniaxial stress S along z (Goodier 1933): displacement u and\n")
    f.write("// gradient G_ij = du_i/dx_j outside (inclusionOuter) and inside (inclusionInner).\n")
    f.write("// Matrix sigma_zz(a,0,0)/S = %s.\n" % sp.ccode(scf))
    f.write("#pragma once\n\n#include <Eigen/Dense>\n#include <cmath>\n\nnamespace igabem {\n\n")
    f.write(emit("inclusionOuter", ueo) + "\n\n")
    f.write(emit("inclusionInner", uei) + "\n")
    f.write("\n}  // namespace igabem\n")
print("wrote", os.path.normpath(out))
