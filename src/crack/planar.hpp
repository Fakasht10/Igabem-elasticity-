// Singular coefficients of a flat open crack face, seen from a point of the coincident opposite face.
//
// A dual-BEM collocation point x on one crack face lies on the other face too. Every regularization in this
// repository (sbie.hpp, hsbie.hpp) subtracts a rigid or linear field on a CLOSED surface; on the coincident face
// the subtracted field does not vanish at x (it differs by the opening), and what is left is a singular integral
// over the open face of the kernel times a constant (DBIE) or a linear field (TBIE). For a flat face both are
// homogeneous in r around x and reduce to one integral over the polar angle theta:
//
//   K(x, x + r e(theta)) = r^-k K1(theta),   rho(theta) = distance from x to the face boundary along e(theta),
//
//   PV  int_F T dA        = int T1(theta) ln rho dtheta                         (T1 odd: the ln eps terms cancel)
//   FP  int_F S a dA      = - int S1(theta) a / rho dtheta                      (Hadamard: the 1/eps term dropped)
//   FP  int_F S B(y-x) dA = sum_k int S1(theta) B e_k(theta) ln rho dtheta      (S1 even, e odd: ln eps cancels)
//
// The one-sided limits that the BIEs need are these values plus the double-layer jump +-I/2 for T and nothing for
// S on a flat face (the traction of a displacement-discontinuity sheet is continuous across it); tests/test_crack.cpp
// checks both against off-plane quadrature as the point approaches the face. A face that is not star-shaped from x
// is handled by summing over every crossing of the ray with the boundary (exits +, entries -).
//
// Flat faces only: a curved face needs Guiggiani's expansion instead (the paper's SST), behind the same interface.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../kelvin.hpp"
#include "../nurbs.hpp"

namespace igabem {
namespace crack {

// Boundary of a flat face: closed set of edge curves C(t), t in [t0, t1] (patch edges not shared by two patches
// of the face), each with a polyline for the first guess of a crossing.
struct PlanarRegion {
  Vec3 point, m, e1, e2;  // plane point, unit normal, in-plane orthonormal axes (e1 x e2 = m)
  struct Edge {
    std::function<Vec3(double)> C;
    double t0, t1;
    std::vector<double> ts;
    std::vector<Eigen::Vector2d> xy;  // polyline in plane coordinates
  };
  std::vector<Edge> edges;

  Eigen::Vector2d planar(const Vec3& x) const { return {(x - point).dot(e1), (x - point).dot(e2)}; }

  void setPlane(const Vec3& p, const Vec3& normal) {
    point = p;
    m = normal.normalized();
    e1 = (std::abs(m(0)) < 0.9 ? Vec3::UnitX() : Vec3::UnitY());
    e1 = (e1 - m * m.dot(e1)).normalized();
    e2 = m.cross(e1);
  }

  void addEdge(std::function<Vec3(double)> C, double t0, double t1, int nseg = 256) {
    Edge E{std::move(C), t0, t1, {}, {}};
    for (int i = 0; i <= nseg; ++i) {
      const double t = t0 + (t1 - t0) * i / nseg;
      E.ts.push_back(t);
      E.xy.push_back(planar(E.C(t)));
    }
    // edges meet at shared end points: snap them to bit-identical coordinates so the half-open crossing rule
    // below counts a ray through a junction exactly once
    for (Eigen::Vector2d* q : {&E.xy.front(), &E.xy.back()})
      for (const Edge& F : edges)
        for (const Eigen::Vector2d& r : {F.xy.front(), F.xy.back()})
          if ((*q - r).norm() < 1e-10 * (1.0 + r.norm())) *q = r;
    edges.push_back(std::move(E));
  }

  // Signed crossing distances of the ray x + s d (s > 0, d in plane) with the boundary: (s, +1 exit / -1 entry).
  void crossings(const Vec3& x, const Vec3& d, std::vector<std::pair<double, int>>& out) const {
    out.clear();
    const Eigen::Vector2d o = planar(x), dd(d.dot(e1), d.dot(e2));
    auto cross2 = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) { return a(0) * b(1) - a(1) * b(0); };
    for (const Edge& E : edges) {
      for (size_t i = 0; i + 1 < E.xy.size(); ++i) {
        const Eigen::Vector2d a = E.xy[i] - o, b = E.xy[i + 1] - o;
        const double fa = cross2(dd, a), fb = cross2(dd, b);
        // half-open rule (a vertex on the ray counts as the side f >= 0): a ray through a polyline vertex or a
        // junction of two edges is counted once, and a ray touching a corner from outside twice or not at all
        if ((fa >= 0) == (fb >= 0)) continue;
        // bracketed: secant guess, then Newton/bisection on f(t) = d x (C(t) - x)
        double ta = E.ts[i], tb = E.ts[i + 1], f0 = fa, f1 = fb;
        double t = (f1 == f0) ? 0.5 * (ta + tb) : ta + (tb - ta) * f0 / (f0 - f1);
        for (int it = 0; it < 60; ++it) {
          const double f = cross2(dd, planar(E.C(t)) - o);
          if ((f >= 0) == (f0 >= 0)) { ta = t; f0 = f; } else { tb = t; f1 = f; }
          const double h = 1e-7 * (E.t1 - E.t0);
          const double df = (cross2(dd, planar(E.C(t + h)) - o) - cross2(dd, planar(E.C(t - h)) - o)) / (2 * h);
          double tn = (df != 0.0) ? t - f / df : 0.5 * (ta + tb);
          if (!(tn > ta && tn < tb)) tn = 0.5 * (ta + tb);
          if (std::abs(tn - t) < 1e-13 * (E.t1 - E.t0)) { t = tn; break; }
          t = tn;
        }
        const Eigen::Vector2d c = planar(E.C(t)) - o;
        const double s = c.dot(dd);
        if (s <= 0.0) continue;
        // exit if the edge's outward side (to the right of travel along the ray... ) is decided by the turn:
        // with the boundary oriented arbitrarily, classify by parity below instead
        out.push_back({s, 0});
      }
    }
    std::sort(out.begin(), out.end());
    std::vector<std::pair<double, int>>& u = out;
    // x inside: crossings alternate exit, entry, exit, ...
    for (size_t i = 0; i < u.size(); ++i) u[i].second = (i % 2 == 0) ? 1 : -1;
    if (u.empty() || u.size() % 2 == 0) {
      std::string msg = "PlanarRegion: point not inside the face: x " + std::to_string(o(0)) + " " +
                        std::to_string(o(1)) + " dir " + std::to_string(dd(0)) + " " + std::to_string(dd(1)) + " s";
      for (const auto& c : out) msg += " " + std::to_string(c.first);
      throw std::runtime_error(msg);
    }
  }

  // Angles (about m, from e1) of the edge end points seen from x: kinks of rho(theta).
  std::vector<double> kinks(const Vec3& x) const {
    std::vector<double> a;
    const Eigen::Vector2d o = planar(x);
    for (const Edge& E : edges)
      for (double t : {E.t0, E.t1}) {
        const Eigen::Vector2d c = planar(E.C(t)) - o;
        double th = std::atan2(c(1), c(0));
        if (th < 0) th += 2 * M_PI;
        a.push_back(th);
      }
    std::sort(a.begin(), a.end());
    return a;
  }
};

// Adaptive Gauss-Kronrod (7-15) of a vector-valued f over [a, b]; f returns an Eigen vector of fixed size.
template <class V, class F>
V gaussKronrod(F&& f, double a, double b, double tol, int depth = 0) {
  static const double xk[8] = {0.991455371120812639, 0.949107912342758525, 0.864864423359769073,
                               0.741531185599394440, 0.586087235467691130, 0.405845151377397167,
                               0.207784955007898468, 0.0};
  static const double wk[8] = {0.022935322010529225, 0.063092092629978553, 0.104790010322250184,
                               0.140653259715525919, 0.169004726639267903, 0.190350578064785410,
                               0.204432940075298892, 0.209482141084727828};
  static const double wg[4] = {0.129484966168869693, 0.279705391489276668, 0.381830050505118945,
                               0.417959183673469388};
  const double c = 0.5 * (a + b), h = 0.5 * (b - a);
  V K = f(c) * wk[7], G = f(c) * wg[3];
  for (int i = 0; i < 7; ++i) {
    const V s = f(c - h * xk[i]) + f(c + h * xk[i]);
    K += wk[i] * s;
    if (i % 2 == 1) G += wg[i / 2] * s;
  }
  K *= h;
  G *= h;
  if ((K - G).cwiseAbs().maxCoeff() <= tol || depth >= 14) return K;
  return gaussKronrod<V>(f, a, c, 0.5 * tol, depth + 1) + gaussKronrod<V>(f, c, b, 0.5 * tol, depth + 1);
}

// Coefficients at x (on the plane, inside the face) of the face with outward normal n (unit):
//   CT = PV int_F T(x, y; n) dA                          (3 x 3)
//   Fa = FP int_F S(x, y; N, n) dA                       (3 x 3, acting on a constant)
//   FB[k] = FP int_F S(x, y; N, n) (y - x)_k dA          (3 x 3 each, world k, acting on B.col(k))
struct FaceCoefficients {
  Eigen::Matrix3d CT, Fa, FB[3];
};

inline FaceCoefficients faceCoefficients(const PlanarRegion& F, const Material& mat, const Vec3& x, const Vec3& n,
                                         const Vec3& N, double tol = 1e-12) {
  const Kelvin K(mat);
  const KelvinTraction KT(mat);
  using V45 = Eigen::Matrix<double, 45, 1>;
  std::vector<std::pair<double, int>> cr;
  auto f = [&](double th) {
    const Vec3 e = std::cos(th) * F.e1 + std::sin(th) * F.e2;
    F.crossings(x, e, cr);
    double inv = 0.0, lg = 0.0;  // sum sign/s, sum sign ln s
    for (const auto& [s, sg] : cr) { inv += sg / s; lg += sg * std::log(s); }
    Eigen::Matrix3d U, T, D, S;
    K.eval(e, n, U, T);
    KT.eval(e, N, n, D, S);
    V45 v;
    Eigen::Map<Eigen::Matrix3d>(v.data()) = T * lg;
    Eigen::Map<Eigen::Matrix3d>(v.data() + 9) = -S * inv;
    for (int k = 0; k < 3; ++k) Eigen::Map<Eigen::Matrix3d>(v.data() + 18 + 9 * k) = S * (e(k) * lg);
    return v;
  };
  std::vector<double> br = F.kinks(x);
  br.insert(br.begin(), 0.0);
  br.push_back(2 * M_PI);
  std::sort(br.begin(), br.end());
  V45 acc = V45::Zero();
  // scale of the integrand for an absolute tolerance: 1 / (distance to the boundary)
  double dmin = 1e300;
  for (const auto& E : F.edges)
    for (const auto& p : E.xy) dmin = std::min(dmin, (p - F.planar(x)).norm());
  const double scale = 1.0 / std::max(dmin, 1e-12);
  for (size_t i = 0; i + 1 < br.size(); ++i) {
    if (br[i + 1] - br[i] < 1e-14) continue;
    // split each interval into a few pieces first, so the adaptive rule starts resolved
    const int np = 8;
    for (int j = 0; j < np; ++j) {
      const double a = br[i] + (br[i + 1] - br[i]) * j / np, b = br[i] + (br[i + 1] - br[i]) * (j + 1) / np;
      acc += gaussKronrod<V45>(f, a, b, tol * scale * (b - a));
    }
  }
  FaceCoefficients c;
  c.CT = Eigen::Map<const Eigen::Matrix3d>(acc.data());
  c.Fa = Eigen::Map<const Eigen::Matrix3d>(acc.data() + 9);
  for (int k = 0; k < 3; ++k) c.FB[k] = Eigen::Map<const Eigen::Matrix3d>(acc.data() + 18 + 9 * k);
  return c;
}

}  // namespace crack
}  // namespace igabem
