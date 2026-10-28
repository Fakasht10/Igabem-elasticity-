// Boundary stress recovery from a displacement solution and the traction:
// the tangential gradient comes from u_h, the normal derivative from t through
// Hooke's law, so the full Cauchy stress is available on the surface without
// any domain evaluation.
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "kelvin.hpp"
#include "mesh.hpp"

namespace igabem {

// Full displacement gradient at a boundary point from the tangential gradient
// Gt (Gt N = 0) and the traction t on the unit normal N.
inline Eigen::Matrix3d surfaceGradient(const Material& mat, const Eigen::Matrix3d& Gt,
                                       const Vec3& t, const Vec3& N) {
  const double lam = mat.lambda, mu = mat.mu;
  const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - N * N.transpose();
  const Vec3 a = P * t / mu - P * (Gt.transpose() * N) +
                 N * (t.dot(N) - lam * Gt.trace()) / (lam + 2.0 * mu);
  return Gt + a * N.transpose();
}

// Tangential gradient of a field with coefficients c (3 per global dof of the
// conforming space) at an evaluated point.
inline Eigen::Matrix3d tangentialGradient(const Mesh& m, int patch, const PatchEval& E,
                                          const Eigen::VectorXd& c) {
  Vec3 uu = Vec3::Zero(), uv = Vec3::Zero();
  for (int l = 0; l < E.n_loc; ++l) {
    const Vec3 cl = c.segment<3>(3 * m.l2g[patch][E.idx[l]]);
    uu += E.Ru[l] * cl;
    uv += E.Rv[l] * cl;
  }
  Eigen::Matrix<double, 3, 2> Ax, Ud;
  Ax << E.xu, E.xv;
  Ud << uu, uv;
  return Ud * (Ax.transpose() * Ax).inverse() * Ax.transpose();
}

struct SurfacePoint {
  int patch, su, sv;
  Vec2 xi;
  PatchEval E;
  Vec3 N;  // unit normal outward from the solid
};

// Closest boundary point to x among the patches accepted by use(patch).
template <class Pred>
inline SurfacePoint locate(const Mesh& m, const Vec3& x, Pred&& use) {
  SurfacePoint s{};
  double best = 1e300;
  for (const auto& e : m.elements) {
    if (!use(e.patch)) continue;
    double d;
    const Vec2 xi = m.closestPoint(e, x, d);
    if (d < best) { best = d; s.patch = e.patch; s.su = e.su; s.sv = e.sv; s.xi = xi; }
  }
  m.patches[s.patch].evalSpan(s.su, s.sv, s.xi(0), s.xi(1), s.E);
  s.N = (s.E.normal() * m.normal_sign).normalized();
  return s;
}

// Closest boundary point to x among the patches [p0, p1).
inline SurfacePoint locate(const Mesh& m, const Vec3& x, int p0, int p1) {
  return locate(m, x, [=](int p) { return p >= p0 && p < p1; });
}

// Cauchy stress of u_h at a boundary point with known traction t.
inline Eigen::Matrix3d surfaceStress(const Mesh& m, const Material& mat, const SurfacePoint& s,
                                     const Eigen::VectorXd& c, const Vec3& t) {
  return mat.stress(surfaceGradient(mat, tangentialGradient(m, s.patch, s.E, c), t, s.N));
}

}  // namespace igabem
