// Half-space z < 0 under a uniform pressure p on the disk r <= a of its surface
// (Love 1929): the Boussinesq point-load field (boussinesq_exact_gen.hpp)
// integrated over the disk, adaptively in polar coordinates. Away from the disk
// the integrand is smooth; on the disk it is weakly singular (1/rho) and the
// adaptive split converges all the same.
//
// Closed forms on the axis (Johnson, Contact Mechanics, 3.4):
//   sigma_zz(0, 0, -zeta) = -p [1 - zeta^3 / (a^2 + zeta^2)^(3/2)]
//   sigma_rr(0, 0, -zeta) = -p/2 [(1 + 2 nu) - 2 (1 + nu) zeta / sqrt(a^2 + zeta^2)
//                                 + zeta^3 / (a^2 + zeta^2)^(3/2)]
//   u_z(0, 0, 0) = -2 (1 - nu^2) p a / E
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>

#include "../boussinesq_exact_gen.hpp"
#include "../kelvin.hpp"
#include "../quadrature.hpp"

namespace igabem {

struct HalfSpaceDisk {
  Material mat;
  double a = 1.0, p = 1.0, tol = 1e-10;

  // u and du/dx at x (z <= 0). On the loaded surface the gradient integral is
  // strongly singular (principal value): grad = false integrates u only. Points
  // on the surface z = 0 use polar coordinates about themselves (the 1/rho
  // displacement kernel times the polar Jacobian is smooth).
  void field(const Vec3& x, Vec3& u, Eigen::Matrix3d& Gr, bool grad = true) const {
    using V12 = Eigen::Matrix<double, 12, 1>;
    static const GaussRule g10 = gaussLegendre(10);
    const double G = mat.mu, nu = mat.nu();
    const Vec2 x0(x(0), x(1));
    const double c = x0.squaredNorm() - a * a;
    // on the loaded part of the surface (outside it the disk polar rule is fine)
    const bool surface = std::abs(x(2)) < 1e-12 && c <= 0.0;
    // (t, th) in [0, 1] x [th0, th1] -> load point xi and Jacobian (area element per dt dth)
    auto map = [&](double t, double th, Vec2& xi) -> double {
      const Vec2 e(std::cos(th), std::sin(th));
      if (!surface) { xi = a * t * e; return a * a * t; }
      const double b = x0.dot(e), smax = -b + std::sqrt(std::max(0.0, b * b - c));
      xi = x0 + t * smax * e;
      return smax * smax * t;
    };
    auto rule = [&](double r0, double r1, double t0, double t1) {
      V12 I = V12::Zero();
      Vec3 uu;
      Eigen::Matrix3d gg;
      Vec2 xi;
      for (size_t i = 0; i < g10.x.size(); ++i)
        for (size_t j = 0; j < g10.x.size(); ++j) {
          const double J = map(r0 + (r1 - r0) * g10.x[i], t0 + (t1 - t0) * g10.x[j], xi);
          if (J == 0.0) continue;
          const Vec2 d = x0 - xi;
          if (surface && d.norm() < 1e-14 * a) continue;
          boussinesqField(d(0), d(1), x(2), G, nu, uu, gg);
          const double w = g10.w[i] * g10.w[j] * (r1 - r0) * (t1 - t0) * J * p;
          I.head<3>() += w * uu;
          for (int k = 0; k < 9; ++k) I(3 + k) += w * gg(k / 3, k % 3);
        }
      return I;
    };
    // scales: |u| ~ p a / G, |grad u| ~ p / G
    const double su = p * a / G, sg = p / G;
    std::function<V12(double, double, double, double, const V12&, int)> rec =
        [&](double r0, double r1, double t0, double t1, const V12& coarse, int depth) -> V12 {
      const double rm = 0.5 * (r0 + r1), tm = 0.5 * (t0 + t1);
      V12 parts[4] = {rule(r0, rm, t0, tm), rule(rm, r1, t0, tm), rule(r0, rm, tm, t1), rule(rm, r1, tm, t1)};
      const V12 fine = parts[0] + parts[1] + parts[2] + parts[3];
      const V12 d = fine - coarse;
      const double area = (r1 - r0) * (t1 - t0) / (2 * M_PI);
      if ((d.head<3>().norm() <= tol * su * area && (!grad || d.tail<9>().norm() <= tol * sg * area)) || depth > 18)
        return fine;
      return rec(r0, rm, t0, tm, parts[0], depth + 1) + rec(rm, r1, t0, tm, parts[1], depth + 1) +
             rec(r0, rm, tm, t1, parts[2], depth + 1) + rec(rm, r1, tm, t1, parts[3], depth + 1);
    };
    V12 I = V12::Zero();
    for (int q = 0; q < 4; ++q) {
      const double t0 = q * M_PI / 2, t1 = t0 + M_PI / 2;
      I += rec(0.0, 1.0, t0, t1, rule(0.0, 1.0, t0, t1), 0);
    }
    u = I.head<3>();
    for (int k = 0; k < 9; ++k) Gr(k / 3, k % 3) = I(3 + k);
  }

  Vec3 displacement(const Vec3& x) const {
    Vec3 u;
    Eigen::Matrix3d Gr;
    field(x, u, Gr, false);
    return u;
  }
  Eigen::Matrix3d stress(const Vec3& x) const {
    Vec3 u;
    Eigen::Matrix3d Gr;
    field(x, u, Gr);
    return mat.stress(Gr);
  }

  double szzAxis(double zeta) const {
    const double s = a * a + zeta * zeta;
    return -p * (1.0 - zeta * zeta * zeta / std::pow(s, 1.5));
  }
  double srrAxis(double zeta) const {
    const double s = a * a + zeta * zeta, nu = mat.nu();
    return -0.5 * p * ((1 + 2 * nu) - 2 * (1 + nu) * zeta / std::sqrt(s) + zeta * zeta * zeta / std::pow(s, 1.5));
  }
};

}  // namespace igabem
