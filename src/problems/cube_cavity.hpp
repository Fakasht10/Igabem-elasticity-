// Unit cube with a spherical cavity of radius a = 0.2 at c = (1/2, 1/2, 1/2)
// (python/make_geometries.py: 6 cube patches first, then 6 cavity patches).
//
// Load cases:
//   Patch     u = alpha x, t = sigma n on both surfaces
//   Pressure  exact Lame field u = p a^3 (x - c) / (4 mu |x - c|^3), p = 1
//   Goodier   exact infinite-medium field for remote tension S = 1 along z
//   Tension   t = S e_z on z = 0, 1, zero elsewhere (no exact solution)
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <string>

#include "../cavity_exact_gen.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"

namespace igabem {

struct CubeCavity {
  enum Case { Patch, Pressure, Goodier, Tension };
  static constexpr double a = 0.2, S = 1.0, p = 1.0;
  static constexpr int ncube = 6;  // cube patches come first
  Material mat;
  Vec3 c = Vec3(0.5, 0.5, 0.5);
  Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();

  static bool parse(const std::string& s, Case& k) {
    if (s == "patch") k = Patch;
    else if (s == "pressure") k = Pressure;
    else if (s == "goodier") k = Goodier;
    else if (s == "tension") k = Tension;
    else return false;
    return true;
  }
  static bool exact(Case k) { return k != Tension; }

  // Exact displacement and gradient at x.
  void field(Case k, const Vec3& x, Vec3& u, Eigen::Matrix3d& G) const {
    const Vec3 X = x - c;
    if (k == Patch) {
      u = alpha * x;
      G = alpha;
    } else if (k == Pressure) {
      const double A = p * a * a * a / (4.0 * mat.mu), r = X.norm();
      u = A * X / (r * r * r);
      G = A * (Eigen::Matrix3d::Identity() / (r * r * r) - 3.0 * X * X.transpose() / std::pow(r, 5));
    } else {
      goodierField(X(0), X(1), X(2), a, S, mat.mu, mat.nu(), u, G);
    }
  }
  Vec3 displacement(Case k, const Vec3& x) const {
    Vec3 u;
    Eigen::Matrix3d G;
    field(k, x, u, G);
    return u;
  }
  Eigen::Matrix3d stress(Case k, const Vec3& x) const {
    Vec3 u;
    Eigen::Matrix3d G;
    field(k, x, u, G);
    return mat.stress(G);
  }
  Vec3 traction(Case k, const Vec3& y, const Vec3& n) const {
    if (k == Tension) return (y - c).norm() > 2.0 * a ? Vec3(S * n(2) * Vec3::UnitZ()) : Vec3::Zero();
    return stress(k, y) * n;
  }
};

}  // namespace igabem
