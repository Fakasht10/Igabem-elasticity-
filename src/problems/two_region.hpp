// Exact two-region fields for the multi-region validation (app/region_validate),
// on the concentric spheres of python/make_regions.py (outer radius b = 1, core
// radius a = 0.5, centre 0). Region 0 is the shell or matrix (mat1), region 1 the
// core (mat2).
//
//   LameSphere2    two-layer sphere under external pressure p: u = A x in the core,
//                  u = B x + C x / r^3 in the shell, A, B, C from continuity of u_r
//                  and sigma_rr at r = a and sigma_rr = -p at r = b.
//   EshelbySphere  elastic sphere in an infinite matrix under remote tension S along z
//                  (Goodier 1933; inclusion_exact_gen.hpp). Prescribing its traction on
//                  r = b makes it the exact solution of the finite two-region body.
#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "../inclusion_exact_gen.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"

namespace igabem {

struct LameSphere2 {
  Material mat1, mat2;
  double a = 0.5, b = 1.0, p = 1.0;
  double A = 0, B = 0, C = 0;

  void init() {
    const double K1 = mat1.lambda + 2.0 * mat1.mu / 3.0, K2 = mat2.lambda + 2.0 * mat2.mu / 3.0;
    Eigen::Matrix3d M;
    Eigen::Vector3d r;
    M << a, -a, -1.0 / (a * a),                                  // u_r continuous
        3 * K2, -3 * K1, 4 * mat1.mu / (a * a * a),              // sigma_rr continuous
        0, 3 * K1, -4 * mat1.mu / (b * b * b);                   // sigma_rr(b) = -p
    r << 0, 0, -p;
    const Eigen::Vector3d s = M.partialPivLu().solve(r);
    A = s(0);
    B = s(1);
    C = s(2);
  }
  void field(int region, const Vec3& x, Vec3& u, Eigen::Matrix3d& G) const {
    if (region == 1) {
      u = A * x;
      G = A * Eigen::Matrix3d::Identity();
      return;
    }
    const double r = x.norm(), r3 = r * r * r;
    u = B * x + C * x / r3;
    G = B * Eigen::Matrix3d::Identity() + C * (Eigen::Matrix3d::Identity() / r3 - 3.0 * x * x.transpose() / (r3 * r * r));
  }
  const Material& mat(int region) const { return region == 1 ? mat2 : mat1; }
};

struct EshelbySphere {
  Material mat1, mat2;
  double a = 0.5, b = 1.0, S = 1.0;

  void init() {}
  void field(int region, const Vec3& x, Vec3& u, Eigen::Matrix3d& G) const {
    if (region == 1)
      inclusionInner(x(0), x(1), x(2), a, S, mat1.mu, mat1.nu(), mat2.mu, mat2.nu(), u, G);
    else
      inclusionOuter(x(0), x(1), x(2), a, S, mat1.mu, mat1.nu(), mat2.mu, mat2.nu(), u, G);
  }
  const Material& mat(int region) const { return region == 1 ? mat2 : mat1; }
};

}  // namespace igabem
