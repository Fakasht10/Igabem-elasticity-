// Kelvin fundamental solution of 3D isotropic elastostatics.
// Convention (Brebbia & Dominguez): U_ij(x,y) is the displacement at y in
// direction j due to a unit point load at x in direction i; T_ij is the
// corresponding traction on a surface with unit normal n at y; r = y - x.
#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace igabem {

struct Material {
  double lambda = 1.0, mu = 1.0;
  double nu() const { return lambda / (2.0 * (lambda + mu)); }
  // Cauchy stress of the linear field u = alpha x.
  Eigen::Matrix3d stress(const Eigen::Matrix3d& alpha) const {
    const Eigen::Matrix3d eps = 0.5 * (alpha + alpha.transpose());
    return lambda * eps.trace() * Eigen::Matrix3d::Identity() + 2.0 * mu * eps;
  }
};

struct Kelvin {
  double mu, nu, cU, cT;
  explicit Kelvin(const Material& m) : mu(m.mu), nu(m.nu()) {
    cU = 1.0 / (16.0 * M_PI * mu * (1.0 - nu));
    cT = -1.0 / (8.0 * M_PI * (1.0 - nu));
  }

  // Both kernels at once; d = y - x, n = unit normal at y.
  inline void eval(const Eigen::Vector3d& d, const Eigen::Vector3d& n,
                   Eigen::Matrix3d& U, Eigen::Matrix3d& T) const {
    const double r = d.norm();
    const Eigen::Vector3d g = d / r;
    const double drdn = g.dot(n);
    const double a = 3.0 - 4.0 * nu, b = 1.0 - 2.0 * nu;
    const double fu = cU / r, ft = cT / (r * r);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        const double dij = (i == j) ? 1.0 : 0.0;
        U(i, j) = fu * (a * dij + g(i) * g(j));
        T(i, j) = ft * (drdn * (b * dij + 3.0 * g(i) * g(j)) -
                        b * (g(i) * n(j) - g(j) * n(i)));
      }
    }
  }
};

}  // namespace igabem
