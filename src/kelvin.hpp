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

// Kernels of the traction BIE (HSBIE), obtained by applying the traction
// operator at x, t_i = C_ijkl N_j d/dx_l, to the representation
// u_k(x) = int U_km t_m - int T_km u_m:
//
//   t_i(x) = int D_im(x,y) t_m(y) dGamma_y - int S_im(x,y) u_m(y) dGamma_y,
//
//   D_im = C_ijkl N_j dU_km/dx_l   (adjoint double layer, O(1/r^2))
//   S_im = C_ijkl N_j dT_km/dx_l   (hypersingular, O(1/r^3))
//
// N is the unit normal at the collocation point x, n the one at y.
struct KelvinTraction {
  double lambda, mu, nu, cU;
  explicit KelvinTraction(const Material& m) : lambda(m.lambda), mu(m.mu), nu(m.nu()) {
    cU = 1.0 / (16.0 * M_PI * mu * (1.0 - nu));
  }

  // Traction of a gradient field P (P_kl = d/dx_l of component k) on normal N.
  inline Eigen::Vector3d traction(const Eigen::Matrix3d& P, const Eigen::Vector3d& N) const {
    return lambda * P.trace() * N + mu * (P * N + P.transpose() * N);
  }

  inline void eval(const Eigen::Vector3d& d, const Eigen::Vector3d& N, const Eigen::Vector3d& n,
                   Eigen::Matrix3d& D, Eigen::Matrix3d& S) const {
    const double r = d.norm();
    const Eigen::Vector3d g = d / r;
    const double a = 3.0 - 4.0 * nu;
    auto del = [](int i, int j) { return i == j ? 1.0 : 0.0; };
    const double c2 = cU / (r * r), c3 = cU / (r * r * r);
    for (int m = 0; m < 3; ++m) {
      // P_kl = dU_km/dx_l = -dU_km/dy_l
      Eigen::Matrix3d P;
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          P(k, l) = -c2 * (-a * del(k, m) * g(l) + del(k, l) * g(m) + del(m, l) * g(k) -
                           3.0 * g(k) * g(m) * g(l));
      D.col(m) = traction(P, N);
    }
    // Q^{(m)}_kl = dT_km/dx_l = -C_mnpq n_n d2U_kp/dy_l dy_q
    Eigen::Matrix3d Q[3];
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        Eigen::Matrix3d H;  // H_pq = d2 U_kp / dy_l dy_q
        for (int p = 0; p < 3; ++p)
          for (int q = 0; q < 3; ++q)
            H(p, q) = c3 * (a * del(k, p) * (3.0 * g(l) * g(q) - del(l, q)) + del(k, q) * del(p, l) +
                            del(p, q) * del(k, l) -
                            3.0 * (del(k, q) * g(p) * g(l) + del(p, q) * g(k) * g(l) +
                                   del(k, l) * g(p) * g(q) + del(p, l) * g(k) * g(q) +
                                   del(q, l) * g(k) * g(p)) +
                            15.0 * g(k) * g(p) * g(q) * g(l));
        const Eigen::Vector3d tk = -traction(H, n);  // index m
        for (int m = 0; m < 3; ++m) Q[m](k, l) = tk(m);
      }
    }
    for (int m = 0; m < 3; ++m) S.col(m) = traction(Q[m], N);
  }
};

}  // namespace igabem
