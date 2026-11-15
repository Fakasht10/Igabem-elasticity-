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

  // One kernel times w into a column-major 3 x 3 block with leading dimension ld
  // (out(i, j) = out[i + ld j]); the same values as eval.
  inline void evalU(const Eigen::Vector3d& d, double w, double* out, int ld) const {
    const double r = d.norm();
    const Eigen::Vector3d g = d / r;
    const double fu = w * cU / r, a = 3.0 - 4.0 * nu;
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i) out[i + ld * j] = fu * ((i == j ? a : 0.0) + g(i) * g(j));
  }
  inline void evalT(const Eigen::Vector3d& d, const Eigen::Vector3d& n, double w, double* out, int ld) const {
    const double r = d.norm();
    const Eigen::Vector3d g = d / r;
    const double drdn = g.dot(n), b = 1.0 - 2.0 * nu, ft = w * cT / (r * r);
    for (int j = 0; j < 3; ++j)
      for (int i = 0; i < 3; ++i)
        out[i + ld * j] = ft * (drdn * ((i == j ? b : 0.0) + 3.0 * g(i) * g(j)) - b * (g(i) * n(j) - g(j) * n(i)));
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

  // Displacement-gradient kernels (no normal at x): P[m]_kl = dU_km/dx_l and
  // Q[m]_kl = dT_km/dx_l, so that at an interior point
  //   du_k/dx_l = int P[m]_kl t_m - int Q[m]_kl u_m.
  inline void gradients(const Eigen::Vector3d& d, const Eigen::Vector3d& n, Eigen::Matrix3d P[3],
                        Eigen::Matrix3d Q[3]) const {
    const double r = d.norm();
    const Eigen::Vector3d g = d / r;
    const double a = 3.0 - 4.0 * nu;
    auto del = [](int i, int j) { return i == j ? 1.0 : 0.0; };
    const double c2 = cU / (r * r), c3 = cU / (r * r * r);
    for (int m = 0; m < 3; ++m) {
      // P_kl = dU_km/dx_l = -dU_km/dy_l
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          P[m](k, l) = -c2 * (-a * del(k, m) * g(l) + del(k, l) * g(m) + del(m, l) * g(k) -
                              3.0 * g(k) * g(m) * g(l));
    }
    // Q^{(m)}_kl = dT_km/dx_l = -C_mnpq n_n d2U_kp/dy_l dy_q
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
  }

  // D (traction kernel of U at x on N) and S (hypersingular: traction at x on N of T with normal n at y), the
  // contractions of evalRef in closed form, as 3 x 3 outer products of g, N, n (72 ns against 310 ns for evalRef;
  // equal to 1e-14 relative, tests/test_kernels.cpp). With g = (y - x)/r, a = 3 - 4 nu, c2 = cU / r^2, c3 = c2 / r:
  //   D_km = 2 lambda c2 (1 - 2 nu) g_m N_k - mu c2 [(1 - a) (delta_km gN + N_m g_k) + 2 N_k g_m - 6 g_k g_m gN],
  //   S_im = -(lambda N_i A_m + mu B_im), A_m = sum_k t^kk_m, B_im = sum_l N_l (t^il_m + t^li_m), with
  //   t^kl_m = c3 [lambda T1_kl n_m + mu (T2 + T3)^kl_m] the traction on n of H^kl_pq = d2 U_kp / dy_l dy_q:
  //   T1 = (a - 1)(3 g_k g_l - delta_kl), T2^kl_m = sum_q W_kmql n_q, T3^kl_m = sum_p W_kpml n_p (W = H / c3).
  inline void eval(const Eigen::Vector3d& d, const Eigen::Vector3d& Nv, const Eigen::Vector3d& nv,
                   Eigen::Matrix3d& Dm, Eigen::Matrix3d& Sm) const {
    using V = Eigen::Vector3d;
    using M = Eigen::Matrix3d;
    const double r2 = d.squaredNorm(), r = std::sqrt(r2), ir = 1.0 / r;
    const V g = d * ir, N = Nv, n = nv;  // copies: the outputs may alias the inputs
    const double a = 3.0 - 4.0 * nu, am1 = a - 1.0, c2 = cU / r2, c3 = c2 * ir;
    const double gN = g.dot(N), gn = g.dot(n), Nn = N.dot(n);
    // D = c2 [2 lambda (1 - 2 nu) N g^T + mu ((a - 1)(gN I + g N^T) - 2 N g^T + 6 gN g g^T)]
    M D = (c2 * (2.0 * lambda * (1.0 - 2.0 * nu) - 2.0 * mu)) * N * g.transpose() +
          (c2 * mu) * (am1 * g * N.transpose() + 6.0 * gN * g * g.transpose());
    D.diagonal().array() += c2 * mu * am1 * gN;
    // S = -(lambda N A^T + mu B): A = c3 mu (a - 1)(6 gn g - 2 n),
    // B = c3 [2 lambda (a - 1)(3 gN g n^T - N n^T) + mu (X + Y)], X + Y collected into outer products
    const V A = (c3 * mu * am1) * (6.0 * gn * g - 2.0 * n);
    M XY = ((3.0 * a - 9.0) * Nn + 60.0 * gN * gn) * g * g.transpose() + ((3.0 * a - 9.0) * gN) * n * g.transpose() -
           (12.0 * gN) * g * n.transpose() - (12.0 * gn) * N * g.transpose() + ((3.0 * a - 9.0) * gn) * g * N.transpose() -
           (2.0 * am1) * n * N.transpose() + 4.0 * N * n.transpose();
    XY.diagonal().array() += (3.0 * a - 9.0) * gN * gn - 2.0 * am1 * Nn;
    const M B = c3 * ((2.0 * lambda * am1) * (3.0 * gN * g - N) * n.transpose() + mu * XY);
    Sm = -(lambda * N * A.transpose() + mu * B);
    Dm = D;
  }

  // Reference: the contractions of gradients() spelled out (tests only).
  inline void evalRef(const Eigen::Vector3d& d, const Eigen::Vector3d& N, const Eigen::Vector3d& n,
                      Eigen::Matrix3d& D, Eigen::Matrix3d& S) const {
    Eigen::Matrix3d P[3], Q[3];
    gradients(d, n, P, Q);
    for (int m = 0; m < 3; ++m) D.col(m) = traction(P[m], N);
    for (int m = 0; m < 3; ++m) S.col(m) = traction(Q[m], N);
  }
};

}  // namespace igabem
