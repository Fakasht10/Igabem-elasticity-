// Finite-difference checks of the Kelvin kernels:
//   T_km  = C_mnpq n_n dU_kp/dy_q             (traction of U at y)
//   D_im  = C_ijkl N_j dU_km/dx_l             (traction of U at x)
//   S_im  = C_ijkl N_j dT_km/dx_l             (traction of T at x)
// Returns non-zero if any relative error exceeds 1e-7.
#include <algorithm>
#include <cstdio>

#include "kelvin.hpp"

using namespace igabem;
using M3 = Eigen::Matrix3d;
using V3 = Eigen::Vector3d;

int main() {
  Material mat{1.3, 0.7};
  Kelvin K(mat);
  KelvinTraction KT(mat);
  const V3 x(0.1, -0.2, 0.3), y(0.7, 0.4, -0.5);
  const V3 N = V3(0.3, -0.5, 0.8).normalized(), n = V3(-0.6, 0.2, 0.4).normalized();
  const double h = 1e-5;
  auto U = [&](const V3& xx, const V3& yy) { M3 u, t; K.eval(yy - xx, n, u, t); return u; };
  auto T = [&](const V3& xx, const V3& yy) { M3 u, t; K.eval(yy - xx, n, u, t); return t; };
  // Central-difference gradient in x or y of a kernel, column m -> P_kl
  auto grad = [&](auto F, int m, bool wrt_x) {
    M3 P;
    for (int l = 0; l < 3; ++l) {
      V3 e = V3::Zero(); e(l) = h;
      const M3 Fp = wrt_x ? F(x + e, y) : F(x, y + e);
      const M3 Fm = wrt_x ? F(x - e, y) : F(x, y - e);
      P.col(l) = (Fp.col(m) - Fm.col(m)) / (2 * h);
    }
    return P;
  };
  M3 Tfd, Dfd, Sfd, Ua, Ta, Da, Sa;
  K.eval(y - x, n, Ua, Ta);
  KT.eval(y - x, N, n, Da, Sa);
  for (int k = 0; k < 3; ++k) {
    // dU_kp/dy_q with k fixed: U symmetric, so take column k and vary p as rows
    const M3 G = grad(U, k, false);  // G(p, q) = dU_pk/dy_q = dU_kp/dy_q
    Tfd.row(k) = KT.traction(G, n).transpose();
  }
  for (int m = 0; m < 3; ++m) {
    Dfd.col(m) = KT.traction(grad(U, m, true), N);
    Sfd.col(m) = KT.traction(grad(T, m, true), N);
  }
  // Displacement-gradient kernels P[m] = dU_.m/dx, Q[m] = dT_.m/dx (interior evaluation).
  M3 P[3], Q[3];
  KT.gradients(y - x, n, P, Q);
  double eP = 0.0, eQ = 0.0;
  for (int m = 0; m < 3; ++m) {
    eP = std::max(eP, (grad(U, m, true) - P[m]).norm() / P[m].norm());
    eQ = std::max(eQ, (grad(T, m, true) - Q[m]).norm() / Q[m].norm());
  }
  const double eT = (Tfd - Ta).norm() / Ta.norm(), eD = (Dfd - Da).norm() / Da.norm(),
               eS = (Sfd - Sa).norm() / Sa.norm();
  std::printf("rel. error  T: %.2e  D: %.2e  S: %.2e  dU/dx: %.2e  dT/dx: %.2e\n", eT, eD, eS, eP, eQ);
  return (eT < 1e-7 && eD < 1e-7 && eS < 1e-7 && eP < 1e-7 && eQ < 1e-7) ? 0 : 1;
}
