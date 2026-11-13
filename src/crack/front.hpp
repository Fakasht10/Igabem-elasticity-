// Crack front and its Williams enrichment (paper Sec. 2.3, Eqs. 5 and 8).
//
// The front is the v = 1 edge of each crown patch of the upper face: one piece per crown patch, C_k(u), with the
// edge's univariate rational basis phi_s(u) (s = 0..nu-1) interpolating the SIFs K_{s,M}, M = I, II, III. A point y
// of crown patch k (upper or lower face) is projected onto piece k: u*(y), r = |y - C_k(u*)|. The local frame at
// C_k(u*) is (n, b, t): b = m the crack normal (pointing to the upper face's solid), t the front tangent, n = b x t
// pointing away from the crack (ahead of the front). Behind the front the Williams displacements on the two faces
// (theta = +-pi) reduce to
//
//   u(y) = +-(1 / 2 mu) sqrt(r / 2 pi) [ (kappa + 1) K_I b + (kappa + 1) K_II n + 4 K_III t ],   + upper, - lower,
//
// kappa = 3 - 4 nu (plane strain along a 3D front). The enrichment is phi_s(u*) times that, one scalar unknown per
// (s, M); it vanishes on the front, so the face tying (Eq. 13) involves only the NURBS parts.
//
// Flat cracks: b is the plane normal. A curved crack needs the Frenet-Serret frame of App. C here.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <utility>
#include <vector>

#include "../kelvin.hpp"
#include "../nurbs.hpp"

namespace igabem {
namespace crack {

// B-spline basis functions and derivatives up to order 2 at u on span s (Piegl & Tiller A2.3).
inline void basisFunsDers2(int s, double u, int p, const std::vector<double>& U, double N[3][16]) {
  double ndu[16][16], left[16], right[16], a[2][16];
  ndu[0][0] = 1.0;
  for (int j = 1; j <= p; ++j) {
    left[j] = u - U[s + 1 - j];
    right[j] = U[s + j] - u;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const double temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  for (int j = 0; j <= p; ++j) N[0][j] = ndu[j][p];
  for (int r = 0; r <= p; ++r) {
    int s1 = 0, s2 = 1;
    a[0][0] = 1.0;
    for (int k = 1; k <= 2; ++k) {
      double d = 0.0;
      const int rk = r - k, pk = p - k;
      if (k > p) { N[k][r] = 0.0; continue; }
      if (r >= k) {
        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
        d = a[s2][0] * ndu[rk][pk];
      }
      const int j1 = (rk >= -1) ? 1 : -rk, j2 = (r - 1 <= pk) ? k - 1 : p - r;
      for (int j = j1; j <= j2; ++j) {
        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
        d += a[s2][j] * ndu[rk + j][pk];
      }
      if (r <= pk) {
        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
        d += a[s2][k] * ndu[r][pk];
      }
      N[k][r] = d;
      std::swap(s1, s2);
    }
  }
  double f = p;
  for (int k = 1; k <= 2; ++k) {
    for (int j = 0; j <= p; ++j) N[k][j] *= f;
    f *= (p - k);
  }
}

// One front piece: the v = V.back() edge of an upper crown patch, as a rational curve of its own (control points
// and weights of the patch's last row), with the univariate basis phi_s, s = 0..nu-1.
struct FrontPiece {
  const Patch* P = nullptr;
  int patch = -1;       // upper-face crown patch
  double u0 = 0, u1 = 1;
  int first_col = 0;    // first scalar column of K_{s,M}: first_col + 3 s + M
  std::vector<Vec4> Pw;  // homogeneous control points of the edge

  void init(const Patch* Pa, int p) {
    P = Pa;
    patch = p;
    u0 = P->U.front();
    u1 = P->U.back();
    Pw.clear();
    for (int i = 0; i < P->nu; ++i) Pw.push_back(P->Pw[i + (P->nv - 1) * P->nu]);
  }
  int nbasis() const { return P->nu; }

  // Point, first and second derivative, and the nonzero rational basis (indices s, values R) at u.
  void eval(double u, Vec3& C, Vec3& Cu, int& n, int* s, double* R, Vec3* Cuu = nullptr) const {
    const int p = P->p, sp = findSpan(P->nu, p, u, P->U);
    double N[3][16];
    basisFunsDers2(sp, u, p, P->U, N);
    Vec4 A = Vec4::Zero(), A1 = Vec4::Zero(), A2 = Vec4::Zero();
    for (int a = 0; a <= p; ++a) {
      const Vec4& Q = Pw[sp - p + a];
      A += N[0][a] * Q;
      A1 += N[1][a] * Q;
      A2 += N[2][a] * Q;
    }
    const double W = A(3), W1 = A1(3), W2 = A2(3);
    C = A.head<3>() / W;
    Cu = (A1.head<3>() - W1 * C) / W;
    if (Cuu) *Cuu = (A2.head<3>() - 2 * W1 * Cu - W2 * C) / W;
    n = 0;
    for (int a = 0; a <= p; ++a) {
      const double r = N[0][a] * Pw[sp - p + a](3) / W;
      if (std::abs(r) < 1e-15) continue;
      s[n] = sp - p + a;
      R[n] = r;
      ++n;
    }
  }

  // Closest point parameter to y (Newton on g(u) = C'(u).(C(u) - y) from the guess, clamped to the piece).
  double project(const Vec3& y, double guess) const {
    double u = std::min(std::max(guess, u0), u1);
    int n, s[16];
    double R[16];
    Vec3 C, Cu, Cuu;
    for (int it = 0; it < 20; ++it) {
      eval(u, C, Cu, n, s, R, &Cuu);
      const double g = Cu.dot(C - y), dg = Cu.squaredNorm() + Cuu.dot(C - y);
      double un = u - g / (dg > 0 ? dg : Cu.squaredNorm());
      un = std::min(std::max(un, u0), u1);
      if (std::abs(un - u) < 1e-15 * (u1 - u0)) { u = un; break; }
      u = un;
    }
    return u;
  }
};

struct Front {
  std::vector<FrontPiece> pieces;  // by crown index
  Vec3 m;                          // crack normal (upper face's solid side)
  Vec3 centroid;                   // a point of the crack (orients n)
  double mu = 1, kappa = 1;
  int ncols = 0;

  // The 3 nb enrichment vectors at a point y of crown piece k on face sign (+1 upper, -1 lower), u guess from the
  // point's own patch parameter. cols[i], vals[i]: scalar column and its 3-vector.
  int shapes(int k, int sign, const Vec3& y, double uguess, int* cols, Vec3* vals) const {
    const FrontPiece& F = pieces[k];
    const double u = F.project(y, uguess);
    Vec3 C, Cu;
    int n, s[16];
    double R[16];
    F.eval(u, C, Cu, n, s, R);
    const double r = (y - C).norm();
    const Vec3 t0 = Cu.normalized();
    Vec3 nv = m.cross(t0);
    if (nv.dot(C - centroid) < 0) nv = -nv;
    const Vec3 tv = nv.cross(m);
    const double a = sign * std::sqrt(r / (2 * M_PI)) / (2 * mu);
    const Vec3 w[3] = {(kappa + 1) * a * m, (kappa + 1) * a * nv, 4.0 * a * tv};
    int c = 0;
    for (int i = 0; i < n; ++i)
      for (int M = 0; M < 3; ++M) {
        cols[c] = F.first_col + 3 * s[i] + M;
        vals[c] = R[i] * w[M];
        ++c;
      }
    return c;
  }

  // K_M(u) on piece k from the solution's K columns (offset into x).
  Vec3 sif(int k, double u, const Eigen::VectorXd& x) const {
    Vec3 C, Cu;
    int n, s[16];
    double R[16];
    pieces[k].eval(u, C, Cu, n, s, R);
    Vec3 K = Vec3::Zero();
    for (int i = 0; i < n; ++i) K += R[i] * x.segment<3>(pieces[k].first_col + 3 * s[i]);
    return K;
  }
};

}  // namespace crack
}  // namespace igabem
