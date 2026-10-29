// Tensor-product NURBS patches and multipatch surfaces.
// Reads the Bembel / GeoPDEs ".dat" format (control points stored in
// homogeneous form x*w, y*w, z*w, w; u index runs fastest).
#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace igabem {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;

// Index of the knot span [U[s], U[s+1]) containing u (Piegl & Tiller A2.1).
inline int findSpan(int n_cp, int p, double u, const std::vector<double>& U) {
  if (u >= U[n_cp]) return n_cp - 1;
  if (u <= U[p]) return p;
  int lo = p, hi = n_cp, mid = (lo + hi) / 2;
  while (u < U[mid] || u >= U[mid + 1]) {
    if (u < U[mid]) hi = mid; else lo = mid;
    mid = (lo + hi) / 2;
  }
  return mid;
}

// Non-zero B-spline basis functions and first derivatives at u on span s
// (Piegl & Tiller A2.3 with n = 1). N[0][k], N[1][k], k = 0..p.
inline void basisFunsDers(int s, double u, int p, const std::vector<double>& U,
                          double N[2][16]) {
  double ndu[16][16], left[16], right[16];
  ndu[0][0] = 1.0;
  for (int j = 1; j <= p; ++j) {
    left[j] = u - U[s + 1 - j];
    right[j] = U[s + j] - u;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      ndu[j][r] = right[r + 1] + left[j - r];
      double temp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j][j] = saved;
  }
  for (int j = 0; j <= p; ++j) N[0][j] = ndu[j][p];
  if (p == 0) { N[1][0] = 0.0; return; }
  for (int j = 0; j <= p; ++j) {
    double d = 0.0;
    if (j - 1 >= 0) d += ndu[j - 1][p - 1] / ndu[p][j - 1];
    if (j <= p - 1) d -= ndu[j][p - 1] / ndu[p][j];
    N[1][j] = p * d;
  }
}

// Rational basis and geometry evaluated at one parametric point of a patch.
struct PatchEval {
  int n_loc = 0;
  int idx[256];        // local control point indices within the patch
  double R[256];       // rational basis values
  double Ru[256], Rv[256];
  Vec3 x, xu, xv;
  Vec3 normal() const { return xu.cross(xv); }  // un-normalized, |.| = surface Jacobian
};

struct Patch {
  int p = 0, q = 0;    // degrees in u and v
  int nu = 0, nv = 0;  // number of control points
  std::vector<double> U, V;
  std::vector<Vec4> Pw;  // homogeneous control points, index i + j*nu

  Vec3 cp(int k) const { return Pw[k].head<3>() / Pw[k](3); }
  double w(int k) const { return Pw[k](3); }

  std::vector<double> uniqueKnots(const std::vector<double>& K) const {
    std::vector<double> out;
    for (double k : K) if (out.empty() || k > out.back() + 1e-14) out.push_back(k);
    return out;
  }

  double greville(int i, bool in_u) const {
    const auto& K = in_u ? U : V;
    int deg = in_u ? p : q;
    if (deg == 0) return 0.5 * (K[i] + K[i + 1]);
    double s = 0.0;
    for (int k = 1; k <= deg; ++k) s += K[i + k];
    return s / deg;
  }

  void eval(double u, double v, PatchEval& E) const {
    const int su = findSpan(nu, p, u, U), sv = findSpan(nv, q, v, V);
    evalSpan(su, sv, u, v, E);
  }

  void evalSpan(int su, int sv, double u, double v, PatchEval& E) const {
    double Nu[2][16], Nv[2][16];
    basisFunsDers(su, u, p, U, Nu);
    basisFunsDers(sv, v, q, V, Nv);
    double W = 0, Wu = 0, Wv = 0;
    Vec3 A = Vec3::Zero(), Au = Vec3::Zero(), Av = Vec3::Zero();
    int k = 0;
    for (int b = 0; b <= q; ++b) {
      for (int a = 0; a <= p; ++a, ++k) {
        const int id = (su - p + a) + (sv - q + b) * nu;
        const Vec4& Pk = Pw[id];  // homogeneous: (w x, w y, w z, w)
        const double B = Nu[0][a] * Nv[0][b], Bu = Nu[1][a] * Nv[0][b], Bv = Nu[0][a] * Nv[1][b];
        const double N = B * Pk(3), Nu_ = Bu * Pk(3), Nv_ = Bv * Pk(3);
        E.idx[k] = id;
        E.R[k] = N; E.Ru[k] = Nu_; E.Rv[k] = Nv_;
        W += N; Wu += Nu_; Wv += Nv_;
        A += B * Pk.head<3>(); Au += Bu * Pk.head<3>(); Av += Bv * Pk.head<3>();
      }
    }
    E.n_loc = k;
    const double iW = 1.0 / W;
    for (int i = 0; i < k; ++i) {
      const double R = E.R[i] * iW;
      E.Ru[i] = (E.Ru[i] - R * Wu) * iW;
      E.Rv[i] = (E.Rv[i] - R * Wv) * iW;
      E.R[i] = R;
    }
    E.x = A * iW;
    E.xu = (Au - E.x * Wu) * iW;
    E.xv = (Av - E.x * Wv) * iW;
  }
};

inline std::vector<Patch> loadDat(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file);
  std::string line;
  std::vector<std::string> lines;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    lines.push_back(line);
  }
  std::istringstream head(lines.at(0));
  int dim, rdim, npatch;
  head >> dim >> rdim >> npatch;
  std::vector<Patch> out;
  size_t l = 1;
  for (int ip = 0; ip < npatch; ++ip) {
    if (lines.at(l).rfind("PATCH", 0) == 0) ++l;
    Patch P;
    { std::istringstream s(lines.at(l++)); s >> P.p >> P.q; }
    { std::istringstream s(lines.at(l++)); s >> P.nu >> P.nv; }
    auto readRow = [&](int n) {
      std::istringstream s(lines.at(l++));
      std::vector<double> r(n);
      for (auto& x : r) s >> x;
      return r;
    };
    P.U = readRow(P.nu + P.p + 1);
    P.V = readRow(P.nv + P.q + 1);
    const int n = P.nu * P.nv;
    auto X = readRow(n), Y = readRow(n), Z = readRow(n), W = readRow(n);
    P.Pw.resize(n);
    for (int k = 0; k < n; ++k) P.Pw[k] = Vec4(X[k], Y[k], Z[k], W[k]);
    out.push_back(std::move(P));
  }
  return out;
}

}  // namespace igabem
