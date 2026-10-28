// Gauss-Legendre rules and the element partitioning of Taus et al. (2019),
// Appendix B: curvature (kappa*h <= 1/8), aspect ratio (chi <= 3) and
// near-singular refinement (refine while d <= 3h).
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "nurbs.hpp"

namespace igabem {

struct GaussRule {
  std::vector<double> x, w;  // on [0, 1]
};

inline GaussRule gaussLegendre(int n) {
  GaussRule g;
  g.x.resize(n);
  g.w.resize(n);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int k = 1; k <= n; ++k) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p2) / k;
      }
      dp = n * (z * p0 - p1) / (z * z - 1.0);
      const double dz = p0 / dp;
      z -= dz;
      if (std::abs(dz) < 1e-16) break;
    }
    g.x[i] = 0.5 * (1.0 - z);
    g.w[i] = 1.0 / ((1.0 - z * z) * dp * dp);  // = w_[-1,1] / 2
  }
  return g;
}

// Axis-aligned rectangle in a patch's parameter domain.
struct Rect {
  double u0, u1, v0, v1;
  double du() const { return u1 - u0; }
  double dv() const { return v1 - v0; }
  bool contains(const Vec2& p, double tol) const {
    return p(0) >= u0 - tol && p(0) <= u1 + tol && p(1) >= v0 - tol && p(1) <= v1 + tol;
  }
  Vec2 clamp(const Vec2& p) const {
    return Vec2(std::min(std::max(p(0), u0), u1), std::min(std::max(p(1), v0), v1));
  }
};

struct PartitionOptions {
  bool enabled = true;
  double kappa_h = 0.125;  // max normal-angle change across a cell (radians)
  double aspect = 3.0;
  int max_depth = 8;
};

// Recursively split an element into integration cells (App. B, steps 1-7,
// with the normal-angle span across a cell used as the kappa*h estimate).
inline void partitionElement(const Patch& P, int su, int sv, const Rect& R,
                             const PartitionOptions& opt, int depth,
                             std::vector<Rect>& cells) {
  if (!opt.enabled || depth >= opt.max_depth) { cells.push_back(R); return; }
  PatchEval E;
  const double us[3] = {R.u0, 0.5 * (R.u0 + R.u1), R.u1};
  const double vs[3] = {R.v0, 0.5 * (R.v0 + R.v1), R.v1};
  Vec3 X[3][3], N[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      P.evalSpan(su, sv, us[i], vs[j], E);
      X[i][j] = E.x;
      N[i][j] = E.normal().normalized();
    }
  double ang = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ang = std::max(ang, std::acos(std::min(1.0, N[1][1].dot(N[i][j]))));
  ang *= 2.0;  // centre-to-corner angle doubled ~ angle across the cell
  const double Lu = ((X[2][0] - X[0][0]).norm() + (X[2][2] - X[0][2]).norm()) * 0.5;
  const double Lv = ((X[0][2] - X[0][0]).norm() + (X[2][2] - X[2][0]).norm()) * 0.5;
  const bool curv = ang > opt.kappa_h;
  const bool split_u = curv || Lu > opt.aspect * Lv;
  const bool split_v = curv || Lv > opt.aspect * Lu;
  if (!split_u && !split_v) { cells.push_back(R); return; }
  const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
  if (split_u && split_v) {
    partitionElement(P, su, sv, {R.u0, um, R.v0, vm}, opt, depth + 1, cells);
    partitionElement(P, su, sv, {um, R.u1, R.v0, vm}, opt, depth + 1, cells);
    partitionElement(P, su, sv, {R.u0, um, vm, R.v1}, opt, depth + 1, cells);
    partitionElement(P, su, sv, {um, R.u1, vm, R.v1}, opt, depth + 1, cells);
  } else if (split_u) {
    partitionElement(P, su, sv, {R.u0, um, R.v0, R.v1}, opt, depth + 1, cells);
    partitionElement(P, su, sv, {um, R.u1, R.v0, R.v1}, opt, depth + 1, cells);
  } else {
    partitionElement(P, su, sv, {R.u0, R.u1, R.v0, vm}, opt, depth + 1, cells);
    partitionElement(P, su, sv, {R.u0, R.u1, vm, R.v1}, opt, depth + 1, cells);
  }
}

}  // namespace igabem
