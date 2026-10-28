// Traction space: discontinuous per-patch NURBS space for tractions, with its
// collocation points and interpolation of traction data.
#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "mesh.hpp"

namespace igabem {

// Traction space: the geometry's rational basis on each patch, discontinuous
// across patches (no merging), collocated at Greville points whose first and
// last entries are moved halfway inwards so that every point is interior to
// its patch.
struct TractionSpace {
  struct Point {
    int patch, su, sv;
    Vec2 xi;
    PatchEval E;  // geometry and basis at the point
    Vec3 x, N;
  };
  std::vector<int> offset;
  std::vector<Point> points;
  int ndof = 0;

  static double shifted(const Patch& P, int i, bool in_u) {
    const int n = in_u ? P.nu : P.nv;
    if (i == 0) return 0.5 * (P.greville(0, in_u) + P.greville(1, in_u));
    if (i == n - 1) return 0.5 * (P.greville(n - 2, in_u) + P.greville(n - 1, in_u));
    return P.greville(i, in_u);
  }

  void build(const Mesh& m) {
    offset.clear();
    points.clear();
    ndof = 0;
    for (size_t ip = 0; ip < m.patches.size(); ++ip) {
      const Patch& P = m.patches[ip];
      if (P.nu < 3 || P.nv < 3) throw std::runtime_error("traction space needs >= 3 cps per direction");
      offset.push_back(ndof);
      for (int k = 0; k < P.nu * P.nv; ++k) {
        Point c;
        c.patch = int(ip);
        c.xi = Vec2(shifted(P, k % P.nu, true), shifted(P, k / P.nu, false));
        c.su = findSpan(P.nu, P.p, c.xi(0), P.U);
        c.sv = findSpan(P.nv, P.q, c.xi(1), P.V);
        P.evalSpan(c.su, c.sv, c.xi(0), c.xi(1), c.E);
        c.x = c.E.x;
        c.N = (c.E.normal() * m.normal_sign).normalized();
        points.push_back(c);
      }
      ndof += P.nu * P.nv;
    }
    // Per-patch interpolation matrices (point i, basis j), factored once.
    lu.clear();
    for (size_t ip = 0; ip < m.patches.size(); ++ip) {
      const int n0 = offset[ip], n = (ip + 1 < m.patches.size() ? offset[ip + 1] : ndof) - n0;
      Eigen::MatrixXd Np = Eigen::MatrixXd::Zero(n, n);
      for (int i = 0; i < n; ++i) {
        const PatchEval& E = points[n0 + i].E;
        for (int l = 0; l < E.n_loc; ++l) Np(i, E.idx[l]) += E.R[l];
      }
      lu.emplace_back(Np);
    }
  }

  // Coefficients (3 per traction dof, one column per load case) of the
  // interpolant of traction data t(y, N(y)) returning a 3 x K block.
  template <class F>
  Eigen::MatrixXd interpolate(F&& traction, int K) const {
    Eigen::MatrixXd C(3 * ndof, K);
    for (size_t ip = 0; ip < lu.size(); ++ip) {
      const int n0 = offset[ip], n = int(lu[ip].rows());
      Eigen::MatrixXd T(n, 3 * K);
      for (int i = 0; i < n; ++i) {
        const auto t = traction(points[n0 + i].x, points[n0 + i].N);
        for (int k = 0; k < K; ++k) T.block(i, 3 * k, 1, 3) = t.col(k).transpose();
      }
      const Eigen::MatrixXd c = lu[ip].solve(T);
      for (int i = 0; i < n; ++i)
        for (int k = 0; k < K; ++k) C.block(3 * (n0 + i), k, 3, 1) = c.block(i, 3 * k, 1, 3).transpose();
    }
    return C;
  }

  std::vector<Eigen::PartialPivLU<Eigen::MatrixXd>> lu;
};

}  // namespace igabem
