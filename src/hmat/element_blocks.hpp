// Exact SBIE blocks of a set of collocation points against one element, in the
// element's local basis (unassembled columns): the kernel used by the
// hierarchical operator (elem_hmatrix.hpp) for its near field, its ACA samples
// and its dense leaves.
//
//   H(3k + i, 3l + j) = int_e T_ij(x_k, y) R_l(y) dS,   G(3k + i, 3l + j) = int_e U_ij(x_k, y) R_l(y) dS
//
// for the points x_k = colloc[pts[k]] and the element's n_loc basis functions
// R_l (local control points idx[l] of its patch). Separated pairs take the
// cached Gauss rule of order farRuleOrder(tol) (grouped by order, one kernel
// call per group, vectorized over points, contracted with the basis by GEMM:
// the scheme of DenseSbie); close and own pairs the exact quadrature
// (ElementIntegrator::integrate). The same rule for a pair as DenseSbie with the
// same tol / min_order, so the entries agree with the dense operator.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "../integrator.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"

namespace igabem {

class ElementBlocks {
 public:
  ElementBlocks(const Mesh& m, const Material& mat, const SolverOptions& o, double tol, int min_order = -1)
      : mesh_(m), K_(mat), quad_(m, o, true, maxOrder(o, tol)), tol_(tol), min_order_(min_order) {}
  // Highest far-rule order farRuleOrder can ask for: the closest far pair (d = 1.5 r).
  static int maxOrder(const SolverOptions& o, double tol) {
    const double rho = 1.5 + std::sqrt(1.25);
    return int(std::ceil(std::log(1.0 / std::max(tol, o.far_tol)) / (2.0 * std::log(rho))));
  }

  const ElementIntegrator& quad() const { return quad_; }
  void releaseCache() { quad_.releaseCache(); }
  const Mesh& mesh() const { return mesh_; }
  const Kelvin& kelvin() const { return K_; }
  int nloc(int ie) const {
    ElementIntegrator::RuleData r;
    quad_.ruleData(ie, 2, r);
    return r.n_loc;
  }
  const int* localIdx(int ie) const {
    ElementIntegrator::RuleData r;
    quad_.ruleData(ie, 2, r);
    return r.idx;
  }
  // Rule order of the pair (point a, element ie): 0 = exact (close or own) quadrature.
  int order(int a, int ie) const {
    const Collocation& p = mesh_.colloc[a];
    const Element& e = mesh_.elements[ie];
    if (e.patch == p.patch && e.rect.contains(p.xi, 1e-14)) return 0;
    return quad_.farRuleOrder(ie, (quad_.elementCentre(ie) - p.x).norm(), tol_, min_order_);
  }

  struct Work {
    std::vector<std::vector<int>> group;  // per order: positions k in pts
    std::vector<int> near;
    std::vector<double> X, Y, Z, KT, KU, Out, yq, nq, wq;
  };

  // H, G: 3 g x 3 n_loc, column-major with leading dimension ld (>= 3 g), overwritten. G may be null.
  void block(const int* pts, int g, int ie, double* H, double* G, long ld, Work& w) const {
    ElementIntegrator::RuleData r;
    quad_.ruleData(ie, 2, r);
    const int nl = r.n_loc;
    for (int c = 0; c < 3 * nl; ++c) {
      std::fill(H + ld * c, H + ld * c + 3 * g, 0.0);
      if (G) std::fill(G + ld * c, G + ld * c + 3 * g, 0.0);
    }
    const int mmax = quad_.maxCachedOrder();
    w.group.resize(mmax + 1);
    for (auto& v : w.group) v.clear();
    w.near.clear();
    for (int k = 0; k < g; ++k) {
      const int m = order(pts[k], ie);
      if (m) w.group[m].push_back(k);
      else w.near.push_back(k);
    }
    const double b = 1.0 - 2.0 * K_.nu, a = 3.0 - 4.0 * K_.nu;
    for (int m = 2; m <= mmax; ++m) {
      const std::vector<int>& P = w.group[m];
      const int gg = int(P.size());
      if (!gg) continue;
      quad_.ruleData(ie, m, r);
      const int n = r.n;
      w.X.resize(gg); w.Y.resize(gg); w.Z.resize(gg);
      for (int k = 0; k < gg; ++k) {
        const Vec3& x = mesh_.colloc[pts[P[k]]].x;
        w.X[k] = x(0); w.Y[k] = x(1); w.Z[k] = x(2);
      }
      w.yq.resize(3 * n); w.nq.resize(3 * n); w.wq.resize(n);
      for (int q = 0; q < n; ++q) {
        const Vec3 nJ = r.nJ[q] * mesh_.normal_sign;
        const double J = nJ.norm();
        for (int d = 0; d < 3; ++d) { w.yq[3 * q + d] = r.x[q](d); w.nq[3 * q + d] = nJ(d) / J; }
        w.wq[q] = r.w[q] * J;
      }
      const long M3 = 3L * gg, blk = M3 * n;  // component j: 3 gg x n, row i gg + k
      w.KT.resize(3 * blk);
      if (G) w.KU.resize(3 * blk);
      const double* X = w.X.data();
      const double* Y = w.Y.data();
      const double* Z = w.Z.data();
      for (int q = 0; q < n; ++q) {
        const double yx = w.yq[3 * q], yy = w.yq[3 * q + 1], yz = w.yq[3 * q + 2];
        const double nx = w.nq[3 * q], ny = w.nq[3 * q + 1], nz = w.nq[3 * q + 2];
        const double cT = K_.cT * w.wq[q], cU = K_.cU * w.wq[q];
        double* T = w.KT.data() + M3 * q;
#pragma omp simd
        for (int k = 0; k < gg; ++k) {
          const double dx = yx - X[k], dy = yy - Y[k], dz = yz - Z[k];
          const double ir = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
          const double gx = dx * ir, gy = dy * ir, gz = dz * ir;
          const double dn = gx * nx + gy * ny + gz * nz, ft = cT * ir * ir, s = 3.0 * dn;
          const double xy = b * (gx * ny - gy * nx), xz = b * (gx * nz - gz * nx), yz2 = b * (gy * nz - gz * ny);
          T[k] = ft * (dn * b + s * gx * gx);
          T[gg + k] = ft * (s * gy * gx + xy);
          T[2 * gg + k] = ft * (s * gz * gx + xz);
          T[blk + k] = ft * (s * gx * gy - xy);
          T[blk + gg + k] = ft * (dn * b + s * gy * gy);
          T[blk + 2 * gg + k] = ft * (s * gz * gy + yz2);
          T[2 * blk + k] = ft * (s * gx * gz - xz);
          T[2 * blk + gg + k] = ft * (s * gy * gz - yz2);
          T[2 * blk + 2 * gg + k] = ft * (dn * b + s * gz * gz);
        }
        if (G) {
          double* U = w.KU.data() + M3 * q;
#pragma omp simd
          for (int k = 0; k < gg; ++k) {
            const double dx = yx - X[k], dy = yy - Y[k], dz = yz - Z[k];
            const double ir = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
            const double gx = dx * ir, gy = dy * ir, gz = dz * ir, fu = cU * ir;
            const double u01 = fu * gx * gy, u02 = fu * gx * gz, u12 = fu * gy * gz;
            U[k] = fu * (a + gx * gx); U[gg + k] = u01; U[2 * gg + k] = u02;
            U[blk + k] = u01; U[blk + gg + k] = fu * (a + gy * gy); U[blk + 2 * gg + k] = u12;
            U[2 * blk + k] = u02; U[2 * blk + gg + k] = u12; U[2 * blk + 2 * gg + k] = fu * (a + gz * gz);
          }
        }
      }
      // Out_j (3 gg x nl) = K_j (3 gg x n) R^T, scattered to rows 3 P[k] + i, columns 3 l + j.
      const Eigen::Map<const Eigen::MatrixXd> R(r.R, nl, n);
      w.Out.resize(M3 * nl);
      Eigen::Map<Eigen::MatrixXd> Om(w.Out.data(), M3, nl);
      for (int op = 0; op < (G ? 2 : 1); ++op) {
        double* B = op ? G : H;
        const double* Kd = op ? w.KU.data() : w.KT.data();
        for (int j = 0; j < 3; ++j) {
          Om.noalias() = Eigen::Map<const Eigen::MatrixXd>(Kd + j * blk, M3, n) * R.transpose();
          for (int l = 0; l < nl; ++l) {
            double* col = B + ld * (3L * l + j);
            const double* src = w.Out.data() + M3 * l;
            for (int i = 0; i < 3; ++i)
              for (int k = 0; k < gg; ++k) col[3 * P[k] + i] = src[i * gg + k];
          }
        }
      }
    }
    // Close and own pairs: the exact quadrature, both kernels from one call.
    const Element& e = mesh_.elements[ie];
    for (int k : w.near) {
      const Collocation& p = mesh_.colloc[pts[k]];
      const SourcePoint src{p.patch, p.xi, p.x};
      quad_.integrate(src, e, [&](const QPoint& E, double wq) {
        const Vec3 nJ = E.normal() * mesh_.normal_sign;
        const double J = nJ.norm();
        Eigen::Matrix3d U, T;
        K_.eval(E.x - p.x, Vec3(nJ / J), U, T);
        T *= wq * J;
        U *= wq * J;
        for (int l = 0; l < E.n_loc; ++l) {
          const double Rl = E.R[l];
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
              H[3 * k + i + ld * (3 * l + j)] += Rl * T(i, j);
              if (G) G[3 * k + i + ld * (3 * l + j)] += Rl * U(i, j);
            }
        }
      });
    }
  }

  // One point against one element: H, G 3 x 3 n_loc (ld 3), overwritten; vectorized over the
  // quadrature points (the row samples of the compression). G may be null.
  void row(int a, int ie, double* H, double* G, Work& w) const {
    const int m = order(a, ie);
    if (!m) { block(&a, 1, ie, H, G, 3, w); return; }
    ElementIntegrator::RuleData r;
    quad_.ruleData(ie, m, r);
    const int n = r.n, nl = r.n_loc;
    const Vec3& x = mesh_.colloc[a].x;
    const double b = 1.0 - 2.0 * K_.nu, aa = 3.0 - 4.0 * K_.nu, sg = mesh_.normal_sign;
    w.KT.resize(9 * size_t(n));
    if (G) w.KU.resize(6 * size_t(n));
    double* T = w.KT.data();
    double* U = G ? w.KU.data() : nullptr;
#pragma omp simd
    for (int q = 0; q < n; ++q) {
      const double nx0 = r.nJ[q](0) * sg, ny0 = r.nJ[q](1) * sg, nz0 = r.nJ[q](2) * sg;
      const double J = std::sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0), iJ = 1.0 / J;
      const double nx = nx0 * iJ, ny = ny0 * iJ, nz = nz0 * iJ, wq = r.w[q] * J;
      const double dx = r.x[q](0) - x(0), dy = r.x[q](1) - x(1), dz = r.x[q](2) - x(2);
      const double ir = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
      const double gx = dx * ir, gy = dy * ir, gz = dz * ir;
      const double dn = gx * nx + gy * ny + gz * nz, ft = K_.cT * wq * ir * ir, s = 3.0 * dn;
      const double xy = b * (gx * ny - gy * nx), xz = b * (gx * nz - gz * nx), yz2 = b * (gy * nz - gz * ny);
      // T(i, j) at T[(3 j + i) n + q]
      T[0 * n + q] = ft * (dn * b + s * gx * gx);
      T[1 * n + q] = ft * (s * gy * gx + xy);
      T[2 * n + q] = ft * (s * gz * gx + xz);
      T[3 * n + q] = ft * (s * gx * gy - xy);
      T[4 * n + q] = ft * (dn * b + s * gy * gy);
      T[5 * n + q] = ft * (s * gz * gy + yz2);
      T[6 * n + q] = ft * (s * gx * gz - xz);
      T[7 * n + q] = ft * (s * gy * gz - yz2);
      T[8 * n + q] = ft * (dn * b + s * gz * gz);
      if (U) {
        const double fu = K_.cU * wq * ir;
        U[0 * n + q] = fu * (aa + gx * gx); U[1 * n + q] = fu * (aa + gy * gy); U[2 * n + q] = fu * (aa + gz * gz);
        U[3 * n + q] = fu * gx * gy; U[4 * n + q] = fu * gx * gz; U[5 * n + q] = fu * gy * gz;
      }
    }
    // contraction with the basis by one GEMM: Out (15 x nl) = K (15 x n) R^T, K rows: T (3 j + i), U (6)
    const int nk = U ? 15 : 9;
    w.Out.resize(size_t(nk) * nl);
    Eigen::Map<Eigen::MatrixXd> Om(w.Out.data(), nk, nl);
    const Eigen::Map<const Eigen::MatrixXd> R(r.R, nl, n);
    Om.topRows(9).noalias() = Eigen::Map<const Eigen::MatrixXd>(T, n, 9).transpose() * R.transpose();
    if (U) Om.bottomRows(6).noalias() = Eigen::Map<const Eigen::MatrixXd>(U, n, 6).transpose() * R.transpose();
    static constexpr int um[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
    for (int l = 0; l < nl; ++l)
      for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
          H[3 * (3 * l + j) + i] = Om(3 * j + i, l);
          if (G) G[3 * (3 * l + j) + i] = Om(9 + um[i][j], l);
        }
  }

 private:
  const Mesh& mesh_;
  Kelvin K_;
  ElementIntegrator quad_;
  double tol_;
  int min_order_;
};

}  // namespace igabem
