// Dense SBIE operators by element-batched assembly: the same entries as
// SbieEntries (cached Gauss rules at tolerance tol for separated element pairs,
// the exact quadrature for close ones), computed in a different order.
//
// Rows are split into chunks of collocation points (one per task, so threads
// write disjoint rows). Per chunk and element, the points that take a cached
// rule are grouped by its order m; for each group the kernel is evaluated once
// for the whole group (SoA, vectorized over points), U and T together, and
// contracted with the element's basis values by one GEMM per component:
//
//   A(3a + i, 3 col(l) + j) += sum_q K_ij(x_a, y_q) w_q J_q R_l(y_q).
//
// Each task accumulates one patch at a time into a buffer of its rows times the
// patch's control points (cache resident) and adds it to A once: written
// directly, every entry of A would be read and written once per element of its
// column's support (25 at p = 4), which made the 17k build memory bound.
//
// Known tractions need no G: with t(y_q) = sum_l R_l(y_q) c_l precomputed per
// cached point, G c costs 9 multiply-adds per (point, quadrature point, case)
// instead of a contraction against the n_loc basis functions. Close pairs use
// ElementIntegrator::integrate with U and T from one kernel call.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../integrator.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"
#include "../traction_space.hpp"

namespace igabem {

class DenseSbie {
 public:
  // Output of assemble. H: raw H (no rigid-body term), 3 M x 3 M, column-major
  // with leading dimension ldh. G: 3 M x 3 Mt (ldg), or null. C: traction
  // coefficients (3 Mt x K) whose image G C goes to GC (3 M x K), or null.
  struct Out {
    double* H = nullptr;
    long ldh = 0;
    double* G = nullptr;
    long ldg = 0;
    const Eigen::MatrixXd* C = nullptr;
    Eigen::MatrixXd* GC = nullptr;
    // Optional block row sums of raw H, S(3a + i, j) = sum_B Hraw(3a + i, 3B + j)
    // (3 M x 3, column-major), for the rigid-body term.
    double* S = nullptr;
  };

  // Profile counters of the last assemble (summed over threads, seconds of CPU).
  struct Stats {
    long far_pairs = 0, far_qp = 0, near_pairs = 0, near_qp = 0;
    double t_far = 0.0, t_near = 0.0, t_class = 0.0, t_flush = 0.0, t_busy = 0.0;
  };

  DenseSbie(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& o, double tol,
            int min_order = -1, int chunk = 96)
      : mesh_(m), ts_(ts), K_(mat), quad_(m, o), tol_(tol), min_order_(min_order), chunk_(chunk) {
    patch_elems_.assign(m.patches.size(), {});
    for (size_t ie = 0; ie < m.elements.size(); ++ie) patch_elems_[m.elements[ie].patch].push_back(int(ie));
  }

  const Stats& stats() const { return stats_; }

  void assemble(const Out& o) const {
    const int np = int(mesh_.colloc.size()), ne = int(mesh_.elements.size());
    const int K = o.C ? int(o.C->cols()) : 0;
    if (K > 8) throw std::runtime_error("DenseSbie: at most 8 load cases per pass");
    if (o.GC) o.GC->setZero(3 * np, K);
    // Traction data at every cached point of every element (only the orders used are touched).
    std::vector<std::vector<double>> tq(ne);  // per element: [order][q][j][case], orders 2..max
    std::vector<std::vector<long>> tq0(ne);
    if (K) {
#pragma omp parallel for schedule(dynamic, 16)
      for (int ie = 0; ie < ne; ++ie) {
        const Element& e = mesh_.elements[ie];
        tq0[ie].assign(quad_.maxCachedOrder() + 1, -1);
        ElementIntegrator::RuleData r;
        for (int m = 2; m <= quad_.maxCachedOrder(); ++m) {
          quad_.ruleData(ie, m, r);
          tq0[ie][m] = long(tq[ie].size());
          tq[ie].resize(tq[ie].size() + size_t(3) * K * r.n, 0.0);
          double* t = tq[ie].data() + tq0[ie][m];
          for (int q = 0; q < r.n; ++q)
            for (int l = 0; l < r.n_loc; ++l) {
              const double R = r.R[r.n_loc * q + l];
              const int c = ts_.offset[e.patch] + r.idx[l];
              for (int j = 0; j < 3; ++j)
                for (int k = 0; k < K; ++k) t[(3 * q + j) * K + k] += R * (*o.C)(3 * c + j, k);
            }
        }
      }
    }
    Stats st;
    const int nchunk = (np + chunk_ - 1) / chunk_;
    if (o.S) std::fill(o.S, o.S + 9L * np, 0.0);
#pragma omp parallel
    {
      Work w;
      Stats ls;
#pragma omp for schedule(dynamic, 1)
      for (int ch = 0; ch < nchunk; ++ch) {
        const double tch = now();
        const int a0 = ch * chunk_, a1 = std::min(np, a0 + chunk_);
        const long ldb = 3L * (a1 - a0);
        for (size_t ip = 0; ip < mesh_.patches.size(); ++ip) {
          const Patch& P = mesh_.patches[ip];
          const int ncp = P.nu * P.nv;
          w.BH.assign(ldb * 3 * ncp, 0.0);
          if (o.G) w.BG.assign(ldb * 3 * ncp, 0.0);
          for (int ie : patch_elems_[ip]) element(ie, a0, a1, ldb, o, K, tq, tq0, w, ls);
          const double tf0 = now();
          // Add the patch's columns to H (and G), and its row sums to S.
          const int cs = a1 - a0;
          for (int c = 0; c < ncp; ++c)
            for (int j = 0; j < 3; ++j) {
              const double* src = w.BH.data() + ldb * (3L * c + j);
              double* d = o.H + o.ldh * (3L * mesh_.l2g[ip][c] + j) + 3L * a0;
              for (int k = 0; k < cs; ++k)
                for (int i = 0; i < 3; ++i) d[3 * k + i] += src[i * cs + k];
              if (o.S) {
                double* Sd = o.S + 3L * np * j + 3L * a0;
                for (int k = 0; k < cs; ++k)
                  for (int i = 0; i < 3; ++i) Sd[3 * k + i] += src[i * cs + k];
              }
              if (o.G) {
                const double* sg = w.BG.data() + ldb * (3L * c + j);
                double* dg = o.G + o.ldg * (3L * (ts_.offset[ip] + c) + j) + 3L * a0;
                for (int k = 0; k < cs; ++k)
                  for (int i = 0; i < 3; ++i) dg[3 * k + i] += sg[i * cs + k];
              }
            }
          ls.t_flush += now() - tf0;
        }
        ls.t_busy += now() - tch;
      }
#pragma omp critical
      {
        st.far_pairs += ls.far_pairs; st.far_qp += ls.far_qp; st.near_pairs += ls.near_pairs;
        st.near_qp += ls.near_qp; st.t_far += ls.t_far; st.t_near += ls.t_near;
        st.t_class += ls.t_class; st.t_flush += ls.t_flush; st.t_busy += ls.t_busy;
      }
    }
    stats_ = st;
  }

 private:
  struct Work {
    std::vector<std::vector<int>> group;  // per order: chunk points
    std::vector<int> near;
    std::vector<double> X, Y, Z, KT, KU, Out, bacc, yq, nq, wq, accT, accU, BH, BG, u6;
    std::vector<int> idx;
  };

  const Mesh& mesh_;
  const TractionSpace& ts_;
  Kelvin K_;
  ElementIntegrator quad_;
  double tol_;
  int min_order_, chunk_;
  mutable Stats stats_;
  std::vector<std::vector<int>> patch_elems_;

  static double now() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
  }

  // Element ie against points a0 .. a1 - 1 into the patch buffers (rows i (a1 - a0) + a - a0,
  // columns 3 c + j for the patch's control point c).
  void element(int ie, int a0, int a1, long ldb, const Out& o, int K, const std::vector<std::vector<double>>& tq,
               const std::vector<std::vector<long>>& tq0, Work& w, Stats& ls) const {
    const Element& e = mesh_.elements[ie];
    const Vec3 c = quad_.elementCentre(ie);
    const int mmax = quad_.maxCachedOrder();
    w.group.resize(mmax + 1);
    for (auto& g : w.group) g.clear();
    w.near.clear();
    const double tc0 = now();
    for (int a = a0; a < a1; ++a) {
      const Collocation& p = mesh_.colloc[a];
      const bool own = e.patch == p.patch && e.rect.contains(p.xi, 1e-14);
      const int m = own ? 0 : quad_.farRuleOrder(ie, (c - p.x).norm(), tol_, min_order_);
      if (m) w.group[m].push_back(a);
      else w.near.push_back(a);
    }
    const double t0 = now();
    ls.t_class += t0 - tc0;
    const bool wantU = o.G || K;
    ElementIntegrator::RuleData r;
    for (int m = 2; m <= mmax; ++m) {
      const std::vector<int>& G = w.group[m];
      const int g = int(G.size());
      if (!g) continue;
      quad_.ruleData(ie, m, r);
      const int n = r.n, nl = r.n_loc;
      ls.far_pairs += g;
      ls.far_qp += long(g) * n;
      w.X.resize(g); w.Y.resize(g); w.Z.resize(g);
      for (int k = 0; k < g; ++k) {
        const Vec3& x = mesh_.colloc[G[k]].x;
        w.X[k] = x(0); w.Y[k] = x(1); w.Z[k] = x(2);
      }
      w.yq.resize(3 * n); w.nq.resize(3 * n); w.wq.resize(n);
      for (int q = 0; q < n; ++q) {
        const Vec3 nJ = r.nJ[q] * mesh_.normal_sign;
        const double J = nJ.norm();
        for (int d = 0; d < 3; ++d) { w.yq[3 * q + d] = r.x[q](d); w.nq[3 * q + d] = nJ(d) / J; }
        w.wq[q] = r.w[q] * J;
      }
      const long M3 = 3L * g, blk = M3 * n;  // one component j: 3g x n
      w.KT.resize(3 * blk);
      if (o.G) w.KU.resize(3 * blk);
      if (K) w.bacc.assign(size_t(3) * K * g, 0.0);
      if (wantU) w.u6.resize(6 * size_t(g));
      const double b = 1.0 - 2.0 * K_.nu, a = 3.0 - 4.0 * K_.nu;
      const double* X = w.X.data();
      const double* Y = w.Y.data();
      const double* Z = w.Z.data();
      for (int q = 0; q < n; ++q) {
        const double yx = w.yq[3 * q], yy = w.yq[3 * q + 1], yz = w.yq[3 * q + 2];
        const double nx = w.nq[3 * q], ny = w.nq[3 * q + 1], nz = w.nq[3 * q + 2];
        const double cT = K_.cT * w.wq[q], cU = K_.cU * w.wq[q];
        double* T = w.KT.data() + M3 * q;
        double* U = o.G ? w.KU.data() + M3 * q : nullptr;
        const double* tqq = K ? tq[ie].data() + tq0[ie][m] + size_t(3) * K * q : nullptr;
#pragma omp simd
        for (int k = 0; k < g; ++k) {
          const double dx = yx - X[k], dy = yy - Y[k], dz = yz - Z[k];
          const double ir = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
          const double gx = dx * ir, gy = dy * ir, gz = dz * ir;
          const double dn = gx * nx + gy * ny + gz * nz, ft = cT * ir * ir;
          // T_ij = ft (dn (b d_ij + 3 g_i g_j) - b (g_i n_j - g_j n_i)), row i*g + k of component j.
          const double s = 3.0 * dn;
          const double xy = b * (gx * ny - gy * nx), xz = b * (gx * nz - gz * nx), yz2 = b * (gy * nz - gz * ny);
          T[k] = ft * (dn * b + s * gx * gx);
          T[g + k] = ft * (s * gy * gx + xy);
          T[2 * g + k] = ft * (s * gz * gx + xz);
          T[blk + k] = ft * (s * gx * gy - xy);
          T[blk + g + k] = ft * (dn * b + s * gy * gy);
          T[blk + 2 * g + k] = ft * (s * gz * gy + yz2);
          T[2 * blk + k] = ft * (s * gx * gz - xz);
          T[2 * blk + g + k] = ft * (s * gy * gz - yz2);
          T[2 * blk + 2 * g + k] = ft * (dn * b + s * gz * gz);
        }
        if (wantU) {
          double* u = w.u6.data();  // U_00, U_11, U_22, U_01, U_02, U_12 per point
#pragma omp simd
          for (int k = 0; k < g; ++k) {
            const double dx = yx - X[k], dy = yy - Y[k], dz = yz - Z[k];
            const double ir = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
            const double gx = dx * ir, gy = dy * ir, gz = dz * ir, fu = cU * ir;
            u[k] = fu * (a + gx * gx); u[g + k] = fu * (a + gy * gy); u[2 * g + k] = fu * (a + gz * gz);
            u[3 * g + k] = fu * gx * gy; u[4 * g + k] = fu * gx * gz; u[5 * g + k] = fu * gy * gz;
          }
          if (U) {
            static constexpr int map[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) std::copy(u + map[i][j] * g, u + map[i][j] * g + g, U + j * blk + i * g);
          }
          for (int kc = 0; kc < K; ++kc) {
            const double t0 = tqq[kc], t1 = tqq[K + kc], t2 = tqq[2 * K + kc];
            double* b0 = w.bacc.data() + 3 * kc * g;
            double* b1 = b0 + g;
            double* b2 = b1 + g;
#pragma omp simd
            for (int k = 0; k < g; ++k) {
              b0[k] += u[k] * t0 + u[3 * g + k] * t1 + u[4 * g + k] * t2;
              b1[k] += u[3 * g + k] * t0 + u[g + k] * t1 + u[5 * g + k] * t2;
              b2[k] += u[4 * g + k] * t0 + u[5 * g + k] * t1 + u[2 * g + k] * t2;
            }
          }
        }
      }
      // Contraction with the basis: Out_j (3g x n_loc) = K_j (3g x n) R^T.
      const Eigen::Map<const Eigen::MatrixXd> R(r.R, nl, n);
      w.Out.resize(M3 * nl);
      Eigen::Map<Eigen::MatrixXd> Om(w.Out.data(), M3, nl);
      for (int op = 0; op < (o.G ? 2 : 1); ++op) {
        double* B = op ? w.BG.data() : w.BH.data();
        const double* Kd = op ? w.KU.data() : w.KT.data();
        for (int j = 0; j < 3; ++j) {
          Om.noalias() = Eigen::Map<const Eigen::MatrixXd>(Kd + j * blk, M3, n) * R.transpose();
          const int cs = a1 - a0;
          const bool contiguous = G[g - 1] - G[0] == g - 1;
          for (int l = 0; l < nl; ++l) {
            double* Bc = B + ldb * (3L * r.idx[l] + j);
            const double* src = w.Out.data() + M3 * l;
            for (int i = 0; i < 3; ++i) {
              double* d = Bc + i * cs - a0;
              const double* sr = src + i * g;
              if (contiguous) {
                d += G[0];
#pragma omp simd
                for (int k = 0; k < g; ++k) d[k] += sr[k];
              } else {
                for (int k = 0; k < g; ++k) d[G[k]] += sr[k];
              }
            }
          }
        }
      }
      if (K)
        for (int kc = 0; kc < K; ++kc)
          for (int i = 0; i < 3; ++i)
            for (int k = 0; k < g; ++k) (*o.GC)(3 * G[k] + i, kc) += w.bacc[(3 * kc + i) * g + k];
    }
    const double t1 = now();
    ls.t_far += t1 - t0;
    // Close pairs: the exact quadrature, both kernels from one call, summed per
    // local basis function (an element is one knot span: the same n_loc functions
    // at every point) and written once.
    for (int a : w.near) {
      const Collocation& p = mesh_.colloc[a];
      const SourcePoint src{p.patch, p.xi, p.x};
      ++ls.near_pairs;
      int nl = 0;
      const int* idx = nullptr;
      w.accT.assign(9 * 256, 0.0);
      if (o.G) w.accU.assign(9 * 256, 0.0);
      Eigen::Matrix<double, 3, 8> gc = Eigen::Matrix<double, 3, 8>::Zero();
      quad_.integrate(src, e, [&](const QPoint& E, double wq) {
        ++ls.near_qp;
        if (!idx) { idx = E.idx; nl = E.n_loc; w.idx.assign(E.idx, E.idx + nl); }
        const Vec3 nJ = E.normal() * mesh_.normal_sign;
        const double J = nJ.norm();
        Eigen::Matrix3d U, T;
        K_.eval(E.x - p.x, Vec3(nJ / J), U, T);
        T *= wq * J;
        U *= wq * J;
        Eigen::Matrix<double, 3, 8> tE = Eigen::Matrix<double, 3, 8>::Zero();
        for (int l = 0; l < nl; ++l) {
          const double Rl = E.R[l];
          Eigen::Map<Eigen::Matrix3d>(&w.accT[9 * l]) += Rl * T;
          if (o.G) Eigen::Map<Eigen::Matrix3d>(&w.accU[9 * l]) += Rl * U;
          for (int kc = 0; kc < K; ++kc) tE.col(kc) += Rl * o.C->block<3, 1>(3 * (ts_.offset[e.patch] + E.idx[l]), kc);
        }
        for (int kc = 0; kc < K; ++kc) gc.col(kc) += U * tE.col(kc);
      });
      for (int l = 0; l < nl; ++l)
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i) {
            const long at = long(i) * (a1 - a0) + (a - a0) + ldb * (3L * w.idx[l] + j);
            w.BH[at] += w.accT[9 * l + 3 * j + i];
            if (o.G) w.BG[at] += w.accU[9 * l + 3 * j + i];
          }
      for (int kc = 0; kc < K; ++kc) o.GC->block<3, 1>(3 * a, kc) += gc.col(kc);
    }
    ls.t_near += now() - t1;
  }
};

}  // namespace igabem
