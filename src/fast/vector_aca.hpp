// Pieces of the hierarchical compression that know the 3 x 3 structure of the
// elastostatic operators (rows and columns come in triples, one per point):
//
// TriplePartitioning  geometric bisection (principal axis) that never splits the
//                     three components of a point, so every cluster is a set of
//                     whole points;
// BlockACA            partially pivoted ACA on 3 x 3 blocks: each step takes the
//                     three rows of a point and the three columns of a point and
//                     adds a cross of rank <= 3 through the pseudo-inverse of the
//                     3 x 3 pivot. Scalar ACA misses whole components where the
//                     kernel decouples them (e.g. U_xz = 0 for coplanar points on
//                     a flat face), leaving O(1) errors in those entries; the
//                     block pivot sees all components at once. After the usual
//                     stopping test one more row, not yet used, is checked.
// NearField           all dense leaves at once, grouped by row cluster, so that
//                     each (collocation point, element) pair of a row cluster is
//                     integrated once instead of once per leaf.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include <htool/clustering/implementations/partitioning.hpp>
#include <htool/clustering/interfaces/virtual_partitioning.hpp>
#include <htool/hmatrix/interfaces/virtual_dense_blocks_generator.hpp>
#include <htool/hmatrix/interfaces/virtual_lrmat_generator.hpp>

namespace igabem {

class TriplePartitioning final : public htool::VirtualPartitioning<double> {
 public:
  std::vector<std::pair<int, int>> compute_partitioning(htool::Cluster<double>& c, int dim, const double* x,
                                                        const double* const radii, const double* const weights,
                                                        int nparts) override {
    std::vector<int>& perm = c.get_permutation();
    const int off = c.get_offset(), size = c.get_size();
    const auto dirs = htool::ComputeLargestExtent<double>::compute_direction(c, dim, x, radii, weights).first;
    auto key = [&](int i) {
      double k = 0.0;
      for (int d = 0; d < dim; ++d) k += x[dim * i + d] * dirs(d, 0);
      return std::make_tuple(k, i / 3, i);
    };
    std::sort(perm.begin() + off, perm.begin() + off + size, [&](int a, int b) { return key(a) < key(b); });
    const int half = 3 * ((size / 3) / 2);
    if (nparts != 2 || half == 0 || half == size) return {};  // no split
    return {{off, half}, {off + half, size - half}};
  }
};

template <class Entries>
class BlockACA final : public htool::VirtualInternalLowRankGenerator<double> {
 public:
  BlockACA(const Entries& e, const int* target_perm, const int* source_perm)
      : e_(e), tp_(target_perm), sp_(source_perm) {}

  bool copy_low_rank_approximation(int M, int N, int ro, int co, htool::LowRankMatrix<double>& lr) const override {
    return aca(M, N, tp_ + ro, sp_ + co, lr.get_epsilon(), lr);
  }
  bool copy_low_rank_approximation(int M, int N, int ro, int co, int, htool::LowRankMatrix<double>& lr) const override {
    return aca(M, N, tp_ + ro, sp_ + co, lr.get_epsilon(), lr);
  }

 private:
  const Entries& e_;
  const int *tp_, *sp_;

  bool aca(int M, int N, const int* rows, const int* cols, double eps, htool::LowRankMatrix<double>& lr) const {
    using Mat = Eigen::MatrixXd;
    const int m = M / 3, n = N / 3;
    if (3 * m != M || 3 * n != N) return false;
    Mat U(M, 0), V(0, N);
    std::vector<char> row_used(m, 0), col_used(n, 0);
    double frob2 = 0.0;
    int i = 0, checked = -1;
    for (int step = 0; step < std::min(m, n); ++step) {
      if (i < 0) break;
      row_used[i] = 1;
      Mat R(3, N);
      e_.block(3, N, rows + 3 * i, cols, R.data());
      if (U.cols()) R -= U.middleRows(3 * i, 3) * V;
      int j = -1;
      double best = 0.0;
      for (int q = 0; q < n; ++q)
        if (!col_used[q]) {
          const double v = R.middleCols(3 * q, 3).squaredNorm();
          if (v > best) { best = v; j = q; }
        }
      if (j < 0 || best <= 1e-30 * std::max(frob2, 1e-300)) {  // residual row is zero: try another row
        i = nextUnused(row_used);
        continue;
      }
      col_used[j] = 1;
      Mat C(M, 3);
      e_.block(M, 3, rows, cols + 3 * j, C.data());
      if (U.cols()) C -= U * V.middleCols(3 * j, 3);
      // Cross through the pseudo-inverse of the 3 x 3 pivot.
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(Eigen::Matrix3d(R.middleCols(3 * j, 3)),
                                            Eigen::ComputeFullU | Eigen::ComputeFullV);
      const auto& sv = svd.singularValues();
      int r = 0;
      while (r < 3 && sv(r) > 1e-12 * sv(0)) ++r;
      const Mat u = C * svd.matrixV().leftCols(r) * sv.head(r).cwiseInverse().asDiagonal();
      const Mat v = svd.matrixU().leftCols(r).transpose() * R;
      const double uv2 = ((u.transpose() * u).cwiseProduct(v * v.transpose())).sum();
      const double cross = U.cols() ? ((U.transpose() * u).cwiseProduct(V * v.transpose())).sum() : 0.0;
      frob2 += uv2 + 2.0 * cross;
      const int k = int(U.cols());
      if ((k + r) * (M + N) > M * N) return false;  // not worth storing as low rank
      U.conservativeResize(M, k + r);
      U.rightCols(r) = u;
      V.conservativeResize(k + r, N);
      V.bottomRows(r) = v;
      // Next row: largest entry of the new columns among unused rows.
      i = -1;
      best = -1.0;
      for (int p = 0; p < m; ++p)
        if (!row_used[p]) {
          const double w = u.middleRows(3 * p, 3).squaredNorm();
          if (w > best) { best = w; i = p; }
        }
      if (uv2 <= eps * eps * frob2) {
        // Converged by the usual test: confirm on one more unused row (it becomes
        // the next pivot row if its residual is not small).
        if (checked >= 0 || i < 0) break;
        i = farthestUnused(row_used);
        if (i < 0) break;
        Mat Rc(3, N);
        e_.block(3, N, rows + 3 * i, cols, Rc.data());
        const double full = Rc.squaredNorm();
        Rc -= U.middleRows(3 * i, 3) * V;
        if (Rc.squaredNorm() <= eps * eps * std::max(full, 1e-300)) break;
        checked = i;
      } else {
        checked = -1;
      }
    }
    if (U.cols() == 0) return false;
    lr.get_U().resize(M, int(U.cols()));
    lr.get_V().resize(int(V.rows()), N);
    std::copy(U.data(), U.data() + U.size(), lr.get_U().data());
    std::copy(V.data(), V.data() + V.size(), lr.get_V().data());
    return true;
  }

  static int nextUnused(const std::vector<char>& used) {
    for (size_t p = 0; p < used.size(); ++p)
      if (!used[p]) return int(p);
    return -1;
  }
  // An unused row far (in index, i.e. along the cluster ordering) from the used ones.
  static int farthestUnused(const std::vector<char>& used) {
    int best = -1, bd = -1;
    const int m = int(used.size());
    std::vector<int> d(m, m);
    for (int p = 0, last = -m; p < m; ++p) { if (used[p]) last = p; d[p] = p - last; }
    for (int p = m - 1, last = 2 * m; p >= 0; --p) { if (used[p]) last = p; d[p] = std::min(d[p], last - p); }
    for (int p = 0; p < m; ++p)
      if (!used[p] && d[p] > bd) { bd = d[p]; best = p; }
    return best;
  }
};

template <class Entries>
class NearField final : public htool::VirtualDenseBlocksGenerator<double> {
 public:
  NearField(const Entries& e, const int* target_perm, const int* source_perm)
      : e_(e), tp_(target_perm), sp_(source_perm) {}

  mutable double seconds = 0.0;

  void copy_dense_blocks(const std::vector<int>& M, const std::vector<int>& N, const std::vector<int>& ro,
                         const std::vector<int>& co, std::vector<double*>& ptr) const override {
    const auto t0 = std::chrono::steady_clock::now();
    std::map<std::pair<int, int>, std::vector<int>> groups;  // (row offset, rows) -> blocks
    for (size_t b = 0; b < M.size(); ++b) groups[{ro[b], M[b]}].push_back(int(b));
    std::vector<std::pair<std::pair<int, int>, std::vector<int>>> G(groups.begin(), groups.end());
#pragma omp parallel for schedule(dynamic)
    for (long g = 0; g < long(G.size()); ++g) {
      const int r0 = G[g].first.first, m = G[g].first.second;
      const std::vector<int>& blocks = G[g].second;
      std::vector<int> pts;
      for (int p = 0; p < m; p += 3) pts.push_back(tp_[r0 + p] / 3);
      std::vector<int> dofs;
      std::map<int, int> index;
      for (int b : blocks)
        for (int q = 0; q < N[b]; ++q) {
          const int d = sp_[co[b] + q] / 3;
          if (index.emplace(d, int(dofs.size())).second) dofs.push_back(d);
        }
      std::vector<std::vector<int>> cidx(blocks.size());
      for (size_t t = 0; t < blocks.size(); ++t) {
        const int b = blocks[t];
        cidx[t].resize(N[b]);
        for (int q = 0; q < N[b]; ++q) {
          const int s = sp_[co[b] + q];
          cidx[t][q] = 9 * index[s / 3] + 3 * (s % 3);
        }
      }
      e_.rowsAgainst(pts, dofs, [&](int k, const double* acc) {
        for (int c = 0; c < 3; ++c) {
          const int p = 3 * k + c, ri = tp_[r0 + p] % 3;
          for (size_t t = 0; t < blocks.size(); ++t) {
            const int b = blocks[t];
            for (int q = 0; q < N[b]; ++q) ptr[b][p + size_t(m) * q] = acc[cidx[t][q] + ri];
          }
        }
      });
    }
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

 private:
  const Entries& e_;
  const int *tp_, *sp_;
};

}  // namespace igabem
