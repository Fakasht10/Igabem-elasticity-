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
  // far_tol: quadrature tolerance of the far-field rules relative to the ACA
  // tolerance; far_order: their minimum Gauss order (0: the element's default).
  BlockACA(const Entries& e, const int* target_perm, const int* source_perm, double far_tol = 0.0,
           int far_order = 0)
      : e_(e), tp_(target_perm), sp_(source_perm), far_tol_(far_tol), far_order_(far_order) {}

  bool copy_low_rank_approximation(int M, int N, int ro, int co, htool::LowRankMatrix<double>& lr) const override {
    return aca(M, N, tp_ + ro, sp_ + co, lr.get_epsilon(), lr);
  }
  bool copy_low_rank_approximation(int M, int N, int ro, int co, int, htool::LowRankMatrix<double>& lr) const override {
    return aca(M, N, tp_ + ro, sp_ + co, lr.get_epsilon(), lr);
  }

 private:
  const Entries& e_;
  const int *tp_, *sp_;
  double far_tol_;
  int far_order_;

  bool aca(int M, int N, const int* rows, const int* cols, double eps, htool::LowRankMatrix<double>& lr) const {
    using Mat = Eigen::MatrixXd;
    const int m = M / 3, n = N / 3;
    if (3 * m != M || 3 * n != N) return false;
    Mat U, V;
    using FB = typename Entries::FarBlock;
    FB fq;
    const bool far = e_.farBlock(M, rows, N, cols, far_tol_ * eps, far_order_, fq);
    if (far) {
      const FB& fb = fq;
      // Far block: ACA of the point kernel K (cheap rows and columns), then V <- V W.
      const int nq = fb.points();
      auto row = [&](int i, Mat& R) {
        R.resize(3, 3 * nq);
        const int a = rows[3 * i] / 3;
        for (int q = 0; q < nq; ++q) e_.kernel(a, fb, q, R.data() + 9 * size_t(q), 3);
      };
      // Rows of a point come in the order of the cluster (any permutation of x, y, z).
      std::vector<char> plain(m);
      for (int i = 0; i < m; ++i)
        plain[i] = rows[3 * i] % 3 == 0 && rows[3 * i + 1] % 3 == 1 && rows[3 * i + 2] % 3 == 2;
      auto col = [&](int q, Mat& C) {
        C.resize(M, 3);
        double K[9];
        for (int i = 0; i < m; ++i) {
          if (plain[i]) { e_.kernel(rows[3 * i] / 3, fb, q, C.data() + 3 * i, M); continue; }
          e_.kernel(rows[3 * i] / 3, fb, q, K, 3);
          for (int c = 0; c < 3; ++c)
            for (int j = 0; j < 3; ++j) C(3 * i + c, j) = K[rows[3 * i + c] % 3 + 3 * j];
        }
      };
      // Cross approximation of K whose factor V_K = L K(I, :) (I the pivot rows)
      // is never formed: residual rows are tracked contracted, V_A = V_K W, and
      // K residual entries are formed only where needed (the pivot, a column).
      const long bound = (long(M) * N) / (M + N);
      bool ok = implicitACA(m, n, nq, row, col, fb, rows, 0.1 * eps, std::min<long>(2 * bound, std::min(M, N)), U, V);
      if (ok) ok = recompress(U, V, 0.5 * eps) <= bound;
      if (!ok) {
        auto rowx = [&](int i, Mat& R) { R.resize(3, N); e_.block(3, N, rows + 3 * i, cols, R.data()); };
        auto colx = [&](int j, Mat& C) { C.resize(M, 3); e_.block(M, 3, rows, cols + 3 * j, C.data()); };
        if (!blockACA(m, n, rowx, colx, eps, bound, U, V)) return false;
        recompress(U, V, eps);
      }
    } else {
      auto row = [&](int i, Mat& R) { R.resize(3, N); e_.block(3, N, rows + 3 * i, cols, R.data()); };
      auto col = [&](int j, Mat& C) { C.resize(M, 3); e_.block(M, 3, rows, cols + 3 * j, C.data()); };
      if (!blockACA(m, n, row, col, eps, (long(M) * N) / (M + N), U, V)) return false;
      recompress(U, V, eps);
    }
    lr.get_U().resize(M, int(U.cols()));
    lr.get_V().resize(int(V.rows()), N);
    std::copy(U.data(), U.data() + U.size(), lr.get_U().data());
    std::copy(V.data(), V.data() + V.size(), lr.get_V().data());
    return true;
  }

  // U V <- truncated SVD of U V (relative Frobenius tolerance eps); returns the rank.
  static long recompress(Eigen::MatrixXd& U, Eigen::MatrixXd& V, double eps) {  // rank <= min(rows, cols)
    if (U.cols() == 0) return 0;
    const long k = U.cols();
    // Triangular factors from the Gram matrices of the column-scaled factors (one
    // GEMM each; the truncation at eps >> sqrt(machine eps) is unaffected), and
    // Householder QR if a Gram matrix is numerically singular.
    const Eigen::VectorXd su = U.colwise().norm().transpose().cwiseMax(1e-300);
    const Eigen::VectorXd sv0 = V.rowwise().norm().cwiseMax(1e-300);
    const Eigen::MatrixXd Us = U * su.cwiseInverse().asDiagonal(), Vs = sv0.cwiseInverse().asDiagonal() * V;
    Eigen::MatrixXd Gu = Us.transpose() * Us, Gv = Vs * Vs.transpose();
    Eigen::LLT<Eigen::MatrixXd> lu(Gu), lv(Gv);
    auto good = [](const Eigen::LLT<Eigen::MatrixXd>& l) {
      if (l.info() != Eigen::Success) return false;
      const auto d = l.matrixLLT().diagonal().cwiseAbs();
      return d.minCoeff() > 1e-6 * d.maxCoeff();
    };
    if (!good(lu) || !good(lv)) return recompressQR(U, V, eps);
    const Eigen::MatrixXd Ru = Eigen::MatrixXd(lu.matrixU()) * su.asDiagonal();
    const Eigen::MatrixXd Rv = Eigen::MatrixXd(lv.matrixU()) * sv0.asDiagonal();
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Ru * Rv.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& sv = svd.singularValues();
    const long r = truncation(sv, eps);
    // U = (Us Lu^-T) X, Lu^-T X by a triangular solve.
    const Eigen::MatrixXd X = lu.matrixU().solve(Eigen::MatrixXd(svd.matrixU().leftCols(r) * sv.head(r).asDiagonal()));
    const Eigen::MatrixXd Y = lv.matrixU().solve(Eigen::MatrixXd(svd.matrixV().leftCols(r)));
    U = Us * X;
    V = Y.transpose() * Vs;
    return r;
  }

  static long truncation(const Eigen::VectorXd& sv, double eps) {
    const double total = sv.squaredNorm();
    long r = sv.size();
    double tail = 0.0;
    while (r > 1 && tail + sv(r - 1) * sv(r - 1) <= eps * eps * total) {
      --r;
      tail += sv(r) * sv(r);
    }
    return r;
  }

  static long recompressQR(Eigen::MatrixXd& U, Eigen::MatrixXd& V, double eps) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qu(U), qv(V.transpose());
    const long k = U.cols();
    const Eigen::MatrixXd Ru = qu.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    const Eigen::MatrixXd Rv = qv.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Ru * Rv.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& sv = svd.singularValues();
    const long r = truncation(sv, eps);
    // Q (core factor padded with zeros): the Householder reflections applied to r
    // columns only, Q itself is never formed.
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(U.rows(), r), Y = Eigen::MatrixXd::Zero(V.cols(), r);
    X.topRows(k) = svd.matrixU().leftCols(r) * sv.head(r).asDiagonal();
    Y.topRows(k) = svd.matrixV().leftCols(r);
    U = qu.householderQ() * X;
    V = (qv.householderQ() * Y).transpose();
    return r;
  }

  // Partially pivoted ACA on 3 x 3 blocks of an (3m x 3n) matrix given by row(i, R)
  // (rows of point i, 3 x 3n) and col(j, C) (columns of point j, 3m x 3). False if
  // the rank would exceed max_rank or nothing was found.
  template <class Row, class Col>
  static bool blockACA(int m, int n, Row&& row, Col&& col, double eps, long max_rank, Eigen::MatrixXd& U,
                       Eigen::MatrixXd& V) {
    using Mat = Eigen::MatrixXd;
    const int M = 3 * m, N = 3 * n;
    // Factors grow in place (capacity doubling); the first k columns are used. V is
    // kept transposed (N x k) so that appending a cross and the products with it
    // (residual rows, the Frobenius cross term) stream over contiguous columns.
    Mat Uc(M, 12), Vt(N, 12);
    int k = 0;
    std::vector<char> row_used(m, 0), col_used(n, 0);
    double frob2 = 0.0;
    int i = 0, checked = -1;
    Mat R, C;
    for (int step = 0; step < std::min(m, n); ++step) {
      if (i < 0) break;
      row_used[i] = 1;
      row(i, R);
      if (k) R.noalias() -= Uc.block(3 * i, 0, 3, k) * Vt.leftCols(k).transpose();
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
      col(j, C);
      if (k) C.noalias() -= Uc.leftCols(k) * Vt.block(3 * j, 0, 3, k).transpose();
      // Cross through the pseudo-inverse of the 3 x 3 pivot.
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(Eigen::Matrix3d(R.middleCols(3 * j, 3)),
                                            Eigen::ComputeFullU | Eigen::ComputeFullV);
      const auto& sv = svd.singularValues();
      int r = 0;
      while (r < 3 && sv(r) > 1e-12 * sv(0)) ++r;
      const Mat u = C * svd.matrixV().leftCols(r) * sv.head(r).cwiseInverse().asDiagonal();
      const Mat vt = R.transpose() * svd.matrixU().leftCols(r);  // N x r
      const double uv2 = ((u.transpose() * u).cwiseProduct(vt.transpose() * vt)).sum();
      const double cross =
          k ? ((Uc.leftCols(k).transpose() * u).cwiseProduct(Vt.leftCols(k).transpose() * vt)).sum() : 0.0;
      frob2 += uv2 + 2.0 * cross;
      if (k + r > max_rank) return false;  // not worth storing as low rank
      if (k + r > Uc.cols()) {
        const int cap = std::max<int>(2 * int(Uc.cols()), k + r);
        Uc.conservativeResize(M, cap);
        Vt.conservativeResize(N, cap);
      }
      Uc.middleCols(k, r) = u;
      Vt.middleCols(k, r) = vt;
      k += r;
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
        Mat Rc;
        row(i, Rc);
        const double full = Rc.squaredNorm();
        Rc.noalias() -= Uc.block(3 * i, 0, 3, k) * Vt.leftCols(k).transpose();
        if (Rc.squaredNorm() <= eps * eps * std::max(full, 1e-300)) break;
        checked = i;
      } else {
        checked = -1;
      }
    }
    U = Uc.leftCols(k);
    V = Vt.leftCols(k).transpose();
    return k > 0;
  }

  // ACA of A = K W (K: 3m x 3nq point kernel, W: 3nq x 3n sparse) with the pivot
  // rows of K and the pivot columns of K, but every residual kept in dof space: V_K
  // = L K(I, :) with I the pivot row points, so V_K(:, q) = L K(I, q) costs one
  // kernel call per pivot row. The pivot column of a row: among the quadrature
  // points supporting the dof with the largest contracted residual, the one with
  // the largest K residual (tried on its few largest basis weights).
  template <class Row, class Col, class FB>
  bool implicitACA(int m, int n, int nq, Row&& row, Col&& col, const FB& fb, const int* rows, double eps,
                   long max_rank, Eigen::MatrixXd& U, Eigen::MatrixXd& V) const {
    using Mat = Eigen::MatrixXd;
    const int M = 3 * m, N = 3 * n;
    Mat Uc(M, 12), VAt(N, 12), L(12, 12);  // L: k x 3s
    std::vector<int> piv;                   // pivot row points (indices into the block's points)
    int k = 0;
    std::vector<char> row_used(m, 0), dof_used(n, 0), q_used(nq, 0);
    double frob2 = 0.0;
    int i = 0, checked = -1;
    Mat R, RA, C, VKq;
    // V_K(:, q) = L K(I, q): k x 3.
    auto vk = [&](int q, Mat& out) {
      const int s = int(piv.size());
      Mat KI(3 * s, 3);
      double K9[9];
      for (int t = 0; t < s; ++t) {
        e_.kernel(rows[3 * piv[t]] / 3, fb, q, K9, 3);
        for (int c = 0; c < 3; ++c)
          for (int d = 0; d < 3; ++d) KI(3 * t + c, d) = K9[rows[3 * piv[t] + c] % 3 + 3 * d];
      }
      out.noalias() = L.topLeftCorner(k, 3 * s) * KI;
    };
    Mat Racc;
    Mat Ke;
    auto contract = [&](const Mat& Kr, Mat& Ra) {  // Ra = K row block (3 x 3nq) times W
      // Kr as 9 x nq (one 3 x 3 block per point), times the basis values per element.
      const Eigen::Map<const Mat> K9(Kr.data(), 9, nq);
      Racc.setZero(9, Eigen::Index(fb.dof_elems.size()));
      for (const auto& el : fb.elems) {
        Ke.noalias() = K9.middleCols(el.q0, el.nq) * Eigen::Map<const Mat>(el.R, el.n_loc, el.nq).transpose();
        for (const auto& [l, p] : el.lp) Racc.col(p) += Ke.col(l);
      }
      Ra.resize(3, N);
      for (int jj = 0; jj < N; ++jj) Ra.col(jj) = Racc.col(fb.col_pt[jj]).template segment<3>(3 * fb.col_comp[jj]);
    };
    auto contracted = [&](int ii, Mat& Kr, Mat& Ra) {
      row(ii, Kr);
      contract(Kr, Ra);
      if (k) Ra.noalias() -= Uc.block(3 * ii, 0, 3, k) * VAt.leftCols(k).transpose();
    };
    for (int step = 0; step < std::min(m, n); ++step) {
      if (i < 0) break;
      row_used[i] = 1;
      contracted(i, R, RA);
      // Pivot dof, then pivot quadrature point among its support.
      int j = -1;
      double best = 0.0;
      for (int p = 0; p < n; ++p)
        if (!dof_used[p]) {
          const double v = RA.middleCols(3 * p, 3).squaredNorm();
          if (v > best) { best = v; j = p; }
        }
      if (j < 0 || best <= 1e-30 * std::max(frob2, 1e-300)) { i = nextUnused(row_used); continue; }
      dof_used[j] = 1;
      std::vector<std::pair<double, int>> cand;
      for (const auto& [el, l] : fb.dof_elems[fb.col_pt[3 * j]]) {
        const auto& E = fb.elems[el];
        for (int t = 0; t < E.nq; ++t)
          if (!q_used[E.q0 + t]) cand.emplace_back(-std::abs(E.R[l + size_t(E.n_loc) * t]), E.q0 + t);
      }
      std::sort(cand.begin(), cand.end());
      int q = -1;
      double bq = -1.0;
      Eigen::Matrix3d P;
      for (size_t t = 0; t < std::min<size_t>(cand.size(), 6); ++t) {
        const int qq = cand[t].second;
        Eigen::Matrix3d Pq = R.middleCols(3 * qq, 3);
        if (k) { vk(qq, VKq); Pq -= Uc.block(3 * i, 0, 3, k) * VKq; }
        if (Pq.squaredNorm() > bq) { bq = Pq.squaredNorm(); q = qq; P = Pq; }
      }
      if (q < 0 || bq <= 1e-30 * std::max(frob2, 1e-300)) continue;  // try the next dof on another row
      q_used[q] = 1;
      col(q, C);
      if (k) { vk(q, VKq); C.noalias() -= Uc.leftCols(k) * VKq; }
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(P, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const auto& sv = svd.singularValues();
      int r = 0;
      while (r < 3 && sv(r) > 1e-12 * sv(0)) ++r;
      const Mat u = C * svd.matrixV().leftCols(r) * sv.head(r).cwiseInverse().asDiagonal();
      const Mat Us = svd.matrixU().leftCols(r);
      const Mat vt = RA.transpose() * Us;  // N x r
      const double uv2 = ((u.transpose() * u).cwiseProduct(vt.transpose() * vt)).sum();
      const double cross =
          k ? ((Uc.leftCols(k).transpose() * u).cwiseProduct(VAt.leftCols(k).transpose() * vt)).sum() : 0.0;
      frob2 += uv2 + 2.0 * cross;
      if (k + r > max_rank) return false;
      // New rows of L: Us^T [-U_i L, I_3] (the new pivot row block last).
      const int s = int(piv.size());
      Mat Lnew = Mat::Zero(r, 3 * (s + 1));
      if (k) Lnew.leftCols(3 * s) = -Us.transpose() * Uc.block(3 * i, 0, 3, k) * L.topLeftCorner(k, 3 * s);
      Lnew.rightCols(3) = Us.transpose();
      if (k + r > Uc.cols()) {
        const int cap = std::max<int>(2 * int(Uc.cols()), k + r);
        Uc.conservativeResize(M, cap);
        VAt.conservativeResize(N, cap);
      }
      if (k + r > L.rows() || 3 * (s + 1) > L.cols()) {
        Mat L2 = Mat::Zero(std::max<int>(2 * int(L.rows()), k + r), std::max<int>(2 * int(L.cols()), 3 * (s + 1)));
        L2.topLeftCorner(k, 3 * s) = L.topLeftCorner(k, 3 * s);
        L.swap(L2);
      }
      L.block(0, 3 * s, k, 3).setZero();
      L.block(k, 0, r, 3 * (s + 1)) = Lnew;
      piv.push_back(i);
      Uc.middleCols(k, r) = u;
      VAt.middleCols(k, r) = vt;
      k += r;
      i = -1;
      best = -1.0;
      for (int p = 0; p < m; ++p)
        if (!row_used[p]) {
          const double w = u.middleRows(3 * p, 3).squaredNorm();
          if (w > best) { best = w; i = p; }
        }
      if (uv2 <= eps * eps * frob2) {
        // Converged by the usual test: confirm on a few unused rows spread over the
        // cluster (the pivots can miss a part of it); the first failing one becomes
        // the next pivot row.
        if (checked >= 0 || i < 0) break;
        std::vector<int> probe = {farthestUnused(row_used)};
        for (int f : {m / 3, (2 * m) / 3, m - 1})
          for (int p = f; p >= 0; --p)
            if (!row_used[p]) { probe.push_back(p); break; }
        i = -1;
        for (int p : probe) {
          if (p < 0 || row_used[p]) continue;
          Mat Rc, Rac;
          row(p, Rc);
          contract(Rc, Rac);
          const double full = Rac.squaredNorm();
          Rac.noalias() -= Uc.block(3 * p, 0, 3, k) * VAt.leftCols(k).transpose();
          if (Rac.squaredNorm() > eps * eps * std::max(full, 1e-300)) { i = p; break; }
        }
        if (i < 0) break;
        checked = i;
      } else {
        checked = -1;
      }
    }
    U = Uc.leftCols(k);
    V = VAt.leftCols(k).transpose();
    return k > 0;
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
