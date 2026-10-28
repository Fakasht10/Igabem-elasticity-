// A hierarchical (H-) matrix built from an entry callback, on top of Htool
// (third_party/htool, MIT): geometric cluster trees on the row points and on the
// column anchors (inflated by each column's support radius), Rjasanow-Steinbach
// admissibility 2 min(r_t, r_s) < eta dist, adaptive cross approximation with
// SVD recompression to the relative block tolerance eps.
//
// Saved to and loaded from a binary stream leaf by leaf; loading rebuilds the
// (deterministic) cluster and block trees and fills the leaves from the file,
// so no entry is recomputed.
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/lrmat/recompressed_low_rank_generator.hpp>
#include <htool/hmatrix/lrmat/sympartialACA.hpp>
#include <htool/hmatrix/lrmat/utils/SVD_recompression.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/wrappers/wrapper_blas.hpp>

#include "vector_aca.hpp"

namespace igabem {

struct HOptions {
  double eps = 1e-6;  // relative (Frobenius) tolerance per low-rank block
  double eta = 2.0;   // admissibility
  int leaf = 48;      // maximal cluster size (rows or columns, 3 per point)
  std::string key() const {
    return "blockaca1_eps" + std::to_string(eps) + "_eta" + std::to_string(eta) + "_leaf" + std::to_string(leaf);
  }
};

struct HStats {
  long rows = 0, cols = 0;
  long dense_leaves = 0, lowrank_leaves = 0, max_rank = 0;
  double stored = 0.0, stored_dense = 0.0;  // doubles stored (all leaves, dense leaves)
  double bytes() const { return 8.0 * stored; }
  double denseBytes() const { return 8.0 * double(rows) * double(cols); }
  double ratio() const { return stored / (double(rows) * double(cols)); }  // stored / dense
};

// Adapts any object with block(M, N, rows, cols, out) to Htool's generator.
template <class Entries>
class HtoolGenerator final : public htool::VirtualGenerator<double> {
 public:
  explicit HtoolGenerator(const Entries& e) : e_(e) {}
  void copy_submatrix(int M, int N, const int* rows, const int* cols, double* ptr) const override {
    e_.block(M, N, rows, cols, ptr);
  }

 private:
  const Entries& e_;
};

class HOperator {
 public:
  HOperator() = default;
  HOperator(HOperator&&) = default;
  HOperator& operator=(HOperator&&) = default;

  // Entries: block(M, N, rows, cols, out) and rowsAgainst(points, dofs, f) as in
  // SbieEntries (3 rows and 3 columns per point). row_xyz: 3 coordinates per row;
  // col_xyz, col_radius: per column.
  template <class Entries>
  void build(const Entries& e, const std::vector<double>& row_xyz, const std::vector<double>& col_xyz,
             const std::vector<double>& col_radius, const HOptions& o) {
    clusters(row_xyz, col_xyz, col_radius, o);
    // The generators only serve the build; matrix-vector products need none of them.
    const int *tp = target_->get_permutation().data(), *sp = source_->get_permutation().data();
    const HtoolGenerator<Entries> gen(e);
    htool::InternalGeneratorWithPermutation<double> internal(gen, tp, sp);
    BlockACA<Entries> aca(e, tp, sp);
    htool::RecompressedLowRankGenerator<double> lr(aca, [](htool::LowRankMatrix<double>& m) {
      htool::SVD_recompression(m);
    });
    NearField<Entries> near(e, tp, sp);
    const auto t0 = std::chrono::steady_clock::now();
    assemble(internal, lr, o, &near);
    t_build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t_near = near.seconds;
  }

  double t_build = 0.0, t_near = 0.0;  // last build: total and near field (dense leaves), seconds

  int rows() const { return target_ ? target_->get_size() : 0; }
  int cols() const { return source_ ? source_->get_size() : 0; }

  // y = alpha A x + beta y, in the user numbering.
  void apply(const double* x, double* y, double alpha = 1.0, double beta = 0.0) const {
    htool::add_hmatrix_vector_product(exec_compat::par, 'N', alpha, *H_, x, beta, y);
  }
  Eigen::VectorXd operator*(const Eigen::VectorXd& x) const {
    Eigen::VectorXd y = Eigen::VectorXd::Zero(rows());
    apply(x.data(), y.data());
    return y;
  }
  Eigen::MatrixXd operator*(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(rows(), X.cols());
    for (int k = 0; k < X.cols(); ++k) apply(X.col(k).data(), Y.col(k).data());
    return Y;
  }

  HStats stats() const {
    HStats s;
    s.rows = rows();
    s.cols = cols();
    for (const auto* L : leaves()) {
      const long m = L->get_target_cluster().get_size(), n = L->get_source_cluster().get_size();
      if (L->is_dense()) { ++s.dense_leaves; s.stored += double(m) * n; s.stored_dense += double(m) * n; }
      else {
        const long k = L->get_low_rank_data()->rank_of();
        ++s.lowrank_leaves;
        s.max_rank = std::max(s.max_rank, k);
        s.stored += double(k) * (m + n);
      }
    }
    return s;
  }

  // Dense copy (tests on small meshes only).
  Eigen::MatrixXd dense() const {
    Eigen::MatrixXd A(rows(), cols());
    Eigen::VectorXd e = Eigen::VectorXd::Zero(cols());
    for (int j = 0; j < cols(); ++j) {
      e(j) = 1.0;
      A.col(j) = (*this) * e;
      e(j) = 0.0;
    }
    return A;
  }

  void save(std::ostream& f) const {
    const auto L = leaves();
    put(f, std::int64_t(rows()));
    put(f, std::int64_t(cols()));
    put(f, std::int64_t(L.size()));
    for (const auto* l : L) {
      const int ro = l->get_target_cluster().get_offset(), m = l->get_target_cluster().get_size();
      const int co = l->get_source_cluster().get_offset(), n = l->get_source_cluster().get_size();
      for (int v : {ro, m, co, n}) put(f, std::int32_t(v));
      if (l->is_dense()) {
        put(f, std::int32_t(-1));
        f.write(reinterpret_cast<const char*>(l->get_dense_data()->data()), std::streamsize(8 * size_t(m) * n));
      } else {
        const auto* lr = l->get_low_rank_data();
        const int k = lr->rank_of();
        put(f, std::int32_t(k));
        f.write(reinterpret_cast<const char*>(lr->get_U().data()), std::streamsize(8 * size_t(m) * k));
        f.write(reinterpret_cast<const char*>(lr->get_V().data()), std::streamsize(8 * size_t(k) * n));
      }
    }
  }

  // Same geometry and options as the saved operator; false if the file does not match.
  bool load(std::istream& f, const std::vector<double>& row_xyz, const std::vector<double>& col_xyz,
            const std::vector<double>& col_radius, const HOptions& o) {
    std::int64_t r, c, nl;
    if (!get(f, r) || !get(f, c) || !get(f, nl)) return false;
    if (3 * r != std::int64_t(row_xyz.size()) || 3 * c != std::int64_t(col_xyz.size())) return false;
    Cached cached;
    for (std::int64_t i = 0; i < nl; ++i) {
      std::int32_t ro, m, co, n, k;
      if (!get(f, ro) || !get(f, m) || !get(f, co) || !get(f, n) || !get(f, k)) return false;
      Leaf& L = cached.leaves[{ro, co, m, n}];
      L.rank = k;
      L.U.resize(size_t(m) * (k < 0 ? n : k));
      f.read(reinterpret_cast<char*>(L.U.data()), std::streamsize(8 * L.U.size()));
      if (k >= 0) {
        L.V.resize(size_t(k) * n);
        f.read(reinterpret_cast<char*>(L.V.data()), std::streamsize(8 * L.V.size()));
      }
      if (!f) return false;
    }
    clusters(row_xyz, col_xyz, col_radius, o);
    assemble(cached, cached, o);
    return !cached.missed;
  }

 private:
  std::unique_ptr<htool::Cluster<double>> target_, source_;
  std::unique_ptr<htool::HMatrix<double>> H_;

  struct Leaf {
    int rank = -1;  // -1: dense, U holds the block
    std::vector<double> U, V;
  };
  // Serves saved leaves to the tree builder, by (row offset, col offset, size).
  struct Cached final : htool::VirtualInternalGenerator<double>, htool::VirtualInternalLowRankGenerator<double> {
    std::map<std::tuple<int, int, int, int>, Leaf> leaves;
    mutable bool missed = false;
    const Leaf* find(int M, int N, int ro, int co) const {
      auto it = leaves.find({ro, co, M, N});
      if (it == leaves.end()) { missed = true; return nullptr; }
      return &it->second;
    }
    void copy_submatrix(int M, int N, int ro, int co, double* ptr) const override {
      const Leaf* L = find(M, N, ro, co);
      if (!L || L->rank >= 0) { missed = true; std::fill(ptr, ptr + size_t(M) * N, 0.0); return; }
      std::copy(L->U.begin(), L->U.end(), ptr);
    }
    bool copy_low_rank_approximation(int M, int N, int ro, int co, htool::LowRankMatrix<double>& lr) const override {
      auto it = leaves.find({ro, co, M, N});
      if (it == leaves.end() || it->second.rank < 0) return false;  // dense leaf: served above
      const Leaf& L = it->second;
      lr.get_U().resize(M, L.rank);
      lr.get_V().resize(L.rank, N);
      std::copy(L.U.begin(), L.U.end(), lr.get_U().data());
      std::copy(L.V.begin(), L.V.end(), lr.get_V().data());
      return true;
    }
    bool copy_low_rank_approximation(int M, int N, int ro, int co, int, htool::LowRankMatrix<double>& lr) const override {
      return copy_low_rank_approximation(M, N, ro, co, lr);
    }
  };

  void clusters(const std::vector<double>& row_xyz, const std::vector<double>& col_xyz,
                const std::vector<double>& col_radius, const HOptions& o) {
    htool::ClusterTreeBuilder<double> b;
    b.set_maximal_leaf_size(std::max(6, o.leaf));
    b.set_partitioning_strategy(std::make_shared<TriplePartitioning>());
    const int nr = int(row_xyz.size() / 3), nc = int(col_xyz.size() / 3);
    target_ = std::make_unique<htool::Cluster<double>>(
        b.create_cluster_tree(nr, 3, row_xyz.data(), nullptr, nullptr, 2, 2, nullptr, false));
    source_ = std::make_unique<htool::Cluster<double>>(
        b.create_cluster_tree(nc, 3, col_xyz.data(), col_radius.data(), nullptr, 2, 2, nullptr, false));
  }

  void assemble(const htool::VirtualInternalGenerator<double>& gen,
                const htool::VirtualInternalLowRankGenerator<double>& lr, const HOptions& o,
                htool::VirtualDenseBlocksGenerator<double>* near = nullptr) {
    htool::HMatrixTreeBuilder<double> tb(o.eps, o.eta, 'N', 'N');
    using DB = htool::VirtualDenseBlocksGenerator<double>;
    if (near) tb.set_dense_blocks_generator(std::shared_ptr<DB>(std::shared_ptr<DB>(), near));  // non-owning
    using LR = htool::VirtualInternalLowRankGenerator<double>;
    tb.set_low_rank_generator(std::shared_ptr<LR>(std::shared_ptr<LR>(), const_cast<LR*>(&lr)));  // non-owning
    H_ = std::make_unique<htool::HMatrix<double>>(tb.build(exec_compat::par, gen, *target_, *source_));
    H_->set_low_rank_generator(nullptr);  // lr does not outlive the build
  }

  std::vector<const htool::HMatrix<double>*> leaves() const {
    return htool::get_leaves_from(*H_).first;
  }

  template <class T>
  static void put(std::ostream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); }
  template <class T>
  static bool get(std::istream& f, T& v) { f.read(reinterpret_cast<char*>(&v), sizeof v); return bool(f); }
};

}  // namespace igabem
