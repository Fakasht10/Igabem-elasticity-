// Hierarchical SBIE operator for large collocation IgBEM problems: raw H
// (Hraw(a, B) = int T(x_a, y) N_B(y)) compressed, with G applied to given
// traction coefficients during the build, and the rigid-body diagonal term,
// so that H = Hraw - D as in CompressedSbie / DenseNeumann.
//
// Columns are clustered by ELEMENT, not by control point. A block is (points I)
// x (elements E): its entries are the integrals over the elements of E only,
// in their local bases, and are assembled to global control points afterwards.
// A control point's support straddles element clusters, so its column is the
// sum over several blocks; but no block ever needs a partial integral, the
// near-pair guard is the element's own distance, and the near field of a point
// cluster is one dense row block (no duplicated columns).
//
//   far blocks (admissible: max(diam I, diam E) <= eta dist(I, E))
//       ACA (partial pivoting, with an a-posteriori check on random rows) of the
//       joint unassembled block [H | gamma G] (rows: 3 per point, columns: 3 per
//       local basis function per element, H then G), sampled by point rows and
//       element columns of ElementBlocks; gamma = 2 mu / distance balances the
//       two kernels (|U| / |T| ~ r / 2 mu). The factors are assembled to global
//       columns and recompressed (QR + SVD, Frobenius tail <= eps). The G factor
//       is applied to the load cases at once; its columns on the kept traction
//       dofs (gkeep: supported and interface faces of a mixed problem) are stored,
//       the rest dropped. U, W_H, W_G are kept in single precision.
//   near field (every other pair)
//       per point leaf, the dense block of its points against the union of its
//       inadmissible element leaves, assembled (float); G applied at once, its
//       kept columns stored.
//
// Rows are the points `rows` (default mesh.colloc): a mixed problem adds the
// traction collocation points of its supported faces.
//
// Rigid-body term: D(a, B) = N_B(x_a) sum_B Hraw(a, B); the row sums come from
// the compressed Hraw (three products with the translations) and D is folded
// into the near rows (x_a lies in its own element, which is near).
#pragma once

#include <sys/resource.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../traction_space.hpp"
#include "cluster_tree.hpp"
#include "coarse.hpp"
#include "element_blocks.hpp"

namespace igabem {

struct ElemHOptions {
  double eps = 1e-6;      // relative Frobenius accuracy of a far block
  double eta = 2.0;       // admissibility
  int leaf_points = 96;   // point leaf size
  int leaf_elems = 64;    // element leaf size
  double tol = 1e-7;      // element far-field quadrature tolerance (entry rule)
  int min_order = -1;     // far-field minimum Gauss order, -1: from the degree
  int checks = 4;         // ACA a-posteriori check: random point rows
  int max_rank = 400;     // ACA fallback to dense beyond this rank
  int s0 = 32;            // CUR: initial point samples
  int s0e = 12;           // CUR: initial element samples
  bool use_aca = false;   // ACA instead of CUR
  bool adm_min = false;   // admissibility on min(diam I, diam E) instead of max
  bool merge_small = false;  // far blocks too small to sample join the near rows
  bool rsvd = true;       // randomized SVD for the exactly evaluated small far blocks
};

class ElemHMatrix {
 public:
  struct Stats {
    int far_blocks = 0, near_leaves = 0, dense_fallbacks = 0, max_rank = 0;
    double rank_mean = 0.0;
    double bytes_far = 0.0, bytes_near = 0.0;
    double t_tree = 0.0, t_near = 0.0, t_far = 0.0, t_rigid = 0.0;
    long near_pairs = 0;
    double cpu_row = 0.0, cpu_col = 0.0, cpu_aca = 0.0, cpu_rc = 0.0;  // far build profile (cpu s)
    long n_rows = 0, n_cols = 0;
    double cpu_cw = 0, cpu_qr = 0, cpu_rr = 0, cpu_test = 0;
    long attempts = 0;
    double rss_quad = 0.0, rss_near = 0.0;  // peak RSS (GB) before / after the near field
    int cur_failed = 0;
    double cur_failed_entries = 0.0;
  };

  // C: traction coefficients (3 Mt x K) whose image G C is returned by gc(). rows: collocation
  // points (default m.colloc); gkeep: per traction dof, store its G column (default none).
  ElemHMatrix(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& sopt,
              const Eigen::MatrixXd& C, const ElemHOptions& o = ElemHOptions(),
              const std::vector<Collocation>* rows = nullptr, const std::vector<char>* gkeep = nullptr)
      : mesh_(m), ts_(ts), rows_(rows ? *rows : m.colloc), o_(o), eb_(m, mat, sopt, o.tol, o.min_order, &rows_),
        mu_(mat.mu) {
    np_ = int(rows_.size());
    if (gkeep) gk_ = *gkeep;
    gk_.resize(ts.ndof, 0);
    for (char k : gk_) any_g_ = any_g_ || k;
    ne_ = int(m.elements.size());
    K_ = int(C.cols());
    GC_ = Eigen::MatrixXd::Zero(3 * np_, K_);
    auto t0 = now();
    // Element data: local -> global dof and traction dof.
    edof_.resize(ne_);
    etdof_.resize(ne_);
    for (int ie = 0; ie < ne_; ++ie) {
      const Element& e = m.elements[ie];
      const int nl = eb_.nloc(ie);
      const int* idx = eb_.localIdx(ie);
      for (int l = 0; l < nl; ++l) {
        edof_[ie].push_back(m.l2g[e.patch][idx[l]]);
        etdof_[ie].push_back(ts.offset[e.patch] + idx[l]);
      }
    }
    std::vector<Vec3> px(np_), ex(ne_);
    std::vector<double> er(ne_);
    for (int a = 0; a < np_; ++a) px[a] = rows_[a].x;
    for (int ie = 0; ie < ne_; ++ie) {
      ex[ie] = eb_.quad().elementCentre(ie);
      er[ie] = eb_.quad().elementRadius(ie);
    }
    pt_.build(px, {}, o.leaf_points);
    et_.build(ex, er, o.leaf_elems);
    partition(0, 0);
    st_.t_tree = seconds(t0);
    t0 = now();
    st_.rss_quad = peakGB();
    buildNear(C);
    st_.t_near = seconds(t0);
    st_.rss_near = peakGB();
    t0 = now();
    buildFar(C);
    st_.t_far = seconds(t0);
    t0 = now();
    rigid();
    st_.t_rigid = seconds(t0);
  }

  int rows() const { return 3 * np_; }
  int cols() const { return 3 * mesh_.ndof; }
  const Eigen::MatrixXd& gc() const { return GC_; }
  const Stats& stats() const { return st_; }

  // y = H x (with the rigid-body term); x: 3 ndof, y: 3 rows.
  void apply(const double* x, double* y) const { apply(x, nullptr, y, true); }
  // y = H u - G t, t (3 ts.ndof) read on the kept traction dofs only.
  void apply(const double* u, const double* t, double* y) const { apply(u, t, y, true); }
  const std::vector<Collocation>& rowPoints() const { return rows_; }
  // Largest element diameter in the support of each dof (the preconditioner radius unit).
  std::vector<double> dofSize() const {
    std::vector<double> h(mesh_.ndof, 0.0);
    for (int ie = 0; ie < ne_; ++ie)
      for (int B : edof_[ie]) h[B] = std::max(h[B], 2.0 * eb_.quad().elementRadius(ie));
    return h;
  }

  // Exact G(a, J) of the points `pts` against every traction dof J with mask[J]: add(k, i, J, j, v)
  // receives each element's part of G(3 pts[k] + i, 3 J + j), from one thread per point k. For
  // the preconditioner: the traction block of the supported faces is a first-kind single layer,
  // which a local pattern truncates too closely.
  template <class Add>
  void gBlock(const std::vector<int>& pts, const std::vector<char>& mask, Add&& add) const {
    std::vector<int> elems;
    for (int ie = 0; ie < ne_; ++ie)
      for (int J : etdof_[ie])
        if (mask[J]) { elems.push_back(ie); break; }
#pragma omp parallel
    {
      ElementBlocks::Work w;
      std::vector<double> Hb, Gb;
#pragma omp for schedule(dynamic, 4)
      for (int k = 0; k < int(pts.size()); ++k)
        for (int ie : elems) {
          const int nl = int(etdof_[ie].size());
          Hb.resize(9 * edof_[ie].size());
          Gb.resize(9 * nl);
          eb_.row(pts[k], ie, Hb.data(), Gb.data(), w);
          for (int l = 0; l < nl; ++l)
            if (mask[etdof_[ie][l]])
              for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) add(k, i, etdof_[ie][l], j, Gb[3 * (3 * l + j) + i]);
        }
    }
  }

  // Bytes of the integrator's far-rule cache; release it once no more entries are needed
  // (after the preconditioner pattern): only apply / applyCoarse work afterwards.
  double quadratureBytes() const { return eb_.quad().cacheBytes(); }
  void releaseQuadrature() { eb_.releaseCache(); }

  // H Zu - G Zt (3 rows x nc, float) for a coarse basis on the displacement dofs (Zu, 3 ndof rows)
  // and the kept traction dofs (Zt, 3 ts.ndof rows, may be null): each block once, the basis
  // restricted to its columns; rows scattered under a lock per point leaf.
  Eigen::MatrixXf applyCoarse(const SparseBasis& Zu, const SparseBasis* Zt) const {
    const int nc = Zu.nc;
    const int N = 3 * np_;
    Eigen::MatrixXf HZ = Eigen::MatrixXf::Zero(N, nc);
    std::vector<int> leaf_of(np_, 0);
    int nleaf = 0;
    for (const ClusterTree::Node& P : pt_.nodes)
      if (P.leaf()) {
        for (int i = P.lo; i < P.hi; ++i) leaf_of[i] = nleaf;
        ++nleaf;
      }
    std::vector<omp_lock_t> lk(nleaf);
    for (auto& l : lk) omp_init_lock(&l);
    const int nb = int(near_.size() + far_.size());
#pragma omp parallel
    {
      std::vector<int> mark(nc, -1), ids;
      Eigen::MatrixXf Zl, Zg, Y;
      // Z restricted to the columns cols (dof or traction dof, 3 components each), local coarse ids
      auto restrict = [&](const SparseBasis& Z, const std::vector<int>& cols, Eigen::MatrixXf& Out) {
        const int n3 = 3 * int(cols.size());
        Out.setZero(n3, int(ids.size()));
        for (int c = 0; c < n3; ++c) {
          const int g = 3 * cols[c / 3] + c % 3;
          for (int p = Z.ptr[g]; p < Z.ptr[g + 1]; ++p) Out(c, mark[Z.id[p]]) += float(Z.val[p]);
        }
      };
      auto collect = [&](const SparseBasis& Z, const std::vector<int>& cols) {
        for (int c = 0; c < 3 * int(cols.size()); ++c) {
          const int g = 3 * cols[c / 3] + c % 3;
          for (int p = Z.ptr[g]; p < Z.ptr[g + 1]; ++p)
            if (mark[Z.id[p]] < 0) { mark[Z.id[p]] = int(ids.size()); ids.push_back(Z.id[p]); }
        }
      };
#pragma omp for schedule(dynamic, 4)
      for (int q = 0; q < nb; ++q) {
        const bool nearb = q < int(near_.size());
        const NearLeaf* L = nearb ? &near_[q] : nullptr;
        const FarBlock* F = nearb ? nullptr : &far_[q - near_.size()];
        const ClusterTree::Node& P = pt_.nodes[nearb ? L->node : F->pnode];
        const std::vector<int>& cols = nearb ? L->cols : F->cols;
        const std::vector<int>& tcols = nearb ? L->tcols : F->tcols;
        if (!nearb && F->rank == 0) continue;
        const int M = 3 * P.size(), n3 = 3 * int(cols.size()), t3 = Zt ? 3 * int(tcols.size()) : 0;
        ids.clear();
        collect(Zu, cols);
        if (t3) collect(*Zt, tcols);
        const int nl = int(ids.size());
        if (!nl) continue;
        restrict(Zu, cols, Zl);
        if (t3) restrict(*Zt, tcols, Zg);
        for (int k : ids) mark[k] = -1;
        using MapF = Eigen::Map<const Eigen::MatrixXf>;
        if (nearb) {
          Y.noalias() = MapF(L->H.data(), M, n3) * Zl;
          if (t3) Y.noalias() -= MapF(L->G.data(), M, t3) * Zg;
        } else if (F->rank < 0) {
          Y.noalias() = MapF(F->D.data(), M, n3) * Zl;
          if (t3) Y.noalias() -= MapF(F->DG.data(), M, t3) * Zg;
        } else {
          Eigen::MatrixXf T = MapF(F->W.data(), F->rank, n3) * Zl;
          if (t3) T.noalias() -= MapF(F->WG.data(), F->rank, t3) * Zg;
          Y.noalias() = MapF(F->U.data(), M, F->rank) * T;
        }
        int cur = -1;
        for (int k = 0; k < P.size(); ++k) {
          const int lf = leaf_of[P.lo + k];
          if (lf != cur) {
            if (cur >= 0) omp_unset_lock(&lk[cur]);
            omp_set_lock(&lk[lf]);
            cur = lf;
          }
          const int a = pt_.perm[P.lo + k];
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < nl; ++j) HZ(3 * a + i, ids[j]) += Y(3 * k + i, j);
        }
        if (cur >= 0) omp_unset_lock(&lk[cur]);
      }
    }
    for (auto& l : lk) omp_destroy_lock(&l);
    return HZ;
  }

  // Exact entries of H (rigid term included) for the pairs (point a, dof B) with
  // |x_a - P_B| < radius h_B (and every B with N_B(x_a) != 0): the preconditioner,
  // independent of the admissibility. h_B: largest element diameter in B's support.
  void exactTriplets(double radius, std::vector<Eigen::Triplet<double>>& out) const {
    std::vector<double> hB(mesh_.ndof, 0.0);
    double hmax = 0.0;
    for (int ie = 0; ie < ne_; ++ie) {
      const double d = 2.0 * eb_.quad().elementRadius(ie);
      hmax = std::max(hmax, d);
      for (int B : edof_[ie]) hB[B] = std::max(hB[B], d);
    }
    std::vector<std::vector<Eigen::Triplet<double>>> parts;
#pragma omp parallel
    {
#pragma omp single
#ifdef _OPENMP
      parts.resize(omp_get_num_threads());
#else
      parts.resize(1);
#endif
#ifdef _OPENMP
      auto& o = parts[omp_get_thread_num()];
#else
      auto& o = parts[0];
#endif
      ElementBlocks::Work w;
      std::vector<int> elems, stack;
      std::vector<double> Hb, row;
      std::vector<int> cols;
#pragma omp for schedule(dynamic, 64)
      for (int a = 0; a < np_; ++a) {
        const Vec3& x = rows_[a].x;
        // elements within radius hmax + their radius (tree descent)
        elems.clear();
        stack.assign(1, 0);
        const double R = radius * hmax;
        while (!stack.empty()) {
          const ClusterTree::Node& E = et_.nodes[stack.back()];
          stack.pop_back();
          double d2 = 0.0;
          for (int k = 0; k < 3; ++k) {
            const double g = std::max({0.0, E.bmin(k) - x(k), x(k) - E.bmax(k)});
            d2 += g * g;
          }
          if (d2 > R * R) continue;
          if (E.leaf()) {
            for (int i = E.lo; i < E.hi; ++i) {
              const int ie = et_.perm[i];
              if ((eb_.quad().elementCentre(ie) - x).norm() - eb_.quad().elementRadius(ie) < R) elems.push_back(ie);
            }
          } else {
            stack.push_back(E.kid[0]);
            stack.push_back(E.kid[1]);
          }
        }
        // accumulate exact rows over the support elements of the kept dofs
        std::vector<int> keep;
        for (int ie : elems)
          for (int B : edof_[ie])
            if ((x - mesh_.dof_point[B]).norm() < radius * hB[B]) keep.push_back(B);
        for (const auto& [B, NB] : rows_[a].basis) keep.push_back(B);
        cols = unique(keep);
        const int nc = int(cols.size());
        row.assign(9 * size_t(nc), 0.0);  // 3 x 3nc
        for (int ie : elems) {
          const int nl = int(edof_[ie].size());
          bool any = false;
          for (int B : edof_[ie]) any |= std::binary_search(cols.begin(), cols.end(), B);
          if (!any) continue;
          Hb.resize(9 * nl);
          eb_.row(a, ie, Hb.data(), nullptr, w);
          for (int l = 0; l < nl; ++l) {
            const auto it = std::lower_bound(cols.begin(), cols.end(), edof_[ie][l]);
            if (it == cols.end() || *it != edof_[ie][l]) continue;
            const int c = int(it - cols.begin());
            for (int k = 0; k < 9; ++k) row[9 * c + k] += Hb[9 * l + k];
          }
        }
        for (const auto& [B, NB] : rows_[a].basis) {
          const int c = int(std::lower_bound(cols.begin(), cols.end(), B) - cols.begin());
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) row[9 * c + 3 * j + i] -= NB * S_(3 * a + i, j);
        }
        for (int c = 0; c < nc; ++c)
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) o.emplace_back(3 * a + i, 3 * cols[c] + j, row[9 * c + 3 * j + i]);
      }
    }
    for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
  }

  // Preconditioner entries: exact H(a, B) for |x_a - P_B| < radius h_B, from the near rows
  // completed by the support elements of B outside the leaf's near set (cheap far rules), so
  // no pattern entry is a partial sum (B's support may straddle a low-rank block). gout: the
  // same for G(a, J) on the kept traction dofs J (rows 3 a + i, columns 3 J + j).
  void preconTriplets(double radius, std::vector<Eigen::Triplet<double>>& out,
                      std::vector<Eigen::Triplet<double>>* gout = nullptr) const {
    std::vector<double> hB(mesh_.ndof, 0.0), hJ(ts_.ndof, 0.0);
    std::vector<std::vector<int>> supp(mesh_.ndof), tsupp(ts_.ndof);
    for (int ie = 0; ie < ne_; ++ie) {
      const double d = 2.0 * eb_.quad().elementRadius(ie);
      for (int l = 0; l < int(edof_[ie].size()); ++l) {
        hB[edof_[ie][l]] = std::max(hB[edof_[ie][l]], d);
        supp[edof_[ie][l]].push_back(ie);
        if (gk_[etdof_[ie][l]]) {
          hJ[etdof_[ie][l]] = std::max(hJ[etdof_[ie][l]], d);
          tsupp[etdof_[ie][l]].push_back(ie);
        }
      }
    }
    std::vector<std::vector<Eigen::Triplet<double>>> parts(near_.size()), gparts(gout ? near_.size() : 0);
#pragma omp parallel
    {
      ElementBlocks::Work w;
      std::vector<double> Hb, Gb;
      std::vector<char> inS(ne_, 0);
#pragma omp for schedule(dynamic, 1)
      for (int q = 0; q < int(near_.size()); ++q) {
        const NearLeaf& L = near_[q];
        const ClusterTree::Node& P = pt_.nodes[L.node];
        const int m = P.size(), M = 3 * m, nc = int(L.cols.size());
        for (int ie : L.elems) inS[ie] = 1;
        for (int c = 0; c < nc; ++c) {
          const int B = L.cols[c];
          for (int k = 0; k < m; ++k) {
            const int a = pt_.perm[P.lo + k];
            if ((rows_[a].x - mesh_.dof_point[B]).norm() >= radius * hB[B]) continue;
            Eigen::Matrix3d v;
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) v(i, j) = L.H[(3 * k + i) + size_t(M) * (3 * c + j)];
            for (int ie : supp[B]) {
              if (inS[ie]) continue;
              const int nl = int(edof_[ie].size());
              Hb.resize(9 * nl);
              eb_.row(a, ie, Hb.data(), nullptr, w);
              const int l = int(std::find(edof_[ie].begin(), edof_[ie].end(), B) - edof_[ie].begin());
              for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) v(i, j) += Hb[3 * (3 * l + j) + i];
            }
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) parts[q].emplace_back(3 * a + i, 3 * B + j, v(i, j));
          }
        }
        for (int c = 0; gout && c < int(L.tcols.size()); ++c) {
          const int J = L.tcols[c];
          for (int k = 0; k < m; ++k) {
            const int a = pt_.perm[P.lo + k];
            if ((rows_[a].x - ts_.points[J].x).norm() >= radius * hJ[J]) continue;
            Eigen::Matrix3d v;
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) v(i, j) = L.G[(3 * k + i) + size_t(M) * (3 * c + j)];
            for (int ie : tsupp[J]) {
              if (inS[ie]) continue;
              const int nl = int(etdof_[ie].size());
              Hb.resize(9 * nl);
              Gb.resize(9 * nl);
              eb_.row(a, ie, Hb.data(), Gb.data(), w);
              const int l = int(std::find(etdof_[ie].begin(), etdof_[ie].end(), J) - etdof_[ie].begin());
              for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i) v(i, j) += Gb[3 * (3 * l + j) + i];
            }
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) gparts[q].emplace_back(3 * a + i, 3 * J + j, v(i, j));
          }
        }
        for (int ie : L.elems) inS[ie] = 0;
      }
    }
    for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    if (gout)
      for (auto& p : gparts) gout->insert(gout->end(), p.begin(), p.end());
  }

  // Near-field entries as triplets (row, col, value) of H (rigid term included),
  // for pairs (point a, dof B) accepted by keep(a, B): the preconditioner pattern.
  template <class F>
  void nearTriplets(F&& keep, std::vector<Eigen::Triplet<double>>& out) const {
    for (const NearLeaf& L : near_) {
      const ClusterTree::Node& P = pt_.nodes[L.node];
      const int m = P.size(), nc = int(L.cols.size());
      for (int c = 0; c < nc; ++c)
        for (int k = 0; k < m; ++k) {
          const int a = pt_.perm[P.lo + k];
          if (!keep(a, L.cols[c])) continue;
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
              out.emplace_back(3 * a + i, 3 * L.cols[c] + j, double(L.H[(3 * k + i) + size_t(3 * m) * (3 * c + j)]));
        }
    }
  }

 private:
  struct NearLeaf {
    int node;
    std::vector<int> elems;  // elements of the inadmissible element leaves
    std::vector<int> cols;   // global dofs (sorted)
    std::vector<float> H;    // 3 m x 3 nc, column-major
    std::vector<int> tcols;  // kept traction dofs (sorted)
    std::vector<float> G;    // 3 m x 3 nt
  };
  struct FarBlock {
    int pnode, enode, rank = 0;
    std::vector<int> cols;  // global dofs (sorted)
    std::vector<float> U;   // 3 m x r
    std::vector<float> W;   // r x 3 nc
    std::vector<float> D;   // dense fallback 3 m x 3 nc (rank = -1)
    std::vector<int> tcols;  // kept traction dofs (sorted)
    std::vector<float> WG;   // r x 3 nt
    std::vector<float> DG;   // dense fallback 3 m x 3 nt
  };

  const Mesh& mesh_;
  const TractionSpace& ts_;
  const std::vector<Collocation>& rows_;
  std::vector<char> gk_;  // per traction dof: G column stored
  bool any_g_ = false;
  ElemHOptions o_;
  ElementBlocks eb_;
  double mu_;
  int np_, ne_, K_;
  ClusterTree pt_, et_;
  std::vector<std::vector<int>> edof_, etdof_;
  std::vector<std::pair<int, int>> far_pairs_;
  std::vector<std::vector<int>> near_of_leaf_;  // per point node: inadmissible element leaves
  std::vector<NearLeaf> near_;
  std::vector<FarBlock> far_;
  Eigen::MatrixXd GC_, S_;  // S_: rigid-body row sums of Hraw
  std::vector<double> dres_;  // float rounding residual of the folded rigid term, 9 per (row a, basis B)
  std::vector<size_t> doff_;  // per row: offset into dres_
  Stats st_;

  using clock = std::chrono::steady_clock;
  static clock::time_point now() { return clock::now(); }
  static double seconds(clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); }

  void partition(int p, int e) {
    const ClusterTree::Node &P = pt_.nodes[p], &E = et_.nodes[e];
    const double d = ClusterTree::distance(P, E);
    const double dm = o_.adm_min ? std::min(P.diam(), E.diam()) : std::max(P.diam(), E.diam());
    if (d > 0.0 && dm <= o_.eta * d) {
      // A block the compression would store dense (too few points to sample) joins the near
      // row block of its point leaf instead: assembled there, its control point columns are not
      // duplicated across element clusters (a p = 4 dof touches 25 elements).
      if (o_.merge_small && P.leaf() && 2 * std::min(o_.s0, P.size()) >= P.size()) {
        if (near_of_leaf_.size() < pt_.nodes.size()) near_of_leaf_.resize(pt_.nodes.size());
        near_of_leaf_[p].push_back(e);
      } else {
        far_pairs_.emplace_back(p, e);
      }
      return;
    }
    if (P.leaf() && E.leaf()) {
      if (near_of_leaf_.size() < pt_.nodes.size()) near_of_leaf_.resize(pt_.nodes.size());
      near_of_leaf_[p].push_back(e);
      return;
    }
    const bool splitP = !P.leaf() && (E.leaf() || P.diam() >= E.diam());
    if (splitP) {
      partition(P.kid[0], e);
      partition(P.kid[1], e);
    } else {
      partition(p, E.kid[0]);
      partition(p, E.kid[1]);
    }
  }

  static std::vector<int> unique(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  }

  void buildNear(const Eigen::MatrixXd& C) {
    near_of_leaf_.resize(pt_.nodes.size());
    for (int p = 0; p < int(pt_.nodes.size()); ++p)
      if (!near_of_leaf_[p].empty()) {
        NearLeaf L;
        L.node = p;
        for (int e : near_of_leaf_[p]) {
          const ClusterTree::Node& E = et_.nodes[e];
          for (int i = E.lo; i < E.hi; ++i) L.elems.push_back(et_.perm[i]);
        }
        near_.push_back(std::move(L));
      }
    long pairs = 0;
#pragma omp parallel reduction(+ : pairs)
    {
      ElementBlocks::Work w;
      std::vector<double> Hb, Gb, Gc;
#pragma omp for schedule(dynamic, 1)
      for (int q = 0; q < int(near_.size()); ++q) {
        NearLeaf& L = near_[q];
        const ClusterTree::Node& P = pt_.nodes[L.node];
        const int m = P.size();
        const int* pts = pt_.perm.data() + P.lo;
        std::vector<int> all;
        for (int ie : L.elems) all.insert(all.end(), edof_[ie].begin(), edof_[ie].end());
        L.cols = unique(all);
        const int nc = int(L.cols.size());
        std::vector<int> allt;
        for (int ie : L.elems)
          for (int J : etdof_[ie])
            if (gk_[J]) allt.push_back(J);
        L.tcols = unique(allt);
        const int nt = int(L.tcols.size());
        std::vector<double> H(size_t(3 * m) * 3 * nc, 0.0), Gk(size_t(3 * m) * 3 * nt, 0.0);
        Eigen::MatrixXd gcl = Eigen::MatrixXd::Zero(3 * m, K_);
        for (int ie : L.elems) {
          const int nl = int(edof_[ie].size());
          Hb.resize(size_t(3 * m) * 3 * nl);
          Gb.resize(Hb.size());
          eb_.block(pts, m, ie, Hb.data(), K_ || nt ? Gb.data() : nullptr, 3 * m, w);
          pairs += m;
          for (int l = 0; l < nl; ++l) {
            const int c = int(std::lower_bound(L.cols.begin(), L.cols.end(), edof_[ie][l]) - L.cols.begin());
            for (int j = 0; j < 3; ++j) {
              const double* s = Hb.data() + size_t(3 * m) * (3 * l + j);
              double* d = H.data() + size_t(3 * m) * (3 * c + j);
              for (int r = 0; r < 3 * m; ++r) d[r] += s[r];
            }
            const int J = etdof_[ie][l];
            if (!gk_[J]) continue;
            const int ct = int(std::lower_bound(L.tcols.begin(), L.tcols.end(), J) - L.tcols.begin());
            for (int j = 0; j < 3; ++j) {
              const double* s = Gb.data() + size_t(3 * m) * (3 * l + j);
              double* d = Gk.data() + size_t(3 * m) * (3 * ct + j);
              for (int r = 0; r < 3 * m; ++r) d[r] += s[r];
            }
          }
          if (K_) {
            // G block times the traction coefficients of the element's local functions.
            Eigen::MatrixXd Cl(3 * nl, K_);
            for (int l = 0; l < nl; ++l) Cl.middleRows(3 * l, 3) = C.middleRows(3 * etdof_[ie][l], 3);
            gcl.noalias() += Eigen::Map<const Eigen::MatrixXd>(Gb.data(), 3 * m, 3 * nl) * Cl;
          }
        }
        L.H.assign(H.begin(), H.end());
        L.G.assign(Gk.begin(), Gk.end());
        for (int k = 0; k < m; ++k) GC_.middleRows(3 * pts[k], 3) += gcl.middleRows(3 * k, 3);
      }
    }
    st_.near_leaves = int(near_.size());
    st_.near_pairs = pairs;
    for (const NearLeaf& L : near_)
      st_.bytes_near += 4.0 * (L.H.size() + L.cols.size() + L.G.size() + L.tcols.size());
  }

  // ---------------------------------------------------------------- far blocks
  struct AcaWork {
    ElementBlocks::Work w;
    std::vector<double> Hb, Gb;
    double t_row = 0, t_col = 0, t_aca = 0, t_rc = 0, t_cw = 0, t_qr = 0, t_rr = 0, t_test = 0;
    int attempts = 0, n_fail = 0;  // n_fail: blocks where CUR ran and did not converge
    long n_rows = 0, n_cols = 0;
    double fail_entries = 0.0;
  };
  static double peakGB() {
    rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / 1048576.0;
  }
  static double wt() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return 0.0;
#endif
  }

  // Row block (3 x ncol) of the joint unassembled block for point a over the elements E.
  void pointRow(int a, const std::vector<int>& elems, const std::vector<int>& eoff, double gamma, double* out,
                AcaWork& aw) const {
    for (size_t t = 0; t < elems.size(); ++t) {
      const int ie = elems[t], nl = int(edof_[ie].size());
      aw.Hb.resize(9 * nl);
      aw.Gb.resize(9 * nl);
      eb_.row(a, ie, aw.Hb.data(), aw.Gb.data(), aw.w);
      // out: 3 x ncol column-major (ld 3); columns eoff[t] + c (H: c < 3 nl, G: 3 nl + c)
      for (int c = 0; c < 3 * nl; ++c)
        for (int i = 0; i < 3; ++i) {
          out[3 * (eoff[t] + c) + i] = aw.Hb[3 * c + i];
          out[3 * (eoff[t] + 3 * nl + c) + i] = gamma * aw.Gb[3 * c + i];
        }
    }
  }
  // Assembled row block (3 x ncolA, zeroed here) for point a: unassembled columns c go to amap[c].
  void pointRowA(int a, const std::vector<int>& elems, const std::vector<int>& eoff, double gamma,
                 const std::vector<int>& amap, int ncolA, double* out, AcaWork& aw) const {
    std::fill(out, out + 3 * size_t(ncolA), 0.0);
    for (size_t t = 0; t < elems.size(); ++t) {
      const int ie = elems[t], nl = int(edof_[ie].size());
      aw.Hb.resize(9 * nl);
      aw.Gb.resize(9 * nl);
      eb_.row(a, ie, aw.Hb.data(), aw.Gb.data(), aw.w);
      for (int c = 0; c < 3 * nl; ++c) {
        double* h = out + 3 * size_t(amap[eoff[t] + c]);
        double* g = out + 3 * size_t(amap[eoff[t] + 3 * nl + c]);
        for (int i = 0; i < 3; ++i) {
          h[i] += aw.Hb[3 * c + i];
          g[i] += gamma * aw.Gb[3 * c + i];
        }
      }
    }
  }
  // Column block (3 m x 6 nl) of element ie for the points pts.
  void elemCols(const int* pts, int m, int ie, double gamma, double* out, AcaWork& aw) const {
    const int nl = int(edof_[ie].size());
    eb_.block(pts, m, ie, out, out + size_t(3 * m) * 3 * nl, 3 * m, aw.w);
    double* g = out + size_t(3 * m) * 3 * nl;
    for (size_t i = 0; i < size_t(3 * m) * 3 * nl; ++i) g[i] *= gamma;
  }


  // Farthest-point order of a set of positions (start: the one nearest the centroid).
  static std::vector<int> fpsOrder(const std::vector<Vec3>& x) {
    const int n = int(x.size());
    std::vector<int> ord;
    if (!n) return ord;
    Vec3 c = Vec3::Zero();
    for (const Vec3& p : x) c += p;
    c /= n;
    int s = 0;
    for (int i = 1; i < n; ++i)
      if ((x[i] - c).squaredNorm() < (x[s] - c).squaredNorm()) s = i;
    std::vector<double> d(n, 1e300);
    std::vector<char> used(n, 0);
    for (int k = 0; k < n; ++k) {
      ord.push_back(s);
      used[s] = 1;
      int nx = -1;
      double best = -1.0;
      for (int i = 0; i < n; ++i) {
        if (used[i]) continue;
        d[i] = std::min(d[i], (x[i] - x[s]).squaredNorm());
        if (d[i] > best) { best = d[i]; nx = i; }
      }
      if (nx < 0) break;
      s = nx;
    }
    return ord;
  }

  // Sampled cross approximation (CUR) of the joint unassembled block, as bemkit's cur_block:
  // s point rows (all three components) and s elements (all their columns) in farthest-point
  // order, A ~ C W^+ R with W^+ from a truncated SVD; accepted when random test point rows and
  // test elements agree to eps / 2; otherwise the worst test points join and s doubles.
  // U (M x q), V (q x ncol); false if the samples cover the block (dense is cheaper).
  // amap: unassembled column -> assembled column (of ncolA); V is returned assembled.
  bool cur(const int* pts, int m, const std::vector<int>& elems, const std::vector<int>& eoff, int ncol, double gamma,
           const std::vector<int>& amap, int ncolA, Eigen::MatrixXd& U, Eigen::MatrixXd& V, AcaWork& aw,
           std::mt19937& rng) const {
    auto assembleRows = [&](const double* src, int nr, Eigen::MatrixXd& out) {  // src: nr x ncol col-major
      out = Eigen::MatrixXd::Zero(nr, ncolA);
      for (int c = 0; c < ncol; ++c)
        for (int i = 0; i < nr; ++i) out(i, amap[c]) += src[size_t(nr) * c + i];
    };
    const int M = 3 * m, ne = int(elems.size());
    std::vector<Vec3> px(m), ex(ne);
    for (int k = 0; k < m; ++k) px[k] = rows_[pts[k]].x;
    for (int t = 0; t < ne; ++t) ex[t] = eb_.quad().elementCentre(elems[t]);
    std::vector<int> pord = fpsOrder(px), eord = fpsOrder(ex);
    std::vector<int> prow(m, -1), ecol(ne, -1);
    std::vector<std::vector<double>> prows;
    std::vector<std::vector<float>> ecols;  // sampled element columns, float (M can be thousands)
    std::vector<double> cbuf;
    auto rowOf = [&](int k) -> const double* {
      if (prow[k] < 0) {
        prow[k] = int(prows.size());
        prows.emplace_back(size_t(3) * ncolA);
        const double t0 = wt();
        pointRowA(pts[k], elems, eoff, gamma, amap, ncolA, prows.back().data(), aw);
        aw.t_row += wt() - t0;
        ++aw.n_rows;
      }
      return prows[prow[k]].data();
    };
    auto evalCols = [&](int t, double* out) {
      const double t0 = wt();
      elemCols(pts, m, elems[t], gamma, out, aw);
      aw.t_col += wt() - t0;
      ++aw.n_cols;
    };
    auto colOf = [&](int t) -> const float* {
      if (ecol[t] < 0) {
        ecol[t] = int(ecols.size());
        cbuf.resize(size_t(M) * (eoff[t + 1] - eoff[t]));
        evalCols(t, cbuf.data());
        ecols.emplace_back(cbuf.begin(), cbuf.end());
      }
      return ecols[ecol[t]].data();
    };
    std::vector<int> sp, se;  // sampled points (positions in pts), elements (positions in elems)
    int s = std::min(o_.s0, m), sE = std::min(o_.s0e, ne);
    if (2 * s >= m) return false;  // small block: dense
    std::uniform_int_distribution<int> pickP(0, m - 1), pickE(0, ne - 1);
    std::vector<int> extra;  // worst test points, added first
    while (true) {
      // samples: extra points first, then the farthest-point order
      sp.clear();
      std::vector<char> inP(m, 0), inE(ne, 0);
      for (int k : extra)
        if (!inP[k] && int(sp.size()) < s) { sp.push_back(k); inP[k] = 1; }
      for (int k : pord)
        if (int(sp.size()) < s && !inP[k]) { sp.push_back(k); inP[k] = 1; }
      se.assign(eord.begin(), eord.begin() + sE);
      for (int t : se) inE[t] = 1;
      if (4 * int(sp.size()) > 3 * m || sE >= ne) return false;  // samples nearly the whole block
      ++aw.attempts;
      int nC = 0;
      for (int t : se) nC += eoff[t + 1] - eoff[t];
      double tl = wt();
      // W = A(sampled point rows, sampled element columns), from the caches (no C or R copies)
      std::vector<int> coff(se.size() + 1, 0);
      for (size_t u = 0; u < se.size(); ++u) coff[u + 1] = coff[u] + eoff[se[u] + 1] - eoff[se[u]];
      const int ns = int(sp.size());
      std::vector<const double*> rp(ns);
      for (int i = 0; i < ns; ++i) rp[i] = rowOf(sp[i]);
      Eigen::MatrixXd W(3 * ns, nC);
      for (size_t u = 0; u < se.size(); ++u) {
        const float* cb = colOf(se[u]);
        for (int c = coff[u]; c < coff[u + 1]; ++c)
          for (int i = 0; i < ns; ++i)
            for (int r = 0; r < 3; ++r) W(3 * i + r, c) = cb[size_t(M) * (c - coff[u]) + 3 * sp[i] + r];
      }
      // column skeleton J of W (pivoted QR), then A ~ C(:, J) W(:, J)^+ R
      aw.t_cw += wt() - tl;
      tl = wt();
      // truncated column-pivoted Gram-Schmidt (stops at the tolerance: cost rows x cols x rank)
      std::vector<int> piv;
      {
        Eigen::VectorXd cn = W.colwise().squaredNorm().transpose();
        const double c0 = cn.maxCoeff();
        Eigen::MatrixXd Q(W.rows(), std::min<Eigen::Index>(W.rows(), W.cols()));
        const double tol2 = 1e-6 * o_.eps * o_.eps * c0;
        for (int k = 0; k < Q.cols();) {
          Eigen::Index c;
          const double mx = cn.maxCoeff(&c);
          if (mx <= tol2) break;
          Eigen::VectorXd v = W.col(c);
          for (int pass = 0; pass < 2; ++pass) v -= Q.leftCols(k) * (Q.leftCols(k).transpose() * v);
          const double nv = v.norm();
          if (nv * nv <= tol2) { cn(c) = 0.0; continue; }
          Q.col(k) = v / nv;
          piv.push_back(int(c));
          cn -= (Q.col(k).transpose() * W).transpose().cwiseAbs2();
          cn(c) = 0.0;
          ++k;
        }
      }
      const int q = int(piv.size());
      if (q == 0) { U.resize(M, 0); V.resize(0, ncolA); return true; }
      struct { const std::vector<int>& v; int operator()(int i) const { return v[i]; } } perm{piv};
      Eigen::MatrixXd L(M, q), WJ(W.rows(), q);
      for (int c = 0; c < q; ++c) {
        const int pc = perm(c);
        const int u = int(std::upper_bound(coff.begin(), coff.end(), pc) - coff.begin()) - 1;
        L.col(c) = Eigen::Map<const Eigen::VectorXf>(colOf(se[u]) + size_t(M) * (pc - coff[u]), M).cast<double>();
        WJ.col(c) = W.col(pc);
      }
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(WJ, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const Eigen::VectorXd& sg = svd.singularValues();
      int qq = 0;
      while (qq < sg.size() && sg(qq) > 1e-3 * o_.eps * sg(0)) ++qq;
      const Eigen::MatrixXd Pinv =
          (svd.matrixV().leftCols(qq) * sg.head(qq).cwiseInverse().asDiagonal()) * svd.matrixU().leftCols(qq).transpose();
      aw.t_qr += wt() - tl;
      tl = wt();
      Eigen::MatrixXd Rr = Eigen::MatrixXd::Zero(Pinv.rows(), ncolA);  // Pinv R, R from the row cache (no copy)
      for (int i = 0; i < ns; ++i) Rr.noalias() += Pinv.middleCols(3 * i, 3) * Eigen::Map<const Eigen::MatrixXd>(rp[i], 3, ncolA);
      aw.t_rr += wt() - tl;
      tl = wt();
      // tests: random point rows and random elements outside the samples, Frobenius error relative
      // to the tested entries (the block criterion of the recompression; a per-row relative test
      // rejects blocks whose rows nearly cancel and sends them to the exact path)
      double worst = 0.0, er2 = 0.0, nr2 = 0.0, ec2 = 0.0, nc2 = 0.0;
      std::vector<std::pair<double, int>> bad;
      for (int c = 0, tries = 0; c < o_.checks && tries < 4 * m; ++tries) {
        const int k = pickP(rng);
        if (inP[k]) continue;
        inP[k] = 1;
        ++c;
        const Eigen::Map<const Eigen::MatrixXd> ex3(rowOf(k), 3, ncolA);
        const double e2 = (ex3 - L.middleRows(3 * k, 3) * Rr).squaredNorm();
        er2 += e2;
        nr2 += ex3.squaredNorm();
        bad.emplace_back(e2, k);
      }
      for (int c = 0, tries = 0; c < o_.checks / 2 && tries < 4 * ne; ++tries) {
        const int t = pickE(rng);
        if (inE[t]) continue;
        inE[t] = 1;
        ++c;
        const int w = eoff[t + 1] - eoff[t];
        cbuf.resize(size_t(M) * w);  // test columns: evaluated, not cached
        evalCols(t, cbuf.data());
        const Eigen::Map<const Eigen::MatrixXd> exc(cbuf.data(), M, w);
        Eigen::MatrixXd Rs(3 * ns, w);  // sampled rows of the element's columns, from its column block
        const double* cb = cbuf.data();
        for (int i = 0; i < ns; ++i)
          for (int c = 0; c < w; ++c)
            for (int r = 0; r < 3; ++r) Rs(3 * i + r, c) = cb[size_t(M) * c + 3 * sp[i] + r];
        ec2 += (exc - L * (Pinv * Rs)).squaredNorm();
        nc2 += exc.squaredNorm();
      }
      worst = std::max(std::sqrt(er2 / std::max(nr2, 1e-300)), std::sqrt(ec2 / std::max(nc2, 1e-300)));
      aw.t_test += wt() - tl;
      if (worst <= 0.5 * o_.eps) {
        U = std::move(L);
        V = std::move(Rr);
        return true;
      }
      std::sort(bad.rbegin(), bad.rend());
      for (auto& [e, k] : bad)
        if (e > 0.25 * o_.eps * o_.eps * nr2 / std::max<size_t>(bad.size(), 1)) extra.push_back(k);
      s = std::min((3 * s) / 2, m);  // gently: a leaf-sized block (m ~ 2 s0) stays sampled
      sE = std::min((3 * sE + 1) / 2 + 1, ne);
    }
  }

  // Partially pivoted ACA of the joint block; returns false if the rank exceeds max_rank.
  bool aca(const int* pts, int m, const std::vector<int>& elems, const std::vector<int>& eoff, int ncol, double gamma,
           Eigen::MatrixXd& U, Eigen::MatrixXd& V, AcaWork& aw, std::mt19937& rng) const {
    const int M = 3 * m;
    std::vector<Eigen::VectorXd> us, vs;
    std::vector<char> row_used(M, 0), col_used(ncol, 0);
    // caches: point rows (3 x ncol) and element columns (3m x 6nl)
    std::vector<int> prow(m, -1), ecol(elems.size(), -1);
    std::vector<std::vector<double>> prows, ecols;
    std::vector<int> col_elem(ncol);
    for (size_t t = 0; t < elems.size(); ++t)
      for (int c = eoff[t]; c < eoff[t + 1]; ++c) col_elem[c] = int(t);
    auto getRow = [&](int r, Eigen::VectorXd& row) {
      const int k = r / 3;
      if (prow[k] < 0) {
        prow[k] = int(prows.size());
        prows.emplace_back(size_t(3) * ncol);
        const double t0 = wt();
        pointRow(pts[k], elems, eoff, gamma, prows.back().data(), aw);
        aw.t_row += wt() - t0;
        ++aw.n_rows;
      }
      const double* s = prows[prow[k]].data();
      row.resize(ncol);
      for (int c = 0; c < ncol; ++c) row(c) = s[3 * c + r % 3];
      for (size_t t = 0; t < us.size(); ++t) row -= us[t](r) * vs[t];
    };
    auto getCol = [&](int c, Eigen::VectorXd& col) {
      const int t = col_elem[c];
      if (ecol[t] < 0) {
        ecol[t] = int(ecols.size());
        const int nl = int(edof_[elems[t]].size());
        ecols.emplace_back(size_t(M) * 6 * nl);
        const double t0 = wt();
        elemCols(pts, m, elems[t], gamma, ecols.back().data(), aw);
        aw.t_col += wt() - t0;
        ++aw.n_cols;
      }
      col = Eigen::Map<const Eigen::VectorXd>(ecols[ecol[t]].data() + size_t(M) * (c - eoff[t]), M);
      for (size_t s = 0; s < us.size(); ++s) col -= vs[s](c) * us[s];
    };
    double S2 = 0.0;
    int r = 0;
    Eigen::VectorXd row, col;
    auto step = [&](int i) -> int {  // one cross from row i; returns next row or -1 on convergence; -2 zero row
      row_used[i] = 1;
      getRow(i, row);
      int j = -1;
      double best = 0.0;
      for (int c = 0; c < ncol; ++c)
        if (!col_used[c] && std::abs(row(c)) > best) { best = std::abs(row(c)); j = c; }
      if (j < 0 || best == 0.0) return -2;
      col_used[j] = 1;
      Eigen::VectorXd v = row / row(j);
      getCol(j, col);
      double cross = 0.0;
      for (size_t t = 0; t < us.size(); ++t) cross += col.dot(us[t]) * vs[t].dot(v);
      const double nu2 = col.squaredNorm() * v.squaredNorm();
      S2 += nu2 + 2.0 * cross;
      us.push_back(col);
      vs.push_back(v);
      ++r;
      if (std::sqrt(nu2) <= o_.eps * std::sqrt(std::max(S2, 1e-300))) return -1;
      int nx = -1;
      double bu = -1.0;
      for (int q = 0; q < M; ++q)
        if (!row_used[q] && std::abs(col(q)) > bu) { bu = std::abs(col(q)); nx = q; }
      return nx;
    };
    std::uniform_int_distribution<int> pick(0, M - 1);
    int i = pick(rng);
    int zero_rows = 0;
    for (int round = 0; round < 8; ++round) {
      while (i >= 0) {
        if (r >= o_.max_rank || r >= std::min(M, ncol)) return false;
        const int nx = step(i);
        if (nx == -2) {
          // zero residual row: try another unused row
          if (++zero_rows > 8) break;
          int q = pick(rng), tries = 0;
          while (row_used[q] && tries++ < 4 * M) q = (q + 1) % M;
          i = row_used[q] ? -1 : q;
          continue;
        }
        if (nx == -1) break;
        i = nx;
      }
      // a-posteriori check on random unused point rows (all three components)
      int worst = -1;
      double wres = 0.0;
      for (int c = 0; c < o_.checks; ++c) {
        int q = pick(rng);
        for (int tries = 0; row_used[q] && tries < M; ++tries) q = (q + 1) % M;
        if (row_used[q]) break;
        getRow(q, row);
        const double res = row.norm();
        // the exact row's norm from the cache
        const double* s = prows[prow[q / 3]].data();
        double nr = 0.0;
        for (int cc = 0; cc < ncol; ++cc) nr += s[3 * cc + q % 3] * s[3 * cc + q % 3];
        const double rel = res / std::max(std::sqrt(nr), 1e-300);
        if (rel > wres) { wres = rel; worst = q; }
      }
      if (worst < 0 || wres <= o_.eps) break;
      i = worst;
    }
    U.resize(M, r);
    V.resize(r, ncol);
    for (int t = 0; t < r; ++t) { U.col(t) = us[t]; V.row(t) = vs[t]; }
    return true;
  }

  void buildFar(const Eigen::MatrixXd& C) {
    far_.resize(far_pairs_.size());
    // larger blocks first (dynamic schedule)
    std::vector<int> order(far_.size());
    for (size_t b = 0; b < far_.size(); ++b) order[b] = int(b);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return double(pt_.nodes[far_pairs_[a].first].size()) * et_.nodes[far_pairs_[a].second].size() >
             double(pt_.nodes[far_pairs_[b].first].size()) * et_.nodes[far_pairs_[b].second].size();
    });
    std::atomic<int> fallbacks{0};
    std::vector<Eigen::MatrixXd> gc_t;
#pragma omp parallel
    {
#pragma omp single
      {
#ifdef _OPENMP
        gc_t.assign(omp_get_num_threads(), Eigen::MatrixXd::Zero(3 * np_, K_));
#else
        gc_t.assign(1, Eigen::MatrixXd::Zero(3 * np_, K_));
#endif
      }
#ifdef _OPENMP
      Eigen::MatrixXd& gct = gc_t[omp_get_thread_num()];
      std::mt19937 rng(1234 + omp_get_thread_num());
#else
      Eigen::MatrixXd& gct = gc_t[0];
      std::mt19937 rng(1234);
#endif
      AcaWork aw;
#pragma omp for schedule(dynamic, 1)
      for (int oi = 0; oi < int(order.size()); ++oi) {
        const int b = order[oi];
        FarBlock& F = far_[b];
        F.pnode = far_pairs_[b].first;
        F.enode = far_pairs_[b].second;
        const ClusterTree::Node &P = pt_.nodes[F.pnode], &E = et_.nodes[F.enode];
        const int m = P.size(), M = 3 * m;
        const int* pts = pt_.perm.data() + P.lo;
        std::vector<int> elems(et_.perm.begin() + E.lo, et_.perm.begin() + E.hi);
        std::vector<int> eoff(elems.size() + 1, 0);
        std::vector<int> all, allt;
        for (size_t t = 0; t < elems.size(); ++t) {
          eoff[t + 1] = eoff[t] + 6 * int(edof_[elems[t]].size());
          all.insert(all.end(), edof_[elems[t]].begin(), edof_[elems[t]].end());
          allt.insert(allt.end(), etdof_[elems[t]].begin(), etdof_[elems[t]].end());
        }
        F.cols = unique(all);
        const std::vector<int> tcols = unique(allt);
        const double gamma = 2.0 * mu_ / std::max((P.centre() - E.centre()).norm(), 1e-300);
        std::vector<int> kept;  // positions in tcols of the kept traction dofs
        for (int c = 0; c < int(tcols.size()); ++c)
          if (gk_[tcols[c]]) { kept.push_back(c); F.tcols.push_back(tcols[c]); }
        // kept G columns (3 per dof) of a matrix whose columns are the joint block's G part
        auto keptG = [&](const Eigen::MatrixXd& Gpart) {
          Eigen::MatrixXd o(Gpart.rows(), 3 * kept.size());
          for (size_t c = 0; c < kept.size(); ++c) o.middleCols(3 * c, 3) = Gpart.middleCols(3 * kept[c], 3) / gamma;
          return o;
        };
        const int ncol = eoff.back(), nc = int(F.cols.size()), nt = int(tcols.size());
        // assembly of unassembled columns -> [H global | G global]
        auto assemble = [&](const Eigen::MatrixXd& V, Eigen::MatrixXd& WH, Eigen::MatrixXd& WG) {
          WH = Eigen::MatrixXd::Zero(V.rows(), 3 * nc);
          WG = Eigen::MatrixXd::Zero(V.rows(), 3 * nt);
          for (size_t t = 0; t < elems.size(); ++t) {
            const int ie = elems[t], nl = int(edof_[ie].size());
            for (int l = 0; l < nl; ++l) {
              const int c = int(std::lower_bound(F.cols.begin(), F.cols.end(), edof_[ie][l]) - F.cols.begin());
              const int ct = int(std::lower_bound(tcols.begin(), tcols.end(), etdof_[ie][l]) - tcols.begin());
              for (int j = 0; j < 3; ++j) {
                WH.col(3 * c + j) += V.col(eoff[t] + 3 * l + j);
                WG.col(3 * ct + j) += V.col(eoff[t] + 3 * nl + 3 * l + j);
              }
            }
          }
        };
        Eigen::MatrixXd Ua, Va, WH, WG;
        std::vector<int> amap(ncol);
        for (size_t t = 0; t < elems.size(); ++t) {
          const int ie = elems[t], nl = int(edof_[ie].size());
          for (int l = 0; l < nl; ++l) {
            const int c = int(std::lower_bound(F.cols.begin(), F.cols.end(), edof_[ie][l]) - F.cols.begin());
            const int ct = int(std::lower_bound(tcols.begin(), tcols.end(), etdof_[ie][l]) - tcols.begin());
            for (int j = 0; j < 3; ++j) {
              amap[eoff[t] + 3 * l + j] = 3 * c + j;
              amap[eoff[t] + 3 * nl + 3 * l + j] = 3 * nc + 3 * ct + j;
            }
          }
        }
        Eigen::MatrixXd Cl(3 * nt, K_);
        for (int c = 0; c < nt; ++c) Cl.middleRows(3 * c, 3) = C.middleRows(3 * tcols[c], 3);
        const double ta = wt();
        const bool ok = o_.use_aca ? aca(pts, m, elems, eoff, ncol, gamma, Ua, Va, aw, rng)
                                  : cur(pts, m, elems, eoff, ncol, gamma, amap, 3 * (nc + nt), Ua, Va, aw, rng);
        aw.t_aca += wt() - ta;
        const double trc = wt();
        if (!ok) {
          // dense fallback
          ++fallbacks;
          if (!o_.use_aca && 2 * std::min(o_.s0, m) < m) { ++aw.n_fail; aw.fail_entries += double(M) * ncol; }
          Eigen::MatrixXd Full(M, ncol);
          for (size_t t = 0; t < elems.size(); ++t) {
            std::vector<double> buf(size_t(M) * (eoff[t + 1] - eoff[t]));
            elemCols(pts, m, elems[t], gamma, buf.data(), aw);
            Full.middleCols(eoff[t], eoff[t + 1] - eoff[t]) = Eigen::Map<Eigen::MatrixXd>(buf.data(), M, eoff[t + 1] - eoff[t]);
          }
          Eigen::MatrixXd DH, DG;
          // assemble rows of Full^T: treat Full as V with M "rank" rows transposed
          assemble(Full, DH, DG);  // Full: M x ncol -> DH: M x 3nc
          if (K_) {
            const Eigen::MatrixXd g = (DG / gamma) * Cl;
            for (int k = 0; k < m; ++k) gct.middleRows(3 * pts[k], 3) += g.middleRows(3 * k, 3);
          }
          // exact block: truncated SVD of the joint [H | gamma G]; low rank if it pays
          Eigen::MatrixXd Jt(M, DH.cols() + DG.cols());
          Jt << DH, DG;
          // randomized range finder (doubling until the Frobenius residual is below eps), then the
          // SVD of the small projection: ~10x cheaper than an SVD of the whole block
          const double nA = Jt.norm();
          int kk = std::min<int>(48, std::min<int>(M, int(Jt.cols())));
          Eigen::MatrixXd Q, Bm;
          if (!o_.rsvd) {
            Q = Eigen::MatrixXd::Identity(M, M);
            Bm = Jt;
          }
          while (o_.rsvd) {
            Eigen::MatrixXd Om(Jt.cols(), kk);
            std::normal_distribution<double> nd;
            for (int i = 0; i < Om.size(); ++i) Om.data()[i] = nd(rng);
            Eigen::HouseholderQR<Eigen::MatrixXd> hq(Jt * Om);
            Q = hq.householderQ() * Eigen::MatrixXd::Identity(M, kk);
            Bm = Q.transpose() * Jt;
            if (kk >= std::min<int>(M, int(Jt.cols())) || (Jt - Q * Bm).norm() <= 0.3 * o_.eps * nA) break;
            kk = std::min<int>(2 * kk, std::min<int>(M, int(Jt.cols())));
          }
          Eigen::BDCSVD<Eigen::MatrixXd> sv(Bm, Eigen::ComputeThinU | Eigen::ComputeThinV);
          const Eigen::VectorXd& sg = sv.singularValues();
          const double tot = nA * nA;
          int r = int(sg.size());
          double tail = std::max(0.0, tot - sg.squaredNorm());
          while (r > 1 && tail + sg(r - 1) * sg(r - 1) <= o_.eps * o_.eps * tot) {
            --r;
            tail += sg(r) * sg(r);
          }
          if (double(r) * (M + DH.cols()) < 0.8 * double(M) * DH.cols()) {
            const Eigen::MatrixXd Uf = Q * sv.matrixU().leftCols(r) * sg.head(r).asDiagonal();
            const Eigen::MatrixXd Wf = sv.matrixV().topRows(DH.cols()).leftCols(r).transpose();
            F.rank = r;
            F.U.assign(Uf.data(), Uf.data() + Uf.size());
            F.W.assign(Wf.data(), Wf.data() + Wf.size());
            if (!kept.empty()) {
              const Eigen::MatrixXd Wg = keptG(sv.matrixV().bottomRows(DG.cols()).leftCols(r).transpose());
              F.WG.assign(Wg.data(), Wg.data() + Wg.size());
            }
          } else {
            F.rank = -1;
            F.D.assign(DH.data(), DH.data() + DH.size());
            if (!kept.empty()) {
              const Eigen::MatrixXd Dg = keptG(DG);
              F.DG.assign(Dg.data(), Dg.data() + Dg.size());
            }
          }
          aw.t_rc += wt() - trc;
          continue;
        }
        if (o_.use_aca) assemble(Va, WH, WG);
        else { WH = Va.leftCols(3 * nc); WG = Va.rightCols(3 * nt); }
        // recompression: [U] [WH | WG] -> QR, SVD, Frobenius tail <= eps
        const int k = int(Ua.cols());
        if (k == 0) { F.rank = 0; aw.t_rc += wt() - trc; continue; }
        Eigen::MatrixXd Wj(k, WH.cols() + WG.cols());
        Wj << WH, WG;
        // QR of U only; B = R_u W (k x cols) has the block's singular values, from the k x k Gram
        // B B^T (accurate to ~1e-8 relative, far below eps); the reflectors are applied, not formed.
        Eigen::HouseholderQR<Eigen::MatrixXd> qu(Ua);
        const int kq = std::min<int>(k, M);
        const Eigen::MatrixXd B = qu.matrixQR().topRows(kq).triangularView<Eigen::Upper>() * Wj;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B * B.transpose());
        const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);  // ascending
        const double tot = lam.sum();
        int r = kq;
        double tail = 0.0;
        while (r > 1 && tail + lam(kq - r) <= o_.eps * o_.eps * tot) {
          tail += lam(kq - r);
          --r;
        }
        const Eigen::MatrixXd Vr = es.eigenvectors().rightCols(r);
        Eigen::MatrixXd Uf = Eigen::MatrixXd::Zero(M, r);
        Uf.topRows(kq) = Vr;
        Uf.applyOnTheLeft(qu.householderQ());
        const Eigen::MatrixXd Wf = Vr.transpose() * B;  // r x (3nc + 3nt)
        F.rank = r;
        F.U.assign(Uf.data(), Uf.data() + Uf.size());
        const Eigen::MatrixXd WHf = Wf.leftCols(3 * nc);
        F.W.assign(WHf.data(), WHf.data() + WHf.size());
        if (!kept.empty()) {
          const Eigen::MatrixXd Wg = keptG(Wf.rightCols(3 * nt));
          F.WG.assign(Wg.data(), Wg.data() + Wg.size());
        }
        if (K_) {
          const Eigen::MatrixXd g = Uf * ((Wf.rightCols(3 * nt) / gamma) * Cl);
          for (int kk = 0; kk < m; ++kk) gct.middleRows(3 * pts[kk], 3) += g.middleRows(3 * kk, 3);
        }
        aw.t_rc += wt() - trc;
      }
#pragma omp critical
      {
        st_.cpu_row += aw.t_row; st_.cpu_col += aw.t_col; st_.cpu_aca += aw.t_aca; st_.cpu_rc += aw.t_rc;
        st_.n_rows += aw.n_rows; st_.n_cols += aw.n_cols;
        st_.cpu_cw += aw.t_cw; st_.cpu_qr += aw.t_qr; st_.cpu_rr += aw.t_rr; st_.cpu_test += aw.t_test;
        st_.attempts += aw.attempts;
        st_.cur_failed += aw.n_fail;
        st_.cur_failed_entries += aw.fail_entries;
      }
    }
    for (const auto& g : gc_t) GC_ += g;
    st_.far_blocks = int(far_.size());
    st_.dense_fallbacks = fallbacks;
    double rs = 0.0;
    int nr = 0;
    for (const FarBlock& F : far_) {
      st_.bytes_far += 4.0 * (F.U.size() + F.W.size() + F.D.size() + F.cols.size() + F.WG.size() + F.DG.size() +
                              F.tcols.size());
      if (F.rank >= 0) { rs += F.rank; ++nr; st_.max_rank = std::max(st_.max_rank, F.rank); }
    }
    st_.rank_mean = nr ? rs / nr : 0.0;
  }

  // y = Hraw x (+ near rigid term if folded) - G t (kept columns; t may be null); with_far false: near only
  void apply(const double* x, const double* t, double* y, bool with_far) const {
    const int N = 3 * np_;
    std::fill(y, y + N, 0.0);
#pragma omp parallel
    {
      std::vector<double> xl, yl;
#pragma omp for schedule(dynamic, 4) nowait
      for (int q = 0; q < int(near_.size()); ++q) {
        const NearLeaf& L = near_[q];
        const ClusterTree::Node& P = pt_.nodes[L.node];
        const int M = 3 * P.size(), nc = int(L.cols.size());
        xl.resize(3 * nc);
        for (int c = 0; c < nc; ++c)
          for (int j = 0; j < 3; ++j) xl[3 * c + j] = x[3 * L.cols[c] + j];
        yl.assign(M, 0.0);
        const float* H = L.H.data();
        for (int c = 0; c < 3 * nc; ++c) {
          const double v = xl[c];
          const float* h = H + size_t(M) * c;
#pragma omp simd
          for (int r = 0; r < M; ++r) yl[r] += h[r] * v;
        }
        for (int c = 0; t && c < 3 * int(L.tcols.size()); ++c) {
          const double v = -t[3 * L.tcols[c / 3] + c % 3];
          const float* h = L.G.data() + size_t(M) * c;
#pragma omp simd
          for (int r = 0; r < M; ++r) yl[r] += h[r] * v;
        }
        for (int k = 0; !dres_.empty() && k < P.size(); ++k) {
          const int a = pt_.perm[P.lo + k];
          const double* r = dres_.data() + doff_[a];
          for (const auto& [B, NB] : rows_[a].basis) {
            for (int j = 0; j < 3; ++j)
              for (int i = 0; i < 3; ++i) yl[3 * k + i] += r[3 * j + i] * x[3 * B + j];
            r += 9;
          }
        }
        for (int k = 0; k < P.size(); ++k)
          for (int i = 0; i < 3; ++i) y[3 * pt_.perm[P.lo + k] + i] += yl[3 * k + i];  // rows disjoint per leaf
      }
    }
    if (!with_far) return;
#pragma omp parallel
    {
      std::vector<double> yp(N, 0.0), xl, tl, yl;
#pragma omp for schedule(dynamic, 4) nowait
      for (int b = 0; b < int(far_.size()); ++b) {
        const FarBlock& F = far_[b];
        const ClusterTree::Node& P = pt_.nodes[F.pnode];
        const int M = 3 * P.size(), nc = int(F.cols.size());
        xl.resize(3 * nc);
        for (int c = 0; c < nc; ++c)
          for (int j = 0; j < 3; ++j) xl[3 * c + j] = x[3 * F.cols[c] + j];
        yl.assign(M, 0.0);
        const int nt = t ? int(F.tcols.size()) : 0;
        if (F.rank < 0) {
          for (int c = 0; c < 3 * nc; ++c) {
            const double v = xl[c];
            const float* h = F.D.data() + size_t(M) * c;
#pragma omp simd
            for (int r = 0; r < M; ++r) yl[r] += h[r] * v;
          }
          for (int c = 0; c < 3 * nt; ++c) {
            const double v = -t[3 * F.tcols[c / 3] + c % 3];
            const float* h = F.DG.data() + size_t(M) * c;
#pragma omp simd
            for (int r = 0; r < M; ++r) yl[r] += h[r] * v;
          }
        } else {
          const int r = F.rank;
          tl.assign(r, 0.0);
          for (int c = 0; c < 3 * nc; ++c) {
            const double v = xl[c];
            const float* w = F.W.data() + size_t(r) * c;
            for (int a = 0; a < r; ++a) tl[a] += w[a] * v;
          }
          for (int c = 0; c < 3 * nt; ++c) {
            const double v = -t[3 * F.tcols[c / 3] + c % 3];
            const float* w = F.WG.data() + size_t(r) * c;
            for (int a = 0; a < r; ++a) tl[a] += w[a] * v;
          }
          for (int a = 0; a < r; ++a) {
            const double v = tl[a];
            const float* u = F.U.data() + size_t(M) * a;
#pragma omp simd
            for (int q = 0; q < M; ++q) yl[q] += u[q] * v;
          }
        }
        for (int k = 0; k < P.size(); ++k)
          for (int i = 0; i < 3; ++i) yp[3 * pt_.perm[P.lo + k] + i] += yl[3 * k + i];
      }
#pragma omp critical
      for (int i = 0; i < N; ++i) y[i] += yp[i];
    }
  }

  // Rigid-body term folded into the near rows: H = Hraw - D.
  void rigid() {
    const int N = 3 * mesh_.ndof;
    Eigen::MatrixXd& S = S_;
    S.resize(3 * np_, 3);
    Eigen::VectorXd t(N), s(3 * np_);
    for (int j = 0; j < 3; ++j) {
      t.setZero();
      for (int g = 0; g < mesh_.ndof; ++g) t(3 * g + j) = 1.0;
      apply(t.data(), nullptr, s.data(), true);
      S.col(j) = s;
    }
    // The fold H = Hraw - D is rounded to float; its rounding residual on these
    // (self) entries is kept in double and added in apply: the diagonal term is
    // a large difference, and its float error alone shifts stiff mixed problems
    // (cantilever) by ~1e-4.
    std::vector<double> dres;
    doff_.assign(np_, 0);
    for (NearLeaf& L : near_) {
      const ClusterTree::Node& P = pt_.nodes[L.node];
      const int M = 3 * P.size();
      for (int k = 0; k < P.size(); ++k) {
        const int a = pt_.perm[P.lo + k];
        doff_[a] = dres.size();
        for (const auto& [B, NB] : rows_[a].basis) {
          const auto it = std::lower_bound(L.cols.begin(), L.cols.end(), B);
          if (it == L.cols.end() || *it != B) throw std::runtime_error("ElemHMatrix: own dof not in the near field");
          const int c = int(it - L.cols.begin());
          for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
              float& h = L.H[(3 * k + i) + size_t(M) * (3 * c + j)];
              const double v = double(h) - NB * S(3 * a + i, j);
              h = float(v);
              dres.push_back(v - double(h));
            }
        }
      }
    }
    dres_ = std::move(dres);
  }
};

}  // namespace igabem
