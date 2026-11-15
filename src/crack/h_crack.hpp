// Hierarchical (H-matrix) path for the dual crack model (dual_model.hpp): the same collocation rows, columns,
// regularization and Dirichlet handling as CrackModel::solve, with the O(N^2) part compressed.
//
//   A = H Q + F
//
//   H   one H-matrix (Htool: block ACA + SVD recompression to eps, admissibility 2 min(r) < eta dist) of the raw
//       element integrals of every collocation row against three kinds of columns:
//         displacement dofs        (DBIE rows: T, TBIE rows: -S),
//         traction-space dofs, all patches (DBIE: -U, TBIE: D): the Dirichlet patches' ones are unknowns, the rest
//                                  carry the loads (b = -H t, t the L2 projection of the data on each patch, exact
//                                  for constant and linear data),
//         enrichment columns       (SIFs and r^3/2 amplitudes: T or -S times the enrichment vector);
//       Q places the unknowns in that column space;
//   F   sparse: each row's finish (dbieFinish / tbieFinish: the rigid / linear field terms at its own and
//       coincident points) and the tying rows (Eq. 13). The finish needs the row's element sums, which come from
//       products of H with fields: T0, S0 with constants (partition of unity), S1 with the linear fields
//       (isoparametric: sum_l R_l P_l = y), Mn = sum int D n_k with n_k on the traction columns (exact on flat
//       patches; a moment H-matrix of int_e D n_k otherwise).
//
// The near leaves use the dense path's quadrature (ElementIntegrator::integrate, far_tol of the model), so they
// equal the dense entries up to round-off; the entries that the ACA samples in admissible blocks take a looser far
// tolerance (entry_tol), except on enriched elements (the enrichment's sqrt(r) is not polynomial: a loose whole-
// element rule there moved K by 7e-6). Solved by right-preconditioned restarted GMRES; the preconditioner is the
// sparse LU (UMFPACK) of the near-leaf entries within `radius` support radii of the row point, plus F.
#pragma once

#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <chrono>
#include <memory>
#include <random>
#include <vector>

#include "../fast/hmatrix_operator.hpp"
#include "../hmat/two_level.hpp"
#include "dual_model.hpp"

namespace igabem {
namespace crack {

// Raw entries of the crack collocation rows (the interface of SbieEntries for HOperator / BlockACA / NearField).
// Main: column triples [0, nd) displacement dofs, [nd, nd + nts) traction-space dofs, then the enrichment columns
// in triples (scalar column cK + 3 t + c). Moment: column triples 3 ie + k, int_e D n_k (TBIE rows).
class CrackEntries {
 public:
  enum Kind { Main, Moment };
  using Row = CrackModel::Row;

  CrackEntries(const CrackModel& M, const std::vector<Row>& rows, const ElementIntegrator& quad,
               const ElementIntegrator& quad_aca, Kind kind)
      : M_(M), mesh_(M.mesh), rows_(rows), quad_(quad), quadA_(quad_aca), K_(M.mat), KT_(M.mat), kind_(kind) {
    const int ne = int(mesh_.elements.size());
    nd_ = mesh_.ndof;
    nts_ = M.ts.ndof;
    ne3_ = (M.cT - M.cK) / 3;
    ncol_ = kind == Main ? nd_ + nts_ + ne3_ : 3 * ne;
    support_.assign(ncol_, {});
    enriched_.assign(ne, 0);
    for (int ie = 0; ie < ne; ++ie) {
      if (kind == Moment) { for (int k = 0; k < 3; ++k) support_[3 * ie + k].push_back(ie); continue; }
      const Element& e = mesh_.elements[ie];
      PatchEval E;
      mesh_.patches[e.patch].evalSpan(e.su, e.sv, 0.5 * (e.rect.u0 + e.rect.u1), 0.5 * (e.rect.v0 + e.rect.v1), E);
      for (int l = 0; l < E.n_loc; ++l) {
        support_[mesh_.l2g[e.patch][E.idx[l]]].push_back(ie);
        support_[nd_ + M.ts.offset[e.patch] + E.idx[l]].push_back(ie);
      }
      if (!M.opt.enrich || M.piece[e.patch] < 0) continue;
      // enrichment columns: supported on every enriched element, so they stay in the near field (a dense border,
      // one integration per row and element with the rest of the row). Their exact supports (front piece)
      // compress, but each ACA sample of such a column integrates a whole piece with the exact rules and the build
      // took 1.5x longer; sampling the enrichment for supports missed elements (7e-8 errors in A x, penny n11).
      for (int t = 0; t < ne3_; ++t) support_[nd_ + nts_ + t].push_back(ie);
      enriched_[ie] = 1;
    }
  }

  int rows() const { return 3 * int(rows_.size()); }
  int cols() const { return 3 * ncol_; }

  // Anchor and support radius of every scalar column (3 per triple), and 3 coordinates per scalar row.
  void columnGeometry(std::vector<double>& xyz, std::vector<double>& radius) const {
    xyz.resize(3 * size_t(cols()));
    radius.resize(size_t(cols()));
    for (int c = 0; c < ncol_; ++c) {
      Vec3 lo = Vec3::Constant(1e300), hi = -lo;
      for (int ie : support_[c]) {
        const Vec3& x = quad_.elementCentre(ie);
        const double r = quad_.elementRadius(ie);
        lo = lo.cwiseMin(x - Vec3::Constant(r));
        hi = hi.cwiseMax(x + Vec3::Constant(r));
      }
      if (support_[c].empty()) lo = hi = Vec3::Zero();
      const Vec3 x = 0.5 * (lo + hi);
      const double r = 0.5 * (hi - lo).norm();
      for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 3; ++d) xyz[3 * (3 * size_t(c) + i) + d] = x(d);
        radius[3 * size_t(c) + i] = r;
      }
    }
  }
  void rowGeometry(std::vector<double>& xyz) const {
    xyz.resize(3 * size_t(rows()));
    for (size_t a = 0; a < rows_.size(); ++a)
      for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) xyz[3 * (3 * a + i) + d] = rows_[a].x(d);
  }

  // f(k, acc): acc the 3 x 3 blocks (9 doubles each, column-major) of row point row_points[k] against col_dofs,
  // dense-path quadrature (near leaves).
  template <class F>
  void rowsAgainst(const std::vector<int>& row_points, const std::vector<int>& col_dofs, F&& f) const {
    rowsAgainstQ(row_points, col_dofs, false, f);
  }

  // out(i + M j) = A(rows[i], cols[j]), column-major M x N (ACA samples: the looser far rules).
  void block(int M, int N, const int* rws, const int* cls, double* out) const {
    std::vector<int> pts, dofs;
    std::vector<int> rp(M), cp(N);
    for (int i = 0; i < M; ++i) {
      if (pts.empty() || pts.back() != rws[i] / 3) pts.push_back(rws[i] / 3);
      rp[i] = int(pts.size()) - 1;
    }
    {
      Scratch& s = scratch();
      s.pos.resize(ncol_, -1);
      for (int j = 0; j < N; ++j) {
        const int c = cls[j] / 3;
        if (s.pos[c] < 0) { s.pos[c] = int(dofs.size()); dofs.push_back(c); }
        cp[j] = s.pos[c];
      }
      for (int c : dofs) s.pos[c] = -1;
    }
    int i0 = 0;
    rowsAgainstQ(pts, dofs, true, [&](int k, const double* acc) {
      for (; i0 < M && rp[i0] == k; ++i0) {
        const int ri = rws[i0] % 3;
        for (int j = 0; j < N; ++j) out[i0 + size_t(M) * j] = acc[9 * size_t(cp[j]) + 3 * (cls[j] % 3) + ri];
      }
    });
  }

  // No point-kernel fast path: crack rows are coplanar with the crack columns, their kernels decouple whole
  // components (T: dn = 0) and the implicit ACA of the point kernel stopped early there (1e-2 errors in A x on the
  // penny n7, the entry ACA 5e-10); and the columns mix three kernels.
  struct FarBlock {
    struct Elem { int q0, nq, n_loc; const double* R; std::vector<std::pair<int, int>> lp; };
    std::vector<Elem> elems;
    std::vector<std::vector<std::pair<int, int>>> dof_elems;
    std::vector<int> col_pt, col_comp;
    int points() const { return 0; }
  };
  bool farBlock(int, const int*, int, const int*, double, int, FarBlock&) const { return false; }
  void kernel(int, const FarBlock&, int, double*, int) const {}

 private:
  using Matrix3 = Eigen::Matrix3d;
  const CrackModel& M_;
  const Mesh& mesh_;
  const std::vector<Row>& rows_;
  const ElementIntegrator& quad_;
  const ElementIntegrator& quadA_;
  Kelvin K_;
  KelvinTraction KT_;
  Kind kind_;
  int nd_ = 0, nts_ = 0, ne3_ = 0, ncol_ = 0;
  std::vector<std::vector<int>> support_;
  std::vector<char> enriched_;

  struct Scratch {
    std::vector<int> pos, elems;
    std::vector<char> emark;
    std::vector<double> acc;
  };
  static Scratch& scratch() {
    thread_local Scratch s;
    return s;
  }

  template <class F>
  void rowsAgainstQ(const std::vector<int>& row_points, const std::vector<int>& col_dofs, bool loose, F&& f) const {
    Scratch& s = scratch();
    s.pos.resize(ncol_, -1);
    s.emark.resize(mesh_.elements.size(), 0);
    s.elems.clear();
    for (size_t k = 0; k < col_dofs.size(); ++k) s.pos[col_dofs[k]] = int(k);
    for (int c : col_dofs)
      for (int ie : support_[c])
        if (!s.emark[ie]) { s.emark[ie] = 1; s.elems.push_back(ie); }
    bool enr = false;
    for (int c : col_dofs) enr = enr || (kind_ == Main && c >= nd_ + nts_);
    s.acc.resize(9 * col_dofs.size());
    for (size_t k = 0; k < row_points.size(); ++k) {
      std::fill(s.acc.begin(), s.acc.end(), 0.0);
      for (int ie : s.elems)
        integrate(loose && !enriched_[ie] ? quadA_ : quad_, row_points[k], ie, s.pos, enr, s.acc.data());
      f(int(k), s.acc.data());
    }
    for (int c : col_dofs) s.pos[c] = -1;
    for (int ie : s.elems) s.emark[ie] = 0;
  }

  // Row point a against element ie, the column triples with pos >= 0 (acc: 9 per triple).
  void integrate(const ElementIntegrator& qd, int a, int ie, const std::vector<int>& pos, bool enr, double* acc) const {
    const Row& r = rows_[a];
    const Element& e = mesh_.elements[ie];
    const SourcePoint src{r.patch, r.xi, r.x};
    const bool tbie = r.kind == Row::Lower;
    const auto& l2g = mesh_.l2g[e.patch];
    const int off = nd_ + M_.ts.offset[e.patch];
    enr = enr && enriched_[ie];
    qd.integrate(src, e, [&](const QPoint& q, double w) {
      const Vec3 nJ = q.normal() * mesh_.normal_sign;
      const double J = nJ.norm();
      const Vec3 n = nJ / J;
      Matrix3 A, B, Kd, Kt;
      if (!tbie) {
        K_.eval(q.x - r.x, n, A, B);  // U, T
        Kd = B * (w * J);
        Kt = -A * (w * J);
      } else {
        KT_.eval(q.x - r.x, r.N, n, A, B);  // D, S
        Kd = -B * (w * J);
        Kt = A * (w * J);
      }
      if (kind_ == Moment) {
        for (int k = 0; k < 3; ++k) {
          const int p = pos[3 * ie + k];
          if (p >= 0) Eigen::Map<Matrix3>(acc + 9 * size_t(p)) += n(k) * Kt;
        }
        return;
      }
      for (int l = 0; l < q.n_loc; ++l) {
        const int p = pos[l2g[q.idx[l]]], pt = pos[off + q.idx[l]];
        if (p >= 0) Eigen::Map<Matrix3>(acc + 9 * size_t(p)) += q.R[l] * Kd;
        if (pt >= 0) Eigen::Map<Matrix3>(acc + 9 * size_t(pt)) += q.R[l] * Kt;
      }
      if (!enr) return;
      int cols[kMaxEnr];
      Vec3 vals[kMaxEnr];
      const int ne = M_.quadEnrichment(e.patch, q, cols, vals);
      for (int i = 0; i < ne; ++i) {
        if (cols[i] < M_.cK || cols[i] >= M_.cT) continue;
        const int t = (cols[i] - M_.cK) / 3, c = (cols[i] - M_.cK) % 3, p = pos[nd_ + nts_ + t];
        if (p >= 0) Eigen::Map<Vec3>(acc + 9 * size_t(p) + 3 * c) += Kd * vals[i];
      }
    });
  }
};

struct CrackHOptions {
  HOptions h;              // eps, eta, leaf of the compression
  double gtol = 1e-10;     // GMRES relative residual (the floor is ~1e-12 at cond ~1e6)
  double radius = 2.0;     // preconditioner: near entries with |x_row - anchor| <= radius * support radius
  int restart = 200;
  double entry_tol = 1e-10;  // far-field quadrature tolerance of the ACA's samples (not on enriched elements)
  CrackHOptions() { h.eps = 1e-8; h.eta = 2.0; h.leaf = 192; }
};

class CrackH {
 public:
  using Row = CrackModel::Row;
  using RowAcc = CrackModel::RowAcc;

  struct Stats {
    double t_h = 0, t_hm = 0, t_sums = 0, t_finish = 0, t_precond = 0, t_solve = 0, t_near = 0;
    double ratio = 0, mem_gb = 0;
    long precond_nnz = 0;
    std::vector<int> iterations;
    std::vector<double> residuals;
    double build() const { return t_h + t_sums + t_finish + t_precond; }
  };
  Stats st;

  CrackH(CrackModel& M, const CrackHOptions& o)
      : M_(M), o_(o), quad_(M.mesh, M.opt.quad), quadA_(M.mesh, looser(M.opt.quad, o.entry_tol)) { build(); }

  // Solutions (full scalar vectors, as CrackModel::solve) for the loads (and Dirichlet displacements).
  std::vector<Eigen::VectorXd> solve(const std::vector<TractionFn>& loads,
                                     const std::vector<DisplacementFn>& disps = {}) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Eigen::VectorXd> out;
    for (size_t l = 0; l < loads.size(); ++l) {
      Eigen::VectorXd zfix = Eigen::VectorXd::Zero(M_.ncols);  // known Dirichlet displacements
      if (!disps.empty()) zfix.head(3 * M_.mesh.ndof) = M_.projectDirichlet(disps[l]);
      Eigen::VectorXd b = rhs(loads[l]);
      if (!disps.empty()) b -= full(zfix);
      Eigen::VectorXd x;
      double res = 0;
      const VecOp A = [&](const Eigen::VectorXd& v, Eigen::VectorXd& y) { y = full(scatter(v)); };
      const VecOp P = [&](const Eigen::VectorXd& v, Eigen::VectorXd& y) { y = lu_.solve(v); };
      st.iterations.push_back(gmresRight(A, P, b, x, o_.gtol, res, o_.restart, 2000));
      st.residuals.push_back(res);
      Eigen::VectorXd z = scatter(x);
      z.head(3 * M_.mesh.ndof) += zfix.head(3 * M_.mesh.ndof);
      out.push_back(z);
    }
    st.t_solve = secs(t0);
    M_.unknowns = long(cols_.size());
    M_.t_solve = st.t_solve;
    M_.t_assemble = st.build();
    return out;
  }

  // A z for a full scalar vector z (ncols): all rows (collocation then tying).
  Eigen::VectorXd full(const Eigen::VectorXd& z) const {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(H_.cols()), y = Eigen::VectorXd::Zero(nrows_);
    v.head(3 * nd_) = z.head(3 * nd_);
    for (int j = 0; j < nts_; ++j)
      if (M_.tcol[j] >= 0) v.segment<3>(3 * (nd_ + j)) = z.segment<3>(M_.tcol[j]);
    v.tail(M_.cT - M_.cK) = z.segment(M_.cK, M_.cT - M_.cK);
    H_.apply(v.data(), y.data());  // rows [0, 3 nrp)
    y += S_ * z;
    return y;
  }

  // Unknown vector (model column order) -> full scalar vector (fixed displacement columns zero).
  Eigen::VectorXd scatter(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z = Eigen::VectorXd::Zero(M_.ncols);
    for (size_t j = 0; j < cols_.size(); ++j) z(cols_[j]) = x(j);
    return z;
  }

  int unknowns() const { return int(cols_.size()); }

  // Diagnostic against the dense operator Ad (unknown numbering, CrackModel::factor's f_A): relative error of A x
  // for a random x, per row kind (outer, upper, lower, tying).
  void check(const Eigen::MatrixXd& Ad) const {
    std::mt19937 g(7);
    std::normal_distribution<double> nd;
    Eigen::VectorXd x(cols_.size());
    for (int i = 0; i < x.size(); ++i) x(i) = nd(g);
    const Eigen::VectorXd yh = full(scatter(x)), yd = Ad * x;
    double e[4] = {0, 0, 0, 0}, n[4] = {0, 0, 0, 0};
    for (int i = 0; i < nrows_; ++i) {
      const int k = i >= 3 * nrp_ ? 3 : rows_[i / 3].kind == Row::Outer ? 0 : rows_[i / 3].kind == Row::Upper ? 1 : 2;
      e[k] += std::pow(yh(i) - yd(i), 2);
      n[k] += yd(i) * yd(i);
    }
    const char* nm[4] = {"outer", "upper", "lower", "tying"};
    for (int k = 0; k < 4; ++k)
      std::fprintf(stderr, "check %-6s |Ax_H - Ax_D| / |Ax_D| = %.2e\n", nm[k], std::sqrt(e[k] / std::max(n[k], 1e-300)));
  }

 private:
  CrackModel& M_;
  CrackHOptions o_;
  ElementIntegrator quad_, quadA_;
  static SolverOptions looser(SolverOptions q, double tol) {
    q.far_tol = std::max(q.far_tol, tol);
    return q;
  }
  std::vector<Row> rows_, trows_;  // all collocation rows; the TBIE ones (moment operator)
  std::vector<int> tidx_;          // trows_[k] = rows_[tidx_[k]]
  int nrp_ = 0, nrows_ = 0, nd_ = 0, nts_ = 0;
  std::vector<int> cols_;          // unknown -> model scalar column
  HOperator H_;
  std::vector<RowAcc> acc_;
  Eigen::SparseMatrix<double> S_;  // finish + tying rows, nrows x ncols
  Eigen::SparseMatrix<double> Pm_;  // preconditioner matrix (unknown numbering)
  Eigen::UmfPackLU<Eigen::SparseMatrix<double>> lu_;

  static double secs(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  void build() {
    rows_ = M_.collocationRows();
    nrp_ = int(rows_.size());
    nrows_ = 3 * nrp_ + M_.extraRows();
    nd_ = M_.mesh.ndof;
    nts_ = M_.ts.ndof;
    if ((M_.cT - M_.cK) % 3) throw std::runtime_error("CrackH: enrichment columns not in triples");
    for (int r = 0; r < nrp_; ++r)
      if (rows_[r].kind == Row::Lower) { trows_.push_back(rows_[r]); tidx_.push_back(r); }
    // unknowns: free displacement columns, then [cK, ncols)
    std::vector<char> fixed(nd_, 0);
    for (int p = 0; p < M_.n_outer; ++p)
      if (M_.dirichlet[p]) for (int g : M_.mesh.l2g[p]) fixed[g] = 1;
    for (int g = 0; g < nd_; ++g)
      if (!fixed[g]) for (int c = 0; c < 3; ++c) cols_.push_back(3 * g + c);
    for (int c = M_.cK; c < M_.ncols; ++c) cols_.push_back(c);
    if (int(cols_.size()) != nrows_)
      throw std::runtime_error("crack system not square: " + std::to_string(nrows_) + " rows, " +
                               std::to_string(cols_.size()) + " unknowns");

    auto t = std::chrono::steady_clock::now();
    CrackEntries E(M_, rows_, quad_, quadA_, CrackEntries::Main);
    std::vector<double> rx, cx, cr;
    E.rowGeometry(rx);
    E.columnGeometry(cx, cr);
    H_.build(E, rx, cx, cr, o_.h);
    st.t_h = secs(t);
    st.t_near = H_.t_near;
    st.ratio = H_.stats().ratio();
    st.mem_gb = H_.stats().bytes() / 1e9;
    t = std::chrono::steady_clock::now();
    sums();
    st.t_sums = secs(t);
    t = std::chrono::steady_clock::now();
    finishRows();
    st.t_finish = secs(t);
    t = std::chrono::steady_clock::now();
    precond(cx, cr);
    st.t_precond = secs(t);
  }

  // A patch's unit normal if it is flat (to 1e-12), else NaN.
  Vec3 flatNormal(int p) const {
    const Patch& P = M_.mesh.patches[p];
    Vec3 n0 = Vec3::Constant(NAN);
    for (int i = 0; i <= 4; ++i)
      for (int j = 0; j <= 4; ++j) {
        PatchEval E;
        P.eval(P.U[P.p] + (P.U[P.nu] - P.U[P.p]) * i / 4.0, P.V[P.q] + (P.V[P.nv] - P.V[P.q]) * j / 4.0, E);
        const Vec3 n = (E.normal() * M_.mesh.normal_sign).normalized();
        if (std::isnan(n0(0))) n0 = n;
        else if ((n - n0).norm() > 1e-12) return Vec3::Constant(NAN);
      }
    return n0;
  }

  // Element sums of every row from products of H with constant and linear fields.
  void sums() {
    const Mesh& mesh = M_.mesh;
    std::vector<int> fg(nd_, 0);
    std::vector<Vec3> P(nd_, Vec3::Zero());
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      for (size_t k = 0; k < mesh.l2g[p].size(); ++k) {
        fg[mesh.l2g[p][k]] = M_.face[p];
        P[mesh.l2g[p][k]] = mesh.patches[p].cp(int(k));
      }
    acc_.assign(nrp_, RowAcc{});
    const long nc = H_.cols();
    // 3 groups (lower, outer, upper) x 3 components x (constant, y_0, y_1, y_2): one product with 36 columns
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(nc, 36);
    for (int d = 0; d < nd_; ++d) {
      const int g = fg[d] + 1;
      for (int j = 0; j < 3; ++j)
        for (int k = -1; k < 3; ++k) V(3 * d + j, 12 * g + 4 * j + (k + 1)) = k < 0 ? 1.0 : P[d](k);
    }
    const Eigen::MatrixXd Y = H_ * V;
    for (int r = 0; r < nrp_; ++r) {
      const Row& row = rows_[r];
      RowAcc& A = acc_[r];
      for (int g = 0; g < 3; ++g)
        for (int j = 0; j < 3; ++j)
          for (int k = -1; k < 3; ++k) {
            const Vec3 h = Y.block<3, 1>(3 * r, 12 * g + 4 * j + (k + 1));
            if (row.kind != Row::Lower) {  // raw T: T0[grp], grp 1 = lower-face elements of an upper row
              if (k >= 0) continue;
              A.T0[(row.kind == Row::Upper && g == 0) ? 1 : 0].col(j) += h;
            } else {  // raw -S: S0[grp], S1[grp][k] (sum S y_k so far), grp 1 = upper-face elements
              const int grp = g == 2 ? 1 : 0;
              if (k < 0) A.S0[grp].col(j) -= h;
              else A.S1[grp][k].col(j) -= h;
            }
          }
      if (row.kind == Row::Lower)
        for (int grp = 0; grp < 2; ++grp)
          for (int k = 0; k < 3; ++k) A.S1[grp][k] -= row.x(k) * A.S0[grp];
    }
    if (trows_.empty()) return;
    // Mn[k].col(j) = sum int D(:, j) n_k: the traction columns (raw D) times n_k e_j, exact where n is constant
    std::vector<Vec3> fn(mesh.patches.size());
    bool flat = true;
    for (size_t p = 0; p < mesh.patches.size(); ++p) {
      fn[p] = flatNormal(int(p));
      flat = flat && !std::isnan(fn[p](0));
    }
    if (flat) {
      Eigen::MatrixXd W = Eigen::MatrixXd::Zero(nc, 9);
      for (size_t p = 0; p < mesh.patches.size(); ++p) {
        const int n1 = p + 1 < M_.ts.offset.size() ? M_.ts.offset[p + 1] : nts_;
        for (int t = M_.ts.offset[p]; t < n1; ++t)
          for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j) W(3 * (nd_ + t) + j, 3 * k + j) = fn[p](k);
      }
      const Eigen::MatrixXd Z = H_ * W;
      for (size_t q = 0; q < trows_.size(); ++q)
        for (int k = 0; k < 3; ++k)
          for (int j = 0; j < 3; ++j) acc_[tidx_[q]].Mn[k].col(j) = Z.block<3, 1>(3 * tidx_[q], 3 * k + j);
      return;
    }
    // curved patches: an H-matrix of the element moments int_e D n_k (TBIE rows)
    const auto t = std::chrono::steady_clock::now();
    CrackEntries Em(M_, trows_, quad_, quadA_, CrackEntries::Moment);
    std::vector<double> mx, mcx, mcr;
    Em.rowGeometry(mx);
    Em.columnGeometry(mcx, mcr);
    HOperator Hm;
    Hm.build(Em, mx, mcx, mcr, o_.h);
    const int ne = int(mesh.elements.size());
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(9 * ne, 9);
    for (int ie = 0; ie < ne; ++ie)
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j) W(3 * (3 * ie + k) + j, 3 * k + j) = 1.0;
    const Eigen::MatrixXd Z = Hm * W;
    for (size_t q = 0; q < trows_.size(); ++q)
      for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j) acc_[tidx_[q]].Mn[k].col(j) = Z.block<3, 1>(3 * q, 3 * k + j);
    st.t_hm = secs(t);
  }

  // Finish rows (own / coincident point terms) and tying rows into S_.
  void finishRows() {
    std::vector<Eigen::Triplet<double>> tr;
    std::vector<SparseRow> fr(nrp_);
    const std::vector<TractionFn> none;
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < nrp_; ++r) {
      Eigen::MatrixXd b(3, 0);
      if (rows_[r].kind == Row::Lower) M_.tbieFinish(rows_[r], none, fr[r], b, false, acc_[r]);
      else M_.dbieFinish(rows_[r], fr[r], acc_[r]);
      fr[r].compress();
    }
    for (int r = 0; r < nrp_; ++r)
      for (const auto& [c, v] : fr[r].e)
        for (int i = 0; i < 3; ++i)
          if (v(i) != 0.0) tr.emplace_back(3 * r + i, c, v(i));
    if (M_.extraRows()) {
      Eigen::MatrixXd T = Eigen::MatrixXd::Zero(M_.extraRows(), M_.ncols);
      M_.tyingRows(T, 0);
      for (int c = 0; c < T.cols(); ++c)
        for (int i = 0; i < T.rows(); ++i)
          if (T(i, c) != 0.0) tr.emplace_back(3 * nrp_ + i, c, T(i, c));
    }
    S_.resize(nrows_, M_.ncols);
    S_.setFromTriplets(tr.begin(), tr.end());
  }

  // Right-hand side of one load: -H t (t the traction data projected on the known-traction patches) and the TBIE
  // finish's load terms.
  Eigen::VectorXd rhs(const TractionFn& load) const {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(H_.cols()), y = Eigen::VectorXd::Zero(nrows_);
    v.segment(3 * nd_, 3 * nts_) = projectTraction(load);
    H_.apply(v.data(), y.data());
    Eigen::VectorXd b = -y;
    const std::vector<TractionFn> L{load};
    for (int r = 0; r < nrp_; ++r) {
      if (rows_[r].kind != Row::Lower) continue;
      SparseRow a;
      Eigen::MatrixXd bb = Eigen::MatrixXd::Zero(3, 1);
      M_.tbieFinish(rows_[r], L, a, bb, true, acc_[r]);
      b.segment<3>(3 * r) += bb.col(0);
    }
    return b;
  }

  // L2 projection of the traction data on each known-traction patch's basis (zero on Dirichlet patches).
  Eigen::VectorXd projectTraction(const TractionFn& load) const {
    const Mesh& mesh = M_.mesh;
    Eigen::VectorXd out = Eigen::VectorXd::Zero(3 * nts_);
    const GaussRule gr = gaussLegendre(8);
    for (size_t p = 0; p < mesh.patches.size(); ++p) {
      if (int(p) < M_.n_outer && M_.dirichlet[p]) continue;
      const Patch& Pa = mesh.patches[p];
      const int nb = Pa.nu * Pa.nv;
      Eigen::MatrixXd Mm = Eigen::MatrixXd::Zero(nb, nb), F = Eigen::MatrixXd::Zero(nb, 3);
      for (const Element& e : mesh.elements) {
        if (e.patch != int(p)) continue;
        for (size_t i = 0; i < gr.x.size(); ++i)
          for (size_t j = 0; j < gr.x.size(); ++j) {
            const double u = e.rect.u0 + (e.rect.u1 - e.rect.u0) * gr.x[i], v = e.rect.v0 + (e.rect.v1 - e.rect.v0) * gr.x[j];
            PatchEval E;
            Pa.evalSpan(e.su, e.sv, u, v, E);
            const Vec3 nJ = E.normal() * mesh.normal_sign;
            const double J = nJ.norm();
            const double w = gr.w[i] * gr.w[j] * (e.rect.u1 - e.rect.u0) * (e.rect.v1 - e.rect.v0) * J;
            const Vec3 tv = load(E.x, nJ / J, int(p));
            for (int a = 0; a < E.n_loc; ++a) {
              F.row(E.idx[a]) += w * E.R[a] * tv.transpose();
              for (int c = 0; c < E.n_loc; ++c) Mm(E.idx[a], E.idx[c]) += w * E.R[a] * E.R[c];
            }
          }
      }
      const Eigen::MatrixXd C = Mm.ldlt().solve(F);
      for (int a = 0; a < nb; ++a) out.segment<3>(3 * (M_.ts.offset[p] + a)) = C.row(a).transpose();
    }
    return out;
  }

  // Sparse LU of the local part: near-leaf entries within radius of their column's support, plus S_.
  void precond(const std::vector<double>& cx, const std::vector<double>& cr) {
    std::vector<int> pos(M_.ncols, -1);  // model column -> unknown
    for (size_t j = 0; j < cols_.size(); ++j) pos[cols_[j]] = int(j);
    std::vector<int> hc(H_.cols(), -1);  // H column (scalar) -> unknown
    for (int i = 0; i < 3 * nd_; ++i) hc[i] = pos[i];
    for (int j = 0; j < nts_; ++j)
      if (M_.tcol[j] >= 0) for (int c = 0; c < 3; ++c) hc[3 * (nd_ + j) + c] = pos[M_.tcol[j] + c];
    for (int c = 0; c < M_.cT - M_.cK; ++c) hc[3 * (nd_ + nts_) + c] = pos[M_.cK + c];
    std::vector<Eigen::Triplet<double>> tr;
    const double R = o_.radius;
    H_.forDenseLeaves([&](const int* rr, int m, const int* cc, int n, const double* D) {
      for (int j = 0; j < n; ++j) {
        const int u = hc[cc[j]];
        if (u < 0) continue;
        const Vec3 x(cx[3 * size_t(cc[j])], cx[3 * size_t(cc[j]) + 1], cx[3 * size_t(cc[j]) + 2]);
        const double rad = R * cr[cc[j]];
        for (int i = 0; i < m; ++i) {
          if ((rows_[rr[i] / 3].x - x).norm() > rad) continue;
          tr.emplace_back(rr[i], u, D[i + size_t(m) * j]);
        }
      }
    });
    for (int k = 0; k < S_.outerSize(); ++k)
      for (Eigen::SparseMatrix<double>::InnerIterator it(S_, k); it; ++it)
        if (pos[it.col()] >= 0) tr.emplace_back(int(it.row()), pos[it.col()], it.value());
    Pm_.resize(nrows_, nrows_);
    Pm_.setFromTriplets(tr.begin(), tr.end());
    Pm_.makeCompressed();
    st.precond_nnz = Pm_.nonZeros();
    lu_.compute(Pm_);
    if (lu_.info() != Eigen::Success) throw std::runtime_error("CrackH: preconditioner LU failed");
  }
};

}  // namespace crack
}  // namespace igabem
