// Mixed boundary conditions (BcModel) on the hierarchical operator: the system
//
//   [H(:, u_free) + S_spring   -G(:, t_free)   Bt] [u_free]   [G t_data - H u_fixed + b]
//   [Bb                                       0 ] [t_free] = [0]                      ...
//
// over the rows `rowsel` of the points `rowpts` (dof collocation points, then the
// traction points of the supported faces), solved by GMRES with the two-level
// preconditioner. The operator stores G only on the traction dofs that are
// unknown (supported faces) or condensed (interfaces of bonded inclusions,
// multi_region.hpp: t_I = -S_I u_I, applied in the product, not assembled), or
// on every traction dof with full_g (smooth traction data on the whole surface).
// Springs stay the dense block BcModel builds (local supports: rows x spring dofs).
//
// Preconditioner (two_level.hpp): affine coarse level (displacement and traction
// aggregates apart), then the sparse LU of the near field and a dense LU of the
// whole traction block of the supported faces (rows at their traction points x
// unknown tractions), swept forward. That block is a first-kind single layer: cut
// to a local pattern (even 4 h) GMRES stalls on a bolt hole; whole, it converges
// in 20-60 iterations, at nt^2 memory and nt^3 factor work.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../multi_region.hpp"
#include "elem_hmatrix.hpp"
#include "two_level.hpp"

namespace igabem {

struct HMixedOptions {
  ElemHOptions h;
  double gtol = 1e-10;    // GMRES relative residual
  double radius = 1.0;   // preconditioner pattern radius, in local element diameters
  int coarse = 256;      // coarse aggregate (points; 0: off)
  bool full_g = false;   // store G on every traction dof (smooth traction data)
  HMixedOptions() { h.eps = 1e-5; }
};

class HMixed {
 public:
  struct Stats {
    double t_build = 0.0, t_precond = 0.0, t_coarse = 0.0, t_solve = 0.0;
    double t_matvec = 0.0, t_pc = 0.0;  // of t_solve: products, preconditioner applications
    long nnz = 0;
    int coarse_dim = 0;
    std::vector<int> iterations;
    std::vector<double> residuals;
  };

  HMixed(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& opt,
         const std::vector<Collocation>& rowpts, const std::vector<int>& rowsel, const std::vector<int>& u_free,
         const std::vector<int>& t_free, const MultiRegion* regions, const HMixedOptions& o)
      : mesh_(m), ts_(ts), rowsel_(rowsel), u_free_(u_free), t_free_(t_free), o_(o) {
    if (regions && !regions->empty()) regions_ = regions;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<char> keep(ts.ndof, char(o.full_g));
    for (int j : t_free) keep[j / 3] = 1;
    if (regions_)
      for (int j : regions_->t_interface) keep[j / 3] = 1;
    H_ = std::make_unique<ElemHMatrix>(m, ts, mat, opt, Eigen::MatrixXd(3 * ts.ndof, 0), o.h, &rowpts, &keep);
    st.t_build = since(t0);
    nu_ = int(u_free.size());
    n_ = nu_ + int(t_free.size());
    rowinv_.assign(3 * rowpts.size(), -1);
    for (int r = 0; r < n_; ++r) rowinv_[rowsel[r]] = r;
    ucol_.assign(3 * m.ndof, -1);
    for (int k = 0; k < nu_; ++k) ucol_[u_free[k]] = k;
    tcol_.assign(3 * ts.ndof, -1);
    for (size_t k = 0; k < t_free.size(); ++k) tcol_[t_free[k]] = nu_ + int(k);
  }

  bool fullG() const { return o_.full_g; }
  const HMixedOptions& options() const { return o_; }
  const ElemHMatrix& op() const { return *H_; }

  // After BcModel's springs and free modes: S (n x 3 spring dofs, dense), free modes on u_free (nu x m).
  void factor(const Eigen::MatrixXd& S, const std::vector<int>& spring_dofs, const Eigen::MatrixXd& Cu) {
    const auto t0 = std::chrono::steady_clock::now();
    S_ = S;
    spring_dofs_ = spring_dofs;
    std::vector<Eigen::Triplet<double>> ht, gt, trip;
    H_->preconTriplets(o_.radius, ht, &gt);
    // The traction rows against the unknown tractions: the whole block, exact and dense
    // (TwoLevel factors it apart); its near entries leave the sparse pattern.
    const int nt = n_ - nu_;
    Eigen::MatrixXd D;
    if (nt) {
      std::vector<char> tmask(ts_.ndof, 0);
      for (int j : t_free_) tmask[j / 3] = 1;
      std::vector<int> tpts, prow(H_->rowPoints().size(), -1);  // block rows per point and component
      for (int r = nu_; r < n_; ++r) {
        const int a = rowsel_[r] / 3;
        if (prow[a] < 0) { prow[a] = int(tpts.size()); tpts.push_back(a); }
      }
      std::vector<std::array<int, 3>> brow(tpts.size(), {-1, -1, -1});
      for (int r = nu_; r < n_; ++r) brow[prow[rowsel_[r] / 3]][rowsel_[r] % 3] = r - nu_;
      gt.erase(std::remove_if(gt.begin(), gt.end(),
                              [&](const auto& e) { return rowinv_[e.row()] >= nu_ && tcol_[e.col()] >= 0; }),
               gt.end());
      D = Eigen::MatrixXd::Zero(nt, nt);
      H_->gBlock(tpts, tmask, [&](int k, int i, int J, int j, double v) {
        const int r = brow[k][i], c = tcol_[3 * J + j];
        if (r >= 0 && c >= 0) D(r, c - nu_) -= v;
      });
    }
    const std::vector<double> hB = H_->dofSize();
    H_->releaseQuadrature();
    for (const auto& e : ht) {
      const int r = rowinv_[e.row()], c = ucol_[e.col()];
      if (r >= 0 && c >= 0) trip.emplace_back(r, c, e.value());
    }
    for (const auto& e : gt) {
      const int r = rowinv_[e.row()], c = tcol_[e.col()];
      if (r >= 0 && c >= 0) trip.emplace_back(r, c, -e.value());
    }
    if (regions_) interfaceTriplets(gt, trip);
    std::vector<Eigen::Triplet<double>>().swap(ht);
    std::vector<Eigen::Triplet<double>>().swap(gt);
    for (size_t i = 0; i < spring_dofs_.size(); ++i) {
      const int B = spring_dofs_[i];
      for (int r = 0; r < n_; ++r) {
        const Vec3& x = H_->rowPoints()[rowsel_[r] / 3].x;
        if ((x - mesh_.dof_point[B]).norm() >= o_.radius * hB[B]) continue;
        for (int j = 0; j < 3; ++j) {
          const int c = ucol_[3 * B + j];
          if (c >= 0) trip.emplace_back(r, c, S_(r, 3 * i + j));
        }
      }
    }
    double s = 0.0;
    for (const auto& e : trip)
      if (e.row() == e.col()) s = std::max(s, std::abs(e.value()));
    m_ = int(Cu.cols());
    Bt_ = Eigen::MatrixXd::Zero(n_, m_);
    Bt_.topRows(nu_) = s * Cu;
    SparseBasis Zr, Zc;
    Eigen::MatrixXf AZ;
    if (o_.coarse > 0) {
      const auto tc = std::chrono::steady_clock::now();
      std::vector<Vec3> xt(n_), xr(n_);
      std::vector<int> comp(n_), group(n_);
      for (int k = 0; k < n_; ++k) {
        xr[k] = H_->rowPoints()[rowsel_[k] / 3].x;
        const bool u = k < nu_;
        const int j = u ? u_free_[k] : t_free_[k - nu_];
        xt[k] = u ? mesh_.dof_point[j / 3] : ts_.points[j / 3].x;
        comp[k] = j % 3;
        group[k] = u ? 0 : 1;
      }
      affineCoarse(xt, xr, comp, group, o_.coarse, Zc, Zr);
      AZ = productCoarse(Zc);
      st.t_coarse = since(tc);
    }
    pc_.build(n_, trip, Bt_, Bt_.transpose(), std::move(Zr), std::move(Zc), std::move(AZ), std::move(D));
    st.nnz = pc_.nonZeros();
    st.coarse_dim = pc_.coarseDim();
    st.t_precond = since(t0);
  }

  // G t - H u over the system rows (u: 3 ndof x K, t: 3 ts.ndof x K, t nonzero on stored columns only).
  Eigen::MatrixXd data(const Eigen::MatrixXd& u, const Eigen::MatrixXd& t) const {
    Eigen::MatrixXd out(n_, u.cols());
    Eigen::VectorXd y(H_->rows());
    for (int k = 0; k < u.cols(); ++k) {
      const Eigen::VectorXd uk = u.col(k), tk = t.col(k);
      H_->apply(uk.data(), tk.data(), y.data());
      for (int r = 0; r < n_; ++r) out(r, k) = -y(rowsel_[r]);
    }
    return out;
  }

  // Unknowns (n + m) x K of the bordered system for right-hand sides (n + m) x K.
  Eigen::MatrixXd solve(const Eigen::MatrixXd& B) const {
    const auto t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd X(n_ + m_, B.cols());
    auto A = [&](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
      const auto q = std::chrono::steady_clock::now();
      product(x, y);
      st.t_matvec += since(q);
    };
    auto P = [&](const Eigen::VectorXd& v, Eigen::VectorXd& z) {
      const auto q = std::chrono::steady_clock::now();
      pc_.apply(v, z);
      st.t_pc += since(q);
    };
    for (int k = 0; k < B.cols(); ++k) {
      Eigen::VectorXd x;
      double res;
      st.iterations.push_back(gmresRight(A, P, B.col(k), x, o_.gtol, res));
      st.residuals.push_back(res);
      X.col(k) = x;
    }
    st.t_solve += since(t0);
    return X;
  }

  mutable Stats st;

 private:
  const Mesh& mesh_;
  const TractionSpace& ts_;
  const std::vector<int>& rowsel_;
  const std::vector<int>& u_free_;
  const std::vector<int>& t_free_;
  const MultiRegion* regions_ = nullptr;
  HMixedOptions o_;
  std::unique_ptr<ElemHMatrix> H_;
  int n_ = 0, nu_ = 0, m_ = 0;
  std::vector<int> rowinv_, ucol_, tcol_;
  Eigen::MatrixXd S_, Bt_;
  std::vector<int> spring_dofs_;
  TwoLevel pc_;

  static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  // Interface tractions of the inclusions from the host displacement: t(tc) = -S_I u(uc).
  void interfaceTraction(const Eigen::VectorXd& u, Eigen::VectorXd& t) const {
    for (const Inclusion& I : regions_->inc) {
      Eigen::VectorXd ui(3 * I.u_host.size());
      for (size_t g = 0; g < I.u_host.size(); ++g) ui.segment<3>(3 * g) = u.segment<3>(3 * I.u_host[g]);
      const Eigen::VectorXd ti = I.S * ui;
      for (size_t j = 0; j < I.t_host.size(); ++j) t.segment<3>(3 * I.t_host[j]) = -ti.segment<3>(3 * j);
    }
  }

  // y = A x (bordered).
  void product(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(3 * mesh_.ndof), t = Eigen::VectorXd::Zero(3 * ts_.ndof);
    for (int k = 0; k < nu_; ++k) u(u_free_[k]) = x(k);
    for (int k = nu_; k < n_; ++k) t(t_free_[k - nu_]) = x(k);
    if (regions_) interfaceTraction(u, t);
    Eigen::VectorXd full(H_->rows());
    H_->apply(u.data(), t.data(), full.data());
    y.resize(n_ + m_);
    for (int r = 0; r < n_; ++r) y(r) = full(rowsel_[r]);
    if (S_.size()) {
      Eigen::VectorXd us(S_.cols());
      for (size_t i = 0; i < spring_dofs_.size(); ++i) us.segment<3>(3 * i) = u.segment<3>(3 * spring_dofs_[i]);
      y.head(n_) += S_ * us;
    }
    if (m_) {
      y.head(n_) += Bt_ * x.tail(m_);
      y.tail(m_) = Bt_.transpose() * x.head(n_);
    }
  }

  // A Zc (n x nc, float): the operator on the coarse basis split into displacement and traction
  // parts (interface tractions -S_I Zu added to the traction part), plus the springs.
  Eigen::MatrixXf productCoarse(const SparseBasis& Zc) const {
    const int nc = Zc.nc;
    std::vector<std::vector<std::pair<int, double>>> zu(3 * mesh_.ndof), zt(3 * ts_.ndof);
    for (int k = 0; k < n_; ++k)
      for (int p = Zc.ptr[k]; p < Zc.ptr[k + 1]; ++p)
        (k < nu_ ? zu[u_free_[k]] : zt[t_free_[k - nu_]]).push_back({Zc.id[p], Zc.val[p]});
    if (regions_)
      for (const Inclusion& I : regions_->inc) {
        Eigen::MatrixXd Zi = Eigen::MatrixXd::Zero(3 * I.u_host.size(), nc);
        for (size_t g = 0; g < I.u_host.size(); ++g)
          for (int c = 0; c < 3; ++c)
            for (const auto& [id, v] : zu[3 * I.u_host[g] + c]) Zi(3 * g + c, id) += v;
        const Eigen::MatrixXd Ti = -(I.S * Zi);
        for (size_t j = 0; j < I.t_host.size(); ++j)
          for (int c = 0; c < 3; ++c)
            for (int id = 0; id < nc; ++id)
              if (Ti(3 * j + c, id) != 0.0) zt[3 * I.t_host[j] + c].push_back({id, Ti(3 * j + c, id)});
      }
    auto pack = [nc](const std::vector<std::vector<std::pair<int, double>>>& e) {
      SparseBasis Z;
      Z.n = int(e.size());
      Z.nc = nc;
      for (const auto& row : e) {
        for (const auto& [id, v] : row) { Z.id.push_back(id); Z.val.push_back(v); }
        Z.ptr.push_back(int(Z.id.size()));
      }
      return Z;
    };
    const SparseBasis Zu = pack(zu), Zt = pack(zt);
    const Eigen::MatrixXf full = H_->applyCoarse(Zu, &Zt);
    Eigen::MatrixXf AZ(n_, nc);
    for (int r = 0; r < n_; ++r) AZ.row(r) = full.row(rowsel_[r]);
    if (S_.size()) {
      Eigen::MatrixXd Zs = Eigen::MatrixXd::Zero(S_.cols(), nc);
      for (size_t i = 0; i < spring_dofs_.size(); ++i)
        for (int c = 0; c < 3; ++c)
          for (const auto& [id, v] : zu[3 * spring_dofs_[i] + c]) Zs(3 * i + c, id) += v;
      AZ += (S_ * Zs).cast<float>();
    }
    return AZ;
  }

  // Preconditioner entries of the condensed interfaces: G(r, tc) S_I on the inclusion's host dofs.
  void interfaceTriplets(const std::vector<Eigen::Triplet<double>>& gt, std::vector<Eigen::Triplet<double>>& trip) const {
    for (const Inclusion& I : regions_->inc) {
      std::vector<int> trow(3 * ts_.ndof, -1);  // host traction comp -> row of S_I
      for (size_t j = 0; j < I.t_host.size(); ++j)
        for (int c = 0; c < 3; ++c) trow[3 * I.t_host[j] + c] = 3 * int(j) + c;
      std::vector<std::vector<std::pair<int, double>>> byRow(n_);
      for (const auto& e : gt) {
        const int r = rowinv_[e.row()];
        if (r >= 0 && trow[e.col()] >= 0) byRow[r].push_back({trow[e.col()], e.value()});
      }
      for (int r = 0; r < n_; ++r) {
        if (byRow[r].empty()) continue;
        Eigen::RowVectorXd v = Eigen::RowVectorXd::Zero(I.S.cols());
        for (const auto& [i, g] : byRow[r]) v += g * I.S.row(i);
        for (size_t g = 0; g < I.u_host.size(); ++g)
          for (int c = 0; c < 3; ++c) {
            const int col = ucol_[3 * I.u_host[g] + c];
            if (col >= 0 && v(3 * g + c) != 0.0) trip.emplace_back(r, col, v(3 * g + c));
          }
      }
    }
  }
};

}  // namespace igabem
