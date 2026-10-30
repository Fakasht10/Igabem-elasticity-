// Traction-only (Neumann) problem on the hierarchical operator (ElemHMatrix):
// the bordered system [H sR; sR^T 0] x = [G C; 0] (rigid modes as Lagrange
// border, as DenseNeumann) solved by right-preconditioned restarted GMRES.
//
// Preconditioner: sparse LU (UMFPACK) of the near field of H restricted to the
// pairs (point a, control point B) with |x_a - P_B| < radius h_B, h_B the
// largest element diameter in the support of B (bemkit: 1.5 h); the six border
// unknowns are left unpreconditioned. Coarse correction (bemkit): 12 affine modes
// per aggregate of at most `coarse` points, bordered with the rigid modes,
// Eb = Zb^T A Zb with A Z kept in float; applied multiplicatively before the LU,
// z = q + LU (r - A q), q = Zb Eb^-1 Zb^T r (no extra matvec).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <sys/resource.h>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

#include "../sbie.hpp"
#include "elem_hmatrix.hpp"

namespace igabem {

// Restarted GMRES (modified Gram-Schmidt), right preconditioning.
inline int gmresRight(const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& A,
                      const std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>& P, const Eigen::VectorXd& b,
                      Eigen::VectorXd& x, double tol, double& res, int restart = 80, int maxit = 1000) {
  const int n = int(b.size());
  x = Eigen::VectorXd::Zero(n);
  const double nb = b.norm();
  Eigen::VectorXd r = b, z, w;
  Eigen::MatrixXd V(n, restart + 1);
  int it = 0;
  res = 1.0;
  while (it < maxit) {
    const double beta = r.norm();
    res = beta / nb;
    if (res < tol) break;
    Eigen::MatrixXd Hh = Eigen::MatrixXd::Zero(restart + 1, restart);
    Eigen::VectorXd cs(restart), sn(restart), gv = Eigen::VectorXd::Zero(restart + 1);
    V.col(0) = r / beta;
    gv(0) = beta;
    int k = 0;
    for (; k < restart && it < maxit; ++k, ++it) {
      P(V.col(k), z);
      A(z, w);
      for (int i = 0; i <= k; ++i) { Hh(i, k) = V.col(i).dot(w); w -= Hh(i, k) * V.col(i); }
      Hh(k + 1, k) = w.norm();
      V.col(k + 1) = w / Hh(k + 1, k);
      for (int i = 0; i < k; ++i) {
        const double t = cs(i) * Hh(i, k) + sn(i) * Hh(i + 1, k);
        Hh(i + 1, k) = -sn(i) * Hh(i, k) + cs(i) * Hh(i + 1, k);
        Hh(i, k) = t;
      }
      const double h = std::hypot(Hh(k, k), Hh(k + 1, k));
      cs(k) = Hh(k, k) / h;
      sn(k) = Hh(k + 1, k) / h;
      Hh(k, k) = h;
      Hh(k + 1, k) = 0.0;
      gv(k + 1) = -sn(k) * gv(k);
      gv(k) = cs(k) * gv(k);
      if (std::abs(gv(k + 1)) / nb < tol) { ++k; ++it; break; }
    }
    const Eigen::VectorXd y = Hh.topLeftCorner(k, k).triangularView<Eigen::Upper>().solve(gv.head(k));
    P(V.leftCols(k) * y, z);
    x += z;
    A(x, r);
    r = b - r;
    res = r.norm() / nb;
  }
  return it;
}

struct HNeumannOptions {
  ElemHOptions h;
  double gtol = 1e-8;     // GMRES relative residual
  double radius = 1.0;    // preconditioner pattern radius, in local element diameters (1.5 without the coarse level)
  int coarse = 256;       // coarse correction aggregate (points; 0: off)
};

class HNeumann {
 public:
  struct Times {
    double build = 0.0, precond = 0.0, solve = 0.0;
    long precond_nnz = 0;
    double t_pattern = 0.0, coarse = 0.0;
    int coarse_dim = 0;
    double rss_build = 0.0, rss_precond = 0.0;
    double quad_GB = 0.0;  // far-rule cache, released before the LU  // peak RSS (GB) after the operator, after the LU
  };

  HNeumann(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& sopt,
           const Eigen::MatrixXd& C, const HNeumannOptions& o = HNeumannOptions())
      : mesh_(m), o_(o) {
    auto t0 = std::chrono::steady_clock::now();
    H_ = std::make_unique<ElemHMatrix>(m, ts, mat, sopt, C, o.h);
    times.build = since(t0);
    times.rss_build = peakGB();
    N_ = 3 * m.ndof;
    R_ = rigidModes(m);
    // border scale: a typical diagonal magnitude
    t0 = std::chrono::steady_clock::now();
    std::vector<Eigen::Triplet<double>> trip;
    if (o.h.adm_min) H_->exactTriplets(o.radius, trip);
    else H_->preconTriplets(o.radius, trip);
    times.t_pattern = since(t0);
    times.quad_GB = H_->quadratureBytes() / 1e9;
    H_->releaseQuadrature();
    Eigen::SparseMatrix<double>& S = S_;  // UmfPackLU keeps a reference to it
    S.resize(N_, N_);
    S.setFromTriplets(trip.begin(), trip.end());
    S.makeCompressed();
    times.precond_nnz = S.nonZeros();
    double dmax = 0.0;
    for (int i = 0; i < N_; ++i) dmax = std::max(dmax, std::abs(S.coeff(i, i)));
    s_ = dmax;
    lu_.compute(S);
    if (lu_.info() != Eigen::Success) throw std::runtime_error("HNeumann: preconditioner factorization failed");
    times.precond = since(t0);
    times.rss_precond = peakGB();
    if (o.coarse > 0) {
      t0 = std::chrono::steady_clock::now();
      nc_ = H_->coarseSpace(o.coarse, cid_, zr_, zc_);
      HZ_ = H_->applyCoarse(cid_, zc_, nc_);
      Eigen::MatrixXd E = Eigen::MatrixXd::Zero(nc_ + 6, nc_ + 6);
#pragma omp parallel for schedule(static)
      for (int j0 = 0; j0 < nc_; j0 += 64) {
        const int nj = std::min(64, nc_ - j0);
        for (int g = 0; g < N_; ++g)
          for (int f = 0; f < 4; ++f)
            if (cid_[4 * g + f] >= 0)
              E.row(cid_[4 * g + f]).segment(j0, nj) += zr_[4 * g + f] * HZ_.row(g).segment(j0, nj).cast<double>();
      }
      for (int g = 0; g < N_; ++g)
        for (int f = 0; f < 4; ++f)
          if (cid_[4 * g + f] >= 0) {
            E.row(cid_[4 * g + f]).tail(6) += s_ * zr_[4 * g + f] * R_.row(g);
            E.col(cid_[4 * g + f]).tail(6) += s_ * zc_[4 * g + f] * R_.row(g).transpose();
          }
      Elu_.compute(E);
      times.coarse = since(t0);
      times.coarse_dim = nc_;
    }
  }

  int unknowns() const { return N_; }
  const ElemHMatrix& op() const { return *H_; }

  // Displacement coefficients (3 M x K); iterations per case.
  Eigen::MatrixXd solve(std::vector<int>* iterations = nullptr, std::vector<double>* residuals = nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    const Eigen::MatrixXd& GC = H_->gc();
    Eigen::MatrixXd U(N_, GC.cols());
    auto A = [&](const Eigen::VectorXd& x, Eigen::VectorXd& y) {
      y.resize(N_ + 6);
      H_->apply(x.data(), y.data());
      y.head(N_) += s_ * (R_ * x.tail(6));
      y.tail(6) = s_ * (R_.transpose() * x.head(N_));
    };
    auto P = [&](const Eigen::VectorXd& v, Eigen::VectorXd& z) {
      z.resize(N_ + 6);
      if (nc_ == 0) {
        z.head(N_) = lu_.solve(v.head(N_));
        z.tail(6) = v.tail(6);
        return;
      }
      Eigen::VectorXd e = Eigen::VectorXd::Zero(nc_ + 6);  // Zb^T v
      for (int g = 0; g < N_; ++g)
        for (int f = 0; f < 4; ++f)
          if (cid_[4 * g + f] >= 0) e(cid_[4 * g + f]) += zr_[4 * g + f] * v(g);
      e.tail(6) = v.tail(6);
      const Eigen::VectorXd y = Elu_.solve(e);
      Eigen::VectorXd q = Eigen::VectorXd::Zero(N_);  // Zc y
      for (int g = 0; g < N_; ++g)
        for (int f = 0; f < 4; ++f)
          if (cid_[4 * g + f] >= 0) q(g) += zc_[4 * g + f] * y(cid_[4 * g + f]);
      const Eigen::VectorXf yf = y.head(nc_).cast<float>();
      const Eigen::VectorXd w = v.head(N_) - (HZ_ * yf).cast<double>() - s_ * (R_ * y.tail(6));
      z.head(N_) = q + lu_.solve(w);
      z.tail(6) = y.tail(6) + v.tail(6) - s_ * (R_.transpose() * q);
    };
    for (int k = 0; k < GC.cols(); ++k) {
      Eigen::VectorXd b = Eigen::VectorXd::Zero(N_ + 6), x;
      b.head(N_) = GC.col(k);
      double res;
      const int it = gmresRight(A, P, b, x, o_.gtol, res);
      if (iterations) iterations->push_back(it);
      if (residuals) residuals->push_back(res);
      U.col(k) = x.head(N_);
    }
    times.solve = since(t0);
    return U;
  }

  Times times;

 private:
  const Mesh& mesh_;
  HNeumannOptions o_;
  std::unique_ptr<ElemHMatrix> H_;
  int N_;
  Eigen::MatrixXd R_;
  double s_ = 1.0;
  Eigen::SparseMatrix<double> S_;
  Eigen::UmfPackLU<Eigen::SparseMatrix<double>> lu_;
  int nc_ = 0;  // coarse correction
  std::vector<int> cid_;
  std::vector<double> zr_, zc_;
  Eigen::MatrixXf HZ_;
  Eigen::PartialPivLU<Eigen::MatrixXd> Elu_;

  static double peakGB() {
    rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_maxrss / 1048576.0;
  }
  static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }
};

}  // namespace igabem
