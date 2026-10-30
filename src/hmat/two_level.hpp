// Two-level preconditioner of the hierarchical solvers (bemkit's): sparse LU
// (UMFPACK) of the near-field pattern, preceded multiplicatively by an affine
// coarse correction on the bordered system
//
//   [A   Bt] [x]      Bt: n x m, Bb: m x n (free rigid modes; m may be 0)
//   [Bb  0 ] [l]
//
// z = q + M (r - Ab q), q = Zb Eb^-1 Zb^T r, Zb = diag(Z, I_m), Eb = Zb^T Ab Zb,
// with A Zc given (float, from the hierarchical operator), so applying it costs
// one LU solve and one n x nc product, no extra matvec. The border unknowns are
// not preconditioned beyond the coarse level. Also the restarted GMRES.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "coarse.hpp"

namespace igabem {

using VecOp = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

// Restarted GMRES (modified Gram-Schmidt), right preconditioning.
inline int gmresRight(const VecOp& A, const VecOp& P, const Eigen::VectorXd& b, Eigen::VectorXd& x, double tol,
                      double& res, int restart = 80, int maxit = 1000) {
  const int n = int(b.size());
  x = Eigen::VectorXd::Zero(n);
  const double nb = b.norm();
  res = 0.0;
  if (nb == 0.0) return 0;
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

class TwoLevel {
 public:
  // trip: near pattern of A (n x n). Zr, Zc: coarse basis (nc = 0: LU only), AZ = A Zc (n x nc).
  // D (nd x nd, may be empty): the last nd unknowns form a dense block, factored apart (dense LU)
  // and coupled to the sparse rest by its rows (forward block Gauss-Seidel: on the bracket fewer
  // iterations than the symmetric sweep, at one sparse solve); pattern entries in it are added.
  void build(int n, std::vector<Eigen::Triplet<double>>& trip, const Eigen::MatrixXd& Bt, const Eigen::MatrixXd& Bb,
             SparseBasis Zr, SparseBasis Zc, Eigen::MatrixXf AZ, Eigen::MatrixXd D = Eigen::MatrixXd()) {
    n_ = n;
    m_ = int(Bt.cols());
    nd_ = int(D.rows());
    const int n0 = n - nd_;
    Bt_ = Bt;
    Bb_ = Bb;
    if (nd_) {
      std::vector<Eigen::Triplet<double>> lo;
      size_t k = 0;
      for (const auto& e : trip) {
        if (e.row() >= n0 && e.col() >= n0) D(e.row() - n0, e.col() - n0) += e.value();
        else if (e.row() >= n0) lo.emplace_back(e.row() - n0, e.col(), e.value());
        else if (e.col() < n0) trip[k++] = e;  // upper coupling: unused by the forward sweep
      }
      trip.resize(k);
      L_.resize(nd_, n0);
      L_.setFromTriplets(lo.begin(), lo.end());
      Dlu_.compute(D);
    }
    S_.resize(n0, n0);
    S_.setFromTriplets(trip.begin(), trip.end());
    S_.makeCompressed();
    std::vector<Eigen::Triplet<double>>().swap(trip);
    lu_.compute(S_);
    if (lu_.info() != Eigen::Success) throw std::runtime_error("TwoLevel: near-field factorization failed");
    nc_ = Zc.nc;
    if (!nc_) return;
    Zr_ = std::move(Zr);
    Zc_ = std::move(Zc);
    AZ_ = std::move(AZ);
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(nc_ + m_, nc_ + m_);
#pragma omp parallel for schedule(static)
    for (int j0 = 0; j0 < nc_; j0 += 64) {
      const int nj = std::min(64, nc_ - j0);
      for (int i = 0; i < n_; ++i)
        for (int p = Zr_.ptr[i]; p < Zr_.ptr[i + 1]; ++p)
          E.row(Zr_.id[p]).segment(j0, nj) += Zr_.val[p] * AZ_.row(i).segment(j0, nj).cast<double>();
    }
    for (int i = 0; i < n_; ++i) {
      for (int p = Zr_.ptr[i]; p < Zr_.ptr[i + 1]; ++p) E.row(Zr_.id[p]).tail(m_) += Zr_.val[p] * Bt_.row(i);
      for (int p = Zc_.ptr[i]; p < Zc_.ptr[i + 1]; ++p) E.col(Zc_.id[p]).tail(m_) += Zc_.val[p] * Bb_.col(i);
    }
    Elu_.compute(E);
  }

  long nonZeros() const { return S_.nonZeros() + L_.nonZeros() + long(nd_) * nd_; }
  int coarseDim() const { return nc_; }
  double maxDiag() const {
    double d = 0.0;
    for (int i = 0; i < n_ - nd_; ++i) d = std::max(d, std::abs(S_.coeff(i, i)));
    return d;
  }

  // Near-field solve: the sparse LU, then the dense block (forward block Gauss-Seidel).
  Eigen::VectorXd nearSolve(const Eigen::VectorXd& v) const {
    if (!nd_) return lu_.solve(v);
    const int n0 = n_ - nd_;
    Eigen::VectorXd z(n_), r = v.head(n0);
    z.head(n0) = lu_.solve(r);
    z.tail(nd_) = Dlu_.solve(v.tail(nd_) - L_ * z.head(n0));
    return z;
  }

  void apply(const Eigen::VectorXd& v, Eigen::VectorXd& z) const {
    z.resize(n_ + m_);
    if (!nc_) {
      z.head(n_) = nearSolve(v.head(n_));
      z.tail(m_) = v.tail(m_);
      return;
    }
    Eigen::VectorXd e = Eigen::VectorXd::Zero(nc_ + m_);
    Zr_.applyT(v.data(), e.data());
    e.tail(m_) = v.tail(m_);
    const Eigen::VectorXd y = Elu_.solve(e);
    Eigen::VectorXd q(n_);
    Zc_.apply(y.data(), q.data());
    const Eigen::VectorXf yf = y.head(nc_).cast<float>();
    Eigen::VectorXd w = v.head(n_) - (AZ_ * yf).cast<double>();
    if (m_) w -= Bt_ * y.tail(m_);
    z.head(n_) = q + nearSolve(w);
    z.tail(m_) = y.tail(m_) + v.tail(m_);
    if (m_) z.tail(m_) -= Bb_ * q;
  }

 private:
  int n_ = 0, m_ = 0, nc_ = 0, nd_ = 0;
  Eigen::SparseMatrix<double> L_;  // dense-block rows on the sparse columns
  Eigen::PartialPivLU<Eigen::MatrixXd> Dlu_;
  Eigen::MatrixXd Bt_, Bb_;
  Eigen::SparseMatrix<double> S_;  // UmfPackLU keeps a reference to it
  Eigen::UmfPackLU<Eigen::SparseMatrix<double>> lu_;
  SparseBasis Zr_, Zc_;
  Eigen::MatrixXf AZ_;
  Eigen::PartialPivLU<Eigen::MatrixXd> Elu_;
};

}  // namespace igabem
