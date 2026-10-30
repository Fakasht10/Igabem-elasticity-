// Traction-only (Neumann) problem on the hierarchical operator (ElemHMatrix):
// the bordered system [H sR; sR^T 0] x = [G C; 0] (rigid modes as Lagrange
// border, as DenseNeumann) solved by right-preconditioned restarted GMRES.
//
// Preconditioner (two_level.hpp): sparse LU (UMFPACK) of the near field of H
// restricted to the pairs (point a, control point B) with |x_a - P_B| < radius
// h_B, h_B the largest element diameter in the support of B, after an affine
// coarse correction (12 modes per aggregate of at most `coarse` points, bordered
// with the rigid modes; bemkit's).
#pragma once

#include <sys/resource.h>

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <vector>

#include "../sbie.hpp"
#include "elem_hmatrix.hpp"
#include "two_level.hpp"

namespace igabem {

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
    double rss_build = 0.0, rss_precond = 0.0;  // peak RSS (GB) after the operator, after the LU
    double quad_GB = 0.0;  // far-rule cache, released before the LU
  };

  HNeumann(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& sopt,
           const Eigen::MatrixXd& C, const HNeumannOptions& o = HNeumannOptions())
      : o_(o) {
    auto t0 = std::chrono::steady_clock::now();
    H_ = std::make_unique<ElemHMatrix>(m, ts, mat, sopt, C, o.h);
    times.build = since(t0);
    times.rss_build = peakGB();
    N_ = 3 * m.ndof;
    R_ = rigidModes(m);
    t0 = std::chrono::steady_clock::now();
    std::vector<Eigen::Triplet<double>> trip;
    if (o.h.adm_min) H_->exactTriplets(o.radius, trip);
    else H_->preconTriplets(o.radius, trip);
    times.t_pattern = since(t0);
    times.quad_GB = H_->quadratureBytes() / 1e9;
    H_->releaseQuadrature();
    // border scale: the largest diagonal entry of the pattern
    s_ = 0.0;
    for (const auto& e : trip)
      if (e.row() == e.col()) s_ = std::max(s_, std::abs(e.value()));
    SparseBasis Zr, Zc;
    Eigen::MatrixXf AZ;
    if (o.coarse > 0) {
      const auto tc = std::chrono::steady_clock::now();
      std::vector<Vec3> xt(N_), xr(N_);
      std::vector<int> comp(N_), group(N_, 0);
      for (int g = 0; g < m.ndof; ++g)
        for (int c = 0; c < 3; ++c) {
          xt[3 * g + c] = m.dof_point[g];
          xr[3 * g + c] = m.colloc[g].x;
          comp[3 * g + c] = c;
        }
      affineCoarse(xt, xr, comp, group, o.coarse, Zc, Zr);
      AZ = H_->applyCoarse(Zc, nullptr);
      times.coarse = since(tc);
    }
    pc_.build(N_, trip, s_ * R_, s_ * R_.transpose(), std::move(Zr), std::move(Zc), std::move(AZ));
    times.precond_nnz = pc_.nonZeros();
    times.coarse_dim = pc_.coarseDim();
    times.precond = since(t0);
    times.rss_precond = peakGB();
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
    auto P = [&](const Eigen::VectorXd& v, Eigen::VectorXd& z) { pc_.apply(v, z); };
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
  HNeumannOptions o_;
  std::unique_ptr<ElemHMatrix> H_;
  int N_;
  Eigen::MatrixXd R_;
  double s_ = 1.0;
  TwoLevel pc_;

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
