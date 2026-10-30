// Traction-only (Neumann) problem on dense operators, fast path: the bordered
// system [H sR; sR^T 0] assembled once by DenseSbie (no G formed: G c for the
// given load cases is applied during assembly) and solved by DenseGmres, one
// solve per load case against the same matrix and preconditioner.
//
// Same discretization as NeumannModel (rigid-body diagonal term, rigid modes
// as Lagrange border); entries equal to SbieEntries with the same entry rule.
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "../sbie.hpp"
#include "dense_gmres.hpp"
#include "dense_sbie.hpp"
#include "huge_buffer.hpp"

namespace igabem {

struct DenseNeumannOptions {
  double tol = 1e-7;    // far-field entry rule tolerance
  int min_order = -1;   // far-field minimum Gauss order, -1: from the basis degree
  double gtol = 1e-10;  // GMRES relative residual
  int blocks = 4;       // preconditioner blocks (bisection)
};

class DenseNeumann {
 public:
  // C: traction coefficients (3 Mt x K, TractionSpace::interpolate), K <= 8.
  DenseNeumann(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& opt,
               const Eigen::MatrixXd& C, const DenseNeumannOptions& o = DenseNeumannOptions())
      : mesh_(m), o_(o), N_(3 * m.ndof), buf_(size_t(N_ + 6) * (N_ + 6)) {
    const int N = N_, np = int(m.colloc.size());
    Eigen::Map<Eigen::MatrixXd> A(buf_.data(), N + 6, N + 6);
    DenseSbie ds(m, ts, mat, opt, o.tol, o.min_order);
    Eigen::MatrixXd GC, S = Eigen::MatrixXd::Zero(N, 3);
    DenseSbie::Out out;
    out.H = A.data();
    out.ldh = N + 6;
    out.C = &C;
    out.GC = &GC;
    out.S = S.data();
    ds.assemble(out);
    stats = ds.stats();
    // Rigid-body term H = Hraw - N_B(x_a) sum_B Hraw(a, B), and the rigid-mode border.
    for (int a = 0; a < np; ++a)
      for (const auto& [B, NB] : m.colloc[a].basis) A.block<3, 3>(3 * a, 3 * B) -= NB * S.block<3, 3>(3 * a, 0);
    const Eigen::MatrixXd R = rigidModes(m);
    const double s = A.topLeftCorner(N, N).cwiseAbs().maxCoeff();
    A.topRightCorner(N, 6) = s * R;
    A.bottomLeftCorner(6, N) = s * R.transpose();
    rhs_ = Eigen::MatrixXd::Zero(N + 6, C.cols());
    rhs_.topRows(N) = GC;
  }

  int unknowns() const { return N_; }
  Eigen::Map<const Eigen::MatrixXd> matrix() const { return {buf_.data(), N_ + 6, N_ + 6}; }

  // Displacement coefficients (3 M x K); iterations[k]: GMRES iterations of case k.
  Eigen::MatrixXd solve(std::vector<int>* iterations = nullptr) {
    if (!gmres_) gmres_ = std::make_unique<DenseGmres>(buf_.data(), N_ + 6, bisectionBlocks(mesh_, o_.blocks));
    Eigen::MatrixXd U(N_, rhs_.cols());
    if (iterations) iterations->clear();
    for (int k = 0; k < rhs_.cols(); ++k) {
      Eigen::VectorXd x;
      double res;
      const int it = gmres_->solve(rhs_.col(k), x, o_.gtol, res);
      if (iterations) iterations->push_back(it);
      U.col(k) = x.head(N_);
    }
    return U;
  }

  DenseSbie::Stats stats;

 private:
  const Mesh& mesh_;
  DenseNeumannOptions o_;
  int N_;
  HugeBuffer buf_;
  Eigen::MatrixXd rhs_;
  std::unique_ptr<DenseGmres> gmres_;
};

}  // namespace igabem
