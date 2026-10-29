// Iterative solve of the dense bordered SBIE system [H sR; sR^T 0] x = b:
// restarted GMRES (modified Gram-Schmidt) with multithreaded dgemv matvecs and a
// right block-Jacobi preconditioner on a few large spatial blocks.
//
// Blocks: recursive bisection of the control points across their longest
// extent, so both faces of a thin plate stay in one block (blocks cut by patch
// or into small clusters let the iteration count grow from ~30 to 110-260 at
// t/W = 0.02). Each block is factored in double precision (dgetrf): with a
// single-precision factorization the iterations doubled (26 -> 52 at 17k). The
// 6 border unknowns are left unpreconditioned.
//
// Cost: a matvec reads the matrix once (0.065 s at 17k on 4 cores), the block
// factorization is (number of blocks)^-2 of a full LU.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "../mesh.hpp"

extern "C" {
void dgemv_(const char* t, const int* m, const int* n, const double* al, const double* a, const int* lda,
            const double* x, const int* incx, const double* be, double* y, const int* incy);
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* t, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b,
             const int* ldb, int* info);
}

namespace igabem {

// Unknowns (3 per control point) of nblocks spatial blocks (a power of 2) of
// the mesh's dofs, by recursive bisection at the median across the longest extent.
inline std::vector<std::vector<int>> bisectionBlocks(const Mesh& m, int nblocks) {
  std::vector<std::vector<int>> out;
  std::vector<int> all(m.ndof);
  for (int g = 0; g < m.ndof; ++g) all[g] = g;
  std::function<void(std::vector<int>&, int)> split = [&](std::vector<int>& v, int k) {
    if (k <= 1 || v.size() < 2) {
      out.emplace_back();
      for (int g : v)
        for (int i = 0; i < 3; ++i) out.back().push_back(3 * g + i);
      return;
    }
    Vec3 lo = Vec3::Constant(1e300), hi = -lo;
    for (int g : v) { lo = lo.cwiseMin(m.dof_point[g]); hi = hi.cwiseMax(m.dof_point[g]); }
    int d;
    (hi - lo).maxCoeff(&d);
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end(),
                     [&](int a, int b) { return m.dof_point[a](d) < m.dof_point[b](d); });
    std::vector<int> a(v.begin(), v.begin() + v.size() / 2), b(v.begin() + v.size() / 2, v.end());
    split(a, k / 2);
    split(b, k - k / 2);
  };
  split(all, nblocks);
  return out;
}

class DenseGmres {
 public:
  // A: n x n column-major (kept by reference), blocks: preconditioner index sets.
  DenseGmres(const double* A, int n, std::vector<std::vector<int>> blocks) : A_(A), n_(n), blocks_(std::move(blocks)) {
    lu_.resize(blocks_.size());
    piv_.resize(blocks_.size());
    for (size_t p = 0; p < blocks_.size(); ++p) {
      const std::vector<int>& I = blocks_[p];
      const int m = int(I.size());
      lu_[p].resize(size_t(m) * m);
      piv_[p].resize(m);
#pragma omp parallel for schedule(static)
      for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i) lu_[p][i + size_t(m) * j] = (A_[I[i] + size_t(n_) * I[j]]);
      int info = 0;
      dgetrf_(&m, &m, lu_[p].data(), &m, piv_[p].data(), &info);
    }
  }

  // Solves to a relative residual tol; returns the number of iterations (res: final relative residual).
  int solve(const Eigen::VectorXd& b, Eigen::VectorXd& x, double tol, double& res, int restart = 100,
            int maxit = 2000) const {
    const int n = n_;
    x = Eigen::VectorXd::Zero(n);
    const double nb = b.norm();
    Eigen::VectorXd r = b, z;
    Eigen::MatrixXd V(n, restart + 1);
    int it = 0;
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
        precondition(V.col(k), z);
        matvec(z.data(), V.col(k + 1).data());
        for (int i = 0; i <= k; ++i) { Hh(i, k) = V.col(i).dot(V.col(k + 1)); V.col(k + 1) -= Hh(i, k) * V.col(i); }
        Hh(k + 1, k) = V.col(k + 1).norm();
        V.col(k + 1) /= Hh(k + 1, k);
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
      precondition(V.leftCols(k) * y, z);
      x += z;
      matvec(x.data(), r.data());
      r = b - r;
    }
    res = r.norm() / nb;
    return it;
  }

  void matvec(const double* v, double* y) const {
    const int one = 1;
    const double d1 = 1.0, d0 = 0.0;
    dgemv_("N", &n_, &n_, &d1, A_, &n_, v, &one, &d0, y, &one);
  }

  void precondition(const Eigen::VectorXd& v, Eigen::VectorXd& z) const {
    z = v;
    for (size_t p = 0; p < blocks_.size(); ++p) {
      const std::vector<int>& I = blocks_[p];
      const int m = int(I.size()), one = 1;
      Eigen::VectorXd y(m);
      for (int i = 0; i < m; ++i) y(i) = v(I[i]);
      int info = 0;
      dgetrs_("N", &m, &one, lu_[p].data(), &m, piv_[p].data(), y.data(), &m, &info);
      for (int i = 0; i < m; ++i) z(I[i]) = y(i);
    }
  }

 private:
  const double* A_;
  int n_;
  std::vector<std::vector<int>> blocks_;
  std::vector<std::vector<double>> lu_;
  std::vector<std::vector<int>> piv_;
};

}  // namespace igabem
