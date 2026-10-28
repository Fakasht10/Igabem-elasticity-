// Collocation IgBEM for the displacement boundary integral equation (SBIE)
// of 3D elastostatics, interior problems, after Taus, Rodin, Hughes & Scott
// (CMAME 357, 2019).
//
// Regularized SBIE at a collocation point x (free term never formed, since
// sigma(x) = -PV int T dGamma for rigid translations):
//
//   int_Gamma T(x,y) [u(y) - u(x)] dGamma_y = int_Gamma U(x,y) t(y) dGamma_y
//
// Both integrands are O(1/r), so a Duffy (polar) split at x makes them smooth.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "integrator.hpp"
#include "kelvin.hpp"
#include "mesh.hpp"

namespace igabem {

// Assembles A (3M x 3M) and b for the interior Neumann problem with
// prescribed traction t(y) = sigma n(y).
class SbieAssembler {
 public:
  SbieAssembler(const Mesh& m, const Material& mat, const SolverOptions& o)
      : mesh_(m), K_(mat), quad_(m, o) {}

  void assemble(const Eigen::Matrix3d& sigma, Eigen::MatrixXd& A, Eigen::VectorXd& b) const {
    const int M = mesh_.ndof;
    A.setZero(3 * M, 3 * M);
    b.setZero(3 * M);
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < int(mesh_.colloc.size()); ++a) {
      const Collocation& c = mesh_.colloc[a];
      const SourcePoint src{c.patch, c.xi, c.x};
      Eigen::MatrixXd Arow = Eigen::MatrixXd::Zero(3, 3 * M);
      Eigen::Vector3d brow = Eigen::Vector3d::Zero();
      Eigen::Matrix3d Tsum = Eigen::Matrix3d::Zero();  // times N_B(x) afterwards
      for (const auto& e : mesh_.elements) {
        const auto& l2g = mesh_.l2g[e.patch];
        quad_.integrate(src, e, [&](const PatchEval& E, double w) {
          const Vec3 nJ = E.normal() * mesh_.normal_sign;
          const double J = nJ.norm();
          const Vec3 n = nJ / J;
          Eigen::Matrix3d U, T;
          K_.eval(E.x - c.x, n, U, T);
          const Eigen::Matrix3d Tw = T * (w * J);
          for (int l = 0; l < E.n_loc; ++l) Arow.block<3, 3>(0, 3 * l2g[E.idx[l]]) += E.R[l] * Tw;
          Tsum += Tw;
          brow += U * (sigma * n) * (w * J);
        });
      }
      for (const auto& [B, NB] : c.basis) Arow.block<3, 3>(0, 3 * B) -= NB * Tsum;
      A.middleRows(3 * a, 3) = Arow;
      b.segment<3>(3 * a) = brow;
    }
  }

 private:
  const Mesh& mesh_;
  Kelvin K_;
  ElementIntegrator quad_;
};

// Rigid-body modes in coefficient space: 3 translations, 3 rotations about 0.
inline Eigen::MatrixXd rigidModes(const Mesh& m) {
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3 * m.ndof, 6);
  for (int g = 0; g < m.ndof; ++g) {
    const Vec3& P = m.dof_point[g];
    for (int i = 0; i < 3; ++i) R(3 * g + i, i) = 1.0;
    for (int k = 0; k < 3; ++k) R.block<3, 1>(3 * g, 3 + k) = Vec3::Unit(k).cross(P);
  }
  for (int k = 0; k < 6; ++k) R.col(k).normalize();
  return R;
}

// Relative L2(Gamma) error of u_h against u = alpha x, after removing the
// L2-best rigid-body motion (undetermined by the pure Neumann problem).
inline double relativeL2Error(const Mesh& m, const Eigen::VectorXd& c,
                              const Eigen::Matrix3d& alpha, int ng = 10) {
  const GaussRule g = gaussLegendre(ng);
  struct QP { Vec3 x, e; double w; };
  std::vector<QP> qp;
  PatchEval E;
  for (const auto& el : m.elements)
    for (const auto& R : el.cells)
      for (int i = 0; i < ng; ++i)
        for (int j = 0; j < ng; ++j) {
          m.patches[el.patch].evalSpan(el.su, el.sv, R.u0 + R.du() * g.x[i],
                                       R.v0 + R.dv() * g.x[j], E);
          Vec3 uh = Vec3::Zero();
          for (int l = 0; l < E.n_loc; ++l)
            uh += E.R[l] * c.segment<3>(3 * m.l2g[el.patch][E.idx[l]]);
          qp.push_back({E.x, uh - alpha * E.x, g.w[i] * g.w[j] * R.du() * R.dv() * E.normal().norm()});
        }
  Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
  Eigen::Matrix<double, 6, 1> r = Eigen::Matrix<double, 6, 1>::Zero();
  auto phi = [](const Vec3& x) {
    Eigen::Matrix<double, 3, 6> F;
    F.leftCols<3>().setIdentity();
    for (int k = 0; k < 3; ++k) F.col(3 + k) = Vec3::Unit(k).cross(x);
    return F;
  };
  for (const auto& q : qp) {
    const auto F = phi(q.x);
    G += q.w * F.transpose() * F;
    r += q.w * F.transpose() * q.e;
  }
  const Eigen::Matrix<double, 6, 1> beta = G.ldlt().solve(r);
  double num = 0.0, den = 0.0;
  for (const auto& q : qp) {
    num += q.w * (q.e - phi(q.x) * beta).squaredNorm();
    den += q.w * (alpha * q.x).squaredNorm();
  }
  return std::sqrt(num / den);
}

}  // namespace igabem
