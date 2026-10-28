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
#include "traction_space.hpp"

namespace igabem {

// Assembles A (3M x 3M) and b for the interior Neumann problem with
// prescribed traction t(y) = sigma n(y).
// Traction data of up to 8 load cases at one point, on the stack (no heap
// allocation in the quadrature loop).
using TractionBlock = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor, 3, 8>;

class SbieAssembler {
 public:
  SbieAssembler(const Mesh& m, const Material& mat, const SolverOptions& o)
      : mesh_(m), K_(mat), quad_(m, o) {}

  void assemble(const Eigen::Matrix3d& sigma, Eigen::MatrixXd& A, Eigen::VectorXd& b) const {
    assembleTraction([&](const Vec3&, const Vec3& n) { return Vec3(sigma * n); }, A, b);
  }

  // General traction data t(y, n(y)), n the unit normal outward from the solid.
  template <class F>
  void assembleTraction(F&& traction, Eigen::MatrixXd& A, Eigen::VectorXd& b) const {
    Eigen::MatrixXd B;
    assembleTractions(
        [&](const Vec3& y, const Vec3& n) { return TractionBlock(traction(y, n)); },
        1, A, B);
    b = B.col(0);
  }

  // Several load cases at once (K <= 8): traction(y, n) returns a 3 x K TractionBlock, one
  // column per case; B gets one right-hand side per column. The matrix depends
  // only on the mesh, so it is assembled once for all cases.
  template <class F>
  void assembleTractions(F&& traction, int K, Eigen::MatrixXd& A, Eigen::MatrixXd& B) const {
    const int M = mesh_.ndof;
    if (K > 8) throw std::runtime_error("assembleTractions: at most 8 load cases");
    A.setZero(3 * M, 3 * M);
    B.setZero(3 * M, K);
    // Traction data at cached far-field points, evaluated once for all rows.
    std::vector<TractionBlock> Tc(quad_.numCached());
#pragma omp parallel for schedule(static)
    for (long id = 0; id < quad_.numCached(); ++id) {
      const Vec3 nJ = quad_.cachedNJ(id) * mesh_.normal_sign;
      Tc[id] = traction(quad_.cachedX(id), Vec3(nJ / nJ.norm()));
    }
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < int(mesh_.colloc.size()); ++a) {
      const Collocation& c = mesh_.colloc[a];
      const SourcePoint src{c.patch, c.xi, c.x};
      Eigen::MatrixXd Arow = Eigen::MatrixXd::Zero(3, 3 * M);
      TractionBlock brow = TractionBlock::Zero(3, K);
      Eigen::Matrix3d Tsum = Eigen::Matrix3d::Zero();  // times N_B(x) afterwards
      for (const auto& e : mesh_.elements) {
        const auto& l2g = mesh_.l2g[e.patch];
        quad_.integrate(src, e, [&](const QPoint& E, double w) {
          const Vec3 nJ = E.normal() * mesh_.normal_sign;
          const double J = nJ.norm();
          const Vec3 n = nJ / J;
          Eigen::Matrix3d U, T;
          K_.eval(E.x - c.x, n, U, T);
          const Eigen::Matrix3d Tw = T * (w * J);
          for (int l = 0; l < E.n_loc; ++l) Arow.block<3, 3>(0, 3 * l2g[E.idx[l]]) += E.R[l] * Tw;
          Tsum += Tw;
          if (E.id >= 0) brow.noalias() += U * Tc[E.id] * (w * J);
          else brow.noalias() += U * traction(E.x, n) * (w * J);
        });
      }
      for (const auto& [Bg, NB] : c.basis) Arow.block<3, 3>(0, 3 * Bg) -= NB * Tsum;
      A.middleRows(3 * a, 3) = Arow;
      B.middleRows(3 * a, 3) = brow;
    }
  }

  // Both SBIE operators for any boundary data: H (the matrix above, acting on
  // displacement coefficients) and G (acting on coefficients of the traction
  // space ts), so that H u = G t. Assembled once per mesh; every load case and,
  // later, every mixed boundary condition is a choice of columns and a
  // right-hand side.
  void assembleHG(const TractionSpace& ts, Eigen::MatrixXd& H, Eigen::MatrixXd& G) const {
    assembleHG(mesh_.colloc, ts, H, G);
  }

  // The same rows at any set of collocation points on the boundary (3 rows each),
  // e.g. the traction collocation points of Dirichlet patches (mixed problems).
  void assembleHG(const std::vector<Collocation>& rows, const TractionSpace& ts, Eigen::MatrixXd& H,
                  Eigen::MatrixXd& G) const {
    const int M = mesh_.ndof, Nt = ts.ndof;
    H.setZero(3 * int(rows.size()), 3 * M);
    G.setZero(3 * int(rows.size()), 3 * Nt);
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < int(rows.size()); ++a) {
      const Collocation& c = rows[a];
      const SourcePoint src{c.patch, c.xi, c.x};
      Eigen::MatrixXd Hrow = Eigen::MatrixXd::Zero(3, 3 * M);
      Eigen::MatrixXd Grow = Eigen::MatrixXd::Zero(3, 3 * Nt);
      Eigen::Matrix3d Tsum = Eigen::Matrix3d::Zero();
      for (const auto& e : mesh_.elements) {
        const auto& l2g = mesh_.l2g[e.patch];
        const int off = ts.offset[e.patch];
        quad_.integrate(src, e, [&](const QPoint& E, double w) {
          const Vec3 nJ = E.normal() * mesh_.normal_sign;
          const double J = nJ.norm();
          Eigen::Matrix3d U, T;
          K_.eval(E.x - c.x, Vec3(nJ / J), U, T);
          const Eigen::Matrix3d Tw = T * (w * J), Uw = U * (w * J);
          for (int l = 0; l < E.n_loc; ++l) {
            Hrow.block<3, 3>(0, 3 * l2g[E.idx[l]]) += E.R[l] * Tw;
            Grow.block<3, 3>(0, 3 * (off + E.idx[l])) += E.R[l] * Uw;
          }
          Tsum += Tw;
        });
      }
      for (const auto& [Bg, NB] : c.basis) Hrow.block<3, 3>(0, 3 * Bg) -= NB * Tsum;
      H.middleRows(3 * a, 3) = Hrow;
      G.middleRows(3 * a, 3) = Grow;
    }
  }

 private:
  const Mesh& mesh_;
  Kelvin K_;
  ElementIntegrator quad_;
};

// SBIE collocation points at the traction collocation points of one patch
// (interior to the patch), with the displacement basis there.
inline std::vector<Collocation> tractionCollocation(const Mesh& m, const TractionSpace& ts, int patch) {
  std::vector<Collocation> out;
  const int n0 = ts.offset[patch];
  const int n1 = patch + 1 < int(ts.offset.size()) ? ts.offset[patch + 1] : ts.ndof;
  for (int i = n0; i < n1; ++i) {
    const TractionSpace::Point& p = ts.points[i];
    Collocation c{p.patch, p.su, p.sv, p.xi, p.x, {}};
    for (int l = 0; l < p.E.n_loc; ++l)
      if (std::abs(p.E.R[l]) > 1e-15) c.basis.push_back({m.l2g[patch][p.E.idx[l]], p.E.R[l]});
    out.push_back(std::move(c));
  }
  return out;
}

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

// Relative L2(Gamma) error of u_h against an exact field uexact(x), after
// removing the L2-best rigid-body motion (undetermined by the pure Neumann
// problem; remove_rigid = false for problems with Dirichlet data).
template <class Fu>
inline double displacementL2Error(const Mesh& m, const Eigen::VectorXd& c, Fu&& uexact, int ng = 10,
                                  bool remove_rigid = true) {
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
          qp.push_back({E.x, uh - uexact(E.x), g.w[i] * g.w[j] * R.du() * R.dv() * E.normal().norm()});
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
  const Eigen::Matrix<double, 6, 1> beta =
      remove_rigid ? Eigen::Matrix<double, 6, 1>(G.ldlt().solve(r)) : Eigen::Matrix<double, 6, 1>::Zero();
  double num = 0.0, den = 0.0;
  for (const auto& q : qp) {
    num += q.w * (q.e - phi(q.x) * beta).squaredNorm();
    den += q.w * uexact(q.x).squaredNorm();
  }
  return std::sqrt(num / den);
}

inline double relativeL2Error(const Mesh& m, const Eigen::VectorXd& c,
                              const Eigen::Matrix3d& alpha, int ng = 10) {
  return displacementL2Error(m, c, [&](const Vec3& x) { return Vec3(alpha * x); }, ng);
}

}  // namespace igabem
