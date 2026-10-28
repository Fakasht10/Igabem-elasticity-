// Collocation IgBEM for the traction boundary integral equation (HSBIE) of 3D
// elastostatics, interior Dirichlet problems.
//
// Regularization by subtracting the linear field that matches u and its full
// gradient at the collocation point x (classical "constant strain" or Taylor
// subtraction; it needs u in C^1 and t continuous at x). With
// u_L(y) = u(x) + grad u(x) (y - x) and t_L(y) = sigma(x) n(y), the identity
// t_L(x) = int D t_L - int S u_L holds exactly, and subtracting it removes the
// free term (t(x) = sigma(x) N):
//
//   int D(x,y) [t(y) - sigma(x) n(y)] dGamma = int S(x,y) [u(y) - u_L(y)] dGamma
//
// Both integrands are O(1/r). grad u(x) is assembled from the tangential
// derivatives of the Dirichlet data and the normal derivative implied by t(x)
// through Hooke's law, so the unknown t(x) enters both sides.
//
// This is a reconstruction: Taus, Rodin & Hughes (M3AS 2016) were not
// available. See README.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "integrator.hpp"
#include "kelvin.hpp"
#include "mesh.hpp"

namespace igabem {

// Traction space: the geometry's rational basis on each patch, discontinuous
// across patches (no merging), collocated at Greville points whose first and
// last entries are moved halfway inwards so that every point is interior to
// its patch.
struct TractionSpace {
  struct Point {
    int patch, su, sv;
    Vec2 xi;
    PatchEval E;  // geometry and basis at the point
    Vec3 x, N;
  };
  std::vector<int> offset;
  std::vector<Point> points;
  int ndof = 0;

  static double shifted(const Patch& P, int i, bool in_u) {
    const int n = in_u ? P.nu : P.nv;
    if (i == 0) return 0.5 * (P.greville(0, in_u) + P.greville(1, in_u));
    if (i == n - 1) return 0.5 * (P.greville(n - 2, in_u) + P.greville(n - 1, in_u));
    return P.greville(i, in_u);
  }

  void build(const Mesh& m) {
    offset.clear();
    points.clear();
    ndof = 0;
    for (size_t ip = 0; ip < m.patches.size(); ++ip) {
      const Patch& P = m.patches[ip];
      if (P.nu < 3 || P.nv < 3) throw std::runtime_error("traction space needs >= 3 cps per direction");
      offset.push_back(ndof);
      for (int k = 0; k < P.nu * P.nv; ++k) {
        Point c;
        c.patch = int(ip);
        c.xi = Vec2(shifted(P, k % P.nu, true), shifted(P, k / P.nu, false));
        c.su = findSpan(P.nu, P.p, c.xi(0), P.U);
        c.sv = findSpan(P.nv, P.q, c.xi(1), P.V);
        P.evalSpan(c.su, c.sv, c.xi(0), c.xi(1), c.E);
        c.x = c.E.x;
        c.N = (c.E.normal() * m.normal_sign).normalized();
        points.push_back(c);
      }
      ndof += P.nu * P.nv;
    }
  }
};

class HsbieAssembler {
 public:
  HsbieAssembler(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& o)
      : mesh_(m), ts_(ts), mat_(mat), K_(mat), quad_(m, o) {}

  // Full displacement gradient at a boundary point from the tangential
  // gradient Gt of the data and the traction t there.
  Eigen::Matrix3d gradient(const Eigen::Matrix3d& Gt, const Vec3& t, const Vec3& N) const {
    const double lam = mat_.lambda, mu = mat_.mu;
    const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - N * N.transpose();
    const Vec3 a = P * t / mu - P * (Gt.transpose() * N) +
                   N * (t.dot(N) - lam * Gt.trace()) / (lam + 2.0 * mu);
    return Gt + a * N.transpose();
  }

  Eigen::Matrix3d stress(const Eigen::Matrix3d& G) const {
    const Eigen::Matrix3d eps = 0.5 * (G + G.transpose());
    return mat_.lambda * eps.trace() * Eigen::Matrix3d::Identity() + 2.0 * mat_.mu * eps;
  }

  // ucoef: Dirichlet data as coefficients of the conforming displacement space.
  void assemble(const Eigen::VectorXd& ucoef, Eigen::MatrixXd& A, Eigen::VectorXd& b) const {
    const int Nt = ts_.ndof;
    A.setZero(3 * Nt, 3 * Nt);
    b.setZero(3 * Nt);
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < int(ts_.points.size()); ++a) {
      const auto& c = ts_.points[a];
      const SourcePoint src{c.patch, c.xi, c.x};
      // Data at x: u(x), its tangential gradient, and the data part of grad u(x).
      const auto& l2gx = mesh_.l2g[c.patch];
      Vec3 ux = Vec3::Zero(), uu = Vec3::Zero(), uv = Vec3::Zero();
      for (int l = 0; l < c.E.n_loc; ++l) {
        const Vec3 ul = ucoef.segment<3>(3 * l2gx[c.E.idx[l]]);
        ux += c.E.R[l] * ul; uu += c.E.Ru[l] * ul; uv += c.E.Rv[l] * ul;
      }
      Eigen::Matrix<double, 3, 2> Ax, Ud;
      Ax << c.E.xu, c.E.xv;
      Ud << uu, uv;
      const Eigen::Matrix3d Gt = Ud * (Ax.transpose() * Ax).inverse() * Ax.transpose();
      const Eigen::Matrix3d G0 = gradient(Gt, Vec3::Zero(), c.N);
      const Eigen::Matrix3d sig0 = stress(G0);

      Eigen::MatrixXd Arow = Eigen::MatrixXd::Zero(3, 3 * Nt);
      Vec3 rhs = Vec3::Zero();  // data part, regularized pointwise
      Eigen::Matrix3d M[3], S1[3];
      for (int k = 0; k < 3; ++k) { M[k].setZero(); S1[k].setZero(); }
      for (const auto& e : mesh_.elements) {
        const auto& l2g = mesh_.l2g[e.patch];
        const int off = ts_.offset[e.patch];
        quad_.integrate(src, e, [&](const PatchEval& E, double w) {
          const Vec3 nJ = E.normal() * mesh_.normal_sign;
          const double J = nJ.norm();
          const Vec3 n = nJ / J, d = E.x - c.x;
          Eigen::Matrix3d D, S;
          K_.eval(d, c.N, n, D, S);
          const double wJ = w * J;
          const Eigen::Matrix3d Dw = D * wJ, Sw = S * wJ;
          Vec3 uy = Vec3::Zero();
          for (int l = 0; l < E.n_loc; ++l) {
            Arow.block<3, 3>(0, 3 * (off + E.idx[l])) += E.R[l] * Dw;
            uy += E.R[l] * ucoef.segment<3>(3 * l2g[E.idx[l]]);
          }
          // int S [u - u(x) - G0 d] - D sig0 n: O(1/r) once combined at each point
          rhs += Sw * (uy - ux - G0 * d) + Dw * (sig0 * n);
          for (int k = 0; k < 3; ++k) { M[k] += Dw * n(k); S1[k] += Sw * d(k); }
        });
      }
      auto mapM = [&](const Eigen::Matrix3d& sig) {
        Vec3 v = Vec3::Zero();
        for (int k = 0; k < 3; ++k) v += M[k] * sig.col(k);
        return v;
      };
      auto mapS = [&](const Eigen::Matrix3d& G) {
        Vec3 v = Vec3::Zero();
        for (int l = 0; l < 3; ++l) v += S1[l] * G.col(l);
        return v;
      };
      // Affine dependence on t(x): data part (t = 0) and one column per t_m.
      const Eigen::Matrix3d Gt0 = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d Bt;  // coefficient of t(x) moved to the left-hand side
      for (int m = 0; m < 3; ++m) {
        const Eigen::Matrix3d Gm = gradient(Gt0, Vec3::Unit(m), c.N);
        Bt.col(m) = -mapM(stress(Gm)) + mapS(Gm);
      }
      const int off = ts_.offset[c.patch];
      for (int l = 0; l < c.E.n_loc; ++l)
        if (std::abs(c.E.R[l]) > 1e-15) Arow.block<3, 3>(0, 3 * (off + c.E.idx[l])) += c.E.R[l] * Bt;
      A.middleRows(3 * a, 3) = Arow;
      b.segment<3>(3 * a) = rhs;
    }
  }

 private:
  const Mesh& mesh_;
  const TractionSpace& ts_;
  Material mat_;
  KelvinTraction K_;
  ElementIntegrator quad_;
};

// Relative L2(Gamma) error of the traction against an exact field
// texact(y, n(y)).
template <class F>
inline double tractionL2Error(const Mesh& m, const TractionSpace& ts, const Eigen::VectorXd& tc,
                              F&& texact, int ng = 10) {
  const GaussRule g = gaussLegendre(ng);
  PatchEval E;
  double num = 0.0, den = 0.0;
  for (const auto& el : m.elements)
    for (const auto& R : el.cells)
      for (int i = 0; i < ng; ++i)
        for (int j = 0; j < ng; ++j) {
          m.patches[el.patch].evalSpan(el.su, el.sv, R.u0 + R.du() * g.x[i],
                                       R.v0 + R.dv() * g.x[j], E);
          const Vec3 nJ = E.normal() * m.normal_sign;
          const double w = g.w[i] * g.w[j] * R.du() * R.dv() * nJ.norm();
          Vec3 th = Vec3::Zero();
          for (int l = 0; l < E.n_loc; ++l)
            th += E.R[l] * tc.segment<3>(3 * (ts.offset[el.patch] + E.idx[l]));
          const Vec3 te = texact(E.x, Vec3(nJ.normalized()));
          num += w * (th - te).squaredNorm();
          den += w * te.squaredNorm();
        }
  return std::sqrt(num / den);
}

}  // namespace igabem

namespace igabem {

// Coefficients of the conforming displacement space interpolating u(x) at the
// SBIE collocation (Greville) points.
template <class F>
inline Eigen::VectorXd interpolateDisplacement(const Mesh& m, F&& u) {
  Eigen::MatrixXd Nm = Eigen::MatrixXd::Zero(m.ndof, m.ndof);
  Eigen::MatrixXd rhs(m.ndof, 3);
  for (int a = 0; a < m.ndof; ++a) {
    for (const auto& [B, NB] : m.colloc[a].basis) Nm(a, B) += NB;
    rhs.row(a) = u(m.colloc[a].x).transpose();
  }
  const Eigen::MatrixXd c = Nm.partialPivLu().solve(rhs);
  Eigen::VectorXd out(3 * m.ndof);
  for (int g = 0; g < m.ndof; ++g) out.segment<3>(3 * g) = c.row(g).transpose();
  return out;
}

}  // namespace igabem
