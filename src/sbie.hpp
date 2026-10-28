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
#include <map>
#include <vector>

#include "kelvin.hpp"
#include "nurbs.hpp"
#include "quadrature.hpp"

namespace igabem {

struct Element {
  int patch, su, sv;
  Rect rect;                // parametric extent
  std::vector<Rect> cells;  // App. B partition
  double h;                 // physical diameter estimate
};

struct Collocation {
  int patch, su, sv;
  Vec2 xi;
  Vec3 x;
  std::vector<std::pair<int, double>> basis;  // global dof, N_B(x) != 0
};

struct Mesh {
  std::vector<Patch> patches;
  std::vector<std::vector<int>> l2g;  // per patch: local cp -> global dof
  std::vector<Vec3> dof_point;        // control point of each global dof
  std::vector<Element> elements;
  std::vector<Collocation> colloc;
  int ndof = 0;
  double normal_sign = 1.0;  // flips normals if the files are inward oriented

  void build(std::vector<Patch> P, const PartitionOptions& popt, double merge_tol = 1e-9) {
    patches = std::move(P);
    // Merge coincident control points into conforming global dofs.
    l2g.assign(patches.size(), {});
    dof_point.clear();
    std::vector<std::pair<int, int>> owner;
    for (size_t ip = 0; ip < patches.size(); ++ip) {
      const Patch& Pa = patches[ip];
      l2g[ip].resize(Pa.nu * Pa.nv);
      for (int k = 0; k < Pa.nu * Pa.nv; ++k) {
        const Vec3 X = Pa.cp(k);
        int found = -1;
        for (size_t g = 0; g < dof_point.size(); ++g)
          if ((dof_point[g] - X).norm() < merge_tol) { found = int(g); break; }
        if (found < 0) {
          found = int(dof_point.size());
          dof_point.push_back(X);
          owner.push_back({int(ip), k});
        }
        l2g[ip][k] = found;
      }
    }
    ndof = int(dof_point.size());

    // Elements = non-empty knot spans.
    elements.clear();
    for (size_t ip = 0; ip < patches.size(); ++ip) {
      const Patch& Pa = patches[ip];
      for (int su = Pa.p; su < Pa.nu; ++su) {
        if (Pa.U[su + 1] <= Pa.U[su]) continue;
        for (int sv = Pa.q; sv < Pa.nv; ++sv) {
          if (Pa.V[sv + 1] <= Pa.V[sv]) continue;
          Element e{int(ip), su, sv, {Pa.U[su], Pa.U[su + 1], Pa.V[sv], Pa.V[sv + 1]}, {}, 0.0};
          partitionElement(Pa, su, sv, e.rect, popt, 0, e.cells);
          PatchEval E;
          Vec3 c[4];
          int k = 0;
          for (double u : {e.rect.u0, e.rect.u1})
            for (double v : {e.rect.v0, e.rect.v1}) { Pa.evalSpan(su, sv, u, v, E); c[k++] = E.x; }
          e.h = std::max((c[3] - c[0]).norm(), (c[2] - c[1]).norm());
          elements.push_back(std::move(e));
        }
      }
    }

    // Collocation at the Greville point of each dof's first owner.
    colloc.clear();
    for (int g = 0; g < ndof; ++g) {
      const int ip = owner[g].first, k = owner[g].second;
      const Patch& Pa = patches[ip];
      const int i = k % Pa.nu, j = k / Pa.nu;
      Collocation c;
      c.patch = ip;
      c.xi = Vec2(Pa.greville(i, true), Pa.greville(j, false));
      c.su = findSpan(Pa.nu, Pa.p, c.xi(0), Pa.U);
      c.sv = findSpan(Pa.nv, Pa.q, c.xi(1), Pa.V);
      PatchEval E;
      Pa.evalSpan(c.su, c.sv, c.xi(0), c.xi(1), E);
      c.x = E.x;
      for (int l = 0; l < E.n_loc; ++l)
        if (std::abs(E.R[l]) > 1e-15) c.basis.push_back({l2g[ip][E.idx[l]], E.R[l]});
      colloc.push_back(std::move(c));
    }

    // Orientation: the volume (1/3) int x.n must be positive for outward n.
    normal_sign = volume() < 0 ? -1.0 : 1.0;
  }

  double volume() const {
    const GaussRule g = gaussLegendre(8);
    double V = 0.0;
    PatchEval E;
    for (const auto& e : elements)
      for (const auto& R : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a],
                                      R.v0 + R.dv() * g.x[b], E);
            V += g.w[a] * g.w[b] * R.du() * R.dv() * E.x.dot(E.normal()) / 3.0;
          }
    return V * normal_sign;
  }

  // Closest point of an element to x: grid sample, then projected Gauss-Newton.
  Vec2 closestPoint(const Element& e, const Vec3& x, double& dist) const {
    const Patch& Pa = patches[e.patch];
    PatchEval E;
    Vec2 best;
    double bd = 1e300;
    for (int a = 0; a <= 6; ++a)
      for (int b = 0; b <= 6; ++b) {
        Vec2 p(e.rect.u0 + e.rect.du() * a / 6.0, e.rect.v0 + e.rect.dv() * b / 6.0);
        Pa.evalSpan(e.su, e.sv, p(0), p(1), E);
        const double d = (E.x - x).norm();
        if (d < bd) { bd = d; best = p; }
      }
    for (int it = 0; it < 30; ++it) {
      Pa.evalSpan(e.su, e.sv, best(0), best(1), E);
      Eigen::Matrix<double, 3, 2> J;
      J << E.xu, E.xv;
      const Vec2 step = (J.transpose() * J).ldlt().solve(J.transpose() * (x - E.x));
      const Vec2 nb = e.rect.clamp(best + step);
      if ((nb - best).norm() < 1e-15 * (1.0 + e.rect.du())) { best = nb; break; }
      best = nb;
    }
    Pa.evalSpan(e.su, e.sv, best(0), best(1), E);
    dist = (E.x - x).norm();
    return best;
  }
};

struct SolverOptions {
  int n_gauss = 8;          // integration points per direction per cell
  bool near_refine = true;  // App. B near-singular refinement (d <= 3h)
  int near_max_depth = 16;
};

// Assembles A (3M x 3M) and b for the interior Neumann problem with
// prescribed traction t(y) = sigma n(y).
class SbieAssembler {
 public:
  SbieAssembler(const Mesh& m, const Material& mat, const SolverOptions& o)
      : mesh_(m), K_(mat), opt_(o), g_(gaussLegendre(o.n_gauss)) {}

  void assemble(const Eigen::Matrix3d& sigma, Eigen::MatrixXd& A, Eigen::VectorXd& b) const {
    const int M = mesh_.ndof;
    A.setZero(3 * M, 3 * M);
    b.setZero(3 * M);
#pragma omp parallel for schedule(dynamic)
    for (int a = 0; a < int(mesh_.colloc.size()); ++a) {
      Row row{Eigen::MatrixXd::Zero(3, 3 * M), Eigen::Vector3d::Zero(), Eigen::Matrix3d::Zero()};
      const Collocation& c = mesh_.colloc[a];
      for (const auto& e : mesh_.elements) integrateElement(c, e, sigma, row);
      for (const auto& [B, NB] : c.basis) row.A.block<3, 3>(0, 3 * B) -= NB * row.Tsum;
      A.middleRows(3 * a, 3) = row.A;
      b.segment<3>(3 * a) = row.b;
    }
  }

 private:
  struct Row {
    Eigen::MatrixXd A;
    Eigen::Vector3d b;
    Eigen::Matrix3d Tsum;  // int T dGamma, multiplied by N_B(x) afterwards
  };

  const Mesh& mesh_;
  Kelvin K_;
  SolverOptions opt_;
  GaussRule g_;

  inline void addPoint(const Collocation& c, const Element& e, const Vec2& p, double w,
                       const Eigen::Matrix3d& sigma, Row& row) const {
    PatchEval E;
    mesh_.patches[e.patch].evalSpan(e.su, e.sv, p(0), p(1), E);
    Vec3 nJ = E.normal() * mesh_.normal_sign;
    const double J = nJ.norm();
    const Vec3 n = nJ / J;
    Eigen::Matrix3d U, T;
    K_.eval(E.x - c.x, n, U, T);
    const double wJ = w * J;
    const Eigen::Matrix3d Tw = T * wJ;
    const auto& l2g = mesh_.l2g[e.patch];
    for (int l = 0; l < E.n_loc; ++l)
      row.A.block<3, 3>(0, 3 * l2g[E.idx[l]]) += E.R[l] * Tw;
    row.Tsum += Tw;
    row.b += U * (sigma * n) * wJ;
  }

  void gaussRect(const Collocation& c, const Element& e, const Rect& R,
                 const Eigen::Matrix3d& sigma, Row& row) const {
    const int n = int(g_.x.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        addPoint(c, e, Vec2(R.u0 + R.du() * g_.x[i], R.v0 + R.dv() * g_.x[j]),
                 g_.w[i] * g_.w[j] * R.du() * R.dv(), sigma, row);
  }

  // Duffy split of a rectangle into triangles with a common vertex at xs.
  void duffyRect(const Collocation& c, const Element& e, const Rect& R, const Vec2& xs,
                 const Eigen::Matrix3d& sigma, Row& row) const {
    const Vec2 C[4] = {{R.u0, R.v0}, {R.u1, R.v0}, {R.u1, R.v1}, {R.u0, R.v1}};
    const int n = int(g_.x.size());
    for (int k = 0; k < 4; ++k) {
      const Vec2 P1 = C[k], P2 = C[(k + 1) % 4];
      const Vec2 a = P1 - xs, d = P2 - P1;
      const double det = std::abs(a(0) * d(1) - a(1) * d(0));
      if (det < 1e-14 * R.du() * R.dv()) continue;  // xs on this edge
      for (int i = 0; i < n; ++i) {
        const double s = g_.x[i];
        for (int j = 0; j < n; ++j) {
          const Vec2 p = xs + s * (a + g_.x[j] * d);
          addPoint(c, e, p, g_.w[i] * g_.w[j] * s * det, sigma, row);
        }
      }
    }
  }

  void nearRect(const Collocation& c, const Element& e, const Rect& R, const Vec2& xs,
                int depth, const Eigen::Matrix3d& sigma, Row& row) const {
    if (opt_.near_refine && depth < opt_.near_max_depth) {
      const Patch& Pa = mesh_.patches[e.patch];
      PatchEval E;
      const Vec2 q = R.clamp(xs);
      Pa.evalSpan(e.su, e.sv, q(0), q(1), E);
      const double d = (E.x - c.x).norm();
      Vec3 X[4];
      int k = 0;
      for (double u : {R.u0, R.u1})
        for (double v : {R.v0, R.v1}) { Pa.evalSpan(e.su, e.sv, u, v, E); X[k++] = E.x; }
      const double h = std::max((X[3] - X[0]).norm(), (X[2] - X[1]).norm());
      if (d <= 3.0 * h) {
        const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
        nearRect(c, e, {R.u0, um, R.v0, vm}, xs, depth + 1, sigma, row);
        nearRect(c, e, {um, R.u1, R.v0, vm}, xs, depth + 1, sigma, row);
        nearRect(c, e, {R.u0, um, vm, R.v1}, xs, depth + 1, sigma, row);
        nearRect(c, e, {um, R.u1, vm, R.v1}, xs, depth + 1, sigma, row);
        return;
      }
    }
    gaussRect(c, e, R, sigma, row);
  }

  void integrateElement(const Collocation& c, const Element& e,
                        const Eigen::Matrix3d& sigma, Row& row) const {
    Vec2 xs;
    double dist;
    const bool own = (e.patch == c.patch) && e.rect.contains(c.xi, 1e-14);
    if (own) { xs = c.xi; dist = 0.0; }
    else xs = mesh_.closestPoint(e, c.x, dist);
    const bool singular = dist < 1e-11 * e.h;
    const double tol = 1e-12 * std::max(e.rect.du(), e.rect.dv());
    for (const Rect& R : e.cells) {
      if (singular && R.contains(xs, tol)) duffyRect(c, e, R, R.clamp(xs), sigma, row);
      else if (singular || dist <= 3.0 * e.h) nearRect(c, e, R, xs, 0, sigma, row);
      else gaussRect(c, e, R, sigma, row);
    }
  }
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
