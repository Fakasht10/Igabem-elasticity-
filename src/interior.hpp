// Displacement, gradient, strain and stress at interior points from a boundary
// solution (Somigliana identity and its gradient):
//
//   u_k(x)       = int U_km t_m dGamma - int T_km u_m dGamma
//   du_k/dx_l(x) = int dU_km/dx_l t_m  - int dT_km/dx_l u_m
//
// Near-boundary evaluation: both identities hold exactly for the linear field
// u_L(y) = u(xb) + grad u(xb) (y - xb), t_L = sigma(xb) n, with xb the closest
// boundary point. Subtracting it,
//
//   u(x)      = u_L(x)       + int U (t - t_L) - int T (u - u_L)
//   grad u(x) = grad u(xb)   + int dU/dx (t - t_L) - int dT/dx (u - u_L)
//
// the integrands vanish at xb to first and second order, so the result stays
// accurate as x approaches the boundary. Its limit differs from the boundary
// recovery at xb only by the discretization error (the traction-equation
// residual of u_h at xb), e.g. 1.5e-5 relative at the plate's hole for p = 4, r3.
// The near-singular integrals use the adaptive quadtree of ElementIntegrator
// (SolverOptions::far_tol > 0 required).
//
// inside(x): the solid angle integral int (y - x).n / (4 pi |y - x|^3) dGamma,
// 1 inside the solid and 0 outside.
#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#include "integrator.hpp"
#include "kelvin.hpp"
#include "recovery.hpp"
#include "regions.hpp"
#include "solution.hpp"

namespace igabem {

struct FieldSample {
  Vec3 x;
  Vec3 u;
  Eigen::Matrix3d grad, strain, stress;
  double winding = 1.0;      // 1 inside, 0 outside, 1/2 on a smooth boundary
  bool on_boundary = false;  // value from the boundary recovery
};

class InteriorEvaluator {
 public:
  InteriorEvaluator(const BoundarySolution& s, const SolverOptions& o)
      : s_(s), m_(*s.mesh), K_(s.mat), KT_(s.mat), quad_(*s.mesh, o) {
    if (o.far_tol <= 0.0) throw std::runtime_error("InteriorEvaluator needs SolverOptions::far_tol > 0");
    for (const auto& d : s.region) region_.emplace_back(*d->R, o);
  }

  // Boundary distance below which x is treated as on the boundary.
  double on_boundary_tol = 1e-10;

  FieldSample eval(const Vec3& x) const {
    FieldSample f;
    f.x = x;
    const SurfacePoint sp = locate(m_, x, [](int) { return true; });
    const Vec3 xb = sp.E.x;
    const Vec3 ub = s_.displacement(sp);
    const Eigen::Matrix3d Gb = s_.gradient(sp);
    const Eigen::Matrix3d Sb = s_.mat.stress(Gb);
    if ((x - xb).norm() < on_boundary_tol) {
      f.u = ub;
      f.grad = Gb;
      f.winding = 0.5;
      f.on_boundary = true;
    } else {
      const SourcePoint src{-1, Vec2(0.0, 0.0), x};
      Vec3 du = Vec3::Zero();
      Eigen::Matrix3d dG = Eigen::Matrix3d::Zero();
      double wind = 0.0;
      for (const auto& e : m_.elements) {
        const int patch = e.patch, off = s_.ts->offset[patch];
        const auto& l2g = m_.l2g[patch];
        quad_.integrate(src, e, [&](const QPoint& q, double w) {
          const Vec3 nJ = q.normal() * m_.normal_sign;
          const double J = nJ.norm();
          const Vec3 n = nJ / J, d = q.x - x;
          Vec3 uy = Vec3::Zero(), ty = Vec3::Zero();
          for (int l = 0; l < q.n_loc; ++l) {
            uy += q.R[l] * s_.u.segment<3>(3 * l2g[q.idx[l]]);
            ty += q.R[l] * s_.t.segment<3>(3 * (off + q.idx[l]));
          }
          const Vec3 eu = uy - ub - Gb * (q.x - xb), et = ty - Sb * n;
          Eigen::Matrix3d U, T, P[3], Q[3];
          K_.eval(d, n, U, T);
          KT_.gradients(d, n, P, Q);
          const double wJ = w * J;
          du += wJ * (U * et - T * eu);
          for (int mm = 0; mm < 3; ++mm) dG += wJ * (P[mm] * et(mm) - Q[mm] * eu(mm));
          const double r = d.norm();
          wind += wJ * d.dot(n) / (4.0 * M_PI * r * r * r);
        });
      }
      // Region tractions (their t_L part is in the loop above, through Sb).
      for (size_t k = 0; k < region_.size(); ++k)
        region_[k].visit(src, *s_.region[k], [&](const Vec3& y, const Vec3& n, const RegionBlock& tw) {
          Eigen::Matrix3d U, T, P[3], Q[3];
          K_.eval(y - x, n, U, T);
          KT_.gradients(y - x, n, P, Q);
          du += U * tw.col(0);
          for (int mm = 0; mm < 3; ++mm) dG += P[mm] * tw(mm, 0);
        });
      f.u = ub + Gb * (x - xb) + du;
      f.grad = Gb + dG;
      f.winding = wind;
    }
    f.strain = strainOf(f.grad);
    f.stress = s_.mat.stress(f.grad);
    return f;
  }

  // Boundary recovery at the closest boundary point to x (x itself need not lie on the
  // spline surface, e.g. a node of a tetrahedral mesh on the CAD face). f.x is the
  // projected point.
  FieldSample evalSurface(const Vec3& x) const {
    FieldSample f;
    const SurfacePoint sp = locate(m_, x, [](int) { return true; });
    f.x = sp.E.x;
    f.u = s_.displacement(sp);
    f.grad = s_.gradient(sp);
    f.winding = 0.5;
    f.on_boundary = true;
    f.strain = strainOf(f.grad);
    f.stress = s_.mat.stress(f.grad);
    return f;
  }

  // Many points, in parallel; on_surface[i] (if given) takes the boundary recovery.
  std::vector<FieldSample> eval(const std::vector<Vec3>& xs, const std::vector<char>& on_surface = {}) const {
    std::vector<FieldSample> out(xs.size());
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < long(xs.size()); ++i)
      out[i] = (!on_surface.empty() && on_surface[i]) ? evalSurface(xs[i]) : eval(xs[i]);
    return out;
  }

 private:
  const BoundarySolution& s_;
  const Mesh& m_;
  Kelvin K_;
  KelvinTraction KT_;
  ElementIntegrator quad_;
  std::vector<RegionIntegrator> region_;
};

}  // namespace igabem
