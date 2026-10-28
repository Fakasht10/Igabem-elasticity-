// Integration of a (weakly) singular integrand over one element, seen from a
// collocation point: Duffy split on cells containing the point, App. B
// near-singular quadtree while d <= 3h, plain n x n Gauss otherwise.
//
// The integrand is a callback f(E, w) receiving the patch evaluation at a
// quadrature point and its parametric weight w (the surface Jacobian |x_u x x_v|
// is not included; the callback applies it).
#pragma once

#include <algorithm>
#include <cmath>

#include "mesh.hpp"

namespace igabem {

struct SourcePoint {
  int patch;  // patch on which xi is given
  Vec2 xi;
  Vec3 x;
};

class ElementIntegrator {
 public:
  ElementIntegrator(const Mesh& m, const SolverOptions& o)
      : mesh_(m), opt_(o), g_(gaussLegendre(o.n_gauss)) {}

  template <class F>
  void integrate(const SourcePoint& c, const Element& e, F&& f) const {
    Vec2 xs;
    double dist;
    const bool own = (e.patch == c.patch) && e.rect.contains(c.xi, 1e-14);
    if (own) { xs = c.xi; dist = 0.0; }
    else xs = mesh_.closestPoint(e, c.x, dist);
    const bool singular = dist < 1e-11 * e.h;
    const double tol = 1e-12 * std::max(e.rect.du(), e.rect.dv());
    for (const Rect& R : e.cells) {
      if (singular && R.contains(xs, tol)) duffyRect(e, R, R.clamp(xs), f);
      else if (singular || dist <= 3.0 * e.h) nearRect(c, e, R, xs, 0, f);
      else gaussRect(e, R, f);
    }
  }

 private:
  const Mesh& mesh_;
  SolverOptions opt_;
  GaussRule g_;

  template <class F>
  void point(const Element& e, const Vec2& p, double w, F& f) const {
    PatchEval E;
    mesh_.patches[e.patch].evalSpan(e.su, e.sv, p(0), p(1), E);
    f(E, w);
  }

  template <class F>
  void gaussRect(const Element& e, const Rect& R, F& f) const {
    const int n = int(g_.x.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        point(e, Vec2(R.u0 + R.du() * g_.x[i], R.v0 + R.dv() * g_.x[j]),
              g_.w[i] * g_.w[j] * R.du() * R.dv(), f);
  }

  // Duffy split of a rectangle into triangles with a common vertex at xs.
  template <class F>
  void duffyRect(const Element& e, const Rect& R, const Vec2& xs, F& f) const {
    const Vec2 C[4] = {{R.u0, R.v0}, {R.u1, R.v0}, {R.u1, R.v1}, {R.u0, R.v1}};
    const int n = int(g_.x.size());
    for (int k = 0; k < 4; ++k) {
      const Vec2 P1 = C[k], P2 = C[(k + 1) % 4];
      const Vec2 a = P1 - xs, d = P2 - P1;
      const double det = std::abs(a(0) * d(1) - a(1) * d(0));
      if (det < 1e-14 * R.du() * R.dv()) continue;  // xs on this edge
      for (int i = 0; i < n; ++i) {
        const double s = g_.x[i];
        for (int j = 0; j < n; ++j)
          point(e, xs + s * (a + g_.x[j] * d), g_.w[i] * g_.w[j] * s * det, f);
      }
    }
  }

  template <class F>
  void nearRect(const SourcePoint& c, const Element& e, const Rect& R, const Vec2& xs,
                int depth, F& f) const {
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
        nearRect(c, e, {R.u0, um, R.v0, vm}, xs, depth + 1, f);
        nearRect(c, e, {um, R.u1, R.v0, vm}, xs, depth + 1, f);
        nearRect(c, e, {R.u0, um, vm, R.v1}, xs, depth + 1, f);
        nearRect(c, e, {um, R.u1, vm, R.v1}, xs, depth + 1, f);
        return;
      }
    }
    gaussRect(e, R, f);
  }
};

}  // namespace igabem
