// Integration of a (weakly) singular integrand over one element, seen from a
// collocation point: Duffy split on cells containing the point, App. B
// near-singular quadtree while d <= 3h, plain n x n Gauss otherwise (optionally
// with a distance-dependent order, SolverOptions::far_tol).
//
// The integrand is a callback f(q, w) receiving a quadrature point q (position,
// unnormalized normal x_u x x_v, nonzero basis functions) and its parametric
// weight w (the surface Jacobian |x_u x x_v| is not included; the callback
// applies it).
//
// With far_tol > 0, far elements use one Gauss rule over the whole element whose
// points are evaluated once and cached (geometry and basis), so a far-field
// quadrature point costs only the kernel evaluation. Cached points carry an id
// (q.id >= 0) so callers can cache their own point data (e.g. traction) too.
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "mesh.hpp"

namespace igabem {

struct QPoint {
  Vec3 x, nJ;  // position and x_u x x_v
  int n_loc;
  const int* idx;   // local control point indices within the patch
  const double* R;  // rational basis values
  long id;          // index of a cached point, -1 if evaluated on the fly
  Vec3 normal() const { return nJ; }
};

struct SourcePoint {
  int patch;  // patch on which xi is given
  Vec2 xi;
  Vec3 x;
};

class ElementIntegrator {
 public:
  // cache = false: no far-field point cache (adaptive mode then splits far
  // elements into cells instead), for short-lived integrators on small meshes,
  // e.g. the sub-cells of a loaded region (regions.hpp).
  ElementIntegrator(const Mesh& m, const SolverOptions& o, bool cache = true)
      : mesh_(m), opt_(o), g_(gaussLegendre(o.n_gauss)) {
    for (int k = 1; k <= o.n_gauss; ++k) rules_.push_back(gaussLegendre(k));
    // Bounding spheres of elements and cells, for cheap far-field tests.
    elem_ball_.resize(m.elements.size());
    cell_ball_.resize(m.elements.size());
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < m.elements.size(); ++i) {
      const Element& e = m.elements[i];
      elem_ball_[i] = ball(e, e.rect);
      for (const Rect& R : e.cells) cell_ball_[i].push_back(ball(e, R));
    }
    if (o.far_tol > 0.0 && cache) buildCache();
  }

  // Adaptive mode (far_tol > 0): one cached rule per far element; otherwise
  // Duffy on the cells touching the source point and, on the remaining cells, a
  // quadtree split until the Gauss order the error estimate asks for is <= n.
  template <class F>
  void integrateAdaptive(const SourcePoint& c, const Element& e, size_t ie, bool own, F& f) const {
    const Ball& B = elem_ball_[ie];
    if (!own && max_cached_ > 0) {
      const int m = std::max(farOrder(c.x, B), min_order_[ie]);
      if (m <= max_cached_ && (c.x - B.first).norm() > 1.5 * B.second) { cachedRule(ie, m, f); return; }
    }
    Vec2 xs(0.0, 0.0);
    double dist = 1e300;
    if (own) { xs = c.xi; dist = 0.0; }
    else if ((c.x - B.first).norm() <= B.second) xs = mesh_.closestPoint(e, c.x, dist);
    const bool singular = dist < 1e-11 * e.h;
    const double tol = 1e-12 * std::max(e.rect.du(), e.rect.dv());
    for (size_t k = 0; k < e.cells.size(); ++k) {
      const Rect& R = e.cells[k];
      if (singular && R.contains(xs, tol)) singularRect(c, e, R, R.clamp(xs), f);
      else adaptiveRect(c, e, R, 0, f);
    }
  }

  // Adaptive mode: the cached rule of element ie meeting the error estimate tol
  // (max(tol, far_tol); tol = far_tol gives the rule integrateAdaptive uses, or a
  // finer one) for every source point at least d away from the centre of the
  // element's bounding ball; f(q, w) for each of its points. False, calling nothing,
  // if no cached rule qualifies (d too small, or the order above the cache).
  // min_order >= 2 replaces the element's minimum order (degree + 2), e.g. for
  // separated blocks whose entries are only needed to a compression tolerance;
  // min_order < 0 takes it from the degree (see farRuleOrder).
  template <class F>
  bool farRule(size_t ie, double d, double tol, F&& f, int min_order = 0) const {
    const int m = farRuleOrder(ie, d, tol, min_order);
    if (!m) return false;
    cachedRule(ie, m, f);
    return true;
  }

  // The rule farRule would use, as arrays for batched evaluation: n points x, nJ
  // (unnormalized normal, |nJ| the Jacobian), weights w, and the n_loc basis values
  // of point k at R + n_loc k (local indices idx).
  struct RuleData {
    const Vec3 *x, *nJ;
    const double *w, *R;
    int n, n_loc;
    const int* idx;
  };
  bool farRuleData(size_t ie, double d, double tol, int min_order, RuleData& r) const {
    const int m = farRuleOrder(ie, d, tol, min_order);
    if (!m) return false;
    ruleData(ie, m, r);
    return true;
  }
  // The cached m x m rule of element ie (2 <= m <= maxCachedOrder()).
  void ruleData(size_t ie, int m, RuleData& r) const {
    const long p0 = first_[ie][m];
    r = {&cx_[p0], &cnj_[p0], &cw_[p0], &cr_[roff_[p0]], m * m, nloc_[ie], eidx_[ie].data()};
  }
  int maxCachedOrder() const { return max_cached_; }

  // Order of farRule's rule, 0 if none qualifies.
  int farRuleOrder(size_t ie, double d, double tol, int min_order) const {
    if (opt_.far_tol <= 0.0) return 0;
    const Ball& B = elem_ball_[ie];
    if (d <= 1.5 * B.second) return 0;
    const double q = d / B.second, rho = q + std::sqrt(q * q - 1.0);
    const int need = int(std::ceil(std::log(1.0 / std::max(tol, opt_.far_tol)) / (2.0 * std::log(rho))));
    // A lower minimum only where the basis is polynomial: rational (curved) elements
    // get one order more (a 4 x 4 rule left 3e-3 errors in a far block on the hole,
    // 5 x 5 is within the tolerance).
    // min_order < 0: from the degree p of the element, the lowest order exact for
    // a degree-p basis times a linear kernel (2m - 1 >= p + 1), one more on rational elements.
    const int deg_lo = (min_order_[ie] + 1) / 2 + rational_[ie];  // min_order_ = p + 2
    const int lo = min_order < 0 ? std::min(min_order_[ie], deg_lo)
                   : min_order >= 2 ? (rational_[ie] ? std::min(min_order_[ie], min_order + 1) : min_order) : min_order_[ie];
    const int m = std::max({2, std::min(opt_.n_gauss, need), lo});
    return m > max_cached_ ? 0 : m;
  }
  const Vec3& elementCentre(size_t ie) const { return elem_ball_[ie].first; }
  double elementRadius(size_t ie) const { return elem_ball_[ie].second; }

  // Cached far-field points (for callers that cache per-point data).
  long numCached() const { return long(cx_.size()); }
  const Vec3& cachedX(long id) const { return cx_[id]; }
  const Vec3& cachedNJ(long id) const { return cnj_[id]; }

  template <class F>
  void integrate(const SourcePoint& c, const Element& e, F&& f) const {
    const size_t ie = &e - mesh_.elements.data();
    const bool own = (e.patch == c.patch) && e.rect.contains(c.xi, 1e-14);
    if (opt_.far_tol > 0.0) { integrateAdaptive(c, e, ie, own, f); return; }
    // Far element: its bounding sphere is more than 3h away, no projection needed.
    if (!own && (c.x - elem_ball_[ie].first).norm() - elem_ball_[ie].second > 3.0 * e.h) {
      if (opt_.far_tol > 0.0) {
        const int m = std::max(farOrder(c.x, elem_ball_[ie]), min_order_[ie]);
        if (m < opt_.n_gauss || int(e.cells.size()) > 1) {
          if (m <= max_cached_) { cachedRule(ie, m, f); return; }
        }
      }
      for (size_t k = 0; k < e.cells.size(); ++k) gaussRect(c.x, e, e.cells[k], cell_ball_[ie][k], f);
      return;
    }
    Vec2 xs;
    double dist;
    if (own) { xs = c.xi; dist = 0.0; }
    else xs = mesh_.closestPoint(e, c.x, dist);
    const bool singular = dist < 1e-11 * e.h;
    const double tol = 1e-12 * std::max(e.rect.du(), e.rect.dv());
    for (size_t k = 0; k < e.cells.size(); ++k) {
      const Rect& R = e.cells[k];
      if (singular && R.contains(xs, tol)) duffyRect(e, R, R.clamp(xs), f);
      else if (singular || dist <= 3.0 * e.h) nearRect(c, e, R, xs, 0, f);
      else gaussRect(c.x, e, R, cell_ball_[ie][k], f);
    }
  }

 private:
  using Ball = std::pair<Vec3, double>;
  const Mesh& mesh_;
  SolverOptions opt_;
  GaussRule g_;
  std::vector<GaussRule> rules_;  // orders 1..n_gauss
  std::vector<Ball> elem_ball_;
  std::vector<std::vector<Ball>> cell_ball_;
  // Far-field cache: element-wide m x m Gauss rules, m = 2..max_cached_.
  int max_cached_ = 0;
  std::vector<int> min_order_;               // per element: max degree + 2
  std::vector<char> rational_;               // per element: weights not all equal
  std::vector<std::vector<long>> first_;     // per element, per order: first point id
  std::vector<std::vector<int>> eidx_;       // per element: local basis indices (constant)
  std::vector<int> nloc_;
  std::vector<Vec3> cx_, cnj_;
  std::vector<double> cw_;
  std::vector<long> roff_;                   // per point: offset into cr_
  std::vector<double> cr_;

  // Points of element i, order m, at first_[i][m]; elements in order, orders 2..max_cached_
  // (n_loc = (p + 1)(q + 1) basis values each), filled in parallel.
  void buildCache() {
    const size_t ne = mesh_.elements.size();
    max_cached_ = std::min(opt_.n_gauss, 10);
    min_order_.assign(ne, 2);
    rational_.assign(ne, 0);
    first_.assign(ne, {});
    eidx_.assign(ne, {});
    nloc_.assign(ne, 0);
    std::vector<long> rfirst(ne + 1, 0);  // first basis value of each element
    long npt = 0;
    for (size_t i = 0; i < ne; ++i) {
      const Patch& Pa = mesh_.patches[mesh_.elements[i].patch];
      min_order_[i] = std::max(Pa.p, Pa.q) + 2;
      nloc_[i] = (Pa.p + 1) * (Pa.q + 1);
      first_[i].assign(max_cached_ + 1, -1);
      for (int m = 2; m <= max_cached_; ++m) { first_[i][m] = npt; npt += long(m) * m; }
      rfirst[i + 1] = rfirst[i] + long(nloc_[i]) * (npt - first_[i][2]);
    }
    cx_.resize(npt);
    cnj_.resize(npt);
    cw_.resize(npt);
    roff_.resize(npt);
    cr_.resize(rfirst[ne]);
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < ne; ++i) {
      const Element& e = mesh_.elements[i];
      const Patch& Pa = mesh_.patches[e.patch];
      PatchEval E;
      long id = first_[i][2], ro = rfirst[i];
      for (int m = 2; m <= max_cached_; ++m) {
        const GaussRule& g = rules_[m - 1];
        const Rect& R = e.rect;
        for (int a = 0; a < m; ++a)
          for (int b = 0; b < m; ++b, ++id) {
            Pa.evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
            if (eidx_[i].empty()) {
              eidx_[i].assign(E.idx, E.idx + E.n_loc);
              for (int l = 1; l < E.n_loc; ++l)
                if (std::abs(Pa.w(E.idx[l]) - Pa.w(E.idx[0])) > 1e-12 * std::abs(Pa.w(E.idx[0]))) rational_[i] = 1;
            }
            cx_[id] = E.x;
            cnj_[id] = E.normal();
            cw_[id] = g.w[a] * g.w[b] * R.du() * R.dv();
            roff_[id] = ro;
            std::copy(E.R, E.R + E.n_loc, cr_.begin() + ro);
            ro += E.n_loc;
          }
      }
    }
  }

  template <class F>
  void cachedRule(size_t ie, int m, F& f) const {
    const long p0 = first_[ie][m];
    for (long id = p0; id < p0 + long(m) * m; ++id) {
      const QPoint q{cx_[id], cnj_[id], nloc_[ie], eidx_[ie].data(), &cr_[roff_[id]], id};
      f(q, cw_[id]);
    }
  }

  // Centre and radius (with a 10% margin for curvature) of a parametric rectangle's image.
  Ball ball(const Element& e, const Rect& R) const {
    PatchEval E;
    Vec3 X[25], c = Vec3::Zero();
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        mesh_.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * i / 4.0, R.v0 + R.dv() * j / 4.0, E);
        X[5 * i + j] = E.x;
        c += E.x / 25.0;
      }
    double r = 0.0;
    for (const Vec3& x : X) r = std::max(r, (x - c).norm());
    return {c, 1.1 * r};
  }

  // Gauss order for a cell seen from x (far_tol > 0), else n_gauss.
  int farOrder(const Vec3& x, const Ball& b) const {
    const int n = opt_.n_gauss;
    if (opt_.far_tol <= 0.0) return n;
    const double q = (x - b.first).norm() / b.second;
    if (q <= 1.5) return n;
    const double rho = q + std::sqrt(q * q - 1.0);
    const int m = int(std::ceil(std::log(1.0 / opt_.far_tol) / (2.0 * std::log(rho))));
    return std::min(n, std::max(2, m));
  }

  template <class F>
  void point(const Element& e, const Vec2& p, double w, F& f) const {
    PatchEval E;
    mesh_.patches[e.patch].evalSpan(e.su, e.sv, p(0), p(1), E);
    const QPoint q{E.x, E.normal(), E.n_loc, E.idx, E.R, -1};
    f(q, w);
  }

  template <class F>
  void gaussRect(const Vec3& x, const Element& e, const Rect& R, const Ball& b, F& f) const {
    const GaussRule& g = rules_[farOrder(x, b) - 1];
    const int n = int(g.x.size());
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        point(e, Vec2(R.u0 + R.du() * g.x[i], R.v0 + R.dv() * g.x[j]),
              g.w[i] * g.w[j] * R.du() * R.dv(), f);
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

  // Adaptive mode, cell containing the source point xs: Duffy on a block around
  // xs whose sides are at most twice the distance from xs to the nearest cell
  // edge (and of aspect <= 2 physically), adaptive quadtree on the rest. A plain
  // Duffy split degrades when xs is close to, but not on, a cell edge (thin
  // triangles): e.g. traction collocation points near patch edges.
  template <class F>
  void singularRect(const SourcePoint& c, const Element& e, const Rect& R, const Vec2& xs, F& f) const {
    PatchEval E;
    mesh_.patches[e.patch].evalSpan(e.su, e.sv, xs(0), xs(1), E);
    const double su = E.xu.norm(), sv = E.xv.norm();  // physical length per unit parameter
    const double tol = 1e-12 * std::max(R.du(), R.dv());
    auto half = [&](double a0, double a1) {  // half-width: nearest edge not touching xs
      if (a0 <= tol) return a1;
      if (a1 <= tol) return a0;
      return std::min(a0, a1);
    };
    double hu = half(xs(0) - R.u0, R.u1 - xs(0)), hv = half(xs(1) - R.v0, R.v1 - xs(1));
    hu = std::min(hu, 2.0 * hv * sv / su);
    hv = std::min(hv, 2.0 * hu * su / sv);
    const double cu[4] = {R.u0, std::max(R.u0, xs(0) - hu), std::min(R.u1, xs(0) + hu), R.u1};
    const double cv[4] = {R.v0, std::max(R.v0, xs(1) - hv), std::min(R.v1, xs(1) + hv), R.v1};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        const Rect S{cu[i], cu[i + 1], cv[j], cv[j + 1]};
        if (S.du() <= tol || S.dv() <= tol) continue;
        if (i == 1 && j == 1) duffyRect(e, S, S.clamp(xs), f);
        else adaptiveRect(c, e, S, 0, f);
      }
  }

  template <class F>
  void adaptiveRect(const SourcePoint& c, const Element& e, const Rect& R, int depth, F& f) const {
    const Patch& Pa = mesh_.patches[e.patch];
    PatchEval E;
    Vec3 X[4];
    int k = 0;
    for (double u : {R.u0, R.u1})
      for (double v : {R.v0, R.v1}) { Pa.evalSpan(e.su, e.sv, u, v, E); X[k++] = E.x; }
    const double h = std::max((X[3] - X[0]).norm(), (X[2] - X[1]).norm());
    const Ball b{0.25 * (X[0] + X[1] + X[2] + X[3]), 0.6 * h};
    const double q = (c.x - b.first).norm() / b.second;
    if ((q <= 1.5 || farOrder(c.x, b) >= opt_.n_gauss + 1 || needsSplit(c.x, b)) &&
        depth < opt_.near_max_depth) {
      const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
      adaptiveRect(c, e, {R.u0, um, R.v0, vm}, depth + 1, f);
      adaptiveRect(c, e, {um, R.u1, R.v0, vm}, depth + 1, f);
      adaptiveRect(c, e, {R.u0, um, vm, R.v1}, depth + 1, f);
      adaptiveRect(c, e, {um, R.u1, vm, R.v1}, depth + 1, f);
      return;
    }
    gaussRect(c.x, e, R, b, f);
  }

  // True when even n_gauss points miss far_tol for this cell.
  bool needsSplit(const Vec3& x, const Ball& b) const {
    const double q = (x - b.first).norm() / b.second;
    if (q <= 1.5) return true;
    const double rho = q + std::sqrt(q * q - 1.0);
    return std::pow(rho, -2.0 * opt_.n_gauss) > opt_.far_tol;
  }

  template <class F>
  void nearRect(const SourcePoint& c, const Element& e, const Rect& R, const Vec2& xs,
                int depth, F& f) const {
    const Patch& Pa = mesh_.patches[e.patch];
    PatchEval E;
    Vec3 X[4];
    int k = 0;
    for (double u : {R.u0, R.u1})
      for (double v : {R.v0, R.v1}) { Pa.evalSpan(e.su, e.sv, u, v, E); X[k++] = E.x; }
    const double h = std::max((X[3] - X[0]).norm(), (X[2] - X[1]).norm());
    if (opt_.near_refine && depth < opt_.near_max_depth) {
      const Vec2 q = R.clamp(xs);
      Pa.evalSpan(e.su, e.sv, q(0), q(1), E);
      const double d = (E.x - c.x).norm();
      if (d <= 3.0 * h) {
        const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
        nearRect(c, e, {R.u0, um, R.v0, vm}, xs, depth + 1, f);
        nearRect(c, e, {um, R.u1, R.v0, vm}, xs, depth + 1, f);
        nearRect(c, e, {R.u0, um, vm, R.v1}, xs, depth + 1, f);
        nearRect(c, e, {um, R.u1, vm, R.v1}, xs, depth + 1, f);
        return;
      }
    }
    // Leaf: centre of the corners, radius 0.6 x diagonal (20% margin).
    const Ball b{0.25 * (X[0] + X[1] + X[2] + X[3]), 0.6 * h};
    gaussRect(c.x, e, R, b, f);
  }
};

}  // namespace igabem
