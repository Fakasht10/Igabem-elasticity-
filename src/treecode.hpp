// Stress at many interior points (Weibull integrals, TET10 nodes for the failure layer) by a barycentric
// Lagrange treecode: a port of nurb-fea-tcd's bemkit/volume.py (MIT, Aleksey Kashtelyan; Wang, Krasny & Tlupova
// 2020, BLTC) from curved tri6 elements to spline elements.
//
//   sources   the tensor Gauss points (order M) of every element, each with 15 densities w [t, u n^T, n]: the
//             stress kernel is linear in t and in m = u n^T, sigma = D(r) t - S(r) m (bemkit interior._acc_tm),
//             and the solid angle (inside test) is linear in n
//   tree      bisection of the elements by centroid (leaves <= leaf elements); per node the box of its sources and
//             a safe radius max_e(|c_e - c_C| + q_far r_e): beyond it every element of C is "far" for M points
//   proxies   (p+1)^3 Chebyshev points (2nd kind) on a node's box, charges Q_m = sum_q L_m(y_q) rho_q
//   traversal per target batch B (bisection of the points, <= batch points):
//             all of C far from all of B: proxies if (r_B + r_C) <= theta |c_B - c_C| and C has more sources
//             than proxies, else its sources directly (leaves) or the children;
//             otherwise, in a leaf: per (target, element) the sources when far, else the plain Somigliana kernel
//             with the adaptive near rule of ElementIntegrator (the one InteriorEvaluator uses)
//
// Differences from bemkit: elements are Bezier spans of the spline patches (sources from PatchEval), and a
// target closer to the surface than `near_frac` x the element size takes InteriorEvaluator's direct evaluation
// with the linear-field subtraction (exact to the boundary), since the plain kernel loses accuracy there.
// Loads on sub-regions (RegionDensity) are added per target as InteriorEvaluator does.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <vector>

#include "interior.hpp"

namespace igabem {

class TreeEvaluator {
 public:
  static constexpr int ND = 15;  // densities per source: t (3), u n^T (9, row-major), n (3); all times w |J|
  using Dens = std::array<double, ND>;

  TreeEvaluator(const BoundarySolution& s, const SolverOptions& o, int p = 8, double theta = 0.4, int leaf = 12,
                int batch = 64)
      : s_(s), m_(*s.mesh), opt_(o), direct_(s, o), quad_(*s.mesh, o), p_(p), theta_(theta), leaf_(leaf),
        batch_(batch) {
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& d : s.region) region_.emplace_back(*d->R, o);
    const double nu = s.mat.nu(), mu = s.mat.mu;
    C_ = {1.0 / (8.0 * M_PI * (1.0 - nu)), mu / (4.0 * M_PI * (1.0 - nu)), 1.0 - 2.0 * nu, nu};
    // far rule: order M is enough (far_tol) for every element seen from beyond q_far its radius
    const double tol = std::max(o.far_tol, src_tol), rho = q_far_ + std::sqrt(q_far_ * q_far_ - 1.0);
    M_ = std::max(4, int(std::ceil(std::log(1.0 / tol) / (2.0 * std::log(rho)))));
    for (const auto& Pa : m_.patches) M_ = std::max(M_, std::max(Pa.p, Pa.q) + 2);
    buildSources();
    buildTree();
    t_setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // Relative accuracy of an element's far rule at q_far (sets the source order M; far_tol if larger).
  static constexpr double src_tol = 1e-8;
  // Fraction of an element's size below which a target takes the direct (subtracted) evaluation.
  double near_frac = 0.05;
  double t_setup = 0.0;
  struct Stats {
    long proxy = 0, far_src = 0, near_pairs = 0, direct_points = 0;
  };
  mutable Stats stats;

  std::vector<FieldSample> eval(const std::vector<Vec3>& xs, const std::vector<char>& on_surface = {}) const {
    const long n = long(xs.size());
    std::vector<FieldSample> out(n);
    std::vector<char> direct(n, 0);
    for (long i = 0; i < n; ++i) direct[i] = !on_surface.empty() && on_surface[i];
    // targets near the surface: direct evaluation (with the subtraction), found from the elements near each one
#pragma omp parallel for schedule(dynamic, 64)
    for (long i = 0; i < n; ++i) {
      if (direct[i]) continue;
      if (tooClose(xs[i])) direct[i] = 2;
    }
    std::vector<long> tree_idx;
    for (long i = 0; i < n; ++i)
      if (!direct[i]) tree_idx.push_back(i);
    // batches: bisection of the tree targets
    std::vector<std::pair<long, long>> batches;
    bisect(xs, tree_idx, 0, long(tree_idx.size()), batches);
    std::vector<Eigen::Matrix<double, 7, 1>> acc(n, Eigen::Matrix<double, 7, 1>::Zero());
    Stats st;
#pragma omp parallel
    {
      Stats loc;
#pragma omp for schedule(dynamic, 1)
      for (long b = 0; b < long(batches.size()); ++b) {
        const auto [lo, hi] = batches[b];
        Vec3 bmin = xs[tree_idx[lo]], bmax = bmin;
        for (long k = lo; k < hi; ++k) {
          bmin = bmin.cwiseMin(xs[tree_idx[k]]);
          bmax = bmax.cwiseMax(xs[tree_idx[k]]);
        }
        const Vec3 cB = 0.5 * (bmin + bmax);
        double rB = 0.0;
        for (long k = lo; k < hi; ++k) rB = std::max(rB, (xs[tree_idx[k]] - cB).norm());
        traverse(0, xs, tree_idx, lo, hi, cB, rB, acc, loc);
      }
#pragma omp critical
      {
        st.proxy += loc.proxy;
        st.far_src += loc.far_src;
        st.near_pairs += loc.near_pairs;
      }
    }
    // regions (local loads) per target, then strain/stress; the direct ones from InteriorEvaluator
#pragma omp parallel for schedule(dynamic, 16)
    for (long i = 0; i < n; ++i) {
      if (direct[i]) {
        out[i] = direct[i] == 1 ? direct_.evalSurface(xs[i]) : direct_.eval(xs[i]);
        continue;
      }
      Eigen::Matrix<double, 7, 1> a = acc[i];
      const SourcePoint src{-1, Vec2(0.0, 0.0), xs[i]};
      for (size_t k = 0; k < region_.size(); ++k)
        region_[k].visit(src, *s_.region[k], [&](const Vec3& y, const Vec3& nn, const RegionBlock& tw) {
          const Vec3 tv = tw.col(0);
          Dens d{};
          d[0] = tv(0), d[1] = tv(1), d[2] = tv(2);
          kernel(y - xs[i], d.data(), a.data(), false);
        });
      FieldSample f;
      f.x = xs[i];
      f.stress << a(0), a(3), a(5), a(3), a(1), a(4), a(5), a(4), a(2);
      const double lam = s_.mat.lambda, mu = s_.mat.mu;
      f.strain = (f.stress - lam / (3.0 * lam + 2.0 * mu) * f.stress.trace() * Eigen::Matrix3d::Identity()) / (2.0 * mu);
      f.grad = f.strain;  // symmetric part only: the stress kernel carries no rotation
      f.u = Vec3::Constant(std::nan(""));
      f.winding = a(6);
      out[i] = f;
    }
    st.direct_points = std::count_if(direct.begin(), direct.end(), [](char c) { return c != 0; });
    stats = st;
    return out;
  }

 private:
  const BoundarySolution& s_;
  const Mesh& m_;
  SolverOptions opt_;
  InteriorEvaluator direct_;
  ElementIntegrator quad_;
  std::vector<RegionIntegrator> region_;
  int p_;
  double theta_;
  int leaf_, batch_;
  double q_far_ = 2.5;  // element far beyond q_far x its radius
  int M_ = 8;
  std::array<double, 4> C_;

  // sources
  std::vector<Vec3> sx_;
  std::vector<Dens> sd_;
  std::vector<long> eoff_;  // sources of element e: [eoff_[e], eoff_[e+1])
  std::vector<Vec3> ec_;    // element centre
  std::vector<double> er_;  // element radius (ball containing it)
  // tree
  struct Node {
    long e0, e1;  // range in perm_
    int kid[2] = {-1, -1};
    Vec3 lo, hi, ctr;
    double rC, safe;
    long nsrc;
    int proxy = -1;
  };
  std::vector<Node> nodes_;
  std::vector<long> perm_;
  std::vector<double> cheb_, wb_;
  std::vector<std::vector<Vec3>> px_;  // proxy points per proxy node
  std::vector<std::vector<Dens>> pq_;  // proxy charges

  // sigma (6: xx yy zz xy yz zx) += D(r) t - S(r) m, r = y - x (bemkit interior._acc_tm), and the solid angle
  // density (y - x).n / (4 pi r^3) into a[6] when `wind`.
  void kernel(const Vec3& r, const double* d, double* a, bool wind = true) const {
    const double c1 = C_[0], c2 = C_[1], b = C_[2], nu = C_[3];
    const double t0 = d[0], t1 = d[1], t2 = d[2];
    const double m00 = d[3], m01 = d[4], m02 = d[5], m10 = d[6], m11 = d[7], m12 = d[8], m20 = d[9], m21 = d[10],
                 m22 = d[11];
    const double P00 = 2 * m00, P11 = 2 * m11, P22 = 2 * m22, P01 = m01 + m10, P12 = m12 + m21, P02 = m02 + m20;
    const double trm = m00 + m11 + m22;
    const double r2 = r.squaredNorm(), ri = 1.0 / std::sqrt(r2);
    const double g0 = r(0) * ri, g1 = r(1) * ri, g2 = r(2) * ri;
    const double tg = t0 * g0 + t1 * g1 + t2 * g2;
    const double q0 = P00 * g0 + P01 * g1 + P02 * g2, q1 = P01 * g0 + P11 * g1 + P12 * g2,
                 q2 = P02 * g0 + P12 * g1 + P22 * g2;
    const double gmg = 0.5 * (q0 * g0 + q1 * g1 + q2 * g2);
    const double fd = c1 * ri * ri, fs = c2 * ri * ri * ri;
    const double h0 = b * fd * t0 - 3 * nu * fs * q0, h1 = b * fd * t1 - 3 * nu * fs * q1,
                 h2 = b * fd * t2 - 3 * nu * fs * q2;
    const double K = 3 * fd * tg + fs * (15 * gmg - 3 * b * trm);
    const double dd = -b * tg * fd - fs * (3 * b * gmg - (1 - 4 * nu) * trm);
    const double fb = -b * fs;
    a[0] += dd + 2 * h0 * g0 + K * g0 * g0 + fb * P00;
    a[1] += dd + 2 * h1 * g1 + K * g1 * g1 + fb * P11;
    a[2] += dd + 2 * h2 * g2 + K * g2 * g2 + fb * P22;
    a[3] += h0 * g1 + g0 * h1 + K * g0 * g1 + fb * P01;
    a[4] += h1 * g2 + g1 * h2 + K * g1 * g2 + fb * P12;
    a[5] += h2 * g0 + g2 * h0 + K * g2 * g0 + fb * P02;
    if (wind) a[6] += (r(0) * d[12] + r(1) * d[13] + r(2) * d[14]) * ri * ri * ri / (4.0 * M_PI);
  }

  Dens density(const Vec3& u, const Vec3& t, const Vec3& n, double w) const {
    Dens d;
    for (int k = 0; k < 3; ++k) {
      d[k] = w * t(k);
      for (int l = 0; l < 3; ++l) d[3 + 3 * k + l] = w * u(k) * n(l);
      d[12 + k] = w * n(k);
    }
    return d;
  }

  void buildSources() {
    const GaussRule g = gaussLegendre(M_);
    const size_t ne = m_.elements.size();
    eoff_.assign(ne + 1, 0);
    ec_.resize(ne);
    er_.resize(ne);
    for (size_t ie = 0; ie < ne; ++ie) {
      const Element& e = m_.elements[ie];
      const Patch& Pa = m_.patches[e.patch];
      const int off = s_.ts->offset[e.patch];
      const auto& l2g = m_.l2g[e.patch];
      PatchEval E;
      eoff_[ie] = long(sx_.size());
      // centre and radius from a 5 x 5 grid (padded: the span bulges between samples)
      Vec3 c = Vec3::Zero();
      std::vector<Vec3> G;
      for (int a = 0; a <= 4; ++a)
        for (int b = 0; b <= 4; ++b) {
          Pa.evalSpan(e.su, e.sv, e.rect.u0 + e.rect.du() * a / 4.0, e.rect.v0 + e.rect.dv() * b / 4.0, E);
          G.push_back(E.x);
          c += E.x;
        }
      c /= double(G.size());
      double r = 0.0;
      for (const auto& x : G) r = std::max(r, (x - c).norm());
      ec_[ie] = c;
      er_[ie] = 1.1 * r;
      for (int a = 0; a < M_; ++a)
        for (int b = 0; b < M_; ++b) {
          const double u = e.rect.u0 + e.rect.du() * g.x[a], v = e.rect.v0 + e.rect.dv() * g.x[b];
          Pa.evalSpan(e.su, e.sv, u, v, E);
          const Vec3 nJ = E.normal() * m_.normal_sign;
          const double J = nJ.norm(), w = g.w[a] * g.w[b] * e.rect.du() * e.rect.dv() * J;
          Vec3 uu = Vec3::Zero(), tt = Vec3::Zero();
          for (int l = 0; l < E.n_loc; ++l) {
            uu += E.R[l] * s_.u.segment<3>(3 * l2g[E.idx[l]]);
            tt += E.R[l] * s_.t.segment<3>(3 * (off + E.idx[l]));
          }
          sx_.push_back(E.x);
          sd_.push_back(density(uu, tt, nJ / J, w));
        }
    }
    eoff_[ne] = long(sx_.size());
  }

  int build(long e0, long e1) {
    const int id = int(nodes_.size());
    nodes_.push_back(Node{});
    nodes_[id].e0 = e0;
    nodes_[id].e1 = e1;
    if (e1 - e0 > leaf_) {
      Vec3 lo = ec_[perm_[e0]], hi = lo;
      for (long k = e0; k < e1; ++k) {
        lo = lo.cwiseMin(ec_[perm_[k]]);
        hi = hi.cwiseMax(ec_[perm_[k]]);
      }
      int ax;
      (hi - lo).maxCoeff(&ax);
      std::stable_sort(perm_.begin() + e0, perm_.begin() + e1,
                       [&](long a, long b) { return ec_[a](ax) < ec_[b](ax); });
      const long h = (e1 - e0) / 2;
      const int a = build(e0, e0 + h), b = build(e0 + h, e1);
      nodes_[id].kid[0] = a;
      nodes_[id].kid[1] = b;
    }
    return id;
  }

  void buildTree() {
    const long ne = long(m_.elements.size());
    perm_.resize(ne);
    for (long i = 0; i < ne; ++i) perm_[i] = i;
    build(0, ne);
    cheb_.resize(p_ + 1);
    wb_.resize(p_ + 1);
    for (int k = 0; k <= p_; ++k) {
      cheb_[k] = std::cos(M_PI * k / p_);
      wb_[k] = (k % 2 ? -1.0 : 1.0) * ((k == 0 || k == p_) ? 0.5 : 1.0);
    }
    const long np3 = long(p_ + 1) * (p_ + 1) * (p_ + 1);
    std::vector<int> want;
    for (auto& nd : nodes_) {
      Vec3 lo = Vec3::Constant(1e300), hi = Vec3::Constant(-1e300);
      nd.nsrc = 0;
      for (long k = nd.e0; k < nd.e1; ++k) {
        const long e = perm_[k];
        for (long q = eoff_[e]; q < eoff_[e + 1]; ++q) {
          lo = lo.cwiseMin(sx_[q]);
          hi = hi.cwiseMax(sx_[q]);
        }
        nd.nsrc += eoff_[e + 1] - eoff_[e];
      }
      const Vec3 ext = hi - lo;
      const double mn = 1e-3 * ext.maxCoeff();
      const Vec3 pad = (Vec3::Constant(mn) - ext).cwiseMax(0.0) / 2.0;
      nd.lo = lo - pad;
      nd.hi = hi + pad;
      nd.ctr = 0.5 * (nd.lo + nd.hi);
      nd.rC = 0.5 * (nd.hi - nd.lo).norm();
      nd.safe = 0.0;
      for (long k = nd.e0; k < nd.e1; ++k) {
        const long e = perm_[k];
        nd.safe = std::max(nd.safe, (ec_[e] - nd.ctr).norm() + q_far_ * er_[e]);
      }
      if (nd.nsrc > np3) {
        nd.proxy = int(px_.size());
        px_.emplace_back();
        pq_.emplace_back();
        want.push_back(int(&nd - nodes_.data()));
      }
    }
    // proxy points and charges (directly from the sources: L is separable, so per source (p+1) x 3 weights)
    px_.assign(want.size(), {});
    pq_.assign(want.size(), {});
#pragma omp parallel for schedule(dynamic, 1)
    for (long w = 0; w < long(want.size()); ++w) {
      const Node& nd = nodes_[want[w]];
      const int n1 = p_ + 1;
      std::vector<Vec3> P(np3);
      Eigen::Matrix<double, Eigen::Dynamic, 3> S(n1, 3);
      for (int a = 0; a < n1; ++a)
        for (int c = 0; c < 3; ++c) S(a, c) = nd.ctr(c) + 0.5 * (nd.hi(c) - nd.lo(c)) * cheb_[a];
      for (int a = 0; a < n1; ++a)
        for (int b = 0; b < n1; ++b)
          for (int c = 0; c < n1; ++c) P[(a * n1 + b) * n1 + c] = Vec3(S(a, 0), S(b, 1), S(c, 2));
      std::vector<Dens> Q(np3);
      for (auto& q : Q) q.fill(0.0);
      std::vector<double> L(3 * n1);
      for (long k = nd.e0; k < nd.e1; ++k) {
        const long e = perm_[k];
        for (long q = eoff_[e]; q < eoff_[e + 1]; ++q) {
          for (int c = 0; c < 3; ++c) lagrange(sx_[q](c), S.col(c), &L[c * n1]);
          const Dens& d = sd_[q];
          for (int a = 0; a < n1; ++a)
            for (int b = 0; b < n1; ++b) {
              const double lab = L[a] * L[n1 + b];
              for (int c = 0; c < n1; ++c) {
                const double l = lab * L[2 * n1 + c];
                Dens& Qm = Q[(a * n1 + b) * n1 + c];
                for (int j = 0; j < ND; ++j) Qm[j] += l * d[j];
              }
            }
        }
      }
      px_[w] = std::move(P);
      pq_[w] = std::move(Q);
    }
  }

  // barycentric Lagrange weights of y on the Chebyshev nodes s (2nd kind)
  void lagrange(double y, const Eigen::Ref<const Eigen::VectorXd>& s, double* L) const {
    const int n1 = p_ + 1;
    double den = 0.0;
    for (int a = 0; a < n1; ++a) {
      const double dy = y - s(a);
      if (std::abs(dy) < 1e-14 * (1.0 + std::abs(y))) {
        for (int b = 0; b < n1; ++b) L[b] = 0.0;
        L[a] = 1.0;
        return;
      }
      L[a] = wb_[a] / dy;
      den += L[a];
    }
    for (int a = 0; a < n1; ++a) L[a] /= den;
  }

  bool tooClose(const Vec3& x) const {
    // elements within q_far of x (tree walk on the element balls); closest point on each
    std::vector<int> stack{0};
    while (!stack.empty()) {
      const Node& nd = nodes_[stack.back()];
      stack.pop_back();
      if ((x - nd.ctr).norm() >= nd.safe) continue;
      if (nd.kid[0] < 0) {
        for (long k = nd.e0; k < nd.e1; ++k) {
          const long e = perm_[k];
          if ((x - ec_[e]).norm() >= q_far_ * er_[e]) continue;
          double d;
          m_.closestPoint(m_.elements[e], x, d);
          if (d < near_frac * m_.elements[e].h) return true;
        }
      } else {
        stack.push_back(nd.kid[0]);
        stack.push_back(nd.kid[1]);
      }
    }
    return false;
  }

  void bisect(const std::vector<Vec3>& xs, std::vector<long>& idx, long lo, long hi,
              std::vector<std::pair<long, long>>& out) const {
    if (hi - lo <= batch_) {
      if (hi > lo) out.push_back({lo, hi});
      return;
    }
    Vec3 a = xs[idx[lo]], b = a;
    for (long k = lo; k < hi; ++k) {
      a = a.cwiseMin(xs[idx[k]]);
      b = b.cwiseMax(xs[idx[k]]);
    }
    int ax;
    (b - a).maxCoeff(&ax);
    const long mid = (lo + hi) / 2;
    std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                     [&](long p, long q) { return xs[p](ax) < xs[q](ax); });
    bisect(xs, idx, lo, mid, out);
    bisect(xs, idx, mid, hi, out);
  }

  void sources(long e, const Vec3& x, double* a, Stats& st) const {
    for (long q = eoff_[e]; q < eoff_[e + 1]; ++q) kernel(sx_[q] - x, sd_[q].data(), a);
    st.far_src += eoff_[e + 1] - eoff_[e];
  }

  // plain Somigliana kernel on element e with the adaptive near rule
  void nearPair(long ie, const Vec3& x, double* a, Stats& st) const {
    const Element& e = m_.elements[ie];
    const int off = s_.ts->offset[e.patch];
    const auto& l2g = m_.l2g[e.patch];
    const SourcePoint src{-1, Vec2(0.0, 0.0), x};
    quad_.integrate(src, e, [&](const QPoint& q, double w) {
      const Vec3 nJ = q.normal() * m_.normal_sign;
      const double J = nJ.norm();
      Vec3 uu = Vec3::Zero(), tt = Vec3::Zero();
      for (int l = 0; l < q.n_loc; ++l) {
        uu += q.R[l] * s_.u.segment<3>(3 * l2g[q.idx[l]]);
        tt += q.R[l] * s_.t.segment<3>(3 * (off + q.idx[l]));
      }
      const Dens d = density(uu, tt, nJ / J, w * J);
      kernel(q.x - x, d.data(), a);
    });
    ++st.near_pairs;
  }

  void traverse(int c, const std::vector<Vec3>& xs, const std::vector<long>& idx, long lo, long hi, const Vec3& cB,
                double rB, std::vector<Eigen::Matrix<double, 7, 1>>& acc, Stats& st) const {
    const Node& nd = nodes_[c];
    const double dist = (cB - nd.ctr).norm();
    if (dist >= rB + nd.safe) {  // every element of C is far for every target of B
      if (nd.proxy >= 0 && rB + nd.rC <= theta_ * dist) {
        const auto& P = px_[nd.proxy];
        const auto& Q = pq_[nd.proxy];
        for (long k = lo; k < hi; ++k) {
          const Vec3& x = xs[idx[k]];
          double* a = acc[idx[k]].data();
          for (size_t m = 0; m < P.size(); ++m) kernel(P[m] - x, Q[m].data(), a);
        }
        st.proxy += (hi - lo) * long(P.size());
        return;
      }
      if (nd.kid[0] < 0) {
        for (long k = lo; k < hi; ++k)
          for (long j = nd.e0; j < nd.e1; ++j) sources(perm_[j], xs[idx[k]], acc[idx[k]].data(), st);
        return;
      }
    } else if (nd.kid[0] < 0) {
      for (long k = lo; k < hi; ++k) {
        const Vec3& x = xs[idx[k]];
        for (long j = nd.e0; j < nd.e1; ++j) {
          const long e = perm_[j];
          if ((x - ec_[e]).norm() >= q_far_ * er_[e]) sources(e, x, acc[idx[k]].data(), st);
          else nearPair(e, x, acc[idx[k]].data(), st);
        }
      }
      return;
    }
    traverse(nd.kid[0], xs, idx, lo, hi, cB, rB, acc, st);
    traverse(nd.kid[1], xs, idx, lo, hi, cB, rB, acc, st);
  }
};

}  // namespace igabem
