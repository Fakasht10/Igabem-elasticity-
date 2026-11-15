// Independent SIFs from the interior field: the interaction (M-) integral on a slice normal to the front.
//
// At a front point C with the frame (n, b, t) of front.hpp (n ahead of the front, b the crack normal, t = n x b),
// local coordinates x1 = n, x2 = b. The 2D domain form of the interaction integral over the annulus
// A = {rho1 < r < rho2} of the (n, b) plane through C, with q(r) = 1 inside rho1 and 0 outside rho2, is
//
//   I = int_A [ sigma_ij u^a_i,1 + sigma^a_ij u_i,1 - sigma^a_ik eps_ik delta_1j ] q_,j dA,     j = 1, 2
//
// (Yau, Wang & Corten 1980; Moran & Shih 1987), with the auxiliary field u^a the Williams displacement of
// front.hpp per unit K^a (plane strain, u^a independent of x3) and sigma^a its stress. On traction-free faces no
// face term arises, and q_,j vanishes outside the annulus, so only interior points away from the front and the
// faces are needed. Then
//
//   I = 2 (1 - nu^2) / E (K_I K^a_I + K_II K^a_II) + K_III K^a_III / mu,
//
// one auxiliary mode at a time gives K_I, K_II, K_III. In 3D the slice drops the x3 derivatives (sigma_i3 u_i,1),3
// and the front curvature; both enter as O(rho2) terms, removed by the linear extrapolation 2 K(rho / 2) - K(rho)
// of two annuli with the same rho1 / rho2.
//
// The routine is field-agnostic: grad(points, G) fills displacement gradients G(i, j) = du_i / dx_j (global axes)
// at the given points, so it is tested on the exact Williams field (tests/test_crack.cpp) and run on the BEM
// field by CrackField.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <vector>

#include "../integrator.hpp"
#include "../kelvin.hpp"
#include "../quadrature.hpp"
#include "dual_model.hpp"
#include "front.hpp"

namespace igabem {
namespace crack {

using GradFn = std::function<void(const std::vector<Vec3>&, std::vector<Eigen::Matrix3d>&)>;

struct InteractionOptions {
  double ratio = 0.25;  // rho1 / rho2
  int n_r = 4;          // Gauss points across the annulus
  int n_theta = 16;     // Gauss points in theta over (-pi, pi), split at 0
  bool curvature = false;  // tube metric of a curved front (off: no gain measured; the O(rho) slice error is removed by extrapolation)
};

struct SliceFrame {
  Vec3 C, n, b, t;
  Vec3 kappa = Vec3::Zero();  // front curvature vector (toward the centre of curvature)
};

inline SliceFrame sliceFrame(const Front& F, int k, double u) {
  Vec3 C, Cu, Cuu;
  int nn, s[16];
  double R[16];
  F.pieces[k].eval(u, C, Cu, nn, s, R, &Cuu);
  SliceFrame f;
  const Vec3 tu = Cu.normalized();
  f.kappa = (Cuu - Cuu.dot(tu) * tu) / Cu.squaredNorm();
  f.C = C;
  f.b = F.m;
  f.n = F.m.cross(Cu.normalized());
  if (f.n.dot(C - F.centroid) < 0) f.n = -f.n;
  f.t = f.n.cross(F.m);
  return f;
}

// Williams displacement gradient (local frame, per unit K_M) at polar (r, th), by central differences in x1, x2.
inline Eigen::Matrix3d williamsGrad(int M, double r, double th, double mu, double kappa) {
  const Vec3 e1 = Vec3::UnitX(), e2 = Vec3::UnitY(), e3 = Vec3::UnitZ();
  const double x1 = r * std::cos(th), x2 = r * std::sin(th), h = 1e-5 * r;
  auto u = [&](double a, double b) {
    Vec3 w[3];
    Front::williams(std::hypot(a, b), std::atan2(b, a), mu, kappa, e1, e2, e3, w);
    return w[M];
  };
  Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
  G.col(0) = (u(x1 + h, x2) - u(x1 - h, x2)) / (2 * h);
  G.col(1) = (u(x1, x2 + h) - u(x1, x2 - h)) / (2 * h);
  return G;
}

// The annulus points of one slice: positions, and weight times q_,j (j = 1, 2) and polar coordinates per point.
struct SliceRule {
  std::vector<Vec3> x;
  std::vector<double> w, q1, q2, r, th;
};

inline SliceRule sliceRule(const SliceFrame& f, double rho2, const InteractionOptions& o) {
  SliceRule S;
  const double rho1 = o.ratio * rho2, L = rho2 - rho1;
  const GaussRule gr = gaussLegendre(o.n_r);
  const GaussRule gt = gaussLegendre(o.n_theta / 2);
  for (int i = 0; i < o.n_r; ++i) {
    const double s = gr.x[i], r = rho1 + L * s;
    const double dq = -6.0 * s * (1.0 - s) / L;  // q = 1 - (3 s^2 - 2 s^3)
    for (int half = 0; half < 2; ++half)
      for (int j = 0; j < o.n_theta / 2; ++j) {
        const double th = half == 0 ? -M_PI + M_PI * gt.x[j] : M_PI * gt.x[j];
        const Vec3 er = std::cos(th) * f.n + std::sin(th) * f.b;
        // the slice stands for a thin tube along the front: the arclength of the parallel curve through the point
        // is (1 - r er.kappa) times the front's
        const double w = gr.w[i] * L * gt.w[j] * M_PI * r * (o.curvature ? 1.0 - r * er.dot(f.kappa) : 1.0);
        S.x.push_back(f.C + r * er);
        S.w.push_back(w);
        S.q1.push_back(dq * std::cos(th));
        S.q2.push_back(dq * std::sin(th));
        S.r.push_back(r);
        S.th.push_back(th);
      }
  }
  return S;
}

// (K_I, K_II, K_III) of one slice from gradients G at the rule's points (global axes).
inline Vec3 sliceK(const SliceFrame& f, const SliceRule& S, const std::vector<Eigen::Matrix3d>& G,
                   const Material& mat) {
  const double mu = mat.mu, nu = mat.lambda / (2 * (mat.lambda + mat.mu)), kappa = 3 - 4 * nu;
  const double E = 2 * mu * (1 + nu);
  Eigen::Matrix3d Rm;
  Rm << f.n, f.b, f.t;
  Vec3 I = Vec3::Zero();
  for (size_t p = 0; p < S.x.size(); ++p) {
    const Eigen::Matrix3d g = Rm.transpose() * G[p] * Rm;
    const Eigen::Matrix3d sg = mat.stress(g), eps = 0.5 * (g + g.transpose());
    for (int M = 0; M < 3; ++M) {
      const Eigen::Matrix3d ga = williamsGrad(M, S.r[p], S.th[p], mu, kappa);
      const Eigen::Matrix3d sa = mat.stress(ga);
      const double W = (sa.array() * eps.array()).sum();
      double v = 0;
      for (int j = 0; j < 2; ++j) {
        double c = 0;
        for (int i = 0; i < 3; ++i) c += sg(i, j) * ga(i, 0) + sa(i, j) * g(i, 0);
        if (j == 0) c -= W;
        v += c * (j == 0 ? S.q1[p] : S.q2[p]);
      }
      I(M) += S.w[p] * v;
    }
  }
  const double Es = E / (1 - nu * nu);
  return Vec3(0.5 * Es * I(0), 0.5 * Es * I(1), mu * I(2));
}

// K at each front station (piece, u) from the slice at rho and at rho / 2, linearly extrapolated to rho -> 0.
// Returns per station {K(rho), K(rho / 2), K_extrapolated}.
struct InteractionK {
  Vec3 K1, K2, K;
};

inline std::vector<InteractionK> interactionK(const Front& F, const std::vector<std::pair<int, double>>& stations,
                                              double rho, const Material& mat, const GradFn& grad,
                                              const InteractionOptions& o = {}) {
  std::vector<SliceFrame> fr;
  std::vector<SliceRule> rules;
  std::vector<Vec3> all;
  for (const auto& [k, u] : stations) {
    fr.push_back(sliceFrame(F, k, u));
    for (double r : {rho, 0.5 * rho}) {
      rules.push_back(sliceRule(fr.back(), r, o));
      all.insert(all.end(), rules.back().x.begin(), rules.back().x.end());
    }
  }
  std::vector<Eigen::Matrix3d> G(all.size());
  grad(all, G);
  std::vector<InteractionK> out;
  size_t at = 0;
  for (size_t s = 0; s < stations.size(); ++s) {
    Vec3 K[2];
    for (int h = 0; h < 2; ++h) {
      const SliceRule& S = rules[2 * s + h];
      std::vector<Eigen::Matrix3d> g(G.begin() + at, G.begin() + at + S.x.size());
      at += S.x.size();
      K[h] = sliceK(fr[s], S, g, mat);
    }
    out.push_back({K[0], K[1], 2 * K[1] - K[0]});
  }
  return out;
}

// Interior displacement gradient of a crack solution (Somigliana gradient identity over the outer boundary and
// both faces, enrichment included). Points must keep off the boundary; near-singular integrals use the adaptive
// quadtree (SolverOptions::far_tol > 0).
class CrackField {
 public:
  CrackField(const CrackModel& M, const Eigen::VectorXd& x, const TractionFn& load)
      : M_(M), x_(x), load_(load), KT_(M.mat), quad_(M.mesh, M.opt.quad) {}

  Eigen::Matrix3d grad(const Vec3& x) const {
    const SourcePoint src{-1, Vec2(0.0, 0.0), x};
    Eigen::Matrix3d dG = Eigen::Matrix3d::Zero();
    for (const Element& e : M_.mesh.elements) {
      const int pe = e.patch;
      quad_.integrate(src, e, [&](const QPoint& q, double w) {
        const Vec3 nJ = q.normal() * M_.mesh.normal_sign;
        const double J = nJ.norm();
        const Vec3 n = nJ / J;
        Vec3 u, t;
        M_.densities(pe, q, n, x_, load_, u, t);
        Eigen::Matrix3d P[3], Q[3];
        KT_.gradients(q.x - x, n, P, Q);
        const double wJ = w * J;
        for (int m = 0; m < 3; ++m) dG += wJ * (P[m] * t(m) - Q[m] * u(m));
      });
    }
    return dG;
  }

  void operator()(const std::vector<Vec3>& xs, std::vector<Eigen::Matrix3d>& G) const {
    G.resize(xs.size());
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < long(xs.size()); ++i) G[i] = grad(xs[i]);
  }

 private:
  const CrackModel& M_;
  const Eigen::VectorXd& x_;
  TractionFn load_;
  KelvinTraction KT_;
  ElementIntegrator quad_;
};

}  // namespace crack
}  // namespace igabem
