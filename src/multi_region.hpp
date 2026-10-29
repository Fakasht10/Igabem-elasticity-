// Two-region (multi-region) bodies: a host region with any number of enclosed
// inclusions, each of its own isotropic material, perfectly bonded (displacement
// continuity and traction equilibrium on the interfaces). The target is a 3D
// printed part: a solid wall shell (host) around a homogenized sparse-infill core
// (inclusion).
//
// Geometry: one .dat per region. The host's boundary is its outer surface plus the
// interface patches, oriented outward from the host (into the inclusion); the
// inclusion's .dat holds the same interface patches oriented outward from the
// inclusion (python/make_regions.py writes the pairs, transposing u and v). The
// interface must be conforming: the same patches and knots on both sides. Dofs
// are matched by coordinates and the bases are checked to coincide.
//
// Formulation: each region has its own SBIE, H u = G t, with its own Kelvin
// kernel. The inclusion's SBIE is collocated at the traction collocation points
// of all its patches, so its G_i is square and the inclusion is condensed to a
// collocation Dirichlet-to-Neumann map
//
//   t_i = S u_i,   S = G_i^{-1} H_i      (3 Nt x 3 M_i; S R = 0 for rigid R;
//                                          S = mu_i S(mu = 1) at fixed nu_i).
//
// On the interface the host traction is t_h = -t_i (opposite normals) and the
// displacement is shared, so the host rows become
//
//   (H_h + G_h[:, I_t] S E) u_h = G_h[:, outer] t_outer,
//
// E gathering the interface displacement from the host's coefficients. That is
// a single-region problem on the host mesh with a modified H and the interface
// columns of G removed; BcModel solves it unchanged (supports, springs, loads,
// reactions, free rigid modes), and the host rigid modes stay in the null space.
// condense() applies the map to any set of cached host rows; complete() fills the
// interface tractions of a host solution; inclusionSolution() returns the
// inclusion's boundary solution for post-processing in that region.
//
// Cost: the inclusion rows (cached per patch, the same entries BcModel uses for a
// support on that patch), one LU of the 3 Nt square G_i, and the product
// G_h[:, I_t] S (rows x 3 Nt x 3 M_i), done once per setup.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "interior.hpp"
#include "operators.hpp"
#include "sbie.hpp"
#include "solution.hpp"
#include "traction_space.hpp"

namespace igabem {

inline Material materialEnu(double E, double nu) {
  Material m;
  m.lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  m.mu = E / (2.0 * (1.0 + nu));
  return m;
}
inline double youngs(const Material& m) { return m.mu * (3.0 * m.lambda + 2.0 * m.mu) / (m.lambda + m.mu); }

// Isotropic effective material of a sparse infill of relative density rho:
// Gibson-Ashby scaling E* = C rho^n E_s, nu* = nu_c. Defaults (C = 1, n = 2,
// nu = 0.3) are the open-cell, bending-dominated law; gyroid fits in the
// literature fall between n = 1.5 and 2.5, so C and n are parameters, to be
// fitted to coupon tests or to a resolved unit-cell computation.
inline Material infillMaterial(const Material& solid, double rho, double C = 1.0, double n = 2.0,
                               double nu_c = 0.3) {
  return materialEnu(C * std::pow(rho, n) * youngs(solid), nu_c);
}

struct Inclusion {
  std::string label, geometry_file;
  Material mat;
  Mesh mesh;
  TractionSpace ts;
  std::vector<int> u_host;  // inclusion dof -> host dof
  std::vector<int> t_host;  // inclusion traction dof -> host traction dof
  Eigen::MatrixXd S;        // 3 ts.ndof x 3 mesh.ndof, t_i = S u_i
  double rcond_G = 0.0;     // reciprocal condition estimate of G_i (1-norm)
  double rigid_residual = 0.0;  // ||S R|| / (||S|| ||R||), R the rigid modes
  bool from_cache = false;
  double t_assemble = 0.0, t_load = 0.0, t_factor = 0.0;
};

class MultiRegion {
 public:
  std::vector<Inclusion> inc;
  std::vector<char> interface_patch;  // per host patch
  std::vector<int> t_interface;       // host traction components (3 j + c) on interfaces
  double t_condense = 0.0;

  // Adds an inclusion bonded to the host (mesh and traction space already built).
  void add(const Mesh& host, const TractionSpace& hts, const std::string& label, const std::string& geometry_file,
           const Material& mat, const SolverOptions& opt, const PartitionOptions& po = PartitionOptions{},
           const std::string& cache_dir = "cache") {
    if (interface_patch.empty()) interface_patch.assign(host.patches.size(), 0);
    Inclusion I;
    I.label = label;
    I.geometry_file = geometry_file;
    I.mat = mat;
    I.mesh.build(loadDat(geometry_file), po);
    I.ts.build(I.mesh);
    match(host, hts, I);

    // Inclusion rows at the traction collocation points of all its patches, for
    // the material scaled to mu = 1: T does not depend on mu and U scales as 1/mu
    // at fixed nu, so S(mu) = mu S(1) and a stiffness sweep at fixed nu (infill
    // density) reuses one cached assembly.
    Material unit;
    unit.lambda = mat.lambda / mat.mu;
    unit.mu = 1.0;
    Eigen::MatrixXd Hi(3 * I.ts.ndof, 3 * I.mesh.ndof), Gi(3 * I.ts.ndof, 3 * I.ts.ndof);
    I.from_cache = true;
    int r = 0;
    for (int p = 0; p < int(I.mesh.patches.size()); ++p) {
      OperatorRows op = cachedOperatorRows(I.mesh, I.ts, unit, opt, po, geometry_file, cache_dir,
                                           "sbie_hg_tpatch" + std::to_string(p), tractionCollocation(I.mesh, I.ts, p));
      I.from_cache = I.from_cache && op.from_cache;
      I.t_assemble += op.t_assemble;
      I.t_load += op.t_load;
      Hi.middleRows(r, op.H.rows()) = op.H;
      Gi.middleRows(r, op.G.rows()) = op.G;
      r += int(op.H.rows());
    }
    const auto t1 = std::chrono::steady_clock::now();
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(Gi);
    I.rcond_G = lu.rcond();
    I.S = mat.mu * lu.solve(Hi);
    const Eigen::MatrixXd R = rigidModes(I.mesh);
    I.rigid_residual = (I.S * R).norm() / (I.S.norm() * R.norm());
    I.t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    for (int j : I.t_host) {
      interface_patch[hts.points[j].patch] = 1;
      for (int c = 0; c < 3; ++c) t_interface.push_back(3 * j + c);
    }
    inc.push_back(std::move(I));
  }

  // New stiffness for inclusion k at the same Poisson ratio (an infill density
  // sweep): S scales with mu, nothing is reassembled or refactored.
  void scaleStiffness(int k, const Material& m) {
    Inclusion& I = inc[k];
    if (std::abs(m.nu() - I.mat.nu()) > 1e-12)
      throw std::runtime_error("MultiRegion::scaleStiffness: Poisson ratio changed, add the inclusion again");
    I.S *= m.mu / I.mat.mu;
    I.mat = m;
  }

  bool empty() const { return inc.empty(); }
  bool onInterface(int host_patch) const {
    return !interface_patch.empty() && interface_patch[host_patch];
  }

  // H <- H + G[:, I_t] S E, G[:, I_t] <- 0, for any rows of the host SBIE.
  void condense(Eigen::MatrixXd& H, Eigen::MatrixXd& G) const {
    const auto t0 = std::chrono::steady_clock::now();
    for (const Inclusion& I : inc) {
      std::vector<int> tc, uc;
      for (int j : I.t_host)
        for (int c = 0; c < 3; ++c) tc.push_back(3 * j + c);
      for (int g : I.u_host)
        for (int c = 0; c < 3; ++c) uc.push_back(3 * g + c);
      const Eigen::MatrixXd W = G(Eigen::all, tc) * I.S;
      H(Eigen::all, uc) += W;
      G(Eigen::all, tc).setZero();
    }
    const_cast<MultiRegion*>(this)->t_condense +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // Interface tractions of a host solution (one column per case), from its displacement.
  void complete(const Eigen::MatrixXd& U, Eigen::MatrixXd& T) const {
    for (const Inclusion& I : inc) {
      const Eigen::MatrixXd Ti = I.S * gatherU(I, U);
      for (size_t j = 0; j < I.t_host.size(); ++j) T.middleRows(3 * I.t_host[j], 3) = -Ti.middleRows(3 * j, 3);
    }
  }

  // Boundary solution of inclusion k for a host solution.
  BoundarySolution inclusionSolution(int k, const BoundarySolution& host) const {
    const Inclusion& I = inc[k];
    const Eigen::VectorXd u = gatherU(I, host.u);
    return BoundarySolution{&I.mesh, &I.ts, I.mat, u, I.S * u, {}};
  }

  // Resultant force and moment (about the origin) of an inclusion's traction:
  // both vanish for an inclusion in equilibrium (discretization error otherwise).
  std::pair<Vec3, Vec3> inclusionResultant(const BoundarySolution& s) const {
    Vec3 F = Vec3::Zero(), M = Vec3::Zero();
    const GaussRule g = gaussLegendre(6);
    PatchEval E;
    for (const Element& e : s.mesh->elements)
      for (const Rect& R : e.cells)
        for (int a = 0; a < 6; ++a)
          for (int b = 0; b < 6; ++b) {
            s.mesh->patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
            const Vec3 f = s.traction(e.patch, E) * (g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm());
            F += f;
            M += E.x.cross(f);
          }
    return {F, M};
  }

 private:
  template <class Mat>
  static Eigen::MatrixXd gatherU(const Inclusion& I, const Mat& U) {
    Eigen::MatrixXd u(3 * I.mesh.ndof, U.cols());
    for (size_t g = 0; g < I.u_host.size(); ++g) u.middleRows(3 * g, 3) = U.middleRows(3 * I.u_host[g], 3);
    return u;
  }

  // Index of the point of P (sorted by x, order) within tol of X, or -1.
  static int nearest(const std::vector<Vec3>& P, const std::vector<int>& order, const Vec3& X, double tol) {
    auto lo = std::lower_bound(order.begin(), order.end(), X(0) - tol,
                               [&](int i, double v) { return P[i](0) < v; });
    for (auto it = lo; it != order.end() && P[*it](0) <= X(0) + tol; ++it)
      if ((P[*it] - X).norm() <= tol) return *it;
    return -1;
  }

  static std::vector<int> sortedByX(const std::vector<Vec3>& P) {
    std::vector<int> o(P.size());
    for (size_t i = 0; i < P.size(); ++i) o[i] = int(i);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return P[a](0) < P[b](0); });
    return o;
  }

  // Interface correspondence: displacement dofs by control point, traction dofs by
  // collocation point; then checks that the bases coincide (basis values at the
  // matched collocation points agree) and that the two sides face each other.
  static void match(const Mesh& host, const TractionSpace& hts, Inclusion& I) {
    double L = 0.0;
    for (const Vec3& P : I.mesh.dof_point) L = std::max(L, P.norm());
    const double tol = 1e-9 * std::max(1.0, L);
    const std::vector<int> ou = sortedByX(host.dof_point);
    I.u_host.resize(I.mesh.ndof);
    for (int g = 0; g < I.mesh.ndof; ++g) {
      I.u_host[g] = nearest(host.dof_point, ou, I.mesh.dof_point[g], tol);
      if (I.u_host[g] < 0)
        throw std::runtime_error("MultiRegion: inclusion " + I.label + " control point not on the host boundary "
                                 "(the interface must be conforming)");
    }
    std::vector<Vec3> hx(hts.points.size());
    for (size_t j = 0; j < hx.size(); ++j) hx[j] = hts.points[j].x;
    const std::vector<int> ot = sortedByX(hx);
    I.t_host.resize(I.ts.ndof);
    for (int j = 0; j < I.ts.ndof; ++j) {
      I.t_host[j] = nearest(hx, ot, I.ts.points[j].x, tol);
      if (I.t_host[j] < 0) throw std::runtime_error("MultiRegion: inclusion " + I.label + " patch not on the host");
    }
    for (int p = 0; p < int(I.mesh.patches.size()); ++p) {
      const int n0 = I.ts.offset[p], n = int(I.ts.lu[p].rows());
      const int hp = hts.points[I.t_host[n0]].patch, h0 = hts.offset[hp];
      if (int(hts.lu[hp].rows()) != n) throw std::runtime_error("MultiRegion: interface patches differ in size");
      // local index map inclusion -> host on this patch
      std::vector<int> loc(n);
      for (int i = 0; i < n; ++i) {
        if (hts.points[I.t_host[n0 + i]].patch != hp) throw std::runtime_error("MultiRegion: patch maps to two");
        loc[i] = I.t_host[n0 + i] - h0;
      }
      // basis values at the collocation points must agree under the map
      double err = 0.0;
      for (int i = 0; i < n; ++i) {
        const PatchEval& Ei = I.ts.points[n0 + i].E;
        const PatchEval& Eh = hts.points[h0 + loc[i]].E;
        Eigen::VectorXd vi = Eigen::VectorXd::Zero(n), vh = Eigen::VectorXd::Zero(n);
        for (int l = 0; l < Ei.n_loc; ++l) vi(Ei.idx[l]) += Ei.R[l];
        for (int l = 0; l < Eh.n_loc; ++l) vh(Eh.idx[l]) += Eh.R[l];
        for (int k = 0; k < n; ++k) err = std::max(err, std::abs(vi(k) - vh(loc[k])));
        if (I.ts.points[n0 + i].N.dot(hts.points[h0 + loc[i]].N) > -0.999)
          throw std::runtime_error("MultiRegion: interface normals must be opposite (host: into the inclusion)");
      }
      if (err > 1e-10) throw std::runtime_error("MultiRegion: interface bases differ (non-conforming interface)");
    }
  }
};

// Interior fields of a multi-region body: the region containing x is the one
// whose solid-angle integral is 1. Inclusions are tried first (bounding box
// test, then the integral), then the host.
struct RegionSample {
  FieldSample f;
  int region = 0;  // 0 host, k + 1 inclusion k
};

class MultiRegionEvaluator {
 public:
  MultiRegionEvaluator(const MultiRegion& mr, const BoundarySolution& host, const SolverOptions& o)
      : host_(host, o) {
    for (int k = 0; k < int(mr.inc.size()); ++k) sol_.push_back(mr.inclusionSolution(k, host));
    ev_.reserve(sol_.size());
    for (const auto& s : sol_) {
      ev_.emplace_back(s, o);
      Eigen::AlignedBox3d b;
      for (const Vec3& P : s.mesh->dof_point) b.extend(P);
      box_.push_back(b);
    }
  }

  RegionSample eval(const Vec3& x) const {
    for (size_t k = 0; k < ev_.size(); ++k) {
      if (!box_[k].contains(x)) continue;
      RegionSample r{ev_[k].eval(x), int(k) + 1};
      if (r.f.winding > 0.5 || r.f.on_boundary) return r;
    }
    return RegionSample{host_.eval(x), 0};
  }

  std::vector<RegionSample> eval(const std::vector<Vec3>& xs) const {
    std::vector<RegionSample> out(xs.size());
#pragma omp parallel for schedule(dynamic)
    for (long i = 0; i < long(xs.size()); ++i) out[i] = eval(xs[i]);
    return out;
  }

  const BoundarySolution& inclusion(int k) const { return sol_[k]; }

 private:
  InteriorEvaluator host_;
  std::vector<BoundarySolution> sol_;  // kept alive: the evaluators hold references
  std::vector<InteriorEvaluator> ev_;
  std::vector<Eigen::AlignedBox3d> box_;
};

}  // namespace igabem
