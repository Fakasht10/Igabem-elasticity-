// Bead-boundary periodic cells: every bead (or bead piece) its own BEM region, joined at the welds.
//
// A print is a lattice of identical beads in contact over flat welds. Here each bead is one region with its own
// SBIE; regions meet on connection patches, matched by coordinates:
//   periodic pairs   a patch on the box's - face a and its copy translated by L_a e_a on the + face (any regions)
//   interfaces       two coincident patches of different regions with opposite normals (a weld inside the box)
// Everything else is free surface (t = 0), including patches that lie on a box plane but have no partner (the
// free parts of a strip, for example). Every pair carries a linear traction-separation law
//   u+ = u-(x - s) + E s - C t+,    t- = -t+ (at matching points),
// + the side whose outward normal points along +e_a, s the translation (L_a e_a, or 0 for an interface), C the
// weld compliance (C = 0: a perfect weld, the regions act as one body). Perfect welds reproduce the single-region
// cell (src/rve.hpp) and spring welds give the stiffness of imperfectly bonded prints.
//
// Unknowns: one displacement per class of dofs tied by the pairs (union-find over all regions), and the + side
// traction of every pair. Rows: each class representative's dof collocation point (in its own region; the
// representative is a - side dof, so the - region's SBIE holds there) and every + side traction collocation point, so both sides of every pair carry equations (the rows of src/rve.hpp's PBC,
// generalized to several regions). The three translations are bordered out.
//
// Identical beads share one operator: a region declared as a rigid copy (rotation R, translation) of an earlier
// region with the same patch structure gets H and G by rotating that region's 3 x 3 blocks (isotropic Kelvin
// kernels are rotation- and translation-equivariant), with no assembly. The 0/90 cell's B bead is A turned 90 deg.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "operators.hpp"
#include "rve.hpp"
#include "sbie.hpp"
#include "solution.hpp"

namespace igabem {

struct BeadRegion {
  std::string file;
  Mesh mesh;
  TractionSpace ts;
  Eigen::MatrixXd H, G;            // rows: 3 per dof collocation point, then 3 per connection traction point
  std::vector<int> conn_row0;      // per patch: first row (point index) of its traction points, -1 if free
  std::vector<char> connection;    // per patch: 1 if a connection patch
  int copy_of = -1;                // >= 0: H, G rotated from that region
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  double t_assemble = 0.0;
};

struct BeadPair {
  int rm, pm, rp, pp;  // region and patch of the - side and the + side
  int axis;            // normal axis
  Vec3 shift;          // + = - shifted by this (L_a e_a, or 0)
  bool internal;
};

class BeadCell {
 public:
  std::vector<BeadRegion> regions;
  std::vector<BeadPair> pairs;
  Material mat;
  SolverOptions opt;
  PartitionOptions popt;
  Vec3 lo, hi;
  double volume_solid = 0.0;
  // weld compliances: periodic pairs by axis, interfaces by normal axis (3 x 3, mm^3 / N; zero = perfect)
  std::array<Eigen::Matrix3d, 3> weld_periodic{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
  std::array<Eigen::Matrix3d, 3> weld_internal{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};

  static Eigen::Matrix3d springCompliance(int a, double Kn, double Kt) {
    const Eigen::Matrix3d nn = Vec3::Unit(a) * Vec3::Unit(a).transpose();
    return Kn > 0 ? Eigen::Matrix3d(nn / Kn + (Eigen::Matrix3d::Identity() - nn) / Kt) : Eigen::Matrix3d::Zero();
  }

  Vec3 lengths() const { return hi - lo; }
  double volume() const { const Vec3 L = lengths(); return L(0) * L(1) * L(2); }
  double porosity() const { return 1.0 - volume_solid / volume(); }
  int ndofTotal() const { int n = 0; for (auto& r : regions) n += r.mesh.ndof; return n; }

  // files[i]: region geometry; copies[i] = (j, R): region i is region j rotated by R (and translated), -1: assemble.
  void setup(const std::vector<std::string>& files, const Material& m, const SolverOptions& o,
             const std::vector<std::pair<int, Eigen::Matrix3d>>& copies = {}, const std::string& cache_dir = "") {
    mat = m;
    opt = o;
    regions.assign(files.size(), BeadRegion{});
    bool first = true;
    for (size_t i = 0; i < files.size(); ++i) {
      BeadRegion& r = regions[i];
      r.file = files[i];
      r.mesh.build(loadDat(files[i]), popt);
      r.ts.build(r.mesh);
      volume_solid += r.mesh.volume();
      for (const Vec3& P : r.mesh.dof_point) {
        if (first) { lo = hi = P; first = false; }
        lo = lo.cwiseMin(P);
        hi = hi.cwiseMax(P);
      }
      if (i < copies.size() && copies[i].first >= 0) { r.copy_of = copies[i].first; r.R = copies[i].second; }
    }
    tol_ = 1e-9 * (hi - lo).maxCoeff();
    findPairs();
    for (size_t i = 0; i < regions.size(); ++i) {
      BeadRegion& r = regions[i];
      const int np = int(r.mesh.patches.size());
      r.conn_row0.assign(np, -1);
      std::vector<Collocation> rows = r.mesh.colloc;
      for (int p = 0; p < np; ++p) {
        if (!r.connection[p]) continue;
        r.conn_row0[p] = int(rows.size());
        for (auto& c : tractionCollocation(r.mesh, r.ts, p)) rows.push_back(std::move(c));
      }
      if (r.copy_of >= 0) {
        copyRotated(regions[r.copy_of], r, int(rows.size()));
        continue;
      }
      OperatorRows A = cachedOperatorRows(r.mesh, r.ts, mat, opt, popt, r.file, cache_dir, "bead_rows", rows);
      r.t_assemble = A.t_assemble;
      r.H = std::move(A.H);
      r.G = std::move(A.G);
    }
  }

  RveResult solve() const {
    RveResult res;
    res.bc = RveBc::Pbc;
    const auto t0 = std::chrono::steady_clock::now();
    const Vec3 L = lengths();
    const double ptol = 1e3 * tol_;
    // global dof and traction numbering
    std::vector<int> u0(regions.size()), t0r(regions.size());
    int nd = 0, ntt = 0;
    for (size_t i = 0; i < regions.size(); ++i) { u0[i] = nd; t0r[i] = ntt; nd += regions[i].mesh.ndof; ntt += regions[i].ts.ndof; }
    std::vector<Vec3> X(nd);
    std::vector<int> reg_of(nd);
    for (size_t i = 0; i < regions.size(); ++i)
      for (int g = 0; g < regions[i].mesh.ndof; ++g) { X[u0[i] + g] = regions[i].mesh.dof_point[g]; reg_of[u0[i] + g] = int(i); }
    // dof edges of the pairs: (minus dof, plus dof, pair)
    struct Edge { int m, p, pair; };
    std::vector<Edge> edges;
    std::vector<int> tplus_col(ntt, -1);  // + side traction dof -> unknown column
    std::vector<int> plus_t, minus_t;     // per unknown traction: global + and - traction dof
    std::vector<std::pair<int, int>> trow;  // per unknown traction: (region, row point) of its + side point
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      const BeadRegion &Rm = regions[P.rm], &Rp = regions[P.rp];
      for (int lm : Rm.mesh.l2g[P.pm]) {
        int found = -1;
        for (int lp : Rp.mesh.l2g[P.pp])
          if ((Rp.mesh.dof_point[lp] - Rm.mesh.dof_point[lm] - P.shift).norm() < ptol) { found = lp; break; }
        if (found < 0) throw std::runtime_error("BeadCell: pair dof without a partner");
        edges.push_back({u0[P.rm] + lm, u0[P.rp] + found, int(k)});
      }
      const int om = Rm.ts.offset[P.pm], op = Rp.ts.offset[P.pp];
      const int nm = tdofs(Rm, P.pm), npp = tdofs(Rp, P.pp);
      if (nm != npp) throw std::runtime_error("BeadCell: pair patches with different traction spaces");
      for (int j = 0; j < npp; ++j) {
        int found = -1;
        for (int i = 0; i < nm; ++i)
          if ((Rp.ts.points[op + j].x - Rm.ts.points[om + i].x - P.shift).norm() < ptol) { found = i; break; }
        if (found < 0) throw std::runtime_error("BeadCell: pair traction point without a partner");
        tplus_col[t0r[P.rp] + op + j] = int(plus_t.size());
        plus_t.push_back(t0r[P.rp] + op + j);
        minus_t.push_back(t0r[P.rm] + om + found);
        trow.push_back({P.rp, Rp.conn_row0[P.pp] + j});
      }
    }
    // union-find classes, representative = member with the smallest coordinate sum (on the - sides)
    std::vector<int> parent(nd);
    for (int g = 0; g < nd; ++g) parent[g] = g;
    std::function<int(int)> root = [&](int g) { return parent[g] == g ? g : parent[g] = root(parent[g]); };
    for (const Edge& e : edges) {
      const int a = root(e.m), b = root(e.p);
      if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
    // Representative: a member that is no pair's + side (so every region of the class gets rows: the - side region
    // at the representative's dof point, the + side region at its traction points), then the smallest coordinate sum.
    std::vector<char> plus(nd, 0);
    for (const Edge& e : edges) plus[e.p] = 1;
    std::vector<int> best(nd, -1), reps, col_of(nd, -1);
    auto better = [&](int g, int b) {
      if (plus[g] != plus[b]) return plus[g] < plus[b];
      return X[g].sum() < X[b].sum() - ptol;
    };
    for (int g = 0; g < nd; ++g) {
      const int r = root(g);
      if (best[r] < 0 || better(g, best[r])) best[r] = g;
    }
    for (int g = 0; g < nd; ++g)
      if (best[root(g)] == g) { col_of[g] = int(reps.size()); reps.push_back(g); }
    // jumps: walk each class from its representative; crossing pair k from - to + adds -C t+
    std::vector<std::vector<std::pair<int, Eigen::Matrix3d>>> jump(nd);
    std::map<std::pair<int, int>, std::vector<std::pair<int, double>>> tplus_at;  // (plus dof, pair) -> traction cols
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      const BeadRegion& Rp = regions[P.rp];
      for (size_t l = 0; l < Rp.mesh.l2g[P.pp].size(); ++l)
        tplus_at[{u0[P.rp] + Rp.mesh.l2g[P.pp][l], int(k)}].push_back({tplus_col[t0r[P.rp] + Rp.ts.offset[P.pp] + int(l)], 1.0});
    }
    std::vector<std::vector<std::tuple<int, int, int>>> adj(nd);
    for (const Edge& e : edges) { adj[e.m].push_back({e.p, e.pair, 1}); adj[e.p].push_back({e.m, e.pair, -1}); }
    std::vector<char> seen(nd, 0);
    for (int r : reps) {
      std::vector<int> queue{r};
      seen[r] = 1;
      for (size_t q = 0; q < queue.size(); ++q) {
        const int g = queue[q];
        for (auto [h, k, dir] : adj[g]) {
          if (seen[h]) continue;
          seen[h] = 1;
          jump[h] = jump[g];
          const Eigen::Matrix3d& C = compliance(pairs[k]);
          if (C.squaredNorm() > 0) {
            // average the + traction coefficient over the + patches of this pair's kind that hold the dof
            const int pd = dir > 0 ? h : g;
            std::vector<std::pair<int, double>> cols;
            for (size_t kk = 0; kk < pairs.size(); ++kk)
              if (pairs[kk].axis == pairs[k].axis && pairs[kk].internal == pairs[k].internal) {
                auto it = tplus_at.find({pd, int(kk)});
                if (it != tplus_at.end()) cols.insert(cols.end(), it->second.begin(), it->second.end());
              }
            for (auto [col, w] : cols) jump[h].push_back({col, -double(dir) / double(cols.size()) * C});
          }
          queue.push_back(h);
        }
      }
    }
    // system: rows = reps' dof points (own region) + + side traction points
    const int nr = int(reps.size()), nt = int(plus_t.size()), nu = 3 * nr, n = nu + 3 * nt;
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n), rhs = Eigen::MatrixXd::Zero(n, 6);
    std::vector<std::pair<int, int>> rowpts;  // (region, row point)
    for (int g : reps) rowpts.push_back({reg_of[g], g - u0[reg_of[g]]});
    for (auto& r : trow) rowpts.push_back(r);
    std::vector<int> tmap(ntt, 0);  // global traction dof -> +col+1 or -(col+1)
    for (int j = 0; j < nt; ++j) { tmap[plus_t[j]] = j + 1; tmap[minus_t[j]] = -(j + 1); }
    std::vector<Vec3> shift(nd);
    for (int g = 0; g < nd; ++g) shift[g] = X[g] - X[best[root(g)]];
    for (int i = 0; i < int(rowpts.size()); ++i) {
      const auto [ri, pt] = rowpts[i];
      const BeadRegion& R = regions[ri];
      const auto Hr = R.H.middleRows<3>(3 * pt);
      const auto Gr = R.G.middleRows<3>(3 * pt);
      for (int gl = 0; gl < R.mesh.ndof; ++gl) {
        const int g = u0[ri] + gl;
        const auto Hb = Hr.middleCols<3>(3 * gl);
        M.block<3, 3>(3 * i, 3 * col_of[best[root(g)]]) += Hb;
        if (shift[g].squaredNorm() > 0)
          for (int k = 0; k < 6; ++k) rhs.block<3, 1>(3 * i, k) -= Hb * (voigtUnit(k, true) * shift[g]);
        for (const auto& [col, C] : jump[g]) M.block<3, 3>(3 * i, nu + 3 * col) += Hb * C;
      }
      for (int dl = 0; dl < R.ts.ndof; ++dl) {
        const int s = tmap[t0r[ri] + dl];
        if (!s) continue;
        M.block<3, 3>(3 * i, nu + 3 * (std::abs(s) - 1)) -= (s > 0 ? 1.0 : -1.0) * Gr.middleCols<3>(3 * dl);
      }
    }
    // border the three translations
    Eigen::MatrixXd modes = Eigen::MatrixXd::Zero(nu, 3);
    for (int j = 0; j < nr; ++j) modes.block<3, 3>(3 * j, 0).setIdentity();
    for (int k = 0; k < 3; ++k) modes.col(k).normalize();
    const double sc = M.cwiseAbs().maxCoeff();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 3, n + 3);
    A.topLeftCorner(n, n) = M;
    A.block(0, n, nu, 3) = sc * modes;
    A.block(n, 0, 3, nu) = sc * modes.transpose();
    res.unknowns = n;
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(A);
    const auto t1 = std::chrono::steady_clock::now();
    res.t_factor = std::chrono::duration<double>(t1 - t0).count();
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 3, 6);
    B.topRows(n) = rhs;
    const Eigen::MatrixXd Xs = lu.solve(B);
    // averages and the boundary solutions per region
    res.cases.fill(RveCaseResult{});
    Mat6 Sig;
    const double V = volume();
    for (int k = 0; k < 6; ++k) {
      const Eigen::Matrix3d Ek = voigtUnit(k, true);
      Eigen::VectorXd tg = Eigen::VectorXd::Zero(3 * ntt);
      for (int j = 0; j < nt; ++j) {
        tg.segment<3>(3 * plus_t[j]) = Xs.col(k).segment<3>(nu + 3 * j);
        tg.segment<3>(3 * minus_t[j]) = -Xs.col(k).segment<3>(nu + 3 * j);
      }
      RveCaseResult& c = res.cases[k];
      c.sigma.setZero();
      c.energy = 0.0;
      for (size_t i = 0; i < regions.size(); ++i) {
        const BeadRegion& R = regions[i];
        BoundarySolution s{&R.mesh, &R.ts, mat, Eigen::VectorXd::Zero(3 * R.mesh.ndof), tg.segment(3 * t0r[i], 3 * R.ts.ndof), {}};
        for (int gl = 0; gl < R.mesh.ndof; ++gl) {
          const int g = u0[i] + gl;
          Vec3 u = Xs.col(k).segment<3>(3 * col_of[best[root(g)]]) + Ek * shift[g];
          for (const auto& [col, C] : jump[g]) u += C * Xs.col(k).segment<3>(nu + 3 * col);
          s.u.segment<3>(3 * gl) = u;
        }
        accumulate(R, s, c, int(i));
        res.solutions.push_back(std::move(s));
      }
      c.sigma = (0.5 / V * (c.sigma + c.sigma.transpose())).eval();
      c.energy /= V;
      c.eps = Ek;
      Sig.col(k) = toVoigt(c.sigma, false);
      const double w = (c.sigma.array() * c.eps.array()).sum();
      res.hill_mandel = std::max(res.hill_mandel, std::abs(c.energy - w) / std::abs(w));
    }
    res.C = Sig;  // unit strains: C = Sigma
    res.asymmetry = (res.C - res.C.transpose()).norm() / res.C.norm();
    res.t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    return res;
  }

 private:
  double tol_ = 1e-12;

  static int tdofs(const BeadRegion& R, int p) {
    return (p + 1 < int(R.ts.offset.size()) ? R.ts.offset[p + 1] : R.ts.ndof) - R.ts.offset[p];
  }
  const Eigen::Matrix3d& compliance(const BeadPair& P) const {
    return P.internal ? weld_internal[P.axis] : weld_periodic[P.axis];
  }

  // planar axis of a patch (all control points on x_a = const), -1 if none
  int planeAxis(const Patch& P, double& c) const {
    for (int a = 0; a < 3; ++a) {
      c = P.cp(0)(a);
      bool on = true;
      for (int k = 1; k < P.nu * P.nv && on; ++k) on = std::abs(P.cp(k)(a) - c) < 1e3 * tol_;
      if (on) return a;
    }
    return -1;
  }
  Vec3 outward(const BeadRegion& R, int p) const {
    PatchEval E;
    const Patch& P = R.mesh.patches[p];
    const double u = 0.5 * (P.U.front() + P.U.back()), v = 0.5 * (P.V.front() + P.V.back());
    P.evalSpan(findSpan(P.nu, P.p, u, P.U), findSpan(P.nv, P.q, v, P.V), u, v, E);
    return E.normal() * R.mesh.normal_sign;
  }
  bool samePoints(const Patch& A, const Patch& B, const Vec3& s) const {
    if (A.nu * A.nv != B.nu * B.nv) return false;
    for (int i = 0; i < A.nu * A.nv; ++i) {
      bool f = false;
      for (int j = 0; j < B.nu * B.nv && !f; ++j) f = (B.cp(j) - A.cp(i) - s).norm() < 1e3 * tol_;
      if (!f) return false;
    }
    return true;
  }

  void findPairs() {
    pairs.clear();
    const Vec3 L = lengths();
    for (auto& r : regions) r.connection.assign(r.mesh.patches.size(), 0);
    for (size_t ri = 0; ri < regions.size(); ++ri)
      for (size_t pi = 0; pi < regions[ri].mesh.patches.size(); ++pi) {
        double c;
        const Patch& P = regions[ri].mesh.patches[pi];
        const int a = planeAxis(P, c);
        if (a < 0) continue;
        const Vec3 n = outward(regions[ri], int(pi));
        if (n(a) >= 0) continue;  // start from - sides (outward normal along -e_a)
        const bool on_lo = std::abs(c - lo(a)) < 1e3 * tol_;
        for (size_t rj = 0; rj < regions.size(); ++rj)
          for (size_t pj = 0; pj < regions[rj].mesh.patches.size(); ++pj) {
            const Patch& Q = regions[rj].mesh.patches[pj];
            double cq;
            if (planeAxis(Q, cq) != a) continue;
            if (outward(regions[rj], int(pj))(a) <= 0) continue;
            Vec3 s = Vec3::Zero();
            bool internal = false;
            if (on_lo && std::abs(cq - hi(a)) < 1e3 * tol_) s = L(a) * Vec3::Unit(a);
            else if (rj != ri && std::abs(cq - c) < 1e3 * tol_) internal = true;
            else continue;
            if (!samePoints(P, Q, s)) continue;
            pairs.push_back({int(ri), int(pi), int(rj), int(pj), a, s, internal});
            regions[ri].connection[pi] = regions[rj].connection[pj] = 1;
          }
      }
  }

  // H, G of a region that is a rotated + translated copy of another with the same patch structure
  void copyRotated(const BeadRegion& S, BeadRegion& T, int nrows) const {
    if (S.mesh.ndof != T.mesh.ndof || S.ts.ndof != T.ts.ndof || 3 * nrows != S.H.rows())
      throw std::runtime_error("BeadCell: copied region does not match its source");
    const Vec3 d = T.mesh.dof_point[0] - T.R * S.mesh.dof_point[0];
    for (int g = 0; g < S.mesh.ndof; ++g)
      if ((T.mesh.dof_point[g] - T.R * S.mesh.dof_point[g] - d).norm() > 1e3 * tol_)
        throw std::runtime_error("BeadCell: copied region is not a rigid copy (dof order or geometry)");
    T.H.resize(S.H.rows(), S.H.cols());
    T.G.resize(S.G.rows(), S.G.cols());
    const Eigen::Matrix3d& R = T.R;
    for (int i = 0; i < nrows; ++i) {
      for (int j = 0; j < S.mesh.ndof; ++j) T.H.block<3, 3>(3 * i, 3 * j) = R * S.H.block<3, 3>(3 * i, 3 * j) * R.transpose();
      for (int j = 0; j < S.ts.ndof; ++j) T.G.block<3, 3>(3 * i, 3 * j) = R * S.G.block<3, 3>(3 * i, 3 * j) * R.transpose();
    }
  }

  void accumulate(const BeadRegion& R, const BoundarySolution& s, RveCaseResult& c, int ri) const {
    const GaussRule g = gaussLegendre(std::max(6, opt.n_gauss / 2));
    PatchEval E;
    std::vector<int> plus_kind(R.mesh.patches.size(), -1);  // + side of a pair: index into pairs
    for (size_t k = 0; k < pairs.size(); ++k)
      if (pairs[k].rp == ri) plus_kind[pairs[k].pp] = int(k);
    for (const Element& e : R.mesh.elements)
      for (const Rect& Rc : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            R.mesh.patches[e.patch].evalSpan(e.su, e.sv, Rc.u0 + Rc.du() * g.x[a], Rc.v0 + Rc.dv() * g.x[b], E);
            const double w = g.w[a] * g.w[b] * Rc.du() * Rc.dv() * E.normal().norm();
            const Vec3 u = s.displacement(e.patch, E), t = s.traction(e.patch, E);
            c.sigma += w * t * E.x.transpose();
            c.energy += w * t.dot(u);
            if (plus_kind[e.patch] >= 0) c.energy += w * t.dot(compliance(pairs[plus_kind[e.patch]]) * t);
          }
  }
};

}  // namespace igabem
