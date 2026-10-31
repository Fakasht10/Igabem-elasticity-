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
#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "cache.hpp"
#include "fast/dense_sbie.hpp"
#include "operators.hpp"
#include "rve.hpp"
#include "sbie.hpp"
#include "solution.hpp"

#ifdef IGABEM_LAPACK
extern "C" {
void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
void dgetrs_(const char* t, const int* n, const int* nrhs, const double* a, const int* lda, const int* ipiv, double* b,
             const int* ldb, int* info);
void dgemm_(const char* ta, const char* tb, const int* m, const int* n, const int* k, const double* al, const double* a,
            const int* lda, const double* b, const int* ldb, const double* be, double* c, const int* ldc);
}
#endif

namespace igabem {

// Dense LU: LAPACK dgetrf/dgetrs when linked (IGABEM_LAPACK; OpenBLAS 3.6x Eigen's PartialPivLU at n = 6000 on
// 4 cores, 1.8 vs 6.6 s), Eigen otherwise.
class DenseLU {
 public:
  explicit DenseLU(Eigen::MatrixXd A) : A_(std::move(A)) {
#ifdef IGABEM_LAPACK
    const int n = int(A_.rows());
    piv_.resize(n);
    int info = 0;
    dgetrf_(&n, &n, A_.data(), &n, piv_.data(), &info);
    if (info < 0) throw std::runtime_error("DenseLU: dgetrf failed");
#else
    lu_.compute(A_);
#endif
  }
  Eigen::MatrixXd solve(Eigen::MatrixXd B) const {
#ifdef IGABEM_LAPACK
    const int n = int(A_.rows()), k = int(B.cols());
    int info = 0;
    dgetrs_("N", &n, &k, A_.data(), &n, piv_.data(), B.data(), &n, &info);
    return B;
#else
    return lu_.solve(B);
#endif
  }

 private:
  Eigen::MatrixXd A_;
#ifdef IGABEM_LAPACK
  std::vector<int> piv_;
#else
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_;
#endif
};

// C += alpha A B (BLAS dgemm when linked: 3x Eigen's product on the condensation's 7.7k x 2.6k x 7.7k).
inline void gemmAdd(double alpha, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
#ifdef IGABEM_LAPACK
  const int m = int(A.rows()), n = int(B.cols()), k = int(A.cols());
  const double one = 1.0;
  dgemm_("N", "N", &m, &n, &k, &alpha, A.data(), &m, B.data(), &k, &one, C.data(), &m);
#else
  C.noalias() += alpha * A * B;
#endif
}

// H (with the rigid-body term) and G at arbitrary collocation rows by the element-batched assembly of the fast dense
// path (src/fast/dense_sbie.hpp, PR #8): one kernel call per (element, point group) for U and T, GEMM contraction.
inline void batchedHG(const Mesh& mesh, const TractionSpace& ts, const Material& mat, const SolverOptions& opt,
                      const std::vector<Collocation>& rows, Eigen::MatrixXd& H, Eigen::MatrixXd& G, double tol = 1e-10) {
  Mesh m = mesh;  // the batched assembler collocates at mesh.colloc
  m.colloc = rows;
  const int n = 3 * int(rows.size());
  H.setZero(n, 3 * m.ndof);
  G.setZero(n, 3 * ts.ndof);
  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, 3);
  DenseSbie ds(m, ts, mat, opt, tol);
  DenseSbie::Out o;
  o.H = H.data();
  o.ldh = n;
  o.G = G.data();
  o.ldg = n;
  o.S = S.data();
  ds.assemble(o);
  for (int a = 0; a < int(rows.size()); ++a)
    for (const auto& [B, NB] : rows[a].basis) H.block<3, 3>(3 * a, 3 * B) -= NB * S.block<3, 3>(3 * a, 0);
}

struct BeadRegion {
  std::string file;
  Mesh mesh;
  TractionSpace ts;
  // Reduced (Schur) operator on the bead's connection unknowns: the free-surface dofs (t = 0) are condensed out,
  //   H_ii u_i + H_ib u_b - G_ic t_c = 0   ->   u_i = Xu u_b + Xt t_c,
  // and the connection rows read Hr u_b - Gr t_c (Hr = H_bb + H_bi Xu, Gr = G_bc - H_bi Xt). Rows: 3 per connection
  // dof (bdof order), then 3 per connection traction point (tcon order); columns: bdof (Hr) and tcon (Gr).
  Eigen::MatrixXd Hr, Gr, Xu, Xt;
  std::vector<int> bdof, idof, tcon;  // connection dofs, interior (free-surface) dofs, connection traction dofs
  std::vector<int> bpos;              // dof -> position in bdof, -1 if interior
  std::vector<int> conn_row0;      // per patch: first row (point index) of its traction points, -1 if free
  std::vector<char> connection;    // per patch: 1 if a connection patch
  int copy_of = -1;                // >= 0: a rotated copy of that region, which holds the matrices (read turned)
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  int axis = -1;                   // filament axis: its periodic pair is the bead itself (always perfect), not a weld
  double t_assemble = 0.0, t_reduce = 0.0;
  bool reduced_from_cache = false;
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
  // weld compliances: periodic pairs by axis, interfaces by normal axis (3 x 3, mm^3 / N; zero = perfect). A periodic
  // pair normal to a region's filament axis (BeadRegion::axis) joins the bead to itself and stays perfect.
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
      r.bpos.assign(r.mesh.ndof, -1);
      r.bdof.clear(); r.idof.clear(); r.tcon.clear();
      for (int p = 0; p < np; ++p)
        if (r.connection[p]) {
          for (int g : r.mesh.l2g[p]) r.bpos[g] = 0;
          for (int j = r.ts.offset[p]; j < r.ts.offset[p] + tdofs(r, p); ++j) r.tcon.push_back(j);
        }
      for (int g = 0; g < r.mesh.ndof; ++g) {
        if (r.bpos[g] < 0) { r.idof.push_back(g); continue; }
        r.bpos[g] = int(r.bdof.size());
        r.bdof.push_back(g);
      }
      if (r.copy_of >= 0) {
        copyRotated(regions[r.copy_of], r);
        continue;
      }
      reduce(r, rows, cache_dir);
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
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  pairs matched %.2f s\n", since(t0));
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
    for (int g = 0; g < nd; ++g)  // free-surface dofs are condensed out (BeadRegion::Xu, Xt): no unknowns
      if (best[root(g)] == g && regions[reg_of[g]].bpos[g - u0[reg_of[g]]] >= 0) { col_of[g] = int(reps.size()); reps.push_back(g); }
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
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  classes and jumps %.2f s\n", since(t0));
    // system: rows = reps' dof points (own region) + + side traction points
    const int nr = int(reps.size()), nt = int(plus_t.size()), nu = 3 * nr, n = nu + 3 * nt;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 3, n + 3), rhs = Eigen::MatrixXd::Zero(n, 6);
    auto M = A.topLeftCorner(n, n);  // the weld system, bordered below by the three translations
    std::vector<std::pair<int, int>> rowpts;  // (region, row point)
    for (int g : reps) rowpts.push_back({reg_of[g], g - u0[reg_of[g]]});
    for (auto& r : trow) rowpts.push_back(r);
    std::vector<int> tmap(ntt, 0);  // global traction dof -> +col+1 or -(col+1)
    for (int j = 0; j < nt; ++j) { tmap[plus_t[j]] = j + 1; tmap[minus_t[j]] = -(j + 1); }
    std::vector<Vec3> shift(nd);
    std::vector<int> ucol(nd);  // dof -> its class's unknown (-1: condensed free-surface dof)
    for (int g = 0; g < nd; ++g) { shift[g] = X[g] - X[best[root(g)]]; ucol[g] = col_of[best[root(g)]]; }
    for (const auto& [ri, pt] : rowpts)
      if (pt < regions[ri].mesh.ndof && regions[ri].bpos[pt] < 0) throw std::runtime_error("BeadCell: row on a free-surface dof");
    std::array<Eigen::Matrix3d, 6> Eu;
    for (int k = 0; k < 6; ++k) Eu[k] = voigtUnit(k, true);
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < int(rowpts.size()); ++i) {  // row block i is written by one thread only
      const auto [ri, pt] = rowpts[i];
      const BeadRegion& R = regions[ri];
      const BeadRegion& S = source(R);  // a rotated copy reads its source's blocks, turned: Rot B Rot^T
      const bool rot = R.copy_of >= 0;
      const Eigen::Matrix3d& Rot = R.R;
      const int rr = pt < R.mesh.ndof ? R.bpos[pt] : int(R.bdof.size()) + pt - R.mesh.ndof;
      const auto Hr = S.Hr.middleRows<3>(3 * rr);
      const auto Gr = S.Gr.middleRows<3>(3 * rr);
      for (int j = 0; j < int(R.bdof.size()); ++j) {
        const int g = u0[ri] + R.bdof[j];
        const Eigen::Matrix3d Hb = rot ? Eigen::Matrix3d(Rot * Hr.middleCols<3>(3 * j) * Rot.transpose())
                                       : Eigen::Matrix3d(Hr.middleCols<3>(3 * j));
        M.block<3, 3>(3 * i, 3 * ucol[g]) += Hb;
        if (shift[g].squaredNorm() > 0)
          for (int k = 0; k < 6; ++k) rhs.block<3, 1>(3 * i, k) -= Hb * (Eu[k] * shift[g]);
        for (const auto& [col, C] : jump[g]) M.block<3, 3>(3 * i, nu + 3 * col) += Hb * C;
      }
      for (int j = 0; j < int(R.tcon.size()); ++j) {
        const int s = tmap[t0r[ri] + R.tcon[j]];
        if (!s) continue;
        const Eigen::Matrix3d Gb = rot ? Eigen::Matrix3d(Rot * Gr.middleCols<3>(3 * j) * Rot.transpose())
                                       : Eigen::Matrix3d(Gr.middleCols<3>(3 * j));
        M.block<3, 3>(3 * i, nu + 3 * (std::abs(s) - 1)) -= (s > 0 ? 1.0 : -1.0) * Gb;
      }
    }
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  blocks placed %.2f s\n", since(t0));
    // border the three translations
    Eigen::MatrixXd modes = Eigen::MatrixXd::Zero(nu, 3);
    for (int j = 0; j < nr; ++j) modes.block<3, 3>(3 * j, 0).setIdentity();
    for (int k = 0; k < 3; ++k) modes.col(k).normalize();
    const double sc = M.cwiseAbs().maxCoeff();
    A.block(0, n, nu, 3) = sc * modes;
    A.block(n, 0, 3, nu) = sc * modes.transpose();
    res.unknowns = n;
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  weld system built %.2f s\n", since(t0));
    const DenseLU lu(std::move(A));
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
        Eigen::VectorXd ub(3 * R.bdof.size()), tc(3 * R.tcon.size());
        for (int j = 0; j < int(R.bdof.size()); ++j) {
          const int g = u0[i] + R.bdof[j];
          Vec3 u = Xs.col(k).segment<3>(3 * ucol[g]) + Ek * shift[g];
          for (const auto& [col, C] : jump[g]) u += C * Xs.col(k).segment<3>(nu + 3 * col);
          ub.segment<3>(3 * j) = u;
          s.u.segment<3>(3 * R.bdof[j]) = u;
        }
        for (int j = 0; j < int(R.tcon.size()); ++j) tc.segment<3>(3 * j) = s.t.segment<3>(3 * R.tcon[j]);
        const BeadRegion& S = source(R);
        auto turn = [&](Eigen::VectorXd v, bool back) {  // every triplet by R (back: R^T)
          Eigen::Map<Eigen::MatrixXd> V(v.data(), 3, v.size() / 3);
          V = (back ? Eigen::Matrix3d(R.R.transpose()) : R.R) * V;
          return v;
        };
        const Eigen::VectorXd ui = R.copy_of < 0 ? Eigen::VectorXd(S.Xu * ub + S.Xt * tc)
                                                 : turn(S.Xu * turn(ub, true) + S.Xt * turn(tc, true), false);
        for (int j = 0; j < int(R.idof.size()); ++j) s.u.segment<3>(3 * R.idof[j]) = ui.segment<3>(3 * j);
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
    static const Eigen::Matrix3d zero = Eigen::Matrix3d::Zero();
    if (!P.internal && P.rm == P.rp && regions[P.rm].axis == P.axis) return zero;  // a bead's own cross-section
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

  // out(:, c0 + j) = s M(rows, cols[j]), threaded over columns (Eigen's indexed view gathers serially: 14 s of the
  // grade-3 condensation)
  static void gather(const Eigen::MatrixXd& M, const std::vector<int>& rows, const std::vector<int>& cols,
                     Eigen::MatrixXd& out, int c0, double s) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < int(cols.size()); ++j) {
      const double* src = M.data() + long(cols[j]) * M.rows();
      double* dst = out.data() + long(c0 + j) * out.rows();
      for (size_t i = 0; i < rows.size(); ++i) dst[i] = s * src[rows[i]];
    }
  }
  static Eigen::MatrixXd gather(const Eigen::MatrixXd& M, const std::vector<int>& rows, const std::vector<int>& cols) {
    Eigen::MatrixXd out(rows.size(), cols.size());
    gather(M, rows, cols, out, 0, 1.0);
    return out;
  }
  static std::vector<int> expand3(const std::vector<int>& v) {
    std::vector<int> e;
    e.reserve(3 * v.size());
    for (int i : v) for (int c = 0; c < 3; ++c) e.push_back(3 * i + c);
    return e;
  }

  // Assemble the bead's operator (rows: dof points, then connection traction points) and condense the free-surface
  // dofs: one LU of H_ii per bead type, cached on disk with the reduced blocks.
  void reduce(BeadRegion& r, const std::vector<Collocation>& rows, const std::string& cache_dir) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string path;
    if (!cache_dir.empty()) {
      std::filesystem::create_directories(cache_dir);
      path = cache_dir + "/" + operatorKey(r.file, "bead_schur", mat, opt, popt) + ".bin";
      r.reduced_from_cache = loadMatrices(path, {&r.Hr, &r.Gr, &r.Xu, &r.Xt}) &&
                             r.Hr.cols() == 3 * int(r.bdof.size()) && r.Gr.cols() == 3 * int(r.tcon.size()) &&
                             r.Xu.rows() == 3 * int(r.idof.size());
      if (r.reduced_from_cache) { r.t_reduce = since(t0); return; }
    }
    OperatorRows A;
    batchedHG(r.mesh, r.ts, mat, opt, rows, A.H, A.G);
    r.t_assemble = since(t0);
    const int nd = r.mesh.ndof, nct = int(r.tcon.size());
    std::vector<int> brow = r.bdof;  // boundary rows: connection dof points, then the connection traction points
    for (int j = 0; j < nct; ++j) brow.push_back(nd + j);
    const auto I = expand3(r.idof), B = expand3(r.bdof), Bt = expand3(brow), T = expand3(r.tcon);
    const auto t1 = std::chrono::steady_clock::now();
    if (I.empty()) {  // nothing to condense
      r.Hr = gather(A.H, Bt, B);
      r.Gr = gather(A.G, Bt, T);
      r.Xu.resize(0, B.size());
      r.Xt.resize(0, T.size());
      r.t_reduce = since(t1);
      return;
    }
    const DenseLU lu(gather(A.H, I, I));
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  lu %.2f s (n %zu)\n", since(t1), I.size());
    Eigen::MatrixXd rhs(I.size(), B.size() + T.size());
    gather(A.H, I, B, rhs, 0, -1.0);
    gather(A.G, I, T, rhs, int(B.size()), 1.0);
    const Eigen::MatrixXd X = lu.solve(std::move(rhs));
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  solve %.2f s (rhs %zu)\n", since(t1), B.size() + T.size());
    r.Xu = X.leftCols(B.size());
    r.Xt = X.rightCols(T.size());
    const Eigen::MatrixXd Hbi = gather(A.H, Bt, I);
    Eigen::MatrixXd HG(Bt.size(), B.size() + T.size());
    gather(A.H, Bt, B, HG, 0, 1.0);
    gather(A.G, Bt, T, HG, int(B.size()), -1.0);
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  gathers %.2f s\n", since(t1));
    gemmAdd(1.0, Hbi, X, HG);
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  gemm %.2f s\n", since(t1));  // [Hr, -Gr] = [H_bb, -G_bc] + H_bi [Xu, Xt]
    r.Hr = HG.leftCols(B.size());
    r.Gr = -HG.rightCols(T.size());
    r.t_reduce = since(t1);
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  products done %.2f s (rows %zu)\n", since(t1), Bt.size());
    if (!path.empty()) saveMatrices(path, {&r.Hr, &r.Gr, &r.Xu, &r.Xt});
  }

  // Reduced operator of a region that is a rotated + translated copy of another with the same patch structure:
  // every 3 x 3 block turns as R B R^T (isotropic kernels are rotation- and translation-equivariant).
  void copyRotated(const BeadRegion& S, BeadRegion& T) const {
    if (S.mesh.ndof != T.mesh.ndof || S.ts.ndof != T.ts.ndof || S.bdof != T.bdof || S.tcon != T.tcon)
      throw std::runtime_error("BeadCell: copied region does not match its source (patches or connections)");
    const Vec3 d = T.mesh.dof_point[0] - T.R * S.mesh.dof_point[0];
    for (int g = 0; g < S.mesh.ndof; ++g)
      if ((T.mesh.dof_point[g] - T.R * S.mesh.dof_point[g] - d).norm() > 1e3 * tol_)
        throw std::runtime_error("BeadCell: copied region is not a rigid copy (dof order or geometry)");
    if (S.copy_of >= 0) throw std::runtime_error("BeadCell: copy of a copy");
    // nothing stored: solve() turns the source's blocks on the fly (3 x 3 products, cheap next to the LU)
  }
  const BeadRegion& source(const BeadRegion& R) const { return R.copy_of >= 0 ? regions[R.copy_of] : R; }
  static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
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
