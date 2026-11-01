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
// Each bead is condensed once onto its weld unknowns. Its free surface (t = 0) and its own cross-sections (the
// periodic pair normal to its filament axis, which continues the bead and is never a weld) are eliminated with one
// LU: what is left is the bead's weld operator K (rows: weld dof and weld traction collocation points; columns: the
// weld dofs' displacements and the weld tractions) and its load F (the macro strain, which enters through the
// cross-section shift). The cell then solves the welds only: one displacement per class of weld dofs tied by the
// pairs (union-find over all regions), and the + side traction of every weld pair; rows at each class
// representative's dof point (a - side dof, so the - region's SBIE holds there) and every + side traction point.
// The three translations are bordered out.
//
// Identical beads share one condensed operator: a region declared as a rigid copy (rotation R) of an earlier one
// with the same patch structure reads that region's 3 x 3 blocks turned (R B R^T; isotropic Kelvin kernels are
// rotation- and translation-equivariant) and its load columns turned in Voigt space, with no assembly and no LU. The
// 0/90 cell's B bead is A turned 90 deg. Debonds (WeldCrack) and weld springs only change the cell's weld system:
// every crack length and weld law reuses the same condensed beads.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
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
  // Condensed weld operator: K [u_W; t_W] = F e (e: the macro strain, Voigt engineering, 6 columns), rows: weld dofs
  // (bdof order), then the weld traction points (tcon order); the eliminated unknowns that carry a traction (the
  // cross-sections: their dofs and tractions, xsel) are x = Xk [u_W; t_W] + Xf e. The free-surface displacements
  // (t = 0) enter neither the cell averages nor the energy, so they are not kept: Xk is ~5x smaller on disk.
  Eigen::MatrixXd K, F, Xk, Xf;
  std::vector<int> xsel, xpos;     // kept eliminated unknowns; eliminated unknown -> position in xsel, -1 dropped
  std::vector<int> bdof, tcon;     // weld dofs, weld traction dofs (kept)
  std::vector<int> bpos;           // dof -> position in bdof, -1 if eliminated
  std::vector<int> rowpos;         // row point -> kept row, -1 if eliminated
  // eliminated unknowns: dof -> column (free surface and cross-section dofs; a + cross-section dof maps to its
  // - partner, with shift), traction dof -> (column, sign) on the cross-sections (t- = -t+)
  std::vector<int> ecol_u, ecol_t;
  std::vector<double> esign_t;
  std::vector<Vec3> eshift;
  int ne = 0;
  std::vector<int> erow;           // row points of the eliminated equations
  Eigen::Matrix<double, 6, 6> Tv = Eigen::Matrix<double, 6, 6>::Identity();  // copy: Voigt strain into the source frame
  std::vector<int> conn_row0;      // per patch: first row (point index) of its traction points, -1 if free
  std::vector<char> connection;    // per patch: 1 if a connection patch
  int copy_of = -1;                // >= 0: a rotated copy of that region, which holds the matrices (read turned)
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  int axis = -1;                   // filament axis: its periodic pair is the bead itself (always perfect), not a weld
  double t_assemble = 0.0, t_reduce = 0.0;
  bool reduced_from_cache = false;
};

// A debond on the welds of one kind: pairs normal to `axis` (internal welds or periodic pairs; region: the + side
// region, -1 any) are cracked where max_c |x_c - centre_c| > limit over the listed coordinates, i.e. a debond of
// depth D = half width - limit advancing from the bond edges (a strip: one coordinate; a square: two). Crack faces
// are traction free. Fronts must be patch edges (python/rve/beads.py debonds=: C0 cuts at every depth), so a debond
// is a set of whole weld patches (by their centres): tractions are discontinuous at the front, and the solution is
// consistent (Hill-Mandel exact) with the crack tip on the front.
struct WeldCrack {
  bool internal = false;
  int axis = 2, region = -1;
  std::vector<std::pair<int, double>> centre;  // (coordinate, centre)
  double limit = 0.0;
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

  std::vector<WeldCrack> cracks;  // debonds (FFM energy: compliance against debond depth, same condensed operators)

  static Eigen::Matrix3d springCompliance(int a, double Kn, double Kt) {
    const Eigen::Matrix3d nn = Vec3::Unit(a) * Vec3::Unit(a).transpose();
    return Kn > 0 ? Eigen::Matrix3d(nn / Kn + (Eigen::Matrix3d::Identity() - nn) / Kt) : Eigen::Matrix3d::Zero();
  }

  Vec3 lengths() const { return hi - lo; }
  double volume() const { const Vec3 L = lengths(); return L(0) * L(1) * L(2); }
  double porosity() const { return 1.0 - volume_solid / volume(); }
  int ndofTotal() const { int n = 0; for (auto& r : regions) n += r.mesh.ndof; return n; }

  // files[i]: region geometry; copies[i] = (j, R): region i is region j rotated by R (and translated), -1: assemble;
  // axes[i]: region i's filament axis (-1: none, every pair is a weld).
  void setup(const std::vector<std::string>& files, const Material& m, const SolverOptions& o,
             const std::vector<std::pair<int, Eigen::Matrix3d>>& copies = {}, const std::string& cache_dir = "",
             const std::vector<int>& axes = {}) {
    mat = m;
    // The operators depend on the material only through nu (H) and 1/mu (G): they are assembled and cached for
    // mu = 1 (unit_), so one cached bead serves every material of the same Poisson ratio. The solve works with
    // t' = t / mu and weld compliances mu C, and scales tractions, stresses and energies back by mu.
    mu_ = m.mu;
    unit_ = Material{m.lambda / m.mu, 1.0};
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
      if (i < axes.size()) r.axis = axes[i];
    }
    tol_ = 1e-9 * (hi - lo).maxCoeff();
    findPairs();
    for (size_t i = 0; i < regions.size(); ++i) {
      BeadRegion& r = regions[i];
      const std::vector<Collocation> rows = structure(int(i));
      if (r.copy_of >= 0) {
        copyRotated(regions[r.copy_of], r);
        continue;
      }
      reduce(r, rows, cache_dir);
    }
  }

  RveResult solve() const { return solveScales(std::vector<double>{1.0})[0]; }
  std::vector<RveResult> solveScales(const std::vector<double>& scales) const {
    std::vector<Vec3> s3;
    for (double s : scales) s3.push_back(Vec3::Constant(s));
    return solveScales(s3);
  }

  // One solve per weld-compliance scale s (every weld compliance times s; s = 0: perfect bond, s = kappa_ref / kappa
  // for a kappa sweep). The scale enters the weld system as A(s) = A0 + s U E^T (U: the spring columns of the + side
  // displacement jumps, E^T: the weld traction unknowns), so several scales share one LU of A0 and the solves
  // Z = A0^-1 U, then each costs one LU of I + s E^T Z (Woodbury, size 3 x weld tractions) and the averages.
  // Per-axis form: scales[q](a) multiplies the compliance of every weld normal to axis a (lateral and interlayer
  // welds apart: A(S) = A0 + U S E^T, S diagonal over the traction unknowns by their pair's axis).
  std::vector<RveResult> solveScales(const std::vector<Vec3>& scales) const {
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
    std::vector<int> taxis;               // per unknown traction: its pair's normal axis (weld scale family)
    std::vector<std::pair<int, int>> trow;  // per unknown traction: (region, row point) of its + side point
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      const BeadRegion &Rm = regions[P.rm], &Rp = regions[P.rp];
      if (cracked(P)) continue;  // crack faces: not tied, t+ = t- = 0 (no unknowns, no rows)
      const bool cont = continuation(P);  // a cross-section: condensed in the bead, only its weld-edge dofs tie here
      const int om = Rm.ts.offset[P.pm], op = Rp.ts.offset[P.pp];
      for (int lm : Rm.mesh.l2g[P.pm]) {
        if (cont && Rm.bpos[lm] < 0) continue;
        int found = -1;
        const auto& l2gp = Rp.mesh.l2g[P.pp];
        for (int l = 0; l < int(l2gp.size()) && found < 0; ++l)
          if ((Rp.mesh.dof_point[l2gp[l]] - Rm.mesh.dof_point[lm] - P.shift).norm() < ptol) found = l;
        if (found < 0) throw std::runtime_error("BeadCell: pair dof without a partner");
        edges.push_back({u0[P.rm] + lm, u0[P.rp] + l2gp[found], int(k)});
      }
      if (cont) continue;
      const int nm = tdofs(Rm, P.pm), npp = tdofs(Rp, P.pp);
      if (nm != npp) throw std::runtime_error("BeadCell: pair patches with different traction spaces");
      for (int j = 0; j < npp; ++j) {
        int found = -1;
        for (int i = 0; i < nm; ++i)
          if ((Rp.ts.points[op + j].x - Rm.ts.points[om + i].x - P.shift).norm() < ptol) { found = i; break; }
        if (found < 0) throw std::runtime_error("BeadCell: pair traction point without a partner");
        tplus_col[t0r[P.rp] + op + j] = int(plus_t.size());
        plus_t.push_back(t0r[P.rp] + op + j);
        taxis.push_back(P.axis);
        minus_t.push_back(t0r[P.rm] + om + found);
        trow.push_back({P.rp, Rp.rowpos[Rp.conn_row0[P.pp] + j]});
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
    for (int g = 0; g < nd; ++g)  // eliminated dofs (free surface, cross-sections) are condensed out
      if (best[root(g)] == g && regions[reg_of[g]].bpos[g - u0[reg_of[g]]] >= 0) { col_of[g] = int(reps.size()); reps.push_back(g); }
    // jumps: walk each class from its representative; crossing pair k from - to + adds -C t+
    std::vector<std::vector<std::pair<int, Eigen::Matrix3d>>> jump(nd);
    std::map<std::pair<int, int>, std::vector<std::pair<int, double>>> tplus_at;  // (plus dof, pair) -> traction cols
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      const BeadRegion& Rp = regions[P.rp];
      for (size_t l = 0; l < Rp.mesh.l2g[P.pp].size(); ++l)
        if (const int c = tplus_col[t0r[P.rp] + Rp.ts.offset[P.pp] + int(l)]; c >= 0)
          tplus_at[{u0[P.rp] + Rp.mesh.l2g[P.pp][l], int(k)}].push_back({c, 1.0});
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
    bool springs = false;
    for (const auto& j : jump) springs = springs || !j.empty();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 3, n + 3), rhs = Eigen::MatrixXd::Zero(n, 6);
    Eigen::MatrixXd U = Eigen::MatrixXd::Zero(springs ? n + 3 : 0, springs ? 3 * nt : 0);  // spring columns (s = 1)
    auto M = A.topLeftCorner(n, n);  // the weld system, bordered below by the three translations
    std::vector<std::pair<int, int>> rowpts;  // (region, row point)
    for (int g : reps) rowpts.push_back({reg_of[g], regions[reg_of[g]].rowpos[g - u0[reg_of[g]]]});
    for (auto& r : trow) rowpts.push_back(r);
    std::vector<int> tmap(ntt, 0);  // global traction dof -> +col+1 or -(col+1)
    for (int j = 0; j < nt; ++j) { tmap[plus_t[j]] = j + 1; tmap[minus_t[j]] = -(j + 1); }
    std::vector<Vec3> shift(nd);
    std::vector<int> ucol(nd);  // dof -> its class's unknown (-1: condensed free-surface dof)
    for (int g = 0; g < nd; ++g) { shift[g] = X[g] - X[best[root(g)]]; ucol[g] = col_of[best[root(g)]]; }
    for (const auto& [ri, rr] : rowpts)
      if (rr < 0) throw std::runtime_error("BeadCell: row on an eliminated point");
    std::array<Eigen::Matrix3d, 6> Eu;
    for (int k = 0; k < 6; ++k) Eu[k] = voigtUnit(k, true);
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < int(rowpts.size()); ++i) {  // row block i is written by one thread only
      const auto [ri, rr] = rowpts[i];
      const BeadRegion& R = regions[ri];
      const BeadRegion& S = source(R);  // a rotated copy reads its source's blocks, turned: Rot B Rot^T
      const bool rot = R.copy_of >= 0;
      const Eigen::Matrix3d& Rot = R.R;
      const auto Kr = S.K.middleRows<3>(3 * rr);
      auto blk = [&](int c) {
        return rot ? Eigen::Matrix3d(Rot * Kr.middleCols<3>(3 * c) * Rot.transpose()) : Eigen::Matrix3d(Kr.middleCols<3>(3 * c));
      };
      const int nb = int(R.bdof.size());
      for (int j = 0; j < nb; ++j) {
        const int g = u0[ri] + R.bdof[j];
        const Eigen::Matrix3d Hb = blk(j);
        M.block<3, 3>(3 * i, 3 * ucol[g]) += Hb;
        if (shift[g].squaredNorm() > 0)
          for (int k = 0; k < 6; ++k) rhs.block<3, 1>(3 * i, k) -= Hb * (Eu[k] * shift[g]);
        for (const auto& [col, C] : jump[g]) U.block<3, 3>(3 * i, 3 * col) += Hb * C;
      }
      for (int j = 0; j < int(R.tcon.size()); ++j) {
        const int s = tmap[t0r[ri] + R.tcon[j]];
        if (!s) continue;  // a crack face (t = 0)
        M.block<3, 3>(3 * i, nu + 3 * (std::abs(s) - 1)) += (s > 0 ? 1.0 : -1.0) * blk(nb + j);
      }
      rhs.middleRows<3>(3 * i) += loadRows(R, rr);
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
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 3, 6);
    B.topRows(n) = rhs;
    std::vector<Eigen::MatrixXd> sols;
    auto svec = [&](const Vec3& s) {  // the diagonal of S
      Eigen::VectorXd v(3 * nt);
      for (int j = 0; j < nt; ++j) v.segment<3>(3 * j).setConstant(s(taxis[j]));
      return v;
    };
    if (scales.size() == 1 || !springs) {  // one scale: factor A(s) itself
      if (springs) A.middleCols(nu, 3 * nt) += U * svec(scales[0]).asDiagonal();
      const DenseLU lu(std::move(A));
      const Eigen::MatrixXd Xs = lu.solve(B);
      sols.assign(scales.size(), Xs);
    } else {
      const DenseLU lu(std::move(A));
      const Eigen::MatrixXd Y = lu.solve(B), Z = lu.solve(U);
      if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  A0 factor and Z = A0^-1 U (%d x %d) %.2f s\n", n + 3, 3 * nt, since(t0));
      const Eigen::MatrixXd W = Z.middleRows(nu, 3 * nt);
      for (const Vec3& sc : scales) {
        if (sc.isZero()) { sols.push_back(Y); continue; }
        const Eigen::VectorXd sv = svec(sc);
        Eigen::MatrixXd S = W * sv.asDiagonal();
        S.diagonal().array() += 1.0;
        const DenseLU ls(std::move(S));
        Eigen::MatrixXd Xs = Y;
        gemmAdd(-1.0, Z * sv.asDiagonal(), ls.solve(Y.middleRows(nu, 3 * nt)), Xs);
        sols.push_back(std::move(Xs));
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    res.t_factor = std::chrono::duration<double>(t1 - t0).count();
    std::vector<RveResult> out;
    for (size_t si = 0; si < scales.size(); ++si) {
    const Vec3 scale = springs ? scales[si] : Vec3::Zero();
    const Eigen::MatrixXd& Xs = sols[si];
    res.solutions.clear();
    res.hill_mandel = 0.0;
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
        const int nb = int(R.bdof.size());
        Eigen::VectorXd kv(3 * (nb + R.tcon.size()));
        for (int j = 0; j < nb; ++j) {
          const int g = u0[i] + R.bdof[j];
          Vec3 u = Xs.col(k).segment<3>(3 * ucol[g]) + Ek * shift[g];
          for (const auto& [col, C] : jump[g]) u += scale(taxis[col]) * C * Xs.col(k).segment<3>(nu + 3 * col);
          kv.segment<3>(3 * j) = u;
          s.u.segment<3>(3 * R.bdof[j]) = u;
        }
        for (int j = 0; j < int(R.tcon.size()); ++j) kv.segment<3>(3 * (nb + j)) = s.t.segment<3>(3 * R.tcon[j]);
        const Eigen::VectorXd x = eliminated(R, kv, k);
        for (int g = 0; g < R.mesh.ndof; ++g)  // free-surface dofs (dropped rows) stay 0: t = 0 there
          if (R.ecol_u[g] >= 0 && R.xpos[R.ecol_u[g]] >= 0)
            s.u.segment<3>(3 * g) = x.segment<3>(3 * R.xpos[R.ecol_u[g]]) + Ek * R.eshift[g];
        for (int d = 0; d < R.ts.ndof; ++d)
          if (R.ecol_t[d] >= 0) s.t.segment<3>(3 * d) = R.esign_t[d] * x.segment<3>(3 * R.xpos[R.ecol_t[d]]);
        accumulate(R, s, c, int(i), scale);
        s.t *= mu_;
        res.solutions.push_back(std::move(s));
      }
      c.sigma = (mu_ * 0.5 / V * (c.sigma + c.sigma.transpose())).eval();
      c.energy *= mu_ / V;
      c.eps = Ek;
      Sig.col(k) = toVoigt(c.sigma, false);
      const double w = (c.sigma.array() * c.eps.array()).sum();
      res.hill_mandel = std::max(res.hill_mandel, std::abs(c.energy - w) / std::abs(w));
    }
    res.C = Sig;  // unit strains: C = Sigma
    res.asymmetry = (res.C - res.C.transpose()).norm() / res.C.norm();
    res.t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    out.push_back(res);
    }
    return out;
  }

  // Weld tractions of a solve (FFM stress part with weld laws): one line per Gauss point of every weld + patch (no
  // cross-sections, no debonded patches): pair axis, internal flag, x (3), area weight, outward unit normal of the +
  // region (3), then the + side traction (3) of each of the six unit macro strains (18). Each element cell carries
  // an ng x ng Gauss rule, so the strip means over a debond depth converge with the mesh.
  void weldTractions(const RveResult& r, std::ostream& os, int ng = 6) const {
    const GaussRule g = gaussLegendre(ng);
    PatchEval E;
    const size_t nr = regions.size();
    char buf[64];
    for (const BeadPair& P : pairs) {
      if (continuation(P) || cracked(P)) continue;
      const BeadRegion& R = regions[P.rp];
      for (const Element& e : R.mesh.elements) {
        if (e.patch != P.pp) continue;
        for (const Rect& Rc : e.cells)
          for (size_t a = 0; a < g.x.size(); ++a)
            for (size_t b = 0; b < g.x.size(); ++b) {
              R.mesh.patches[e.patch].evalSpan(e.su, e.sv, Rc.u0 + Rc.du() * g.x[a], Rc.v0 + Rc.dv() * g.x[b], E);
              const Vec3 nv = E.normal();
              const double w = g.w[a] * g.w[b] * Rc.du() * Rc.dv() * nv.norm();
              const Vec3 n = (nv * R.mesh.normal_sign).normalized();
              os << P.axis << ' ' << int(P.internal);
              for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.10g", E.x(c)); os << buf; }
              std::snprintf(buf, sizeof buf, " %.10g", w);
              os << buf;
              for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.8g", n(c)); os << buf; }
              for (int k = 0; k < 6; ++k) {
                const Vec3 t = r.solutions[k * nr + P.rp].traction(e.patch, E);
                for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.10g", t(c)); os << buf; }
              }
              os << '\n';
            }
      }
    }
  }

 private:
  double tol_ = 1e-12;
  double mu_ = 1.0;   // shear modulus of mat: operators are built for mu = 1 (unit_)
  Material unit_;

  static int tdofs(const BeadRegion& R, int p) {
    return (p + 1 < int(R.ts.offset.size()) ? R.ts.offset[p + 1] : R.ts.ndof) - R.ts.offset[p];
  }
  bool continuation(const BeadPair& P) const {  // a bead's own cross-section pair (never a weld)
    return !P.internal && P.rm == P.rp && regions[P.rm].axis == P.axis;
  }
  bool cracked(const BeadPair& P) const {  // the + patch lies in a debond (fronts are patch edges: beads.py debonds)
    if (continuation(P) || cracks.empty()) return false;
    Vec3 x = Vec3::Zero();
    const auto& l2g = regions[P.rp].mesh.l2g[P.pp];
    for (int g : l2g) x += regions[P.rp].mesh.dof_point[g];
    x /= double(l2g.size());
    for (const WeldCrack& c : cracks) {
      if (c.internal != P.internal || c.axis != P.axis || (c.region >= 0 && c.region != P.rp)) continue;
      double d = 0.0;
      for (auto [k, x0] : c.centre) d = std::max(d, std::abs(x(k) - x0));
      if (d > c.limit) return true;
    }
    return false;
  }
  // weld compliance in the unit-mu system (t' = t / mu): mu C
  Eigen::Matrix3d compliance(const BeadPair& P) const {
    if (continuation(P)) return Eigen::Matrix3d::Zero();
    return mu_ * (P.internal ? weld_internal[P.axis] : weld_periodic[P.axis]);
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

  // Index structure of region ri: weld dofs and tractions (kept), the eliminated unknowns (free-surface dofs, the
  // - dofs of the bead's own cross-sections, the + cross-section tractions) and their rows; returns all row points
  // (dof points, then the traction points of every connection patch).
  std::vector<Collocation> structure(int ri) {
    BeadRegion& r = regions[ri];
    const int np = int(r.mesh.patches.size()), nd = r.mesh.ndof;
    const double ptol = 1e3 * tol_;
    r.conn_row0.assign(np, -1);
    std::vector<Collocation> rows = r.mesh.colloc;
    for (int p = 0; p < np; ++p) {
      if (!r.connection[p]) continue;
      r.conn_row0[p] = int(rows.size());
      for (auto& c : tractionCollocation(r.mesh, r.ts, p)) rows.push_back(std::move(c));
    }
    std::vector<char> weld(np, 0);
    std::vector<const BeadPair*> cont;
    for (const BeadPair& P : pairs) {
      if (continuation(P)) { if (P.rm == ri) cont.push_back(&P); continue; }
      if (P.rm == ri) weld[P.pm] = 1;
      if (P.rp == ri) weld[P.pp] = 1;
    }
    r.bpos.assign(nd, -1);
    r.bdof.clear();
    r.tcon.clear();
    for (int p = 0; p < np; ++p)
      if (weld[p]) {
        for (int g : r.mesh.l2g[p]) r.bpos[g] = 0;
        for (int j = 0; j < tdofs(r, p); ++j) r.tcon.push_back(r.ts.offset[p] + j);
      }
    for (int g = 0; g < nd; ++g)
      if (r.bpos[g] >= 0) { r.bpos[g] = int(r.bdof.size()); r.bdof.push_back(g); }
    // cross-sections: + dofs follow their - partners (u+ = u- + E s), + tractions are unknowns (t- = -t+)
    std::vector<int> partner(nd, -1);
    r.eshift.assign(nd, Vec3::Zero());
    for (const BeadPair* P : cont)
      for (int lm : r.mesh.l2g[P->pm]) {
        int lp = -1;
        for (int g : r.mesh.l2g[P->pp])
          if ((r.mesh.dof_point[g] - r.mesh.dof_point[lm] - P->shift).norm() < ptol) { lp = g; break; }
        if (lp < 0 || (r.bpos[lm] >= 0) != (r.bpos[lp] >= 0)) throw std::runtime_error("BeadCell: cross-section dofs do not match");
        if (r.bpos[lm] < 0) { partner[lp] = lm; r.eshift[lp] = P->shift; }
      }
    r.ecol_u.assign(nd, -1);
    r.erow.clear();
    r.ne = 0;
    for (int g = 0; g < nd; ++g)
      if (r.bpos[g] < 0 && partner[g] < 0) { r.ecol_u[g] = r.ne++; r.erow.push_back(g); }
    for (int g = 0; g < nd; ++g)
      if (partner[g] >= 0) {
        if (partner[partner[g]] >= 0) throw std::runtime_error("BeadCell: chained cross-sections");
        r.ecol_u[g] = r.ecol_u[partner[g]];
      }
    r.ecol_t.assign(r.ts.ndof, -1);
    r.esign_t.assign(r.ts.ndof, 0.0);
    for (const BeadPair* P : cont) {
      const int om = r.ts.offset[P->pm], op = r.ts.offset[P->pp], n = tdofs(r, P->pp);
      for (int j = 0; j < n; ++j) {
        r.ecol_t[op + j] = r.ne;
        r.esign_t[op + j] = 1.0;
        r.erow.push_back(r.conn_row0[P->pp] + j);
        int found = -1;
        for (int i = 0; i < n && found < 0; ++i)
          if ((r.ts.points[op + j].x - r.ts.points[om + i].x - P->shift).norm() < ptol) found = i;
        if (found < 0) throw std::runtime_error("BeadCell: cross-section traction point without a partner");
        r.ecol_t[om + found] = r.ne++;
        r.esign_t[om + found] = -1.0;
      }
    }
    r.xpos.assign(r.ne, -1);
    for (const BeadPair* P : cont)
      for (int p : {P->pm, P->pp})
        for (int g : r.mesh.l2g[p])
          if (r.ecol_u[g] >= 0) r.xpos[r.ecol_u[g]] = 0;
    for (int d = 0; d < r.ts.ndof; ++d)
      if (r.ecol_t[d] >= 0) r.xpos[r.ecol_t[d]] = 0;
    r.xsel.clear();
    for (int e = 0; e < r.ne; ++e)
      if (r.xpos[e] >= 0) { r.xpos[e] = int(r.xsel.size()); r.xsel.push_back(e); }
    r.rowpos.assign(rows.size(), -1);
    const int nb = int(r.bdof.size());
    for (int j = 0; j < nb; ++j) r.rowpos[r.bdof[j]] = j;
    int kt = 0;
    for (int p = 0; p < np; ++p)
      if (weld[p])
        for (int j = 0; j < tdofs(r, p); ++j) r.rowpos[r.conn_row0[p] + j] = nb + kt++;
    return rows;
  }

  // Columns of the bead's equations at the row points `pts`: eliminated unknowns (Ae), weld unknowns (Ak: weld dof
  // displacements, then weld tractions with -G), and the load of the six unit macro strains (F: -H (E s) of the
  // + cross-section dofs, moved to the right-hand side).
  void columns(const BeadRegion& r, const Eigen::MatrixXd& H, const Eigen::MatrixXd& G, const std::vector<int>& pts,
               Eigen::MatrixXd& Ae, Eigen::MatrixXd& Ak, Eigen::MatrixXd& F) const {
    const auto rows = expand3(pts);
    const int m = int(rows.size()), nb = int(r.bdof.size());
    std::vector<int> tpos(r.ts.ndof, -1);
    for (int j = 0; j < int(r.tcon.size()); ++j) tpos[r.tcon[j]] = j;
    // sources per target 3-column: (H or G, source block, factor); targets 0..ne-1 eliminated, then kept
    struct Src { bool g; int col; double f; };
    std::vector<std::vector<Src>> src(r.ne + nb + r.tcon.size());
    for (int g = 0; g < r.mesh.ndof; ++g)
      src[r.bpos[g] >= 0 ? r.ne + r.bpos[g] : r.ecol_u[g]].push_back({false, g, 1.0});
    for (int d = 0; d < r.ts.ndof; ++d) {
      if (tpos[d] >= 0) src[r.ne + nb + tpos[d]].push_back({true, d, -1.0});
      else if (r.ecol_t[d] >= 0) src[r.ecol_t[d]].push_back({true, d, -r.esign_t[d]});
    }
    Ae.setZero(m, 3 * r.ne);
    Ak.setZero(m, 3 * (nb + r.tcon.size()));
#pragma omp parallel for schedule(dynamic, 8)
    for (int t = 0; t < int(src.size()); ++t)
      for (const Src& sc : src[t])
        for (int c = 0; c < 3; ++c) {
          const Eigen::MatrixXd& M = sc.g ? G : H;
          const double* from = M.data() + long(3 * sc.col + c) * M.rows();
          double* to = t < r.ne ? Ae.data() + long(3 * t + c) * m : Ak.data() + long(3 * (t - r.ne) + c) * m;
          for (int i = 0; i < m; ++i) to[i] += sc.f * from[rows[i]];
        }
    F.setZero(m, 6);
    for (int g = 0; g < r.mesh.ndof; ++g) {  // column-wise (H is column-major): F(:, k) -= H(rows, 3g..3g+2) E_k s
      if (r.eshift[g].squaredNorm() == 0) continue;
      for (int c = 0; c < 3; ++c) {
        const double* from = H.data() + long(3 * g + c) * H.rows();
        for (int k = 0; k < 6; ++k) {
          const double v = (voigtUnit(k, true) * r.eshift[g])(c);
          if (v == 0.0) continue;
          double* to = F.data() + long(k) * m;
          for (int i = 0; i < m; ++i) to[i] -= v * from[rows[i]];
        }
      }
    }
  }

  // Assemble the bead's operator (rows: dof points, then connection traction points; batched dense assembly) and
  // condense it onto the weld unknowns with one LU, cached on disk.
  void reduce(BeadRegion& r, const std::vector<Collocation>& rows, const std::string& cache_dir) {
    const auto t0 = std::chrono::steady_clock::now();
    const int nk = 3 * int(r.bdof.size() + r.tcon.size());
    std::vector<int> krow;
    for (int pt = 0; pt < int(rows.size()); ++pt)
      if (r.rowpos[pt] >= 0) krow.push_back(pt);
    std::sort(krow.begin(), krow.end(), [&](int a, int b) { return r.rowpos[a] < r.rowpos[b]; });
    std::string path;
    if (!cache_dir.empty()) {
      std::filesystem::create_directories(cache_dir);
      path = cache_dir + "/" + operatorKey(r.file, "bead_weldu_a" + std::to_string(r.axis), unit_, opt, popt) + ".bin";
      r.reduced_from_cache = loadMatrices(path, {&r.K, &r.F, &r.Xk, &r.Xf}) && r.K.rows() == 3 * int(krow.size()) &&
                             r.K.cols() == nk && r.Xk.rows() == 3 * int(r.xsel.size());
      if (r.reduced_from_cache) { r.t_reduce = since(t0); return; }
    }
    OperatorRows A;
    batchedHG(r.mesh, r.ts, unit_, opt, rows, A.H, A.G);
    r.t_assemble = since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXd Ae, Ak, F, Ake, Akk, Fk;
    columns(r, A.H, A.G, r.erow, Ae, Ak, F);
    columns(r, A.H, A.G, krow, Ake, Akk, Fk);
    A = OperatorRows();
    if (std::getenv("IGABEM_PROFILE")) std::fprintf(stderr, "  columns %.2f s (eliminated %d, kept %d)\n", since(t1), 3 * r.ne, nk);
    const DenseLU lu(std::move(Ae));
    Eigen::MatrixXd rhs(3 * r.ne, nk + 6);
    rhs << -Ak, F;
    const Eigen::MatrixXd X = lu.solve(std::move(rhs));
    const std::vector<int> xr = expand3(r.xsel);
    r.Xk = X(xr, Eigen::seqN(0, nk));
    r.Xf = X(xr, Eigen::seqN(nk, 6));
    Eigen::MatrixXd KF(Akk.rows(), nk + 6);
    KF << Akk, -Fk;
    gemmAdd(1.0, Ake, X, KF);  // [K, -F] = [A_kk, -F_k] + A_ke [Xk, Xf]
    r.K = KF.leftCols(nk);
    r.F = -KF.rightCols(6);
    r.t_reduce = since(t1);
    if (!path.empty()) saveMatrices(path, {&r.K, &r.F, &r.Xk, &r.Xf});
  }

  // A rotated + translated copy of a region with the same patch structure: nothing stored, solve() turns the
  // source's 3 x 3 blocks (R B R^T) and load columns (Voigt strain into the source frame, Tv) on the fly.
  void copyRotated(const BeadRegion& S, BeadRegion& T) const {
    if (S.copy_of >= 0) throw std::runtime_error("BeadCell: copy of a copy");
    if (S.mesh.ndof != T.mesh.ndof || S.ts.ndof != T.ts.ndof || S.bdof != T.bdof || S.tcon != T.tcon ||
        S.ecol_u != T.ecol_u || S.ecol_t != T.ecol_t || S.rowpos != T.rowpos || S.xsel != T.xsel)
      throw std::runtime_error("BeadCell: copied region does not match its source (patches or connections)");
    const Vec3 d = T.mesh.dof_point[0] - T.R * S.mesh.dof_point[0];
    for (int g = 0; g < S.mesh.ndof; ++g)
      if ((T.mesh.dof_point[g] - T.R * S.mesh.dof_point[g] - d).norm() > 1e3 * tol_)
        throw std::runtime_error("BeadCell: copied region is not a rigid copy (dof order or geometry)");
    for (int k = 0; k < 6; ++k) T.Tv.col(k) = toVoigt(Eigen::Matrix3d(T.R.transpose() * voigtUnit(k, true) * T.R), true);
  }
  static Eigen::VectorXd turn(Eigen::VectorXd v, const Eigen::Matrix3d& R) {  // every triplet by R
    Eigen::Map<Eigen::MatrixXd> V(v.data(), 3, v.size() / 3);
    V = (R * V).eval();
    return v;
  }
  // Load rows (3 x 6) of kept row rr of region R.
  Eigen::Matrix<double, 3, 6> loadRows(const BeadRegion& R, int rr) const {
    const BeadRegion& S = source(R);
    const Eigen::Matrix<double, 3, 6> Fv = S.F.middleRows<3>(3 * rr);
    return R.copy_of < 0 ? Fv : Eigen::Matrix<double, 3, 6>(R.R * Fv * R.Tv);
  }
  // Eliminated unknowns of region R for the kept values kv and unit macro strain k.
  Eigen::VectorXd eliminated(const BeadRegion& R, const Eigen::VectorXd& kv, int k) const {
    const BeadRegion& S = source(R);
    if (R.copy_of < 0) return S.Xk * kv + S.Xf.col(k);
    return turn(S.Xk * turn(kv, R.R.transpose()) + S.Xf * R.Tv.col(k), R.R);
  }
  const BeadRegion& source(const BeadRegion& R) const { return R.copy_of >= 0 ? regions[R.copy_of] : R; }
  static double since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
  }

  void accumulate(const BeadRegion& R, const BoundarySolution& s, RveCaseResult& c, int ri, const Vec3& scale) const {
    const GaussRule g = gaussLegendre(std::max(6, opt.n_gauss / 2));
    PatchEval E;
    std::vector<int> plus_kind(R.mesh.patches.size(), -1);  // + side of a pair: index into pairs
    for (size_t k = 0; k < pairs.size(); ++k)
      if (pairs[k].rp == ri) plus_kind[pairs[k].pp] = int(k);
    for (const Element& e : R.mesh.elements) {
      if (!R.connection[e.patch]) continue;  // free surface: t = 0, adds nothing to the averages or the energy
      for (const Rect& Rc : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            R.mesh.patches[e.patch].evalSpan(e.su, e.sv, Rc.u0 + Rc.du() * g.x[a], Rc.v0 + Rc.dv() * g.x[b], E);
            const double w = g.w[a] * g.w[b] * Rc.du() * Rc.dv() * E.normal().norm();
            const Vec3 u = s.displacement(e.patch, E), t = s.traction(e.patch, E);
            c.sigma += w * t * E.x.transpose();
            c.energy += w * t.dot(u);
            if (const int k = plus_kind[e.patch]; k >= 0) c.energy += scale(pairs[k].axis) * w * t.dot(compliance(pairs[k]) * t);
          }
    }
  }
};

}  // namespace igabem
