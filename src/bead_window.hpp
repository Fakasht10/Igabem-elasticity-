// Bead window: a non-periodic patch of bead pieces with driven cut faces, for global-local analysis of a print.
//
// The window tiles copies of the bead pieces of a cell (src/bead_cell.hpp) over a region of a part. Every piece is
// a rigid copy of one condensed operator (BeadCell::copyRotated: no assembly, no LU per piece); pieces meet on
//   welds             spring laws, as in the cell
//   filament joints   the cross-sections between two pieces of the same filament (perfect: the bead continues)
//   periodic pairs    along the periodic axes only (a z-periodic slab: the build direction of a thick part)
// and the window's cut faces (unpaired planar patches on a non-periodic box face) carry a prescribed displacement
// u = u_d(x) (the part solution) with unknown traction, the reaction. A periodic pair carries the jump of the drive,
// u+ = u-(x - s) + u_d(x) - u_d(x - s) - C t+ (the cell's E s for u_d = E x).
//
// Unknowns: one displacement per class of tied weld dofs that holds no cut-face dof (union-find over the pairs, as
// in the cell; a class with a cut-face dof is known), the + side traction of every pair and the traction of every
// cut face. Rows: a class representative's dof point, every + side traction point and every cut-face traction
// point. Each row belongs to one piece and involves only that piece's kept unknowns, so the system is block-sparse:
// one dense block per piece (its rows x its unknowns, ~1.6k x 3.3k at grade 0), taken from the turned piece
// operator. A cell is mostly interface (its faces hold most of its unknowns), so a sparse LU fills in to nearly dense
// beyond 2 x 2 cells (UMFPACK + METIS: 2 x 2 cells 35 s, 3 x 3 out of memory). The default solver is GMRES on the
// piece blocks (one GEMV per piece), right-preconditioned by block Jacobi over cells: every unknown is owned by the
// cell that holds its row, so each cell's block (its rows x its own unknowns, ~3.5k) is square and is factored once
// (dense LU). UMFPACK stays for small windows and checks (direct = true).
#pragma once

#include <Eigen/Sparse>
#include <Eigen/UmfPackSupport>
#include <memory>
#include <ostream>

#include "bead_cell.hpp"

namespace igabem {

class BeadWindow : public BeadCell {
 public:
  // Prescribed displacement at a dof: (region, dof of that region, its point). Point-only drives (u_d = E x, the part
  // solution) ignore the first two; a corrector drive (u = E x + chi E, chi the periodic cell's fluctuation at the
  // same dof of the same piece) uses them, and returns its smooth part alone for region -1 (the periodic jumps: the
  // window's own weld springs make the fluctuation's jumps).
  using Drive = std::function<Vec3(int, int, const Vec3&)>;

  struct Result {
    std::vector<BoundarySolution> solutions;  // drive-major: solutions[q * regions + i] (kept dofs and tractions)
    double t_solve = 0.0;
  };

  int unknowns() const { return n_; }
  long nonzeros() const { long n = 0; for (const auto& M : M_) n += M.size(); return n; }
  double t_build = 0.0, t_factor = 0.0;
  bool direct = false;       // UMFPACK on the assembled matrix instead of GMRES
  double tol = 1e-10;        // GMRES relative residual
  mutable int iterations = 0;  // GMRES iterations of the last solve (max over drives)

  // Window of bead pieces: files[i] placed at offsets[i], copies as in BeadCell::setup, filament axis per region,
  // periodic box axes (the rest are cut).
  void setupWindow(const std::vector<std::string>& files, const std::vector<Vec3>& offsets,
                   const std::vector<std::pair<int, Eigen::Matrix3d>>& copies, const std::vector<int>& filaments,
                   const std::array<bool, 3>& per, const Material& m, const SolverOptions& o,
                   const std::string& cache_dir, const std::vector<int>& groups = {}) {
    periodic = per;
    setup(files, m, o, copies, cache_dir, {}, offsets);
    for (size_t i = 0; i < regions.size() && i < filaments.size(); ++i) regions[i].filament = filaments[i];
    // preconditioner groups (cells): consecutive regions with the same group id; default one group per region
    group_.assign(regions.size(), 0);
    for (size_t i = 0; i < regions.size(); ++i) group_[i] = i < groups.size() ? groups[i] : int(i);
    for (size_t i = 1; i < regions.size(); ++i)
      if (group_[i] != group_[i - 1] && group_[i] != group_[i - 1] + 1) throw std::runtime_error("BeadWindow: groups must be consecutive");
    index();
  }

  // Assemble and factor the window system for weld-compliance scale s (per normal axis, as BeadCell::solveScales).
  void factor(const Vec3& scale) {
    const auto t0 = std::chrono::steady_clock::now();
    scale_ = scale;
    assemble();
    t_build = since(t0);
    const auto t1 = std::chrono::steady_clock::now();
    if (!direct) {
      blockJacobi();
      t_factor = since(t1);
      return;
    }
    compress();
    lu_ = std::make_unique<Eigen::UmfPackLU<Eigen::SparseMatrix<double>>>();
    auto& ctl = lu_->umfpackControl();
    ctl(UMFPACK_STRATEGY) = UMFPACK_STRATEGY_SYMMETRIC;  // nearly symmetric pattern: fill-reducing order of A + A^T
    ctl(UMFPACK_ORDERING) = UMFPACK_ORDERING_METIS;
    if (const char* e = std::getenv("IGABEM_UMF_ORDERING")) ctl(UMFPACK_ORDERING) = std::atoi(e);
    if (const char* e = std::getenv("IGABEM_UMF_STRATEGY")) ctl(UMFPACK_STRATEGY) = std::atoi(e);
    lu_->compute(A_);
    if (lu_->info() != Eigen::Success)
      throw std::runtime_error("BeadWindow: UMFPACK factorization failed, status " + std::to_string(lu_->umfpackFactorizeReturncode()));
    t_factor = since(t1);
  }

  // Solve for each drive u_d (displacements on the cut faces and the periodic jumps).
  Result solve(const std::vector<Drive>& drives) const {
    const auto t0 = std::chrono::steady_clock::now();
    const int nq = int(drives.size()), nreg = int(regions.size());
    // fixed part of every kept dof: u_d(x) on a known class, the drive's periodic jump on an unknown class
    std::vector<Eigen::MatrixXd> fixed(nreg);
    for (int i = 0; i < nreg; ++i) {
      const BeadRegion& R = regions[i];
      fixed[i].setZero(3 * R.bdof.size(), nq);
      for (size_t j = 0; j < R.bdof.size(); ++j) {
        const int g = u0_[i] + R.bdof[j], r = rep_[g], rr = reg_of_[r];
        const bool known = ucol_[g] < 0, jumps = shift_[g].squaredNorm() > 0;
        if (!known && !jumps) continue;
        for (int q = 0; q < nq; ++q) {  // every member is its representative plus the drive's periodic jump
          const Vec3 ur = drives[q](rr, r - u0_[rr], X_[r]);
          Vec3 v = known ? ur : Vec3::Zero();
          if (jumps) v += drives[q](-1, -1, X_[g]) - drives[q](-1, -1, X_[r]);  // smooth part only (no corrector)
          fixed[i].block<3, 1>(3 * j, q) = v;
        }
      }
    }
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_, nq);
    for (int i = 0; i < nreg; ++i) turned(i);  // fill the cache before the threads read it
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nreg; ++i) {
      const BeadRegion& R = regions[i];
      const int nb = int(R.bdof.size()), m = int(rows_[i].size());
      if (!m || fixed[i].isZero(0)) continue;
      const Eigen::MatrixXd& K = turned(i);
      Eigen::MatrixXd Kr(3 * m, 3 * nb);
      for (int k = 0; k < m; ++k) Kr.middleRows<3>(3 * k) = K.block(3 * rows_[i][k], 0, 3, 3 * nb);
      B.middleRows(3 * row0_[i], 3 * m) -= Kr * fixed[i];
    }
    Eigen::MatrixXd Xs(n_, nq);
    iterations = 0;
    if (direct) Xs = lu_->solve(B);
    else {
      double r;
      iterations = gmresBatch(B, Xs, r);
      if (r > 10 * tol) std::fprintf(stderr, "BeadWindow: GMRES stopped at residual %.2e after %d iterations\n", r, iterations);
    }
    Result res;
    for (int q = 0; q < nq; ++q)
      for (int i = 0; i < nreg; ++i) {
        const BeadRegion& R = regions[i];
        BoundarySolution s{&R.mesh, &R.ts, mat, Eigen::VectorXd::Zero(3 * R.mesh.ndof), Eigen::VectorXd::Zero(3 * R.ts.ndof), {}};
        for (size_t j = 0; j < R.bdof.size(); ++j) {
          const int g = u0_[i] + R.bdof[j];
          Vec3 u = fixed[i].block<3, 1>(3 * j, q);
          if (ucol_[g] >= 0) u += Xs.col(q).segment<3>(3 * ucol_[g]);
          for (const auto& [col, C] : jump_[g]) u += scale_(taxis_[col]) * C * Xs.col(q).segment<3>(nu_ + 3 * col);
          s.u.segment<3>(3 * R.bdof[j]) = u;
        }
        for (int d : R.tcon)
          if (const int c = tmap_[t0r_[i] + d]) s.t.segment<3>(3 * d) = (c > 0 ? mu_ : -mu_) * Xs.col(q).segment<3>(nu_ + 3 * (std::abs(c) - 1));
        res.solutions.push_back(std::move(s));
      }
    res.t_solve = since(t0);
    return res;
  }

  // Tractions at Gauss points of every weld + side and every cut face, one line each: kind (0 periodic, 1 internal,
  // 2 filament joint, 3 cut face), normal axis, region, x (3), area weight, outward unit normal (3), then the
  // traction (3) per drive.
  void tractions(const Result& r, std::ostream& os, int ng = 6) const {
    const GaussRule g = gaussLegendre(ng);
    const int nreg = int(regions.size()), nq = int(r.solutions.size()) / nreg;
    PatchEval E;
    char buf[64];
    auto patch = [&](int kind, int axis, int ri, int p) {
      const BeadRegion& R = regions[ri];
      for (const Element& e : R.mesh.elements) {
        if (e.patch != p) continue;
        for (const Rect& Rc : e.cells)
          for (size_t a = 0; a < g.x.size(); ++a)
            for (size_t b = 0; b < g.x.size(); ++b) {
              R.mesh.patches[p].evalSpan(e.su, e.sv, Rc.u0 + Rc.du() * g.x[a], Rc.v0 + Rc.dv() * g.x[b], E);
              const Vec3 nv = E.normal();
              const Vec3 n = (nv * R.mesh.normal_sign).normalized();
              os << kind << ' ' << axis << ' ' << ri;
              for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.10g", E.x(c)); os << buf; }
              std::snprintf(buf, sizeof buf, " %.10g", g.w[a] * g.w[b] * Rc.du() * Rc.dv() * nv.norm());
              os << buf;
              for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.8g", n(c)); os << buf; }
              for (int q = 0; q < nq; ++q) {
                const Vec3 t = r.solutions[q * nreg + ri].traction(p, E);
                for (int c = 0; c < 3; ++c) { std::snprintf(buf, sizeof buf, " %.10g", t(c)); os << buf; }
              }
              os << '\n';
            }
      }
    };
    for (const BeadPair& P : pairs)
      if (!cracked(P)) patch(filamentJoint(P) ? 2 : P.internal ? 1 : 0, P.axis, P.rp, P.pp);
    for (int i = 0; i < nreg; ++i) {
      double c;
      for (size_t p = 0; p < regions[i].driven.size(); ++p)
        if (regions[i].driven[p]) patch(3, planeAxis(regions[i].mesh.patches[p], c), i, int(p));
    }
  }

 private:
  // numbering and classes (independent of the weld scale)
  std::vector<int> u0_, t0r_, ucol_, rep_, reg_of_, taxis_, tmap_, row0_;
  std::vector<Vec3> X_, shift_;
  std::vector<std::vector<int>> rows_;                                   // per region: kept row points it holds
  std::vector<std::vector<std::pair<int, Eigen::Matrix3d>>> jump_;      // dof -> (+ traction unknown, C)
  int nu_ = 0, n_ = 0;
  Vec3 scale_ = Vec3::Ones();
  std::vector<std::vector<int>> rowunk_;   // per region: the unknown block that owns each of its rows
  std::vector<std::vector<int>> cols_;     // per region: unknown blocks its rows involve
  std::vector<Eigen::MatrixXd> M_;         // per region: its rows x its unknowns (3 rows x 3 columns per block)
  std::vector<int> group_;                 // per region: preconditioner group (cell)
  std::vector<std::unique_ptr<DenseLU>> P_;  // per group: LU of its rows x its own unknowns
  std::vector<std::vector<int>> gown_;     // per group: its own unknown blocks, in its row order
  std::vector<int> gfirst_;                // per group: first region
  Eigen::SparseMatrix<double> A_;  // UmfPackLU keeps a reference to it
  std::unique_ptr<Eigen::UmfPackLU<Eigen::SparseMatrix<double>>> lu_;
  mutable std::map<std::pair<int, std::array<double, 9>>, std::shared_ptr<Eigen::MatrixXd>> turned_;

  // The condensed operator of region i in its own frame (a copy: its source's, every 3 x 3 block turned R B R^T).
  const Eigen::MatrixXd& turned(int i) const {
    const BeadRegion& R = regions[i];
    if (R.copy_of < 0 || R.R.isIdentity(1e-14)) return source(R).K;
    std::array<double, 9> key;
    for (int k = 0; k < 9; ++k) key[k] = R.R(k);
    auto& slot = turned_[{R.copy_of, key}];
    if (!slot) {
      const Eigen::MatrixXd& K = regions[R.copy_of].K;
      auto T = std::make_shared<Eigen::MatrixXd>(K.rows(), K.cols());
#pragma omp parallel for schedule(static)
      for (int c = 0; c < int(K.cols() / 3); ++c)
        for (int r = 0; r < int(K.rows() / 3); ++r) T->block<3, 3>(3 * r, 3 * c) = R.R * K.block<3, 3>(3 * r, 3 * c) * R.R.transpose();
      slot = T;
    }
    return *slot;
  }

  void index() {
    const int nreg = int(regions.size());
    const double ptol = 1e3 * tol_;
    u0_.assign(nreg, 0);
    t0r_.assign(nreg, 0);
    int nd = 0, ntt = 0;
    for (int i = 0; i < nreg; ++i) { u0_[i] = nd; t0r_[i] = ntt; nd += regions[i].mesh.ndof; ntt += regions[i].ts.ndof; }
    X_.assign(nd, Vec3::Zero());
    reg_of_.assign(nd, 0);
    std::vector<int>& reg_of = reg_of_;
    std::vector<char> cut(nd, 0);
    for (int i = 0; i < nreg; ++i) {
      const BeadRegion& R = regions[i];
      for (int g = 0; g < R.mesh.ndof; ++g) { X_[u0_[i] + g] = R.mesh.dof_point[g]; reg_of[u0_[i] + g] = i; }
      for (size_t p = 0; p < R.driven.size(); ++p)
        if (R.driven[p]) for (int g : R.mesh.l2g[p]) cut[u0_[i] + g] = 1;
    }
    struct Edge { int m, p, pair; };
    std::vector<Edge> edges;
    std::vector<int> tplus_col(ntt, -1);
    std::vector<std::pair<int, int>> trow;  // per traction unknown: (region, kept row)
    tmap_.assign(ntt, 0);
    taxis_.clear();
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      if (cracked(P)) continue;
      const BeadRegion &Rm = regions[P.rm], &Rp = regions[P.rp];
      const int om = Rm.ts.offset[P.pm], op = Rp.ts.offset[P.pp];
      const auto& l2gp = Rp.mesh.l2g[P.pp];
      for (int lm : Rm.mesh.l2g[P.pm]) {
        int found = -1;
        for (int l = 0; l < int(l2gp.size()) && found < 0; ++l)
          if ((Rp.mesh.dof_point[l2gp[l]] - Rm.mesh.dof_point[lm] - P.shift).norm() < ptol) found = l;
        if (found < 0) throw std::runtime_error("BeadWindow: pair dof without a partner");
        edges.push_back({u0_[P.rm] + lm, u0_[P.rp] + l2gp[found], int(k)});
      }
      const int npp = tdofs(Rp, P.pp);
      for (int j = 0; j < npp; ++j) {
        int found = -1;
        for (int i = 0; i < npp && found < 0; ++i)
          if ((Rp.ts.points[op + j].x - Rm.ts.points[om + i].x - P.shift).norm() < ptol) found = i;
        if (found < 0) throw std::runtime_error("BeadWindow: pair traction point without a partner");
        const int c = int(taxis_.size());
        tplus_col[t0r_[P.rp] + op + j] = c;
        tmap_[t0r_[P.rp] + op + j] = c + 1;
        tmap_[t0r_[P.rm] + om + found] = -(c + 1);
        taxis_.push_back(P.axis);
        trow.push_back({P.rp, Rp.rowpos[Rp.conn_row0[P.pp] + j]});
      }
    }
    for (int i = 0; i < nreg; ++i) {  // cut-face tractions (no spring: axis -1 is never scaled)
      const BeadRegion& R = regions[i];
      for (size_t p = 0; p < R.driven.size(); ++p)
        if (R.driven[p])
          for (int j = 0; j < tdofs(R, int(p)); ++j) {
            const int c = int(taxis_.size());
            tmap_[t0r_[i] + R.ts.offset[p] + j] = c + 1;
            taxis_.push_back(0);
            trow.push_back({i, R.rowpos[R.conn_row0[p] + j]});
          }
    }
    // classes: representative a cut-face dof (the class is known), else no pair's + side, then the smallest sum
    std::vector<int> parent(nd);
    for (int g = 0; g < nd; ++g) parent[g] = g;
    std::function<int(int)> root = [&](int g) { return parent[g] == g ? g : parent[g] = root(parent[g]); };
    for (const Edge& e : edges) {
      const int a = root(e.m), b = root(e.p);
      if (a != b) parent[std::max(a, b)] = std::min(a, b);
    }
    std::vector<char> plus(nd, 0);
    for (const Edge& e : edges) plus[e.p] = 1;
    auto better = [&](int g, int b) {
      if (cut[g] != cut[b]) return cut[g] > cut[b];
      if (plus[g] != plus[b]) return plus[g] < plus[b];
      return X_[g].sum() < X_[b].sum() - ptol;
    };
    std::vector<int> best(nd, -1);
    for (int g = 0; g < nd; ++g) {
      const int r = root(g);
      if (best[r] < 0 || better(g, best[r])) best[r] = g;
    }
    rep_.assign(nd, -1);
    shift_.assign(nd, Vec3::Zero());
    ucol_.assign(nd, -1);
    std::vector<int> col_of(nd, -1), reps;
    for (int g = 0; g < nd; ++g)
      if (best[root(g)] == g && !cut[g] && regions[reg_of[g]].bpos[g - u0_[reg_of[g]]] >= 0) { col_of[g] = int(reps.size()); reps.push_back(g); }
    for (int g = 0; g < nd; ++g) { rep_[g] = best[root(g)]; shift_[g] = X_[g] - X_[rep_[g]]; ucol_[g] = col_of[rep_[g]]; }
    // spring jumps along each class from its representative (BeadCell::solveScales)
    jump_.assign(nd, {});
    std::map<std::pair<int, int>, std::vector<int>> tplus_at;
    for (size_t k = 0; k < pairs.size(); ++k) {
      const BeadPair& P = pairs[k];
      const BeadRegion& Rp = regions[P.rp];
      for (size_t l = 0; l < Rp.mesh.l2g[P.pp].size(); ++l)
        if (const int c = tplus_col[t0r_[P.rp] + Rp.ts.offset[P.pp] + int(l)]; c >= 0)
          tplus_at[{u0_[P.rp] + Rp.mesh.l2g[P.pp][l], int(k)}].push_back(c);
    }
    std::vector<std::vector<std::tuple<int, int, int>>> adj(nd);
    for (const Edge& e : edges) { adj[e.m].push_back({e.p, e.pair, 1}); adj[e.p].push_back({e.m, e.pair, -1}); }
    std::vector<char> seen(nd, 0);
    for (int g0 = 0; g0 < nd; ++g0) {
      if (rep_[g0] != g0 || seen[g0]) continue;
      std::vector<int> queue{g0};
      seen[g0] = 1;
      for (size_t q = 0; q < queue.size(); ++q) {
        const int g = queue[q];
        for (auto [h, k, dir] : adj[g]) {
          if (seen[h]) continue;
          seen[h] = 1;
          jump_[h] = jump_[g];
          const Eigen::Matrix3d C = compliance(pairs[k]);
          if (C.squaredNorm() > 0) {
            const int pd = dir > 0 ? h : g;
            std::vector<int> cols;
            for (size_t kk = 0; kk < pairs.size(); ++kk)
              if (pairs[kk].axis == pairs[k].axis && pairs[kk].internal == pairs[k].internal && !filamentJoint(pairs[kk])) {
                auto it = tplus_at.find({pd, int(kk)});
                if (it != tplus_at.end()) cols.insert(cols.end(), it->second.begin(), it->second.end());
              }
            for (int col : cols) jump_[h].push_back({col, -double(dir) / double(cols.size()) * C});
          }
          queue.push_back(h);
        }
      }
    }
    // rows, region by region (sorted row indices in every column of the compressed matrix)
    rows_.assign(nreg, {});
    rowunk_.assign(nreg, {});
    for (size_t k = 0; k < reps.size(); ++k) {
      const int g = reps[k], ri = reg_of[g];
      rows_[ri].push_back(regions[ri].rowpos[g - u0_[ri]]);
      rowunk_[ri].push_back(int(k));
    }
    for (size_t j = 0; j < trow.size(); ++j) {
      rows_[trow[j].first].push_back(trow[j].second);
      rowunk_[trow[j].first].push_back(int(reps.size() + j));
    }
    row0_.assign(nreg, 0);
    int nr = 0;
    for (int i = 0; i < nreg; ++i) {
      for (int rr : rows_[i]) if (rr < 0) throw std::runtime_error("BeadWindow: row on an eliminated point");
      row0_[i] = nr;
      nr += int(rows_[i].size());
    }
    nu_ = 3 * int(reps.size());
    n_ = nu_ + 3 * int(taxis_.size());
    if (3 * nr != n_) throw std::runtime_error("BeadWindow: rows and unknowns differ (" + std::to_string(3 * nr) + " vs " + std::to_string(n_) + ")");
  }

  // Piece blocks: per region, its rows times the unknowns they involve (classes, tractions, spring jumps at the
  // current scale), from the turned piece operator.
  void assemble() {
    const int nreg = int(regions.size());
    cols_.assign(nreg, {});
    M_.assign(nreg, Eigen::MatrixXd());
    for (int i = 0; i < nreg; ++i) turned(i);  // fill the cache before the threads read it
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nreg; ++i) {
      const BeadRegion& R = regions[i];
      std::map<int, std::vector<std::pair<int, Eigen::Matrix3d>>> m;
      const int nb = int(R.bdof.size());
      for (int j = 0; j < nb; ++j) {
        const int g = u0_[i] + R.bdof[j];
        if (ucol_[g] >= 0) m[ucol_[g]].push_back({j, Eigen::Matrix3d::Identity()});
        for (const auto& [col, C] : jump_[g]) m[nu_ / 3 + col].push_back({j, scale_(taxis_[col]) * C});
      }
      for (int j = 0; j < int(R.tcon.size()); ++j)
        if (const int c = tmap_[t0r_[i] + R.tcon[j]])
          m[nu_ / 3 + std::abs(c) - 1].push_back({nb + j, (c > 0 ? 1.0 : -1.0) * Eigen::Matrix3d::Identity()});
      const Eigen::MatrixXd& K = turned(i);
      const auto& rows = rows_[i];
      const int nr = int(rows.size());
      Eigen::MatrixXd Kr(3 * nr, K.cols());
      for (int k = 0; k < nr; ++k) Kr.middleRows<3>(3 * k) = K.middleRows<3>(3 * rows[k]);
      Eigen::MatrixXd& M = M_[i];
      M.setZero(3 * nr, 3 * long(m.size()));
      int b = 0;
      for (const auto& [c, l] : m) {
        cols_[i].push_back(c);
        for (const auto& [j, Cm] : l) M.middleCols<3>(3 * b) += Kr.middleCols<3>(3 * j) * Cm;
        ++b;
      }
    }
  }

  // Y = A X, piece by piece (rows of different pieces are disjoint); all columns at once (one GEMM per piece).
  Eigen::MatrixXd apply(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd Y(n_, X.cols());
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(regions.size()); ++i) {
      const auto& c = cols_[i];
      Eigen::MatrixXd xg(3 * c.size(), X.cols());
      for (size_t b = 0; b < c.size(); ++b) xg.middleRows<3>(3 * b) = X.middleRows<3>(3 * c[b]);
      Y.middleRows(3 * row0_[i], M_[i].rows()).noalias() = M_[i] * xg;
    }
    return Y;
  }

  // Restarted right-preconditioned GMRES on all right-hand sides in lockstep (each its own Krylov space; the
  // operator and the preconditioner act on all columns at once: GEMMs instead of GEMVs). Returns the iterations.
  int gmresBatch(const Eigen::MatrixXd& B, Eigen::MatrixXd& X, double& res, int restart = 150, int maxit = 3000) const {
    const int n = int(B.rows()), q = int(B.cols());
    X.setZero(n, q);
    const Eigen::VectorXd nb = B.colwise().norm().transpose().cwiseMax(1e-300);
    Eigen::MatrixXd R = B;
    int it = 0;
    res = 0.0;
    while (it < maxit) {
      const Eigen::VectorXd beta = R.colwise().norm().transpose();
      res = beta.cwiseQuotient(nb).maxCoeff();
      if (res < tol) break;
      std::vector<Eigen::MatrixXd> V(q, Eigen::MatrixXd(n, restart + 1));
      std::vector<Eigen::MatrixXd> H(q, Eigen::MatrixXd::Zero(restart + 1, restart));
      Eigen::MatrixXd cs(restart, q), sn(restart, q), g = Eigen::MatrixXd::Zero(restart + 1, q);
      for (int j = 0; j < q; ++j) { V[j].col(0) = beta(j) > 0 ? Eigen::VectorXd(R.col(j) / beta(j)) : Eigen::VectorXd::Zero(n); g(0, j) = beta(j); }
      int k = 0;
      Eigen::MatrixXd Vk(n, q);
      for (; k < restart && it < maxit; ++k, ++it) {
        for (int j = 0; j < q; ++j) Vk.col(j) = V[j].col(k);
        const Eigen::MatrixXd W = apply(precondition(Vk));
        double worst = 0.0;
#pragma omp parallel for schedule(static) reduction(max : worst)
        for (int j = 0; j < q; ++j) {
          Eigen::VectorXd w = W.col(j);
          Eigen::MatrixXd& Hj = H[j];
          for (int i = 0; i <= k; ++i) { Hj(i, k) = V[j].col(i).dot(w); w -= Hj(i, k) * V[j].col(i); }
          Hj(k + 1, k) = w.norm();
          V[j].col(k + 1) = Hj(k + 1, k) > 0 ? Eigen::VectorXd(w / Hj(k + 1, k)) : Eigen::VectorXd::Zero(n);
          for (int i = 0; i < k; ++i) {
            const double t = cs(i, j) * Hj(i, k) + sn(i, j) * Hj(i + 1, k);
            Hj(i + 1, k) = -sn(i, j) * Hj(i, k) + cs(i, j) * Hj(i + 1, k);
            Hj(i, k) = t;
          }
          const double h = std::hypot(Hj(k, k), Hj(k + 1, k));
          cs(k, j) = h > 0 ? Hj(k, k) / h : 1.0;
          sn(k, j) = h > 0 ? Hj(k + 1, k) / h : 0.0;
          Hj(k, k) = h;
          Hj(k + 1, k) = 0.0;
          g(k + 1, j) = -sn(k, j) * g(k, j);
          g(k, j) = cs(k, j) * g(k, j);
          worst = std::max(worst, std::abs(g(k + 1, j)) / nb(j));
        }
        if (worst < tol) { ++k; ++it; break; }
      }
      Eigen::MatrixXd Y(n, q);
      for (int j = 0; j < q; ++j) {
        Eigen::VectorXd y = Eigen::VectorXd::Zero(k);
        for (int i = k - 1; i >= 0; --i) {  // H(0:k, 0:k) upper triangular (zero pivots: a column already solved)
          double s = g(i, j);
          for (int l = i + 1; l < k; ++l) s -= H[j](i, l) * y(l);
          y(i) = H[j](i, i) != 0.0 ? s / H[j](i, i) : 0.0;
        }
        Y.col(j) = V[j].leftCols(k) * y;
      }
      X += precondition(Y);
      R = B - apply(X);
    }
    res = R.colwise().norm().transpose().cwiseQuotient(nb).maxCoeff();
    return it;
  }

  // Block Jacobi over groups: every unknown is owned by the group that holds its row.
  void blockJacobi() {
    const int nreg = int(regions.size()), ng = group_.back() + 1;
    gfirst_.assign(ng + 1, nreg);
    for (int i = nreg - 1; i >= 0; --i) gfirst_[group_[i]] = i;
    std::vector<int> owner(n_ / 3, -1), local(n_ / 3, -1);
    gown_.assign(ng, {});
    for (int g = 0; g < ng; ++g)
      for (int i = gfirst_[g]; i < gfirst_[g + 1]; ++i)
        for (int u : rowunk_[i]) { owner[u] = g; local[u] = int(gown_[g].size()); gown_[g].push_back(u); }
    P_.clear();
    P_.resize(ng);
    for (int g = 0; g < ng; ++g) {  // LAPACK threads inside each LU
      const int m = int(gown_[g].size());
      Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3 * m, 3 * m);
      int r0 = 0;
      for (int i = gfirst_[g]; i < gfirst_[g + 1]; ++i) {
        for (size_t b = 0; b < cols_[i].size(); ++b)
          if (owner[cols_[i][b]] == g) D.block(3 * r0, 3 * local[cols_[i][b]], M_[i].rows(), 3) = M_[i].middleCols<3>(3 * b);
        r0 += int(rows_[i].size());
      }
      P_[g] = std::make_unique<DenseLU>(std::move(D));
    }
  }
  Eigen::MatrixXd precondition(const Eigen::MatrixXd& R) const {
    Eigen::MatrixXd Z(n_, R.cols());
    for (int g = 0; g < int(P_.size()); ++g) {
      const int m = int(gown_[g].size());
      const Eigen::MatrixXd y = P_[g]->solve(R.middleRows(3 * row0_[gfirst_[g]], 3 * m));
      for (int k = 0; k < m; ++k) Z.middleRows<3>(3 * gown_[g][k]) = y.middleRows<3>(3 * k);
    }
    return Z;
  }

  // Compressed columns of the piece blocks (direct solver): regions in row order, so every column's rows are sorted.
  void compress() {
    const int nreg = int(regions.size()), nblk = n_ / 3;
    std::vector<long> cnt(nblk, 0);
    std::vector<std::vector<long>> start(nreg);
    for (int i = 0; i < nreg; ++i)
      for (int c : cols_[i]) { start[i].push_back(cnt[c]); cnt[c] += M_[i].rows(); }
    std::vector<long> outer(n_ + 1, 0);
    for (int c = 0; c < nblk; ++c)
      for (int d = 0; d < 3; ++d) outer[3 * c + d + 1] = outer[3 * c + d] + cnt[c];
    A_ = Eigen::SparseMatrix<double>(n_, n_);
    A_.resizeNonZeros(outer[n_]);
    for (int k = 0; k <= n_; ++k) A_.outerIndexPtr()[k] = int(outer[k]);
    double* val = A_.valuePtr();
    int* inner = A_.innerIndexPtr();
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nreg; ++i)
      for (size_t b = 0; b < cols_[i].size(); ++b)
        for (int d = 0; d < 3; ++d) {
          const long o = outer[3 * cols_[i][b] + d] + start[i][b];
          for (long k = 0; k < M_[i].rows(); ++k) { val[o + k] = M_[i](k, 3 * b + d); inner[o + k] = int(3 * row0_[i] + k); }
        }
  }
};

}  // namespace igabem
