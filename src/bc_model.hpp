// Local boundary conditions and loads on the cached SBIE rows (H u = G t):
//
//   supports  displacement components prescribed on whole patches (CAD faces,
//             patches, or selector regions that cover their patches: clamps,
//             rollers, symmetry planes, prescribed motions), component by
//             component, so the traction is unknown only in the fixed
//             components;
//   springs   elastic supports t = -k(x, n) u on any region (k per unit area,
//             any 3 x 3 tensor: normal, tangential, foundation);
//   loads     tractions on any region (regions.hpp), integrated exactly as the
//             discontinuous data they are: b = int_R U t dGamma, plus smooth
//             traction data given everywhere (interpolated in the traction
//             space, the NeumannModel/MixedModel path).
//
// Unknowns: the free displacement components, and the traction components that
// are fixed in displacement. Equations: the SBIE component rows at the free
// displacement dofs' collocation points, and the rows of the fixed components
// at the traction collocation points of the supported patches (cached per patch,
// the same cache entries as MixedModel). So any change of supports, springs or
// loads reuses every assembled row: only the column selection, the spring
// columns (cost ~ spring area) and one LU are redone, and each load is one
// right-hand side (cost ~ loaded area).
//
// Rigid modes: the six rigid motions are tested against the supports (fixed
// components) and springs; the ones left free (e.g. none held at all, or a
// translation along a symmetry plane) are bordered out of the LU. Loads must not
// do work on a free motion: the solve reports that work, and the driver refuses
// unbalanced loads, as bemkit does for unrestrained supports.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "operators.hpp"
#include "regions.hpp"
#include "sbie.hpp"
#include "solution.hpp"
#include "traction_space.hpp"

namespace igabem {

struct FixedSupport {
  std::string label;
  std::vector<int> patches;                 // whole patches
  std::array<bool, 3> fix{true, true, true};
  std::function<Vec3(const Vec3&)> value;   // prescribed displacement (empty: zero)
};

struct SpringSupport {
  std::string label;
  std::shared_ptr<const RegionMesh> region;
  std::function<Eigen::Matrix3d(const Vec3& x, const Vec3& n)> k;  // stiffness per unit area
};

struct RegionLoad {
  std::string label;
  std::shared_ptr<const RegionMesh> region;
  std::function<Vec3(const RPoint&)> t;  // traction
};

struct LoadCase {
  std::string name;
  std::vector<RegionLoad> loads;
  std::function<Vec3(const Vec3& y, const Vec3& n)> smooth;  // traction data on the whole surface (optional)
  std::function<Vec3(const Vec3& x)> displacement;           // overrides the supports' values (optional)
};

// Resultant force and moment (about the origin) of a traction field.
struct Wrench {
  Vec3 F = Vec3::Zero(), M = Vec3::Zero();
  Wrench& operator+=(const Wrench& o) { F += o.F; M += o.M; return *this; }
};

struct CaseReport {
  std::string name;
  Wrench applied;                                       // all loads (regions + smooth data)
  std::vector<std::pair<std::string, Wrench>> loads;    // per region load
  std::vector<std::pair<std::string, Wrench>> reactions;  // per support and spring
  Wrench total;              // whole boundary traction (should vanish)
  double free_work = 0.0;    // max |work of the applied loads on a free rigid motion| / scale

  // Equilibrium residual: |resultant of all boundary tractions| over the largest
  // applied or reacting force, and the same for moments about the origin (a pure
  // couple's force scale is M / L, a force's moment scale is F L).
  double residual(double L) const {
    double sF = applied.F.norm(), sM = applied.M.norm();
    for (const auto& r : reactions) { sF = std::max(sF, r.second.F.norm()); sM = std::max(sM, r.second.M.norm()); }
    const double F = std::max({sF, sM / L, 1e-300}), M = std::max({sM, sF * L, 1e-300});
    return std::max(total.F.norm() / F, total.M.norm() / M);
  }
};

// Kelvin U only (displacement kernel), d = y - x.
inline Eigen::Matrix3d kelvinU(const Kelvin& K, const Vec3& d) {
  const double r = d.norm();
  const Vec3 g = d / r;
  const double fu = K.cU / r;
  return fu * ((3.0 - 4.0 * K.nu) * Eigen::Matrix3d::Identity() + g * g.transpose());
}

class BcModel {
 public:
  Mesh mesh;
  TractionSpace ts;
  Material mat;
  SolverOptions opt;
  PartitionOptions popt;
  std::vector<FixedSupport> supports;
  std::vector<SpringSupport> springs;
  std::vector<std::array<bool, 3>> dir;   // per patch: fixed components
  std::vector<int> u_free, u_fixed;       // displacement components (3 g + c)
  std::vector<int> t_free;                // traction components unknown (3 j + c)
  std::vector<Collocation> rowpts;        // collocation points of the rows (3 rows each)
  std::vector<int> rowsel;                // selected rows of 3 * rowpts
  std::vector<int> spring_dofs;           // displacement dofs touched by springs
  Eigen::MatrixXd free_modes;             // rigid motions left free (3 ndof x m)
  bool from_cache = false;
  double t_assemble = 0.0, t_load = 0.0, t_springs = 0.0, t_factor = 0.0;

  void setup(const std::string& geometry_file, const Material& m, const SolverOptions& o,
             std::vector<FixedSupport> fixed, std::vector<SpringSupport> spr, const std::string& cache_dir = "cache",
             const PartitionOptions& po = PartitionOptions{}) {
    mat = m;
    opt = o;
    popt = po;
    mesh.build(loadDat(geometry_file), po);
    ts.build(mesh);
    setupOn(geometry_file, std::move(fixed), std::move(spr), cache_dir);
  }

  // Mesh already built (e.g. to resolve regions first): mesh, ts, mat, opt, popt set.
  void setupOn(const std::string& geometry_file, std::vector<FixedSupport> fixed, std::vector<SpringSupport> spr,
               const std::string& cache_dir = "cache") {
    supports = std::move(fixed);
    springs = std::move(spr);
    const int np = int(mesh.patches.size());
    dir.assign(np, {false, false, false});
    for (const auto& s : supports)
      for (int p : s.patches)
        for (int c = 0; c < 3; ++c) dir[p][c] = dir[p][c] || s.fix[c];
    std::vector<char> fixedc(3 * mesh.ndof, 0);
    for (int p = 0; p < np; ++p)
      for (int g : mesh.l2g[p])
        for (int c = 0; c < 3; ++c) if (dir[p][c]) fixedc[3 * g + c] = 1;
    u_free.clear();
    u_fixed.clear();
    for (int k = 0; k < 3 * mesh.ndof; ++k) (fixedc[k] ? u_fixed : u_free).push_back(k);

    // Rows: free displacement components at the dof collocation points, then
    // the fixed components at the traction points of each supported patch.
    OperatorRows main =
        cachedOperatorRows(mesh, ts, mat, opt, popt, geometry_file, cache_dir, "sbie_hg", mesh.colloc);
    from_cache = main.from_cache;
    t_assemble = main.t_assemble;
    t_load = main.t_load;
    rowpts.clear();
    rowsel.clear();
    t_free.clear();
    std::vector<int> mainrows;
    {
      std::vector<int> pt_of(mesh.ndof, -1);
      for (int k : u_free) {
        const int g = k / 3;
        if (pt_of[g] < 0) { pt_of[g] = int(rowpts.size()); rowpts.push_back(mesh.colloc[g]); }
        rowsel.push_back(3 * pt_of[g] + k % 3);
        mainrows.push_back(k);
      }
    }
    const int nmain = int(mainrows.size());
    std::vector<OperatorRows> tp(np);
    int ntp = 0;
    for (int p = 0; p < np; ++p) {
      if (!(dir[p][0] || dir[p][1] || dir[p][2])) continue;
      tp[p] = cachedOperatorRows(mesh, ts, mat, opt, popt, geometry_file, cache_dir,
                                 "sbie_hg_tpatch" + std::to_string(p), tractionCollocation(mesh, ts, p));
      from_cache = from_cache && tp[p].from_cache;
      t_assemble += tp[p].t_assemble;
      t_load += tp[p].t_load;
      const std::vector<Collocation> pts = tractionCollocation(mesh, ts, p);
      for (size_t i = 0; i < pts.size(); ++i) {
        const int base = int(rowpts.size());
        rowpts.push_back(pts[i]);
        for (int c = 0; c < 3; ++c)
          if (dir[p][c]) {
            rowsel.push_back(3 * base + c);
            t_free.push_back(3 * (ts.offset[p] + int(i)) + c);
            ++ntp;
          }
      }
    }
    const int nr = nmain + ntp;
    if (nr != int(u_free.size() + t_free.size())) throw std::logic_error("BcModel: rows != unknowns");
    H_.resize(nr, 3 * mesh.ndof);
    G_.resize(nr, 3 * ts.ndof);
    H_.topRows(nmain) = main.H(mainrows, Eigen::all);
    G_.topRows(nmain) = main.G(mainrows, Eigen::all);
    main = OperatorRows();
    int r = nmain;
    for (int p = 0; p < np; ++p) {
      if (tp[p].H.size() == 0) continue;
      std::vector<int> sel;
      for (int i = 0; i < int(tp[p].H.rows()) / 3; ++i)
        for (int c = 0; c < 3; ++c) if (dir[p][c]) sel.push_back(3 * i + c);
      H_.middleRows(r, sel.size()) = tp[p].H(sel, Eigen::all);
      G_.middleRows(r, sel.size()) = tp[p].G(sel, Eigen::all);
      r += int(sel.size());
      tp[p] = OperatorRows();
    }

    // Springs: S (rows x spring dofs) = sum_s int_R U k N_B.
    const auto t1 = std::chrono::steady_clock::now();
    buildSprings();
    t_springs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    // Free rigid motions and the bordered LU.
    const auto t2 = std::chrono::steady_clock::now();
    findFreeModes();
    factor();
    t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
  }

  int unknowns() const { return int(u_free.size() + t_free.size()); }
  int freeModes() const { return int(free_modes.cols()); }

  // Boundary solutions of the cases (any number; solved together, 8 at a time).
  std::vector<BoundarySolution> solutions(const std::vector<LoadCase>& cases,
                                          std::vector<CaseReport>* reports = nullptr) const {
    std::vector<BoundarySolution> out;
    for (size_t c0 = 0; c0 < cases.size(); c0 += 8) {
      const int K = int(std::min<size_t>(8, cases.size() - c0));
      auto chunk = solveChunk(cases, c0, K);
      for (auto& s : chunk) out.push_back(std::move(s));
    }
    if (reports) {
      reports->clear();
      for (size_t k = 0; k < cases.size(); ++k) reports->push_back(report(cases[k], out[k]));
    }
    return out;
  }

  // Resultant of the boundary traction (traction-space part over the chosen
  // patches plus the region densities' parts on them).
  Wrench resultant(const BoundarySolution& s, const std::function<bool(int)>& use) const {
    Wrench W;
    const GaussRule g = gaussLegendre(std::max(6, opt.n_gauss / 2));
    PatchEval E;
    for (const Element& e : mesh.elements) {
      if (!use(e.patch)) continue;
      const int off = ts.offset[e.patch];
      for (const Rect& R : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            mesh.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
            Vec3 t = Vec3::Zero();
            for (int l = 0; l < E.n_loc; ++l) t += E.R[l] * s.t.segment<3>(3 * (off + E.idx[l]));
            const Vec3 f = t * (g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm());
            W.F += f;
            W.M += E.x.cross(f);
          }
    }
    for (const auto& d : s.region)
      d->R->forCloud([&](const RPoint& p) {
        if (!use(p.patch)) return;
        const Vec3 f = d->t(p).col(0) * p.w;
        W.F += f;
        W.M += p.x.cross(f);
      });
    return W;
  }

 private:
  Eigen::MatrixXd H_, G_;   // selected rows, all columns
  Eigen::MatrixXd S_;       // selected rows x 3 spring_dofs
  std::vector<int> spring_col_;  // 3 ndof -> column of S_ (-1: none)
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_;
  std::vector<int> ufree_col_;   // 3 ndof -> unknown index (-1: fixed)

  void buildSprings() {
    spring_dofs.clear();
    spring_col_.assign(3 * mesh.ndof, -1);
    S_.resize(0, 0);
    if (springs.empty()) return;
    std::vector<char> touched(mesh.ndof, 0);
    for (const auto& s : springs)
      for (const auto& C : s.region->cells)
        for (int l : C.idx) touched[mesh.l2g[C.patch][l]] = 1;
    for (int g = 0; g < mesh.ndof; ++g)
      if (touched[g]) {
        for (int c = 0; c < 3; ++c) spring_col_[3 * g + c] = 3 * int(spring_dofs.size()) + c;
        spring_dofs.push_back(g);
      }
    const int ns = 3 * int(spring_dofs.size());
    Eigen::MatrixXd Sall = Eigen::MatrixXd::Zero(3 * rowpts.size(), ns);
    const Kelvin Kv(mat);
    for (const auto& s : springs) {
      const RegionMesh& RM = *s.region;
      RegionIntegrator ri(RM, opt);
      // far-field node weights per cell, level, node, local basis: int l_a k N_l
      std::vector<std::array<std::vector<Eigen::Matrix3d>, 4>> W(RM.cells.size());
      for (size_t ic = 0; ic < RM.cells.size(); ++ic) {
        const auto& C = RM.cells[ic];
        const int nl = int(C.idx.size());
        for (int lv = 0; lv < 4; ++lv) {
          const int m = RegionMesh::kM[lv];
          const GaussRule g = gaussLegendre(m);
          W[ic][lv].assign(m * m * nl, Eigen::Matrix3d::Zero());
          double la[10], lb[10];
          for (size_t q = 0; q < C.w.size(); ++q) {
            const Eigen::Matrix3d kq = s.k(C.x[q], C.n[q]) * C.w[q];
            lagrange(g, (C.xi[q](0) - C.rect.u0) / C.rect.du(), la);
            lagrange(g, (C.xi[q](1) - C.rect.v0) / C.rect.dv(), lb);
            for (int a = 0; a < m; ++a)
              for (int b = 0; b < m; ++b)
                for (int l = 0; l < nl; ++l) W[ic][lv][(a * m + b) * nl + l] += (la[a] * lb[b] * C.R[q * nl + l]) * kq;
          }
        }
      }
#pragma omp parallel for schedule(dynamic)
      for (long a = 0; a < long(rowpts.size()); ++a) {
        const Collocation& c = rowpts[a];
        const SourcePoint src{c.patch, c.xi, c.x};
        Eigen::MatrixXd row = Eigen::MatrixXd::Zero(3, ns);
        for (size_t ic = 0; ic < RM.cells.size(); ++ic) {
          const auto& C = RM.cells[ic];
          const int nl = int(C.idx.size());
          const auto& l2g = mesh.l2g[C.patch];
          const int lv = ri.farLevel(C, c.x);
          if (lv >= 0) {
            const int m = RegionMesh::kM[lv];
            for (int nd = 0; nd < m * m; ++nd) {
              const Eigen::Matrix3d U = kelvinU(Kv, C.node_x[lv][nd] - c.x);
              for (int l = 0; l < nl; ++l)
                row.middleCols<3>(spring_col_[3 * l2g[C.idx[l]]]) += U * W[ic][lv][nd * nl + l];
            }
          } else {
            ri.near(src, C, [&](const RPoint& p) {
              const Eigen::Matrix3d UK = kelvinU(Kv, p.x - c.x) * s.k(p.x, p.n) * p.w;
              for (int l = 0; l < p.n_loc; ++l)
                row.middleCols<3>(spring_col_[3 * l2g[p.idx[l]]]) += p.R[l] * UK;
            });
          }
        }
        Sall.middleRows(3 * a, 3) += row;
      }
    }
    S_ = Sall(rowsel, Eigen::all);
  }

  // Rigid motions not held by the fixed components or the springs.
  void findFreeModes() {
    Eigen::MatrixXd R6 = Eigen::MatrixXd::Zero(3 * mesh.ndof, 6);
    Vec3 o = Vec3::Zero();
    for (const Vec3& P : mesh.dof_point) o += P / mesh.ndof;
    for (int g = 0; g < mesh.ndof; ++g) {
      for (int i = 0; i < 3; ++i) R6(3 * g + i, i) = 1.0;
      for (int k = 0; k < 3; ++k) R6.block<3, 1>(3 * g, 3 + k) = Vec3::Unit(k).cross(mesh.dof_point[g] - o);
    }
    for (int k = 0; k < 6; ++k) R6.col(k).normalize();
    Eigen::Matrix<double, 6, 6> Gm = Eigen::Matrix<double, 6, 6>::Zero();
    if (!u_fixed.empty()) {
      const Eigen::MatrixXd B = R6(u_fixed, Eigen::all);
      Eigen::Matrix<double, 6, 6> Gf = B.transpose() * B;
      Gm += Gf / Gf.trace();
    }
    if (S_.size()) {
      std::vector<int> rows;
      for (int g : spring_dofs)
        for (int c = 0; c < 3; ++c) rows.push_back(3 * g + c);
      const Eigen::MatrixXd B = S_ * R6(rows, Eigen::all);
      Eigen::Matrix<double, 6, 6> Gs = B.transpose() * B;
      Gm += Gs / Gs.trace();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(Gm);
    std::vector<int> fr;
    for (int k = 0; k < 6; ++k)
      if (es.eigenvalues()(k) < 1e-10) fr.push_back(k);
    free_modes.resize(3 * mesh.ndof, fr.size());
    for (size_t k = 0; k < fr.size(); ++k) free_modes.col(k) = (R6 * es.eigenvectors().col(fr[k])).normalized();
  }

  void factor() {
    const int n = unknowns(), m = freeModes();
    ufree_col_.assign(3 * mesh.ndof, -1);
    for (size_t j = 0; j < u_free.size(); ++j) ufree_col_[u_free[j]] = int(j);
    Eigen::MatrixXd A(n + m, n + m);
    A.setZero();
    A.topLeftCorner(n, u_free.size()) = H_(Eigen::all, u_free);
    if (S_.size())
      for (int g : spring_dofs)
        for (int c = 0; c < 3; ++c) {
          const int j = ufree_col_[3 * g + c];
          if (j >= 0) A.col(j).head(n) += S_.col(spring_col_[3 * g + c]);
        }
    const int c0 = int(u_free.size());
    for (size_t j = 0; j < t_free.size(); ++j) A.col(c0 + j).head(n) = -G_.col(t_free[j]);
    if (m > 0) {
      const double s = A.topLeftCorner(n, n).cwiseAbs().maxCoeff();
      const Eigen::MatrixXd Cu = free_modes(u_free, Eigen::all);
      A.block(0, n, c0, m) = s * Cu;
      A.block(n, 0, m, c0) = s * Cu.transpose();
    }
    lu_.compute(A);
  }

  // Greville interpolation of a displacement field on one patch (as MixedModel).
  void interpolatePatch(int p, const std::function<Vec3(const Vec3&)>& u, const std::array<bool, 3>& mask,
                        Eigen::MatrixXd& UC, int col) const {
    const Patch& P = mesh.patches[p];
    const int n = P.nu * P.nv;
    Eigen::MatrixXd Np = Eigen::MatrixXd::Zero(n, n), rhs(n, 3);
    PatchEval E;
    for (int k = 0; k < n; ++k) {
      P.eval(P.greville(k % P.nu, true), P.greville(k / P.nu, false), E);
      for (int l = 0; l < E.n_loc; ++l) Np(k, E.idx[l]) += E.R[l];
      rhs.row(k) = u(E.x).transpose();
    }
    const Eigen::MatrixXd c = Np.partialPivLu().solve(rhs);
    for (int k = 0; k < n; ++k)
      for (int cc = 0; cc < 3; ++cc)
        if (mask[cc]) UC(3 * mesh.l2g[p][k] + cc, col) = c(k, cc);
  }

  std::vector<BoundarySolution> solveChunk(const std::vector<LoadCase>& cases, size_t c0, int K) const {
    const int n = unknowns(), m = freeModes();
    // smooth traction data
    Eigen::MatrixXd TC = Eigen::MatrixXd::Zero(3 * ts.ndof, K);
    {
      bool any = false;
      for (int k = 0; k < K; ++k) any = any || bool(cases[c0 + k].smooth);
      if (any)
        TC = ts.interpolate(
            [&](const Vec3& y, const Vec3& nn) {
              TractionBlock T = TractionBlock::Zero(3, K);
              for (int k = 0; k < K; ++k)
                if (cases[c0 + k].smooth) T.col(k) = cases[c0 + k].smooth(y, nn);
              return T;
            },
            K);
    }
    // prescribed displacements
    Eigen::MatrixXd UC = Eigen::MatrixXd::Zero(3 * mesh.ndof, K);
    for (int k = 0; k < K; ++k) {
      const LoadCase& lc = cases[c0 + k];
      if (lc.displacement) {
        for (size_t p = 0; p < mesh.patches.size(); ++p)
          if (dir[p][0] || dir[p][1] || dir[p][2]) interpolatePatch(int(p), lc.displacement, dir[p], UC, k);
      } else {
        for (const auto& s : supports)
          if (s.value)
            for (int p : s.patches) interpolatePatch(p, s.value, s.fix, UC, k);
      }
    }
    Eigen::MatrixXd tk = TC, uk = Eigen::MatrixXd::Zero(3 * mesh.ndof, K);
    for (int j : t_free) tk.row(j).setZero();
    for (int j : u_fixed) uk.row(j) = UC.row(j);
    Eigen::MatrixXd rhs = G_ * tk - H_ * uk;
    if (S_.size()) {
      Eigen::MatrixXd us(S_.cols(), K);
      for (size_t i = 0; i < spring_dofs.size(); ++i) us.middleRows(3 * i, 3) = uk.middleRows(3 * spring_dofs[i], 3);
      rhs -= S_ * us;
    }
    // region loads, grouped by region: b = int_R U t
    std::map<const RegionMesh*, std::vector<std::pair<int, const RegionLoad*>>> byRegion;
    for (int k = 0; k < K; ++k)
      for (const auto& L : cases[c0 + k].loads) byRegion[L.region.get()].push_back({k, &L});
    const Kelvin Kv(mat);
    for (const auto& [RM, list] : byRegion) {
      const RegionDensity d(*RM, K, [&, list = list](const RPoint& p) {
        RegionBlock T = RegionBlock::Zero(3, K);
        for (const auto& [k, L] : list) T.col(k) += L->t(p);
        return T;
      });
      const RegionIntegrator ri(*RM, opt);
      Eigen::MatrixXd b(3 * rowpts.size(), K);
#pragma omp parallel for schedule(dynamic)
      for (long a = 0; a < long(rowpts.size()); ++a) {
        const Collocation& c = rowpts[a];
        RegionBlock acc = RegionBlock::Zero(3, K);
        ri.visit(SourcePoint{c.patch, c.xi, c.x}, d,
                 [&](const Vec3& y, const Vec3&, const RegionBlock& tw) { acc.noalias() += kelvinU(Kv, y - c.x) * tw; });
        b.middleRows(3 * a, 3) = acc;
      }
      rhs += b(rowsel, Eigen::all);
    }
    Eigen::MatrixXd bb = Eigen::MatrixXd::Zero(n + m, K);
    bb.topRows(n) = rhs;
    const Eigen::MatrixXd X = lu_.solve(bb);
    std::vector<BoundarySolution> out(K);
    for (int k = 0; k < K; ++k) {
      Eigen::VectorXd u = UC.col(k), t = TC.col(k);
      for (size_t j = 0; j < u_free.size(); ++j) u(u_free[j]) = X(j, k);
      for (size_t j = 0; j < t_free.size(); ++j) t(t_free[j]) = X(u_free.size() + j, k);
      BoundarySolution s{&mesh, &ts, mat, u, t, {}};
      for (const auto& L : cases[c0 + k].loads) {
        auto f = L.t;
        s.region.push_back(std::make_shared<RegionDensity>(*L.region, 1, [f](const RPoint& p) {
          return RegionBlock(f(p));
        }));
      }
      for (const auto& sp : springs) {
        auto kf = sp.k;
        const Mesh* me = &mesh;
        auto uu = std::make_shared<Eigen::VectorXd>(u);
        s.region.push_back(std::make_shared<RegionDensity>(*sp.region, 1, [kf, me, uu](const RPoint& p) {
          Vec3 v = Vec3::Zero();
          for (int l = 0; l < p.n_loc; ++l) v += p.R[l] * uu->segment<3>(3 * me->l2g[p.patch][p.idx[l]]);
          return RegionBlock(-kf(p.x, p.n) * v);
        }));
      }
      out[k] = std::move(s);
    }
    return out;
  }

  CaseReport report(const LoadCase& lc, const BoundarySolution& s) const {
    CaseReport r;
    r.name = lc.name;
    for (size_t i = 0; i < lc.loads.size(); ++i) {
      Wrench W;
      s.region[i]->R->forCloud([&](const RPoint& p) {
        const Vec3 f = s.region[i]->t(p).col(0) * p.w;
        W.F += f;
        W.M += p.x.cross(f);
      });
      r.loads.push_back({lc.loads[i].label, W});
      r.applied += W;
    }
    if (lc.smooth) {
      // smooth data where it is prescribed (not on fixed components)
      BoundarySolution d{&mesh, &ts, mat, s.u, s.t, {}};
      for (int j : t_free) d.t(j) = 0.0;
      r.applied += resultant(d, [](int) { return true; });
    }
    for (const auto& sp : supports) {
      std::vector<char> on(mesh.patches.size(), 0);
      for (int p : sp.patches) on[p] = 1;
      BoundarySolution d{&mesh, &ts, mat, s.u, Eigen::VectorXd::Zero(s.t.size()), {}};
      for (int j : t_free)
        if (on[ts.points[j / 3].patch] && sp.fix[j % 3]) d.t(j) = s.t(j);
      r.reactions.push_back({sp.label, resultant(d, [&](int p) { return bool(on[p]); })});
    }
    for (size_t i = 0; i < springs.size(); ++i) {
      Wrench W;
      const auto& d = s.region[lc.loads.size() + i];
      d->R->forCloud([&](const RPoint& p) {
        const Vec3 f = d->t(p).col(0) * p.w;
        W.F += f;
        W.M += p.x.cross(f);
      });
      r.reactions.push_back({springs[i].label, W});
    }
    r.total = resultant(s, [](int) { return true; });
    // work of the applied loads on the free rigid motions (bordered out)
    if (freeModes() > 0) {
      double scale = 0.0;
      Eigen::VectorXd w = Eigen::VectorXd::Zero(freeModes());
      for (size_t i = 0; i < lc.loads.size(); ++i)
        s.region[i]->R->forCloud([&](const RPoint& p) {
          const Vec3 f = s.region[i]->t(p).col(0) * p.w;
          scale += f.norm();
          for (int k = 0; k < freeModes(); ++k) {
            Vec3 v = Vec3::Zero();
            for (int l = 0; l < p.n_loc; ++l) v += p.R[l] * free_modes.col(k).segment<3>(3 * mesh.l2g[p.patch][p.idx[l]]);
            w(k) += v.dot(f);
          }
        });
      if (lc.smooth) {
        Eigen::VectorXd tt = s.t;
        for (int j : t_free) tt(j) = 0.0;
        const GaussRule g = gaussLegendre(6);
        PatchEval E;
        for (const Element& e : mesh.elements)
          for (const Rect& R : e.cells)
            for (int a = 0; a < 6; ++a)
              for (int b = 0; b < 6; ++b) {
                mesh.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
                Vec3 t = Vec3::Zero();
                for (int l = 0; l < E.n_loc; ++l) t += E.R[l] * tt.segment<3>(3 * (ts.offset[e.patch] + E.idx[l]));
                const Vec3 f = t * (g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm());
                scale += f.norm();
                for (int k = 0; k < freeModes(); ++k) {
                  Vec3 v = Vec3::Zero();
                  for (int l = 0; l < E.n_loc; ++l)
                    v += E.R[l] * free_modes.col(k).segment<3>(3 * mesh.l2g[e.patch][E.idx[l]]);
                  w(k) += v.dot(f);
                }
              }
      }
      // free_modes columns are unit vectors over 3 ndof: |v| ~ 1/sqrt(ndof)
      const double vmax = free_modes.cwiseAbs().maxCoeff();
      r.free_work = scale > 0 ? w.cwiseAbs().maxCoeff() / (scale * vmax) : 0.0;
    }
    return r;
  }
};

}  // namespace igabem
