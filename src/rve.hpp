// Homogenization of a periodic cell (RVE): the effective stiffness C of a box
// [lo, hi] of one isotropic solid with traction-free pores (a printed bead stack,
// a porous cell), from one SBIE assembly and three kinds of cell boundary conditions
// admitted by Hill-Mandel (Hazanov & Huet 1994; Biswas 2019, eq. 2.3-2.6):
//
//   KUBC  u = E x on the cell faces                       (upper bound)
//   PBC   u+ - u- = E (x+ - x-),  t+ = -t-  on face pairs  (the periodic answer)
//   SUBC  t = S n on the cell faces                       (lower bound)
//
// The cell faces are the planar patches lying on the box planes; every other patch
// (pores, bead surfaces) is traction free. With no body force the cell averages are
// boundary integrals,
//
//   sigma_bar = (1/V) int_dOmega sym(t x x) dS,   eps_bar = (1/V) int_faces sym(u x n) dS,
//
// V the box volume. eps_bar is E for KUBC and PBC (by construction); for SUBC it
// needs faces with no pore openings ("closed" cells), else it is not defined. The
// Hill-Mandel energy identity sigma_bar : eps_bar = (1/V) int t.u dS is reported.
//
// One assembly: the SBIE rows at every displacement collocation point and at the
// traction collocation points of the face patches. Each BC picks rows and columns,
// one LU, and the six unit macro cases are six right-hand sides.
//   KUBC  unknowns: u off the faces, t on the faces; rows: dof points off the
//         faces, traction points of the faces. u on the faces is E x exactly
//         (linear fields are in the isoparametric space).
//   PBC   face dofs translated by the cell lengths are merged into one
//         representative on the lo faces (u_g = u_rep + E s_g, s_g a lattice
//         vector); the traction on a - face is minus the traction at the translated
//         + face point. Unknowns: representatives' u, + face t; rows:
//         representatives' dof points (- faces), traction points of the + faces, so
//         both faces of a pair carry equations (rows on one side only are singular). Needs the - faces to be exact
//         translated copies of the + faces (python/rve writes them so). The three
//         translations are bordered out.
//   SUBC  unknowns: all u; rows: all dof points; six rigid modes bordered out.
//
// Voigt order 11, 22, 33, 23, 13, 12 with engineering shear strains.
#pragma once

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

#include "operators.hpp"
#include "sbie.hpp"
#include "solution.hpp"

namespace igabem {

enum class RveBc { Kubc, Pbc, Subc };
inline const char* rveBcName(RveBc b) { return b == RveBc::Kubc ? "KUBC" : b == RveBc::Pbc ? "PBC" : "SUBC"; }

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6r = Eigen::Matrix<double, 6, 1>;

inline constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};

// Unit macro strain (engineering shear: gamma = 1) or unit macro stress of Voigt component k.
inline Eigen::Matrix3d voigtUnit(int k, bool strain) {
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  const int i = kVoigt[k][0], j = kVoigt[k][1];
  const double v = (k < 3 || !strain) ? 1.0 : 0.5;
  M(i, j) = v;
  M(j, i) = v;
  return M;
}
inline Vec6r toVoigt(const Eigen::Matrix3d& M, bool strain) {
  Vec6r v;
  for (int k = 0; k < 6; ++k) v(k) = (k < 3 || !strain ? 1.0 : 2.0) * M(kVoigt[k][0], kVoigt[k][1]);
  return v;
}

// Orthotropic engineering constants of a stiffness (in its own axes).
struct Engineering {
  double E1, E2, E3, G23, G13, G12, nu12, nu13, nu23;
};
inline Engineering engineering(const Mat6& C) {
  const Mat6 S = C.inverse();
  return {1 / S(0, 0), 1 / S(1, 1), 1 / S(2, 2), 1 / S(3, 3), 1 / S(4, 4), 1 / S(5, 5),
          -S(0, 1) / S(0, 0), -S(0, 2) / S(0, 0), -S(1, 2) / S(1, 1)};
}
inline Mat6 isotropicC(const Material& m) {
  Mat6 C = Mat6::Zero();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) C(i, j) = m.lambda;
    C(i, i) += 2 * m.mu;
    C(3 + i, 3 + i) = m.mu;
  }
  return C;
}

struct RveCaseResult {
  Eigen::Matrix3d sigma, eps;  // cell averages
  double energy = 0.0;         // (1/V) int t.u dS
};

struct RveResult {
  RveBc bc;
  Mat6 C;                     // effective stiffness, Voigt
  std::array<RveCaseResult, 6> cases;
  std::vector<BoundarySolution> solutions;
  double hill_mandel = 0.0;   // max_k |energy - sigma:eps| / |sigma:eps|
  double asymmetry = 0.0;     // |C - C^T| / |C|
  int unknowns = 0;
  double t_factor = 0.0, t_solve = 0.0;
};

class RveModel {
 public:
  Mesh mesh;
  TractionSpace ts;
  Material mat;
  SolverOptions opt;
  PartitionOptions popt;
  Vec3 lo, hi;
  std::vector<int> face;  // per patch: 2 axis + side (0: lo, 1: hi), -1: not a face
  bool closed = false;    // faces cover the box faces (no pore openings)
  // Imperfect interfaces on the periodic face pairs (PBC only), per axis a: the + face displacement is
  // u+ = u-(x - L_a e_a) + E L_a e_a - weld[a] t+, t+ the traction on the + face, weld[a] = K^-1 (3 x 3 compliance,
  // mm^3/N). Zero (default): perfect continuity. For a bead-boundary cell (python/rve/beads.py bead_00) the y and z
  // face pairs are the in-layer and interlayer welds and setWeld gives them a linear traction-separation law.
  std::array<Eigen::Matrix3d, 3> weld{Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};

  // Normal stiffness Kn and tangential Kt (N/mm^3) of the welds normal to axis a; Kn <= 0: perfect weld.
  void setWeld(int a, double Kn, double Kt) {
    const Eigen::Matrix3d nn = Vec3::Unit(a) * Vec3::Unit(a).transpose();
    weld[a] = Kn > 0 ? Eigen::Matrix3d(nn / Kn + (Eigen::Matrix3d::Identity() - nn) / Kt) : Eigen::Matrix3d::Zero();
  }
  double t_assemble = 0.0, volume_solid = 0.0;

  Vec3 lengths() const { return hi - lo; }
  double volume() const { const Vec3 L = lengths(); return L(0) * L(1) * L(2); }
  double porosity() const { return 1.0 - volume_solid / volume(); }

  void setup(const std::string& geometry_file, const Material& m, const SolverOptions& o,
             const std::string& cache_dir = "", const PartitionOptions& po = PartitionOptions{}) {
    mat = m;
    opt = o;
    popt = po;
    mesh.build(loadDat(geometry_file), po);
    ts.build(mesh);
    volume_solid = mesh.volume();
    lo = hi = mesh.dof_point[0];
    for (const Vec3& P : mesh.dof_point) { lo = lo.cwiseMin(P); hi = hi.cwiseMax(P); }
    tol_ = 1e-9 * (hi - lo).maxCoeff();
    const int np = int(mesh.patches.size());
    face.assign(np, -1);
    for (int p = 0; p < np; ++p) {
      const Patch& P = mesh.patches[p];
      for (int a = 0; a < 3 && face[p] < 0; ++a)
        for (int s = 0; s < 2 && face[p] < 0; ++s) {
          const double c = s ? hi(a) : lo(a);
          bool on = true;
          for (int k = 0; k < P.nu * P.nv && on; ++k) on = std::abs(P.cp(k)(a) - c) < tol_;
          if (on) face[p] = 2 * a + s;
        }
    }
    // Face areas against the box faces: closed cell or pore openings.
    std::array<double, 6> area{};
    {
      const GaussRule g = gaussLegendre(8);
      PatchEval E;
      for (const Element& e : mesh.elements) {
        if (face[e.patch] < 0) continue;
        for (const Rect& R : e.cells)
          for (size_t a = 0; a < g.x.size(); ++a)
            for (size_t b = 0; b < g.x.size(); ++b) {
              mesh.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
              area[face[e.patch]] += g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm();
            }
      }
    }
    const Vec3 L = lengths();
    closed = true;
    for (int f = 0; f < 6; ++f) {
      const int a = f / 2;
      const double full = L((a + 1) % 3) * L((a + 2) % 3);
      if (area[f] <= 0) throw std::runtime_error("RveModel: no patch on cell face " + std::to_string(f));
      closed = closed && std::abs(area[f] - full) < 1e-8 * full;
    }
    // Rows: every dof point, then the traction points of the face patches.
    face_row0_.assign(np, -1);
    std::vector<Collocation> trows;
    for (int p = 0; p < np; ++p) {
      if (face[p] < 0) continue;
      face_row0_[p] = 3 * (mesh.ndof + int(trows.size()));
      for (auto& c : tractionCollocation(mesh, ts, p)) trows.push_back(std::move(c));
    }
    std::vector<Collocation> rows = mesh.colloc;
    rows.insert(rows.end(), trows.begin(), trows.end());
    OperatorRows A = cachedOperatorRows(mesh, ts, mat, opt, popt, geometry_file, cache_dir, "rve_rows", rows);
    t_assemble = A.t_assemble;
    H_ = std::move(A.H);
    G_ = std::move(A.G);
  }

  // Effective stiffness under one kind of cell BC (six unit macro cases).
  RveResult solve(RveBc bc) const {
    RveResult res;
    res.bc = bc;
    const auto t0 = std::chrono::steady_clock::now();
    System S = bc == RveBc::Kubc ? kubc() : bc == RveBc::Pbc ? pbc() : subc();
    const int n = int(S.A.rows()) - int(S.modes.cols());
    res.unknowns = n;
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(S.A);
    const auto t1 = std::chrono::steady_clock::now();
    res.t_factor = std::chrono::duration<double>(t1 - t0).count();
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(S.A.rows(), 6);
    rhs.topRows(n) = S.rhs;
    const Eigen::MatrixXd X = lu.solve(rhs);
    Mat6 Sig, Eps;
    for (int k = 0; k < 6; ++k) {
      BoundarySolution s{&mesh, &ts, mat, Eigen::VectorXd::Zero(3 * mesh.ndof), Eigen::VectorXd::Zero(3 * ts.ndof), {}};
      S.expand(X.col(k), k, s.u, s.t);
      RveCaseResult& c = res.cases[k];
      averages(s, c);
      if (bc != RveBc::Subc) c.eps = voigtUnit(k, true);
      else if (!closed) throw std::runtime_error("RveModel: SUBC strain needs a closed cell (no pore openings)");
      Sig.col(k) = toVoigt(c.sigma, false);
      Eps.col(k) = toVoigt(c.eps, true);
      const double w = (c.sigma.array() * c.eps.array()).sum();
      res.hill_mandel = std::max(res.hill_mandel, std::abs(c.energy - w) / std::abs(w));
      res.solutions.push_back(std::move(s));
    }
    res.C = Sig * Eps.inverse();
    res.asymmetry = (res.C - res.C.transpose()).norm() / res.C.norm();
    res.t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    return res;
  }

  // Cell averages and energy of a boundary solution.
  void averages(const BoundarySolution& s, RveCaseResult& c) const {
    const GaussRule g = gaussLegendre(std::max(6, opt.n_gauss / 2));
    PatchEval E;
    c.sigma.setZero();
    c.eps.setZero();
    c.energy = 0.0;
    for (const Element& e : mesh.elements)
      for (const Rect& R : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            mesh.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
            const Vec3 nn = E.normal() * mesh.normal_sign;
            const double w = g.w[a] * g.w[b] * R.du() * R.dv() * nn.norm();
            const Vec3 u = s.displacement(e.patch, E), t = s.traction(e.patch, E);
            c.sigma += w * t * E.x.transpose();
            c.energy += w * t.dot(u);
            if (face[e.patch] >= 0 && face[e.patch] % 2 == 1) c.energy += w * t.dot(weld[face[e.patch] / 2] * t);  // weld
            if (face[e.patch] >= 0) c.eps += w * u * nn.normalized().transpose();
          }
    const double V = volume();
    c.sigma = (0.5 / V * (c.sigma + c.sigma.transpose())).eval();
    c.eps = (0.5 / V * (c.eps + c.eps.transpose())).eval();
    c.energy /= V;
  }

 public:  // internals, exposed for tests
  Eigen::MatrixXd H_, G_;      // rows: 3 ndof dof points, then the face traction points
  std::vector<int> face_row0_; // per face patch: first row of its traction points
  double tol_ = 1e-12;

  struct System {
    Eigen::MatrixXd A;    // bordered system matrix
    Eigen::MatrixXd rhs;  // n x 6
    Eigen::MatrixXd modes;
    std::function<void(const Eigen::VectorXd&, int, Eigen::VectorXd&, Eigen::VectorXd&)> expand;
  };

  std::vector<int> tractionDofs(int p) const {
    const int n0 = ts.offset[p], n1 = p + 1 < int(ts.offset.size()) ? ts.offset[p + 1] : ts.ndof;
    std::vector<int> d;
    for (int i = n0; i < n1; ++i) d.push_back(i);
    return d;
  }
  std::vector<char> faceDofs() const {
    std::vector<char> on(mesh.ndof, 0);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0)
        for (int g : mesh.l2g[p]) on[g] = 1;
    return on;
  }
  static std::vector<int> comps(const std::vector<int>& idx, int off = 0) {
    std::vector<int> r;
    for (int i : idx)
      for (int c = 0; c < 3; ++c) r.push_back(off + 3 * i + c);
    return r;
  }
  // Rigid motions (translations, then rotations about the cell centre) on a dof subset.
  Eigen::MatrixXd rigid(const std::vector<int>& dofs, int nmodes) const {
    const Vec3 o = 0.5 * (lo + hi);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3 * dofs.size(), nmodes);
    for (size_t j = 0; j < dofs.size(); ++j) {
      for (int i = 0; i < 3; ++i) R(3 * j + i, i) = 1.0;
      if (nmodes > 3)
        for (int k = 0; k < 3; ++k) R.block<3, 1>(3 * j, 3 + k) = Vec3::Unit(k).cross(mesh.dof_point[dofs[j]] - o);
    }
    for (int k = 0; k < nmodes; ++k) R.col(k).normalize();
    return R;
  }
  static Eigen::MatrixXd bordered(const Eigen::MatrixXd& M, const Eigen::MatrixXd& modes, int nu) {
    const int n = int(M.rows()), m = int(modes.cols());
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + m, n + m);
    A.topLeftCorner(n, n) = M;
    if (m) {
      const double s = M.cwiseAbs().maxCoeff();
      A.block(0, n, nu, m) = s * modes;
      A.block(n, 0, m, nu) = s * modes.transpose();
    }
    return A;
  }

  System kubc() const {
    const auto on = faceDofs();
    std::vector<int> inner, outer, tdofs, trows;
    for (int g = 0; g < mesh.ndof; ++g) (on[g] ? outer : inner).push_back(g);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0) {
        const auto d = tractionDofs(int(p));
        for (size_t i = 0; i < d.size(); ++i) {
          tdofs.push_back(d[i]);
          trows.push_back(face_row0_[p] / 3 + int(i));
        }
      }
    std::vector<int> rows = comps(inner);
    for (int r : comps(trows)) rows.push_back(r);
    const std::vector<int> uc = comps(inner), tc = comps(tdofs), oc = comps(outer);
    const int nu = int(uc.size()), n = nu + int(tc.size());
    if (int(rows.size()) != n) throw std::logic_error("KUBC: rows != unknowns");
    Eigen::MatrixXd M(n, n);
    M.leftCols(nu) = H_(rows, uc);
    M.rightCols(n - nu) = -G_(rows, tc);
    Eigen::MatrixXd Uo(oc.size(), 6);
    for (int k = 0; k < 6; ++k) {
      const Eigen::Matrix3d Ek = voigtUnit(k, true);
      for (size_t j = 0; j < outer.size(); ++j) Uo.block<3, 1>(3 * j, k) = Ek * mesh.dof_point[outer[j]];
    }
    System S;
    S.A = M;
    S.rhs = -H_(rows, oc) * Uo;
    S.modes.resize(n, 0);
    S.expand = [=](const Eigen::VectorXd& x, int k, Eigen::VectorXd& u, Eigen::VectorXd& t) {
      for (int j = 0; j < nu; ++j) u(uc[j]) = x(j);
      for (size_t j = 0; j < oc.size(); ++j) u(oc[j]) = Uo(j, k);
      for (size_t j = 0; j < tc.size(); ++j) t(tc[j]) = x(nu + j);
    };
    return S;
  }

  System subc() const {
    std::vector<int> all(mesh.ndof);
    for (int g = 0; g < mesh.ndof; ++g) all[g] = g;
    const int n = 3 * mesh.ndof;
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(3 * ts.ndof, 6);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0) {
        Vec3 nrm = Vec3::Zero();
        nrm(face[p] / 2) = face[p] % 2 ? 1.0 : -1.0;
        for (int d : tractionDofs(int(p)))
          for (int k = 0; k < 6; ++k) T.block<3, 1>(3 * d, k) = voigtUnit(k, false) * nrm;
      }
    System S;
    S.modes = rigid(all, 6);
    S.A = bordered(H_.topRows(n), S.modes, n);
    S.rhs = G_.topRows(n) * T;
    S.expand = [=](const Eigen::VectorXd& x, int k, Eigen::VectorXd& u, Eigen::VectorXd& t) {
      u = x.head(n);
      t = T.col(k);
    };
    return S;
  }

  // Coordinate lookup of points (tolerance tol_) for the face pairings.
  struct PointIndex {
    double h;
    std::map<std::array<long long, 3>, std::vector<std::pair<Vec3, int>>> cells;
    explicit PointIndex(double tol) : h(1e3 * tol) {}
    std::array<long long, 3> key(const Vec3& x) const {
      return {(long long)std::floor(x(0) / h), (long long)std::floor(x(1) / h), (long long)std::floor(x(2) / h)};
    }
    void add(const Vec3& x, int id) { cells[key(x)].push_back({x, id}); }
    int find(const Vec3& x, double tol) const {
      const auto k = key(x);
      for (long long a = -1; a <= 1; ++a)
        for (long long b = -1; b <= 1; ++b)
          for (long long c = -1; c <= 1; ++c) {
            auto it = cells.find({k[0] + a, k[1] + b, k[2] + c});
            if (it == cells.end()) continue;
            for (const auto& [y, id] : it->second)
              if ((y - x).norm() < tol) return id;
          }
      return -1;
    }
  };

  System pbc() const {
    const Vec3 L = lengths();
    const auto on = faceDofs();
    // Merge translated face dofs (union-find).
    std::vector<int> parent(mesh.ndof);
    for (int g = 0; g < mesh.ndof; ++g) parent[g] = g;
    std::function<int(int)> root = [&](int g) { return parent[g] == g ? g : parent[g] = root(parent[g]); };
    PointIndex dix(tol_);
    for (int g = 0; g < mesh.ndof; ++g)
      if (on[g]) dix.add(mesh.dof_point[g], g);
    const double ptol = 1e3 * tol_;
    for (int g = 0; g < mesh.ndof; ++g) {
      if (!on[g]) continue;
      for (int a = 0; a < 3; ++a) {
        if (std::abs(mesh.dof_point[g](a) - lo(a)) > ptol) continue;
        const int h = dix.find(mesh.dof_point[g] + L(a) * Vec3::Unit(a), ptol);
        if (h < 0) throw std::runtime_error("PBC: face dof without a translated partner (faces must be copies)");
        const int rg = root(g), rh = root(h);
        if (rg != rh) parent[std::max(rg, rh)] = std::min(rg, rh);
      }
    }
    // Representative of a class: its member on the lo faces (smallest coordinate sum), so the dof rows sit on
    // the - faces and the traction rows (below) on the + faces: every face pair gets equations on both sides.
    std::vector<int> reps, rep_of(mesh.ndof), col_of(mesh.ndof, -1), best(mesh.ndof, -1);
    for (int g = 0; g < mesh.ndof; ++g) {
      const int r = root(g);
      if (best[r] < 0 || mesh.dof_point[g].sum() < mesh.dof_point[best[r]].sum() - ptol) best[r] = g;
    }
    for (int g = 0; g < mesh.ndof; ++g) rep_of[g] = best[root(g)];
    for (int g = 0; g < mesh.ndof; ++g)
      if (rep_of[g] == g) { col_of[g] = int(reps.size()); reps.push_back(g); }
    // Pair - face traction points with + face points.
    PointIndex tix(tol_);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0 && face[p] % 2 == 1)
        for (int d : tractionDofs(int(p))) tix.add(ts.points[d].x, d);
    std::vector<int> plus, tcol(ts.ndof, -1), minus_of, mrows;
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0 && face[p] % 2 == 1)
        for (int d : tractionDofs(int(p))) { tcol[d] = int(plus.size()); plus.push_back(d); }
    minus_of.assign(plus.size(), -1);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0 && face[p] % 2 == 0) {
        const int a = face[p] / 2;
        const auto d = tractionDofs(int(p));
        for (size_t i = 0; i < d.size(); ++i) {
          const int q = tix.find(ts.points[d[i]].x + L(a) * Vec3::Unit(a), ptol);
          if (q < 0 || minus_of[tcol[q]] >= 0) throw std::runtime_error("PBC: - face traction point without a unique + partner");
          minus_of[tcol[q]] = d[i];
        }
      }
    for (int m : minus_of)
      if (m < 0) throw std::runtime_error("PBC: + face traction point without a - partner");
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (face[p] >= 0 && face[p] % 2 == 1)
        for (size_t i = 0; i < tractionDofs(int(p)).size(); ++i) mrows.push_back(face_row0_[p] / 3 + int(i));
    // Weld jumps: every dof's displacement is its representative's plus E times the shift plus, across each + face
    // crossed on the way, -weld[a] times the + face traction there (same NURBS basis for u and t on a face patch,
    // so the jump coefficient of a control point is the traction coefficient of that control point; averaged over
    // the + face patches that share it). Walk each class from its representative.
    std::vector<std::vector<std::pair<int, Eigen::Matrix3d>>> jump(mesh.ndof);
    bool imperfect = false;
    for (int a = 0; a < 3; ++a) imperfect |= weld[a].squaredNorm() > 0;
    if (imperfect) {
      std::vector<std::vector<std::pair<int, double>>> tplus(3 * mesh.ndof);  // (dof, axis) -> (traction col, weight)
      for (size_t p = 0; p < mesh.patches.size(); ++p)
        if (face[p] >= 0 && face[p] % 2 == 1)
          for (size_t l = 0; l < mesh.l2g[p].size(); ++l)
            tplus[3 * mesh.l2g[p][l] + face[p] / 2].push_back({tcol[ts.offset[p] + int(l)], 1.0});
      for (auto& v : tplus)
        for (auto& e : v) e.second = 1.0 / double(v.size());
      std::vector<std::vector<std::tuple<int, int, int>>> adj(mesh.ndof);  // (neighbour, axis, +1: neighbour is +)
      for (int g = 0; g < mesh.ndof; ++g) {
        if (!on[g]) continue;
        for (int a = 0; a < 3; ++a) {
          if (std::abs(mesh.dof_point[g](a) - lo(a)) > ptol) continue;
          const int h = dix.find(mesh.dof_point[g] + L(a) * Vec3::Unit(a), ptol);
          adj[g].push_back({h, a, 1});
          adj[h].push_back({g, a, -1});
        }
      }
      std::vector<char> seen(mesh.ndof, 0);
      for (int r : reps) {
        std::vector<int> queue{r};
        seen[r] = 1;
        for (size_t q = 0; q < queue.size(); ++q) {
          const int g = queue[q];
          for (auto [h, a, dir] : adj[g]) {
            if (seen[h]) continue;
            seen[h] = 1;
            jump[h] = jump[g];
            if (weld[a].squaredNorm() > 0) {
              const int plus_dof = dir > 0 ? h : g;  // the + face end of this translation
              if (tplus[3 * plus_dof + a].empty()) throw std::runtime_error("PBC weld: + face dof without traction");
              for (auto [col, wgt] : tplus[3 * plus_dof + a]) jump[h].push_back({col, -double(dir) * wgt * weld[a]});
            }
            queue.push_back(h);
          }
        }
      }
    }
    const int nr = int(reps.size()), nt = int(plus.size()), nu = 3 * nr, n = nu + 3 * nt;
    std::vector<int> rows = comps(reps);
    for (int r : comps(mrows)) rows.push_back(r);
    if (int(rows.size()) != n) throw std::logic_error("PBC: rows != unknowns");
    const Eigen::MatrixXd Hr = H_(rows, Eigen::all), Gr = G_(rows, Eigen::all);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, n), rhs = Eigen::MatrixXd::Zero(n, 6);
    std::vector<Vec3> shift(mesh.ndof);
    for (int g = 0; g < mesh.ndof; ++g) {
      shift[g] = mesh.dof_point[g] - mesh.dof_point[rep_of[g]];
      M.middleCols<3>(3 * col_of[rep_of[g]]) += Hr.middleCols<3>(3 * g);
      if (shift[g].squaredNorm() > 0)
        for (int k = 0; k < 6; ++k) rhs.col(k) -= Hr.middleCols<3>(3 * g) * (voigtUnit(k, true) * shift[g]);
    }
    for (int j = 0; j < nt; ++j)
      M.middleCols<3>(nu + 3 * j) = -(Gr.middleCols<3>(3 * plus[j]) - Gr.middleCols<3>(3 * minus_of[j]));
    for (int g = 0; g < mesh.ndof; ++g)
      for (const auto& [col, C] : jump[g]) M.middleCols<3>(nu + 3 * col) += Hr.middleCols<3>(3 * g) * C;
    System S;
    S.modes = rigid(reps, 3);
    S.A = bordered(M, S.modes, nu);
    S.rhs = rhs;
    S.expand = [=](const Eigen::VectorXd& x, int k, Eigen::VectorXd& u, Eigen::VectorXd& t) {
      const Eigen::Matrix3d Ek = voigtUnit(k, true);
      for (int g = 0; g < int(rep_of.size()); ++g)
      {
        u.segment<3>(3 * g) = x.segment<3>(3 * col_of[rep_of[g]]) + Ek * shift[g];
        for (const auto& [col, C] : jump[g]) u.segment<3>(3 * g) += C * x.segment<3>(nu + 3 * col);
      }
      for (int j = 0; j < nt; ++j) {
        t.segment<3>(3 * plus[j]) = x.segment<3>(nu + 3 * j);
        t.segment<3>(3 * minus_of[j]) = -x.segment<3>(nu + 3 * j);
      }
    };
    return S;
  }
};

}  // namespace igabem
