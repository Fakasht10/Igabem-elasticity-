// Mixed problem: displacement prescribed on chosen patches (Gamma_u), traction
// on the rest (Gamma_t), same SBIE H u = G t.
//
// Unknowns: the displacement dofs not touched by any Dirichlet patch, and the
// traction dofs of the Dirichlet patches (the traction space is discontinuous
// across patches, so a traction jump at the edge of Gamma_u is allowed).
// Equations: the SBIE at the collocation points of the free displacement dofs
// (the rows of NeumannModel's H and G, same cache entry) plus the SBIE at the
// traction collocation points of each Dirichlet patch (interior to the patch,
// cached per patch). The system is square, and the Dirichlet data fix the
// rigid modes, so no bordering. Changing which patches are clamped reuses every
// cached row; only the column selection and the LU are redone.
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "operators.hpp"
#include "sbie.hpp"
#include "solution.hpp"
#include "traction_space.hpp"

namespace igabem {

class MixedModel {
 public:
  Mesh mesh;
  TractionSpace ts;
  Material mat;
  SolverOptions opt;
  std::vector<bool> dirichlet;           // per patch
  std::vector<int> u_free, u_fixed;      // global displacement dofs
  std::vector<int> t_free, t_fixed;      // traction dofs (unknown on Dirichlet patches)
  bool from_cache = false;               // all rows came from the cache
  double t_assemble = 0.0, t_load = 0.0, t_factor = 0.0;

  // use_dirichlet(patch index, patch centre, unit outward normal at the centre).
  template <class Pred>
  void setup(const std::string& geometry_file, const Material& m, const SolverOptions& o, Pred&& use_dirichlet,
             const std::string& cache_dir = "cache", const PartitionOptions& po = PartitionOptions{}) {
    mat = m;
    opt = o;
    mesh.build(loadDat(geometry_file), po);
    ts.build(mesh);
    dirichlet.assign(mesh.patches.size(), false);
    for (size_t p = 0; p < mesh.patches.size(); ++p) {
      const Patch& P = mesh.patches[p];
      PatchEval E;
      P.eval(0.5 * (P.U.front() + P.U.back()), 0.5 * (P.V.front() + P.V.back()), E);
      dirichlet[p] = use_dirichlet(int(p), E.x, Vec3((E.normal() * mesh.normal_sign).normalized()));
    }
    std::vector<bool> fixed(mesh.ndof, false);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (dirichlet[p]) for (int g : mesh.l2g[p]) fixed[g] = true;
    u_free.clear();
    u_fixed.clear();
    for (int g = 0; g < mesh.ndof; ++g) (fixed[g] ? u_fixed : u_free).push_back(g);
    t_free.clear();
    t_fixed.clear();
    for (size_t p = 0; p < mesh.patches.size(); ++p) {
      const int n1 = p + 1 < mesh.patches.size() ? ts.offset[p + 1] : ts.ndof;
      for (int j = ts.offset[p]; j < n1; ++j) (dirichlet[p] ? t_free : t_fixed).push_back(j);
    }
    if (u_fixed.empty()) throw std::runtime_error("MixedModel: no Dirichlet patch selected");

    // Rows: SBIE at the free displacement dofs, then at the traction points of Dirichlet patches.
    OperatorRows main =
        cachedOperatorRows(mesh, ts, mat, opt, po, geometry_file, cache_dir, "sbie_hg", mesh.colloc);
    from_cache = main.from_cache;
    t_assemble = main.t_assemble;
    t_load = main.t_load;
    const int nr = 3 * int(u_free.size() + t_free.size());
    H_.resize(nr, 3 * mesh.ndof);
    G_.resize(nr, 3 * ts.ndof);
    // One gather per matrix: column-major traversal (copying 3 rows at a time strides
    // across every column and dominated setup on 10k-unknown models).
    std::vector<int> rows;
    rows.reserve(3 * u_free.size());
    for (int g : u_free)
      for (int c = 0; c < 3; ++c) rows.push_back(3 * g + c);
    int r = int(rows.size());
    H_.topRows(r) = main.H(rows, Eigen::placeholders::all);
    G_.topRows(r) = main.G(rows, Eigen::placeholders::all);
    for (size_t p = 0; p < mesh.patches.size(); ++p) {
      if (!dirichlet[p]) continue;
      OperatorRows op = cachedOperatorRows(mesh, ts, mat, opt, po, geometry_file, cache_dir,
                                           "sbie_hg_tpatch" + std::to_string(p),
                                           tractionCollocation(mesh, ts, int(p)));
      from_cache = from_cache && op.from_cache;
      t_assemble += op.t_assemble;
      t_load += op.t_load;
      H_.middleRows(r, op.H.rows()) = op.H;
      G_.middleRows(r, op.G.rows()) = op.G;
      r += int(op.H.rows());
    }
    const auto t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXd A(nr, nr);
    for (size_t j = 0; j < u_free.size(); ++j) A.middleCols(3 * j, 3) = H_.middleCols(3 * u_free[j], 3);
    const int c0 = 3 * int(u_free.size());
    for (size_t j = 0; j < t_free.size(); ++j) A.middleCols(c0 + 3 * j, 3) = -G_.middleCols(3 * t_free[j], 3);
    lu_.compute(A);
    t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  }

  int unknowns() const { return 3 * int(u_free.size() + t_free.size()); }

  // traction(y, n) and displacement(x) each return a 3 x K TractionBlock (K load
  // cases); traction is used on Gamma_t, displacement on Gamma_u.
  template <class Ft, class Fu>
  std::vector<BoundarySolution> solutions(Ft&& traction, Fu&& displacement, int K) const {
    Eigen::MatrixXd TC = ts.interpolate(traction, K);  // Dirichlet patches overwritten below
    Eigen::MatrixXd UC = Eigen::MatrixXd::Zero(3 * mesh.ndof, K);
    for (size_t p = 0; p < mesh.patches.size(); ++p)
      if (dirichlet[p]) interpolatePatch(int(p), displacement, K, UC);
    // A x = G t_known - H u_known
    Eigen::MatrixXd uk = Eigen::MatrixXd::Zero(3 * mesh.ndof, K), tk = TC;
    for (int g : u_fixed) uk.middleRows(3 * g, 3) = UC.middleRows(3 * g, 3);
    for (int j : t_free) tk.middleRows(3 * j, 3).setZero();
    const Eigen::MatrixXd X = lu_.solve(G_ * tk - H_ * uk);
    for (size_t j = 0; j < u_free.size(); ++j) UC.middleRows(3 * u_free[j], 3) = X.middleRows(3 * j, 3);
    const int c0 = 3 * int(u_free.size());
    for (size_t j = 0; j < t_free.size(); ++j) TC.middleRows(3 * t_free[j], 3) = X.middleRows(c0 + 3 * j, 3);
    std::vector<BoundarySolution> out(K);
    for (int k = 0; k < K; ++k) out[k] = {&mesh, &ts, mat, UC.col(k), TC.col(k)};
    return out;
  }

 private:
  Eigen::MatrixXd H_, G_;  // selected rows, all columns
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_;

  // Greville interpolation of the displacement data on one patch, written to the
  // patch's global dofs (edge coefficients depend only on edge data, so dofs
  // shared by two Dirichlet patches get the same value from either).
  template <class Fu>
  void interpolatePatch(int p, Fu& displacement, int K, Eigen::MatrixXd& UC) const {
    const Patch& P = mesh.patches[p];
    const int n = P.nu * P.nv;
    Eigen::MatrixXd Np = Eigen::MatrixXd::Zero(n, n), rhs(n, 3 * K);
    PatchEval E;
    for (int k = 0; k < n; ++k) {
      P.eval(P.greville(k % P.nu, true), P.greville(k / P.nu, false), E);
      for (int l = 0; l < E.n_loc; ++l) Np(k, E.idx[l]) += E.R[l];
      const auto u = displacement(E.x);
      for (int c = 0; c < K; ++c) rhs.block(k, 3 * c, 1, 3) = u.col(c).transpose();
    }
    const Eigen::MatrixXd c = Np.partialPivLu().solve(rhs);
    for (int k = 0; k < n; ++k)
      for (int q = 0; q < K; ++q) UC.block(3 * mesh.l2g[p][k], q, 3, 1) = c.block(k, 3 * q, 1, 3).transpose();
  }
};

}  // namespace igabem
