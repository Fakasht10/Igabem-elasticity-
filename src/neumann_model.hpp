// Traction (Neumann) problem on a closed multipatch body, as a reusable object:
// mesh, traction space, SBIE operators H and G (assembled once and cached on
// disk), and the bordered LU that fixes the six rigid modes (factored once).
// Any number of load cases are then a traction interpolation, a product with G
// and a triangular solve each.
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <string>
#include <vector>

#include "operators.hpp"
#include "sbie.hpp"
#include "solution.hpp"
#include "traction_space.hpp"

namespace igabem {

class NeumannModel {
 public:
  Mesh mesh;
  TractionSpace ts;
  Material mat;
  SolverOptions opt;
  Eigen::MatrixXd H, G, R;
  bool from_cache = false;
  // Seconds. t_assemble is the cost of assembling H and G, kept in the cache file,
  // so it is reported also when the operators were loaded (t_load).
  double t_assemble = 0.0, t_load = 0.0, t_factor = 0.0;

  // cache_dir empty: no disk cache.
  void setup(const std::string& geometry_file, const Material& m, const SolverOptions& o,
             const std::string& cache_dir = "cache", const PartitionOptions& po = PartitionOptions{}) {
    mat = m;
    opt = o;
    mesh.build(loadDat(geometry_file), po);
    ts.build(mesh);
    R = rigidModes(mesh);
    OperatorRows op = cachedOperatorRows(mesh, ts, mat, opt, po, geometry_file, cache_dir, "sbie_hg", mesh.colloc);
    H = std::move(op.H);
    G = std::move(op.G);
    from_cache = op.from_cache;
    t_assemble = op.t_assemble;
    t_load = op.t_load;
    const auto t1 = std::chrono::steady_clock::now();
    factor();
    t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  }

  int unknowns() const { return 3 * mesh.ndof; }

  // Displacement coefficients for traction data t(y, n) returning a 3 x K block
  // (K load cases), interpolated in the traction space.
  template <class F>
  Eigen::MatrixXd solve(F&& traction, int K) const {
    return solveRhs(G * ts.interpolate(traction, K));
  }

  // Full boundary solutions (displacement and interpolated traction), one per case.
  template <class F>
  std::vector<BoundarySolution> solutions(F&& traction, int K) const {
    const Eigen::MatrixXd TC = ts.interpolate(traction, K);
    const Eigen::MatrixXd UH = solveRhs(G * TC);
    std::vector<BoundarySolution> out(K);
    for (int k = 0; k < K; ++k) out[k] = {&mesh, &ts, mat, UH.col(k), TC.col(k)};
    return out;
  }

  // Displacement coefficients for given right-hand sides B (3M x K).
  Eigen::MatrixXd solveRhs(const Eigen::MatrixXd& B) const {
    const int N = unknowns();
    Eigen::MatrixXd bb = Eigen::MatrixXd::Zero(N + 6, B.cols());
    bb.topRows(N) = B;
    return lu_.solve(bb).topRows(N);
  }

  // ||H R_rot|| / ||H||: rotations carry no traction, so this checks the kernel.
  double rotationResidual() const { return (H * R.rightCols(3)).norm() / H.norm(); }

 private:
  Eigen::PartialPivLU<Eigen::MatrixXd> lu_;

  // Bordered system [H s*R; s*R^T 0] fixes the six rigid-body modes.
  void factor() {
    const int N = unknowns();
    const double s = H.cwiseAbs().maxCoeff();
    Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
    Ab.topLeftCorner(N, N) = H;
    Ab.topRightCorner(N, 6) = s * R;
    Ab.bottomLeftCorner(6, N) = s * R.transpose();
    lu_.compute(Ab);
  }
};

}  // namespace igabem
