// SBIE operator rows H, G (H u = G t) at a set of collocation points, loaded
// from the on-disk cache when present, otherwise assembled and saved. The
// assembly time is stored with the matrices, so it is reported also on a cache
// hit.
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "cache.hpp"
#include "sbie.hpp"
#include "traction_space.hpp"

namespace igabem {

struct OperatorRows {
  Eigen::MatrixXd H, G;
  bool from_cache = false;
  double t_assemble = 0.0, t_load = 0.0;  // seconds
};

// kind names the row set in the cache key (e.g. "sbie_hg" for the SBIE
// collocation points); cache_dir empty: no disk cache.
inline OperatorRows cachedOperatorRows(const Mesh& mesh, const TractionSpace& ts, const Material& mat,
                                       const SolverOptions& opt, const PartitionOptions& po,
                                       const std::string& geometry_file, const std::string& cache_dir,
                                       const std::string& kind, const std::vector<Collocation>& rows) {
  OperatorRows op;
  const auto t0 = std::chrono::steady_clock::now();
  std::string path;
  Eigen::MatrixXd T(1, 1);
  if (!cache_dir.empty()) {
    std::filesystem::create_directories(cache_dir);
    path = cache_dir + "/" + operatorKey(geometry_file, kind, mat, opt, po) + ".bin";
    op.from_cache = loadMatrices(path, {&op.H, &op.G, &T}) && op.H.rows() == 3 * int(rows.size()) &&
                    op.H.cols() == 3 * mesh.ndof && op.G.cols() == 3 * ts.ndof && T.size() == 1;
  }
  if (!op.from_cache) {
    SbieAssembler(mesh, mat, opt).assembleHG(rows, ts, op.H, op.G);
    T(0, 0) = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!path.empty()) saveMatrices(path, {&op.H, &op.G, &T});
  }
  op.t_assemble = T(0, 0);
  if (op.from_cache)
    op.t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return op;
}

}  // namespace igabem
