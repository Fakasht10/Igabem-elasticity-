// Compressed SBIE operators H and G (H u = G t) at a set of collocation points,
// the hierarchical counterpart of OperatorRows (operators.hpp): built once per
// geometry, quadrature, material and compression setting, cached on disk, and
// applied to any number of vectors.
//
//   H = Hraw - D,  Hraw(a, B) = int T(x_a, y) N_B(y),  D(a, B) = N_B(x_a) Tsum(a),
//
// with Tsum(a) = int_Gamma T(x_a, y) = sum_B Hraw(a, B) (partition of unity).
// Tsum is taken from the compressed Hraw (three products with the translations),
// so H annihilates rigid translations exactly, as the dense operator does; D is
// block sparse (the basis functions nonzero at each collocation point).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../cache.hpp"
#include "hmatrix_operator.hpp"
#include "sbie_entries.hpp"

namespace igabem {

class CompressedSbie {
 public:
  HOperator Hraw, G;
  Eigen::SparseMatrix<double, Eigen::RowMajor> D;
  HOptions hopt;
  bool from_cache = false;
  double t_assemble = 0.0, t_load = 0.0;  // seconds; t_assemble kept in the cache file

  int rows() const { return Hraw.rows(); }

  Eigen::VectorXd applyH(const Eigen::VectorXd& u) const {
    Eigen::VectorXd y = Hraw * u;
    y -= D * u;
    return y;
  }
  Eigen::MatrixXd applyH(const Eigen::MatrixXd& U) const { return Hraw * U - D * U; }
  Eigen::VectorXd applyG(const Eigen::VectorXd& t) const { return G * t; }
  Eigen::MatrixXd applyG(const Eigen::MatrixXd& T) const { return G * T; }

  double bytes() const {
    return Hraw.stats().bytes() + G.stats().bytes() + 12.0 * double(D.nonZeros());
  }
  double denseBytes() const { return Hraw.stats().denseBytes() + G.stats().denseBytes(); }

  // kind names the row set in the cache key, as for cachedOperatorRows; cache_dir
  // empty: no disk cache.
  void setup(const Mesh& mesh, const TractionSpace& ts, const Material& mat, const SolverOptions& opt,
             const PartitionOptions& po, const HOptions& ho, const std::string& geometry_file,
             const std::string& cache_dir, const std::string& kind, const std::vector<Collocation>& rows) {
    hopt = ho;
    const auto t0 = std::chrono::steady_clock::now();
    SbieEntries eh(mesh, ts, mat, opt, rows, SbieEntries::H), eg(mesh, ts, mat, opt, rows, SbieEntries::G);
    std::vector<double> rxyz, hxyz, hrad, gxyz, grad;
    eh.rowGeometry(rxyz);
    eh.columnGeometry(hxyz, hrad);
    eg.columnGeometry(gxyz, grad);
    std::string path;
    from_cache = false;
    if (!cache_dir.empty()) {
      std::filesystem::create_directories(cache_dir);
      path = cache_dir + "/" + operatorKey(geometry_file, kind + "_" + ho.key(), mat, opt, po) + ".hmat";
      std::ifstream f(path, std::ios::binary);
      char magic[8];
      if (f && f.read(magic, 8) && std::string(magic, 8) == "IGBEMHM1" &&
          f.read(reinterpret_cast<char*>(&t_assemble), sizeof t_assemble))
        from_cache = Hraw.load(f, rxyz, hxyz, hrad, ho) && G.load(f, rxyz, gxyz, grad, ho);
    }
    if (!from_cache) {
      Hraw.build(eh, rxyz, hxyz, hrad, ho);
      G.build(eg, rxyz, gxyz, grad, ho);
      t_assemble = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (!path.empty()) {
        std::ofstream f(path, std::ios::binary);
        f.write("IGBEMHM1", 8);
        f.write(reinterpret_cast<const char*>(&t_assemble), sizeof t_assemble);
        Hraw.save(f);
        G.save(f);
      }
    }
    rigidDiagonal(mesh, rows);
    if (from_cache) t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

 private:
  void rigidDiagonal(const Mesh& mesh, const std::vector<Collocation>& rows) {
    Eigen::MatrixXd Tr = Eigen::MatrixXd::Zero(3 * mesh.ndof, 3);
    for (int g = 0; g < mesh.ndof; ++g) Tr.block<3, 3>(3 * g, 0).setIdentity();
    const Eigen::MatrixXd S = Hraw * Tr;  // rows 3a..3a+2: Tsum(a)
    std::vector<Eigen::Triplet<double>> trip;
    for (size_t a = 0; a < rows.size(); ++a)
      for (const auto& [B, NB] : rows[a].basis)
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) trip.emplace_back(int(3 * a + i), 3 * B + j, NB * S(3 * a + i, j));
    D.resize(3 * int(rows.size()), 3 * mesh.ndof);
    D.setFromTriplets(trip.begin(), trip.end());
  }
};

}  // namespace igabem
