// Hierarchical compression of the SBIE operators (src/fast) on a small plate:
//   1. the entry callback reproduces the dense H and G (round-off),
//   2. compressed H and G match the dense ones to the ACA tolerance,
//   3. compressed H annihilates rigid translations (round-off),
//   4. a cache hit gives the same operator as a fresh build.
//
// usage: test_hmatrix <repository root>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <string>

#include "fast/compressed_sbie.hpp"
#include "operators.hpp"
#include "problems/plate_hole.hpp"

using namespace igabem;

static int failures = 0;

static void check(const char* name, double got, double tol) {
  const bool ok = got <= tol;
  std::printf("%-52s %s  %.3e  (<= %.1e)\n", name, ok ? "ok  " : "FAIL", got, tol);
  if (!ok) ++failures;
}

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const std::string geo = root + "/geo/plate_hole_p2_t0.2_r2.dat";
  const PlateHole ph;
  SolverOptions opt;
  opt.n_gauss = 8;
  opt.far_tol = 1e-10;
  const PartitionOptions po;
  Mesh mesh;
  mesh.build(loadDat(geo), po);
  TractionSpace ts;
  ts.build(mesh);
  Eigen::MatrixXd Hd, Gd;
  SbieAssembler(mesh, ph.mat, opt).assembleHG(ts, Hd, Gd);
  std::printf("# %s: %d unknowns, %d traction unknowns\n", geo.c_str(), 3 * mesh.ndof, 3 * ts.ndof);

  // 1. Entries: full block of the callback, rigid-body term from its row sums.
  {
    SbieEntries eh(mesh, ts, ph.mat, opt, mesh.colloc, SbieEntries::H);
    SbieEntries eg(mesh, ts, ph.mat, opt, mesh.colloc, SbieEntries::G);
    std::vector<int> r(eh.rows()), ch(eh.cols()), cg(eg.cols());
    std::iota(r.begin(), r.end(), 0);
    std::iota(ch.begin(), ch.end(), 0);
    std::iota(cg.begin(), cg.end(), 0);
    Eigen::MatrixXd He(eh.rows(), eh.cols()), Ge(eg.rows(), eg.cols());
    eh.block(eh.rows(), eh.cols(), r.data(), ch.data(), He.data());
    eg.block(eg.rows(), eg.cols(), r.data(), cg.data(), Ge.data());
    const Eigen::MatrixXd Hraw = He;
    for (int a = 0; a < mesh.ndof; ++a) {
      Eigen::Matrix3d Ts = Eigen::Matrix3d::Zero();
      for (int B = 0; B < mesh.ndof; ++B) Ts += Hraw.block<3, 3>(3 * a, 3 * B);
      for (const auto& [B, NB] : mesh.colloc[a].basis) He.block<3, 3>(3 * a, 3 * B) -= NB * Ts;
    }
    check("entries: |H_entries - H_dense| / |H_dense|", (He - Hd).norm() / Hd.norm(), 1e-13);
    check("entries: |G_entries - G_dense| / |G_dense|", (Ge - Gd).norm() / Gd.norm(), 1e-13);
    // A scattered sub-block (rows and columns out of order, points split) = the full block's entries.
    const std::vector<int> rr = {7, 3, 400, 401, 5, 1000}, cc = {2, 999, 12, 13, 14, 600, 0};
    Eigen::MatrixXd S(rr.size(), cc.size());
    eh.block(int(rr.size()), int(cc.size()), rr.data(), cc.data(), S.data());
    double e = 0.0;
    for (size_t i = 0; i < rr.size(); ++i)
      for (size_t j = 0; j < cc.size(); ++j) e = std::max(e, std::abs(S(i, j) - Hraw(rr[i], cc[j])));
    check("entries: scattered sub-block, max abs error", e / Hd.cwiseAbs().maxCoeff(), 1e-13);
  }

  // 2.-4. Compressed operators.
  const std::string cache = (std::filesystem::temp_directory_path() / "igabem_test_hmatrix").string();
  std::filesystem::remove_all(cache);
  for (double eps : {1e-4, 1e-8}) {
    HOptions ho;
    ho.eps = eps;
    ho.leaf = 24;  // small clusters, so that this small mesh has low-rank blocks
    CompressedSbie C;
    C.setup(mesh, ts, ph.mat, opt, po, ho, geo, cache, "sbie_hg", mesh.colloc);
    const Eigen::MatrixXd Hh = C.Hraw.dense() - Eigen::MatrixXd(C.D), Gh = C.G.dense();
    char name[96];
    std::snprintf(name, sizeof name, "eps %.0e: |H_h - H| / |H| / eps", eps);
    check(name, (Hh - Hd).norm() / Hd.norm() / eps, 10.0);
    std::snprintf(name, sizeof name, "eps %.0e: |G_h - G| / |G| / eps", eps);
    check(name, (Gh - Gd).norm() / Gd.norm() / eps, 10.0);
    Eigen::MatrixXd Tr = Eigen::MatrixXd::Zero(3 * mesh.ndof, 3);
    for (int g = 0; g < mesh.ndof; ++g) Tr.block<3, 3>(3 * g, 0).setIdentity();
    std::snprintf(name, sizeof name, "eps %.0e: |H_h translations| / |H_h|", eps);
    check(name, C.applyH(Tr).norm() / Hh.norm(), 1e-13);
    const HStats sh = C.Hraw.stats(), sg = C.G.stats();
    std::printf("#   H: %ld dense + %ld low-rank leaves, max rank %ld, %.1f%% of dense; G: %.1f%%\n",
                sh.dense_leaves, sh.lowrank_leaves, sh.max_rank, 100 * sh.ratio(), 100 * sg.ratio());
    std::snprintf(name, sizeof name, "eps %.0e: compressed below dense (H, G)", eps);
    check(name, std::max(sh.ratio(), sg.ratio()), 1.0);
    // Cache hit: identical products.
    CompressedSbie L;
    L.setup(mesh, ts, ph.mat, opt, po, ho, geo, cache, "sbie_hg", mesh.colloc);
    const Eigen::VectorXd x = Eigen::VectorXd::Random(3 * mesh.ndof), y = Eigen::VectorXd::Random(3 * ts.ndof);
    std::snprintf(name, sizeof name, "eps %.0e: cache hit = fresh (H x, G y), relative", eps);
    const Eigen::VectorXd hx = C.applyH(x), gy = C.applyG(y);
    check(name, L.from_cache ? std::max((L.applyH(x) - hx).norm() / hx.norm(), (L.applyG(y) - gy).norm() / gy.norm())
                             : 1.0, 1e-14);  // parallel summation order only
  }
  std::filesystem::remove_all(cache);
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "passed", failures);
  return failures ? 1 : 0;
}
