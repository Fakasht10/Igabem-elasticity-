// Hierarchical operator (src/hmat) on the traction-only problem with an exact
// solution: the plate with a hole under the 3D Kirsch field (default) or the
// cube with a spherical cavity under the Goodier field (--problem cavity).
// Reports the build (tree, near field, far field, rigid term), memory,
// preconditioner, GMRES and the error against the exact field; --check also
// assembles the dense operator (DenseSbie, same entry rule) and compares H x and
// G c (small meshes only).
//
// usage: hmat_bench <geometry.dat> [--problem plate|cavity] [--eps 1e-6] [--eta 2] [--leaf 96] [--eleaf 64]
//                   [--tol 1e-7] [--gtol 1e-8] [--radius 1.0] [--coarse 256] [--check] [--max-error e]
//                   [--csv file label]
// --check / --max-error make the exit status a test (ctest hmat_kirsch).

#include <malloc.h>
#include <sys/resource.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "fast/dense_sbie.hpp"
#include "hmat/h_neumann.hpp"
#include "problems/cube_cavity.hpp"
#include "problems/plate_hole.hpp"
#include "sbie.hpp"

using namespace igabem;

static double peakRssMB() {
  rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss / 1024.0;
}
static double curRssGB() {
  long pages = 0, rss = 0;
  if (FILE* f = std::fopen("/proc/self/statm", "r")) {
    if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
    std::fclose(f);
  }
  return rss * 4096.0 / 1e9;
}
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: hmat_bench <geometry.dat> [--eps 1e-6] [--eta 1.5] [--leaf 48] [--eleaf 32] [--check]\n");
    return 1;
  }
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  HNeumannOptions ho;
  bool check = false, cavity = false;
  double max_error = 0.0;  // > 0: exit 1 if the Kirsch / Goodier error exceeds it (ctest)
  int status = 0;
  const char *csv = nullptr, *label = "";
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--eps")) ho.h.eps = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--eta")) ho.h.eta = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--leaf")) ho.h.leaf_points = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--eleaf")) ho.h.leaf_elems = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--tol")) ho.h.tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--gtol")) ho.gtol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--radius")) ho.radius = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--check")) check = true;
    else if (!std::strcmp(argv[i], "--max-error")) max_error = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-rsvd")) ho.h.rsvd = false;
    else if (!std::strcmp(argv[i], "--coarse")) ho.coarse = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--s0")) ho.h.s0 = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--adm-min")) ho.h.adm_min = true;
    else if (!std::strcmp(argv[i], "--s0e")) ho.h.s0e = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--problem")) cavity = !std::strcmp(argv[++i], "cavity");
    else if (!std::strcmp(argv[i], "--csv")) { csv = argv[++i]; label = argv[++i]; }
  }
  const PlateHole ph;
  const CubeCavity cc;
  const Material mat = cavity ? cc.mat : ph.mat;
  auto exact_u = [&](const Vec3& x) { return cavity ? cc.displacement(CubeCavity::Goodier, x) : ph.displacement(PlateHole::Kirsch, x); };
  auto exact_t = [&](const Vec3& y, const Vec3& n) {
    return cavity ? Vec3(cc.traction(CubeCavity::Goodier, y, n)) : Vec3(ph.traction(PlateHole::Kirsch, y, n));
  };
  const double t_start = now();
  Mesh mesh;
  mesh.build(loadDat(argv[1]), PartitionOptions());
  TractionSpace ts;
  ts.build(mesh);
  const int N = 3 * mesh.ndof;
  const Eigen::MatrixXd tc = ts.interpolate([&](const Vec3& y, const Vec3& n) { return TractionBlock(exact_t(y, n)); }, 1);
  const double t_mesh = now() - t_start;
  std::printf("# %d unknowns, %zu elements; eps %g eta %g leaves %d/%d tol %g\n", N, mesh.elements.size(), ho.h.eps,
              ho.h.eta, ho.h.leaf_points, ho.h.leaf_elems, ho.h.tol);
  std::fflush(stdout);

  std::printf("# peak RSS %.2f GB before the operator\n", peakRssMB() / 1024);
  HNeumann hn(mesh, ts, mat, opt, tc, ho);
  const auto& st = hn.op().stats();
  std::printf("# build %.2f s (tree %.2f, near %.2f [%d leaves, %ld pairs], far %.2f [%d blocks, rank mean %.1f max %d, "
              "%d dense], rigid %.2f); storage near %.3f GB far %.3f GB (%.2f%% of dense float)\n",
              hn.times.build, st.t_tree, st.t_near, st.near_leaves, st.near_pairs, st.t_far, st.far_blocks, st.rank_mean,
              st.max_rank, st.dense_fallbacks, st.t_rigid, st.bytes_near / 1e9, st.bytes_far / 1e9,
              100.0 * (st.bytes_near + st.bytes_far) / (4.0 * double(N) * N));
  std::printf("# peak RSS %.2f GB before the near field, %.2f after\n", st.rss_quad, st.rss_near);
  std::printf("# far profile (cpu s): aca %.1f (rows %.1f [%ld], element columns %.1f [%ld]), recompression %.1f\n",
              st.cpu_aca, st.cpu_row, st.n_rows, st.cpu_col, st.n_cols, st.cpu_rc);
  std::printf("# cur (cpu s): C/W %.1f, skeleton %.1f, R %.1f, tests %.1f; %ld attempts; %d blocks failed (%.1fM entries)\n", st.cpu_cw, st.cpu_qr,
              st.cpu_rr, st.cpu_test, st.attempts, st.cur_failed, st.cur_failed_entries / 1e6);
  std::printf("# preconditioner %.2f s (pattern %.2f s; %ld nonzeros, %.1f per row); coarse %d dof %.2f s; peak RSS %.2f GB after build (quadrature cache %.2f GB), %.2f after LU\n", hn.times.precond,
              hn.times.t_pattern, hn.times.precond_nnz, double(hn.times.precond_nnz) / N, hn.times.coarse_dim, hn.times.coarse,
              hn.times.rss_build, hn.times.quad_GB, hn.times.rss_precond);
  std::fflush(stdout);
  const double rss0 = curRssGB();
  malloc_trim(0);
  std::printf("# RSS now %.2f GB (operator + preconditioner), %.2f after malloc_trim\n", rss0, curRssGB());
  std::vector<int> its;
  std::vector<double> res;
  const Eigen::MatrixXd U = hn.solve(&its, &res);
  double t0 = now();
  const double err = displacementL2Error(mesh, U.col(0), exact_u);
  const double t_post = now() - t0;
  const double total = now() - t_start;
  std::printf("# solve %.2f s (%d iterations, residual %.1e); error %.2f s; total %.2f s; peak RSS %.0f MB; %s rel L2 %.4e\n",
              hn.times.solve, its[0], res[0], t_post, total, peakRssMB(), cavity ? "Goodier" : "Kirsch", err);

  if (max_error > 0.0 && !(err <= max_error)) {
    std::printf("# FAIL: error %.3e > %.3e\n", err, max_error);
    status = 1;
  }
  if (check) {
    DenseSbie ds(mesh, ts, mat, opt, ho.h.tol, ho.h.min_order);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N), GC, S = Eigen::MatrixXd::Zero(N, 3);
    DenseSbie::Out o;
    o.H = H.data();
    o.ldh = N;
    o.C = &tc;
    o.GC = &GC;
    o.S = S.data();
    ds.assemble(o);
    for (int a = 0; a < int(mesh.colloc.size()); ++a)
      for (const auto& [B, NB] : mesh.colloc[a].basis) H.block<3, 3>(3 * a, 3 * B) -= NB * S.block<3, 3>(3 * a, 0);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    Eigen::VectorXd x(N), y(N);
    for (int i = 0; i < N; ++i) x(i) = nd(rng);
    hn.op().apply(x.data(), y.data());
    const Eigen::VectorXd yd = H * x;
    Eigen::VectorXd u0(N), yu(N);
    u0 = U.col(0);
    hn.op().apply(u0.data(), yu.data());
    std::printf("# check: |Hh x - H x| / |H x| = %.3e (random x), %.3e (x = solution); |Gc_h - Gc| / |Gc| = %.3e\n",
                (y - yd).norm() / yd.norm(), (yu - H * u0).norm() / (H * u0).norm(), (hn.op().gc() - GC).norm() / GC.norm());
    if ((y - yd).norm() > 10.0 * ho.h.eps * yd.norm() || (hn.op().gc() - GC).norm() > 10.0 * ho.h.eps * GC.norm()) {
      std::printf("# FAIL: operator error above 10 eps\n");
      status = 1;
    }
  }
  if (csv) {
    FILE* f = std::fopen(csv, "a");
    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) == 0)
      std::fprintf(f, "label,unknowns,elements,eps,eta,leaf,eleaf,t_mesh,t_tree,t_near,t_far,t_rigid,t_build,t_precond,"
                      "t_solve,iterations,total,near_GB,far_GB,rank_mean,rank_max,rss_MB,err,t_coarse,coarse_dim,radius\n");
    std::fprintf(f, "%s,%d,%zu,%g,%g,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.4f,%.4f,%.1f,%d,%.0f,%.4e,%.2f,%d,%g\n",
                 label, N, mesh.elements.size(), ho.h.eps, ho.h.eta, ho.h.leaf_points, ho.h.leaf_elems, t_mesh, st.t_tree,
                 st.t_near, st.t_far, st.t_rigid, hn.times.build, hn.times.precond, hn.times.solve, its[0], total,
                 st.bytes_near / 1e9, st.bytes_far / 1e9, st.rank_mean, st.max_rank, peakRssMB(), err, hn.times.coarse,
                 hn.times.coarse_dim, ho.radius);
    std::fclose(f);
  }
  return status;
}
