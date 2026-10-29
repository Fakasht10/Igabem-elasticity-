// Dense SBIE operators and the traction-only (Neumann) problem with an exact
// solution, with a profile of every stage: the plate with a hole under the 3D
// Kirsch field (default) or the cube with a spherical cavity under the Goodier
// field (--problem cavity).
//
// Assembly (--assembly):
//   batched  DenseSbie: element-batched far field (one kernel call for U and T,
//            GEMM contraction), G c applied on the fly without forming G (default)
//   legacy   SbieEntries row by row, G formed and applied, then freed
// Both use the same quadrature rule for every (point, element) pair (entry rule
// tol / order for separated pairs, the exact quadrature for close ones), so the
// entries agree to round-off; --check compares them. --order -1 (default) takes
// the far-field minimum Gauss order from the basis degree.
// Solve:
//   --solver gmres  DenseGmres to --gtol (default 1e-10), block Jacobi on --blocks
//                   (default 4) bisection blocks (default)
//   --lu s          single-precision LU (sgetrf) + iterative refinement in double to 1e-14
//   --lu d          double-precision LU (dgetrf)
//
// usage: dense_bench <geometry.dat> [--problem plate|cavity] [--tol 1e-7] [--order -1] [--n 12] [--far-tol 1e-10]
//                    [--assembly batched|legacy] [--solver gmres] [--gtol 1e-10] [--blocks 4] [--lu s|d] [--chunk 96]
//                    [--check] [--save-u file] [--ref-u file] [--csv file label]

#include <sys/resource.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "fast/dense_gmres.hpp"
#include "fast/dense_sbie.hpp"
#include "fast/huge_buffer.hpp"
#include "fast/sbie_entries.hpp"
#include "problems/cube_cavity.hpp"
#include "problems/plate_hole.hpp"
#include "sbie.hpp"

extern "C" void sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info);
extern "C" void sgetrs_(const char* t, const int* n, const int* nrhs, const float* a, const int* lda, const int* ipiv, float* b, const int* ldb, int* info);
extern "C" void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
extern "C" void dgetrs_(const char* t, const int* n, const int* nrhs, const double* a, const int* lda,
                        const int* ipiv, double* b, const int* ldb, int* info);
using namespace igabem;

static double peakRssMB() {
  rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss / 1024.0;
}
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// A (column-major, leading dimension ld) rows 3a + i, columns 3B + j = entries of e.
static void assembleLegacy(const SbieEntries& e, int npts, int ncol, double* A, long ld) {
  std::vector<int> dofs(ncol);
  for (int c = 0; c < ncol; ++c) dofs[c] = c;
#pragma omp parallel for schedule(dynamic, 4)
  for (int a = 0; a < npts; ++a)
    e.rowsAgainst({a}, dofs, [&](int, const double* acc) {
      for (int B = 0; B < ncol; ++B)
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i) A[(3 * a + i) + ld * (3 * B + j)] = acc[9 * B + 3 * j + i];
    });
}

// Restarted GMRES (modified Gram-Schmidt) for A x = b, A dense n x n (multithreaded dgemv);
// iterations to a relative residual tol.
int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: dense_bench <geometry.dat> [--tol 1e-7] [--order 4] [--assembly batched|legacy] [--lu s|d] [--check]\n");
    return 1;
  }
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  double tol = 1e-7;
  int order = -1, chunk = 96;
  bool legacy = false, single = true, check = false, gm = true;
  double gtol = 1e-10;
  int nblocks = 4;
  const char *csv = nullptr, *label = "", *save_u = nullptr, *ref_u = nullptr;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--order")) order = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--chunk")) chunk = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--assembly")) legacy = !std::strcmp(argv[++i], "legacy");
    else if (!std::strcmp(argv[i], "--lu")) { gm = false; single = !std::strcmp(argv[++i], "s"); }
    else if (!std::strcmp(argv[i], "--check")) check = true;
    else if (!std::strcmp(argv[i], "--save-u")) save_u = argv[++i];
    else if (!std::strcmp(argv[i], "--ref-u")) ref_u = argv[++i];
    else if (!std::strcmp(argv[i], "--solver")) gm = !std::strcmp(argv[++i], "gmres");
    else if (!std::strcmp(argv[i], "--gtol")) gtol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--blocks")) nblocks = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--csv")) { csv = argv[++i]; label = argv[++i]; }
  }
  const PlateHole ph;
  const CubeCavity cc;
  bool cavity = false;
  for (int i = 2; i < argc; ++i)
    if (!std::strcmp(argv[i], "--problem")) cavity = !std::strcmp(argv[i + 1], "cavity");
  // Exact displacement and traction: 3D Kirsch field (plate with a hole) or Goodier field (cube with a cavity).
  const Material mat = cavity ? cc.mat : ph.mat;
  auto exact_u = [&](const Vec3& x) { return cavity ? cc.displacement(CubeCavity::Goodier, x) : ph.displacement(PlateHole::Kirsch, x); };
  auto exact_t = [&](const Vec3& y, const Vec3& n) {
    return cavity ? Vec3(cc.traction(CubeCavity::Goodier, y, n)) : Vec3(ph.traction(PlateHole::Kirsch, y, n));
  };
  const double t_start = now();
  double t0 = t_start;
  Mesh mesh;
  mesh.build(loadDat(argv[1]), PartitionOptions());
  TractionSpace ts;
  ts.build(mesh);
  const int np = int(mesh.colloc.size()), N = 3 * mesh.ndof, Nt = 3 * ts.ndof;
  const Eigen::MatrixXd tc = ts.interpolate(
      [&](const Vec3& y, const Vec3& n) { return TractionBlock(exact_t(y, n)); }, 1);
  const double t_mesh = now() - t0;
  std::printf("# %d unknowns, %d traction unknowns, %zu elements; entry rule tol %g order %d; %s assembly, %s LU\n", N,
              Nt, mesh.elements.size(), tol, order, legacy ? "legacy" : "batched", single ? "mixed" : "double");

  // Right-hand side b = G tc and raw H in the top-left of A (N + 6 bordered).
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N + 6);
  t0 = now();
  HugeBuffer Abuf(size_t(N + 6) * (N + 6));
  Eigen::Map<Eigen::MatrixXd> A(Abuf.data(), N + 6, N + 6);
  const double t_alloc = now() - t0;
  double t_g = 0.0, t_h = 0.0;
  DenseSbie::Stats st;
  Eigen::MatrixXd S;  // block row sums of raw H
  t0 = now();
  if (legacy) {
    {
      SbieEntries eg(mesh, ts, mat, opt, mesh.colloc, SbieEntries::G);
      eg.setEntryRule(tol, order);
      Eigen::MatrixXd G(N, Nt);
      assembleLegacy(eg, np, ts.ndof, G.data(), N);
      b.head(N) = G * tc;
    }
    t_g = now() - t0;
    t0 = now();
    SbieEntries eh(mesh, ts, mat, opt, mesh.colloc, SbieEntries::H);
    eh.setEntryRule(tol, order);
    assembleLegacy(eh, np, mesh.ndof, A.data(), N + 6);
    t_h = now() - t0;
  } else {
    DenseSbie ds(mesh, ts, mat, opt, tol, order, chunk);
    t_g = now() - t0;  // quadrature tables (reported as "G": no G is formed)
    t0 = now();
    Eigen::MatrixXd GC;
    DenseSbie::Out o;
    o.H = A.data();
    o.ldh = N + 6;
    o.C = &tc;
    o.GC = &GC;
    S.setZero(N, 3);
    o.S = S.data();
    ds.assemble(o);
    b.head(N) = GC.col(0);
    t_h = now() - t0;
    st = ds.stats();
    std::printf("# H + Gc %.2f s: far %ld pairs %ld qp %.1f s cpu | near %ld pairs %ld qp %.1f s cpu | classify %.1f s, flush %.1f s, busy %.1f s cpu\n", t_h,
                st.far_pairs, st.far_qp, st.t_far, st.near_pairs, st.near_qp, st.t_near, st.t_class, st.t_flush, st.t_busy);
  }
  if (check) {  // entries against the legacy assembly
    SbieEntries eh(mesh, ts, mat, opt, mesh.colloc, SbieEntries::H);
    eh.setEntryRule(tol, order);
    Eigen::MatrixXd H(N, N);
    assembleLegacy(eh, np, mesh.ndof, H.data(), N);
    SbieEntries eg(mesh, ts, mat, opt, mesh.colloc, SbieEntries::G);
    eg.setEntryRule(tol, order);
    Eigen::MatrixXd G(N, Nt);
    assembleLegacy(eg, np, ts.ndof, G.data(), N);
    const Eigen::VectorXd gc = G * tc;
    std::printf("# check: max |H - H_legacy| / max|H| = %.3e, |Gc - Gc_legacy| / |Gc| = %.3e\n",
                (A.topLeftCorner(N, N) - H).cwiseAbs().maxCoeff() / H.cwiseAbs().maxCoeff(),
                (b.head(N) - gc).norm() / gc.norm());
  }
  t0 = now();
  // Rigid-body term: H = Hraw - N_B(x_a) sum_B Hraw(a, B).
  // The block row sums S_a = sum_B Hraw(a, B) as one product with [I I ... I]^T.
  if (legacy) {
    Eigen::MatrixXd E3 = Eigen::MatrixXd::Zero(N, 3);
    for (int B = 0; B < mesh.ndof; ++B) E3.block<3, 3>(3 * B, 0).setIdentity();
    S = A.topLeftCorner(N, N) * E3;
  }
  for (int a = 0; a < np; ++a)
    for (const auto& [B, NB] : mesh.colloc[a].basis) A.block<3, 3>(3 * a, 3 * B) -= NB * S.block<3, 3>(3 * a, 0);
  const Eigen::MatrixXd Rm = rigidModes(mesh);
  const double s = A.topLeftCorner(N, N).cwiseAbs().maxCoeff();
  A.topRightCorner(N, 6) = s * Rm;
  A.bottomLeftCorner(6, N) = s * Rm.transpose();
  const double t_rigid = now() - t0;

  t0 = now();
  const int n = N + 6, one = 1;
  int info = 0, iters = 0;
  std::vector<int> piv(n);
  Eigen::VectorXd u;
  double t_lu = 0.0, t_sol = 0.0, res = 0.0;
  if (gm) {
    const DenseGmres gs(A.data(), n, bisectionBlocks(mesh, nblocks));
    t_lu = now() - t0;  // block preconditioner
    t0 = now();
    Eigen::VectorXd x;
    iters = gs.solve(b, x, gtol, res);
    u = x.head(N);
    t_sol = now() - t0;
  } else if (single) {
    Eigen::MatrixXf Af = A.cast<float>();
    sgetrf_(&n, &n, Af.data(), &n, piv.data(), &info);
    t_lu = now() - t0;
    t0 = now();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n), r = b;
    const double nb = b.norm();
    for (iters = 0; iters < 20; ++iters) {
      res = r.norm() / nb;
      if (res < 1e-14) break;
      Eigen::VectorXf rf = r.cast<float>();
      sgetrs_("N", &n, &one, Af.data(), &n, piv.data(), rf.data(), &n, &info);
      x += rf.cast<double>();
      r.noalias() = b - A * x;
    }
    u = x.head(N);
    t_sol = now() - t0;
  } else {
    Eigen::VectorXd x = b;
    dgetrf_(&n, &n, A.data(), &n, piv.data(), &info);
    t_lu = now() - t0;
    t0 = now();
    dgetrs_("N", &n, &one, A.data(), &n, piv.data(), x.data(), &n, &info);
    u = x.head(N);
    t_sol = now() - t0;
  }
  t0 = now();
  const double err = displacementL2Error(mesh, u, exact_u);
  const double t_post = now() - t0;
  double du = -1.0;  // relative difference to a reference solution (--ref-u)
  if (save_u) { FILE* f = std::fopen(save_u, "wb"); std::fwrite(u.data(), sizeof(double), u.size(), f); std::fclose(f); }
  if (ref_u) {
    Eigen::VectorXd ur(u.size());
    FILE* f = std::fopen(ref_u, "rb");
    if (f && std::fread(ur.data(), sizeof(double), ur.size(), f) == size_t(ur.size())) du = (u - ur).norm() / ur.norm();
    if (f) std::fclose(f);
    std::printf("# |u - u_ref| / |u_ref| = %.3e\n", du);
  }
  const double total = t_mesh + t_alloc + t_g + t_h + t_rigid + t_lu + t_sol + t_post;
  const double wall = now() - t_start;
  std::printf("# wall %.2f s (main), allocation %.2f s\n", wall, t_alloc);
  std::printf("# mesh %.2f s, G %.2f s, H %.2f s, rigid %.2f s, LU %.2f s (info %d), solve %.2f s (%d refinements, res %.1e), "
              "error %.2f s; total %.2f s, peak RSS %.0f MB; %s rel L2 %.4e\n",
              t_mesh, t_g, t_h, t_rigid, t_lu, info, t_sol, iters, res, t_post, total, peakRssMB(), cavity ? "Goodier" : "Kirsch", err);
  if (csv) {
    FILE* f = std::fopen(csv, "a");
    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) == 0)
      std::fprintf(f, "label,unknowns,elements,assembly,lu,tol,order,t_mesh,t_g,t_h,t_rigid,t_lu,t_solve,refinements,t_post,"
                      "t_total,far_qp,near_qp,cpu_far,cpu_near,rss_MB,err_kirsch\n");
    std::fprintf(f, "%s,%d,%zu,%s,%s,%g,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%.3f,%ld,%ld,%.2f,%.2f,%.0f,%.6e\n", label, N,
                 mesh.elements.size(), legacy ? "legacy" : "batched", single ? "s" : "d", tol, order, t_mesh, t_g, t_h,
                 t_rigid, t_lu, t_sol, iters, t_post, total, st.far_qp, st.near_qp, st.t_far, st.t_near, peakRssMB(), err);
    std::fclose(f);
  }
  return 0;
}
