// Dense SBIE operators on the plate with a hole, built with the fast element
// integrals of src/fast (SbieEntries: cached Gauss rules at quadrature tolerance
// tol for separated element pairs, the exact quadrature for close ones), and the
// traction-free Kirsch problem solved by LAPACK LU (dgetrf, multithreaded BLAS) of
// the bordered H. G is applied to the data and freed before H is assembled, so
// the peak memory is one N x N matrix (plus G while it is in use).
//
// usage: dense_bench <geometry.dat> [--tol 1e-7] [--order 4] [--n 12] [--far-tol 1e-10]

#include <sys/resource.h>

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "fast/sbie_entries.hpp"
#include "problems/plate_hole.hpp"
#include "sbie.hpp"

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
static void assemble(const SbieEntries& e, int npts, int ncol, double* A, long ld) {
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

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: dense_bench <geometry.dat> [--tol 1e-7] [--order 4] [--n 12] [--far-tol 1e-10]\n"); return 1; }
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  double tol = 1e-7;
  int order = 4;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--order")) order = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
  }
  const PlateHole ph;
  Mesh mesh;
  mesh.build(loadDat(argv[1]), PartitionOptions());
  TractionSpace ts;
  ts.build(mesh);
  const int np = int(mesh.colloc.size()), N = 3 * mesh.ndof, Nt = 3 * ts.ndof;
  std::printf("# %d unknowns, %d traction unknowns, entry rule tol %g order %d\n", N, Nt, tol, order);
  const Eigen::VectorXd tc = ts.interpolate(
      [&](const Vec3& y, const Vec3& n) { return TractionBlock(ph.traction(PlateHole::Kirsch, y, n)); }, 1);

  double t0 = now();
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N + 6);
  {
    SbieEntries eg(mesh, ts, ph.mat, opt, mesh.colloc, SbieEntries::G);
    eg.setEntryRule(tol, order);
    Eigen::MatrixXd G(N, Nt);
    assemble(eg, np, ts.ndof, G.data(), N);
    b.head(N) = G * tc;
  }
  const double t_g = now() - t0;
  t0 = now();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N + 6, N + 6);
  {
    SbieEntries eh(mesh, ts, ph.mat, opt, mesh.colloc, SbieEntries::H);
    eh.setEntryRule(tol, order);
    assemble(eh, np, mesh.ndof, A.data(), N + 6);
  }
  // Rigid-body term: H = Hraw - N_B(x_a) sum_B Hraw(a, B).
  for (int a = 0; a < np; ++a) {
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    for (int B = 0; B < mesh.ndof; ++B) S += A.block<3, 3>(3 * a, 3 * B);
    for (const auto& [B, NB] : mesh.colloc[a].basis) A.block<3, 3>(3 * a, 3 * B) -= NB * S;
  }
  const Eigen::MatrixXd R = rigidModes(mesh);
  const double s = A.topLeftCorner(N, N).cwiseAbs().maxCoeff();
  A.topRightCorner(N, 6) = s * R;
  A.bottomLeftCorner(6, N) = s * R.transpose();
  const double t_h = now() - t0;
  t0 = now();
  const int n = N + 6, one = 1;
  int info = 0;
  std::vector<int> piv(n);
  dgetrf_(&n, &n, A.data(), &n, piv.data(), &info);
  const double t_lu = now() - t0;
  dgetrs_("N", &n, &one, A.data(), &n, piv.data(), b.data(), &n, &info);
  const Eigen::VectorXd u = b.head(N);
  const double err = displacementL2Error(mesh, u, [&](const Vec3& x) { return ph.displacement(PlateHole::Kirsch, x); });
  std::printf("# G %.1f s, H %.1f s, LU %.1f s (info %d), peak RSS %.0f MB; Kirsch rel L2 %.4e\n", t_g, t_h, t_lu, info,
              peakRssMB(), err);
  return 0;
}
