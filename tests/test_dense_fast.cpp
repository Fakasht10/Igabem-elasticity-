// Fast dense path (DenseSbie + DenseGmres, src/fast) against the reference
// path (SbieEntries row by row, G formed, dense LU) on geometries of different
// kind: the cube with a spherical cavity (degree 2, rational sphere patches,
// inward normals, Goodier field) and plates with a hole (degree 4, rational hole
// patches, 3D Kirsch field) of thickness t = 0.2 and the thin t = 0.02.
//
// Per case: H, G and G c equal to the reference entries with the same entry
// rule (round-off); the GMRES solution equal to the LU solution; the change of
// the solution from the degree-based far-field Gauss order against the
// reference far rule (tol 1e-10, dense assembly orders) far below the
// discretization error; the error against the exact field; the GMRES
// iteration count bounded (thin plates included).
//
// usage: test_dense_fast <repository root>

#include <cmath>
#include <cstdio>
#include <string>

#include "fast/dense_neumann.hpp"
#include "fast/sbie_entries.hpp"
#include "problems/cube_cavity.hpp"
#include "problems/plate_hole.hpp"

using namespace igabem;

static int failures = 0;

static void check(const std::string& name, double got, double lo, double hi) {
  const bool ok = got >= lo && got <= hi;
  std::printf("%-52s %s  %.6e  in [%.1e, %.1e]\n", name.c_str(), ok ? "ok  " : "FAIL", got, lo, hi);
  if (!ok) ++failures;
}

static Eigen::MatrixXd legacyMatrix(const SbieEntries& e, int npts, int ncol) {
  Eigen::MatrixXd A(3 * npts, 3 * ncol);
  std::vector<int> dofs(ncol);
  for (int c = 0; c < ncol; ++c) dofs[c] = c;
#pragma omp parallel for schedule(dynamic, 4)
  for (int a = 0; a < npts; ++a)
    e.rowsAgainst({a}, dofs, [&](int, const double* acc) {
      for (int B = 0; B < ncol; ++B)
        for (int j = 0; j < 3; ++j)
          for (int i = 0; i < 3; ++i) A(3 * a + i, 3 * B + j) = acc[9 * B + 3 * j + i];
    });
  return A;
}

// Reference solution: SbieEntries H and G with entry rule (tol, order), bordered dense LU.
static Eigen::VectorXd referenceSolve(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& opt,
                                      const Eigen::MatrixXd& tc, double tol, int order, Eigen::MatrixXd* Hraw = nullptr,
                                      Eigen::MatrixXd* Graw = nullptr) {
  const int np = int(m.colloc.size()), N = 3 * m.ndof;
  SbieEntries eh(m, ts, mat, opt, m.colloc, SbieEntries::H), eg(m, ts, mat, opt, m.colloc, SbieEntries::G);
  eh.setEntryRule(tol, order);
  eg.setEntryRule(tol, order);
  const Eigen::MatrixXd H = legacyMatrix(eh, np, m.ndof), G = legacyMatrix(eg, np, ts.ndof);
  if (Hraw) *Hraw = H;
  if (Graw) *Graw = G;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N + 6, N + 6);
  A.topLeftCorner(N, N) = H;
  for (int a = 0; a < np; ++a) {
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
    for (int B = 0; B < m.ndof; ++B) S += H.block<3, 3>(3 * a, 3 * B);
    for (const auto& [B, NB] : m.colloc[a].basis) A.block<3, 3>(3 * a, 3 * B) -= NB * S;
  }
  const Eigen::MatrixXd R = rigidModes(m);
  const double s = A.topLeftCorner(N, N).cwiseAbs().maxCoeff();
  A.topRightCorner(N, 6) = s * R;
  A.bottomLeftCorner(6, N) = s * R.transpose();
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N + 6);
  b.head(N) = G * tc.col(0);
  return A.partialPivLu().solve(b).head(N);
}

template <class U, class T>
static void runCase(const std::string& label, const std::string& geo, const Material& mat, U&& exact_u, T&& exact_t,
                    double err_lo, double err_hi, int max_iters) {
  Mesh m;
  m.build(loadDat(geo), PartitionOptions());
  TractionSpace ts;
  ts.build(m);
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  const DenseNeumannOptions o;
  const Eigen::MatrixXd tc =
      ts.interpolate([&](const Vec3& y, const Vec3& n) { return TractionBlock(exact_t(y, n)); }, 1);
  const int N = 3 * m.ndof, np = int(m.colloc.size());
  std::printf("# %s: %d unknowns, %zu elements\n", label.c_str(), N, m.elements.size());

  // Entries: DenseSbie (H, G formed, G c) against SbieEntries with the same rule.
  Eigen::MatrixXd Href, Gref;
  const Eigen::VectorXd u_same = referenceSolve(m, ts, mat, opt, tc, o.tol, o.min_order, &Href, &Gref);
  {
    DenseSbie ds(m, ts, mat, opt, o.tol, o.min_order);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N), G = Eigen::MatrixXd::Zero(N, 3 * ts.ndof), GC,
                    S = Eigen::MatrixXd::Zero(N, 3);
    DenseSbie::Out out;
    out.H = H.data();
    out.ldh = N;
    out.G = G.data();
    out.ldg = N;
    out.C = &tc;
    out.GC = &GC;
    out.S = S.data();
    ds.assemble(out);
    Eigen::MatrixXd E3 = Eigen::MatrixXd::Zero(N, 3);
    for (int B = 0; B < m.ndof; ++B) E3.block<3, 3>(3 * B, 0).setIdentity();
    check(label + " H entries vs reference", (H - Href).cwiseAbs().maxCoeff() / Href.cwiseAbs().maxCoeff(), 0, 1e-12);
    check(label + " G entries vs reference", (G - Gref).cwiseAbs().maxCoeff() / Gref.cwiseAbs().maxCoeff(), 0, 1e-12);
    const Eigen::VectorXd gc = Gref * tc.col(0);
    check(label + " G c vs reference", (GC.col(0) - gc).norm() / gc.norm(), 0, 1e-12);
    check(label + " row sums S vs reference", (S - Href * E3).norm() / (Href * E3).norm(), 0, 1e-12);
    (void)np;
  }
  // Solve: DenseNeumann (GMRES) against LU of the same discretization.
  DenseNeumann dn(m, ts, mat, opt, tc, o);
  std::vector<int> its;
  const Eigen::VectorXd u = dn.solve(&its).col(0);
  check(label + " GMRES vs LU, same rule", (u - u_same).norm() / u_same.norm(), 0, 1e-8);
  check(label + " GMRES iterations", its[0], 1, max_iters);
  // Degree-based far order against the reference far rule (dense assembly orders, tol 1e-10).
  const Eigen::VectorXd u_ref = referenceSolve(m, ts, mat, opt, tc, 1e-10, 0);
  const double err = displacementL2Error(m, u, exact_u), err_ref = displacementL2Error(m, u_ref, exact_u);
  check(label + " |u - u_ref| / |u_ref| (far rule)", (u - u_ref).norm() / u_ref.norm(), 0, 1e-2 * err_ref);
  check(label + " error vs exact field", err, err_lo, err_hi);
}

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const std::string geo = root + "/geo/";
  {
    const CubeCavity cc;
    runCase("cavity r1 Goodier", geo + "cube_cavity_r1.dat", cc.mat,
            [&](const Vec3& x) { return cc.displacement(CubeCavity::Goodier, x); },
            [&](const Vec3& y, const Vec3& n) { return cc.traction(CubeCavity::Goodier, y, n); }, 1.68e-2, 1.72e-2, 60);
  }
  const PlateHole ph;
  auto u = [&](const Vec3& x) { return ph.displacement(PlateHole::Kirsch, x); };
  auto t = [&](const Vec3& y, const Vec3& n) { return ph.traction(PlateHole::Kirsch, y, n); };
  runCase("plate p4 t0.2 r1 Kirsch", geo + "plate_hole_p4_t0.2_r1.dat", ph.mat, u, t, 6.53e-3, 6.66e-3, 60);
  runCase("thin plate p4 t0.02 r1 Kirsch", geo + "plate_hole_p4_t0.02_r1.dat", ph.mat, u, t, 1.54e-2, 1.57e-2, 60);
  std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
