// Regression tests on small meshes: every validated case keeps its numbers.
// Reference values come from the committed result CSVs (cube, sphere, HSBIE,
// cavity) or from the first run of the library path (plate); tolerances are a
// few percent of the reference error, so any change to quadrature, kernels or
// assembly that alters a result shows up here.
//
// usage: test_regression <repository root>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "hsbie.hpp"
#include "interior.hpp"
#include "model.hpp"
#include "neumann_model.hpp"
#include "probes.hpp"
#include "vtk.hpp"
#include "problems/cube_cavity.hpp"
#include "problems/plate_hole.hpp"
#include "sbie.hpp"

using namespace igabem;

static int failures = 0;

static void check(const char* name, double got, double want, double tol) {
  const bool ok = std::abs(got - want) <= tol;
  std::printf("%-44s %s  got %.10e  want %.10e  tol %.1e\n", name, ok ? "ok  " : "FAIL", got, want, tol);
  if (!ok) ++failures;
}

static double neumannPatch(const std::string& geo, int n) {
  Mesh m;
  m.build(loadDat(geo), PartitionOptions{});
  const Material mat;
  const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
  SolverOptions o;
  o.n_gauss = n;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  SbieAssembler(m, mat, o).assemble(mat.stress(alpha), A, b);
  const Eigen::MatrixXd R = rigidModes(m);
  const int N = 3 * m.ndof;
  const double s = A.cwiseAbs().maxCoeff();
  Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
  Ab.topLeftCorner(N, N) = A;
  Ab.topRightCorner(N, 6) = s * R;
  Ab.bottomLeftCorner(6, N) = s * R.transpose();
  Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6);
  bb.head(N) = b;
  return relativeL2Error(m, Ab.partialPivLu().solve(bb).head(N), alpha);
}

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const std::string geo = root + "/geo/";

  // SBIE, interior Neumann patch tests (results/patch_tests.csv).
  check("sbie cube p2, n = 8", neumannPatch(geo + "cube_p2.dat", 8), 1.0830e-07, 1e-10);
  check("sbie sphere r0, n = 5", neumannPatch(geo + "sphere_r0.dat", 5), 8.3746e-09, 1e-12);

  // HSBIE, interior Dirichlet patch test on the cube (results/dirichlet_tests.csv).
  {
    Mesh m;
    m.build(loadDat(geo + "cube_p2.dat"), PartitionOptions{});
    TractionSpace ts;
    ts.build(m);
    const Material mat;
    const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
    SolverOptions o;
    o.n_gauss = 4;
    Eigen::VectorXd uc(3 * m.ndof);
    for (int g = 0; g < m.ndof; ++g) uc.segment<3>(3 * g) = alpha * m.dof_point[g];
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    HsbieAssembler(m, ts, mat, o).assemble(uc, A, b);
    const Eigen::VectorXd tc = A.partialPivLu().solve(b);
    const Eigen::Matrix3d sig = mat.stress(alpha);
    check("hsbie cube p2 Dirichlet, n = 4",
          tractionL2Error(m, ts, tc, [&](const Vec3&, const Vec3& n) { return Vec3(sig * n); }), 0.0, 1e-13);
  }

  // Cavity, Goodier field, 12 elements (results/cavity_tests.csv, fixed n = 12).
  {
    const CubeCavity cc;
    Mesh m;
    m.build(loadDat(geo + "cube_cavity_r0.dat"), PartitionOptions{});
    SolverOptions o;
    o.n_gauss = 12;
    Eigen::MatrixXd A, B;
    auto data = [&](const Vec3& y, const Vec3& n) { return TractionBlock(cc.traction(CubeCavity::Goodier, y, n)); };
    SbieAssembler(m, cc.mat, o).assembleTractions(data, 1, A, B);
    const Eigen::MatrixXd R = rigidModes(m);
    const int N = 3 * m.ndof;
    const double s = A.cwiseAbs().maxCoeff();
    Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
    Ab.topLeftCorner(N, N) = A;
    Ab.topRightCorner(N, 6) = s * R;
    Ab.bottomLeftCorner(6, N) = s * R.transpose();
    Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6);
    bb.head(N) = B.col(0);
    const Eigen::VectorXd uh = Ab.partialPivLu().solve(bb).head(N);
    check("cavity r0 Goodier, u error",
          displacementL2Error(m, uh, [&](const Vec3& x) { return cc.displacement(CubeCavity::Goodier, x); }),
          1.3789e-02, 1e-6);
    const SurfacePoint eq = locate(m, cc.c + cc.a * Vec3::UnitX(), CubeCavity::ncube, int(m.patches.size()));
    check("cavity r0 Goodier, sigma_zz at equator",
          surfaceStress(m, cc.mat, eq, uh, cc.traction(CubeCavity::Goodier, eq.E.x, eq.N))(2, 2),
          2.03305114, 1e-6);
  }

  // Plate with a hole through the library path (NeumannModel, adaptive quadrature,
  // interpolated data, on-disk cache), p = 2, 64 elements.
  {
    const std::string cache = (std::filesystem::temp_directory_path() / "igabem_test_cache").string();
    std::filesystem::remove_all(cache);
    const PlateHole ph;
    SolverOptions o;
    o.n_gauss = 12;
    o.far_tol = 1e-10;
    auto data = [&](const Vec3& y, const Vec3& n) {
      TractionBlock T(3, 3);
      T.col(0) = ph.traction(PlateHole::Patch, y, n);
      T.col(1) = ph.traction(PlateHole::Kirsch, y, n);
      T.col(2) = ph.traction(PlateHole::Tension, y, n);
      return T;
    };
    double scf[2];
    for (int pass = 0; pass < 2; ++pass) {  // second pass loads the operators from the cache
      NeumannModel nm;
      nm.setup(geo + "plate_hole_p2_t0.2_r1.dat", ph.mat, o, cache);
      if (pass == 1) check("plate cache hit", nm.from_cache, 1.0, 0.0);
      const Eigen::MatrixXd UH = nm.solve(data, 3);
      const auto hole = PlateHole::holePatches(nm.mesh);
      if (pass == 0) {
        check("plate p2 r1 patch test, u error",
              displacementL2Error(nm.mesh, UH.col(0), [&](const Vec3& x) { return ph.displacement(PlateHole::Patch, x); }),
              0.0, 1e-6);
        check("plate p2 r1 Kirsch, u error",
              displacementL2Error(nm.mesh, UH.col(1), [&](const Vec3& x) { return ph.displacement(PlateHole::Kirsch, x); }),
              5.5945230057e-02, 1e-6);
        check("plate p2 r1 tension, sigma_xx mid-plane",
              ph.sxxAtHole(nm.mesh, hole, PlateHole::Tension, UH.col(2), 0.0), 4.0234877354, 1e-6);
      }
      scf[pass] = ph.sxxAtHole(nm.mesh, hole, PlateHole::Kirsch, UH.col(1), 0.0);
    }
    check("plate p2 r1 Kirsch SCF, cached = fresh", scf[1], scf[0], 0.0);
    std::filesystem::remove_all(cache);
  }

  // Mixed problem (end face x = -1/2 clamped to the exact field or to zero), interior
  // evaluation, probes and VTK output, on the same p = 2, 64-element plate.
  for (int clamp = 0; clamp < 2; ++clamp) {
    const PlateHole ph;
    SolverOptions o;
    o.n_gauss = 12;
    o.far_tol = 1e-10;
    Model m;
    Model::Selector sel;
    if (clamp) sel = [&](int, const Vec3& c, const Vec3&) { return ph.clampedEnd(c); };
    m.setup(geo + "plate_hole_p2_t0.2_r1.dat", ph.mat, o, sel, "");
    auto tr = [&](const Vec3& y, const Vec3& n) {
      TractionBlock T(3, 3);
      T.col(0) = ph.traction(PlateHole::Patch, y, n);
      T.col(1) = ph.traction(PlateHole::Kirsch, y, n);
      T.col(2) = ph.traction(PlateHole::Tension, y, n);
      return T;
    };
    auto ud = [&](const Vec3& x) {
      TractionBlock U(3, 3);
      U.col(0) = ph.displacement(PlateHole::Patch, x);
      U.col(1) = ph.displacement(PlateHole::Kirsch, x);
      U.col(2).setZero();
      return U;
    };
    const auto S = m.solutions(tr, ud, 3);
    const auto hole = PlateHole::holePatches(m.mesh());
    auto onHole = [&](int p) { return bool(hole[p]); };
    const std::string tag = clamp ? "plate p2 r1 clamped" : "plate p2 r1 free";
    auto uerr = [&](int k, PlateHole::Case c) {
      return displacementL2Error(m.mesh(), S[k].u, [&](const Vec3& x) { return ph.displacement(c, x); }, 10, !clamp);
    };
    if (clamp) {
      check((tag + " patch test, u error").c_str(), uerr(0, PlateHole::Patch), 0.0, 1e-8);
      check((tag + " Kirsch, u error").c_str(), uerr(1, PlateHole::Kirsch), 7.4291109835e-02, 1e-6);
      check((tag + " tension, sigma_xx mid-plane").c_str(),
            S[2].stress(locate(m.mesh(), Vec3(0.0, ph.a, 0.0), onHole))(0, 0), 3.7008899893, 1e-6);
    }
    InteriorEvaluator e0(S[0], o), e1(S[1], o);
    check((tag + " interior patch test, |sigma - sigma_ex|").c_str(),
          (e0.eval(Vec3(0.1, 0.3, 0.05)).stress - ph.stress(PlateHole::Patch, Vec3::Zero())).norm(), 0.0, 1e-8);
    check((tag + " interior Kirsch sigma_xx(0, 1.25a, 0)").c_str(), e1.eval(Vec3(0.0, 0.25, 0.0)).stress(0, 0),
          clamp ? 2.0112161021 : 2.0251376592, 1e-6);
    check((tag + " inside test, point in the hole").c_str(), e1.eval(Vec3::Zero()).winding, 0.0, 1e-10);
    const double sb = S[1].stress(locate(m.mesh(), Vec3(0.0, ph.a, 0.0), onHole))(0, 0);
    const std::function<Vec3(double)> arc = [&](double s) {
      return Vec3(ph.a * std::cos(M_PI * s), ph.a * std::sin(M_PI * s), 0.0);
    };
    check((tag + " hole probe at 90 deg = boundary stress").c_str(),
          boundaryProbe(S[1], arc, 2, onHole).f[1].stress(0, 0), sb, 1e-12);
    const VtuData v = surfaceVtu(S[1], 2, ph.exactField(PlateHole::Kirsch));
    check((tag + " surface VTU points").c_str(), double(v.points.size()), 9.0 * m.mesh().elements.size(), 0.0);
  }

  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
