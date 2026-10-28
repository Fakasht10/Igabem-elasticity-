// Traction-only (Neumann) problems on the unit cube with a spherical cavity of
// radius a = 0.2 at c = (0.5, 0.5, 0.5), solved with the SBIE.
//
// Load cases (lambda = mu = 1, nu = 1/4):
//   patch     u = alpha x, t = sigma n on both surfaces (exact for any geometry)
//   pressure  exact Lame field u = p a^3 (x - c) / (4 mu |x - c|^3), p = 1
//   goodier   exact infinite-medium field for remote tension S = 1 along z
//             (python/gen_cavity_exact.py); its traction is prescribed on the cube
//   tension   the engineering problem: t = S e_z on z = 0, 1, zero elsewhere
//
// For each n: displacement error (rigid motion removed), rotation residual,
// small singular values, and sigma_zz at the cavity equator point c + a e_x,
// recovered from u_h and t (recovery.hpp). --profile writes the tangential
// stresses along the meridian c + a (sin th, 0, cos th) for the last n.
//
// usage: cavity_test <geometry.dat> <label> <case> [--nmin a] [--nmax b]
//                    [--no-svd] [--csv file] [--profile file]

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "cavity_exact_gen.hpp"
#include "recovery.hpp"
#include "sbie.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: cavity_test <geometry.dat> <label> <patch|pressure|goodier|tension> "
                 "[--nmin a] [--nmax b] [--no-svd] [--csv file] [--profile file]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2], lc = argv[3];
  int nmin = 4, nmax = 14;
  bool svd = true;
  std::string csv, profile;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--nmin")) nmin = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--nmax")) nmax = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-svd")) svd = false;
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--profile")) profile = argv[++i];
  }
  if (lc != "patch" && lc != "pressure" && lc != "goodier" && lc != "tension") {
    std::cerr << "unknown case " << lc << "\n";
    return 1;
  }

  Mesh mesh;
  mesh.build(loadDat(geo), PartitionOptions{});
  int ncells = 0;
  for (const auto& e : mesh.elements) ncells += int(e.cells.size());
  const int ncube = 6, np = int(mesh.patches.size());  // cube patches first, then the cavity

  const Material mat;
  const double a = 0.2, S = 1.0, p = 1.0;
  const Vec3 c(0.5, 0.5, 0.5);
  const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
  const bool exact = lc != "tension";

  // Exact displacement and gradient at x.
  auto field = [&](const Vec3& x, Vec3& u, Eigen::Matrix3d& G) {
    const Vec3 X = x - c;
    if (lc == "patch") {
      u = alpha * x;
      G = alpha;
    } else if (lc == "pressure") {
      const double A = p * a * a * a / (4.0 * mat.mu), r = X.norm();
      u = A * X / (r * r * r);
      G = A * (Eigen::Matrix3d::Identity() / (r * r * r) - 3.0 * X * X.transpose() / std::pow(r, 5));
    } else {
      goodierField(X(0), X(1), X(2), a, S, mat.mu, mat.nu(), u, G);
    }
  };
  auto uexact = [&](const Vec3& x) {
    Vec3 u;
    Eigen::Matrix3d G;
    field(x, u, G);
    return u;
  };
  auto traction = [&](const Vec3& y, const Vec3& n) -> Vec3 {
    if (lc == "tension") return (y - c).norm() > 2.0 * a ? Vec3(S * n(2) * Vec3::UnitZ()) : Vec3::Zero();
    Vec3 u;
    Eigen::Matrix3d G;
    field(y, u, G);
    return mat.stress(G) * n;
  };
  auto exactStress = [&](const Vec3& x) {
    Vec3 u;
    Eigen::Matrix3d G;
    field(x, u, G);
    return Eigen::Matrix3d(mat.stress(G));
  };

  const SurfacePoint eq = locate(mesh, c + a * Vec3::UnitX(), ncube, np);
  const double seq_exact = exact ? exactStress(eq.E.x)(2, 2) : std::nan("");

  std::printf("# %s (%s): %zu patches, %zu elements, %d integration cells, %d dofs (%d unknowns)\n",
              label.c_str(), lc.c_str(), mesh.patches.size(), mesh.elements.size(), ncells,
              mesh.ndof, 3 * mesh.ndof);
  std::printf("# volume = %.15f (1 - 4 pi a^3 / 3 = %.15f)\n", mesh.volume(),
              1.0 - 4.0 * M_PI * a * a * a / 3.0);
  std::printf("# %3s %12s %12s %12s %12s %12s %12s %9s\n", "n", "rel_L2_err", "rot_resid",
              "s6/s_max", "s7/s_max", "szz_eq", "szz_eq_ex", "time_s");

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    out.precision(15);
    if (fresh) out << "label,case,n,ndof,elements,cells,rel_L2_err,rot_resid,s6_rel,s7_rel,"
                      "szz_eq,szz_eq_exact,time_s\n";
  }

  const Eigen::MatrixXd R = rigidModes(mesh);
  const int N = 3 * mesh.ndof;
  SolverOptions sopt;
  Eigen::VectorXd uh;
  for (int n = nmin; n <= nmax; ++n) {
    sopt.n_gauss = n;
    const auto t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    SbieAssembler(mesh, mat, sopt).assemble(traction, A, b);

    // Bordered system [A s*R; s*R^T 0] fixes the 6 rigid-body modes.
    const double s = A.cwiseAbs().maxCoeff();
    Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
    Ab.topLeftCorner(N, N) = A;
    Ab.topRightCorner(N, 6) = s * R;
    Ab.bottomLeftCorner(6, N) = s * R.transpose();
    Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6);
    bb.head(N) = b;
    uh = Ab.partialPivLu().solve(bb).head(N);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double err = exact ? displacementL2Error(mesh, uh, uexact) : std::nan("");
    const double rot = (A * R.rightCols(3)).norm() / A.norm();
    double s6 = std::nan(""), s7 = std::nan("");
    if (svd) {
      const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(A).singularValues();
      s6 = sv(N - 6) / sv(0);
      s7 = sv(N - 7) / sv(0);
    }
    const double seq = surfaceStress(mesh, mat, eq, uh, traction(eq.E.x, eq.N))(2, 2);

    std::printf("  %3d %12.4e %12.4e %12.4e %12.4e %12.8f %12.8f %9.2f\n", n, err, rot, s6, s7,
                seq, seq_exact, t);
    std::fflush(stdout);
    if (out) out << label << ',' << lc << ',' << n << ',' << mesh.ndof << ','
                 << mesh.elements.size() << ',' << ncells << ',' << err << ',' << rot << ','
                 << s6 << ',' << s7 << ',' << seq << ',' << seq_exact << ',' << t << '\n';
  }

  if (!profile.empty()) {
    std::ofstream pf(profile);
    pf << "theta_deg,s_mer,s_circ,s_mer_exact,s_circ_exact\n";
    for (int k = 0; k <= 180; ++k) {
      const double th = M_PI * k / 180.0;
      const Vec3 er(std::sin(th), 0.0, std::cos(th)), et(std::cos(th), 0.0, -std::sin(th));
      const Vec3 ep = Vec3::UnitY();
      const SurfacePoint sp = locate(mesh, c + a * er, ncube, np);
      const Eigen::Matrix3d sh = surfaceStress(mesh, mat, sp, uh, traction(sp.E.x, sp.N));
      pf << k << ',' << et.dot(sh * et) << ',' << ep.dot(sh * ep);
      if (exact) {
        const Eigen::Matrix3d se = exactStress(sp.E.x);
        pf << ',' << et.dot(se * et) << ',' << ep.dot(se * ep) << '\n';
      } else {
        pf << ",,\n";
      }
    }
  }
  return 0;
}
