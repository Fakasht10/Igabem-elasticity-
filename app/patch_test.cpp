// Interior Neumann BEM patch test (Taus et al. 2019, Sec. 3.2.1-3.2.2):
// prescribe t = sigma n for u = alpha x on a closed surface, solve the SBIE,
// and report the relative L2(Gamma) displacement error against the number of
// integration points per direction.
//
// usage: patch_test <geometry.dat> <label> [--nmin a] [--nmax b]
//                   [--no-partition] [--no-near] [--csv file]

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "sbie.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: patch_test <geometry.dat> <label> [--nmin a] [--nmax b] "
                 "[--no-partition] [--no-near] [--csv file]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2];
  int nmin = 1, nmax = 20;
  PartitionOptions popt;
  SolverOptions sopt;
  std::string csv;
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--nmin")) nmin = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--nmax")) nmax = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-partition")) popt.enabled = false;
    else if (!std::strcmp(argv[i], "--no-near")) sopt.near_refine = false;
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
  }

  Mesh mesh;
  mesh.build(loadDat(geo), popt);
  int ncells = 0;
  for (const auto& e : mesh.elements) ncells += int(e.cells.size());

  Material mat;  // lambda = mu = 1 (nu = 1/4)
  Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
  const Eigen::Matrix3d sigma = mat.stress(alpha);

  std::printf("# %s: %zu patches, %zu elements, %d integration cells, %d dofs (%d unknowns)\n",
              label.c_str(), mesh.patches.size(), mesh.elements.size(), ncells, mesh.ndof,
              3 * mesh.ndof);
  std::printf("# volume = %.15f, normals %s\n", mesh.volume(),
              mesh.normal_sign > 0 ? "as given" : "flipped");
  std::printf("# %3s %12s %12s %12s %12s %9s\n", "n", "rel_L2_err", "rot_resid",
              "s6/s_max", "s7/s_max", "time_s");

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    if (fresh) out << "label,partition,near,n,ndof,cells,rel_L2_err,rot_resid,s6_rel,s7_rel,time_s\n";
  }

  const Eigen::MatrixXd R = rigidModes(mesh);
  const int N = 3 * mesh.ndof;
  for (int n = nmin; n <= nmax; ++n) {
    sopt.n_gauss = n;
    const auto t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    SbieAssembler(mesh, mat, sopt).assemble(sigma, A, b);

    // Bordered system [A s*R; s*R^T 0] fixes the 6 rigid-body modes.
    const double s = A.cwiseAbs().maxCoeff();
    Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
    Ab.topLeftCorner(N, N) = A;
    Ab.topRightCorner(N, 6) = s * R;
    Ab.bottomLeftCorner(6, N) = s * R.transpose();
    Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6);
    bb.head(N) = b;
    const Eigen::VectorXd sol = Ab.partialPivLu().solve(bb);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const double err = relativeL2Error(mesh, sol.head(N), alpha);
    // Rotations carry zero traction, so A * rotation must vanish (kernel check).
    const double rot = (A * R.rightCols(3)).norm() / A.norm();
    const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(A).singularValues();
    const double s6 = sv(N - 6) / sv(0), s7 = sv(N - 7) / sv(0);

    std::printf("  %3d %12.4e %12.4e %12.4e %12.4e %9.2f\n", n, err, rot, s6, s7, t);
    std::fflush(stdout);
    if (out) out << label << ',' << popt.enabled << ',' << sopt.near_refine << ',' << n << ','
                 << mesh.ndof << ',' << ncells << ',' << err << ',' << rot << ',' << s6 << ','
                 << s7 << ',' << t << '\n';
  }
  return 0;
}
