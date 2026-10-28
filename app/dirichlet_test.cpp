// Interior Dirichlet BEM patch test (Taus et al. 2019, Sec. 3.2.3):
// prescribe u = alpha x on a closed surface, solve the regularized traction
// BIE (HSBIE) for t, and report the relative L2(Gamma) traction error against
// t = sigma n for increasing integration points per direction.
//
// usage: dirichlet_test <geometry.dat> <label> [--nmin a] [--nmax b]
//                       [--no-partition] [--no-near] [--csv file]
//                       [--source sx sy sz]
//
// With --source, the exact field is instead the Kelvin solution of a unit point
// load at (sx, sy, sz) outside the body, direction (1, 2, 3)/sqrt(14). The data
// are interpolated at the Greville points, so the error then also contains the
// discretization error and should converge under h-refinement.

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "hsbie.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: dirichlet_test <geometry.dat> <label> [--nmin a] [--nmax b] "
                 "[--no-partition] [--no-near] [--csv file]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2];
  int nmin = 1, nmax = 20;
  PartitionOptions popt;
  SolverOptions sopt;
  std::string csv;
  bool source = false;
  Eigen::Vector3d xs = Eigen::Vector3d::Zero();
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--nmin")) nmin = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--nmax")) nmax = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-partition")) popt.enabled = false;
    else if (!std::strcmp(argv[i], "--no-near")) sopt.near_refine = false;
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--source")) {
      source = true;
      for (int k = 0; k < 3; ++k) xs(k) = std::atof(argv[++i]);
    }
  }

  Mesh mesh;
  mesh.build(loadDat(geo), popt);
  TractionSpace ts;
  ts.build(mesh);
  int ncells = 0;
  for (const auto& e : mesh.elements) ncells += int(e.cells.size());

  Material mat;  // lambda = mu = 1 (nu = 1/4)
  const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
  const Eigen::Matrix3d sigma = mat.stress(alpha);
  const Kelvin K(mat);
  const Eigen::Vector3d dir = Eigen::Vector3d(1, 2, 3).normalized();
  auto uexact = [&](const Eigen::Vector3d& y) -> Eigen::Vector3d {
    if (!source) return alpha * y;
    Eigen::Matrix3d U, T;
    K.eval(y - xs, Eigen::Vector3d::UnitZ(), U, T);
    return U.transpose() * dir;
  };
  auto texact = [&](const Eigen::Vector3d& y, const Eigen::Vector3d& n) -> Eigen::Vector3d {
    if (!source) return sigma * n;
    Eigen::Matrix3d U, T;
    K.eval(y - xs, n, U, T);
    return T.transpose() * dir;
  };
  // Dirichlet data: u = alpha x is exact in the isoparametric space; the
  // point-load field is interpolated at the Greville points.
  Eigen::VectorXd ucoef(3 * mesh.ndof);
  if (!source)
    for (int g = 0; g < mesh.ndof; ++g) ucoef.segment<3>(3 * g) = alpha * mesh.dof_point[g];
  else
    ucoef = interpolateDisplacement(mesh, uexact);

  const int N = 3 * ts.ndof;
  std::printf("# %s: %zu patches, %zu elements, %d integration cells, %d traction dofs (%d unknowns)\n",
              label.c_str(), mesh.patches.size(), mesh.elements.size(), ncells, ts.ndof, N);
  if (source) std::printf("# exact field: Kelvin point load at (%g, %g, %g)\n", xs(0), xs(1), xs(2));
  std::printf("# %3s %12s %12s %9s\n", "n", "rel_L2_err", "cond_2", "time_s");

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    if (fresh) out << "label,partition,near,n,tdof,cells,rel_L2_err,cond,time_s\n";
  }
  for (int n = nmin; n <= nmax; ++n) {
    sopt.n_gauss = n;
    const auto t0 = std::chrono::steady_clock::now();
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    HsbieAssembler(mesh, ts, mat, sopt).assemble(ucoef, A, b);
    const Eigen::VectorXd t = A.partialPivLu().solve(b);
    const double time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double err = tractionL2Error(mesh, ts, t, texact);
    const Eigen::VectorXd sv = Eigen::BDCSVD<Eigen::MatrixXd>(A).singularValues();
    const double cond = sv(0) / sv(N - 1);
    std::printf("  %3d %12.4e %12.4e %9.2f\n", n, err, cond, time);
    std::fflush(stdout);
    if (out) out << label << ',' << popt.enabled << ',' << sopt.near_refine << ',' << n << ','
                 << ts.ndof << ',' << ncells << ',' << err << ',' << cond << ',' << time << '\n';
  }
  return 0;
}
