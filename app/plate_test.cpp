// Plate with a circular through-hole under traction loading (src/problems/plate_hole.hpp).
// Thin driver: the operators come from NeumannModel (assembled once per mesh and
// quadrature setting, then cached on disk), all load cases are solved together.
//
// usage: plate_test <geometry.dat> <label> <case[,case...]> <t> [--n 12] [--far-tol 1e-10]
//                   [--exact-data] [--length L] [--cache dir|none] [--csv file] [--profile prefix]
//   cases: patch, kirsch, kirsch_free, tension
//   --exact-data  integrate the traction data exactly instead of interpolating it
//                 in the traction space (validation of the interpolation; reassembles)
//   --profile     writes sigma_xx(0, a, z) through the thickness to prefix_<case>.csv

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "neumann_model.hpp"
#include "problems/plate_hole.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: plate_test <geometry.dat> <label> <case[,case...]> <t> [--n 12] "
                 "[--far-tol 1e-10] [--exact-data] [--cache dir|none] [--csv file] [--profile prefix]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2];
  const double thick = std::atof(argv[4]);
  std::vector<std::string> names;
  std::vector<PlateHole::Case> cases;
  {
    std::stringstream ss(argv[3]);
    for (std::string c; std::getline(ss, c, ',');) {
      PlateHole::Case k;
      if (!PlateHole::parse(c, k)) { std::cerr << "unknown case " << c << "\n"; return 1; }
      names.push_back(c);
      cases.push_back(k);
    }
  }
  PlateHole ph;
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  bool exact_data = false;
  std::string cache = "cache", csv, profile;
  for (int i = 5; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--exact-data")) exact_data = true;
    else if (!std::strcmp(argv[i], "--length")) ph.halfL = 0.5 * std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--profile")) profile = argv[++i];
  }
  const int K = int(cases.size());

  NeumannModel model;
  model.setup(geo, ph.mat, opt, cache);
  const Mesh& mesh = model.mesh;
  const auto hole = PlateHole::holePatches(mesh);
  auto data = [&](const Vec3& y, const Vec3& n) {
    TractionBlock T(3, K);
    for (int k = 0; k < K; ++k) T.col(k) = ph.traction(cases[k], y, n);
    return T;
  };

  const auto t0 = std::chrono::steady_clock::now();
  Eigen::MatrixXd UH;
  if (exact_data) {
    Eigen::MatrixXd A, B;
    SbieAssembler(mesh, ph.mat, opt).assembleTractions(data, K, A, B);
    UH = model.solveRhs(B);
  } else {
    UH = model.solve(data, K);
  }
  const double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("# %s (%s): %zu patches, %zu elements, %d dofs (%d unknowns), %d traction dofs, t = %g\n",
              label.c_str(), argv[3], mesh.patches.size(), mesh.elements.size(), mesh.ndof,
              model.unknowns(), 3 * model.ts.ndof, thick);
  std::printf("# volume %.15f (exact %.15f); operators assembled in %.2f s%s, LU %.2f s, %d case(s) %.3f s\n",
              mesh.volume(), (2.0 * ph.halfL - M_PI * ph.a * ph.a) * thick, model.t_assemble,
              model.from_cache ? (" (loaded from cache in " + std::to_string(model.t_load) + " s)").c_str() : "",
              model.t_factor, K, t_solve);
  const double rot = model.rotationResidual();
  std::printf("# n = %d, far_tol = %g, data %s, rotation residual %.3e\n", opt.n_gauss, opt.far_tol,
              exact_data ? "exact" : "interpolated", rot);
  std::printf("# %-12s %12s %12s %12s %12s\n", "case", "rel_L2_err", "sxx_mid", "sxx_face", "sxx_exact");

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    out.precision(15);
    if (fresh) out << "label,case,t,p,n,far_tol,data,ndof,unknowns,elements,rel_L2_err,rot_resid,"
                      "sxx_mid,sxx_face,sxx_exact,t_assemble,t_load,t_factor,t_solve,cached\n";
  }
  for (int k = 0; k < K; ++k) {
    const auto c = cases[k];
    const Eigen::VectorXd uh = UH.col(k);
    const double err = PlateHole::exact(c)
        ? displacementL2Error(mesh, uh, [&](const Vec3& x) { return ph.displacement(c, x); })
        : std::nan("");
    const double sm = ph.sxxAtHole(mesh, hole, c, uh, 0.0);
    const double sf = ph.sxxAtHole(mesh, hole, c, uh, 0.5 * thick);
    const double se = PlateHole::exact(c) ? ph.stress(c, Vec3(0.0, ph.a, 0.0))(0, 0) : std::nan("");
    std::printf("  %-12s %12.4e %12.8f %12.8f %12.8f\n", names[k].c_str(), err, sm, sf, se);
    if (out) out << label << ',' << names[k] << ',' << thick << ',' << mesh.patches[0].p << ','
                 << opt.n_gauss << ',' << opt.far_tol << ',' << (exact_data ? "exact" : "interp") << ','
                 << mesh.ndof << ',' << model.unknowns() << ',' << mesh.elements.size() << ',' << err
                 << ',' << rot << ',' << sm << ',' << sf << ',' << se << ',' << model.t_assemble << ','
                 << model.t_load << ',' << model.t_factor << ',' << t_solve << ',' << model.from_cache << '\n';
    if (!profile.empty()) {
      std::ofstream pf(profile + "_" + names[k] + ".csv");
      pf.precision(12);
      pf << "z_over_t,sxx,sxx_exact\n";
      for (int j = 0; j <= 40; ++j) {
        const double zt = -0.5 + j / 40.0;
        pf << zt << ',' << ph.sxxAtHole(mesh, hole, c, uh, zt * thick) << ',';
        if (PlateHole::exact(c)) pf << ph.stress(c, Vec3(0.0, ph.a, zt * thick))(0, 0);
        pf << '\n';
      }
    }
  }
  return 0;
}
