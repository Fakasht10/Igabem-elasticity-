// Plate with a circular through-hole (src/problems/plate_hole.hpp).
// Thin driver: the operators come from the library (assembled once per mesh and
// quadrature setting, then cached on disk), all load cases are solved together.
//
// usage: plate_test <geometry.dat> <label> <case[,case...]> <t> [--bc free|clamp] [--n 12]
//                   [--far-tol 1e-10] [--exact-data] [--length L] [--cache dir|none] [--csv file]
//                   [--profile prefix] [--fields prefix]
//   cases: patch, kirsch, kirsch_free, tension
//   --bc clamp    displacement prescribed on the end face x = -L/2 (the exact field
//                 for patch and kirsch, zero for kirsch_free and tension), traction
//                 elsewhere; default free: traction everywhere
//   --exact-data  integrate the traction data exactly instead of interpolating it
//                 in the traction space (validation of the interpolation; bc free only)
//   --profile     writes sigma_xx(0, a, z) through the thickness to prefix_<case>.csv
//   --fields      canned probes and VTK output per case (prefix_<case>_*.csv, *.vtu):
//                 ligament (0, a, 0) -> (0, W/2, 0) and bisector at 45 degrees, both
//                 through the solid at z = 0; around the hole at z = 0 and on the face,
//                 and up the hole wall at 90 degrees; whole surface and the mid-plane
//                 section as .vtu

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "model.hpp"
#include "problems/plate_hole.hpp"
#include "probes.hpp"
#include "vtk.hpp"

using namespace igabem;

static void writeFields(const std::string& prefix, const PlateHole& ph, PlateHole::Case c,
                        const BoundarySolution& sol, const std::vector<bool>& hole, double thick,
                        const SolverOptions& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  const ExactField ex = ph.exactField(c);
  const double a = ph.a;
  InteriorEvaluator ev(sol, opt);
  writeProbeCsv(prefix + "_ligament.csv", lineProbe(ev, Vec3(0.0, a, 0.0), Vec3(0.0, ph.half, 0.0), 60), ex);
  const Vec3 d45 = Vec3(1.0, 1.0, 0.0).normalized();
  writeProbeCsv(prefix + "_bisector.csv", lineProbe(ev, a * d45, ph.outer(M_PI / 4), 60), ex);
  auto onHole = [&](int p) { return bool(hole[p]); };
  for (const auto& [name, z] : {std::pair<const char*, double>{"mid", 0.0}, {"face", 0.5 * thick}}) {
    const double zz = z;
    const std::function<Vec3(double)> arc = [&, zz](double s) {
      return Vec3(a * std::cos(M_PI * s), a * std::sin(M_PI * s), zz);
    };
    writeProbeCsv(prefix + "_hole_" + name + ".csv", boundaryProbe(sol, arc, 90, onHole), ex);
  }
  const std::function<Vec3(double)> wall = [&](double s) { return Vec3(0.0, a, (s - 0.5) * thick); };
  writeProbeCsv(prefix + "_hole_z.csv", boundaryProbe(sol, wall, 40, onHole), ex);
  surfaceVtu(sol, 3, ex).write(prefix + "_surface.vtu");
  // Mid-plane section: rays from the hole to the outer boundary, graded towards the hole.
  auto section = [&](double s, double r) {
    const double th = 2.0 * M_PI * s, g = r * r;
    return Vec3((1.0 - g) * a * Vec3(std::cos(th), std::sin(th), 0.0) + g * ph.outer(th));
  };
  gridVtu(ev, section, 96, 24, ex).write(prefix + "_section.vtu");
  std::printf("#   fields -> %s_* in %.2f s\n", prefix.c_str(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: plate_test <geometry.dat> <label> <case[,case...]> <t> [--bc free|clamp] [--n 12] "
                 "[--far-tol 1e-10] [--exact-data] [--length L] [--cache dir|none] [--csv file] "
                 "[--profile prefix] [--fields prefix]\n";
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
  bool exact_data = false, clamp = false;
  std::string cache = "cache", csv, profile, fields;
  for (int i = 5; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--exact-data")) exact_data = true;
    else if (!std::strcmp(argv[i], "--bc")) clamp = !std::strcmp(argv[++i], "clamp");
    else if (!std::strcmp(argv[i], "--length")) ph.halfL = 0.5 * std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--profile")) profile = argv[++i];
    else if (!std::strcmp(argv[i], "--fields")) fields = argv[++i];
  }
  if (exact_data && clamp) { std::cerr << "--exact-data needs --bc free\n"; return 1; }
  const int K = int(cases.size());

  Model model;
  Model::Selector sel;
  if (clamp) sel = [&](int, const Vec3& c, const Vec3&) { return ph.clampedEnd(c); };
  model.setup(geo, ph.mat, opt, sel, cache);
  const Mesh& mesh = model.mesh();
  const auto hole = PlateHole::holePatches(mesh);
  auto traction = [&](const Vec3& y, const Vec3& n) {
    TractionBlock T(3, K);
    for (int k = 0; k < K; ++k) T.col(k) = ph.traction(cases[k], y, n);
    return T;
  };
  auto displacement = [&](const Vec3& x) {
    TractionBlock U(3, K);
    for (int k = 0; k < K; ++k)
      U.col(k) = PlateHole::exact(cases[k]) ? ph.displacement(cases[k], x) : Vec3::Zero();
    return U;
  };

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<BoundarySolution> sols;
  if (exact_data) {
    Eigen::MatrixXd A, B;
    SbieAssembler(mesh, ph.mat, opt).assembleTractions(traction, K, A, B);
    const Eigen::MatrixXd UH = model.neumann()->solveRhs(B);
    const Eigen::MatrixXd TC = model.ts().interpolate(traction, K);  // for stress recovery only
    for (int k = 0; k < K; ++k) sols.push_back({&mesh, &model.ts(), ph.mat, UH.col(k), TC.col(k)});
  } else {
    sols = model.solutions(traction, displacement, K);
  }
  const double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("# %s (%s, bc %s): %zu patches, %zu elements, %d dofs (%d unknowns), %d traction dofs, t = %g\n",
              label.c_str(), argv[3], clamp ? "clamp" : "free", mesh.patches.size(), mesh.elements.size(),
              mesh.ndof, model.unknowns(), 3 * model.ts().ndof, thick);
  std::printf("# volume %.15f (exact %.15f); operators assembled in %.2f s%s, LU %.2f s, %d case(s) %.3f s\n",
              mesh.volume(), (2.0 * ph.halfL - M_PI * ph.a * ph.a) * thick, model.tAssemble(),
              model.fromCache() ? (" (loaded from cache in " + std::to_string(model.tLoad()) + " s)").c_str() : "",
              model.tFactor(), K, t_solve);
  const double rot = model.neumann() ? model.neumann()->rotationResidual() : std::nan("");
  std::printf("# n = %d, far_tol = %g, data %s, rotation residual %.3e\n", opt.n_gauss, opt.far_tol,
              exact_data ? "exact" : "interpolated", rot);
  std::printf("# %-12s %12s %12s %12s %12s\n", "case", "rel_L2_err", "sxx_mid", "sxx_face", "sxx_exact");

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    out.precision(15);
    if (fresh) out << "label,case,bc,t,p,n,far_tol,data,ndof,unknowns,elements,rel_L2_err,rot_resid,"
                      "sxx_mid,sxx_face,sxx_exact,t_assemble,t_load,t_factor,t_solve,cached\n";
  }
  auto onHole = [&](int p) { return bool(hole[p]); };
  for (int k = 0; k < K; ++k) {
    const auto c = cases[k];
    const BoundarySolution& S = sols[k];
    // Rigid motion is undetermined (and removed from the error) only without Dirichlet data.
    const double err = PlateHole::exact(c)
        ? displacementL2Error(mesh, S.u, [&](const Vec3& x) { return ph.displacement(c, x); }, 10, !clamp)
        : std::nan("");
    auto sxx = [&](double z) { return S.stress(locate(mesh, Vec3(0.0, ph.a, z), onHole))(0, 0); };
    const double sm = sxx(0.0), sf = sxx(0.5 * thick);
    const double se = PlateHole::exact(c) ? ph.stress(c, Vec3(0.0, ph.a, 0.0))(0, 0) : std::nan("");
    std::printf("  %-12s %12.4e %12.8f %12.8f %12.8f\n", names[k].c_str(), err, sm, sf, se);
    if (out) out << label << ',' << names[k] << ',' << (clamp ? "clamp" : "free") << ',' << thick << ','
                 << mesh.patches[0].p << ',' << opt.n_gauss << ',' << opt.far_tol << ','
                 << (exact_data ? "exact" : "interp") << ',' << mesh.ndof << ',' << model.unknowns() << ','
                 << mesh.elements.size() << ',' << err << ',' << rot << ',' << sm << ',' << sf << ',' << se
                 << ',' << model.tAssemble() << ',' << model.tLoad() << ',' << model.tFactor() << ','
                 << t_solve << ',' << model.fromCache() << '\n';
    if (!profile.empty()) {
      std::ofstream pf(profile + "_" + names[k] + ".csv");
      pf.precision(12);
      pf << "z_over_t,sxx,sxx_exact\n";
      for (int j = 0; j <= 40; ++j) {
        const double zt = -0.5 + j / 40.0;
        pf << zt << ',' << sxx(zt * thick) << ',';
        if (PlateHole::exact(c)) pf << ph.stress(c, Vec3(0.0, ph.a, zt * thick))(0, 0);
        pf << '\n';
      }
    }
    if (!fields.empty()) writeFields(fields + "_" + names[k], ph, c, S, hole, thick, opt);
  }
  return 0;
}
