// Any part with local supports and loads from a JSON case file (src/case_io.hpp):
// thin driver over BcModel. All cases share the cached operators and one LU.
//
// usage: bc_case <case.json> [--csv file] [--vtu prefix] [--sub 4] [--max-residual r]
//
// Prints, per case: the applied resultant of every load, the reaction of every
// support and spring, the global equilibrium residual |sum of all boundary
// tractions| / |applied|, the work of the loads on any rigid motion the supports
// leave free (refused above 1e-4, as an unbalanced pure-traction problem has no
// solution), max |u| and the peak von Mises and first principal stress on the
// surface (sub x sub points per element). --max-residual: exit 4 when any case's
// equilibrium residual is above r (regression tests).

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "case_io.hpp"
#include "vtk.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bc_case <case.json> [--csv file] [--vtu prefix] [--sub 4] [--max-residual r]\n";
    return 1;
  }
  int sub = 4;
  std::string csv_arg, vtu_arg;
  double max_res = -1.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--csv")) csv_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--vtu")) vtu_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--sub")) sub = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-residual")) max_res = std::atof(argv[++i]);
  }
  const auto t0 = std::chrono::steady_clock::now();
  CaseFile cf;
  BcModel model;
  std::vector<LoadCase> cases;
  try {
    cf.load(argv[1], model, cases);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  if (!csv_arg.empty()) cf.csv = csv_arg;
  if (!vtu_arg.empty()) cf.vtu = vtu_arg;
  const double t_setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const Mesh& mesh = model.mesh;
  std::printf("# %s: %zu patches, %zu elements, %d dofs, %d unknowns, %d free rigid motion(s)\n", argv[1],
              mesh.patches.size(), mesh.elements.size(), mesh.ndof, model.unknowns(), model.freeModes());
  std::printf("# E %g nu %g; operators %.2f s%s, springs %.2f s, LU %.2f s, setup total %.2f s\n", cf.E, cf.nu,
              model.t_assemble, model.from_cache ? " (from cache)" : "", model.t_springs, model.t_factor, t_setup);
  for (const auto& s : model.supports) {
    std::printf("#   support %-14s fix %c%c%c on %zu patch(es)\n", s.label.c_str(), s.fix[0] ? 'x' : '-',
                s.fix[1] ? 'y' : '-', s.fix[2] ? 'z' : '-', s.patches.size());
  }
  for (const auto& s : model.springs)
    std::printf("#   spring  %-14s area %.6g on %zu element(s)\n", s.label.c_str(), s.region->area, s.region->cells.size());

  const auto t1 = std::chrono::steady_clock::now();
  std::vector<CaseReport> rep;
  const auto sols = model.solutions(cases, &rep);
  const double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  std::printf("# %zu case(s) solved in %.2f s (region right-hand sides included)\n", cases.size(), t_solve);

  std::ofstream csv;
  if (!cf.csv.empty()) {
    csv.open(cf.csv);
    csv.precision(12);
    csv << "case,item,kind,Fx,Fy,Fz,Mx,My,Mz\n";
  }
  auto line = [&](const std::string& cs, const std::string& item, const char* kind, const Wrench& W) {
    std::printf("  %-9s %-22s F (%11.4e, %11.4e, %11.4e)  M (%11.4e, %11.4e, %11.4e)\n", kind, item.c_str(), W.F(0),
                W.F(1), W.F(2), W.M(0), W.M(1), W.M(2));
    if (csv) csv << cs << ',' << item << ',' << kind << ',' << W.F(0) << ',' << W.F(1) << ',' << W.F(2) << ','
                 << W.M(0) << ',' << W.M(1) << ',' << W.M(2) << '\n';
  };
  int status = 0;
  for (size_t k = 0; k < cases.size(); ++k) {
    const CaseReport& r = rep[k];
    std::printf("case %s\n", r.name.c_str());
    for (const auto& [n, W] : r.loads) line(r.name, n, "load", W);
    for (const auto& [n, W] : r.reactions) line(r.name, n, "reaction", W);
    std::printf("  equilibrium residual |sum of boundary tractions| / |applied| = %.3e\n", r.residual(cf.size));
    if (csv) csv << r.name << ",equilibrium,residual," << r.residual(cf.size) << ",,,,,\n";
    if (max_res >= 0 && r.residual(cf.size) > max_res) {
      std::printf("  FAIL: equilibrium residual above %.3e\n", max_res);
      if (status == 0) status = 4;
    }
    if (model.freeModes() > 0) {
      std::printf("  work on free rigid motions (relative) = %.3e\n", r.free_work);
      if (r.free_work > 1e-4) {
        std::printf("  REFUSED: the loads are not balanced on a rigid motion the supports leave free; hold it\n");
        status = 3;
      }
    }
    double umax = 0.0, vm = 0.0, s1 = -1e300;
    Vec3 xvm = Vec3::Zero(), xs1 = Vec3::Zero();
    for (const Element& e : mesh.elements)
      for (int j = 0; j <= sub; ++j)
        for (int i = 0; i <= sub; ++i) {
          SurfacePoint sp;
          sp.patch = e.patch;
          sp.su = e.su;
          sp.sv = e.sv;
          sp.xi = Vec2(e.rect.u0 + e.rect.du() * i / sub, e.rect.v0 + e.rect.dv() * j / sub);
          mesh.patches[e.patch].evalSpan(e.su, e.sv, sp.xi(0), sp.xi(1), sp.E);
          sp.N = (sp.E.normal() * mesh.normal_sign).normalized();
          umax = std::max(umax, sols[k].displacement(sp).norm());
          const Eigen::Matrix3d S = sols[k].stress(sp);
          const double v = vonMises(S), p1 = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(S).eigenvalues()(2);
          if (v > vm) { vm = v; xvm = sp.E.x; }
          if (p1 > s1) { s1 = p1; xs1 = sp.E.x; }
        }
    std::printf("  max |u| %.6e; max von Mises %.6e at (%.4f, %.4f, %.4f); max s1 %.6e at (%.4f, %.4f, %.4f)\n", umax,
                vm, xvm(0), xvm(1), xvm(2), s1, xs1(0), xs1(1), xs1(2));
    if (csv) csv << r.name << ",peak,von_mises," << vm << ',' << xvm(0) << ',' << xvm(1) << ',' << xvm(2) << ",,\n"
                 << r.name << ",peak,s1," << s1 << ',' << xs1(0) << ',' << xs1(1) << ',' << xs1(2) << ",,\n"
                 << r.name << ",peak,u_max," << umax << ",,,,,\n";
    if (!cf.vtu.empty()) surfaceVtu(sols[k], 3).write(cf.vtu + "_" + r.name + "_surface.vtu");
  }
  std::printf("# end to end %.2f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return status;
}
