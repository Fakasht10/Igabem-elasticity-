#include <sys/resource.h>
// Any part with local supports and loads from a JSON case file (src/case_io.hpp):
// thin driver over BcModel. All cases share the cached operators and one LU.
//
// usage: bc_case <case.json> [--csv file] [--vtu prefix] [--sub 4] [--max-residual r]
//                [--points file --points-out prefix] [--serve] [--greville support file]
//
// Prints, per case: the applied resultant of every load, the reaction of every
// support and spring, the global equilibrium residual |sum of all boundary
// tractions| / |applied|, the work of the loads on any rigid motion the supports
// leave free (refused above 1e-4, as an unbalanced pure-traction problem has no
// solution), max |u| and the peak von Mises and first principal stress on the
// surface (sub x sub points per element). --max-residual: exit 4 when any case's
// equilibrium residual is above r (regression tests).
//
// --points: stress at arbitrary points for post-processing (the failure layer).
// One point per line, "x y z [s]": s = 1 takes the boundary recovery at the
// closest surface point (surface nodes of a tetrahedral mesh), otherwise the
// near-boundary-safe interior evaluation. Writes prefix_<case>.txt per case:
// x y z sxx syy szz sxy syz szx winding gap, with x as given; winding is 1 inside,
// 0 outside (0.5 for surface points) and gap is |x - projection| for surface points.
// From 1,000 points on, interior points go through the treecode (src/treecode.hpp).
// --serve: after the solve, read commands from stdin ("points <file> <prefix>", "quit"),
// answering "ready" after each, so later probes reuse the solution without a new LU.
// "displacement <file> <prefix>" writes prefix_<case>.txt: x y z ux uy uz winding (direct
// evaluation; s = 1 takes the boundary value at the closest surface point).
// --greville: write the points a support's patches are interpolated at ("x y z" per line) and
// exit without a solve: where a "field" for that support must give its displacement.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#include "case_io.hpp"
#include "interior.hpp"
#include "treecode.hpp"
#include "vtk.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bc_case <case.json> [--csv file] [--vtu prefix] [--sub 4] [--max-residual r]\n";
    return 1;
  }
  int sub = 4;
  std::string csv_arg, vtu_arg, pts_arg, pts_out, grev_label, grev_out;
  double max_res = -1.0;
  bool serve = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--csv")) csv_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--vtu")) vtu_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--sub")) sub = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--max-residual")) max_res = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--points")) pts_arg = argv[++i];
    else if (!std::strcmp(argv[i], "--points-out")) pts_out = argv[++i];
    else if (!std::strcmp(argv[i], "--serve")) serve = true;
    else if (!std::strcmp(argv[i], "--greville") && i + 2 < argc) { grev_label = argv[++i]; grev_out = argv[++i]; }
  }
  if (!pts_arg.empty() && pts_out.empty()) {
    std::cerr << "error: --points needs --points-out\n";
    return 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  CaseFile cf;
  BcModel model;
  std::vector<LoadCase> cases;
  cf.need_fields = grev_label.empty();
  try {
    cf.load(argv[1], model, cases);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  if (!grev_label.empty()) {
    const FixedSupport* s = nullptr;
    for (const auto& f : model.supports)
      if (f.label == grev_label) s = &f;
    if (!s) {
      std::cerr << "error: no support labelled " << grev_label << "\n";
      return 2;
    }
    std::ofstream o(grev_out);
    o.precision(17);
    size_t n = 0;
    for (int p : s->patches) {  // as BcModel::interpolatePatch
      const Patch& P = model.mesh.patches[p];
      PatchEval E;
      for (int k = 0; k < P.nu * P.nv; ++k, ++n) {
        P.eval(P.greville(k % P.nu, true), P.greville(k / P.nu, false), E);
        o << E.x(0) << ' ' << E.x(1) << ' ' << E.x(2) << '\n';
      }
    }
    std::printf("# %zu interpolation points of %s (%zu patches) -> %s\n", n, grev_label.c_str(), s->patches.size(),
                grev_out.c_str());
    return 0;
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
  std::vector<BoundarySolution> sols;
  try {
    sols = model.solutions(cases, &rep);
  } catch (const std::exception& e) {  // a field file without a value at an interpolation point
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
  const double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  std::printf("# %zu case(s) solved in %.2f s (region right-hand sides included)\n", cases.size(), t_solve);
  if (const HMixed* h = model.hmat()) {
    const auto& st = h->st;
    const auto& hs = h->op().stats();
    std::printf("# hmat eps %g: operator %.2f s (near %.2f GB, far %.2f GB, rank mean %.1f), preconditioner %.2f s "
                "(%ld nonzeros, coarse %d), GMRES %.2f s (products %.2f, preconditioner %.2f), iterations", h->options().h.eps,
                st.t_build, hs.bytes_near / 1e9, hs.bytes_far / 1e9, hs.rank_mean, st.t_precond, st.nnz,
                st.coarse_dim, st.t_solve, st.t_matvec, st.t_pc);
    for (size_t k = 0; k < st.iterations.size(); ++k) std::printf(" %d (%.1e)", st.iterations[k], st.residuals[k]);
    rusage u;
    getrusage(RUSAGE_SELF, &u);
    std::printf("; peak RSS %.2f GB\n", u.ru_maxrss / 1048576.0);
  }

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
    for (const auto& [n, q] : r.master_motion)
      std::printf("  master    %-22s u (%11.4e, %11.4e, %11.4e)  theta (%11.4e, %11.4e, %11.4e)\n", n.c_str(), q(0),
                  q(1), q(2), q(3), q(4), q(5));
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
  auto readPoints = [](const std::string& in_path, std::vector<Vec3>& X, std::vector<char>& on) {
    std::ifstream in(in_path);
    std::string row;
    while (std::getline(in, row)) {
      if (row.empty() || row[0] == '#') continue;
      double v[4] = {0, 0, 0, 0};
      const int n = std::sscanf(row.c_str(), "%lf %lf %lf %lf", v, v + 1, v + 2, v + 3);
      if (n < 3) continue;
      X.emplace_back(v[0], v[1], v[2]);
      on.push_back(n == 4 && v[3] != 0.0);
    }
  };
  // Displacement at the points of a file, for every case: a submodel's cut faces from this (global) solve.
  auto displacement = [&](const std::string& in_path, const std::string& prefix) {
    std::vector<Vec3> X;
    std::vector<char> on;
    readPoints(in_path, X, on);
    for (size_t k = 0; k < cases.size(); ++k) {
      InteriorEvaluator ev(sols[k], cf.opt);
      const auto F = ev.eval(X, on);
      std::ofstream o(prefix + "_" + rep[k].name + ".txt");
      o.precision(17);
      o << "# x y z ux uy uz winding\n";
      for (size_t i = 0; i < X.size(); ++i)
        o << X[i](0) << ' ' << X[i](1) << ' ' << X[i](2) << ' ' << F[i].u(0) << ' ' << F[i].u(1) << ' ' << F[i].u(2)
          << ' ' << F[i].winding << '\n';
    }
  };
  // Stress at the points of a file, for every case (see --points above).
  auto points = [&](const std::string& in_path, const std::string& prefix) {
    std::vector<Vec3> X;
    std::vector<char> on;
    readPoints(in_path, X, on);
    for (size_t k = 0; k < cases.size(); ++k) {
      const auto tp = std::chrono::steady_clock::now();
      // many points: the treecode (src/treecode.hpp, ~1e-7 of the peak stress); few: direct
      std::vector<FieldSample> F;
      if (X.size() >= 1000) {
        TreeEvaluator tree(sols[k], cf.opt);
        F = tree.eval(X, on);
      } else {
        InteriorEvaluator ev(sols[k], cf.opt);
        F = ev.eval(X, on);
      }
      std::ofstream o(prefix + "_" + rep[k].name + ".txt");
      o.precision(12);
      o << "# x y z sxx syy szz sxy syz szx winding gap\n";
      for (size_t i = 0; i < X.size(); ++i) {
        const auto& S = F[i].stress;
        o << X[i](0) << ' ' << X[i](1) << ' ' << X[i](2) << ' ' << S(0, 0) << ' ' << S(1, 1) << ' ' << S(2, 2) << ' '
          << S(0, 1) << ' ' << S(1, 2) << ' ' << S(2, 0) << ' ' << F[i].winding << ' '
          << (on[i] ? (F[i].x - X[i]).norm() : 0.0) << '\n';
      }
      std::printf("# case %s: stress at %zu points in %.2f s\n", rep[k].name.c_str(), X.size(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    }
    std::fflush(stdout);
  };
  if (!pts_arg.empty()) points(pts_arg, pts_out);
  if (serve) {
    // Keep the solution: "points <file> <prefix>" evaluates and answers "ready"; "quit" ends.
    std::printf("ready\n");
    std::fflush(stdout);
    std::string cmd, a, b;
    while (std::cin >> cmd) {
      if (cmd == "quit") break;
      if (cmd == "points" && (std::cin >> a >> b)) points(a, b);
      else if (cmd == "displacement" && (std::cin >> a >> b)) displacement(a, b);
      else std::printf("# unknown command %s\n", cmd.c_str());
      std::printf("ready\n");
      std::fflush(stdout);
    }
  }
  std::printf("# end to end %.2f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return status;
}
