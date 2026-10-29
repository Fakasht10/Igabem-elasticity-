// Any closed part, e.g. imported from STEP (python/step_auto.py): thin driver for a
// patch test and a clamped bar in tension. Operators come from the library (assembled
// once, cached on disk); all cases are solved together.
//
// usage: cad_test <geometry.dat> <label> <case[,case...]> [--bc free|clamp] [--S 1]
//                 [--snom s] [--tol 1e-6] [--exclude d] [--region name:x0,y0,z0,x1,y1,z1]...
//                 [--n 12] [--cache dir|none] [--csv file] [--fields prefix]
//   cases: patch    u = alpha x, t = sigma n (exact; with --bc clamp, u prescribed on the
//                   clamped patches)
//          tension  t = S e_x on the end x = x_max (pointwise, |x - x_max| < tol),
//                   free elsewhere; needs --bc clamp
//   --bc clamp      displacement prescribed on the patches whose centre lies on x = x_min
//   --snom          nominal stress for the stress concentration factor (default S)
//   --exclude d     ignore surface points within d of the clamped end (its corners are
//                   singular) when taking the maximum
//   --region        also report the maximum inside an axis-aligned box (repeatable)
//   --fields        whole surface as prefix_<case>_surface.vtu

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
#include "probes.hpp"
#include "vtk.hpp"

using namespace igabem;

namespace {

struct Region {
  std::string name;
  Vec3 lo, hi;
  bool contains(const Vec3& x) const {
    return (x.array() >= lo.array()).all() && (x.array() <= hi.array()).all();
  }
};

// Largest principal stress over sub x sub points per element (and per region).
struct PeakStress {
  double s1 = -1e300;
  Vec3 at = Vec3::Zero();
  void take(double s, const Vec3& x) {
    if (s > s1) { s1 = s; at = x; }
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: cad_test <geometry.dat> <label> <case[,case...]> [--bc free|clamp] [--S 1] "
                 "[--snom s] [--tol 1e-6] [--exclude d] [--region name:x0,y0,z0,x1,y1,z1] [--n 12] "
                 "[--cache dir|none] [--csv file] [--fields prefix]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2];
  std::vector<std::string> names;
  {
    std::stringstream ss(argv[3]);
    for (std::string s; std::getline(ss, s, ',');) {
      if (s != "patch" && s != "tension") { std::cerr << "unknown case " << s << "\n"; return 1; }
      names.push_back(s);
    }
  }
  SolverOptions opt;
  bool clamp = false;
  double S = 1.0, snom = -1.0, tol = 1e-6, exclude = 0.0;
  std::string cache = "cache", csv, fields;
  std::vector<Region> regions;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bc")) clamp = !std::strcmp(argv[++i], "clamp");
    else if (!std::strcmp(argv[i], "--S")) S = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--snom")) snom = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--tol")) tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--exclude")) exclude = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--fields")) fields = argv[++i];
    else if (!std::strcmp(argv[i], "--region")) {
      std::string r = argv[++i];
      Region R;
      R.name = r.substr(0, r.find(':'));
      double v[6];
      if (std::sscanf(r.c_str() + r.find(':') + 1, "%lf,%lf,%lf,%lf,%lf,%lf", v, v + 1, v + 2, v + 3, v + 4,
                      v + 5) != 6) { std::cerr << "bad --region " << r << "\n"; return 1; }
      R.lo = Vec3(v[0], v[1], v[2]);
      R.hi = Vec3(v[3], v[4], v[5]);
      regions.push_back(R);
    }
  }
  if (snom <= 0.0) snom = S;
  for (const auto& n : names)
    if (n == "tension" && !clamp) { std::cerr << "tension needs --bc clamp\n"; return 1; }

  // Extent along x from the control points (the end faces are planar, so exact).
  double xmin = 1e300, xmax = -1e300;
  for (const auto& P : loadDat(geo))
    for (const Vec4& c : P.Pw) { xmin = std::min(xmin, c(0) / c(3)); xmax = std::max(xmax, c(0) / c(3)); }

  Material mat;  // lambda = mu = 1 (nu = 1/4)
  const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.25, -0.25, 1.0).asDiagonal();
  const Eigen::Matrix3d sigma = mat.stress(alpha);
  const int K = int(names.size());

  Model model;
  Model::Selector sel;
  if (clamp) sel = [&](int, const Vec3& c, const Vec3&) { return c(0) < xmin + tol; };
  model.setup(geo, mat, opt, sel, cache);
  const Mesh& mesh = model.mesh();
  auto traction = [&](const Vec3& y, const Vec3& n) {
    TractionBlock T(3, K);
    for (int k = 0; k < K; ++k)
      T.col(k) = names[k] == "patch" ? Vec3(sigma * n)
                                     : (y(0) > xmax - tol ? Vec3(S * n(0) * Vec3::UnitX()) : Vec3::Zero());
    return T;
  };
  auto displacement = [&](const Vec3& x) {
    TractionBlock U(3, K);
    for (int k = 0; k < K; ++k) U.col(k) = names[k] == "patch" ? Vec3(alpha * x) : Vec3::Zero();
    return U;
  };
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<BoundarySolution> sols = model.solutions(traction, displacement, K);
  const double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("# %s (%s, bc %s): %zu patches, %zu elements, %d dofs (%d unknowns), x in [%g, %g]\n",
              label.c_str(), argv[3], clamp ? "clamp" : "free", mesh.patches.size(), mesh.elements.size(),
              mesh.ndof, model.unknowns(), xmin, xmax);
  std::printf("# volume %.15f; operators assembled in %.2f s%s, LU %.2f s, %d case(s) %.3f s\n", mesh.volume(),
              model.tAssemble(), model.fromCache() ? " (from cache)" : "", model.tFactor(), K, t_solve);

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    out.precision(15);
    if (fresh) out << "label,case,bc,ndof,unknowns,elements,rel_L2_err,region,s1_max,scf,x,y,z,t_assemble,t_factor\n";
  }
  const int sub = 4;
  for (int k = 0; k < K; ++k) {
    const BoundarySolution& sol = sols[k];
    double err = std::nan("");
    if (names[k] == "patch")
      err = displacementL2Error(mesh, sol.u, [&](const Vec3& x) { return Vec3(alpha * x); }, 10, !clamp);
    PeakStress all;
    std::vector<PeakStress> peak(regions.size());
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
          const Vec3 x = sp.E.x;
          if (clamp && x(0) < xmin + exclude) continue;
          const double s1 = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(sol.stress(sp)).eigenvalues()(2);
          all.take(s1, x);
          for (size_t r = 0; r < regions.size(); ++r)
            if (regions[r].contains(x)) peak[r].take(s1, x);
        }
    std::printf("  %-8s rel_L2_err %11.4e  max s1 %.6f (SCF %.5f) at (%.4f, %.4f, %.4f)\n", names[k].c_str(), err,
                all.s1, all.s1 / snom, all.at(0), all.at(1), all.at(2));
    auto row = [&](const std::string& rn, const PeakStress& p) {
      if (out) out << label << ',' << names[k] << ',' << (clamp ? "clamp" : "free") << ',' << mesh.ndof << ','
                   << model.unknowns() << ',' << mesh.elements.size() << ',' << err << ',' << rn << ',' << p.s1
                   << ',' << p.s1 / snom << ',' << p.at(0) << ',' << p.at(1) << ',' << p.at(2) << ','
                   << model.tAssemble() << ',' << model.tFactor() << '\n';
    };
    row("all", all);
    for (size_t r = 0; r < regions.size(); ++r) {
      std::printf("  %-8s   %-10s max s1 %.6f (SCF %.5f) at (%.4f, %.4f, %.4f)\n", "", regions[r].name.c_str(),
                  peak[r].s1, peak[r].s1 / snom, peak[r].at(0), peak[r].at(1), peak[r].at(2));
      row(regions[r].name, peak[r]);
    }
    if (!fields.empty()) surfaceVtu(sol, 3).write(fields + "_" + names[k] + "_surface.vtu");
  }
  return 0;
}
