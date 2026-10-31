// Bead-boundary periodic cell (src/bead_cell.hpp): one BEM region per bead, joined by weld laws, PBC.
//
// usage: bead_cell <bead.dat> [<bead2.dat> ...] [--copy i:j:rz] [--E 2200] [--nu 0.35] [--ng 12]
//                  [--weld a:Kn:Kt] [--weld-int a:Kn:Kt] [--cache dir|none] [--json out.json]
//
//   --copy i:j:rz  region i is region j turned by rz degrees about z (and translated): its operator is region j's
//                  rotated, no assembly (python/rve/beads.py bead090: B = A turned 90 deg)
//   --axes x,y     filament axis of each region (the periodic pair normal to it continues the bead: never a weld;
//                  default none: set it with --weld on a cell whose beads cross the box faces)
//   --weld a:Kn:Kt      linear traction-separation law (N/mm^3) on the periodic face pairs normal to axis a
//   --weld-int a:Kn:Kt  the same on the internal welds (coincident faces of two regions) normal to axis a
//                       Kn <= 0: perfect weld (default everywhere)
//   --crack P|I:a:r:c=v[,c=v]:lim  debond on the periodic (P) or internal (I) welds normal to axis a with + side in
//                  region r (-1: any): cracked where max |x_c - v| > lim (src/bead_cell.hpp WeldCrack); repeatable
//   --scales s1,s2,... every weld compliance times s (0: perfect bond), one solve per entry sharing one LU
//                  (BeadCell::solveScales; JSON {"runs": [...]}, tractions f.0, f.1, ...): a kappa sweep in one run
//   --tractions f  weld tractions of the six unit macro strains at Gauss points of every weld (BeadCell::weldTractions)
//   JSON: C (Voigt 11 22 33 23 13 12, engineering shear), engineering constants, porosity, timings, Hill-Mandel.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <tuple>

#include "bead_cell.hpp"
#include "multi_region.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  std::vector<std::string> files;
  std::vector<std::tuple<int, int, double>> copies;
  std::vector<std::tuple<bool, int, double, double>> welds;
  double E = 2200.0, nu = 0.35;
  int ng = 12;
  std::string cache = "none", json, axes, tractions;
  std::vector<double> scales{1.0};
  std::vector<WeldCrack> cracks;
  auto weld = [](const char* s, bool internal) {
    double Kn, Kt;
    char a;
    if (std::sscanf(s, "%c:%lf:%lf", &a, &Kn, &Kt) != 3) throw std::runtime_error("bad --weld value");
    return std::make_tuple(internal, int(a - 'x'), Kn, Kt);
  };
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return argv[++i]; };
    if (argv[i][0] != '-') files.push_back(argv[i]);
    else if (!std::strcmp(argv[i], "--E")) E = std::stod(next());
    else if (!std::strcmp(argv[i], "--nu")) nu = std::stod(next());
    else if (!std::strcmp(argv[i], "--ng")) ng = std::stoi(next());
    else if (!std::strcmp(argv[i], "--cache")) cache = next();
    else if (!std::strcmp(argv[i], "--json")) json = next();
    else if (!std::strcmp(argv[i], "--axes")) axes = next();
    else if (!std::strcmp(argv[i], "--tractions")) tractions = next();
    else if (!std::strcmp(argv[i], "--scales")) {
      scales.clear();
      std::stringstream ss(next());
      for (std::string t; std::getline(ss, t, ',');) scales.push_back(std::stod(t));
    }
    else if (!std::strcmp(argv[i], "--crack")) {
      std::stringstream ss(next());
      std::string f[5];
      for (auto& x : f) std::getline(ss, x, ':');
      WeldCrack c;
      c.internal = f[0] == "I";
      c.axis = f[1][0] - 'x';
      c.region = std::stoi(f[2]);
      std::stringstream cs(f[3]);
      for (std::string t; std::getline(cs, t, ',');) c.centre.push_back({t[0] - 'x', std::stod(t.substr(2))});
      c.limit = std::stod(f[4]);
      cracks.push_back(c);
    }
    else if (!std::strcmp(argv[i], "--weld")) welds.push_back(weld(next(), false));
    else if (!std::strcmp(argv[i], "--weld-int")) welds.push_back(weld(next(), true));
    else if (!std::strcmp(argv[i], "--copy")) {
      int a, b;
      double rz;
      if (std::sscanf(next(), "%d:%d:%lf", &a, &b, &rz) != 3) { std::fprintf(stderr, "bad --copy\n"); return 2; }
      copies.push_back({a, b, rz});
    } else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }
  if (files.empty()) {
    std::fprintf(stderr, "usage: bead_cell <bead.dat> [...] [--copy i:j:rz] [--weld a:Kn:Kt] [--json f]\n");
    return 2;
  }
  std::vector<std::pair<int, Eigen::Matrix3d>> cp(files.size(), {-1, Eigen::Matrix3d::Identity()});
  for (auto [i, j, rz] : copies) {
    const double c = std::cos(rz * M_PI / 180), s = std::sin(rz * M_PI / 180);
    Eigen::Matrix3d R;
    R << c, -s, 0, s, c, 0, 0, 0, 1;
    cp.at(i) = {j, R};
  }
  SolverOptions o;
  o.n_gauss = ng;
  o.far_tol = 1e-10;
  BeadCell m;
  const auto t0 = std::chrono::steady_clock::now();
  for (auto [internal, a, Kn, Kt] : welds) {
    (internal ? m.weld_internal : m.weld_periodic)[a] = BeadCell::springCompliance(a, Kn, Kt);
    std::printf("%s weld normal to %c: Kn %.4g Kt %.4g N/mm^3\n", internal ? "internal" : "periodic", 'x' + a, Kn, Kt);
  }
  std::vector<int> ax;
  for (size_t i = 0; i < axes.size(); i += 2) ax.push_back(axes[i] - 'x');
  m.setup(files, materialEnu(E, nu), o, cp, cache == "none" ? "" : cache, ax);
  m.cracks = cracks;
  const double t_setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const Vec3 L = m.lengths();
  int nper = 0, nint = 0;
  for (auto& p : m.pairs) (p.internal ? nint : nper)++;
  std::printf("%zu regions, %d dofs, box %.4g x %.4g x %.4g, porosity %.4f%%, %d periodic + %d internal weld pairs\n",
              m.regions.size(), m.ndofTotal(), L(0), L(1), L(2), 100 * m.porosity(), nper, nint);
  for (auto& r : m.regions)
    std::printf("  %s: %zu patches, %d dofs (%zu on welds, %d eliminated unknowns), %s %.2f s, condense %.2f s\n", r.file.c_str(), r.mesh.patches.size(), r.mesh.ndof, r.bdof.size(), 3 * r.ne,
                r.copy_of >= 0 ? "rotated copy" : r.reduced_from_cache ? "cached" : "assembly", r.t_assemble, r.t_reduce);
  std::printf("setup %.2f s\n", t_setup);
  const auto t1 = std::chrono::steady_clock::now();
  const std::vector<RveResult> runs = m.solveScales(scales);
  const double t_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  std::printf("PBC: %d unknowns, %zu weld scale(s), factor %.2f s, all solves %.2f s\n", runs[0].unknowns,
              scales.size(), runs[0].t_factor, t_all);
  auto num = [](double x) { return std::isfinite(x) ? std::to_string(x) : std::string("null"); };
  std::ostringstream all;
  all.precision(10);
  for (size_t q = 0; q < runs.size(); ++q) {
    const RveResult& r = runs[q];
    const Engineering e = engineering(r.C);
    std::printf("scale %g: Hill-Mandel %.1e, asymmetry %.1e\n", scales[q], r.hill_mandel, r.asymmetry);
    for (int i = 0; i < 6; ++i) {
      std::printf("  ");
      for (int j = 0; j < 6; ++j) std::printf("%11.3f", r.C(i, j));
      std::printf("\n");
    }
    std::printf("  E1 %.2f  E2 %.2f  E3 %.2f  G23 %.2f  G13 %.2f  G12 %.2f  nu12 %.4f  nu13 %.4f  nu23 %.4f\n", e.E1,
                e.E2, e.E3, e.G23, e.G13, e.G12, e.nu12, e.nu13, e.nu23);
    if (!tractions.empty()) {
      std::ofstream os(scales.size() == 1 ? tractions : tractions + "." + std::to_string(q));
      m.weldTractions(r, os);
    }
    std::ostringstream js;
    js.precision(10);
    js << "{\"E\": " << E << ", \"nu\": " << nu << ", \"scale\": " << scales[q] << ", \"porosity\": " << m.porosity()
       << ", \"box\": [" << L(0) << ", " << L(1) << ", " << L(2) << "], \"dofs\": " << m.ndofTotal()
       << ", \"unknowns\": " << r.unknowns << ", \"t_setup\": " << t_setup << ", \"t_solve\": " << t_all
       << ", \"hill_mandel\": " << r.hill_mandel << ", \"C\": [";
    for (int i = 0; i < 6; ++i) {
      js << (i ? ", " : "") << "[";
      for (int j = 0; j < 6; ++j) js << (j ? ", " : "") << r.C(i, j);
      js << "]";
    }
    js << "], \"eng\": {\"E1\": " << num(e.E1) << ", \"E2\": " << num(e.E2) << ", \"E3\": " << num(e.E3)
       << ", \"G23\": " << num(e.G23) << ", \"G13\": " << num(e.G13) << ", \"G12\": " << num(e.G12)
       << ", \"nu12\": " << num(e.nu12) << ", \"nu13\": " << num(e.nu13) << ", \"nu23\": " << num(e.nu23) << "}}";
    all << (q ? ",\n " : "") << js.str();
  }
  if (!json.empty()) {
    std::ofstream f(json);
    if (scales.size() == 1) f << all.str() << "\n";                       // one object (as before)
    else f << "{\"runs\": [" << all.str() << "]}\n";                     // one per --scales entry
  }
  return 0;
}
