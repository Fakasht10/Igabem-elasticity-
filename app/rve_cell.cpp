// Effective stiffness of a periodic cell (src/rve.hpp): one assembly, KUBC / PBC / SUBC.
//
// usage: rve_cell <cell.dat> [--E 2200] [--nu 0.35] [--bc kubc,pbc,subc] [--ng 10]
//                 [--cache dir|none] [--json out.json] [--check-iso tol] [--check-c c11,c33,c44,c66 --tol t]
//
//   --check-iso tol  regression (ctest): every C must equal the solid's isotropic C to tol (a cell with no pores)
//   --check-c        regression: C11, C33, C44, C66 of the last BC against committed values, relative tol
//   --points f   interior stress (Somigliana, near-boundary safe) at the points of f (x y z per line) for the six
//                unit macro strains of the last BC: JSON "points": stress[case][point] (Voigt), for bond-plane
//                tractions (python/rve/strength.py)
//   --save-solutions f  binary boundary solutions of the six unit strain cases of the last BC (reuse without
//                re-solving: interior fields, strain-concentration moments). Layout: "RVESOL1\0", int32 ndof,
//                int32 traction ndof, int32 ncase (6), float64 E, nu, then per case u (3 ndof), t (3 traction ndof);
//                case k is the unit macro strain e_k (Voigt 11 22 33 23 13 12, engineering shear)
//   --load-solutions f  skip assembly and solve: read f (same cell, same mesh options) and only evaluate --points
//   JSON: per BC the 6x6 C (Voigt 11 22 33 23 13 12, engineering shear), engineering constants, timings.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "interior.hpp"
#include "multi_region.hpp"
#include "rve.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: rve_cell <cell.dat> [--E e] [--nu v] [--bc kubc,pbc,subc] [--json f]\n");
    return 2;
  }
  const std::string geo = argv[1];
  double E = 2200.0, nu = 0.35, iso_tol = -1, ctol = 1e-6;
  std::string bcs = "kubc,pbc", cache = "none", json, points, save_sol, load_sol;
  std::vector<double> cwant;
  int ng = 10;
  for (int i = 2; i < argc; ++i) {
    auto next = [&]() { return std::string(argv[++i]); };
    if (!std::strcmp(argv[i], "--E")) E = std::stod(next());
    else if (!std::strcmp(argv[i], "--nu")) nu = std::stod(next());
    else if (!std::strcmp(argv[i], "--bc")) bcs = next();
    else if (!std::strcmp(argv[i], "--ng")) ng = std::stoi(next());
    else if (!std::strcmp(argv[i], "--cache")) cache = next();
    else if (!std::strcmp(argv[i], "--json")) json = next();
    else if (!std::strcmp(argv[i], "--points")) points = next();
    else if (!std::strcmp(argv[i], "--save-solutions")) save_sol = next();
    else if (!std::strcmp(argv[i], "--load-solutions")) load_sol = next();
    else if (!std::strcmp(argv[i], "--check-iso")) iso_tol = std::stod(next());
    else if (!std::strcmp(argv[i], "--tol")) ctol = std::stod(next());
    else if (!std::strcmp(argv[i], "--check-c")) {
      std::stringstream ss(next());
      for (std::string s; std::getline(ss, s, ',');) cwant.push_back(std::stod(s));
    } else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }
  SolverOptions o;
  o.n_gauss = ng;
  o.far_tol = 1e-10;
  auto readPoints = [&]() {
    std::vector<Vec3> xs;
    std::ifstream pf(points);
    for (double x, y, z; pf >> x >> y >> z;) xs.emplace_back(x, y, z);
    return xs;
  };
  auto writeStress = [&](std::ostream& out, const std::vector<BoundarySolution>& sols, const std::vector<Vec3>& xs) {
    for (int k = 0; k < 6; ++k) {
      const InteriorEvaluator ev(sols[k], o);
      const auto f = ev.eval(xs);
      out << (k ? ",\n    [" : "\n    [");
      for (size_t i = 0; i < f.size(); ++i) {
        const Vec6r v = toVoigt(f[i].stress, false);
        out << (i ? ", [" : "[");
        for (int c = 0; c < 6; ++c) out << (c ? ", " : "") << v(c);
        out << "]";
      }
      out << "]";
    }
  };
  if (!load_sol.empty()) {  // evaluation only, from saved boundary solutions
    if (points.empty() || json.empty()) {
      std::fprintf(stderr, "--load-solutions needs --points and --json\n");
      return 2;
    }
    Mesh mesh;
    TractionSpace ts;
    mesh.build(loadDat(geo), PartitionOptions{});
    ts.build(mesh);
    std::ifstream in(load_sol, std::ios::binary);
    char magic[8];
    int32_t nd, nt, nc;
    double Ef, nuf;
    in.read(magic, 8).read((char*)&nd, 4).read((char*)&nt, 4).read((char*)&nc, 4).read((char*)&Ef, 8).read((char*)&nuf, 8);
    if (!in || std::string(magic, 7) != "RVESOL1" || nd != mesh.ndof || nt != ts.ndof || nc != 6) {
      std::fprintf(stderr, "%s does not match %s\n", load_sol.c_str(), geo.c_str());
      return 1;
    }
    std::vector<BoundarySolution> sols;
    for (int k = 0; k < 6; ++k) {
      BoundarySolution s{&mesh, &ts, materialEnu(Ef, nuf), Eigen::VectorXd(3 * nd), Eigen::VectorXd(3 * nt), {}};
      in.read((char*)s.u.data(), 8 * 3 * nd).read((char*)s.t.data(), 8 * 3 * nt);
      sols.push_back(std::move(s));
    }
    const auto xs = readPoints();
    std::ofstream js(json);
    js << "{\n  \"geometry\": \"" << geo << "\", \"E\": " << Ef << ", \"nu\": " << nuf
       << ",\n  \"points\": {\"bc\": \"saved\", \"n\": " << xs.size() << ", \"stress\": [";
    writeStress(js, sols, xs);
    js << "]}\n}\n";
    std::printf("interior stress at %zu points x 6 cases from %s\n", xs.size(), load_sol.c_str());
    return 0;
  }
  RveModel m;
  const auto t0 = std::chrono::steady_clock::now();
  m.setup(geo, materialEnu(E, nu), o, cache == "none" ? "" : cache);
  const double t_setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const Vec3 L = m.lengths();
  std::printf("cell %s: %zu patches, %d dofs, %d traction dofs, box %.4g x %.4g x %.4g, porosity %.4f%%, %s\n",
              geo.c_str(), m.mesh.patches.size(), m.mesh.ndof, m.ts.ndof, L(0), L(1), L(2), 100 * m.porosity(),
              m.closed ? "closed faces" : "pores open on faces");
  std::printf("setup %.2f s (assembly %.2f s)\n", t_setup, m.t_assemble);

  int fail = 0;
  std::ofstream js;
  if (!json.empty()) {
    js.open(json);
    js << "{\n  \"geometry\": \"" << geo << "\", \"E\": " << E << ", \"nu\": " << nu << ", \"porosity\": " << m.porosity()
       << ", \"box\": [" << L(0) << ", " << L(1) << ", " << L(2) << "], \"dofs\": " << m.mesh.ndof
       << ", \"t_assemble\": " << m.t_assemble << ",\n  \"bc\": {";
  }
  const Mat6 Ciso = isotropicC(materialEnu(E, nu));
  std::stringstream ss(bcs);
  bool first = true;
  RveResult last;
  for (std::string b; std::getline(ss, b, ',');) {
    const RveBc bc = b == "kubc" ? RveBc::Kubc : b == "pbc" ? RveBc::Pbc : RveBc::Subc;
    RveResult r = m.solve(bc);
    const Engineering e = engineering(r.C);
    std::printf("\n%s: %d unknowns, factor %.2f s, 6 cases %.2f s, Hill-Mandel %.1e, asymmetry %.1e\n",
                rveBcName(bc), r.unknowns, r.t_factor, r.t_solve, r.hill_mandel, r.asymmetry);
    for (int i = 0; i < 6; ++i) {
      std::printf("  ");
      for (int j = 0; j < 6; ++j) std::printf("%11.3f", r.C(i, j));
      std::printf("\n");
    }
    std::printf("  E1 %.2f  E2 %.2f  E3 %.2f  G23 %.2f  G13 %.2f  G12 %.2f  nu12 %.4f  nu13 %.4f  nu23 %.4f\n", e.E1,
                e.E2, e.E3, e.G23, e.G13, e.G12, e.nu12, e.nu13, e.nu23);
    if (iso_tol >= 0) {
      const double err = (r.C - Ciso).norm() / Ciso.norm();
      const bool ok = err <= iso_tol;
      std::printf("  check isotropic C: rel err %.2e (tol %.1e) %s\n", err, iso_tol, ok ? "ok" : "FAIL");
      fail += !ok;
    }
    if (js.is_open()) {
      js << (first ? "" : ",") << "\n    \"" << b << "\": {\"C\": [";
      for (int i = 0; i < 6; ++i) {
        js << (i ? ", " : "") << "[";
        for (int j = 0; j < 6; ++j) js << (j ? ", " : "") << r.C(i, j);
        js << "]";
      }
      js << "], \"E1\": " << e.E1 << ", \"E2\": " << e.E2 << ", \"E3\": " << e.E3 << ", \"G23\": " << e.G23
         << ", \"G13\": " << e.G13 << ", \"G12\": " << e.G12 << ", \"nu12\": " << e.nu12 << ", \"nu13\": " << e.nu13
         << ", \"nu23\": " << e.nu23 << ", \"hill_mandel\": " << r.hill_mandel << ", \"unknowns\": " << r.unknowns
         << ", \"t_factor\": " << r.t_factor << ", \"t_solve\": " << r.t_solve << "}";
    }
    first = false;
    last = std::move(r);
  }
  if (!cwant.empty()) {
    const double got[4] = {last.C(0, 0), last.C(2, 2), last.C(3, 3), last.C(5, 5)};
    for (size_t k = 0; k < cwant.size() && k < 4; ++k) {
      const double rel = std::abs(got[k] - cwant[k]) / std::abs(cwant[k]);
      const bool ok = rel <= ctol;
      std::printf("  check C[%zu] got %.6f want %.6f rel %.1e %s\n", k, got[k], cwant[k], rel, ok ? "ok" : "FAIL");
      fail += !ok;
    }
  }
  if (js.is_open()) js << "\n  }";
  if (!save_sol.empty()) {
    std::ofstream out(save_sol, std::ios::binary);
    const int32_t nd = m.mesh.ndof, nt = m.ts.ndof, nc = 6;
    out.write("RVESOL1", 8).write((const char*)&nd, 4).write((const char*)&nt, 4).write((const char*)&nc, 4);
    out.write((const char*)&E, 8).write((const char*)&nu, 8);
    for (const auto& s : last.solutions) out.write((const char*)s.u.data(), 8 * s.u.size()).write((const char*)s.t.data(), 8 * s.t.size());
  }
  if (!points.empty()) {
    const auto xs = readPoints();
    const auto tp = std::chrono::steady_clock::now();
    if (js.is_open()) {
      js << ",\n  \"points\": {\"bc\": \"" << rveBcName(last.bc) << "\", \"n\": " << xs.size() << ", \"stress\": [";
      writeStress(js, last.solutions, xs);
      js << "]}";
    }
    std::printf("\ninterior stress at %zu points x 6 cases: %.2f s\n", xs.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
  }
  if (js.is_open()) js << "\n}\n";
  return fail ? 1 : 0;
}
