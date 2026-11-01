// Bead window (src/bead_window.hpp): nx x ny cells of 0/90 bead pieces, cut faces driven, z-periodic slab.
//
// usage: bead_window <A.dat> <B.dat> --cells nx:ny [--pitch w] [--E 2200] [--nu 0.35] [--ng 12]
//                    [--weld a:Kn:Kt] [--weld-int a:Kn:Kt] [--scales s1,...] [--cache dir|none]
//                    [--drive strain] [--tractions f] [--json f]
//
//   A.dat, B.dat   the pieces of python/rve/beads.py bead090 (B = A turned 90 deg, lifted by h); cell (i, j) holds
//                  A and B translated by (i w, j w, 0); every piece is a rigid copy of A (one condensed operator)
//   --weld a:...   spring law on the periodic welds normal to a (z: B top to the next A bottom)
//   --weld-int a:  the same on the internal welds normal to a (x: B lateral, y: A lateral, z: A to B); the joints
//                  between pieces of one filament stay perfect
//   --scales       every weld compliance times s (or sx:sy:sz), one factorization each
//   --drive strain the six unit macro strains u = E_k x on the cut faces (and as the z jump): kinematic uniform BC
//   --drive corrector  u = E_k x + chi_k, chi_k the periodic cell's fluctuation at the same dof (same weld laws and
//                  scale): the window then reproduces the periodic cell exactly (the consistency check)
//   --direct       UMFPACK on the assembled matrix (small windows; default GMRES + block Jacobi over cells)
//   --tractions f  weld and cut-face tractions per drive (BeadWindow::tractions; f.q per scale with --scales)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "bead_window.hpp"
#include "multi_region.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  std::vector<std::string> files;
  std::vector<std::tuple<bool, int, double, double>> welds;
  double E = 2200.0, nu = 0.35, w = 0.6;
  int ng = 12, nx = 3, ny = 3;
  bool direct = false;
  double tol = 1e-10;
  std::string cache = "none", json, tractions, drive = "strain";
  std::vector<Vec3> scales{Vec3::Ones()};
  auto weld = [](const char* s, bool internal) {
    double Kn, Kt;
    char a;
    if (std::sscanf(s, "%c:%lf:%lf", &a, &Kn, &Kt) != 3) throw std::runtime_error("bad --weld value");
    return std::make_tuple(internal, int(a - 'x'), Kn, Kt);
  };
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { return argv[++i]; };
    if (argv[i][0] != '-') files.push_back(argv[i]);
    else if (!std::strcmp(argv[i], "--cells")) { if (std::sscanf(next(), "%d:%d", &nx, &ny) != 2) return 2; }
    else if (!std::strcmp(argv[i], "--pitch")) w = std::stod(next());
    else if (!std::strcmp(argv[i], "--E")) E = std::stod(next());
    else if (!std::strcmp(argv[i], "--nu")) nu = std::stod(next());
    else if (!std::strcmp(argv[i], "--ng")) ng = std::stoi(next());
    else if (!std::strcmp(argv[i], "--cache")) cache = next();
    else if (!std::strcmp(argv[i], "--json")) json = next();
    else if (!std::strcmp(argv[i], "--drive")) drive = next();
    else if (!std::strcmp(argv[i], "--direct")) direct = true;
    else if (!std::strcmp(argv[i], "--tol")) tol = std::stod(next());
    else if (!std::strcmp(argv[i], "--tractions")) tractions = next();
    else if (!std::strcmp(argv[i], "--weld")) welds.push_back(weld(next(), false));
    else if (!std::strcmp(argv[i], "--weld-int")) welds.push_back(weld(next(), true));
    else if (!std::strcmp(argv[i], "--scales")) {
      scales.clear();
      std::stringstream ss(next());
      for (std::string t; std::getline(ss, t, ',');) {
        double a, b, c;
        if (std::sscanf(t.c_str(), "%lf:%lf:%lf", &a, &b, &c) == 3) scales.push_back(Vec3(a, b, c));
        else scales.push_back(Vec3::Constant(std::stod(t)));
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }
  if (files.size() != 2 || (drive != "strain" && drive != "corrector")) {
    std::fprintf(stderr, "usage: bead_window <A.dat> <B.dat> --cells nx:ny [--weld a:Kn:Kt] [--json f]\n");
    return 2;
  }
  Eigen::Matrix3d R90;
  R90 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  std::vector<std::string> fs;
  std::vector<Vec3> off;
  std::vector<std::pair<int, Eigen::Matrix3d>> cp;
  std::vector<int> fil, grp;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int b = 0; b < 2; ++b) {
        fs.push_back(files[b]);
        off.push_back(Vec3(i * w, j * w, 0.0));
        cp.push_back(fs.size() == 1 ? std::make_pair(-1, Eigen::Matrix3d::Identity()) : std::make_pair(0, b ? R90 : Eigen::Matrix3d::Identity()));
        fil.push_back(b ? 1 : 0);
        grp.push_back(i * ny + j);
      }
  SolverOptions o;
  o.n_gauss = ng;
  o.far_tol = 1e-10;
  BeadWindow m;
  for (auto [internal, a, Kn, Kt] : welds) {
    (internal ? m.weld_internal : m.weld_periodic)[a] = BeadCell::springCompliance(a, Kn, Kt);
    std::printf("%s weld normal to %c: Kn %.4g Kt %.4g N/mm^3\n", internal ? "internal" : "periodic", 'x' + a, Kn, Kt);
  }
  const auto t0 = std::chrono::steady_clock::now();
  m.direct = direct;
  m.tol = tol;
  m.setupWindow(fs, off, cp, fil, {false, false, true}, materialEnu(E, nu), o, cache == "none" ? "" : cache, grp);
  const double t_setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  int nper = 0, nint = 0, ncut = 0;
  for (auto& p : m.pairs) (p.internal ? nint : nper)++;
  for (auto& r : m.regions) for (char d : r.driven) ncut += d;
  const BeadRegion& r0 = m.regions[0];
  std::printf("%d x %d cells, %zu pieces (%d dofs each, %zu kept, %s %.2f s, condense %.2f s), %d periodic + %d internal "
              "pairs, %d cut patches, setup %.2f s\n", nx, ny, m.regions.size(), r0.mesh.ndof, r0.bdof.size() + r0.tcon.size(),
              r0.reduced_from_cache ? "cached" : "assembly", r0.t_assemble, r0.t_reduce, nper, nint, ncut, t_setup);
  // corrector drive: the periodic cell of the same two pieces and weld laws (window internal x, y = the cell's
  // periodic lateral welds; internal z = A to B; periodic z = B top to the next A bottom)
  BeadCell cell;
  if (drive == "corrector") {
    cell.weld_periodic = {m.weld_internal[0], m.weld_internal[1], m.weld_periodic[2]};
    cell.weld_internal[2] = m.weld_internal[2];
    cell.setup(files, materialEnu(E, nu), o, {{-1, Eigen::Matrix3d::Identity()}, {0, R90}}, cache == "none" ? "" : cache, {0, 1});
  }
  std::ostringstream all;
  all.precision(10);
  for (size_t q = 0; q < scales.size(); ++q) {
    std::vector<BeadWindow::Drive> drives;
    std::shared_ptr<std::vector<Eigen::MatrixXd>> chi;  // per piece: fluctuation (3 x ndof) per unit strain
    if (drive == "corrector") {
      const RveResult rc = cell.solveScales(std::vector<Vec3>{scales[q]})[0];
      chi = std::make_shared<std::vector<Eigen::MatrixXd>>();
      for (int k = 0; k < 6; ++k)
        for (int b = 0; b < 2; ++b) {
          const BeadRegion& P = cell.regions[b];
          Eigen::MatrixXd X(3, P.mesh.ndof);
          const Eigen::Matrix3d Ek = voigtUnit(k, true);
          for (int g = 0; g < P.mesh.ndof; ++g) X.col(g) = rc.solutions[k * 2 + b].u.segment<3>(3 * g) - Ek * P.mesh.dof_point[g];
          chi->push_back(X);
        }
      std::printf("periodic cell for the corrector drive: C11 %.2f, %d unknowns\n", rc.C(0, 0), rc.unknowns);
    }
    for (int k = 0; k < 6; ++k) {
      const Eigen::Matrix3d Ek = voigtUnit(k, true);
      if (chi) drives.push_back([Ek, chi, k](int r, int g, const Vec3& x) { return r < 0 ? Vec3(Ek * x) : Vec3(Ek * x + (*chi)[2 * k + r % 2].col(g)); });
      else drives.push_back([Ek](int, int, const Vec3& x) { return Vec3(Ek * x); });
    }
    m.factor(scales[q]);
    const BeadWindow::Result r = m.solve(drives);
    std::printf("scale %g:%g:%g: %d unknowns, %ld block entries (%.1f GB), assembly %.2f s, %s %.2f s, solve %.2f s%s\n",
                scales[q](0), scales[q](1), scales[q](2), m.unknowns(), m.nonzeros(), 8e-9 * m.nonzeros(), m.t_build,
                direct ? "UMFPACK factor" : "block-Jacobi LU", m.t_factor, r.t_solve,
                direct ? "" : (" (GMRES " + std::to_string(m.iterations) + " iterations)").c_str());
    if (!tractions.empty()) {
      std::ofstream os(scales.size() == 1 ? tractions : tractions + "." + std::to_string(q));
      m.tractions(r, os);
    }
    all << (q ? ",\n " : "") << "{\"scale\": [" << scales[q](0) << ", " << scales[q](1) << ", " << scales[q](2)
        << "], \"unknowns\": " << m.unknowns() << ", \"iterations\": " << m.iterations << ", \"nonzeros\": " << m.nonzeros() << ", \"t_build\": " << m.t_build
        << ", \"t_factor\": " << m.t_factor << ", \"t_solve\": " << r.t_solve << "}";
  }
  if (!json.empty()) {
    std::ofstream f(json);
    f << "{\"cells\": [" << nx << ", " << ny << "], \"pitch\": " << w << ", \"E\": " << E << ", \"nu\": " << nu
      << ", \"t_setup\": " << t_setup << ", \"runs\": [" << all.str() << "]}\n";
  }
  return 0;
}
