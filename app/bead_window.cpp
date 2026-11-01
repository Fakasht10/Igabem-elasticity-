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
//   --points-out f the points a field drive needs, then stop: "0 region dof x y z" (displacement) and "1 region
//                  traction-dof x y z" (- side of a z-periodic weld: the drive may give the jump t+ + t- there)
//   --drive field:f | field+chi:f   a part solution at those points ("0 region dof u(3) eps(6)", engineering
//                  shear; optional "1 region tdof jump(3) 0 0 0 0 0 0"),
//                  without / with the periodic fluctuation chi : eps (the consistent drive)
//   --serve        (field drives) after the first solve keep everything and answer "solve <field> <tractions>"
//                  requests on stdin ("ready" after each; GMRES warm-started from the previous solution)
//   --tractions f  weld and cut-face tractions per drive (BeadWindow::tractions; f.q per scale with --scales)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include <map>

#include "bead_window.hpp"
#include "recovery.hpp"
#include "multi_region.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  std::vector<std::string> files;
  std::vector<std::tuple<bool, int, double, double>> welds;
  double E = 2200.0, nu = 0.35, w = 0.6;
  int ng = 12, nx = 3, ny = 3;
  bool direct = false, serve = false;
  double tol = 1e-10;
  std::string cache = "none", json, tractions, drive = "strain", points_out, manifest;
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
    else if (!std::strcmp(argv[i], "--points-out")) points_out = next();
    else if (!std::strcmp(argv[i], "--serve")) serve = true;
    else if (!std::strcmp(argv[i], "--pieces")) manifest = next();
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
  // field drive: a part solution at the drive points (--points-out): lines "region dof ux uy uz e11 e22 e33 g23 g13
  // g12" (window frame, engineering shear); field+chi adds the periodic fluctuation chi : eps
  const bool field = drive.rfind("field:", 0) == 0 || drive.rfind("field+chi:", 0) == 0;
  const bool use_chi = drive == "corrector" || drive.rfind("field+chi:", 0) == 0;
  if (files.size() != 2 || (drive != "strain" && drive != "corrector" && !field)) {
    std::fprintf(stderr, "usage: bead_window <A.dat> <B.dat> --cells nx:ny [--weld a:Kn:Kt] [--json f]\n");
    return 2;
  }
  Eigen::Matrix3d R90;
  R90 << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  std::vector<std::string> fs;
  std::vector<Vec3> off, cell_at;   // cell_at: centre of the region's cell (corrector lookup)
  std::vector<std::pair<int, Eigen::Matrix3d>> cp;
  std::vector<int> fil, grp;
  if (manifest.empty()) {
    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int b = 0; b < 2; ++b) {
          fs.push_back(files[b]);
          off.push_back(Vec3(i * w, j * w, 0.0));
          cell_at.push_back(off.back());
          cp.push_back(fs.size() == 1 ? std::make_pair(-1, Eigen::Matrix3d::Identity()) : std::make_pair(0, b ? R90 : Eigen::Matrix3d::Identity()));
          fil.push_back(b ? 1 : 0);
          grp.push_back(i * ny + j);
        }
  } else {
    // arbitrary pieces (python/rve/notch.py): "file type(0 A, 1 B) ox oy oz cx cy src group"; src = -1: own operator,
    // else the region whose operator it copies (turned 90 deg if the types differ)
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("cannot read " + manifest);
    std::string f;
    int t, src, gr;
    double ox, oy, oz, cx, cy;
    std::vector<int> types;
    while (in >> f >> t >> ox >> oy >> oz >> cx >> cy >> src >> gr) {
      fs.push_back(f);
      off.push_back(Vec3(ox, oy, oz));
      cell_at.push_back(Vec3(cx, cy, 0.0));
      types.push_back(t);
      cp.push_back(src < 0 ? std::make_pair(-1, Eigen::Matrix3d::Identity())
                           : std::make_pair(src, t != types[src] ? R90 : Eigen::Matrix3d::Identity()));
      fil.push_back(t);
      grp.push_back(gr);
    }
    nx = ny = 0;
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
  if (!points_out.empty()) {  // the points a field drive needs, then stop
    std::ofstream os(points_out);
    os.precision(17);
    for (const auto& [r, g, x] : m.drivePoints()) os << "0 " << r << ' ' << g << ' ' << x(0) << ' ' << x(1) << ' ' << x(2) << '\n';
    for (const auto& [r, d, x] : m.tdrivePoints()) os << "1 " << r << ' ' << d << ' ' << x(0) << ' ' << x(1) << ' ' << x(2) << '\n';
    return 0;
  }
  // per point: Q blocks of 9 values (several drives in one file, solved together)
  using Vals = std::vector<Eigen::Matrix<double, 9, 1>>;
  auto fptr = std::make_shared<std::map<std::pair<int, int>, Vals>>();
  auto tptr = std::make_shared<std::map<std::pair<int, int>, Vals>>();
  auto& fval = *fptr;
  auto& tval = *tptr;
  int nfield = 0;
  auto readField = [&](const std::string& path) {  // type 0: region dof [u(3) eps(6)] x Q; type 1: region tdof
    fval.clear();                                  // [jump(3) 0(6)] x Q
    tval.clear();
    std::ifstream in(path);
    if (!in) throw std::runtime_error("field drive: cannot read " + path);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream ls(line);
      int type, r, g;
      if (!(ls >> type >> r >> g)) continue;
      std::vector<double> x;
      for (double v; ls >> v;) x.push_back(v);
      if (x.size() % 9) throw std::runtime_error("field drive: values not a multiple of 9");
      Vals vs(x.size() / 9);
      for (size_t q = 0; q < vs.size(); ++q)
        for (int c = 0; c < 9; ++c) vs[q](c) = x[9 * q + c];
      nfield = int(vs.size());
      (type == 0 ? fval : tval)[{r, g}] = vs;
    }
  };
  if (field) {
    readField(drive.substr(drive.find(':') + 1));
    std::printf("field drive: %zu points, %zu periodic traction jumps\n", fval.size(), tval.size());
  }
  // corrector drive: the periodic cell of the same two pieces and weld laws (window internal x, y = the cell's
  // periodic lateral welds; internal z = A to B; periodic z = B top to the next A bottom)
  BeadCell cell;
  if (use_chi) {
    cell.weld_periodic = {m.weld_internal[0], m.weld_internal[1], m.weld_periodic[2]};
    cell.weld_internal[2] = m.weld_internal[2];
    cell.setup(files, materialEnu(E, nu), o, {{-1, Eigen::Matrix3d::Identity()}, {0, R90}}, cache == "none" ? "" : cache, {0, 1});
  }
  std::ostringstream all;
  all.precision(10);
  for (size_t q = 0; q < scales.size(); ++q) {
    std::vector<BeadWindow::Drive> drives;
    // per window region, per unit strain k: the fluctuation (3 x ndof) of the periodic cell at its dofs
    std::shared_ptr<std::vector<std::vector<Eigen::MatrixXd>>> chi;
    if (use_chi) {
      const RveResult rc = cell.solveScales(std::vector<Vec3>{scales[q]})[0];
      std::vector<Eigen::MatrixXd> cc;  // cc[2 k + b]: cell piece b
      for (int k = 0; k < 6; ++k)
        for (int b = 0; b < 2; ++b) {
          const BeadRegion& P = cell.regions[b];
          Eigen::MatrixXd X(3, P.mesh.ndof);
          const Eigen::Matrix3d Ek = voigtUnit(k, true);
          for (int g = 0; g < P.mesh.ndof; ++g) X.col(g) = rc.solutions[k * 2 + b].u.segment<3>(3 * g) - Ek * P.mesh.dof_point[g];
          cc.push_back(X);
        }
      chi = std::make_shared<std::vector<std::vector<Eigen::MatrixXd>>>(m.regions.size());
      int nmatch = 0, nlocate = 0;
      for (size_t r = 0; r < m.regions.size(); ++r) {
        const int b = fil[r];
        auto& out = (*chi)[r];
        const BeadRegion& R = m.regions[r];
        if (R.file == files[b]) {  // a copy of the cell's piece: same dofs
          for (int k = 0; k < 6; ++k) out.push_back(cc[2 * k + b]);
          continue;
        }
        // own geometry (a cut piece): the cell's coefficient at a coinciding dof point, else chi at the closest
        // point of the cell piece (the cut-face dofs of a trimmed patch)
        const BeadRegion& C = cell.regions[b];
        const double tl = 1e-7 * w;
        std::map<std::array<long, 3>, int> at;
        auto key = [&](const Vec3& x) { return std::array<long, 3>{std::lround(x(0) / tl), std::lround(x(1) / tl), std::lround(x(2) / tl)}; };
        for (int g = 0; g < C.mesh.ndof; ++g) at[key(C.mesh.dof_point[g])] = g;
        for (int k = 0; k < 6; ++k) out.push_back(Eigen::MatrixXd::Zero(3, R.mesh.ndof));
        std::vector<char> drv(R.mesh.ndof, 0);  // only the cut-face dofs are driven
        for (size_t p = 0; p < R.driven.size(); ++p)
          if (R.driven[p])
            for (int g : R.mesh.l2g[p]) drv[g] = 1;
        for (int g = 0; g < R.mesh.ndof; ++g) {
          if (!drv[g]) continue;
          const Vec3 x = R.mesh.dof_point[g] - cell_at[r];
          const auto it = at.find(key(x));
          if (it != at.end()) {
            for (int k = 0; k < 6; ++k) out[k].col(g) = cc[2 * k + b].col(it->second);
            ++nmatch;
            continue;
          }
          const SurfacePoint sp = locate(C.mesh, x, [](int) { return true; });
          for (int k = 0; k < 6; ++k)
            for (int l = 0; l < sp.E.n_loc; ++l) out[k].col(g) += sp.E.R[l] * cc[2 * k + b].col(C.mesh.l2g[sp.patch][sp.E.idx[l]]);
          ++nlocate;
        }
      }
      std::printf("periodic cell for the corrector drive: C11 %.2f, %d unknowns (cut pieces: %d dofs matched, %d located)\n",
                  rc.C(0, 0), rc.unknowns, nmatch, nlocate);
    }
    for (int f = 0; field && f < nfield; ++f) {
      auto F = fptr;
      drives.push_back([F, chi, f](int r, int g, const Vec3&) {
        const int rr = r < 0 ? -1 - r : r;
        const auto it = F->find({rr, g});
        if (it == F->end()) throw std::runtime_error("field drive: no value for region " + std::to_string(rr) + " dof " + std::to_string(g));
        const auto& v = it->second[f];
        Vec3 u = v.head<3>();
        if (chi && r >= 0)
          for (int k = 0; k < 6; ++k) u += v(3 + k) * (*chi)[rr][k].col(g);
        return u;
      });
    }
    for (int k = 0; k < 6 && !field; ++k) {
      const Eigen::Matrix3d Ek = voigtUnit(k, true);
      if (chi) drives.push_back([Ek, chi, k](int r, int g, const Vec3& x) { return r < 0 ? Vec3(Ek * x) : Vec3(Ek * x + (*chi)[r][k].col(g)); });
      else drives.push_back([Ek](int, int, const Vec3& x) { return Vec3(Ek * x); });
    }
    std::vector<BeadWindow::TDrive> tdrives;
    for (int f = 0; field && f < nfield; ++f) {
      auto Tv = tptr;
      tdrives.push_back([Tv, f](int r, int d, const Vec3&) {
        const auto it = Tv->find({r, d});
        return it == Tv->end() ? Vec3(Vec3::Zero()) : Vec3(it->second[f].head<3>());
      });
    }
    m.factor(scales[q]);
    const BeadWindow::Result r = m.solve(drives, tdrives);
    std::printf("scale %g:%g:%g: %d unknowns, %ld block entries (%.1f GB), assembly %.2f s, %s %.2f s, solve %.2f s%s\n",
                scales[q](0), scales[q](1), scales[q](2), m.unknowns(), m.nonzeros(), 8e-9 * m.nonzeros(), m.t_build,
                direct ? "UMFPACK factor" : "block-Jacobi LU", m.t_factor, r.t_solve,
                direct ? "" : (" (GMRES " + std::to_string(m.iterations) + " iterations)").c_str());
    if (!tractions.empty()) {
      std::ofstream os(scales.size() == 1 ? tractions : tractions + "." + std::to_string(q));
      m.tractions(r, os);
    }
    if (serve && field) {  // keep everything (operator, preconditioner, chi); "solve <field> <tractions>" per request
      m.warm = true;
      std::printf("ready\n");
      std::fflush(stdout);
      std::string cmd, a, b;
      while (std::cin >> cmd) {
        if (cmd == "quit") break;
        if (cmd == "solve" && (std::cin >> a >> b)) {
          const auto ts = std::chrono::steady_clock::now();
          readField(a);
          const BeadWindow::Result rs = m.solve(drives, tdrives);
          std::ofstream os(b);
          m.tractions(rs, os);
          std::printf("# solved in %.2f s (GMRES %d iterations, warm), output %.2f s\n", rs.t_solve, m.iterations,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count() - rs.t_solve);
        } else std::printf("# unknown command %s\n", cmd.c_str());
        std::printf("ready\n");
        std::fflush(stdout);
      }
      return 0;
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
