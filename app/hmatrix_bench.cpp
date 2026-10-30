// Hierarchical compression of the SBIE operators on the plate with a hole
// (src/fast, src/problems/plate_hole.hpp): memory, compression, assembly time
// and, against the dense operators, the matrix-vector and solution error as a
// function of the ACA tolerance eps.
//
// usage: hmatrix_bench <geometry.dat> <label> <t> [--eps 1e-4,1e-6,1e-8] [--eta 3] [--leaf 192]
//                      [--block-tol 0.1] [--block-order 4] [--dense] [--leaf-errors] [--n 12]
//                      [--far-tol 1e-10] [--cache dir|none] [--csv file]
//   --block-tol, --block-order  quadrature of the separated element integrals (HOptions::far_tol,
//            far_order): error estimate block-tol * eps, Gauss order at least block-order
//   --dense  also assemble (or load) the dense H, G: errors of the compressed products, and the
//            traction-free Kirsch problem solved with the compressed operators (defect
//            correction preconditioned by the dense bordered LU, so the result is the solution of
//            the compressed system) against the dense solve and the exact field
//   --leaf-errors  with --dense: the G leaves farthest from the dense operator
//
// Run with OPENBLAS_NUM_THREADS=1 (the compression is parallel over blocks).

#include <sys/resource.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fast/compressed_sbie.hpp"
#include "operators.hpp"
#include "problems/plate_hole.hpp"
#include "solution.hpp"

using namespace igabem;

static double peakRssMB() {
  rusage u;
  getrusage(RUSAGE_SELF, &u);
  return u.ru_maxrss / 1024.0;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: hmatrix_bench <geometry.dat> <label> <t> [--eps 1e-4,1e-6,1e-8] [--eta 3] [--leaf 192] "
                 "[--dense] [--n 12] [--far-tol 1e-10] [--cache dir|none] [--csv file]\n";
    return 1;
  }
  const std::string geo = argv[1], label = argv[2];
  const double thick = std::atof(argv[3]);
  std::vector<double> eps_list = {1e-4, 1e-6, 1e-8};
  HOptions ho;
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  bool dense = false, leaf_errors = false;
  std::string cache = "cache", csv;
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--eps")) {
      eps_list.clear();
      std::stringstream ss(argv[++i]);
      for (std::string e; std::getline(ss, e, ',');) eps_list.push_back(std::atof(e.c_str()));
    } else if (!std::strcmp(argv[i], "--eta")) ho.eta = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--leaf")) ho.leaf = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--dense")) dense = true;
    else if (!std::strcmp(argv[i], "--leaf-errors")) leaf_errors = true;
    else if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--block-tol")) ho.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--block-order")) ho.far_order = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
    else if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
  }

  const PlateHole ph;
  const PartitionOptions po;
  Mesh mesh;
  mesh.build(loadDat(geo), po);
  TractionSpace ts;
  ts.build(mesh);
  const int N = 3 * mesh.ndof, Nt = 3 * ts.ndof;
  const auto hole = PlateHole::holePatches(mesh);
  auto onHole = [&](int p) { return bool(hole[p]); };
  const SurfacePoint sp = locate(mesh, Vec3(0.0, ph.a, 0.0), onHole);
  const double sxx_exact = ph.stress(PlateHole::Kirsch, Vec3(0.0, ph.a, 0.0))(0, 0);
  std::printf("# %s: %zu patches, %zu elements, %d unknowns, %d traction unknowns, t = %g, n = %d, far_tol = %g\n",
              label.c_str(), mesh.patches.size(), mesh.elements.size(), N, Nt, thick, opt.n_gauss, opt.far_tol);

  // Kirsch data (traction-free hole), interpolated in the traction space.
  const Eigen::VectorXd tc = ts.interpolate(
      [&](const Vec3& y, const Vec3& n) { return TractionBlock(ph.traction(PlateHole::Kirsch, y, n)); }, 1);
  auto kirschError = [&](const Eigen::VectorXd& u) {
    return displacementL2Error(mesh, u, [&](const Vec3& x) { return ph.displacement(PlateHole::Kirsch, x); });
  };
  auto sxx = [&](const Eigen::VectorXd& u) {
    const BoundarySolution S{&mesh, &ts, ph.mat, u, tc};
    return S.stress(sp)(0, 0);
  };

  // Dense reference: operators, bordered LU, Kirsch solution.
  OperatorRows Dn;
  Eigen::PartialPivLU<Eigen::MatrixXd> lu;
  Eigen::MatrixXd R;
  double s = 0.0, err_d = std::nan(""), sxx_d = std::nan(""), t_dense = std::nan("");
  Eigen::VectorXd ud;
  if (dense) {
    Dn = cachedOperatorRows(mesh, ts, ph.mat, opt, po, geo, cache, "sbie_hg", mesh.colloc);
    t_dense = Dn.t_assemble;
    R = rigidModes(mesh);
    s = Dn.H.cwiseAbs().maxCoeff();
    Eigen::MatrixXd Ab = Eigen::MatrixXd::Zero(N + 6, N + 6);
    Ab.topLeftCorner(N, N) = Dn.H;
    Ab.topRightCorner(N, 6) = s * R;
    Ab.bottomLeftCorner(6, N) = s * R.transpose();
    lu.compute(Ab);
    Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6);
    bb.head(N) = Dn.G * tc;
    ud = lu.solve(bb).head(N);
    err_d = kirschError(ud);
    sxx_d = sxx(ud);
    std::printf("# dense: H + G %.1f MB, assembled in %.1f s%s; Kirsch rel L2 %.4e, sxx(0,a,0) %.8f (exact %.8f)\n",
                8e-6 * (double(N) * N + double(N) * Nt), t_dense, Dn.from_cache ? " (cached)" : "", err_d, sxx_d,
                sxx_exact);
  }

  std::ofstream out;
  if (!csv.empty()) {
    std::ifstream probe(csv);
    const bool fresh = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    out.open(csv, std::ios::app);
    out.precision(10);
    if (fresh) out << "label,t,p,elements,unknowns,traction_unknowns,eps,eta,leaf,dense_MB,hmat_MB,ratio,ratio_H,"
                      "ratio_G,max_rank_H,max_rank_G,leaves_dense,leaves_lowrank,t_assemble,t_assemble_dense,"
                      "cached,rss_MB,err_Hx,err_Gt,err_kirsch_dense,err_kirsch_h,sxx_dense,sxx_h,sxx_exact,"
                      "du_rel,dc_iters\n";
  }
  std::printf("# %-8s %9s %9s %7s %7s %6s %6s %9s %9s %10s %10s %11s %10s %10s %4s\n", "eps", "dense_MB", "hmat_MB",
              "ratio", "rank_H", "rank_G", "", "t_asm_s", "rss_MB", "err_Hx", "err_Gt", "err_kirsch", "sxx", "du_rel", "it");
  for (double eps : eps_list) {
    ho.eps = eps;
    CompressedSbie C;
    C.setup(mesh, ts, ph.mat, opt, po, ho, geo, cache, "sbie_hg", mesh.colloc);
    const HStats sh = C.Hraw.stats(), sg = C.G.stats();
    const double mb = 1e-6 * C.bytes(), dmb = 1e-6 * C.denseBytes();
    double err_h = std::nan(""), err_g = std::nan(""), err_k = std::nan(""), sxx_h = std::nan(""),
           du = std::nan("");
    int it = 0;
    if (dense) {
      const Eigen::VectorXd x = Eigen::VectorXd::Random(N);
      err_h = (C.applyH(ud) - Dn.H * ud).norm() / (Dn.H * ud).norm();
      err_g = (C.applyG(tc) - Dn.G * tc).norm() / (Dn.G * tc).norm();
      // Defect correction: z <- z + LU^-1 (b_h - A_h z), A_h = [H_h sR; sR^T 0].
      Eigen::VectorXd bb = Eigen::VectorXd::Zero(N + 6), z = Eigen::VectorXd::Zero(N + 6);
      bb.head(N) = C.applyG(tc);
      for (it = 1; it <= 30; ++it) {
        Eigen::VectorXd r = bb;
        r.head(N) -= C.applyH(Eigen::VectorXd(z.head(N))) + s * R * z.tail(6);
        r.tail(6) -= s * R.transpose() * z.head(N);
        if (r.norm() <= 1e-14 * bb.norm()) break;
        z += lu.solve(r);
      }
      const Eigen::VectorXd uh = z.head(N);
      err_k = kirschError(uh);
      sxx_h = sxx(uh);
      du = (uh - ud).norm() / ud.norm();
    }
    if (dense && leaf_errors) {
      struct Bad { double e, n; int m, k; bool d; };
      std::vector<Bad> bad;
      C.G.leafErrors(Dn.G, [&](int mm, int nn, bool d, double e, double nr) { bad.push_back({e, nr, mm, nn, d}); });
      std::sort(bad.begin(), bad.end(), [](auto& a, auto& b) { return a.e > b.e; });
      for (int t = 0; t < 8 && t < int(bad.size()); ++t)
        std::printf("# G leaf %d x %d %s: |err| %.3e |A| %.3e\n", bad[t].m, bad[t].k, bad[t].d ? "dense" : "lowrank", bad[t].e, bad[t].n);
    }
    if (!C.from_cache)
      std::printf("#   build: H %.1f s (near field %.1f s), G %.1f s (near field %.1f s); dense leaves %.0f%% of H, %.0f%% of G storage\n",
                  C.Hraw.t_build, C.Hraw.t_near, C.G.t_build, C.G.t_near, 100 * sh.stored_dense / sh.stored,
                  100 * sg.stored_dense / sg.stored);
    std::printf("  %-8.0e %9.1f %9.1f %6.2f%% %7ld %6ld %6s %9.1f %9.0f %10.2e %10.2e %11.4e %10.7f %10.2e %4d%s\n", eps,
                dmb, mb, 100 * mb / dmb, sh.max_rank, sg.max_rank, "", C.t_assemble, peakRssMB(), err_h, err_g, err_k,
                sxx_h, du, it, C.from_cache ? "  (cached)" : "");
    if (out)
      out << label << ',' << thick << ',' << mesh.patches[0].p << ',' << mesh.elements.size() << ',' << N << ','
          << Nt << ',' << eps << ',' << ho.eta << ',' << ho.leaf << ',' << dmb << ',' << mb << ',' << mb / dmb << ','
          << sh.ratio() << ',' << sg.ratio() << ',' << sh.max_rank << ',' << sg.max_rank << ','
          << sh.dense_leaves + sg.dense_leaves << ',' << sh.lowrank_leaves + sg.lowrank_leaves << ','
          << C.t_assemble << ',' << t_dense << ',' << C.from_cache << ',' << peakRssMB() << ',' << err_h << ','
          << err_g << ',' << err_d << ',' << err_k << ',' << sxx_d << ',' << sxx_h << ',' << sxx_exact << ','
          << du << ',' << it << '\n';
  }
  return 0;
}
