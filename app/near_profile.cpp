// Cost of the element integrals of the SBIE rows by distance class: for every
// (collocation point, element) pair closer than --dmax element sizes, the number
// of quadrature points and the time of the integral (kernel U, T and the basis
// contraction, as SbieEntries does), binned by d / h with d the distance from the
// point to the element's bounding ball. Single thread.
//
// usage: near_profile <geometry.dat> [--n 12] [--far-tol 1e-10] [--dmax 8]

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#include "integrator.hpp"
#include "kelvin.hpp"
#include "problems/plate_hole.hpp"

using namespace igabem;

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: near_profile <geometry.dat> [--n 12] [--far-tol 1e-10] [--dmax 8]\n"); return 1; }
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  double dmax = 8.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n")) opt.n_gauss = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol")) opt.far_tol = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--dmax")) dmax = std::atof(argv[++i]);
  }
  const PlateHole ph;
  Mesh mesh;
  mesh.build(loadDat(argv[1]), PartitionOptions());
  const ElementIntegrator quad(mesh, opt);
  const Kelvin K(ph.mat);
  // Classes: 0 own element (singular), 1 touching (d = 0), then d/h bins.
  const std::vector<double> edges = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, dmax};
  const int nb = int(edges.size()) + 1;
  std::vector<double> pairs(nb), qp(nb), cached(nb), sec(nb);
  double acc[9 * 64];
  for (const Collocation& c : mesh.colloc) {
    const SourcePoint src{c.patch, c.xi, c.x};
    for (size_t ie = 0; ie < mesh.elements.size(); ++ie) {
      const Element& e = mesh.elements[ie];
      const double d = std::max(0.0, (c.x - quad.elementCentre(ie)).norm() - quad.elementRadius(ie)) / e.h;
      if (d >= dmax) continue;
      const bool own = e.patch == c.patch && e.rect.contains(c.xi, 1e-14);
      int b = own ? 0 : 1;
      if (!own && d > 0.0) { b = 2; while (b < nb - 1 && d >= edges[b - 1]) ++b; }
      long n = 0, nc = 0;
      const auto t0 = std::chrono::steady_clock::now();
      quad.integrate(src, e, [&](const QPoint& E, double w) {
        ++n;
        nc += E.id >= 0;
        const Vec3 nJ = E.normal() * mesh.normal_sign;
        const double J = nJ.norm();
        Eigen::Matrix3d U, T;
        K.eval(E.x - c.x, Vec3(nJ / J), U, T);
        const Eigen::Matrix3d Tw = T * (w * J), Uw = U * (w * J);
        for (int l = 0; l < E.n_loc; ++l) {
          Eigen::Map<Eigen::Matrix3d>(&acc[9 * l]) += E.R[l] * Tw;
          Eigen::Map<Eigen::Matrix3d>(&acc[9 * l]) += E.R[l] * Uw;
        }
      });
      sec[b] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      pairs[b] += 1;
      qp[b] += n;
      cached[b] += nc;
    }
  }
  std::printf("# %zu points, %zu elements, n = %d, far_tol = %g; H and G together, 1 thread (checksum %g)\n",
              mesh.colloc.size(), mesh.elements.size(), opt.n_gauss, opt.far_tol, acc[0]);
  std::printf("# %-14s %10s %12s %10s %9s %10s %9s\n", "class", "pairs", "qp", "qp/pair", "cached%", "us/pair", "time_s");
  double tot = 0.0;
  for (int b = 0; b < nb; ++b) {
    char name[32];
    if (b == 0) std::snprintf(name, 32, "own (singular)");
    else if (b == 1) std::snprintf(name, 32, "touching");
    else std::snprintf(name, 32, "d/h %.1f-%.1f", edges[b - 2], edges[b - 1]);
    if (pairs[b] == 0) continue;
    std::printf("  %-14s %10.0f %12.0f %10.1f %8.1f%% %10.2f %9.2f\n", name, pairs[b], qp[b], qp[b] / pairs[b],
                100 * cached[b] / qp[b], 1e6 * sec[b] / pairs[b], sec[b]);
    tot += sec[b];
  }
  std::printf("# total %.2f s for the pairs with d < %g h\n", tot, dmax);
}
