// Multipatch boundary mesh: conforming isoparametric dofs, elements with
// their App. B partitions, and SBIE collocation points.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "nurbs.hpp"
#include "quadrature.hpp"

namespace igabem {

struct Element {
  int patch, su, sv;
  Rect rect;                // parametric extent
  std::vector<Rect> cells;  // App. B partition
  double h;                 // physical diameter estimate
};

struct Collocation {
  int patch, su, sv;
  Vec2 xi;
  Vec3 x;
  std::vector<std::pair<int, double>> basis;  // global dof, N_B(x) != 0
};

struct Mesh {
  std::vector<Patch> patches;
  std::vector<std::vector<int>> l2g;  // per patch: local cp -> global dof
  std::vector<Vec3> dof_point;        // control point of each global dof
  std::vector<Element> elements;
  std::vector<Collocation> colloc;
  int ndof = 0;
  double normal_sign = 1.0;  // flips normals if the files are inward oriented

  void build(std::vector<Patch> P, const PartitionOptions& popt, double merge_tol = 1e-9) {
    patches = std::move(P);
    // Merge coincident control points into conforming global dofs.
    l2g.assign(patches.size(), {});
    dof_point.clear();
    std::vector<std::pair<int, int>> owner;
    for (size_t ip = 0; ip < patches.size(); ++ip) {
      const Patch& Pa = patches[ip];
      l2g[ip].resize(Pa.nu * Pa.nv);
      for (int k = 0; k < Pa.nu * Pa.nv; ++k) {
        const Vec3 X = Pa.cp(k);
        int found = -1;
        for (size_t g = 0; g < dof_point.size(); ++g)
          if ((dof_point[g] - X).norm() < merge_tol) { found = int(g); break; }
        if (found < 0) {
          found = int(dof_point.size());
          dof_point.push_back(X);
          owner.push_back({int(ip), k});
        }
        l2g[ip][k] = found;
      }
    }
    ndof = int(dof_point.size());

    // Elements = non-empty knot spans.
    elements.clear();
    for (size_t ip = 0; ip < patches.size(); ++ip) {
      const Patch& Pa = patches[ip];
      for (int su = Pa.p; su < Pa.nu; ++su) {
        if (Pa.U[su + 1] <= Pa.U[su]) continue;
        for (int sv = Pa.q; sv < Pa.nv; ++sv) {
          if (Pa.V[sv + 1] <= Pa.V[sv]) continue;
          Element e{int(ip), su, sv, {Pa.U[su], Pa.U[su + 1], Pa.V[sv], Pa.V[sv + 1]}, {}, 0.0};
          partitionElement(Pa, su, sv, e.rect, popt, 0, e.cells);
          PatchEval E;
          Vec3 c[4];
          int k = 0;
          for (double u : {e.rect.u0, e.rect.u1})
            for (double v : {e.rect.v0, e.rect.v1}) { Pa.evalSpan(su, sv, u, v, E); c[k++] = E.x; }
          e.h = std::max((c[3] - c[0]).norm(), (c[2] - c[1]).norm());
          elements.push_back(std::move(e));
        }
      }
    }

    // Collocation at the Greville point of each dof's first owner.
    colloc.clear();
    for (int g = 0; g < ndof; ++g) {
      const int ip = owner[g].first, k = owner[g].second;
      const Patch& Pa = patches[ip];
      const int i = k % Pa.nu, j = k / Pa.nu;
      Collocation c;
      c.patch = ip;
      c.xi = Vec2(Pa.greville(i, true), Pa.greville(j, false));
      c.su = findSpan(Pa.nu, Pa.p, c.xi(0), Pa.U);
      c.sv = findSpan(Pa.nv, Pa.q, c.xi(1), Pa.V);
      PatchEval E;
      Pa.evalSpan(c.su, c.sv, c.xi(0), c.xi(1), E);
      c.x = E.x;
      for (int l = 0; l < E.n_loc; ++l)
        if (std::abs(E.R[l]) > 1e-15) c.basis.push_back({l2g[ip][E.idx[l]], E.R[l]});
      colloc.push_back(std::move(c));
    }

    // Orientation: the volume (1/3) int x.n must be positive for outward n.
    normal_sign = volume() < 0 ? -1.0 : 1.0;
  }

  double volume() const {
    const GaussRule g = gaussLegendre(8);
    double V = 0.0;
    PatchEval E;
    for (const auto& e : elements)
      for (const auto& R : e.cells)
        for (size_t a = 0; a < g.x.size(); ++a)
          for (size_t b = 0; b < g.x.size(); ++b) {
            patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a],
                                      R.v0 + R.dv() * g.x[b], E);
            V += g.w[a] * g.w[b] * R.du() * R.dv() * E.x.dot(E.normal()) / 3.0;
          }
    return V * normal_sign;
  }

  // Closest point of an element to x: grid sample, then projected Gauss-Newton.
  Vec2 closestPoint(const Element& e, const Vec3& x, double& dist) const {
    const Patch& Pa = patches[e.patch];
    PatchEval E;
    Vec2 best;
    double bd = 1e300;
    for (int a = 0; a <= 6; ++a)
      for (int b = 0; b <= 6; ++b) {
        Vec2 p(e.rect.u0 + e.rect.du() * a / 6.0, e.rect.v0 + e.rect.dv() * b / 6.0);
        Pa.evalSpan(e.su, e.sv, p(0), p(1), E);
        const double d = (E.x - x).norm();
        if (d < bd) { bd = d; best = p; }
      }
    for (int it = 0; it < 30; ++it) {
      Pa.evalSpan(e.su, e.sv, best(0), best(1), E);
      Eigen::Matrix<double, 3, 2> J;
      J << E.xu, E.xv;
      const Vec2 step = (J.transpose() * J).ldlt().solve(J.transpose() * (x - E.x));
      const Vec2 nb = e.rect.clamp(best + step);
      if ((nb - best).norm() < 1e-15 * (1.0 + e.rect.du())) { best = nb; break; }
      best = nb;
    }
    Pa.evalSpan(e.su, e.sv, best(0), best(1), E);
    dist = (E.x - x).norm();
    return best;
  }
};

struct SolverOptions {
  int n_gauss = 8;          // integration points per direction per cell
  bool near_refine = true;  // App. B near-singular refinement (d <= 3h)
  int near_max_depth = 16;
};

}  // namespace igabem
