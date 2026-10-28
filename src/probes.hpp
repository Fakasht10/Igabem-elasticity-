// Canned field probes:
//   lineProbe      straight segment through the solid (interior evaluation,
//                  boundary recovery at end points on the boundary)
//   boundaryProbe  a curve on the boundary, e.g. around a hole (boundary
//                  recovery only; points are projected onto the chosen patches)
// and a CSV writer with every displacement, strain and stress component, von
// Mises, and the exact field next to it when one is given. python/plot_probes.py
// turns the CSVs into figures (Cartesian or cylindrical components).
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "interior.hpp"
#include "recovery.hpp"
#include "solution.hpp"

namespace igabem {

struct Probe {
  std::vector<double> s;  // arc length from the first point
  std::vector<FieldSample> f;
};

inline Probe lineProbe(const InteriorEvaluator& ev, const Vec3& a, const Vec3& b, int n) {
  std::vector<Vec3> xs;
  for (int i = 0; i <= n; ++i) xs.push_back(a + (b - a) * (double(i) / n));
  Probe p;
  p.f = ev.eval(xs);
  for (int i = 0; i <= n; ++i) p.s.push_back((xs[i] - a).norm());
  return p;
}

// curve(s), s in [0, 1]; use(patch) selects the patches the points are projected on.
template <class Pred>
inline Probe boundaryProbe(const BoundarySolution& sol, const std::function<Vec3(double)>& curve, int n,
                           Pred&& use) {
  Probe p;
  p.f.resize(n + 1);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i <= n; ++i) {
    const SurfacePoint sp = locate(*sol.mesh, curve(double(i) / n), use);
    FieldSample& f = p.f[i];
    f.x = sp.E.x;
    f.u = sol.displacement(sp);
    f.grad = sol.gradient(sp);
    f.strain = strainOf(f.grad);
    f.stress = sol.mat.stress(f.grad);
    f.winding = 0.5;
    f.on_boundary = true;
  }
  double s = 0.0;
  for (int i = 0; i <= n; ++i) {
    if (i) s += (p.f[i].x - p.f[i - 1].x).norm();
    p.s.push_back(s);
  }
  return p;
}

inline void writeProbeCsv(const std::string& path, const Probe& p, const ExactField& exact = nullptr) {
  std::ofstream o(path);
  o.precision(12);
  const char* sym[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
  const int I[6] = {0, 1, 2, 0, 0, 1}, J[6] = {0, 1, 2, 1, 2, 2};
  o << "s,x,y,z,u_x,u_y,u_z";
  for (const char* c : sym) o << ",s_" << c;
  for (const char* c : sym) o << ",e_" << c;
  o << ",von_mises,on_boundary";
  if (exact) {
    o << ",u_x_ex,u_y_ex,u_z_ex";
    for (const char* c : sym) o << ",s_" << c << "_ex";
    o << ",von_mises_ex";
  }
  o << '\n';
  for (size_t i = 0; i < p.f.size(); ++i) {
    const FieldSample& f = p.f[i];
    o << p.s[i] << ',' << f.x(0) << ',' << f.x(1) << ',' << f.x(2) << ',' << f.u(0) << ',' << f.u(1) << ','
      << f.u(2);
    for (int k = 0; k < 6; ++k) o << ',' << f.stress(I[k], J[k]);
    for (int k = 0; k < 6; ++k) o << ',' << f.strain(I[k], J[k]);
    o << ',' << vonMises(f.stress) << ',' << int(f.on_boundary);
    if (exact) {
      Vec3 u;
      Eigen::Matrix3d S;
      exact(f.x, u, S);
      o << ',' << u(0) << ',' << u(1) << ',' << u(2);
      for (int k = 0; k < 6; ++k) o << ',' << S(I[k], J[k]);
      o << ',' << vonMises(S);
    }
    o << '\n';
  }
}

}  // namespace igabem
