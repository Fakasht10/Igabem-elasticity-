// VTK XML unstructured-grid (.vtu, ASCII) output for ParaView:
//   writeSurfaceVtu  the whole boundary, each element sampled on a sub x sub
//                    grid: displacement, traction, stress and strain tensors,
//                    von Mises, patch id, and the exact field when given
//   writeGridVtu     a mapped (nu+1) x (nv+1) grid of interior samples (e.g. a
//                    section through the solid from InteriorEvaluator)
#pragma once

#include <Eigen/Dense>
#include <array>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "interior.hpp"
#include "probes.hpp"
#include "solution.hpp"

namespace igabem {

struct VtuData {
  std::vector<Vec3> points;
  std::vector<std::array<int, 4>> quads;
  std::vector<std::pair<std::string, int>> fields;  // name, components (point data)
  std::vector<std::vector<double>> values;          // per field, components x points
  std::vector<int> cell_patch;

  int field(const std::string& name, int ncomp) {
    fields.push_back({name, ncomp});
    values.emplace_back();
    return int(fields.size()) - 1;
  }
  void push(int k, const Vec3& v) { values[k].insert(values[k].end(), {v(0), v(1), v(2)}); }
  void push(int k, const Eigen::Matrix3d& A) {
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) values[k].push_back(A(i, j));
  }
  void push(int k, double v) { values[k].push_back(v); }

  void write(const std::string& path) const {
    std::ofstream o(path);
    o.precision(10);
    o << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
      << "<UnstructuredGrid>\n<Piece NumberOfPoints=\"" << points.size() << "\" NumberOfCells=\"" << quads.size()
      << "\">\n<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const Vec3& p : points) o << p(0) << ' ' << p(1) << ' ' << p(2) << '\n';
    o << "</DataArray>\n</Points>\n<Cells>\n<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& q : quads) o << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << '\n';
    o << "</DataArray>\n<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (size_t i = 0; i < quads.size(); ++i) o << 4 * (i + 1) << '\n';
    o << "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (size_t i = 0; i < quads.size(); ++i) o << "9\n";
    o << "</DataArray>\n</Cells>\n<PointData>\n";
    for (size_t k = 0; k < fields.size(); ++k) {
      o << "<DataArray type=\"Float64\" Name=\"" << fields[k].first << "\" NumberOfComponents=\""
        << fields[k].second << "\" format=\"ascii\">\n";
      const auto& v = values[k];
      for (size_t i = 0; i < v.size(); ++i) o << v[i] << ((i + 1) % fields[k].second ? ' ' : '\n');
      o << "</DataArray>\n";
    }
    o << "</PointData>\n";
    if (!cell_patch.empty()) {
      o << "<CellData>\n<DataArray type=\"Int32\" Name=\"patch\" format=\"ascii\">\n";
      for (int p : cell_patch) o << p << '\n';
      o << "</DataArray>\n</CellData>\n";
    }
    o << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
  }
};

// Adds displacement, stress, strain, von Mises (and exact stress, von Mises,
// stress error) for one sample.
struct FieldColumns {
  int u, sig, eps, vm, sig_ex = -1, vm_ex = -1, err = -1;
  FieldColumns(VtuData& d, bool exact)
      : u(d.field("displacement", 3)), sig(d.field("stress", 9)), eps(d.field("strain", 9)),
        vm(d.field("von_mises", 1)) {
    if (exact) {
      sig_ex = d.field("stress_exact", 9);
      vm_ex = d.field("von_mises_exact", 1);
      err = d.field("stress_error", 1);  // |sigma_h - sigma| / max |sigma|, normalized after
    }
  }
  void push(VtuData& d, const Vec3& x, const Vec3& u_, const Eigen::Matrix3d& S, const Eigen::Matrix3d& E,
            const ExactField& exact) const {
    d.push(u, u_);
    d.push(sig, S);
    d.push(eps, E);
    d.push(vm, vonMises(S));
    if (exact) {
      Vec3 ue;
      Eigen::Matrix3d Se;
      exact(x, ue, Se);
      d.push(sig_ex, Se);
      d.push(vm_ex, vonMises(Se));
      d.push(err, (S - Se).norm());
    }
  }
  void normalizeError(VtuData& d) const {
    if (err < 0) return;
    double smax = 0.0;
    for (size_t i = 0; i < d.values[vm_ex].size(); ++i) {
      Eigen::Map<const Eigen::Matrix3d> Se(&d.values[sig_ex][9 * i]);
      smax = std::max(smax, Se.norm());
    }
    if (smax > 0.0) for (double& v : d.values[err]) v /= smax;
  }
};

inline VtuData surfaceVtu(const BoundarySolution& sol, int sub = 4, const ExactField& exact = nullptr) {
  const Mesh& m = *sol.mesh;
  VtuData d;
  FieldColumns col(d, bool(exact));
  const int tr = d.field("traction", 3);
  for (const Element& e : m.elements) {
    const int base = int(d.points.size());
    for (int j = 0; j <= sub; ++j)
      for (int i = 0; i <= sub; ++i) {
        SurfacePoint sp;
        sp.patch = e.patch;
        sp.su = e.su;
        sp.sv = e.sv;
        sp.xi = Vec2(e.rect.u0 + e.rect.du() * i / sub, e.rect.v0 + e.rect.dv() * j / sub);
        m.patches[e.patch].evalSpan(e.su, e.sv, sp.xi(0), sp.xi(1), sp.E);
        sp.N = (sp.E.normal() * m.normal_sign).normalized();
        const Eigen::Matrix3d G = sol.gradient(sp);
        d.points.push_back(sp.E.x);
        col.push(d, sp.E.x, sol.displacement(sp), sol.mat.stress(G), strainOf(G), exact);
        d.push(tr, sol.traction(sp));
      }
    for (int j = 0; j < sub; ++j)
      for (int i = 0; i < sub; ++i) {
        const int a = base + j * (sub + 1) + i;
        d.quads.push_back({a, a + 1, a + sub + 2, a + sub + 1});
        d.cell_patch.push_back(e.patch);
      }
  }
  col.normalizeError(d);
  return d;
}

// Interior samples on a mapped grid x(s, r), s in [0, 1] (nu + 1 points), r in
// [0, 1] (nv + 1 points).
template <class Map>
inline VtuData gridVtu(const InteriorEvaluator& ev, Map&& map, int nu, int nv, const ExactField& exact = nullptr) {
  VtuData d;
  FieldColumns col(d, bool(exact));
  for (int j = 0; j <= nv; ++j)
    for (int i = 0; i <= nu; ++i) d.points.push_back(map(double(i) / nu, double(j) / nv));
  const std::vector<FieldSample> f = ev.eval(d.points);
  for (size_t k = 0; k < f.size(); ++k) col.push(d, f[k].x, f[k].u, f[k].stress, f[k].strain, exact);
  for (int j = 0; j < nv; ++j)
    for (int i = 0; i < nu; ++i) {
      const int a = j * (nu + 1) + i;
      d.quads.push_back({a, a + 1, a + nu + 2, a + nu + 1});
    }
  col.normalizeError(d);
  return d;
}

}  // namespace igabem
