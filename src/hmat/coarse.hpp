// Affine coarse space for the hierarchical solvers (bemkit's coarse correction):
// the unknowns of each group (displacement or traction components) are clustered
// by position into aggregates of at most `agg` points, and every aggregate gets
// the modes 1, x, y, z per component, normalized. Trial (Zc) and test (Zr) sides
// share the aggregates; unknown k pairs with row k (its collocation point), so
// E = Zr^T A Zc is square.
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "cluster_tree.hpp"

namespace igabem {

// Sparse n x nc basis stored by row: entries of row i at [ptr[i], ptr[i + 1]).
struct SparseBasis {
  int n = 0, nc = 0;
  std::vector<int> ptr{0}, id;
  std::vector<double> val;

  // y (nc) += Z^T v
  void applyT(const double* v, double* y) const {
    for (int i = 0; i < n; ++i)
      for (int p = ptr[i]; p < ptr[i + 1]; ++p) y[id[p]] += val[p] * v[i];
  }
  // v (n) = Z y
  void apply(const double* y, double* v) const {
    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int p = ptr[i]; p < ptr[i + 1]; ++p) s += val[p] * y[id[p]];
      v[i] = s;
    }
  }
  // Z with its rows placed at map[i] of an m-row basis (rows not hit stay empty).
  SparseBasis scatter(const std::vector<int>& map, int m) const {
    std::vector<std::vector<std::pair<int, double>>> rows(m);
    for (int i = 0; i < n; ++i)
      for (int p = ptr[i]; p < ptr[i + 1]; ++p) rows[map[i]].push_back({id[p], val[p]});
    SparseBasis o;
    o.n = m;
    o.nc = nc;
    for (int r = 0; r < m; ++r) {
      for (const auto& [c, v] : rows[r]) { o.id.push_back(c); o.val.push_back(v); }
      o.ptr.push_back(int(o.id.size()));
    }
    return o;
  }
};

// xt / xr: trial / test position of unknown k, comp: its component (0..2), group: its kind.
inline void affineCoarse(const std::vector<Vec3>& xt, const std::vector<Vec3>& xr, const std::vector<int>& comp,
                         const std::vector<int>& group, int agg, SparseBasis& Zc, SparseBasis& Zr) {
  const int n = int(xt.size());
  std::vector<std::vector<std::pair<int, double>>> ec(n), er(n);
  int nc = 0, ngroup = 0;
  for (int g : group) ngroup = std::max(ngroup, g + 1);
  for (int g = 0; g < ngroup; ++g) {
    std::vector<int> idx;
    std::vector<Vec3> x;
    for (int k = 0; k < n; ++k)
      if (group[k] == g) { idx.push_back(k); x.push_back(xt[k]); }
    if (idx.empty()) continue;
    ClusterTree T;
    T.build(x, {}, 3 * agg);
    for (const ClusterTree::Node& L : T.nodes) {
      if (!L.leaf()) continue;
      const double h = std::max(1e-300, 0.5 * L.diam());
      for (int c = 0; c < 3; ++c) {
        std::vector<int> ks;
        for (int i = L.lo; i < L.hi; ++i)
          if (comp[idx[T.perm[i]]] == c) ks.push_back(idx[T.perm[i]]);
        if (ks.empty()) continue;
        Vec3 ct = Vec3::Zero(), cr = Vec3::Zero();
        for (int k : ks) { ct += xt[k]; cr += xr[k]; }
        ct /= double(ks.size());
        cr /= double(ks.size());
        for (int f = 0; f < 4; ++f) {
          auto ft = [&](int k) { return f == 0 ? 1.0 : (xt[k](f - 1) - ct(f - 1)) / h; };
          auto fr = [&](int k) { return f == 0 ? 1.0 : (xr[k](f - 1) - cr(f - 1)) / h; };
          double st = 0.0, sr = 0.0;
          for (int k : ks) { st += ft(k) * ft(k); sr += fr(k) * fr(k); }
          if (st < 1e-8 * double(ks.size()) || sr < 1e-8 * double(ks.size())) continue;  // flat aggregate
          st = 1.0 / std::sqrt(st);
          sr = 1.0 / std::sqrt(sr);
          for (int k : ks) {
            ec[k].push_back({nc, ft(k) * st});
            er[k].push_back({nc, fr(k) * sr});
          }
          ++nc;
        }
      }
    }
  }
  auto pack = [&](const std::vector<std::vector<std::pair<int, double>>>& e, SparseBasis& Z) {
    Z = SparseBasis();
    Z.n = n;
    Z.nc = nc;
    for (int k = 0; k < n; ++k) {
      for (const auto& [c, v] : e[k]) { Z.id.push_back(c); Z.val.push_back(v); }
      Z.ptr.push_back(int(Z.id.size()));
    }
  };
  pack(ec, Zc);
  pack(er, Zr);
}

}  // namespace igabem
