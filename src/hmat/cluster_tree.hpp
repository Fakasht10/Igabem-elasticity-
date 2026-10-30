// Binary cluster tree by bisection at the median across the longest extent of
// the bounding box, over points with an optional radius each (a collocation
// point has radius 0, an element its bounding ball). Indices are permuted so
// every node is a contiguous range perm[lo, hi).
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../mesh.hpp"

namespace igabem {

struct ClusterTree {
  struct Node {
    int lo, hi;
    int kid[2] = {-1, -1};
    Vec3 bmin, bmax;  // box of the balls
    bool leaf() const { return kid[0] < 0; }
    int size() const { return hi - lo; }
    double diam() const { return (bmax - bmin).norm(); }
    Vec3 centre() const { return 0.5 * (bmin + bmax); }
  };
  std::vector<Node> nodes;  // nodes[0]: root
  std::vector<int> perm;

  void build(const std::vector<Vec3>& x, const std::vector<double>& rad, int leaf) {
    const int n = int(x.size());
    perm.resize(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    nodes.clear();
    nodes.reserve(4 * n / std::max(leaf, 1) + 4);
    split(x, rad, 0, n, leaf);
  }

  // Distance between the boxes of two nodes (0 if they overlap).
  static double distance(const Node& a, const Node& b) {
    double s = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double g = std::max({0.0, a.bmin(d) - b.bmax(d), b.bmin(d) - a.bmax(d)});
      s += g * g;
    }
    return std::sqrt(s);
  }

 private:
  int split(const std::vector<Vec3>& x, const std::vector<double>& rad, int lo, int hi, int leaf) {
    const int id = int(nodes.size());
    nodes.push_back(Node{lo, hi});
    Vec3 bmin = Vec3::Constant(1e300), bmax = -bmin, cmin = bmin, cmax = bmax;
    for (int i = lo; i < hi; ++i) {
      const int p = perm[i];
      const double r = rad.empty() ? 0.0 : rad[p];
      bmin = bmin.cwiseMin(x[p] - Vec3::Constant(r));
      bmax = bmax.cwiseMax(x[p] + Vec3::Constant(r));
      cmin = cmin.cwiseMin(x[p]);
      cmax = cmax.cwiseMax(x[p]);
    }
    nodes[id].bmin = bmin;
    nodes[id].bmax = bmax;
    if (hi - lo > leaf) {
      int d;
      (cmax - cmin).maxCoeff(&d);
      const int mid = (lo + hi) / 2;
      std::nth_element(perm.begin() + lo, perm.begin() + mid, perm.begin() + hi,
                       [&](int a, int b) { return x[a](d) < x[b](d); });
      const int k0 = split(x, rad, lo, mid, leaf);
      const int k1 = split(x, rad, mid, hi, leaf);
      nodes[id].kid[0] = k0;
      nodes[id].kid[1] = k1;
    }
    return id;
  }
};

}  // namespace igabem
