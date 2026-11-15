// Where a crack collocation row's 3 x ncols coefficients go. The dense path writes a 3 x ncols block of the system
// matrix; the hierarchical path (crack/h_crack.hpp) keeps only the row's nonzero columns (near elements, the finish
// at the row's own and coincident points, enrichment), so a row costs O(its support), not O(ncols).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <utility>
#include <vector>

namespace igabem {
namespace crack {

inline void addCol(Eigen::MatrixXd& a, int c, const Eigen::Vector3d& v) { a.col(c) += v; }
inline void addBlk(Eigen::MatrixXd& a, int c, const Eigen::Matrix3d& M) { a.block<3, 3>(0, c) += M; }

// Sparse row: (column, 3-vector) entries, duplicates summed by compress().
struct SparseRow {
  std::vector<std::pair<int, Eigen::Vector3d>> e;
  void compress() {
    std::sort(e.begin(), e.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    size_t k = 0;
    for (size_t i = 0; i < e.size(); ++i) {
      if (k > 0 && e[k - 1].first == e[i].first) e[k - 1].second += e[i].second;
      else e[k++] = e[i];
    }
    e.resize(k);
  }
};
inline void addCol(SparseRow& a, int c, const Eigen::Vector3d& v) { a.e.push_back({c, v}); }
inline void addBlk(SparseRow& a, int c, const Eigen::Matrix3d& M) {
  for (int k = 0; k < 3; ++k) a.e.push_back({c + k, M.col(k)});
}

// Dense window of a row: columns [lo, lo + a.cols()) kept, the rest dropped (the enrichment border).
struct WindowRow {
  int lo = 0;
  Eigen::Matrix<double, 3, Eigen::Dynamic> a;
  WindowRow(int lo_, int n) : lo(lo_), a(Eigen::Matrix<double, 3, Eigen::Dynamic>::Zero(3, n)) {}
};
inline void addCol(WindowRow& w, int c, const Eigen::Vector3d& v) {
  if (c >= w.lo && c < w.lo + w.a.cols()) w.a.col(c - w.lo) += v;
}
inline void addBlk(WindowRow& w, int c, const Eigen::Matrix3d& M) {
  if (c + 3 > w.lo && c < w.lo + w.a.cols())
    for (int k = 0; k < 3; ++k) addCol(w, c + k, M.col(k));
}

}  // namespace crack
}  // namespace igabem
