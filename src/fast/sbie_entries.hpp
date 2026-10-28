// Entries of the SBIE operators H (raw, without the rigid-body diagonal term)
// and G for any rows x columns, as needed by hierarchical compression: rows are
// collocation points (3 rows each), columns displacement dofs (H) or traction
// dofs (G), 3 columns each. An entry is the same quadrature as in
// SbieAssembler::assembleHG restricted to the elements supporting the requested
// columns, so every entry equals the dense one up to round-off.
//
// The rigid-body term -N_B(x_a) sum_Gamma int T is not an entry of a local
// integral (it spans the whole boundary); CompressedSbie adds it afterwards from
// the compressed raw H (partition of unity: sum_B Hraw(a, B) = int_Gamma T).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "../integrator.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"
#include "../traction_space.hpp"

namespace igabem {

class SbieEntries {
 public:
  enum Kind { H, G };

  SbieEntries(const Mesh& m, const TractionSpace& ts, const Material& mat, const SolverOptions& o,
              const std::vector<Collocation>& rows, Kind kind)
      : mesh_(m), ts_(ts), K_(mat), quad_(m, o), rows_(rows), kind_(kind) {
    ncol_ = kind == H ? m.ndof : ts.ndof;
    support_.assign(ncol_, {});
    elem_cols_.assign(m.elements.size(), {});
    for (size_t ie = 0; ie < m.elements.size(); ++ie) {
      const Element& e = m.elements[ie];
      const Patch& P = m.patches[e.patch];
      PatchEval E;
      P.evalSpan(e.su, e.sv, 0.5 * (e.rect.u0 + e.rect.u1), 0.5 * (e.rect.v0 + e.rect.v1), E);
      for (int l = 0; l < E.n_loc; ++l) {
        const int c = column(e.patch, E.idx[l]);
        support_[c].push_back(int(ie));
        elem_cols_[ie].push_back(c);
      }
    }
  }

  int rows() const { return 3 * int(rows_.size()); }
  int cols() const { return 3 * ncol_; }

  // Column of local control point idx of patch p.
  int column(int p, int idx) const { return kind_ == H ? mesh_.l2g[p][idx] : ts_.offset[p] + idx; }

  // Anchor point of each column (3 per dof: x, y, z) and the radius of a ball
  // around it containing the support of its basis function, for clustering.
  void columnGeometry(std::vector<double>& xyz, std::vector<double>& radius) const {
    xyz.resize(3 * size_t(cols()));
    radius.resize(size_t(cols()));
    std::vector<Vec3> lo(mesh_.elements.size()), hi(mesh_.elements.size());
    for (size_t ie = 0; ie < mesh_.elements.size(); ++ie) elementBox(ie, lo[ie], hi[ie]);
    for (int c = 0; c < ncol_; ++c) {
      const Vec3 x = kind_ == H ? mesh_.colloc[c].x : ts_.points[c].x;
      double r = 0.0;
      for (int ie : support_[c])
        for (int k = 0; k < 8; ++k) {
          const Vec3 corner((k & 1) ? hi[ie](0) : lo[ie](0), (k & 2) ? hi[ie](1) : lo[ie](1),
                            (k & 4) ? hi[ie](2) : lo[ie](2));
          r = std::max(r, (corner - x).norm());
        }
      for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 3; ++d) xyz[3 * (3 * size_t(c) + i) + d] = x(d);
        radius[3 * size_t(c) + i] = r;
      }
    }
  }

  void rowGeometry(std::vector<double>& xyz) const {
    xyz.resize(3 * size_t(rows()));
    for (size_t a = 0; a < rows_.size(); ++a)
      for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) xyz[3 * (3 * a + i) + d] = rows_[a].x(d);
  }

  // f(k, acc) for each row point rows[k], acc holding its 3 x 3 blocks against
  // the column dofs col_dofs (9 doubles per dof, column-major, in their order).
  // Each (row point, element) pair is integrated once.
  template <class F>
  void rowsAgainst(const std::vector<int>& row_points, const std::vector<int>& col_dofs, F&& f) const {
    Scratch& s = scratch();
    s.pos.resize(ncol_, -1);
    s.emark.resize(mesh_.elements.size(), 0);
    s.elems.clear();
    for (size_t k = 0; k < col_dofs.size(); ++k) s.pos[col_dofs[k]] = int(k);
    for (int c : col_dofs)
      for (int ie : support_[c])
        if (!s.emark[ie]) { s.emark[ie] = 1; s.elems.push_back(ie); }
    s.acc.resize(9 * col_dofs.size());
    for (size_t k = 0; k < row_points.size(); ++k) {
      rowBlocks(row_points[k], s);
      f(int(k), s.acc.data());
    }
    for (int c : col_dofs) s.pos[c] = -1;
    for (int ie : s.elems) s.emark[ie] = 0;
  }

  // out(i + M j) = A(rows[i], cols[j]), column-major M x N.
  void block(int M, int N, const int* rws, const int* cls, double* out) const {
    std::vector<int> pts, dofs;
    std::vector<int> rp(M), cp(N);  // row -> index in pts, column -> index in dofs
    for (int i = 0; i < M; ++i) {
      if (pts.empty() || pts.back() != rws[i] / 3) pts.push_back(rws[i] / 3);
      rp[i] = int(pts.size()) - 1;
    }
    Scratch& s = scratch();
    s.pos.resize(ncol_, -1);
    for (int j = 0; j < N; ++j) {
      const int c = cls[j] / 3;
      if (s.pos[c] < 0) { s.pos[c] = int(dofs.size()); dofs.push_back(c); }
      cp[j] = s.pos[c];
    }
    for (int c : dofs) s.pos[c] = -1;
    int i0 = 0;
    rowsAgainst(pts, dofs, [&](int k, const double* acc) {
      for (; i0 < M && rp[i0] == k; ++i0) {
        const int ri = rws[i0] % 3;
        for (int j = 0; j < N; ++j) out[i0 + size_t(M) * j] = acc[9 * size_t(cp[j]) + 3 * (cls[j] % 3) + ri];
      }
    });
  }

 private:
  struct Scratch {
    std::vector<int> pos, elems;
    std::vector<char> emark;
    std::vector<double> acc;  // per column dof: 3 x 3 block, column-major
  };
  static Scratch& scratch() {
    thread_local Scratch s;
    return s;
  }

  const Mesh& mesh_;
  const TractionSpace& ts_;
  Kelvin K_;
  ElementIntegrator quad_;
  const std::vector<Collocation>& rows_;
  Kind kind_;
  int ncol_ = 0;
  std::vector<std::vector<int>> support_;    // per column dof: supporting elements
  std::vector<std::vector<int>> elem_cols_;  // per element: column dofs

  // 3 x 3 blocks of row point a against the column dofs marked in s.pos, into s.acc.
  void rowBlocks(int a, Scratch& s) const {
    std::fill(s.acc.begin(), s.acc.end(), 0.0);
    const Collocation& c = rows_[a];
    const SourcePoint src{c.patch, c.xi, c.x};
    for (int ie : s.elems) {
      const Element& e = mesh_.elements[ie];
      quad_.integrate(src, e, [&](const QPoint& E, double w) {
        const Vec3 nJ = E.normal() * mesh_.normal_sign;
        const double J = nJ.norm();
        Eigen::Matrix3d U, T;
        K_.eval(E.x - c.x, Vec3(nJ / J), U, T);
        const Eigen::Matrix3d Kw = (kind_ == H ? T : U) * (w * J);
        for (int l = 0; l < E.n_loc; ++l) {
          const int p = s.pos[column(e.patch, E.idx[l])];
          if (p < 0) continue;
          Eigen::Map<Eigen::Matrix3d>(&s.acc[9 * size_t(p)]) += E.R[l] * Kw;
        }
      });
    }
  }

  void elementBox(size_t ie, Vec3& lo, Vec3& hi) const {
    const Element& e = mesh_.elements[ie];
    PatchEval E;
    lo = Vec3::Constant(1e300);
    hi = Vec3::Constant(-1e300);
    for (int i = 0; i <= 4; ++i)
      for (int j = 0; j <= 4; ++j) {
        mesh_.patches[e.patch].evalSpan(e.su, e.sv, e.rect.u0 + e.rect.du() * i / 4.0,
                                        e.rect.v0 + e.rect.dv() * j / 4.0, E);
        lo = lo.cwiseMin(E.x);
        hi = hi.cwiseMax(E.x);
      }
    const Vec3 m = 0.05 * (hi - lo).norm() * Vec3::Ones();  // curvature margin
    lo -= m;
    hi += m;
  }
};

}  // namespace igabem
