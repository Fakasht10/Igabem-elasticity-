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
#include <Eigen/Sparse>
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

  // Quadrature of the entries for elements that get a cached rule: error estimate
  // tol, Gauss order at least min_order (0, 0: the dense assembly's rules, so that
  // entries equal the dense ones up to round-off).
  void setEntryRule(double tol, int min_order) { near_tol_ = tol; near_order_ = min_order; }
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

  // Far-field factorization of a block, A(rows, cols) = K W: K the kernel (T or U,
  // times the quadrature weight) between the row points and the quadrature points
  // of the elements supporting the columns, W the basis values (sparse). Each
  // element gets the cached rule that is valid for every row point (the one the
  // dense assembly uses for the closest of them, or finer), so K costs one kernel
  // evaluation per entry and has rows and columns equally cheap. The rules are
  // chosen for the quadrature tolerance tol (at most the assembly's far_tol is
  // needed for entries that the ACA reproduces only to its own tolerance). False if some
  // element is too close to the rows for a cached rule.
  struct FarBlock {
    std::vector<Vec3> y, n;
    std::vector<double> w;
    // Per element: its points q0 .. q0 + nq - 1, the basis values R (n_loc x nq,
    // column-major, from the integrator's cache) and which local basis functions
    // are block dofs (l, dof index p, in order of first appearance). Column j of
    // the block is component col_comp[j] of dof col_pt[j]:
    //   A(:, j) = sum_q K(:, 3 q + col_comp[j]) N_{col_pt[j]}(y_q).
    struct Elem {
      int q0, nq, n_loc;
      const double* R;
      std::vector<std::pair<int, int>> lp;
    };
    std::vector<Elem> elems;
    std::vector<std::vector<std::pair<int, int>>> dof_elems;  // per dof p: (element, l)
    std::vector<int> col_pt, col_comp;
    int points() const { return int(y.size()); }
  };

  bool farBlock(int M, const int* rws, int N, const int* cls, double tol, int min_order, FarBlock& fb) const {
    Vec3 c = Vec3::Zero();
    for (int i = 0; i < M; ++i) c += rows_[rws[i] / 3].x;
    c /= M;
    double rho = 0.0;
    for (int i = 0; i < M; ++i) rho = std::max(rho, (rows_[rws[i] / 3].x - c).norm());
    std::vector<std::vector<int>> at;  // per column dof in the block: its columns
    std::vector<int> dofs;
    Scratch& s = scratch();
    s.pos.resize(ncol_, -1);
    for (int j = 0; j < N; ++j) {
      const int d = cls[j] / 3;
      if (s.pos[d] < 0) { s.pos[d] = int(dofs.size()); dofs.push_back(d); at.emplace_back(); }
      at[s.pos[d]].push_back(j);
    }
    std::vector<int> elems;
    s.emark.resize(mesh_.elements.size(), 0);
    for (int d : dofs)
      for (int ie : support_[d])
        if (!s.emark[ie]) { s.emark[ie] = 1; elems.push_back(ie); }
    for (int ie : elems) s.emark[ie] = 0;
    fb.y.clear();
    fb.n.clear();
    fb.w.clear();
    fb.elems.clear();
    fb.dof_elems.assign(dofs.size(), {});
    bool ok = true;
    ElementIntegrator::RuleData r;
    for (int ie : elems) {
      const Element& e = mesh_.elements[ie];
      const double d = (quad_.elementCentre(ie) - c).norm() - rho;
      ok = quad_.farRuleData(size_t(ie), d, tol, min_order, r);
      if (!ok) break;
      FarBlock::Elem E{fb.points(), r.n, r.n_loc, r.R, {}};
      for (int l = 0; l < r.n_loc; ++l) {
        const int p = s.pos[column(e.patch, r.idx[l])];
        if (p < 0) continue;
        E.lp.emplace_back(l, p);
        fb.dof_elems[p].emplace_back(int(fb.elems.size()), l);
      }
      fb.elems.push_back(std::move(E));
      for (int q = 0; q < r.n; ++q) {
        const Vec3 nJ = r.nJ[q] * mesh_.normal_sign;
        const double J = nJ.norm();
        fb.y.push_back(r.x[q]);
        fb.n.push_back(nJ / J);
        fb.w.push_back(r.w[q] * J);
      }
    }
    fb.col_pt.resize(N);
    fb.col_comp.resize(N);
    for (int j = 0; j < N; ++j) { fb.col_pt[j] = s.pos[cls[j] / 3]; fb.col_comp[j] = cls[j] % 3; }
    for (int d : dofs) s.pos[d] = -1;
    return ok;
  }

  // 3 x 3 kernel block of row point a and quadrature point q of fb, times its
  // weight, into out (column-major, leading dimension ld).
  void kernel(int a, const FarBlock& fb, int q, double* out, int ld) const {
    if (kind_ == H) K_.evalT(fb.y[q] - rows_[a].x, fb.n[q], fb.w[q], out, ld);
    else K_.evalU(fb.y[q] - rows_[a].x, fb.w[q], out, ld);
  }

 private:
  struct Scratch {
    std::vector<int> pos, elems;
    std::vector<char> emark;
    std::vector<double> acc;  // per column dof: 3 x 3 block, column-major
    Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor> kw;
    Eigen::Matrix<double, Eigen::Dynamic, 9> ke;
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
  double near_tol_ = 0.0;
  int near_order_ = 0;
  std::vector<std::vector<int>> support_;    // per column dof: supporting elements
  std::vector<std::vector<int>> elem_cols_;  // per element: column dofs

  // 3 x 3 blocks of row point a against the column dofs marked in s.pos, into s.acc.
  // Elements far enough for a cached rule are done in a batch: kernel times weight
  // at the rule's points (n x 9), contracted with the basis values in one product.
  void rowBlocks(int a, Scratch& s) const {
    std::fill(s.acc.begin(), s.acc.end(), 0.0);
    const Collocation& c = rows_[a];
    const SourcePoint src{c.patch, c.xi, c.x};
    ElementIntegrator::RuleData r;
    for (int ie : s.elems) {
      const Element& e = mesh_.elements[ie];
      const bool own = e.patch == c.patch && e.rect.contains(c.xi, 1e-14);
      if (!own && quad_.farRuleData(size_t(ie), (quad_.elementCentre(ie) - c.x).norm(), near_tol_, near_order_, r)) {
        s.kw.resize(r.n, 9);
        for (int q = 0; q < r.n; ++q) {
          const Vec3 nJ = r.nJ[q] * mesh_.normal_sign;
          const double J = nJ.norm();
          if (kind_ == H) K_.evalT(r.x[q] - c.x, Vec3(nJ / J), r.w[q] * J, s.kw.data() + 9 * q, 3);
          else K_.evalU(r.x[q] - c.x, r.w[q] * J, s.kw.data() + 9 * q, 3);
        }
        s.ke.noalias() = Eigen::Map<const Eigen::MatrixXd>(r.R, r.n_loc, r.n) * s.kw;
        for (int l = 0; l < r.n_loc; ++l) {
          const int p = s.pos[column(e.patch, r.idx[l])];
          if (p < 0) continue;
          double* A = &s.acc[9 * size_t(p)];
          for (int k = 0; k < 9; ++k) A[k] += s.ke(l, k);
        }
        continue;
      }
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
