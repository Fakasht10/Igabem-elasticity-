// Crack growth on a cached operator: assemble step k+1 from step k's raw rows.
//
// A collocation row is a sum of raw per-element contributions (Kelvin kernel times the element's shape functions,
// and the load integrals) plus a finish that depends only on the element sums of the regularization (RowAcc) and on
// the row's own point (own/mirror shapes, the closed-form open-face coefficients). So for a row present in both
// steps (same patch geometry, same point)
//
//   raw_{k+1} = P(raw_k - sum_{removed elements} raw_k(e)) + sum_{added elements} raw_{k+1}(e),
//   acc_{k+1} = acc_k - acc_k(removed) + acc_{k+1}(added),
//
// with P the map of the columns that survive (same dof by patch geometry and local index). An element is the same
// in both steps when its patch geometry, parameter rectangle, quadrature cells, face role, Dirichlet flag and (if
// enriched) the front all agree; the front's move changes every enriched element, so the moving window and the
// crown are recomputed, the far body is not. New rows (the grown crack band) are assembled in full; every row is
// finished afresh, then the tying rows and the LU solve run as in CrackModel::solve. The mesh generator must keep
// the patches outside the window fixed (make_crack.senb_grow).
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dual_model.hpp"

namespace igabem {
namespace crack {

struct GrowthStats {
  long rows_reused = 0, rows_new = 0, elems_removed = 0, elems_added = 0, elems = 0;
  double t_raw = 0, t_finish = 0, t_solve = 0, t_quad = 0, t_rem = 0, t_add = 0;
};

struct Growth {
  // One step's raw operator, kept for the next step.
  struct Raw {
    std::vector<CrackModel::Row> rows;
    std::vector<std::string> row_key, col_key, elem_key;
    Eigen::MatrixXd A, B;  // raw collocation rows (3 per point), unfinished
    std::vector<CrackModel::RowAcc> acc;
    std::unique_ptr<ElementIntegrator> quad;
  };
  using RawOperator = Raw;

  static std::string num(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.13g,", v);
    return b;
  }
  static std::string patchKey(const Patch& P) {
    std::string k = std::to_string(P.p) + "," + std::to_string(P.q) + "|";
    for (double u : P.U) k += num(u);
    k += "|";
    for (double v : P.V) k += num(v);
    k += "|";
    for (const Vec4& w : P.Pw) for (int i = 0; i < 4; ++i) k += num(w(i));
    return std::to_string(std::hash<std::string>{}(k));
  }
  static std::string frontKey(const CrackModel& M) {
    std::string k;
    for (const FrontPiece& F : M.front.pieces) k += patchKey(*F.P);
    return k;
  }
  static std::vector<std::string> patchKeys(const CrackModel& M) {
    std::vector<std::string> out;
    for (const Patch& P : M.mesh.patches) out.push_back(patchKey(P));
    return out;
  }

  static void keys(const CrackModel& M, const std::vector<CrackModel::Row>& rows, RawOperator& R) {
    const std::vector<std::string> pk = patchKeys(M);
    const std::string fk = frontKey(M);
    R.row_key.clear();
    for (const auto& r : rows) R.row_key.push_back(pk[r.patch] + "/" + std::to_string(int(r.kind)) + "/" + num(r.xi(0)) + num(r.xi(1)));
    R.elem_key.clear();
    for (const Element& e : M.mesh.elements) {
      std::string k = pk[e.patch] + "/" + std::to_string(M.face[e.patch]) + std::to_string(int(M.dirichlet[e.patch])) + "/" +
                      num(e.rect.u0) + num(e.rect.u1) + num(e.rect.v0) + num(e.rect.v1);
      for (const Rect& c : e.cells) k += num(c.u0) + num(c.u1) + num(c.v0) + num(c.v1);
      if (M.opt.enrich && M.piece[e.patch] >= 0) k += "/F" + fk;
      R.elem_key.push_back(k);
    }
    // columns: a dof is named by the smallest (patch key, local index) that holds it; tractions by patch and
    // index; enrichment columns never match (the front moved)
    R.col_key.assign(M.ncols, "");
    std::vector<std::string> dk(M.mesh.ndof);
    for (size_t p = 0; p < M.mesh.patches.size(); ++p)
      for (size_t i = 0; i < M.mesh.l2g[p].size(); ++i) {
        const std::string k = pk[p] + "#" + std::to_string(i);
        std::string& d = dk[M.mesh.l2g[p][i]];
        if (d.empty() || k < d) d = k;
      }
    for (int g = 0; g < M.mesh.ndof; ++g)
      for (int c = 0; c < 3; ++c) R.col_key[3 * g + c] = dk[g] + "." + std::to_string(c);
    for (int p = 0; p < M.n_outer; ++p) {
      if (!M.dirichlet[p]) continue;
      const int n1 = p + 1 < int(M.ts.offset.size()) ? M.ts.offset[p + 1] : M.ts.ndof;
      for (int j = M.ts.offset[p]; j < n1; ++j)
        for (int c = 0; c < 3; ++c)
          R.col_key[M.tcol[j] + c] = pk[p] + "t" + std::to_string(j - M.ts.offset[p]) + "." + std::to_string(c);
    }
  }

  static void sub(CrackModel::RowAcc& a, const CrackModel::RowAcc& b) {
    for (int i = 0; i < 2; ++i) { a.T0[i] -= b.T0[i]; a.S0[i] -= b.S0[i]; }
    for (int k = 0; k < 3; ++k) a.Mn[k] -= b.Mn[k];
    for (int g = 0; g < 2; ++g) for (int k = 0; k < 3; ++k) a.S1[g][k] -= b.S1[g][k];
  }

  static void rawRow(const CrackModel& M, const ElementIntegrator& q, const CrackModel::Row& r,
                     const std::vector<TractionFn>& loads, Eigen::MatrixXd& a, Eigen::MatrixXd& b,
                     const std::vector<char>* use, CrackModel::RowAcc& acc, bool finish) {
    if (r.kind == CrackModel::Row::Lower) M.tbieRow(q, r, loads, a, b, false, use, &acc, finish);
    else M.dbieRow(q, r, loads, a, b, false, use, &acc, finish);
  }

  // Solve model M for the loads, reusing the previous step (P, its loads and raw operator) when given; R receives
  // this step's raw operator for the next one. Only the plain (unenriched) elements' raw part is kept: the enriched
  // ones (window and crown) change with every move of the front, so they are integrated afresh each step.
  static std::vector<Eigen::VectorXd> solve(CrackModel& M, const std::vector<TractionFn>& loads,
                                            const std::vector<DisplacementFn>& disps, RawOperator& R,
                                            const CrackModel* P, const std::vector<TractionFn>* Ploads,
                                            const RawOperator* PR, GrowthStats& st) {
    const auto t0 = std::chrono::steady_clock::now();
    R.rows = M.collocationRows();
    keys(M, R.rows, R);
    R.quad = std::make_unique<ElementIntegrator>(M.mesh, M.opt.quad);
    st.t_quad = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int n = int(R.rows.size()), L = int(loads.size()), nc = M.ncols;
    const size_t ne = M.mesh.elements.size();
    auto enriched = [](const CrackModel& C, const Element& e) { return C.opt.enrich && C.piece[e.patch] >= 0; };
    // stable (plain) and enriched element sets of this step
    std::vector<char> plain(ne, 0), enr(ne, 0), addS(ne, 0);
    for (size_t e = 0; e < ne; ++e) (enriched(M, M.mesh.elements[e]) ? enr : plain)[e] = 1;
    addS = plain;
    // previous-step maps
    std::vector<int> prow(n, -1), pcol(nc, -1);
    std::vector<char> rem;
    if (P) {
      std::unordered_map<std::string, int> rk, ck, ek;
      for (size_t i = 0; i < PR->row_key.size(); ++i) rk[PR->row_key[i]] = int(i);
      for (size_t i = 0; i < PR->col_key.size(); ++i) if (!PR->col_key[i].empty()) ck[PR->col_key[i]] = int(i);
      for (size_t i = 0; i < PR->elem_key.size(); ++i)
        if (!enriched(*P, P->mesh.elements[i])) ek[PR->elem_key[i]] = int(i);
      for (int i = 0; i < n; ++i) { auto it = rk.find(R.row_key[i]); if (it != rk.end()) prow[i] = it->second; }
      for (int j = 0; j < nc; ++j) { auto it = ck.find(R.col_key[j]); if (it != ck.end()) pcol[j] = it->second; }
      rem.assign(P->mesh.elements.size(), 0);
      for (const auto& [k, i] : ek) rem[i] = 1;
      for (size_t e = 0; e < ne; ++e) {
        if (!plain[e]) continue;
        auto it = ek.find(R.elem_key[e]);
        if (it != ek.end()) { addS[e] = 0; rem[it->second] = 0; }
      }
      for (char c : rem) st.elems_removed += c;
    }
    for (size_t e = 0; e < ne; ++e) st.elems_added += addS[e] + enr[e];
    st.elems = long(ne);
    const int nr = 3 * n + (M.opt.enrich ? M.nK : 0);
    R.A.setZero(3 * n, nc);
    R.B.setZero(3 * n, L);
    R.acc.assign(n, CrackModel::RowAcc{});
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nr, nc), B = Eigen::MatrixXd::Zero(nr, L);
    long reused = 0;
    const std::vector<char> none(ne, 0);
    double t_fin = 0, t_rem = 0, t_add = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : reused, t_fin, t_rem, t_add)
    for (int i = 0; i < n; ++i) {
      Eigen::MatrixXd a = Eigen::MatrixXd::Zero(3, nc), b = Eigen::MatrixXd::Zero(3, L);
      CrackModel::RowAcc acc;
      const int o = prow[i];
      auto t = std::chrono::steady_clock::now();
      if (o >= 0) {  // stable part: previous one, less the plain elements that left, plus the plain ones that came
        ++reused;
        Eigen::MatrixXd ar = Eigen::MatrixXd::Zero(3, P->ncols), br = Eigen::MatrixXd::Zero(3, L);
        CrackModel::RowAcc accr;
        rawRow(*P, *PR->quad, PR->rows[o], *Ploads, ar, br, &rem, accr, false);
        const Eigen::MatrixXd ao = PR->A.middleRows(3 * o, 3) - ar;
        for (int j = 0; j < nc; ++j) if (pcol[j] >= 0) a.col(j) = ao.col(pcol[j]);
        b = PR->B.middleRows(3 * o, 3) - br;
        acc = PR->acc[o];
        sub(acc, accr);
        t_rem += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        t = std::chrono::steady_clock::now();
        rawRow(M, *R.quad, R.rows[i], loads, a, b, &addS, acc, false);
      } else {
        rawRow(M, *R.quad, R.rows[i], loads, a, b, &plain, acc, false);
      }
      R.A.middleRows(3 * i, 3) = a;
      R.B.middleRows(3 * i, 3) = b;
      R.acc[i] = acc;
      rawRow(M, *R.quad, R.rows[i], loads, a, b, &enr, acc, false);  // the moving part, always fresh
      t_add += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
      const auto tf = std::chrono::steady_clock::now();
      rawRow(M, *R.quad, R.rows[i], loads, a, b, &none, acc, true);
      t_fin += std::chrono::duration<double>(std::chrono::steady_clock::now() - tf).count();
      A.middleRows(3 * i, 3) = a;
      B.middleRows(3 * i, 3) = b;
    }
    st.rows_reused = reused;
    st.rows_new = n - reused;
    st.t_raw = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    st.t_finish = t_fin;
    st.t_rem = t_rem;
    st.t_add = t_add;
    auto X = M.solveSystem(A, B, n, false, disps, t0);
    st.t_solve = M.t_solve;
    return X;
  }
};

using RawOperator = Growth::Raw;

}  // namespace crack
}  // namespace igabem
