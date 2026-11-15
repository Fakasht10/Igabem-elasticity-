// Dual isogeometric BEM for a body with one flat crack, with the crack-front enrichment of Rocha, Trevelyan &
// Leonel, CMAME 423 (2024) 116872 (XIGABEM, Sec. 2.3-2.4). Standard IgBEM when the enrichment is off.
//
// Geometry (python/make_crack.py): outer patches (conforming displacement, discontinuous traction space), then the
// upper crack face (outward normal -m) and the lower face (+m), patch for patch coincident, each crack patch with
// dofs of its own (patch-discontinuous, as in the paper). Crown patches carry the enrichment; the front is their
// v = 1 edge.
//
// Unknowns: displacement coefficients except those of Dirichlet outer patches, tractions of the Dirichlet patches
// (traction-space coefficients), and with the enrichment the SIF coefficients K_{s,M} of each front piece.
// Equations (one 3-row block each):
//   outer   DBIE at the Greville point of every free outer dof, and at the traction points of Dirichlet patches
//           (exactly MixedModel's rows, crack faces included in the integrals);
//   upper   DBIE at the interior (shifted Greville) points of the upper face;
//   lower   TBIE at the same points of the lower face;
//   tying   int_front [u+ - u-] phi_s dv = 0 per front basis function (Eq. 13), enrichment only.
//
// Regularization. Every row subtracts a rigid (DBIE) or linear (TBIE) field built at the collocation point x, as
// sbie.hpp / hsbie.hpp do, but on the crack the coincident face gets the field of ITS OWN side at the coincident
// point xb, which makes every element integral O(1/r); what is left is the opening at x over the open coincident
// face, a singular integral evaluated in closed form by crack/planar.hpp:
//
//   DBIE, x on the upper face (limit from the solid above):
//     sum_e int T u - (int_{not lower} T) u+(x) - (int_{lower} T) u-(xb) - (PV int_{lower} T + I/2)(u+(x) - u-(xb))
//       = int U t
//   TBIE, x on the lower face (limit from below), u_L-(y) = u-(x) + grad u-(x)(y - x), u_L+ likewise at xb:
//     int D [t - sigma_L- n] - int_{not upper} S [u - u_L-] - int_{upper} S [u - u_L+]
//       - FP int_{upper} S [u_L+ - u_L-] = 0
//   with grad u from the tangential derivatives and the known traction (recovery.hpp surfaceGradient). The jumps:
//   T's double-layer jump is +I/2 on the side the lower face's normal points to; S has none on a flat face
//   (tests/test_crack.cpp verifies both).
#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../integrator.hpp"
#include "../json.hpp"
#include "../kelvin.hpp"
#include "../mesh.hpp"
#include "../recovery.hpp"
#include "../traction_space.hpp"
#include "front.hpp"
#include "planar.hpp"

namespace igabem {
namespace crack {

// Traction data t(y, n, patch) of one load case: on Neumann outer patches and on both crack faces.
using TractionFn = std::function<Vec3(const Vec3& y, const Vec3& n, int patch)>;
// Prescribed displacement g(y, patch) of one load case on the Dirichlet outer patches (patch-discontinuous dofs;
// L2-projected onto each patch's basis). None: zero.
using DisplacementFn = std::function<Vec3(const Vec3& y, int patch)>;

struct CrackOptions {
  bool enrich = true;
  int grade = 6;                        // geometric cells toward the front in front elements (sqrt(r) integrands)
  double end_shift = 0.02;              // crack collocation: end Greville points moved in by this much of a span
  std::vector<int> dirichlet;           // outer patches with prescribed displacement
  bool discontinuous = false;           // every patch with dofs of its own, outer collocation like the crack's
                                        // (paper Sec. 4.3: no continuity between patches)
  SolverOptions quad;
  PartitionOptions part;
};

class CrackModel {
 public:
  Mesh mesh;
  TractionSpace ts;
  Material mat;
  CrackOptions opt;
  Json meta;
  int n_outer = 0;
  std::vector<int> face;    // per patch: 0 outer, +1 upper, -1 lower
  std::vector<int> mirror;  // per patch: the coincident patch of the other face, -1 outer
  std::vector<int> piece;   // per patch: crown piece index (both faces), -1 none
  std::vector<char> dirichlet;
  PlanarRegion upper_region, lower_region;  // the same face seen with the upper / lower outward normal
  Front front;
  Vec3 m;
  // scalar columns: [0, 3M) displacement coefficients, [cK, cK + nK) SIFs, [cT, cT + nT) Dirichlet tractions
  int cK = 0, nK = 0, cT = 0, nT = 0, ncols = 0;
  std::vector<int> tcol;  // traction-space dof -> scalar column / 3 offset (-1: known traction)
  double t_assemble = 0, t_solve = 0;
  long unknowns = 0;
  // factored operator (factor / solveFactored)
  std::vector<int> f_cols, f_fixed;
  Eigen::MatrixXd f_A, f_Afixed;
  Eigen::PartialPivLU<Eigen::MatrixXd> f_lu;

  void setup(const std::string& dat, const std::string& meta_file, const Material& mt, const CrackOptions& o) {
    mat = mt;
    opt = o;
    meta = Json::load(meta_file);
    std::vector<Patch> P = loadDat(dat);
    const int np = int(P.size());
    n_outer = int(meta["outer"][1].asNumber());
    const int p0 = meta["plus"][0].asInt(), p1 = meta["plus"][1].asInt();
    const int q0 = meta["minus"][0].asInt(), q1 = meta["minus"][1].asInt();
    if (p0 != n_outer || q0 != p1 || q1 != np || p1 - p0 != q1 - q0) throw std::runtime_error("bad crack sidecar");
    face.assign(np, 0);
    mirror.assign(np, -1);
    piece.assign(np, -1);
    std::vector<char> sep(np, opt.discontinuous ? 1 : 0);
    for (int k = 0; k < p1 - p0; ++k) {
      face[p0 + k] = 1;
      face[q0 + k] = -1;
      mirror[p0 + k] = q0 + k;
      mirror[q0 + k] = p0 + k;
      sep[p0 + k] = sep[q0 + k] = 1;
    }
    mesh.build(std::move(P), opt.part, 1e-9, &sep);
    ts.build(mesh);
    m = Vec3(meta["plane"]["normal"][0].asNumber(), meta["plane"]["normal"][1].asNumber(),
             meta["plane"]["normal"][2].asNumber()).normalized();
    const Vec3 pt(meta["plane"]["point"][0].asNumber(), meta["plane"]["point"][1].asNumber(),
                  meta["plane"]["point"][2].asNumber());
    // flatness and orientation checks
    for (int p = p0; p < q1; ++p) {
      const Patch& Pa = mesh.patches[p];
      for (int k = 0; k < Pa.nu * Pa.nv; ++k)
        if (std::abs((Pa.cp(k) - pt).dot(m)) > 1e-10) throw std::runtime_error("crack face is not flat");
      PatchEval E;
      Pa.eval(0.5 * (Pa.U.front() + Pa.U.back()), 0.5 * (Pa.V.front() + Pa.V.back()), E);
      if (E.normal().dot(m) * face[p] * mesh.normal_sign > 0) throw std::runtime_error("crack face orientation");
    }
    dirichlet.assign(np, 0);
    for (int p : opt.dirichlet) dirichlet.at(p) = 1;

    // front pieces (upper crown patches) and their lower mirrors
    const Json& crown = meta["crown"];
    for (size_t k = 0; k < crown.size(); ++k) {
      const int p = crown[k].asInt();
      piece[p] = piece[mirror[p]] = int(k);
      FrontPiece F;
      F.init(&mesh.patches[p], p);
      front.pieces.push_back(F);
    }
    front.m = m;
    front.centroid = pt;
    front.mu = mat.mu;
    front.kappa = 3.0 - 4.0 * mat.nu();
    // outer patches the front ends on: full Williams field (sidecar "enriched_outer": [[patch, piece], ...])
    if (meta.has("enriched_outer"))
      for (size_t i = 0; i < meta["enriched_outer"].size(); ++i)
        piece[meta["enriched_outer"][i][0].asInt()] = meta["enriched_outer"][i][1].asInt();

    // regions: boundary = edges of upper patches shared with no other upper patch
    upper_region.setPlane(pt, m);
    lower_region.setPlane(pt, m);
    for (int p = p0; p < p1; ++p) addFreeEdges(p, p0, p1);

    // geometric cells toward the front (sqrt r enrichment)
    if (opt.enrich && opt.grade > 0)
      for (Element& e : mesh.elements) {
        if (piece[e.patch] < 0 || face[e.patch] == 0) continue;
        const Patch& Pa = mesh.patches[e.patch];
        if (e.rect.v1 < Pa.V.back() - 1e-14) continue;
        std::vector<Rect> cells;
        for (const Rect& R : e.cells) {
          if (R.v1 < e.rect.v1 - 1e-14) { cells.push_back(R); continue; }
          double a = R.v0;
          for (int g = 1; g <= opt.grade; ++g) {
            const double b = R.v1 - (R.v1 - R.v0) * std::pow(0.5, g);
            cells.push_back({R.u0, R.u1, a, b});
            a = b;
          }
          cells.push_back({R.u0, R.u1, a, R.v1});
        }
        e.cells = cells;
      }

    // enriched outer elements with a front end point at a corner: geometric squares toward it
    if (opt.enrich && opt.grade > 0) {
      std::vector<Vec3> ends;
      for (const FrontPiece& F : front.pieces)
        for (double u : {F.u0, F.u1}) {
          Vec3 C, Cu;
          int n, s[16];
          double R[16];
          F.eval(u, C, Cu, n, s, R);
          ends.push_back(C);
        }
      for (Element& e : mesh.elements) {
        if (piece[e.patch] < 0 || face[e.patch] != 0) continue;
        const Rect& Q = e.rect;
        for (int c = 0; c < 4; ++c) {
          const double cu = (c & 1) ? Q.u1 : Q.u0, cv = (c & 2) ? Q.v1 : Q.v0;
          PatchEval E;
          mesh.patches[e.patch].eval(cu, cv, E);
          bool hit = false;
          for (const Vec3& X : ends) hit = hit || (E.x - X).norm() < 1e-9;
          if (!hit) continue;
          // nested squares s_g = 2^-g of the element toward (cu, cv): three rects per ring
          std::vector<Rect> cells;
          const double du = (cu == Q.u0 ? 1 : -1) * (Q.u1 - Q.u0), dv = (cv == Q.v0 ? 1 : -1) * (Q.v1 - Q.v0);
          auto rect = [&](double a0, double a1, double b0, double b1) {
            const double x0 = cu + a0 * du, x1 = cu + a1 * du, y0 = cv + b0 * dv, y1 = cv + b1 * dv;
            cells.push_back({std::min(x0, x1), std::max(x0, x1), std::min(y0, y1), std::max(y0, y1)});
          };
          double so = 1.0;
          for (int g = 1; g <= opt.grade; ++g) {
            const double si = 0.5 * so;
            rect(si, so, 0, si);
            rect(si, so, si, so);
            rect(0, si, si, so);
            so = si;
          }
          rect(0, so, 0, so);
          e.cells = cells;
          break;
        }
      }
    }

    // columns
    cK = 3 * mesh.ndof;
    nK = 0;
    if (opt.enrich)
      for (FrontPiece& F : front.pieces) { F.first_col = cK + nK; nK += 3 * F.nbasis(); }
    cT = cK + nK;
    tcol.assign(ts.ndof, -1);
    nT = 0;
    for (int p = 0; p < n_outer; ++p) {
      if (!dirichlet[p]) continue;
      const int n1 = p + 1 < int(ts.offset.size()) ? ts.offset[p + 1] : ts.ndof;
      for (int j = ts.offset[p]; j < n1; ++j) { tcol[j] = cT + nT; nT += 3; }
    }
    ncols = cT + nT;
  }

  // Assemble all rows for the given load cases and solve. Returns the full scalar vector per case.
  std::vector<Eigen::VectorXd> solve(const std::vector<TractionFn>& loads, bool cond = false,
                                     const std::vector<DisplacementFn>& disps = {}) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Row> rows = collocationRows();
    const int nr = 3 * int(rows.size()) + (opt.enrich ? nK : 0);
    const int L = int(loads.size());
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nr, ncols), B = Eigen::MatrixXd::Zero(nr, L);
    ElementIntegrator quad(mesh, opt.quad);
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < int(rows.size()); ++r) {
      Eigen::MatrixXd a = Eigen::MatrixXd::Zero(3, ncols), b = Eigen::MatrixXd::Zero(3, L);
      if (rows[r].kind == Row::Lower) tbieRow(quad, rows[r], loads, a, b);
      else dbieRow(quad, rows[r], loads, a, b);
      A.middleRows(3 * r, 3) = a;
      B.middleRows(3 * r, 3) = b;
    }
    if (opt.enrich) tyingRows(A, 3 * int(rows.size()));
    // column selection: drop the (zero) Dirichlet displacement columns
    std::vector<char> fixed(mesh.ndof, 0);
    for (int p = 0; p < n_outer; ++p)
      if (dirichlet[p]) for (int g : mesh.l2g[p]) fixed[g] = 1;
    std::vector<int> cols;
    for (int g = 0; g < mesh.ndof; ++g)
      if (!fixed[g]) for (int c = 0; c < 3; ++c) cols.push_back(3 * g + c);
    for (int c = cK; c < ncols; ++c) cols.push_back(c);
    // known Dirichlet displacement coefficients: to the right-hand side
    std::vector<Eigen::VectorXd> g(L, Eigen::VectorXd::Zero(3 * mesh.ndof));
    if (!disps.empty()) {
      for (int l = 0; l < L; ++l) g[l] = projectDirichlet(disps[l]);
      std::vector<int> fc;
      for (int d = 0; d < mesh.ndof; ++d)
        if (fixed[d]) for (int c = 0; c < 3; ++c) fc.push_back(3 * d + c);
      for (int l = 0; l < L; ++l) B.col(l) -= A(Eigen::placeholders::all, fc) * g[l](fc);
    }
    if (int(cols.size()) != nr)
      throw std::runtime_error("crack system not square: " + std::to_string(nr) + " rows, " +
                               std::to_string(cols.size()) + " unknowns");
    unknowns = nr;
    const Eigen::MatrixXd As = A(Eigen::placeholders::all, cols);
    if (cond) {  // row scaling: largest over median row norm (one bad row inflates cond without a near-null mode)
      std::vector<double> rn(nr);
      for (int i = 0; i < nr; ++i) rn[i] = As.row(i).norm();
      std::nth_element(rn.begin(), rn.begin() + nr / 2, rn.end());
      row_ratio = *std::max_element(rn.begin(), rn.end()) / rn[nr / 2];
    }
    t_assemble = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(As);
    const Eigen::MatrixXd X = lu.solve(B);
    t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (cond) {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(As);
      condition = svd.singularValues()(0) / svd.singularValues().tail(1)(0);
    }
    residual = (As * X - B).norm() / std::max(B.norm(), 1e-300);
    std::vector<Eigen::VectorXd> out(L, Eigen::VectorXd::Zero(ncols));
    for (int l = 0; l < L; ++l) {
      out[l].head(3 * mesh.ndof) = g[l];
      for (size_t j = 0; j < cols.size(); ++j) out[l](cols[j]) = X(j, l);
    }
    return out;
  }
  // Factored path (crack_serve): the operator depends on geometry, nu and the Dirichlet set only, so it is
  // assembled and LU-factored once; every later load case assembles its right-hand side alone (rhs_only rows) and
  // costs one back-substitution. Same rows, columns and Dirichlet handling as solve().
  void factor(bool cond = false) {
    const auto t0 = std::chrono::steady_clock::now();
    f_rows = collocationRows();
    const int nr = 3 * int(f_rows.size()) + (opt.enrich ? nK : 0);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nr, ncols);
    ElementIntegrator quad(mesh, opt.quad);
    const std::vector<TractionFn> none;
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < int(f_rows.size()); ++r) {
      Eigen::MatrixXd a = Eigen::MatrixXd::Zero(3, ncols), b(3, 0);
      if (f_rows[r].kind == Row::Lower) tbieRow(quad, f_rows[r], none, a, b);
      else dbieRow(quad, f_rows[r], none, a, b);
      A.middleRows(3 * r, 3) = a;
    }
    if (opt.enrich) tyingRows(A, 3 * int(f_rows.size()));
    std::vector<char> fixed(mesh.ndof, 0);
    for (int p = 0; p < n_outer; ++p)
      if (dirichlet[p]) for (int g : mesh.l2g[p]) fixed[g] = 1;
    f_cols.clear();
    f_fixed.clear();
    for (int g = 0; g < mesh.ndof; ++g)
      for (int c = 0; c < 3; ++c) (fixed[g] ? f_fixed : f_cols).push_back(3 * g + c);
    for (int c = cK; c < ncols; ++c) f_cols.push_back(c);
    if (int(f_cols.size()) != nr)
      throw std::runtime_error("crack system not square: " + std::to_string(nr) + " rows, " +
                               std::to_string(f_cols.size()) + " unknowns");
    unknowns = nr;
    f_A = A(Eigen::placeholders::all, f_cols);
    f_Afixed = A(Eigen::placeholders::all, f_fixed);
    t_assemble = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const auto t1 = std::chrono::steady_clock::now();
    f_lu.compute(f_A);
    t_factor = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (cond) {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(f_A);
      condition = svd.singularValues()(0) / svd.singularValues().tail(1)(0);
    }
    factored = true;
  }

  std::vector<Eigen::VectorXd> solveFactored(const std::vector<TractionFn>& loads,
                                             const std::vector<DisplacementFn>& disps = {}) {
    if (!factored) throw std::runtime_error("solveFactored before factor");
    const auto t0 = std::chrono::steady_clock::now();
    const int L = int(loads.size()), nr = int(f_A.rows());
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nr, L);
    ElementIntegrator quad(mesh, opt.quad);
#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < int(f_rows.size()); ++r) {
      Eigen::MatrixXd a(3, 0), b = Eigen::MatrixXd::Zero(3, L);
      if (f_rows[r].kind == Row::Lower) tbieRow(quad, f_rows[r], loads, a, b, true);
      else dbieRow(quad, f_rows[r], loads, a, b, true);
      B.middleRows(3 * r, 3) = b;
    }
    std::vector<Eigen::VectorXd> g(L, Eigen::VectorXd::Zero(3 * mesh.ndof));
    if (!disps.empty()) {
      for (int l = 0; l < L; ++l) g[l] = projectDirichlet(disps[l]);
      for (int l = 0; l < L; ++l) B.col(l) -= f_Afixed * g[l](f_fixed);
    }
    t_rhs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Eigen::MatrixXd X = f_lu.solve(B);
    residual = (f_A * X - B).norm() / std::max(B.norm(), 1e-300);
    std::vector<Eigen::VectorXd> out(L, Eigen::VectorXd::Zero(ncols));
    for (int l = 0; l < L; ++l) {
      out[l].head(3 * mesh.ndof) = g[l];
      for (size_t j = 0; j < f_cols.size(); ++j) out[l](f_cols[j]) = X(j, l);
    }
    return out;
  }
  bool factored = false;
  double t_factor = 0, t_rhs = 0;
  double condition = 0, residual = 0, row_ratio = 0;

  // L2 projection of g onto the displacement basis of every Dirichlet patch (dofs of their own required unless g
  // vanishes on the shared ones).
  Eigen::VectorXd projectDirichlet(const DisplacementFn& gfn) const {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(3 * mesh.ndof);
    const GaussRule gr = gaussLegendre(6);
    for (int p = 0; p < n_outer; ++p) {
      if (!dirichlet[p]) continue;
      const auto& l2g = mesh.l2g[p];
      const int nb = int(l2g.size());
      Eigen::MatrixXd M = Eigen::MatrixXd::Zero(nb, nb), F = Eigen::MatrixXd::Zero(nb, 3);
      for (const Element& e : mesh.elements) {
        if (e.patch != p) continue;
        for (size_t i = 0; i < gr.x.size(); ++i)
          for (size_t j = 0; j < gr.x.size(); ++j) {
            const double u = e.rect.u0 + (e.rect.u1 - e.rect.u0) * gr.x[i], v = e.rect.v0 + (e.rect.v1 - e.rect.v0) * gr.x[j];
            PatchEval E;
            mesh.patches[p].evalSpan(e.su, e.sv, u, v, E);
            const double w = gr.w[i] * gr.w[j] * (e.rect.u1 - e.rect.u0) * (e.rect.v1 - e.rect.v0) * E.normal().norm();
            const Vec3 gv = gfn(E.x, p);
            for (int a = 0; a < E.n_loc; ++a) {
              F.row(E.idx[a]) += w * E.R[a] * gv.transpose();
              for (int b = 0; b < E.n_loc; ++b) M(E.idx[a], E.idx[b]) += w * E.R[a] * E.R[b];
            }
          }
      }
      const Eigen::MatrixXd C = M.ldlt().solve(F);
      for (int a = 0; a < nb; ++a) out.segment<3>(3 * l2g[a]) = C.row(a).transpose();
    }
    return out;
  }

  // Displacement on face point (patch p, parameter uv) from a solution, enrichment included.
  Vec3 displacement(int p, const Vec2& uv, const Eigen::VectorXd& x) const {
    PatchEval E;
    mesh.patches[p].eval(uv(0), uv(1), E);
    Shape s;
    shapeAt(p, uv, E, s, false);
    Vec3 u = Vec3::Zero();
    for (size_t i = 0; i < s.col.size(); ++i) u += s.val[i] * x(s.col[i]);
    return u;
  }

  // Boundary densities at a quadrature point of patch pe (unit normal n) from a solution x: displacement (enrichment
  // included) and traction (Dirichlet patches: the solved traction coefficients; elsewhere the load).
  void densities(int pe, const QPoint& q, const Vec3& n, const Eigen::VectorXd& x, const TractionFn& load, Vec3& u,
                 Vec3& t) const {
    const auto& l2g = mesh.l2g[pe];
    u.setZero();
    for (int l = 0; l < q.n_loc; ++l) u += q.R[l] * x.segment<3>(3 * l2g[q.idx[l]]);
    int cols[64];
    Vec3 vals[64];
    const int ne = quadEnrichment(pe, q, cols, vals);
    for (int i = 0; i < ne; ++i) u += vals[i] * x(cols[i]);
    if (pe < n_outer && dirichlet[pe]) {
      t.setZero();
      const int off = ts.offset[pe];
      for (int l = 0; l < q.n_loc; ++l) t += q.R[l] * x.segment<3>(tcol[off + q.idx[l]]);
    } else {
      t = load(q.x, n, pe);
    }
  }

  // The lower-face point coincident with x on the mirror patch.
  SurfacePoint coincident(int p, const Vec3& x) const {
    const int q = mirror[p];
    return locate(mesh, x, [q](int pp) { return pp == q; });
  }

 private:
  struct Row {
    enum Kind { Outer, Upper, Lower } kind;
    int patch;
    Vec2 xi;
    Vec3 x, N;
    PatchEval E;
    std::vector<std::pair<int, double>> basis;  // outer rows: global dof, N_B(x)
  };
  std::vector<Row> f_rows;  // collocation rows of the factored operator

  // Values and parametric derivatives of every scalar column's shape vector at a point.
  struct Shape {
    std::vector<int> col;
    std::vector<Vec3> val, du, dv;
    void add(int c, const Vec3& v, const Vec3& a, const Vec3& b) {
      col.push_back(c); val.push_back(v); du.push_back(a); dv.push_back(b);
    }
  };

  void shapeAt(int p, const Vec2& uv, const PatchEval& E, Shape& s, bool derivs) const {
    s = Shape{};
    for (int l = 0; l < E.n_loc; ++l) {
      if (std::abs(E.R[l]) < 1e-15 && (!derivs || (std::abs(E.Ru[l]) < 1e-15 && std::abs(E.Rv[l]) < 1e-15)))
        continue;
      const int g = mesh.l2g[p][E.idx[l]];
      for (int c = 0; c < 3; ++c)
        s.add(3 * g + c, E.R[l] * Vec3::Unit(c), E.Ru[l] * Vec3::Unit(c), E.Rv[l] * Vec3::Unit(c));
    }
    if (!opt.enrich || piece[p] < 0) return;
    int cols[64];
    Vec3 v0[64], vp[64], vm[64];
    const int k = piece[p], sg = face[p];
    const int n = front.shapes(k, sg, E.x, guess(p, uv(0)), cols, v0);
    if (!derivs) {
      for (int i = 0; i < n; ++i) s.add(cols[i], v0[i], Vec3::Zero(), Vec3::Zero());
      return;
    }
    const Patch& Pa = mesh.patches[p];
    PatchEval Ep;
    std::vector<Vec3> du(n), dv(n);
    for (int dir = 0; dir < 2; ++dir) {
      const double span = dir == 0 ? Pa.U.back() - Pa.U.front() : Pa.V.back() - Pa.V.front();
      const double h = 1e-6 * span;
      Vec2 a = uv, b = uv;
      a(dir) += h;
      b(dir) -= h;
      // stay inside the patch (the front edge v = 1 is never a collocation point)
      int cp[64], cm[64];
      Pa.eval(a(0), a(1), Ep);
      const int n1 = front.shapes(k, sg, Ep.x, a(0), cp, vp);
      Pa.eval(b(0), b(1), Ep);
      const int n2 = front.shapes(k, sg, Ep.x, b(0), cm, vm);
      for (int i = 0; i < n; ++i) {
        Vec3 fp = Vec3::Zero(), fm = Vec3::Zero();
        for (int j = 0; j < n1; ++j) if (cp[j] == cols[i]) fp = vp[j];
        for (int j = 0; j < n2; ++j) if (cm[j] == cols[i]) fm = vm[j];
        (dir == 0 ? du : dv)[i] = (fp - fm) / (2 * h);
      }
    }
    for (int i = 0; i < n; ++i) s.add(cols[i], v0[i], du[i], dv[i]);
  }

  // Tangential gradient of a shape vector with parametric derivatives (a, b) at a point with tangents xu, xv.
  static Eigen::Matrix3d tangential(const PatchEval& E, const Vec3& a, const Vec3& b) {
    Eigen::Matrix<double, 3, 2> Ax, Ud;
    Ax << E.xu, E.xv;
    Ud << a, b;
    return Ud * (Ax.transpose() * Ax).inverse() * Ax.transpose();
  }

  void addFreeEdges(int p, int p0, int p1) {
    const Patch& Pa = mesh.patches[p];
    auto edgeMid = [](const Patch& P, int e) {
      PatchEval E;
      const double um = 0.5 * (P.U.front() + P.U.back()), vm = 0.5 * (P.V.front() + P.V.back());
      const double uu[4] = {um, P.U.back(), um, P.U.front()}, vv[4] = {P.V.front(), vm, P.V.back(), vm};
      P.eval(uu[e], vv[e], E);
      return E.x;
    };
    for (int e = 0; e < 4; ++e) {
      const Vec3 X = edgeMid(Pa, e);
      bool shared = false;
      for (int q = p0; q < p1 && !shared; ++q) {
        if (q == p) continue;
        for (int f = 0; f < 4 && !shared; ++f) shared = (edgeMid(mesh.patches[q], f) - X).norm() < 1e-9;
      }
      if (shared) continue;
      const Patch* P = &Pa;
      std::function<Vec3(double)> C;
      double t0, t1;
      if (e == 0 || e == 2) {
        const double v = e == 0 ? Pa.V.front() : Pa.V.back();
        C = [P, v](double t) { PatchEval E; P->eval(t, v, E); return E.x; };
        t0 = Pa.U.front(); t1 = Pa.U.back();
      } else {
        const double u = e == 1 ? Pa.U.back() : Pa.U.front();
        C = [P, u](double t) { PatchEval E; P->eval(u, t, E); return E.x; };
        t0 = Pa.V.front(); t1 = Pa.V.back();
      }
      upper_region.addEdge(C, t0, t1, 128);
      lower_region.addEdge(C, t0, t1, 128);
    }
  }

  // Projection guess on the front piece: the crown patch's own u, the piece's middle on outer patches.
  double guess(int p, double u) const {
    if (face[p] != 0) return u;
    const FrontPiece& F = front.pieces[piece[p]];
    return 0.5 * (F.u0 + F.u1);
  }

  void Pa_eval(Row& r) const {
    mesh.patches[r.patch].eval(r.xi(0), r.xi(1), r.E);
    r.x = r.E.x;
  }

  std::vector<Row> collocationRows() const {
    std::vector<Row> rows;
    std::vector<char> fixed(mesh.ndof, 0);
    for (int p = 0; p < n_outer; ++p)
      if (dirichlet[p]) for (int g : mesh.l2g[p]) fixed[g] = 1;
    if (!opt.discontinuous)
      for (int g = 0; g < mesh.ndof; ++g) {
        const Collocation& c = mesh.colloc[g];
        if (c.patch >= n_outer || fixed[g]) continue;
        Row r{Row::Outer, c.patch, c.xi, c.x, Vec3::Zero(), {}, c.basis};
        Pa_eval(r);
        rows.push_back(r);
      }
    else
      for (int p = 0; p < n_outer; ++p) {
        if (dirichlet[p]) continue;
        const Patch& Pa = mesh.patches[p];
        for (int k = 0; k < Pa.nu * Pa.nv; ++k) {
          Row r{Row::Outer, p, Vec2(crackPoint(Pa, k % Pa.nu, true), crackPoint(Pa, k / Pa.nu, false)), Vec3::Zero(),
                Vec3::Zero(), PatchEval{}, {}};
          Pa_eval(r);
          for (int l = 0; l < r.E.n_loc; ++l)
            if (std::abs(r.E.R[l]) > 1e-15) r.basis.push_back({mesh.l2g[p][r.E.idx[l]], r.E.R[l]});
          rows.push_back(r);
        }
      }
    for (int p = 0; p < n_outer; ++p) {
      if (!dirichlet[p]) continue;
      const int n1 = p + 1 < int(ts.offset.size()) ? ts.offset[p + 1] : ts.ndof;
      for (int j = ts.offset[p]; j < n1; ++j) {
        const TractionSpace::Point& t = ts.points[j];
        Row r{Row::Outer, p, t.xi, t.x, t.N, t.E, {}};
        for (int l = 0; l < t.E.n_loc; ++l)
          if (std::abs(t.E.R[l]) > 1e-15) r.basis.push_back({mesh.l2g[p][t.E.idx[l]], t.E.R[l]});
        rows.push_back(r);
      }
    }
    for (int sgn : {1, -1})
      for (int p = n_outer; p < int(mesh.patches.size()); ++p) {
        if (face[p] != sgn) continue;
        const Patch& Pa = mesh.patches[p];
        for (int k = 0; k < Pa.nu * Pa.nv; ++k) {
          Row r{sgn > 0 ? Row::Upper : Row::Lower, p, Vec2(crackPoint(Pa, k % Pa.nu, true), crackPoint(Pa, k / Pa.nu, false)),
                Vec3::Zero(), Vec3::Zero(), PatchEval{}, {}};
          Pa.eval(r.xi(0), r.xi(1), r.E);
          r.x = r.E.x;
          r.N = (r.E.normal() * mesh.normal_sign).normalized();
          rows.push_back(r);
        }
      }
    return rows;
  }

  // Collocation abscissa i of a crack patch: Greville, the two end points moved inside the patch, by
  // opt.end_shift of the end knot span (paper Eq. A.13: 0.1; the default 0.02 cuts the penny K error 8x) or,
  // end_shift <= 0, halfway to the next Greville point (the traction space's rule).
  double crackPoint(const Patch& P, int i, bool in_u) const {
    const int n = in_u ? P.nu : P.nv, deg = in_u ? P.p : P.q;
    const auto& K = in_u ? P.U : P.V;
    if (opt.end_shift <= 0.0) {
      if (i == 0) return 0.5 * (P.greville(0, in_u) + P.greville(1, in_u));
      if (i == n - 1) return 0.5 * (P.greville(n - 2, in_u) + P.greville(n - 1, in_u));
      return P.greville(i, in_u);
    }
    if (i == 0) return K[deg] + opt.end_shift * (K[deg + 1] - K[deg]);
    if (i == n - 1) return K[n] - opt.end_shift * (K[n] - K[n - 1]);
    return P.greville(i, in_u);
  }

  // Integrand-side shape of a quadrature point: standard basis (3 x 3 blocks) and enrichment vectors.
  int quadEnrichment(int p, const QPoint& q, int* cols, Vec3* vals) const {
    if (!opt.enrich || piece[p] < 0) return 0;
    return front.shapes(piece[p], face[p], q.x, guess(p, q.uv(0)), cols, vals);
  }

  // rhs_only: the load columns b alone (a untouched), for new load cases on a factored operator.
  void dbieRow(const ElementIntegrator& quad, const Row& row, const std::vector<TractionFn>& loads,
               Eigen::MatrixXd& a, Eigen::MatrixXd& b, bool rhs_only = false) const {
    const Kelvin K(mat);
    const int L = int(loads.size());
    const SourcePoint src{row.patch, row.xi, row.x};
    Eigen::Matrix3d T0[2] = {Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
    for (const Element& e : mesh.elements) {
      const int pe = e.patch;
      const auto& l2g = mesh.l2g[pe];
      const int grp = (row.kind == Row::Upper && face[pe] == -1) ? 1 : 0;
      const bool unknown_t = pe < n_outer && dirichlet[pe];
      const int off = ts.offset[pe];
      quad.integrate(src, e, [&](const QPoint& q, double w) {
        const Vec3 nJ = q.normal() * mesh.normal_sign;
        const double J = nJ.norm();
        const Vec3 n = nJ / J;
        Eigen::Matrix3d U, T;
        K.eval(q.x - row.x, n, U, T);
        const Eigen::Matrix3d Tw = T * (w * J), Uw = U * (w * J);
        if (rhs_only) {
          if (!unknown_t) for (int c = 0; c < L; ++c) b.col(c) += Uw * loads[c](q.x, n, pe);
          return;
        }
        for (int l = 0; l < q.n_loc; ++l) a.block<3, 3>(0, 3 * l2g[q.idx[l]]) += q.R[l] * Tw;
        int cols[64];
        Vec3 vals[64];
        const int ne = quadEnrichment(pe, q, cols, vals);
        for (int i = 0; i < ne; ++i) a.col(cols[i]) += Tw * vals[i];
        T0[grp] += Tw;
        if (unknown_t) {
          for (int l = 0; l < q.n_loc; ++l) a.block<3, 3>(0, tcol[off + q.idx[l]]) -= q.R[l] * Uw;
        } else {
          for (int c = 0; c < L; ++c) b.col(c) += Uw * loads[c](q.x, n, pe);
        }
      });
    }
    if (rhs_only) return;
    if (row.kind == Row::Outer) {
      if (piece[row.patch] >= 0) {  // enriched outer patch: the whole field at x
        Shape own;
        shapeAt(row.patch, row.xi, row.E, own, false);
        for (size_t i = 0; i < own.col.size(); ++i) a.col(own.col[i]) -= T0[0] * own.val[i];
      } else {
        for (const auto& [g, NB] : row.basis) a.block<3, 3>(0, 3 * g) -= NB * T0[0];
      }
      return;
    }
    // upper-face DBIE: own face, coincident lower point, opening coefficient
    Shape own, mir;
    shapeAt(row.patch, row.xi, row.E, own, false);
    const SurfacePoint sb = coincident(row.patch, row.x);
    shapeAt(sb.patch, sb.xi, sb.E, mir, false);
    const FaceCoefficients fc = faceCoefficients(lower_region, mat, row.x, m, m);
    const Eigen::Matrix3d J = fc.CT + 0.5 * Eigen::Matrix3d::Identity();
    for (size_t i = 0; i < own.col.size(); ++i) a.col(own.col[i]) -= (T0[0] + J) * own.val[i];
    for (size_t i = 0; i < mir.col.size(); ++i) a.col(mir.col[i]) -= (T0[1] - J) * mir.val[i];
  }

  void tbieRow(const ElementIntegrator& quad, const Row& row, const std::vector<TractionFn>& loads,
               Eigen::MatrixXd& a, Eigen::MatrixXd& b, bool rhs_only = false) const {
    const KelvinTraction K(mat);
    const int L = int(loads.size());
    const SourcePoint src{row.patch, row.xi, row.x};
    const Vec3 N = row.N;
    Eigen::Matrix3d Mn[3], S0[2], S1[2][3];
    for (int k = 0; k < 3; ++k) Mn[k].setZero();
    for (int g = 0; g < 2; ++g) {
      S0[g].setZero();
      for (int k = 0; k < 3; ++k) S1[g][k].setZero();
    }
    for (const Element& e : mesh.elements) {
      const int pe = e.patch;
      const auto& l2g = mesh.l2g[pe];
      const int grp = face[pe] == 1 ? 1 : 0;
      const bool unknown_t = pe < n_outer && dirichlet[pe];
      const int off = ts.offset[pe];
      quad.integrate(src, e, [&](const QPoint& q, double w) {
        const Vec3 nJ = q.normal() * mesh.normal_sign;
        const double J = nJ.norm();
        const Vec3 n = nJ / J, d = q.x - row.x;
        Eigen::Matrix3d D, S;
        K.eval(d, N, n, D, S);
        const Eigen::Matrix3d Dw = D * (w * J), Sw = S * (w * J);
        if (!rhs_only) {
          for (int l = 0; l < q.n_loc; ++l) a.block<3, 3>(0, 3 * l2g[q.idx[l]]) -= q.R[l] * Sw;
          int cols[64];
          Vec3 vals[64];
          const int ne = quadEnrichment(pe, q, cols, vals);
          for (int i = 0; i < ne; ++i) a.col(cols[i]) -= Sw * vals[i];
        }
        if (unknown_t) {
          if (!rhs_only)
            for (int l = 0; l < q.n_loc; ++l) a.block<3, 3>(0, tcol[off + q.idx[l]]) += q.R[l] * Dw;
        } else {
          for (int c = 0; c < L; ++c) b.col(c) -= Dw * loads[c](q.x, n, pe);
        }
        for (int k = 0; k < 3; ++k) {
          Mn[k] += Dw * n(k);
          S1[grp][k] += Sw * d(k);
        }
        S0[grp] += Sw;
      });
    }
    const SurfacePoint sb = coincident(row.patch, row.x);
    const FaceCoefficients fc = faceCoefficients(upper_region, mat, row.x, sb.N, N);
    auto sigmaTerm = [&](const Eigen::Matrix3d& G) {  // sum_k Mn[k] sigma(G).col(k)
      const Eigen::Matrix3d sg = mat.stress(G);
      Vec3 v = Vec3::Zero();
      for (int k = 0; k < 3; ++k) v += Mn[k] * sg.col(k);
      return v;
    };
    auto lin = [&](const Eigen::Matrix3d (&S1g)[3], const Eigen::Matrix3d& G) {
      Vec3 v = Vec3::Zero();
      for (int k = 0; k < 3; ++k) v += S1g[k] * G.col(k);
      return v;
    };
    auto fb = [&](const Eigen::Matrix3d& G) {
      Vec3 v = Vec3::Zero();
      for (int k = 0; k < 3; ++k) v += fc.FB[k] * G.col(k);
      return v;
    };
    if (!rhs_only) {
      // own (lower) side: u_L-
      Shape own;
      shapeAt(row.patch, row.xi, row.E, own, true);
      for (size_t i = 0; i < own.col.size(); ++i) {
        const Eigen::Matrix3d G = surfaceGradient(mat, tangential(row.E, own.du[i], own.dv[i]), Vec3::Zero(), N);
        a.col(own.col[i]) += -sigmaTerm(G) + S0[0] * own.val[i] + lin(S1[0], G) + fc.Fa * own.val[i] + fb(G);
      }
      // coincident (upper) side: u_L+
      Shape mir;
      shapeAt(sb.patch, sb.xi, sb.E, mir, true);
      for (size_t i = 0; i < mir.col.size(); ++i) {
        const Eigen::Matrix3d G = surfaceGradient(mat, tangential(sb.E, mir.du[i], mir.dv[i]), Vec3::Zero(), sb.N);
        a.col(mir.col[i]) += S0[1] * mir.val[i] + lin(S1[1], G) - fc.Fa * mir.val[i] - fb(G);
      }
    }
    // known crack-face tractions enter the linear fields through Hooke's law
    for (int c = 0; c < L; ++c) {
      const Vec3 tm = loads[c](row.x, N, row.patch), tp = loads[c](sb.E.x, sb.N, sb.patch);
      const Eigen::Matrix3d Gm = surfaceGradient(mat, Eigen::Matrix3d::Zero(), tm, N);
      const Eigen::Matrix3d Gp = surfaceGradient(mat, Eigen::Matrix3d::Zero(), tp, sb.N);
      b.col(c) -= -sigmaTerm(Gm) + lin(S1[0], Gm) + fb(Gm) + lin(S1[1], Gp) - fb(Gp);
    }
  }

  // Eq. 13: int_front (u+ - u-) phi_s dv = 0, one row per (piece, s, component); Gauss on the edge's knot spans.
  void tyingRows(Eigen::MatrixXd& A, int r0) const {
    const GaussRule g = gaussLegendre(8);
    int r = r0;
    for (const FrontPiece& F : front.pieces) {
      const Patch& P = *F.P;
      const int q = mirror[F.patch];
      std::vector<double> kn = P.uniqueKnots(P.U);
      for (int s = 0; s < F.nbasis(); ++s) {
        Eigen::Matrix<double, 3, Eigen::Dynamic> row = Eigen::MatrixXd::Zero(3, ncols);
        for (size_t k = 0; k + 1 < kn.size(); ++k)
          for (size_t i = 0; i < g.x.size(); ++i) {
            const double u = kn[k] + (kn[k + 1] - kn[k]) * g.x[i], w = g.w[i] * (kn[k + 1] - kn[k]);
            Vec3 C, Cu;
            int n, idx[16];
            double R[16];
            F.eval(u, C, Cu, n, idx, R);
            double phi = 0.0;
            for (int j = 0; j < n; ++j) if (idx[j] == s) phi = R[j];
            if (phi == 0.0) continue;
            PatchEval E;
            P.eval(u, P.V.back(), E);
            for (int l = 0; l < E.n_loc; ++l)
              row.block<3, 3>(0, 3 * mesh.l2g[F.patch][E.idx[l]]) += (w * phi * E.R[l]) * Eigen::Matrix3d::Identity();
            const SurfacePoint sb = locate(mesh, C, [q](int pp) { return pp == q; });
            for (int l = 0; l < sb.E.n_loc; ++l)
              row.block<3, 3>(0, 3 * mesh.l2g[q][sb.E.idx[l]]) -= (w * phi * sb.E.R[l]) * Eigen::Matrix3d::Identity();
          }
        A.middleRows(r, 3) = row;
        r += 3;
      }
    }
  }
};

}  // namespace crack
}  // namespace igabem
