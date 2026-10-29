// Regions of the boundary for local boundary conditions and loads: CAD faces
// (patch -> face map written by python/face_ids.py), patch lists and geometric
// selectors (box, plane, half-space, cylinder, ball, normal cone), combined by
// AND within a selector and OR across selectors.
//
// A region is resolved on the mesh once (RegionMesh): every element it touches
// becomes a cell with a parametric quadtree that separates the parts inside,
// outside and on the region boundary. Leaves on the boundary are clipped by
// marching squares (crossings found by bisection, boundary taken as the chord;
// area error ~ c^2 / (6 R^2) relative, c the leaf size and R the boundary's
// radius of curvature). So a load on a sub-region is integrated as the
// discontinuous datum it is, not forced into the per-patch traction space,
// and patches never have to follow the load regions.
//
// RegionIntegrator integrates kernel x density over the region from any
// source point: Duffy on the sub-cell holding the point, a near-singular
// quadtree close to it, and for far cells (|x - c| >= 5 r) a product rule on
// the cell's m x m Gauss nodes whose weights carry the density, precomputed
// from a fine cloud: sum_q l_a(xi_q) t(y_q) w_q, exact for kernels polynomial of
// degree m - 1 on the cell (error ~ rho^-m, rho the Bernstein-ellipse parameter).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kelvin.hpp"
#include "mesh.hpp"
#include "quadrature.hpp"

namespace igabem {

// ------------------------------------------------------------------ selectors
struct Selector {
  std::vector<int> faces, patches;
  bool box = false;
  Vec3 lo = Vec3::Zero(), hi = Vec3::Zero();
  bool plane = false;  // |(x - p0) . pn| <= ptol
  Vec3 p0 = Vec3::Zero(), pn = Vec3::UnitZ();
  double ptol = 1e-6;
  bool half = false;  // (x - h0) . hn >= 0
  Vec3 h0 = Vec3::Zero(), hn = Vec3::UnitZ();
  bool cyl = false;  // | dist(x, axis) - cr | <= ctol, axis through c0 along cd
  Vec3 c0 = Vec3::Zero(), cd = Vec3::UnitZ();
  double cr = 0.0, ctol = 1e-6;
  bool ball = false;  // |x - bc| <= br
  Vec3 bc = Vec3::Zero();
  double br = 0.0;
  bool normal = false;  // n . nd >= ncos
  Vec3 nd = Vec3::UnitZ();
  double ncos = 1.0;

  bool patchwise() const { return !(box || plane || half || cyl || ball || normal); }

  bool test(int patch, int face, const Vec3& x, const Vec3& n) const {
    if (!faces.empty() && std::find(faces.begin(), faces.end(), face) == faces.end()) return false;
    if (!patches.empty() && std::find(patches.begin(), patches.end(), patch) == patches.end()) return false;
    if (box && ((x.array() < lo.array()).any() || (x.array() > hi.array()).any())) return false;
    if (plane && std::abs((x - p0).dot(pn)) > ptol) return false;
    if (half && (x - h0).dot(hn) < 0.0) return false;
    if (cyl) {
      Vec3 r = x - c0;
      r -= r.dot(cd) * cd;
      if (std::abs(r.norm() - cr) > ctol) return false;
    }
    if (ball && (x - bc).norm() > br) return false;
    if (normal && n.dot(nd) < ncos) return false;
    return true;
  }

  // A box containing every selected point, if the selector is bounded.
  bool bounds(Vec3& a, Vec3& b) const {
    bool ok = false;
    a = Vec3::Constant(-1e300);
    b = Vec3::Constant(1e300);
    if (box) { a = a.cwiseMax(lo); b = b.cwiseMin(hi); ok = true; }
    if (ball) { a = a.cwiseMax(bc - Vec3::Constant(br)); b = b.cwiseMin(bc + Vec3::Constant(br)); ok = true; }
    return ok;
  }
};

struct Region {
  std::string label;
  std::vector<Selector> any;        // union of selectors
  std::vector<int> face_of_patch;   // CAD face id of each patch (empty: no face map)

  int face(int patch) const {
    return patch < int(face_of_patch.size()) ? face_of_patch[patch] : -1;
  }
  bool contains(int patch, const Vec3& x, const Vec3& n) const {
    for (const Selector& s : any)
      if (s.test(patch, face(patch), x, n)) return true;
    return false;
  }
  bool patchwise() const {
    for (const Selector& s : any)
      if (!s.patchwise()) return false;
    return true;
  }
  bool bounds(Vec3& a, Vec3& b) const {
    a = Vec3::Constant(1e300);
    b = Vec3::Constant(-1e300);
    for (const Selector& s : any) {
      Vec3 sa, sb;
      if (!s.bounds(sa, sb)) return false;
      a = a.cwiseMin(sa);
      b = b.cwiseMax(sb);
    }
    return !any.empty();
  }
};

// CAD face id per patch from "<geometry>.faces" (python/face_ids.py): comment
// lines start with '#', then the number of patches and one id per patch.
// Missing file: empty map (face selectors then select nothing).
inline std::vector<int> loadFaceMap(const std::string& path) {
  std::ifstream f(path);
  std::vector<int> out;
  if (!f) return out;
  std::string line, all;
  while (std::getline(f, line))
    if (!line.empty() && line[0] != '#') all += line + ' ';
  std::istringstream s(all);
  int n = 0;
  s >> n;
  out.resize(n);
  for (int& v : out)
    if (!(s >> v)) throw std::runtime_error("bad face map " + path);
  return out;
}

inline std::string faceMapPath(const std::string& geometry_file) {
  const auto dot = geometry_file.rfind(".dat");
  return (dot == std::string::npos ? geometry_file : geometry_file.substr(0, dot)) + ".faces";
}

// ---------------------------------------------------------------- region mesh
// A quadrature point of the region: position, unit outward normal, weight
// (parametric weight x surface Jacobian), and the patch basis there.
struct RPoint {
  int patch;
  Vec3 x, n;
  double w;
  int n_loc;
  const int* idx;
  const double* R;
};

class RegionMesh {
 public:
  struct Node {
    Rect r;
    char state;  // 'i' inside, 'c' cut (children or clipped triangles), 'o' outside
    int child = -1;  // first of 4 children, -1: leaf
    std::vector<std::array<Vec2, 3>> tris;  // inside part of a cut leaf (parametric)
  };
  struct Cell {
    int elem, patch, su, sv;
    Rect rect;
    Vec3 c;       // bounding ball of the element
    double r, h;  // ball radius, element diameter
    bool full;    // the whole element is inside
    std::vector<Node> tree;
    std::vector<int> idx;  // local basis indices (constant on the knot span)
    // fine cloud: inside quadrature points
    std::vector<Vec2> xi;
    std::vector<Vec3> x, n;
    std::vector<double> w, R;  // R: n_loc values per point
    // far-field product rules, m = 4, 6, 8, 10 nodes per direction
    std::array<std::vector<Vec2>, 4> node_xi;
    std::array<std::vector<Vec3>, 4> node_x;
    std::array<std::vector<Vec3>, 4> node_n;
  };
  static constexpr int kM[4] = {4, 6, 8, 10};

  const Mesh& mesh;
  Region region;
  std::vector<Cell> cells;
  double area = 0.0;
  Vec3 centroid = Vec3::Zero();
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();  // int (|r|^2 I - r r^T), r = x - centroid
  Vec3 normal_sum = Vec3::Zero();                      // int n
  std::vector<double> patch_area;                      // region area on each patch
  int cut_depth;

  RegionMesh(const Mesh& m, Region r, int depth = 6, int order = 6)
      : mesh(m), region(std::move(r)), cut_depth(depth) {
    patch_area.assign(m.patches.size(), 0.0);
    const bool pw = region.patchwise();
    Vec3 blo, bhi;
    const bool bounded = !pw && region.bounds(blo, bhi);
    for (size_t ie = 0; ie < m.elements.size(); ++ie) {
      const Element& e = m.elements[ie];
      Cell C;
      C.elem = int(ie);
      C.patch = e.patch;
      C.su = e.su;
      C.sv = e.sv;
      C.rect = e.rect;
      C.h = e.h;
      ballOf(e.patch, e.su, e.sv, e.rect, C.c, C.r);
      if (bounded && ((C.c.array() + C.r < blo.array()).any() || (C.c.array() - C.r > bhi.array()).any())) continue;
      if (pw) {
        PatchEval E;
        mesh.patches[e.patch].evalSpan(e.su, e.sv, 0.5 * (e.rect.u0 + e.rect.u1), 0.5 * (e.rect.v0 + e.rect.v1), E);
        if (!region.contains(e.patch, E.x, unitNormal(E))) continue;
        C.tree.push_back({e.rect, 'i', -1, {}});
      } else {
        build(C, 0, e.rect, 0, bounded, blo, bhi);
        if (C.tree[0].state == 'o') continue;
      }
      C.full = C.tree[0].state == 'i';
      makeCloud(C, e, order);
      if (C.w.empty()) continue;
      makeNodes(C);
      cells.push_back(std::move(C));
    }
    // moments
    for (const Cell& C : cells)
      for (size_t q = 0; q < C.w.size(); ++q) {
        area += C.w[q];
        centroid += C.w[q] * C.x[q];
        normal_sum += C.w[q] * C.n[q];
        patch_area[C.patch] += C.w[q];
      }
    if (area > 0.0) centroid /= area;
    for (const Cell& C : cells)
      for (size_t q = 0; q < C.w.size(); ++q) {
        const Vec3 d = C.x[q] - centroid;
        inertia += C.w[q] * (d.squaredNorm() * Eigen::Matrix3d::Identity() - d * d.transpose());
      }
  }

  // Region area on each patch over the patch's area.
  std::vector<double> coverage() const {
    std::vector<double> out(mesh.patches.size(), 0.0), tot(mesh.patches.size(), 0.0);
    const GaussRule g = gaussLegendre(6);
    PatchEval E;
    for (const Element& e : mesh.elements)
      for (const Rect& R : e.cells)
        for (int a = 0; a < 6; ++a)
          for (int b = 0; b < 6; ++b) {
            mesh.patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
            tot[e.patch] += g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm();
          }
    for (size_t p = 0; p < out.size(); ++p) out[p] = tot[p] > 0 ? patch_area[p] / tot[p] : 0.0;
    return out;
  }

  // Sum of f(RPoint) over the fine cloud (area integrals: resultants, moments, work).
  template <class F>
  void forCloud(F&& f) const {
    for (const Cell& C : cells) {
      const int nl = int(C.idx.size());
      for (size_t q = 0; q < C.w.size(); ++q)
        f(RPoint{C.patch, C.x[q], C.n[q], C.w[q], nl, C.idx.data(), &C.R[q * nl]});
    }
  }

  bool contains(int patch, const Vec3& x, const Vec3& n) const { return region.contains(patch, x, n); }

  Vec3 unitNormal(const PatchEval& E) const { return (E.normal() * mesh.normal_sign).normalized(); }

 private:
  void ballOf(int patch, int su, int sv, const Rect& R, Vec3& c, double& r) const {
    PatchEval E;
    Vec3 X[25];
    c.setZero();
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        mesh.patches[patch].evalSpan(su, sv, R.u0 + R.du() * i / 4.0, R.v0 + R.dv() * j / 4.0, E);
        X[5 * i + j] = E.x;
        c += E.x / 25.0;
      }
    r = 0.0;
    for (const Vec3& x : X) r = std::max(r, (x - c).norm());
    r *= 1.1;
  }

  bool inside(const Cell& C, const Vec2& p) const {
    PatchEval E;
    mesh.patches[C.patch].evalSpan(C.su, C.sv, p(0), p(1), E);
    return region.contains(C.patch, E.x, unitNormal(E));
  }

  // Quadtree: 3 x 3 samples per rectangle; mixed rectangles are split down to
  // cut_depth, where the inside part is clipped (marching squares).
  void build(Cell& C, int node, const Rect& R, int depth, bool bounded, const Vec3& blo, const Vec3& bhi) {
    if (node == int(C.tree.size())) C.tree.push_back({R, 'o', -1, {}});
    // samples just inside the rectangle: a region that only touches its edge (e.g. a
    // plane selecting the neighbouring face) has no area in it
    static const double fr[3] = {1e-3, 0.5, 1.0 - 1e-3};
    int nin = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) nin += inside(C, Vec2(R.u0 + R.du() * fr[i], R.v0 + R.dv() * fr[j]));
    char st = nin == 9 ? 'i' : (nin == 0 ? 'o' : 'c');
    if (st == 'o' && bounded && depth < cut_depth) {
      // a region smaller than the rectangle can fall between the samples
      Vec3 c;
      double r;
      ballOf(C.patch, C.su, C.sv, R, c, r);
      const bool overlap = !((c.array() + r < blo.array()).any() || (c.array() - r > bhi.array()).any());
      if (overlap && 2.0 * r > 0.5 * (bhi - blo).norm()) st = 'c';
    }
    C.tree[node].state = st;
    if (st != 'c') return;
    if (depth >= cut_depth) { clip(C, node); return; }
    const int first = int(C.tree.size());
    C.tree[node].child = first;
    const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
    const Rect S[4] = {{R.u0, um, R.v0, vm}, {um, R.u1, R.v0, vm}, {R.u0, um, vm, R.v1}, {um, R.u1, vm, R.v1}};
    for (int k = 0; k < 4; ++k) C.tree.push_back({S[k], 'o', -1, {}});
    for (int k = 0; k < 4; ++k) build(C, first + k, S[k], depth + 1, bounded, blo, bhi);
    int nin4 = 0, nout4 = 0;
    for (int k = 0; k < 4; ++k) {
      nin4 += C.tree[first + k].state == 'i';
      nout4 += C.tree[first + k].state == 'o';
    }
    if (nout4 == 4) { C.tree[node].state = 'o'; C.tree[node].child = -1; C.tree.resize(first); }
  }

  // Inside part of a boundary leaf as triangles: corners and edge crossings
  // (bisection) in order around the leaf, the boundary between the two crossings
  // refined to a 4-segment polyline (points found by bisection across the chord;
  // area error ~ c^2 / (96 R^2) relative). Fan triangles carry signed weights, so a
  // slightly non-convex polygon (concave boundary) still integrates exactly.
  // Ambiguous (saddle) leaves fall back to the inside sub-squares of a 4 x 4 split.
  void clip(Cell& C, int node) {
    const Rect R = C.tree[node].r;
    const Vec2 P[4] = {{R.u0, R.v0}, {R.u1, R.v0}, {R.u1, R.v1}, {R.u0, R.v1}};
    bool in[4];
    for (int k = 0; k < 4; ++k) in[k] = inside(C, P[k]);
    std::vector<Vec2> poly;
    std::vector<char> crossing;
    int changes = 0;
    for (int k = 0; k < 4; ++k) {
      const Vec2 &a = P[k], &b = P[(k + 1) % 4];
      if (in[k]) { poly.push_back(a); crossing.push_back(0); }
      if (in[k] != in[(k + 1) % 4]) {
        ++changes;
        double s0 = 0.0, s1 = 1.0;  // in[k] at s0
        for (int it = 0; it < 45; ++it) {
          const double sm = 0.5 * (s0 + s1);
          if (inside(C, a + sm * (b - a)) == in[k]) s0 = sm; else s1 = sm;
        }
        poly.push_back(a + 0.5 * (s0 + s1) * (b - a));
        crossing.push_back(1);
      }
    }
    auto& T = C.tree[node].tris;
    if (changes == 2) {
      const int n = int(poly.size());
      for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        if (!(crossing[i] && crossing[j])) continue;
        std::vector<Vec2> arc{poly[i], poly[j]};
        for (int level = 0; level < 2; ++level) {
          std::vector<Vec2> finer{arc[0]};
          for (size_t k = 0; k + 1 < arc.size(); ++k) {
            finer.push_back(boundaryBetween(C, R, arc[k], arc[k + 1]));
            finer.push_back(arc[k + 1]);
          }
          arc = finer;
        }
        poly.insert(poly.begin() + j + (j == 0 ? n : 0), arc.begin() + 1, arc.end() - 1);
        break;
      }
      for (size_t k = 1; k + 1 < poly.size(); ++k) {
        const Vec2 u = poly[k] - poly[0], v = poly[k + 1] - poly[0];
        if (std::abs(u(0) * v(1) - u(1) * v(0)) > 1e-12 * R.du() * R.dv()) T.push_back({poly[0], poly[k], poly[k + 1]});
      }
    } else {  // 0 crossings on the edges (region inside the leaf) or a saddle
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          const Rect S{R.u0 + R.du() * i / 4.0, R.u0 + R.du() * (i + 1) / 4.0, R.v0 + R.dv() * j / 4.0,
                       R.v0 + R.dv() * (j + 1) / 4.0};
          if (!inside(C, Vec2(0.5 * (S.u0 + S.u1), 0.5 * (S.v0 + S.v1)))) continue;
          T.push_back({Vec2(S.u0, S.v0), Vec2(S.u1, S.v0), Vec2(S.u1, S.v1)});
          T.push_back({Vec2(S.u0, S.v0), Vec2(S.u1, S.v1), Vec2(S.u0, S.v1)});
        }
    }
    // a leaf whose inside part has no area (the region only touches it)
    double area = 0.0;
    for (const auto& t : T) area += std::abs(signedDet(t));
    if (area < 1e-10 * R.du() * R.dv()) { T.clear(); C.tree[node].state = 'o'; }
  }

  // Boundary point across the chord p-q (on the perpendicular through its middle,
  // within half the chord length and the leaf); the middle itself if none.
  Vec2 boundaryBetween(const Cell& C, const Rect& R, const Vec2& p, const Vec2& q) const {
    const Vec2 m = 0.5 * (p + q), d = q - p;
    const Vec2 nrm(-d(1), d(0));
    const bool sm = inside(C, m);
    for (double sgn : {1.0, -1.0}) {
      const Vec2 e = R.clamp(m + sgn * 0.5 * nrm);
      if (inside(C, e) == sm) continue;
      double s0 = 0.0, s1 = 1.0;
      for (int it = 0; it < 45; ++it) {
        const double sh = 0.5 * (s0 + s1);
        if (inside(C, m + sh * (e - m)) == sm) s0 = sh; else s1 = sh;
      }
      return m + 0.5 * (s0 + s1) * (e - m);
    }
    return m;
  }

 public:
  static double signedDet(const std::array<Vec2, 3>& t) {
    const Vec2 a = t[1] - t[0], d = t[2] - t[1];
    return a(0) * d(1) - a(1) * d(0);
  }

 private:
  void addPoint(Cell& C, const Vec2& p, double wp) {
    PatchEval E;
    mesh.patches[C.patch].evalSpan(C.su, C.sv, p(0), p(1), E);
    if (C.idx.empty()) C.idx.assign(E.idx, E.idx + E.n_loc);
    C.xi.push_back(p);
    C.x.push_back(E.x);
    C.n.push_back(unitNormal(E));
    C.w.push_back(wp * E.normal().norm());
    C.R.insert(C.R.end(), E.R, E.R + E.n_loc);
  }

  void makeCloud(Cell& C, const Element& e, int order) {
    const GaussRule g = gaussLegendre(order);
    auto rect = [&](const Rect& R) {
      for (int a = 0; a < order; ++a)
        for (int b = 0; b < order; ++b)
          addPoint(C, Vec2(R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b]), g.w[a] * g.w[b] * R.du() * R.dv());
    };
    for (const Node& N : C.tree) {
      if (N.child >= 0) continue;
      if (N.state == 'i') {
        if (C.full) for (const Rect& R : e.cells) rect(R);  // App. B partition of the whole element
        else rect(N.r);
      } else if (N.state == 'c') {
        for (const auto& t : N.tris) triangle(C, t, g, order);
      }
    }
  }

  // Collapsed-square rule on a parametric triangle (t[0] the collapsed vertex).
  void triangle(Cell& C, const std::array<Vec2, 3>& t, const GaussRule& g, int order) {
    const Vec2 a = t[1] - t[0], d = t[2] - t[1];
    const double det = signedDet(t);  // signed: fan triangles of a non-convex polygon
    for (int i = 0; i < order; ++i)
      for (int j = 0; j < order; ++j) {
        const double s = g.x[i];
        addPoint(C, t[0] + s * (a + g.x[j] * d), g.w[i] * g.w[j] * s * det);
      }
  }

  void makeNodes(Cell& C) {
    PatchEval E;
    for (int k = 0; k < 4; ++k) {
      const GaussRule g = gaussLegendre(kM[k]);
      for (int a = 0; a < kM[k]; ++a)
        for (int b = 0; b < kM[k]; ++b) {
          const Vec2 p(C.rect.u0 + C.rect.du() * g.x[a], C.rect.v0 + C.rect.dv() * g.x[b]);
          mesh.patches[C.patch].evalSpan(C.su, C.sv, p(0), p(1), E);
          C.node_xi[k].push_back(p);
          C.node_x[k].push_back(E.x);
          C.node_n[k].push_back(unitNormal(E));
        }
    }
  }
};

// Lagrange basis on the m Gauss nodes g (on [0, 1]) at s.
inline void lagrange(const GaussRule& g, double s, double* l) {
  const int m = int(g.x.size());
  for (int a = 0; a < m; ++a) {
    double v = 1.0;
    for (int b = 0; b < m; ++b)
      if (b != a) v *= (s - g.x[b]) / (g.x[a] - g.x[b]);
    l[a] = v;
  }
}

// ------------------------------------------------------------ region integrals
// A traction density on a region with K <= 8 columns (load cases), and its
// far-field node weights W = sum_q l(xi_q) t(y_q) w_q per cell and node level.
using RegionBlock = Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor, 3, 8>;

struct RegionDensity {
  const RegionMesh* R = nullptr;
  int K = 1;
  std::function<RegionBlock(const RPoint&)> t;
  std::vector<std::array<std::vector<RegionBlock>, 4>> W;  // per cell, per level, per node

  RegionDensity() = default;
  RegionDensity(const RegionMesh& rm, int k, std::function<RegionBlock(const RPoint&)> f)
      : R(&rm), K(k), t(std::move(f)) {
    W.resize(rm.cells.size());
#pragma omp parallel for schedule(dynamic)
    for (long ic = 0; ic < long(rm.cells.size()); ++ic) {
      const RegionMesh::Cell& C = rm.cells[ic];
      const int nl = int(C.idx.size());
      std::vector<RegionBlock> tq(C.w.size());
      for (size_t q = 0; q < C.w.size(); ++q)
        tq[q] = t(RPoint{C.patch, C.x[q], C.n[q], C.w[q], nl, C.idx.data(), &C.R[q * nl]}) * C.w[q];
      for (int lv = 0; lv < 4; ++lv) {
        const int m = RegionMesh::kM[lv];
        const GaussRule g = gaussLegendre(m);
        std::vector<RegionBlock> Wl(m * m, RegionBlock::Zero(3, K));
        double la[10], lb[10];
        for (size_t q = 0; q < C.w.size(); ++q) {
          lagrange(g, (C.xi[q](0) - C.rect.u0) / C.rect.du(), la);
          lagrange(g, (C.xi[q](1) - C.rect.v0) / C.rect.dv(), lb);
          for (int a = 0; a < m; ++a)
            for (int b = 0; b < m; ++b) Wl[a * m + b].noalias() += (la[a] * lb[b]) * tq[q];
        }
        W[ic][lv] = std::move(Wl);
      }
    }
  }

  // Resultant force and moment about o, from the cloud.
  void resultant(int col, const Vec3& o, Vec3& F, Vec3& M) const {
    F.setZero();
    M.setZero();
    R->forCloud([&](const RPoint& p) {
      const Vec3 f = t(p).col(col) * p.w;
      F += f;
      M += (p.x - o).cross(f);
    });
  }
};

class RegionIntegrator {
 public:
  RegionIntegrator(const RegionMesh& R, const SolverOptions& o) : R_(R), opt_(o) {
    for (int k = 1; k <= std::max(o.n_gauss, 12); ++k) rules_.push_back(gaussLegendre(k));
  }

  // Far-field level for a cell seen from x: -1 = near (use the sub-cells).
  int farLevel(const RegionMesh::Cell& C, const Vec3& x) const {
    const double q = (x - C.c).norm() / C.r;
    if (q < 5.0) return -1;
    const double rho = q + std::sqrt(q * q - 1.0), tol = std::max(opt_.far_tol, 1e-14);
    for (int k = 0; k < 4; ++k)
      if (std::pow(rho, -RegionMesh::kM[k]) <= tol) return k;
    return 3;
  }

  // Near part: f(RPoint) at every quadrature point of cell C, refined around the source.
  template <class F>
  void near(const SourcePoint& src, const RegionMesh::Cell& C, F&& f) const {
    const Element& e = R_.mesh.elements[C.elem];
    Vec2 xs(0.0, 0.0);
    double dist = 1e300;
    if (src.patch == C.patch && C.rect.contains(src.xi, 1e-14)) { xs = src.xi; dist = 0.0; }
    else if ((src.x - C.c).norm() <= C.r) xs = R_.mesh.closestPoint(e, src.x, dist);
    const bool singular = dist < 1e-11 * C.h;
    walk(src, C, 0, singular, xs, f);
  }

  // Visits the density over the region from src: f(y, n, tw) with tw (3 x K)
  // the density times the weight at a near quadrature point, or a far cell's
  // node weight. Kernel x density integrals are sums of kernel(y) * tw.
  template <class F>
  void visit(const SourcePoint& src, const RegionDensity& d, F&& f) const {
    for (size_t ic = 0; ic < R_.cells.size(); ++ic) {
      const RegionMesh::Cell& C = R_.cells[ic];
      const int lv = farLevel(C, src.x);
      if (lv >= 0) {
        const auto& W = d.W[ic][lv];
        for (size_t a = 0; a < W.size(); ++a) f(C.node_x[lv][a], C.node_n[lv][a], W[a]);
      } else {
        near(src, C, [&](const RPoint& p) { f(p.x, p.n, RegionBlock(d.t(p) * p.w)); });
      }
    }
  }

  const RegionMesh& region() const { return R_; }

 private:
  const RegionMesh& R_;
  SolverOptions opt_;
  std::vector<GaussRule> rules_;

  int order(const Vec3& x, const Vec3& c, double r) const {
    const int n = opt_.n_gauss;
    const double q = (x - c).norm() / r;
    if (q <= 1.5 || opt_.far_tol <= 0.0) return n;
    const double rho = q + std::sqrt(q * q - 1.0);
    const int m = int(std::ceil(std::log(1.0 / opt_.far_tol) / (2.0 * std::log(rho))));
    return std::min(n, std::max(3, m));
  }

  template <class F>
  void point(const RegionMesh::Cell& C, const Vec2& p, double w, F& f) const {
    PatchEval E;
    R_.mesh.patches[C.patch].evalSpan(C.su, C.sv, p(0), p(1), E);
    f(RPoint{C.patch, E.x, R_.unitNormal(E), w * E.normal().norm(), E.n_loc, E.idx, E.R});
  }

  void rectBall(const RegionMesh::Cell& C, const Rect& R, Vec3& c, double& h) const {
    PatchEval E;
    Vec3 X[4];
    int k = 0;
    for (double u : {R.u0, R.u1})
      for (double v : {R.v0, R.v1}) { R_.mesh.patches[C.patch].evalSpan(C.su, C.sv, u, v, E); X[k++] = E.x; }
    h = std::max((X[3] - X[0]).norm(), (X[2] - X[1]).norm());
    c = 0.25 * (X[0] + X[1] + X[2] + X[3]);
  }

  template <class F>
  void gaussRect(const RegionMesh::Cell& C, const Rect& R, int m, F& f) const {
    const GaussRule& g = rules_[m - 1];
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        point(C, Vec2(R.u0 + R.du() * g.x[i], R.v0 + R.dv() * g.x[j]), g.w[i] * g.w[j] * R.du() * R.dv(), f);
  }

  // Duffy split of R into triangles with a common vertex at xs (on R).
  template <class F>
  void duffyRect(const RegionMesh::Cell& C, const Rect& R, const Vec2& xs, F& f) const {
    const Vec2 P[4] = {{R.u0, R.v0}, {R.u1, R.v0}, {R.u1, R.v1}, {R.u0, R.v1}};
    const GaussRule& g = rules_[opt_.n_gauss - 1];
    const int n = opt_.n_gauss;
    for (int k = 0; k < 4; ++k) {
      const Vec2 a = P[k] - xs, d = P[(k + 1) % 4] - P[k];
      const double det = std::abs(a(0) * d(1) - a(1) * d(0));
      if (det < 1e-14 * R.du() * R.dv()) continue;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) point(C, xs + g.x[i] * (a + g.x[j] * d), g.w[i] * g.w[j] * g.x[i] * det, f);
    }
  }

  // Fully inside rectangle: Duffy around a source on it (block of aspect <= 2
  // around xs, as ElementIntegrator::singularRect), quadtree while the source
  // is within 3 h, Gauss of the far-field order otherwise.
  template <class F>
  void inRect(const SourcePoint& src, const RegionMesh::Cell& C, const Rect& R, bool singular, const Vec2& xs,
              int depth, F& f) const {
    const double tol = 1e-12 * std::max(R.du(), R.dv());
    if (singular && R.contains(xs, tol)) {
      const Vec2 p = R.clamp(xs);
      PatchEval E;
      R_.mesh.patches[C.patch].evalSpan(C.su, C.sv, p(0), p(1), E);
      const double su = E.xu.norm(), sv = E.xv.norm();
      auto half = [&](double a0, double a1) {
        if (a0 <= tol) return a1;
        if (a1 <= tol) return a0;
        return std::min(a0, a1);
      };
      double hu = half(p(0) - R.u0, R.u1 - p(0)), hv = half(p(1) - R.v0, R.v1 - p(1));
      hu = std::min(hu, 2.0 * hv * sv / su);
      hv = std::min(hv, 2.0 * hu * su / sv);
      const double cu[4] = {R.u0, std::max(R.u0, p(0) - hu), std::min(R.u1, p(0) + hu), R.u1};
      const double cv[4] = {R.v0, std::max(R.v0, p(1) - hv), std::min(R.v1, p(1) + hv), R.v1};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const Rect S{cu[i], cu[i + 1], cv[j], cv[j + 1]};
          if (S.du() <= tol || S.dv() <= tol) continue;
          if (i == 1 && j == 1) duffyRect(C, S, S.clamp(p), f);
          else inRect(src, C, S, false, xs, depth + 1, f);
        }
      return;
    }
    Vec3 c;
    double h;
    rectBall(C, R, c, h);
    // as ElementIntegrator::adaptiveRect: split while the source is within 1.5
    // radii or even n_gauss points miss far_tol
    const double q = (src.x - c).norm() / (0.6 * h);
    bool split = q <= 1.5;
    if (!split && opt_.far_tol > 0.0) {
      const double rho = q + std::sqrt(q * q - 1.0);
      split = std::pow(rho, -2.0 * opt_.n_gauss) > opt_.far_tol;
    }
    if (split && depth < opt_.near_max_depth) {
      const double um = 0.5 * (R.u0 + R.u1), vm = 0.5 * (R.v0 + R.v1);
      inRect(src, C, {R.u0, um, R.v0, vm}, singular, xs, depth + 1, f);
      inRect(src, C, {um, R.u1, R.v0, vm}, singular, xs, depth + 1, f);
      inRect(src, C, {R.u0, um, vm, R.v1}, singular, xs, depth + 1, f);
      inRect(src, C, {um, R.u1, vm, R.v1}, singular, xs, depth + 1, f);
      return;
    }
    const int m = order(src.x, c, 0.6 * h);
    gaussRect(C, R, m, f);
  }

  // Clipped triangle: split in 4 while the source is close, collapsed Gauss otherwise.
  template <class F>
  void triangle(const SourcePoint& src, const RegionMesh::Cell& C, const std::array<Vec2, 3>& t, int depth,
                F& f) const {
    PatchEval E;
    Vec3 X[3];
    for (int k = 0; k < 3; ++k) { R_.mesh.patches[C.patch].evalSpan(C.su, C.sv, t[k](0), t[k](1), E); X[k] = E.x; }
    const Vec3 c = (X[0] + X[1] + X[2]) / 3.0;
    const double h = std::max({(X[1] - X[0]).norm(), (X[2] - X[1]).norm(), (X[0] - X[2]).norm()});
    if ((src.x - c).norm() <= 1.5 * h && depth < 6) {
      const Vec2 m01 = 0.5 * (t[0] + t[1]), m12 = 0.5 * (t[1] + t[2]), m20 = 0.5 * (t[2] + t[0]);
      triangle(src, C, {t[0], m01, m20}, depth + 1, f);
      triangle(src, C, {m01, t[1], m12}, depth + 1, f);
      triangle(src, C, {m20, m12, t[2]}, depth + 1, f);
      triangle(src, C, {m01, m12, m20}, depth + 1, f);
      return;
    }
    const int m = std::max(4, order(src.x, c, 0.6 * h));
    const GaussRule& g = rules_[m - 1];
    const Vec2 a = t[1] - t[0], d = t[2] - t[1];
    const double det = RegionMesh::signedDet(t);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j) point(C, t[0] + g.x[i] * (a + g.x[j] * d), g.w[i] * g.w[j] * g.x[i] * det, f);
  }

  template <class F>
  void walk(const SourcePoint& src, const RegionMesh::Cell& C, int node, bool singular, const Vec2& xs,
            F& f) const {
    const RegionMesh::Node& N = C.tree[node];
    if (N.state == 'o') return;
    if (N.child >= 0) {
      for (int k = 0; k < 4; ++k) walk(src, C, N.child + k, singular, xs, f);
      return;
    }
    if (N.state == 'i') {
      if (C.full) {
        for (const Rect& R : R_.mesh.elements[C.elem].cells) inRect(src, C, R, singular, xs, 0, f);
      } else {
        inRect(src, C, N.r, singular, xs, 0, f);
      }
      return;
    }
    for (const auto& t : N.tris) triangle(src, C, t, 0, f);
  }
};

}  // namespace igabem
