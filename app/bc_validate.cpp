// Validation of local supports and loads (BcModel) against exact solutions.
//
// usage: bc_validate halfspace <geo/box_hs_q_p3_rK.dat> [--csv f] [--probe f] [--cache dir|none]
//        bc_validate bar <geo/bar_L4_p3_n8.dat> [--csv f]
//        bc_validate cantilever <geo/bar_L10_p3_nN.dat> [--csv f] [--probe f]
//   --check   regression mode (ctest): compare the key numbers with the committed reference
//             values (results/bc_*.csv) and exit 2 on any mismatch
//
//   halfspace   quarter block [0,4]^2 x [-4,0], pressure p on the quarter disk r <= 1 of
//               the top face (a region the patches do not follow: cut cells), symmetry on
//               x = 0 and y = 0, the exact half-space (Love) traction on the far faces:
//               no truncation error. One rigid motion (z) is free. Errors: boundary
//               displacement (L2, z translation removed) and the stress along a line under
//               the load, against the integrated Boussinesq field.
//   bar         bar [0,4] x [-1/2,1/2]^2, symmetry on x = 0, loads on the end face x = 4:
//               a resultant moment (exact pure bending), a resultant axial force, and a
//               spring end (t = sigma0 e_x - k u_x e_x), all exact polynomial fields;
//               3 rigid motions free (y, z, rotation about x).
//   cantilever  bar [0,10], clamped at x = 0, resultant shear force at x = 10: deflection
//               and bending stress against Timoshenko beam theory (not exact: clamp
//               and load-distribution effects decay over ~ one depth).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "bc_model.hpp"
#include "interior.hpp"
#include "treecode.hpp"
#include "loads.hpp"
#include "problems/half_space.hpp"

using namespace igabem;

namespace {

bool g_check = false;
static BcModel::HSolver g_hsolver;  // --hmat eps
int g_fail = 0;

// |got - want| <= tol, reported only in --check mode
void check(const char* name, double got, double want, double tol) {
  if (!g_check) return;
  if (g_hsolver.on) tol = std::max(tol, 1e-7 * std::max(1.0, std::abs(want)));  // float operator storage floor
  const bool ok = std::abs(got - want) <= tol;
  std::printf("  check %-34s %s  got %.8e  want %.8e  tol %.1e\n", name, ok ? "ok  " : "FAIL", got, want, tol);
  if (!ok) ++g_fail;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<int> patchesOn(const Mesh& m, int axis, double v, double tol = 1e-9) {
  std::vector<int> out;
  for (size_t p = 0; p < m.patches.size(); ++p) {
    bool on = true;
    for (int k = 0; k < m.patches[p].nu * m.patches[p].nv; ++k)
      on = on && std::abs(m.patches[p].cp(k)(axis) - v) < tol;
    if (on) out.push_back(int(p));
  }
  return out;
}

Region planeRegion(int axis, double v, const std::string& label) {
  Region R;
  R.label = label;
  Selector s;
  s.plane = true;
  s.p0 = Vec3::Unit(axis) * v;
  s.pn = Vec3::Unit(axis);
  s.ptol = 1e-9;
  R.any.push_back(s);
  return R;
}

// Relative L2(Gamma) displacement error; removes the best translation along the
// directions in `free` (the rigid motions the supports leave free).
template <class Fu>
double l2Error(const Mesh& m, const Eigen::VectorXd& c, Fu&& exact, const std::vector<Vec3>& free, int ng = 5) {
  const GaussRule g = gaussLegendre(ng);
  struct Q { Vec3 e, ex; double w; };
  std::vector<std::vector<Q>> per(m.elements.size());
#pragma omp parallel for schedule(dynamic)
  for (long ie = 0; ie < long(m.elements.size()); ++ie) {
    const Element& el = m.elements[ie];
    PatchEval E;
    for (const Rect& R : el.cells)
      for (int i = 0; i < ng; ++i)
        for (int j = 0; j < ng; ++j) {
          m.patches[el.patch].evalSpan(el.su, el.sv, R.u0 + R.du() * g.x[i], R.v0 + R.dv() * g.x[j], E);
          Vec3 uh = Vec3::Zero();
          for (int l = 0; l < E.n_loc; ++l) uh += E.R[l] * c.segment<3>(3 * m.l2g[el.patch][E.idx[l]]);
          const Vec3 ue = exact(E.x);
          per[ie].push_back({uh - ue, ue, g.w[i] * g.w[j] * R.du() * R.dv() * E.normal().norm()});
        }
  }
  Vec3 shift = Vec3::Zero();
  double W = 0.0;
  for (const auto& v : per)
    for (const Q& q : v) { shift += q.w * q.e; W += q.w; }
  shift /= W;
  Vec3 s = Vec3::Zero();
  for (const Vec3& d : free) s += shift.dot(d) * d;
  double num = 0.0, den = 0.0;
  std::vector<double> pn(m.patches.size(), 0.0);
  for (size_t ie = 0; ie < per.size(); ++ie)
    for (const Q& q : per[ie]) {
      num += q.w * (q.e - s).squaredNorm();
      den += q.w * q.ex.squaredNorm();
      pn[m.elements[ie].patch] += q.w * (q.e - s).squaredNorm();
    }
  if (std::getenv("IGABEM_PATCH_ERRORS"))
    for (size_t p = 0; p < pn.size(); ++p) std::printf("    patch %zu: error share %.3f\n", p, pn[p] / num);
  return std::sqrt(num / den);
}

void printReport(const CaseReport& r, double L = 1.0) {
  std::printf("  %s: applied F (%.6e, %.6e, %.6e); ", r.name.c_str(), r.applied.F(0), r.applied.F(1), r.applied.F(2));
  Vec3 react = Vec3::Zero();
  for (const auto& [n, W] : r.reactions) react += W.F;
  std::printf("reactions (%.6e, %.6e, %.6e); equilibrium residual %.2e; free work %.2e\n", react(0), react(1),
              react(2), r.residual(L), r.free_work);
}

int halfspace(const std::string& geo, const std::string& csv, const std::string& probe, const std::string& cache) {
  const double t0 = now();
  HalfSpaceDisk hs;
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  BcModel model;
  model.mat = hs.mat;
  model.opt = opt;
  model.mesh.build(loadDat(geo), model.popt);
  model.ts.build(model.mesh);
  const Mesh& mesh = model.mesh;
  Region disk;
  disk.label = "pressure disk";
  Selector s;
  s.ball = true;
  s.bc = Vec3::Zero();
  s.br = hs.a;
  s.plane = true;
  s.p0 = Vec3::Zero();
  s.pn = Vec3::UnitZ();
  s.ptol = 1e-9;
  disk.any.push_back(s);
  const double t_r0 = now();
  auto RM = std::make_shared<RegionMesh>(mesh, disk);
  const double t_region = now() - t_r0;
  std::vector<FixedSupport> fixed = {{"symmetry x", patchesOn(mesh, 0, 0.0), {true, false, false}, nullptr},
                                     {"symmetry y", patchesOn(mesh, 1, 0.0), {false, true, false}, nullptr}};
  model.hsolver = g_hsolver;
    model.setupOn(geo, fixed, {}, cache);
  LoadCase lc;
  lc.name = "disk";
  lc.loads.push_back(pressureLoad("pressure", RM, hs.p));
  lc.smooth = [&](const Vec3& y, const Vec3& n) -> Vec3 {
    if (y(2) > -1e-9) return Vec3::Zero();  // the free surface (the disk is the region load)
    return hs.stress(y) * n;
  };
  const double t_s0 = now();
  std::vector<CaseReport> rep;
  const auto sol = model.solutions({lc}, &rep);
  const double t_solve = now() - t_s0;
  std::printf("# %s: %zu elements, %d dofs, %d unknowns, %d free rigid motion(s); quarter-disk area %.10f "
              "(exact %.10f), %zu region cells\n",
              geo.c_str(), mesh.elements.size(), mesh.ndof, model.unknowns(), model.freeModes(), RM->area,
              M_PI / 4 * hs.a * hs.a, RM->cells.size());
  std::printf("# region %.2f s, operators %.2f s%s, LU %.2f s, solve (exact data + region RHS) %.2f s\n", t_region,
              model.t_assemble, model.from_cache ? " (cache)" : "", model.t_factor, t_solve);
  printReport(rep[0]);
  const double t_e0 = now();
  const double eu = l2Error(mesh, sol[0].u, [&](const Vec3& x) { return hs.displacement(x); }, {Vec3::UnitZ()});
  // stress along a line under the load (just off the symmetry edge)
  InteriorEvaluator ev(sol[0], opt);
  const double eps = 0.02;
  std::vector<Vec3> xs;
  const int np = 40;
  for (int i = 0; i <= np; ++i) xs.push_back(Vec3(eps, eps, -(0.05 + 2.95 * i / double(np))));
  const auto f = ev.eval(xs);
  double emax = 0.0;
  std::ofstream pr;
  if (!probe.empty()) {
    pr.open(probe);
    pr.precision(12);
    pr << "zeta,szz,srr,sxx,szz_exact,sxx_exact,szz_axis,srr_axis\n";
  }
  double szz_at[3] = {0, 0, 0}, szz_ex_at[3] = {0, 0, 0};
  for (size_t i = 0; i < xs.size(); ++i) {
    const Eigen::Matrix3d S = hs.stress(xs[i]);
    emax = std::max(emax, (f[i].stress - S).cwiseAbs().maxCoeff() / hs.p);
    const double zeta = -xs[i](2);
    // radial direction at 45 degrees: s_rr = (sxx + syy)/2 + sxy
    const double srr = 0.5 * (f[i].stress(0, 0) + f[i].stress(1, 1)) + f[i].stress(0, 1);
    if (pr) pr << zeta << ',' << f[i].stress(2, 2) << ',' << srr << ',' << f[i].stress(0, 0) << ',' << S(2, 2) << ','
               << S(0, 0) << ',' << hs.szzAxis(zeta) << ',' << hs.srrAxis(zeta) << '\n';
  }
  const double zq[3] = {0.5, 1.0, 2.0};
  for (int k = 0; k < 3; ++k) {
    const auto fz = ev.eval(Vec3(eps, eps, -zq[k]));
    szz_at[k] = fz.stress(2, 2);
    szz_ex_at[k] = hs.stress(Vec3(eps, eps, -zq[k]))(2, 2);
  }
  std::printf("  u L2 error %.4e; max |sigma_h - sigma| / p on the line %.4e; sigma_zz/p at zeta = 0.5, 1, 2: "
              "%.6f %.6f %.6f (exact %.6f %.6f %.6f); errors %.1f s\n",
              eu, emax, szz_at[0], szz_at[1], szz_at[2], szz_ex_at[0], szz_ex_at[1], szz_ex_at[2], now() - t_e0);
  std::printf("# end to end %.2f s\n", now() - t0);
  if (geo.find("box_hs_q_p3_r0") != std::string::npos) {  // results/bc_halfspace.csv, r0
    check("halfspace u L2 error", eu, 5.631042258e-3, 2e-5);
    check("halfspace max sigma error / p", emax, 1.890449078e-3, 2e-5);
    check("halfspace sigma_zz/p at zeta = 1", szz_at[1], -0.6474473022, 1e-5);
    check("halfspace equilibrium residual", rep[0].residual(4.0), 0.0, 5e-3);
  }
  if (!csv.empty()) {
    std::ifstream probe_f(csv);
    const bool fresh = !probe_f.good() || probe_f.peek() == std::ifstream::traits_type::eof();
    std::ofstream o(csv, std::ios::app);
    o.precision(10);
    if (fresh) o << "geometry,elements,ndof,unknowns,area_err,u_l2_err,sigma_line_max_err,szz_05,szz_1,szz_2,"
                    "szz_05_ex,szz_1_ex,szz_2_ex,equilibrium,free_work,t_assemble,t_factor,t_solve\n";
    o << geo << ',' << mesh.elements.size() << ',' << mesh.ndof << ',' << model.unknowns() << ','
      << std::abs(RM->area - M_PI / 4) / (M_PI / 4) << ',' << eu << ',' << emax << ',' << szz_at[0] << ','
      << szz_at[1] << ',' << szz_at[2] << ',' << szz_ex_at[0] << ',' << szz_ex_at[1] << ',' << szz_ex_at[2] << ','
      << rep[0].residual(4.0) << ',' << rep[0].free_work << ',' << model.t_assemble
      << ',' << model.t_factor << ',' << t_solve << '\n';
  }
  return 0;
}

int bar(const std::string& geo, const std::string& csv) {
  const double t0 = now();
  Material mat;  // lambda = mu = 1: E = 2.5, nu = 1/4
  const double E = 2.5, nu = 0.25, L = 4.0, I = 1.0 / 12.0;
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  std::ofstream o;
  if (!csv.empty()) {
    o.open(csv);
    o.precision(10);
    o << "case,unknowns,free_modes,u_l2_err,sigma_err,equilibrium,free_work\n";
  }
  // symmetry on x = 0; loads on the end x = L
  for (int pass = 0; pass < 2; ++pass) {
    BcModel model;
    model.mat = mat;
    model.opt = opt;
    model.mesh.build(loadDat(geo), model.popt);
    model.ts.build(model.mesh);
    auto end = std::make_shared<RegionMesh>(model.mesh, planeRegion(0, L, "end"));
    std::vector<FixedSupport> fixed = {{"symmetry x", patchesOn(model.mesh, 0, 0.0), {true, false, false}, nullptr}};
    std::vector<SpringSupport> springs;
    const double k = E / L, s0 = 1.0, sig = s0 / (1.0 + k * L / E);
    if (pass == 1) springs.push_back({"end spring", end, springNT(k, 0.0)});
    model.hsolver = g_hsolver;
    model.setupOn(geo, fixed, springs, "");
    std::vector<LoadCase> cases;
    std::vector<std::function<Vec3(const Vec3&)>> uex;
    std::vector<std::function<Eigen::Matrix3d(const Vec3&)>> sex;
    if (pass == 0) {
      const double M = 1.0, kap = M / (E * I);
      LoadCase a;
      a.name = "moment";
      a.loads.push_back(wrenchLoad("end moment", end, Vec3::Zero(), Vec3(0, 0, M), Vec3(L, 0, 0)));
      cases.push_back(a);
      uex.push_back([=](const Vec3& x) {
        return Vec3(-kap * x(0) * x(1), 0.5 * kap * (x(0) * x(0) + nu * (x(1) * x(1) - x(2) * x(2))), nu * kap * x(1) * x(2));
      });
      sex.push_back([=](const Vec3& x) {
        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        S(0, 0) = -M * x(1) / I;
        return S;
      });
      LoadCase b;
      b.name = "axial";
      b.loads.push_back(forceLoad("end force", end, Vec3(1, 0, 0)));
      cases.push_back(b);
      uex.push_back([=](const Vec3& x) { return Vec3(x(0) / E, -nu * x(1) / E, -nu * x(2) / E); });
      sex.push_back([=](const Vec3&) {
        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        S(0, 0) = 1.0;
        return S;
      });
    } else {
      LoadCase c;
      c.name = "spring";
      c.loads.push_back(tractionLoad("end traction", end, Vec3(s0, 0, 0)));
      cases.push_back(c);
      uex.push_back([=](const Vec3& x) { return Vec3(sig * x(0) / E, -nu * sig * x(1) / E, -nu * sig * x(2) / E); });
      sex.push_back([=](const Vec3&) {
        Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
        S(0, 0) = sig;
        return S;
      });
    }
    std::vector<CaseReport> rep;
    const auto sol = model.solutions(cases, &rep);
    std::printf("# %s (%s): %d unknowns, %d free rigid motion(s), LU %.2f s, springs %.2f s\n", geo.c_str(),
                pass ? "spring end" : "loaded end", model.unknowns(), model.freeModes(), model.t_factor,
                model.t_springs);
    for (size_t k = 0; k < cases.size(); ++k) {
      printReport(rep[k], L);
      const double eu = displacementL2Error(model.mesh, sol[k].u, uex[k], 10, true);
      InteriorEvaluator ev(sol[k], opt);
      double es = 0.0;
      for (const Vec3& x : {Vec3(1.0, 0.2, -0.1), Vec3(2.5, -0.3, 0.25), Vec3(3.6, 0.1, 0.3)})
        es = std::max(es, (ev.eval(x).stress - sex[k](x)).cwiseAbs().maxCoeff());
      std::printf("    u L2 error %.3e (rigid part removed), interior stress error %.3e\n", eu, es);
      // exact polynomial fields: only quadrature and solver round-off remain
      check((cases[k].name + " u L2 error").c_str(), eu, 0.0, 1e-6);
      check((cases[k].name + " stress error").c_str(), es, 0.0, 1e-5);
      check((cases[k].name + " equilibrium residual").c_str(), rep[k].residual(L), 0.0, 1e-6);
      if (o) o << cases[k].name << ',' << model.unknowns() << ',' << model.freeModes() << ',' << eu << ',' << es
               << ',' << rep[k].residual(L) << ',' << rep[k].free_work << '\n';
    }
  }
  std::printf("# end to end %.2f s\n", now() - t0);
  return 0;
}

int cantilever(const std::string& geo, const std::string& csv, const std::string& probe) {
  const double t0 = now();
  Material mat;
  const double E = 2.5, G = 1.0, nu = 0.25, L = 10.0, I = 1.0 / 12.0, A = 1.0, F = 1.0;
  const double kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu);  // rectangle (Cowper)
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  BcModel model;
  model.mat = mat;
  model.opt = opt;
  model.mesh.build(loadDat(geo), model.popt);
  model.ts.build(model.mesh);
  auto end = std::make_shared<RegionMesh>(model.mesh, planeRegion(0, L, "tip"));
  model.hsolver = g_hsolver;
    model.setupOn(geo, {{"clamp", patchesOn(model.mesh, 0, 0.0), {true, true, true}, nullptr}}, {}, "");
  LoadCase lc;
  lc.name = "tip shear";
  lc.loads.push_back(forceLoad("tip force", end, Vec3(0, 0, -F)));
  std::vector<CaseReport> rep;
  const auto sol = model.solutions({lc}, &rep);
  std::printf("# %s: %zu elements, %d unknowns, %d free rigid motion(s); operators %.2f s, LU %.2f s\n", geo.c_str(),
              model.mesh.elements.size(), model.unknowns(), model.freeModes(), model.t_assemble, model.t_factor);
  printReport(rep[0]);
  const Wrench& Rc = rep[0].reactions[0].second;
  std::printf("  clamp reaction F (%.6f, %.6f, %.6f), M_y %.6f (statics: F_z = %.1f, M_y = %.1f)\n", Rc.F(0), Rc.F(1),
              Rc.F(2), Rc.M(1), F, -F * L);
  auto wTim = [&](double x) { return -(F / (E * I) * (L * x * x / 2 - x * x * x / 6) + F * x / (kappa * G * A)); };
  InteriorEvaluator ev(sol[0], opt);
  std::ofstream pr;
  if (!probe.empty()) {
    pr.open(probe);
    pr.precision(12);
    pr << "x,w,w_timoshenko,w_euler,sxx_top,sxx_beam\n";
  }
  double tip = 0.0, smid = 0.0;
  for (int i = 1; i <= 20; ++i) {
    const double x = L * i / 20.0;
    const Vec3 p(std::min(x, L - 1e-3), 0.0, 0.0);
    const double w = ev.eval(p).u(2);
    const SurfacePoint sp = locate(model.mesh, Vec3(std::min(x, L - 1e-3), 0.0, 0.5), [](int) { return true; });
    const double sxx = sol[0].stress(sp)(0, 0), sbeam = F * (L - x) * 0.5 / I;
    if (i == 20) tip = w;
    if (i == 10) smid = sxx;
    if (pr) pr << x << ',' << w << ',' << wTim(x) << ',' << -F / (E * I) * (L * x * x / 2 - x * x * x / 6) << ','
               << sxx << ',' << sbeam << '\n';
  }
  std::printf("  tip deflection %.6f vs Timoshenko %.6f (%+.3f%%), Euler-Bernoulli %.6f; sigma_xx top at mid-span "
              "%.5f vs beam %.5f (%+.3f%%)\n",
              tip, wTim(L - 1e-3), 100 * (tip / wTim(L - 1e-3) - 1), -F * L * L * L / (3 * E * I), smid,
              F * (L / 2) * 0.5 / I, 100 * (smid / (F * (L / 2) * 0.5 / I) - 1));
  std::printf("# end to end %.2f s\n", now() - t0);
  if (geo.find("bar_L10_p3_n20") != std::string::npos) {  // results/bc_cantilever.csv, n20
    check("cantilever tip deflection", tip, -1620.910682, 2e-3);
    check("cantilever mid-span sigma_xx", smid, 30.16049364, 1e-4);
    check("cantilever clamp reaction F_z", Rc.F(2), 0.9482568948, 1e-5);
  }
  if (!csv.empty()) {
    std::ifstream probe_f(csv);
    const bool fresh = !probe_f.good() || probe_f.peek() == std::ifstream::traits_type::eof();
    std::ofstream o(csv, std::ios::app);
    o.precision(10);
    if (fresh) o << "geometry,unknowns,tip,tip_timoshenko,smid,smid_beam,reaction_Fz,reaction_My,equilibrium\n";
    o << geo << ',' << model.unknowns() << ',' << tip << ',' << wTim(L - 1e-3) << ',' << smid << ','
      << F * (L / 2) * 0.5 / I << ',' << Rc.F(2) << ',' << Rc.M(1) << ','
      << rep[0].residual(4.0) << '\n';
  }
  return 0;
}

}  // namespace

// Rigid coupling (RBE2) and the treecode on the cantilever bar [0,10] x [-1/2,1/2]^2, clamped at x = 0: the tip
// face moves with a master at (10, 0, 0) loaded by (a) a shear force F = 1, against Timoshenko (+0.27 %: the
// clamp's root flexibility, as in the cantilever case) and (b) a torque T = 0.5, against Saint-Venant torsion of
// the unit square (J = 0.140577 a^4; -0.22 %: the rigid end restrains warping). Then the treecode against the direct evaluation at 2,000
// points inside the bar.
int rbe2Tree(const std::string& geo) {
  const double t0 = now();
  Material mat;  // E = 2.5, G = 1, nu = 0.25
  const double E = 2.5, G = 1.0, nu = 0.25, L = 10.0, I = 1.0 / 12.0, A = 1.0, F = 1.0, T = 0.5, J = 0.140577;
  const double kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu);
  SolverOptions opt;
  opt.n_gauss = 12;
  opt.far_tol = 1e-10;
  BcModel model;
  model.mat = mat;
  model.opt = opt;
  model.mesh.build(loadDat(geo), model.popt);
  model.ts.build(model.mesh);
  model.hsolver = g_hsolver;
    model.setupOn(geo, {{"clamp", patchesOn(model.mesh, 0, 0.0), {true, true, true}, nullptr},
                      {"tip", patchesOn(model.mesh, 0, L), {true, true, true}, nullptr}}, {}, "");
  Rbe2 m;
  m.label = "tip master";
  m.support = 1;
  m.ref = Vec3(L, 0.0, 0.0);
  model.rbe2 = {m};
  LoadCase shear, twist;
  shear.name = "shear";
  shear.master_wrench.push_back({0, (Vec6() << 0, 0, -F, 0, 0, 0).finished()});
  twist.name = "torsion";
  twist.master_wrench.push_back({0, (Vec6() << 0, 0, 0, T, 0, 0).finished()});
  std::vector<CaseReport> rep;
  const auto sol = model.solutions({shear, twist}, &rep);
  std::printf("# %s: %zu elements, %d unknowns; operators %.2f s, LU %.2f s\n", geo.c_str(), model.mesh.elements.size(),
              model.unknowns(), model.t_assemble, model.t_factor);
  for (const auto& r : rep) printReport(r);
  const double w = rep[0].master_motion[0].second(2), th = rep[1].master_motion[0].second(3);
  const double wT = -(F * L * L * L / (3 * E * I) + F * L / (kappa * G * A)), thSV = T * L / (G * J);
  std::printf("  tip deflection %.6f vs Timoshenko %.6f (%+.3f%%); twist %.6f vs Saint-Venant %.6f (%+.3f%%)\n", w, wT,
              100 * (w / wT - 1), th, thSV, 100 * (th / thSV - 1));
  std::printf("  master reaction on the bar: F_z %.8f (applied %.1f), M_x %.8f (applied %.1f)\n",
              rep[0].reactions[1].second.F(2), -F, rep[1].reactions[1].second.M(0), T);
  // treecode against the direct evaluation (torsion case)
  std::vector<Vec3> X;
  unsigned s = 12345;
  auto rnd = [&]() { s = 1103515245u * s + 12345u; return double((s >> 8) & 0xffffff) / double(0xffffff); };
  for (int i = 0; i < 2000; ++i) X.emplace_back(0.2 + 9.6 * rnd(), -0.45 + 0.9 * rnd(), -0.45 + 0.9 * rnd());
  double ta = now();
  TreeEvaluator tree(sol[1], opt);
  const auto Ft = tree.eval(X);
  const double t_tree = now() - ta;
  InteriorEvaluator ev(sol[1], opt);
  std::vector<Vec3> Y(X.begin(), X.begin() + 200);
  ta = now();
  const auto Fd = ev.eval(Y);
  const double t_dir = now() - ta;
  double emax = 0.0, smax = 0.0;
  for (size_t i = 0; i < Y.size(); ++i) {
    emax = std::max(emax, (Ft[i].stress - Fd[i].stress).norm());
    smax = std::max(smax, Fd[i].stress.norm());
  }
  std::printf("  treecode: %zu points in %.2f s (setup %.2f s), direct %.2f ms/point; max error %.3e of max |sigma| %.3e\n",
              X.size(), t_tree, tree.t_setup, 1e3 * t_dir / Y.size(), emax / smax, smax);
  std::printf("# end to end %.2f s\n", now() - t0);
  check("rbe2 tip deflection vs Timoshenko", w / wT - 1, 0.0, 6e-3);
  check("rbe2 twist vs Saint-Venant", th / thSV - 1, 0.0, 6e-3);
  check("rbe2 master equilibrium F_z", rep[0].reactions[1].second.F(2), -F, 1e-6);
  check("rbe2 master equilibrium M_x", rep[1].reactions[1].second.M(0), T, 1e-6);
  check("rbe2 torsion equilibrium residual", rep[1].residual(L), 0.0, 1e-6);
  check("treecode vs direct, max |dsigma| / max |sigma|", emax / smax, 0.0, 1e-6);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: bc_validate halfspace|bar|cantilever|rbe2 <geometry.dat> [--csv f] [--probe f] [--cache d]\n";
    return 1;
  }
  const std::string what = argv[1], geo = argv[2];
  std::string csv, probe, cache = "cache";
  for (int i = 3; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--probe")) probe = argv[++i];
    else if (!std::strcmp(argv[i], "--check")) g_check = true;
    else if (!std::strcmp(argv[i], "--hmat")) {  // hierarchical backend at tolerance eps
      g_hsolver.on = true;
      g_hsolver.o.full_g = true;  // the exact cases carry smooth traction data
      g_hsolver.o.h.eps = std::atof(argv[++i]);
      g_hsolver.o.h.tol = 1e-10;  // entry quadrature as the dense far_tol
    } else if (!std::strcmp(argv[i], "--hmat-tol")) g_hsolver.o.h.tol = std::atof(argv[++i]);  // entry quadrature
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
  }
  int rc = -1;
  if (what == "halfspace") rc = halfspace(geo, csv, probe, cache);
  else if (what == "bar") rc = bar(geo, csv);
  else if (what == "cantilever") rc = cantilever(geo, csv, probe);
  else if (what == "rbe2") rc = rbe2Tree(geo);
  if (rc >= 0) {
    if (g_check) std::printf("# %d check(s) failed\n", g_fail);
    return rc ? rc : (g_fail ? 2 : 0);
  }
  std::cerr << "unknown case " << what << "\n";
  return 1;
}
