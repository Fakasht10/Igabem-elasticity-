// Validation of two-region bodies (MultiRegion on BcModel) against exact solutions,
// and the printed-coupon stiffness study.
//
// usage: region_validate patch|lame|eshelby <host.dat> <core.dat> [--contrast c] [--csv f] [--probe f]
//                        [--cache dir|none] [--check]
//        region_validate coupon <host.dat> <core.dat> [--rho 0.1,0.2,...] [--csv f] [--check]
//
//   patch    same material in both regions, u = alpha x, t = sigma n on the outer surface:
//            the coupled system must reproduce the linear field to quadrature accuracy
//            (checks the interface matching, the condensation and the signs).
//   lame     two-layer sphere (shell 0.5 < r < 1, core r < 0.5) under external pressure:
//            exact radial field (src/problems/two_region.hpp); core E = c E_shell.
//   eshelby  elastic sphere r < 0.5 in a matrix sphere r < 1 under the exact traction of
//            Goodier's infinite-medium solution for remote tension along z: exact
//            non-radial field, uniform stress in the inclusion (Eshelby).
//   coupon   printed coupon [0,4] x [0,1]^2 with solid walls around a homogenized infill
//            core, ends pulled by a prescribed u_x (rollers): effective modulus against the
//            infill density, the solid and hollow limits, and the Voigt / Reuss /
//            series-parallel estimates.
//   --check  regression mode (ctest): compare the key numbers with the committed values.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "bc_model.hpp"
#include "multi_region.hpp"
#include "problems/two_region.hpp"

using namespace igabem;

namespace {

bool g_check = false;
int g_fail = 0;

void check(const char* name, double got, double want, double tol) {
  if (!g_check) return;
  const bool ok = std::abs(got - want) <= tol;
  std::printf("  check %-38s %s  got %.8e  want %.8e  tol %.1e\n", name, ok ? "ok  " : "FAIL", got, want, tol);
  if (!ok) ++g_fail;
}

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

SolverOptions options() {
  SolverOptions o;
  o.n_gauss = 12;
  o.far_tol = 1e-10;
  return o;
}

// Relative L2(Gamma) error of the boundary traction against sigma(x) n.
template <class Fs>
double tractionL2Error(const BoundarySolution& s, Fs&& sigma, int ng = 8) {
  const GaussRule g = gaussLegendre(ng);
  double num = 0.0, den = 0.0;
  PatchEval E;
  for (const Element& e : s.mesh->elements)
    for (const Rect& R : e.cells)
      for (int a = 0; a < ng; ++a)
        for (int b = 0; b < ng; ++b) {
          s.mesh->patches[e.patch].evalSpan(e.su, e.sv, R.u0 + R.du() * g.x[a], R.v0 + R.dv() * g.x[b], E);
          const Vec3 n = (E.normal() * s.mesh->normal_sign).normalized();
          const Vec3 te = sigma(E.x) * n;
          const double w = g.w[a] * g.w[b] * R.du() * R.dv() * E.normal().norm();
          num += w * (s.traction(e.patch, E) - te).squaredNorm();
          den += w * te.squaredNorm();
        }
  return std::sqrt(num / den);
}

struct Composite {
  BcModel model;
  MultiRegion mr;
  double t_setup = 0.0;

  void build(const std::string& host, const std::string& core, const Material& m1, const Material& m2,
             const std::string& cache, std::vector<FixedSupport> sup = {}) {
    const double t0 = now();
    model.mat = m1;
    model.opt = options();
    model.mesh.build(loadDat(host), model.popt);
    model.ts.build(model.mesh);
    if (!core.empty()) {
      mr.add(model.mesh, model.ts, "core", core, m2, model.opt, model.popt, cache);
      model.regions = &mr;
    }
    model.setupOn(host, std::move(sup), {}, cache);
    t_setup = now() - t0;
  }
  void print(const char* geo) const {
    const Inclusion* I = mr.empty() ? nullptr : &mr.inc[0];
    std::printf("# %s: host %zu elements, %d unknowns", geo, model.mesh.elements.size(), model.unknowns());
    if (I)
      std::printf("; core %zu elements, %d traction unknowns, rcond(G_core) %.2e, |S R|/|S||R| %.1e", I->mesh.elements.size(),
                  3 * I->ts.ndof, I->rcond_G, I->rigid_residual);
    std::printf("\n#   free rigid motions %d; host rows %.2f s%s", model.freeModes(), model.t_assemble,
                model.from_cache ? " (cached)" : "");
    if (I)
      std::printf(", core rows %.2f s%s, core LU + S %.2f s, condensation %.2f s", I->t_assemble,
                  I->from_cache ? " (cached)" : "", I->t_factor, mr.t_condense);
    std::printf(", host LU %.2f s; setup %.2f s\n", model.t_factor, t_setup);
  }
};

Vec3 pointOn(const BoundarySolution& s, const Vec3& x, Eigen::Matrix3d& sig) {
  const SurfacePoint sp = locate(*s.mesh, x, [](int) { return true; });
  sig = s.stress(sp);
  return sp.E.x;
}

template <class P>
int sphere(const std::string& what, const std::string& host, const std::string& core, double contrast,
           const std::string& csv, const std::string& probe, const std::string& cache) {
  const double t0 = now();
  P pb;
  if (what == "patch") {
    pb.mat1 = materialEnu(1.0, 0.3);
    pb.mat2 = pb.mat1;
  } else {
    pb.mat1 = materialEnu(1.0, 0.35);              // printed wall (PLA-like nu)
    pb.mat2 = materialEnu(contrast, 0.3);         // homogenized infill
  }
  pb.init();
  const Eigen::Matrix3d alpha = Eigen::Vector3d(-0.3, -0.3, 1.0).asDiagonal();
  const bool patch = what == "patch";
  auto exactU = [&](int reg, const Vec3& x) {
    if (patch) return Vec3(alpha * x);
    Vec3 u;
    Eigen::Matrix3d G;
    pb.field(reg, x, u, G);
    return u;
  };
  auto exactS = [&](int reg, const Vec3& x) {
    if (patch) return Eigen::Matrix3d(pb.mat1.stress(alpha));
    Vec3 u;
    Eigen::Matrix3d G;
    pb.field(reg, x, u, G);
    return Eigen::Matrix3d(pb.mat(reg).stress(G));
  };

  Composite C;
  C.build(host, core, pb.mat1, pb.mat2, cache);
  LoadCase lc;
  lc.name = what;
  lc.smooth = [&](const Vec3& y, const Vec3& n) { return Vec3(exactS(0, y) * n); };
  std::vector<CaseReport> rep;
  const auto sol = C.model.solutions({lc}, &rep);
  const BoundarySolution cs = C.mr.inclusionSolution(0, sol[0]);
  C.print(host.c_str());

  const double eu0 = displacementL2Error(C.model.mesh, sol[0].u, [&](const Vec3& x) { return exactU(0, x); });
  const double eu1 = displacementL2Error(*cs.mesh, cs.u, [&](const Vec3& x) { return exactU(1, x); });
  const double et1 = tractionL2Error(cs, [&](const Vec3& x) { return exactS(1, x); });
  const auto [Fc, Mc] = C.mr.inclusionResultant(cs);
  std::printf("  u error (L2, rigid removed): shell %.3e, core %.3e; interface traction error %.3e; "
              "host resultant |F| %.1e |M| %.1e; core resultant |F| %.1e |M| %.1e\n",
              eu0, eu1, et1, rep[0].total.F.norm(), rep[0].total.M.norm(), Fc.norm(), Mc.norm());

  // stresses at the interface (both sides) and on the outer surface
  Eigen::Matrix3d s0, s1, sb;
  pointOn(cs, Vec3(pb.a, 0, 0), s1);
  const SurfacePoint hp = locate(C.model.mesh, Vec3(pb.a, 0, 0), [&](int p) { return C.mr.onInterface(p); });
  s0 = sol[0].stress(hp);
  const SurfacePoint bp = locate(C.model.mesh, Vec3(0, 0, pb.b), [&](int p) { return !C.mr.onInterface(p); });
  sb = sol[0].stress(bp);
  const Eigen::Matrix3d e0 = exactS(0, Vec3(pb.a, 0, 0)), e1 = exactS(1, Vec3(pb.a, 0, 0)),
                        eb = exactS(0, Vec3(0, 0, pb.b));
  const double scale = std::max(e0.norm(), e1.norm());
  std::printf("  interface (a,0,0), shell side: s_xx %.6f (exact %.6f) s_zz %.6f (exact %.6f)\n", s0(0, 0), e0(0, 0),
              s0(2, 2), e0(2, 2));
  std::printf("  interface (a,0,0), core side:  s_xx %.6f (exact %.6f) s_zz %.6f (exact %.6f)\n", s1(0, 0), e1(0, 0),
              s1(2, 2), e1(2, 2));
  const double es0 = (s0 - e0).norm() / scale, es1 = (s1 - e1).norm() / scale, esb = (sb - eb).norm() / scale;
  std::printf("  stress errors / |sigma|: shell side %.3e, core side %.3e, outer pole %.3e\n", es0, es1, esb);

  // interior line probes through both regions (x axis and z axis)
  double emax_in = 0.0;
  {
    MultiRegionEvaluator ev(C.mr, sol[0], C.model.opt);
    std::vector<Vec3> xs;
    const int n = 40;
    for (int ax : {0, 2})
      for (int i = 1; i < n; ++i) xs.push_back(Vec3::Unit(ax) * (pb.b * i / n + 1e-3 * (i == n / 2)));
    const auto f = ev.eval(xs);
    std::ofstream pr;
    if (!probe.empty()) {
      pr.open(probe);
      pr.precision(12);
      pr << "axis,s,region,sxx,syy,szz,sxx_exact,syy_exact,szz_exact,ux,uz,ux_exact,uz_exact\n";
    }
    for (size_t i = 0; i < xs.size(); ++i) {
      const int reg = xs[i].norm() < pb.a ? 1 : 0;
      if (f[i].region != reg) std::printf("  probe %zu: region %d, expected %d\n", i, f[i].region, reg);
      const Eigen::Matrix3d se = exactS(reg, xs[i]);
      const Vec3 ue = exactU(reg, xs[i]);
      emax_in = std::max(emax_in, (f[i].f.stress - se).norm() / scale);
      if (pr)
        pr << (i < xs.size() / 2 ? "x" : "z") << ',' << xs[i].norm() << ',' << f[i].region << ',' << f[i].f.stress(0, 0)
           << ',' << f[i].f.stress(1, 1) << ',' << f[i].f.stress(2, 2) << ',' << se(0, 0) << ',' << se(1, 1) << ','
           << se(2, 2) << ',' << f[i].f.u(0) << ',' << f[i].f.u(2) << ',' << ue(0) << ',' << ue(2) << '\n';
    }
    std::printf("  interior stress, max error / |sigma| on the x and z axes: %.3e\n", emax_in);
  }
  const double t_end = now() - t0;
  std::printf("# end to end %.2f s\n", t_end);
  if (!csv.empty()) {
    std::ifstream in(csv);
    const bool fresh = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    std::ofstream o(csv, std::ios::app);
    o.precision(10);
    if (fresh)
      o << "case,geometry,contrast,host_unknowns,core_unknowns,elements,eu_shell,eu_core,et_interface,es_shell,"
           "es_core,es_outer,es_interior,szz_matrix,szz_matrix_exact,szz_core,szz_core_exact,equilibrium,"
           "t_setup,t_total\n";
    o << what << ',' << host << ',' << contrast << ',' << C.model.unknowns() << ',' << 3 * C.mr.inc[0].ts.ndof << ','
      << C.model.mesh.elements.size() << ',' << eu0 << ',' << eu1 << ',' << et1 << ',' << es0 << ',' << es1 << ','
      << esb << ',' << emax_in << ',' << s0(2, 2) << ',' << e0(2, 2) << ',' << s1(2, 2) << ',' << e1(2, 2) << ','
      << std::max(rep[0].total.F.norm(), rep[0].total.M.norm()) << ',' << C.t_setup << ',' << t_end << '\n';
  }
  const bool r1 = host.find("_r1_") != std::string::npos;
  if (patch) {
    check("patch shell u error", eu0, 0.0, 1e-7);
    check("patch core u error", eu1, 0.0, 1e-7);
    check("patch interface traction error", et1, 0.0, 1e-8);
    check("patch interior stress error", emax_in, 0.0, 1e-7);
  } else if (r1 && what == "lame" && std::abs(contrast - 0.1) < 1e-12) {  // exact in the space
    check("lame r1 shell u error", eu0, 0.0, 1e-8);
    check("lame r1 core u error", eu1, 0.0, 1e-8);
    check("lame r1 interface traction error", et1, 0.0, 1e-8);
    check("lame r1 interior stress error", emax_in, 0.0, 1e-7);
  } else if (r1 && what == "eshelby" && std::abs(contrast - 0.1) < 1e-12) {  // results/regions/spheres.csv
    check("eshelby r1 shell u error", eu0, 4.396421899e-4, 2e-7);
    check("eshelby r1 core u error", eu1, 2.562106418e-4, 2e-7);
    check("eshelby r1 interface traction error", et1, 2.483304651e-4, 2e-7);
    check("eshelby r1 matrix sigma_zz(a,0,0)", s0(2, 2), 1.875790966, 1e-6);
    check("eshelby r1 core sigma_zz(a,0,0)", s1(2, 2), 0.1875282364, 1e-7);
  }
  return 0;
}

std::vector<double> parseList(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
  return v;
}

std::vector<int> patchesOn(const Mesh& m, int axis, double v, double tol = 1e-9) {
  std::vector<int> out;
  for (size_t p = 0; p < m.patches.size(); ++p) {
    bool on = true;
    for (int k = 0; k < m.patches[p].nu * m.patches[p].nv; ++k) on = on && std::abs(m.patches[p].cp(k)(axis) - v) < tol;
    if (on) out.push_back(int(p));
  }
  return out;
}

int coupon(const std::string& host, const std::string& core, const std::vector<double>& rhos, const std::string& csv,
           const std::string& cache) {
  const double t0 = now(), L = 4.0, W = 1.0, H = 1.0, w = 0.15, delta = 0.01 * L;
  const Material wall = materialEnu(1.0, 0.35);  // E_s = 1: moduli below are E / E_s
  std::ofstream o;
  if (!csv.empty()) {
    o.open(csv);
    o.precision(10);
    o << "rho,E_core,E_eff,E_voigt,E_reuss,E_series_parallel,unknowns,equilibrium,max_vm_wall,t_case\n";
  }
  const double A = W * H, Ac = (W - 2 * w) * (H - 2 * w), Vc = Ac * (L - 2 * w), V = A * L, phi = Vc / V;
  std::printf("# coupon %s: wall w = %.2f, core volume fraction %.4f\n", host.c_str(), w, phi);
  std::printf("  %-6s %-9s %-9s %-9s %-9s %-9s %-9s %s\n", "rho", "E_core", "E_eff", "Voigt", "Reuss", "ser-par",
              "vM_max", "time");
  double Eeff1 = 0.0, Ehollow = 0.0, Erho02 = 0.0;
  // One host mesh, one inclusion: a density sweep at fixed nu_core only rescales the
  // core's Dirichlet-to-Neumann map; the host rows come from the cache, and the
  // condensation and the LU are redone per density.
  BcModel model;
  model.mat = wall;
  model.opt = options();
  model.mesh.build(loadDat(host), model.popt);
  model.ts.build(model.mesh);
  const std::vector<FixedSupport> sup = {
      {"x0", patchesOn(model.mesh, 0, 0.0), {true, false, false}, nullptr},
      {"xL", patchesOn(model.mesh, 0, L), {true, false, false}, [=](const Vec3&) { return Vec3(delta, 0, 0); }}};
  MultiRegion mr;
  for (double rho : rhos) {
    const double tc = now();
    // rho = 0: hollow (no core), rho = 1: solid (the core is the wall material, nu included)
    const Material mc = rho >= 1.0 ? wall : infillMaterial(wall, rho);
    if (rho > 0.0) {
      if (!mr.empty() && std::abs(mr.inc[0].mat.nu() - mc.nu()) < 1e-12) {
        mr.scaleStiffness(0, mc);
      } else {
        mr = MultiRegion();
        mr.add(model.mesh, model.ts, "core", core, mc, model.opt, model.popt, cache);
      }
    }
    model.regions = rho > 0.0 ? &mr : nullptr;
    mr.t_condense = 0.0;
    model.setupOn(host, sup, {}, cache);
    LoadCase lc;
    lc.name = "pull";
    std::vector<CaseReport> rep;
    const auto sol = model.solutions({lc}, &rep);
    const double F = rep[0].reactions[1].second.F(0);
    const double Eeff = F * L / (delta * A);
    const double Ec = rho > 0.0 ? youngs(mc) : 0.0;
    const double Ev = (1 - phi) * 1.0 + phi * Ec;
    const double Er = Ec > 0.0 ? 1.0 / ((1 - phi) / 1.0 + phi / Ec) : 0.0;
    const double Epar = ((A - Ac) * 1.0 + Ac * Ec) / A;
    const double Esp = 1.0 / ((2 * w / L) / 1.0 + ((L - 2 * w) / L) / Epar);
    // max von Mises on the outer wall surface (corner sampling of each element)
    double vm = 0.0;
    PatchEval E;
    for (const Element& e : model.mesh.elements) {
      if (mr.onInterface(e.patch) && rho > 0.0) continue;
      for (int a = 0; a <= 4; ++a)
        for (int b = 0; b <= 4; ++b) {
          SurfacePoint sp;
          sp.patch = e.patch;
          sp.su = e.su;
          sp.sv = e.sv;
          sp.xi = Vec2(e.rect.u0 + e.rect.du() * (0.05 + 0.9 * a / 4), e.rect.v0 + e.rect.dv() * (0.05 + 0.9 * b / 4));
          model.mesh.patches[e.patch].evalSpan(e.su, e.sv, sp.xi(0), sp.xi(1), sp.E);
          sp.N = (sp.E.normal() * model.mesh.normal_sign).normalized();
          vm = std::max(vm, vonMises(sol[0].stress(sp)));
        }
    }
    const double t_case = now() - tc;
    if (rho >= 1.0) Eeff1 = Eeff;
    if (rho == 0.0) Ehollow = Eeff;
    if (std::abs(rho - 0.2) < 1e-12) Erho02 = Eeff;
    if (std::getenv("IGABEM_VERBOSE"))
      std::printf("#   setup: host LU %.2f s, condensation %.2f s, core %s\n", model.t_factor, mr.t_condense,
                  rho > 0.0 ? (mr.inc[0].from_cache ? "rows cached" : "rows assembled") : "none");
    std::printf("  %-6.2f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %.1f s (%d unknowns, equilibrium %.1e)\n", rho, Ec,
                Eeff, Ev, Er, Esp, vm / (delta / L), t_case, model.unknowns(), rep[0].residual(L));
    if (o)
      o << rho << ',' << Ec << ',' << Eeff << ',' << Ev << ',' << Er << ',' << Esp << ',' << model.unknowns() << ','
        << rep[0].residual(L) << ',' << vm / (delta / L) << ',' << t_case << '\n';
  }
  std::printf("# end to end %.2f s\n", now() - t0);
  if (Eeff1 > 0.0) check("coupon solid limit E_eff / E_s", Eeff1, 1.0, 1e-6);
  if (host.find("_n2_") != std::string::npos) {  // results/regions/coupon_n2.csv
    check("coupon n2 hollow E_eff / E_s", Ehollow, 0.519943627, 1e-6);
    check("coupon n2 rho = 0.2 E_eff / E_s", Erho02, 0.540600904, 1e-6);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: region_validate patch|lame|eshelby|coupon <host.dat> <core.dat> [--contrast c] [--rho list]"
                 " [--csv f] [--probe f] [--cache d|none] [--check]\n";
    return 1;
  }
  const std::string what = argv[1], host = argv[2], core = argv[3];
  std::string csv, probe, cache = "cache";
  double contrast = 0.1;
  std::vector<double> rhos = {0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0};
  for (int i = 4; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--csv")) csv = argv[++i];
    else if (!std::strcmp(argv[i], "--probe")) probe = argv[++i];
    else if (!std::strcmp(argv[i], "--contrast")) contrast = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--rho")) rhos = parseList(argv[++i]);
    else if (!std::strcmp(argv[i], "--check")) g_check = true;
    else if (!std::strcmp(argv[i], "--cache")) { cache = argv[++i]; if (cache == "none") cache.clear(); }
  }
  int rc = -1;
  if (what == "patch" || what == "lame") rc = sphere<LameSphere2>(what, host, core, contrast, csv, probe, cache);
  else if (what == "eshelby") rc = sphere<EshelbySphere>(what, host, core, contrast, csv, probe, cache);
  else if (what == "coupon") rc = coupon(host, core, rhos, csv, cache);
  if (rc >= 0) {
    if (g_check) std::printf("# %d check(s) failed\n", g_fail);
    return rc ? rc : (g_fail ? 2 : 0);
  }
  std::cerr << "unknown case " << what << "\n";
  return 1;
}
