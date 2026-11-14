// Penny-shaped crack in a large cube (Rocha, Trevelyan & Leonel 2024, Sec. 4.1): dual IgBEM, with or without the
// crack-front enrichment (XIGABEM), SIFs straight from the solution vector.
//
//   crack_case geo/crack/penny_i0_n5.dat [--plain] [--load far|pressure] [--out prefix] [--cond]
//                                          [--check-k tol]
//
// Loads: far   sigma0 = 1 on the top face (outward +e2), bottom face clamped (the paper's set-up; nu = 0 makes the
//              uncracked field exactly uniaxial);
//        pressure  p = 1 on both crack faces, bottom clamped, no far load (same K_I by superposition).
// Reference (Tada; paper Eqs. 25-26), radius R, plane inclined by theta_i about e3, load sigma0 along e2:
//   K_I = 2/pi sigma0 cos^2(theta_i) sqrt(pi R), K_II = 4/(pi (2 - nu)) sigma0 sin cos cos(theta_c) sqrt(pi R),
//   K_III = -4 (1 - nu)/(pi (2 - nu)) sigma0 sin cos sin(theta_c) sqrt(pi R);  opening (8 (1 - nu^2)/(pi E)) sigma_nn
//   sqrt(R^2 - rho^2) (only the in-plane case is checked here).
// Output: prefix_sif.csv (front samples: angle, K_I..K_III computed and exact), prefix_cod.csv (opening along the
// e1 radius), one summary line on stdout (and prefix_summary.json).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "crack/dual_model.hpp"

using namespace igabem;
using namespace igabem::crack;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: crack_case geometry.dat [--plain] [--load far|pressure] [--out prefix] [--cond] [--check-k tol]\n");
    return 2;
  }
  std::string dat = argv[1], load = "far", out;
  bool enrich = true, cond = false;
  int ng = 8, grade = 6;
  double shift = 0.1, far_tol = 1e-12;
  double check_k = -1, E = 1.0, nu = 0.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--plain")) enrich = false;
    else if (!std::strcmp(argv[i], "--cond")) cond = true;
    else if (!std::strcmp(argv[i], "--load") && i + 1 < argc) load = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--check-k") && i + 1 < argc) check_k = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--nu") && i + 1 < argc) nu = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--ng") && i + 1 < argc) ng = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--grade") && i + 1 < argc) grade = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shift") && i + 1 < argc) shift = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol") && i + 1 < argc) far_tol = std::atof(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  const std::string meta = dat.substr(0, dat.size() - 4) + ".crack.json";
  Material mat;
  mat.mu = E / (2 * (1 + nu));
  mat.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

  CrackModel M;
  CrackOptions o;
  o.enrich = enrich;
  o.quad.n_gauss = ng;
  o.grade = grade;
  o.end_shift = shift;
  o.quad.far_tol = far_tol;
  // outer patches: bottom (outward -e2) clamped
  {
    std::vector<Patch> P = loadDat(dat);
    const Json J = Json::load(meta);
    for (int p = 0; p < J["outer"][1].asInt(); ++p) {
      PatchEval Ev;
      P[p].eval(0.5, 0.5, Ev);
      if (Ev.normal().normalized().dot(Vec3::UnitY()) < -0.99) o.dirichlet.push_back(p);
    }
  }
  M.setup(dat, meta, mat, o);
  const double R = M.meta["R"].asNumber(), sig = 1.0;
  const double thi = M.meta.get("incline_deg", 0.0) * M_PI / 180.0;
  const int no = M.n_outer;
  TractionFn far = [&](const Vec3&, const Vec3& n, int p) -> Vec3 {
    if (p < no && n.dot(Vec3::UnitY()) > 0.99) return sig * Vec3::UnitY();
    return Vec3::Zero();
  };
  // pressure: sigma_nn of the far field on the faces, t = -p n with p = sigma0 cos^2 (closing the opening load)
  TractionFn pres = [&](const Vec3&, const Vec3& n, int p) -> Vec3 {
    if (p < no) return Vec3::Zero();
    const Vec3 s = sig * Vec3::UnitY() * Vec3::UnitY().dot(n);  // traction of the uniaxial far field, sign flipped
    return -s;
  };
  const std::vector<Eigen::VectorXd> X = M.solve({load == "far" ? far : pres}, cond);
  const Eigen::VectorXd& x = X[0];

  // SIFs along the front
  const double KI0 = 2.0 / M_PI * sig * std::pow(std::cos(thi), 2) * std::sqrt(M_PI * R);
  const double Ks = sig * std::sin(thi) * std::cos(thi) * std::sqrt(M_PI * R) / (M_PI * (2 - nu));
  const Vec3 e1(M.meta["plane"]["e1"][0].asNumber(), M.meta["plane"]["e1"][1].asNumber(), M.meta["plane"]["e1"][2].asNumber());
  const Vec3 e3(M.meta["plane"]["e3"][0].asNumber(), M.meta["plane"]["e3"][1].asNumber(), M.meta["plane"]["e3"][2].asNumber());
  FILE* fs = out.empty() ? nullptr : std::fopen((out + "_sif.csv").c_str(), "w");
  if (fs) std::fprintf(fs, "piece,u,theta_c,KI,KII,KIII,KI_exact,KII_exact,KIII_exact\n");
  double errI = 0, sgnI = 0, maxI = 0, absII = 0, absIII = 0, errII = 0, errIII = 0;
  int ns = 0;
  if (enrich)
    for (size_t k = 0; k < M.front.pieces.size(); ++k) {
      const FrontPiece& F = M.front.pieces[k];
      const int nsm = 40;
      for (int i = 0; i < nsm; ++i) {
        const double u = F.u0 + (F.u1 - F.u0) * (i + 0.5) / nsm;
        Vec3 C, Cu;
        int n, s[16];
        double Rb[16];
        F.eval(u, C, Cu, n, s, Rb);
        const double thc = std::atan2(C.dot(e3), C.dot(e1));
        const Vec3 K = M.front.sif(int(k), u, x);
        const double kII = 4 * Ks * std::cos(thc), kIII = -4 * (1 - nu) * Ks * std::sin(thc);
        if (fs) std::fprintf(fs, "%zu,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n", k, u, thc, K(0), K(1), K(2), KI0, kII, kIII);
        const double e = std::abs(K(0) - KI0) / KI0;
        errI += e;
        sgnI += (K(0) - KI0) / KI0;
        maxI = std::max(maxI, e);
        absII += std::abs(K(1));
        absIII += std::abs(K(2));
        if (Ks != 0) { errII += std::abs(K(1) - kII) / (4 * Ks); errIII += std::abs(K(2) - kIII) / (4 * (1 - nu) * Ks); }
        ++ns;
      }
    }
  if (fs) std::fclose(fs);
  if (ns) { errI /= ns; sgnI /= ns; absII /= ns; absIII /= ns; errII /= ns; errIII /= ns; }

  // crack opening: relative L2 over the upper face (in-plane crack only), and the profile along e1
  double cod_err = -1;
  const Vec3 m = M.m;
  auto codExact = [&](const Vec3& y) {
    const double rho2 = std::pow(y.dot(e1), 2) + std::pow(y.dot(e3), 2);
    return Vec3(m * (8 * (1 - nu * nu) / (M_PI * E) * sig * std::pow(std::cos(thi), 2) * std::sqrt(std::max(0.0, R * R - rho2))));
  };
  auto opening = [&](int p, const Vec2& uv, const Vec3& y) {
    const SurfacePoint sb = M.coincident(p, y);
    return Vec3(M.displacement(p, uv, x) - M.displacement(sb.patch, sb.xi, x));
  };
  if (thi == 0.0) {
    const GaussRule g = gaussLegendre(6);
    double num = 0, den = 0;
    for (const Element& e : M.mesh.elements) {
      if (M.face[e.patch] != 1) continue;
      for (const Rect& Rc : e.cells)
        for (int i = 0; i < 6; ++i)
          for (int j = 0; j < 6; ++j) {
            const Vec2 uv(Rc.u0 + Rc.du() * g.x[i], Rc.v0 + Rc.dv() * g.x[j]);
            PatchEval Ev;
            M.mesh.patches[e.patch].evalSpan(e.su, e.sv, uv(0), uv(1), Ev);
            const double w = g.w[i] * g.w[j] * Rc.du() * Rc.dv() * Ev.normal().norm();
            const Vec3 ex = codExact(Ev.x);
            num += w * (opening(e.patch, uv, Ev.x) - ex).squaredNorm();
            den += w * ex.squaredNorm();
          }
    }
    cod_err = std::sqrt(num / den);
  }
  if (!out.empty()) {
    FILE* fc = std::fopen((out + "_cod.csv").c_str(), "w");
    std::fprintf(fc, "rho,cod_n,cod_exact\n");
    for (int i = 0; i <= 400; ++i) {
      const double rho = R * (i == 400 ? 0.99999 : i / 400.0);
      const Vec3 y = rho * e1 + 1e-9 * e3;
      const SurfacePoint sp = locate(M.mesh, y, [&](int p) { return M.face[p] == 1; });
      const Vec3 d = opening(sp.patch, sp.xi, sp.E.x);
      std::fprintf(fc, "%.12g,%.12g,%.12g\n", rho, d.dot(m), codExact(sp.E.x).dot(m));
    }
    std::fclose(fc);
  }
  char line[2048];
  std::snprintf(line, sizeof line,
                "{\"geometry\": \"%s\", \"enrich\": %d, \"load\": \"%s\", \"unknowns\": %ld, \"KI_exact\": %.12g, "
                "\"KI_avg_err\": %.6e, \"KI_signed_err\": %.6e, \"KI_max_err\": %.6e, \"KII_avg_abs\": %.6e, \"KIII_avg_abs\": %.6e, "
                "\"KII_avg_err\": %.6e, \"KIII_avg_err\": %.6e, \"cod_L2\": %.6e, \"cond\": %.6e, \"residual\": %.3e, "
                "\"t_assemble\": %.3f, \"t_solve\": %.3f}",
                dat.c_str(), int(enrich), load.c_str(), M.unknowns, KI0, errI, sgnI, maxI, absII, absIII, errII, errIII,
                cod_err, M.condition, M.residual, M.t_assemble, M.t_solve);
  std::printf("%s\n", line);
  if (!out.empty()) { std::ofstream(out + "_summary.json") << line << "\n"; }
  if (check_k > 0 && enrich && !(errI <= check_k)) {
    std::fprintf(stderr, "K_I average error %.3e above %.3e\n", errI, check_k);
    return 1;
  }
  return 0;
}
