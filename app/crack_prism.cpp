// Prism with an edge crack, pure-mode loading (Rocha, Trevelyan & Leonel 2024, Sec. 4.3, Table 3, Fig. 21).
//
//   crack_prism geo/crack/prism_k2.dat [--out prefix] [--check-k tol] [--nu NU] [--cond] [--factored] [--a3 A]
//
// [-1, 1]^2 x [0, 2], crack y = 0, -1 <= x <= 0, straight front x = 0 along z (python/make_crack.py prism). The
// single-term Williams field with K_M = 1 (Eq. 29; plane strain for I and II, antiplane for III) is an exact
// elasticity solution in the prism: its displacement is prescribed on x = +-1 and y = +-1, its traction on z = 0, 2
// (sigma_tt = nu (sigma_nn + sigma_bb) for I and II, zero at nu = 0), the crack faces are traction free. Every patch
// has dofs of its own (paper: no continuity between patches); the z faces carry the front-end enrichment, so they
// need no alignment with the crack. The three modes are three right-hand sides of one assembled operator.
//
// Output: prefix_sif.csv (front samples, z and K_I..K_III for each mode), one summary line (prefix_summary.json):
// avg |K_M| per mode (the paper's Table 3), mean |K_M - delta_MN|, and the relative L2 error of the displacement
// over the unknown-displacement surfaces (z faces and crack faces; the paper's Fig. 21 measure).
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
    std::fprintf(stderr, "usage: crack_prism geometry.dat [--out prefix] [--check-k tol] [--nu NU] [--cond]\n");
    return 2;
  }
  std::string dat = argv[1], out;
  bool cond = false;
  bool factored = false;  // --factored: factor() + solveFactored(), as crack_serve runs
  int ng = 12, grade = 6;  // 8 Gauss points leave a 1e-4 floor in K (z-face elements next to the front end)
  double shift = 0.02, far_tol = 1e-12, check_k = -1, E = 1.0, nu = 0.0, a3 = 0.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--cond")) cond = true;
    else if (!std::strcmp(argv[i], "--factored")) factored = true;
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--check-k") && i + 1 < argc) check_k = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--nu") && i + 1 < argc) nu = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--ng") && i + 1 < argc) ng = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--grade") && i + 1 < argc) grade = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shift") && i + 1 < argc) shift = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--a3") && i + 1 < argc) a3 = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--far-tol") && i + 1 < argc) far_tol = std::atof(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  const std::string meta = dat.substr(0, dat.size() - 4) + ".crack.json";
  Material mat;
  mat.mu = E / (2 * (1 + nu));
  mat.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  const double kappa = 3 - 4 * nu;

  CrackModel M;
  CrackOptions o;
  o.discontinuous = true;
  o.quad.n_gauss = ng;
  o.quad.far_tol = far_tol;
  o.grade = grade;
  o.end_shift = shift;
  {
    const Json J = Json::load(meta);
    for (size_t i = 0; i < J["dirichlet"].size(); ++i) o.dirichlet.push_back(J["dirichlet"][i].asInt());
  }
  M.setup(dat, meta, mat, o);
  auto vec = [&](const char* k) {
    const Json& a = M.meta["front"][k];
    return Vec3(a[0].asNumber(), a[1].asNumber(), a[2].asNumber());
  };
  const Vec3 F0 = vec("point"), fn = vec("n"), fb = vec("b"), ft = vec("t");
  auto polar = [&](const Vec3& y, double& r, double& th) {
    const Vec3 d = y - F0;
    r = std::hypot(d.dot(fn), d.dot(fb));
    th = std::atan2(d.dot(fb), d.dot(fn));
  };
  // exact displacement of mode M (the crack faces' sides by patch: theta = +-pi there)
  auto uExact = [&](int Mo, const Vec3& y, int p) {
    double r, th;
    polar(y, r, th);
    if (M.face[p] != 0) th = M.face[p] * M_PI;
    Vec3 w[3];
    Front::williams(r, th, mat.mu, kappa, fn, fb, ft, w);
    if (Mo == 0 && a3 != 0.0) {  // Williams n = 3: u = a3 r^h / (2 mu) [(k + h - 1) cos(h th) - h cos((h - 2) th), ...]
      const double h = 1.5, c = a3 * std::pow(r, h) / (2 * mat.mu);
      w[0] += c * ((kappa + h - 1) * std::cos(h * th) - h * std::cos((h - 2) * th)) * fn +
              c * ((kappa - h + 1) * std::sin(h * th) + h * std::sin((h - 2) * th)) * fb;
    }
    return w[Mo];
  };
  // exact stress of mode M in the (n, b, t) frame, plane strain sigma_tt = nu (sigma_nn + sigma_bb)
  auto sExact = [&](int Mo, const Vec3& y) {
    double r, th;
    polar(y, r, th);
    const double f = 1.0 / std::sqrt(2 * M_PI * r), c2 = std::cos(th / 2), s2 = std::sin(th / 2),
                 c32 = std::cos(1.5 * th), s32 = std::sin(1.5 * th);
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();  // local (n, b, t)
    if (Mo == 0) {
      S(0, 0) = c2 * (1 - s2 * s32);
      S(1, 1) = c2 * (1 + s2 * s32);
      S(0, 1) = S(1, 0) = s2 * c2 * c32;
    } else if (Mo == 1) {
      S(0, 0) = -s2 * (2 + c2 * c32);
      S(1, 1) = s2 * c2 * c32;
      S(0, 1) = S(1, 0) = c2 * (1 - s2 * s32);
    } else {
      S(0, 2) = S(2, 0) = -s2;
      S(1, 2) = S(2, 1) = c2;
    }
    if (Mo == 0 && a3 != 0.0) {  // Williams n = 3 (h = 3/2), stresses a3 h r^(h-1) [...], in units of f
      const double h = 1.5, g = a3 * h * std::sqrt(r) / f;
      S(0, 0) += g * ((2 + h - 1) * std::cos((h - 1) * th) - (h - 1) * std::cos((h - 3) * th));
      S(1, 1) += g * ((2 - h + 1) * std::cos((h - 1) * th) + (h - 1) * std::cos((h - 3) * th));
      const double sxy = g * (-(h - 1) * std::sin((h - 1) * th) + (h - 1) * std::sin((h - 3) * th));
      S(0, 1) += sxy;
      S(1, 0) += sxy;
    }
    S(2, 2) += nu * (S(0, 0) + S(1, 1));
    Eigen::Matrix3d Q;
    Q << fn, fb, ft;
    return Eigen::Matrix3d(f * Q * S * Q.transpose());
  };
  const int no = M.n_outer;
  std::vector<TractionFn> loads;
  std::vector<DisplacementFn> disps;
  for (int Mo = 0; Mo < 3; ++Mo) {
    loads.push_back([&, Mo](const Vec3& y, const Vec3& n, int p) -> Vec3 {
      if (p >= no) return Vec3::Zero();  // traction-free crack faces
      return sExact(Mo, y) * n;
    });
    disps.push_back([&, Mo](const Vec3& y, int p) { return uExact(Mo, y, p); });
  }
  std::vector<Eigen::VectorXd> X;
  if (factored) {  // the crack_serve path: factor once, then right-hand sides alone
    M.factor(cond);
    X = M.solveFactored(loads, disps);
  } else {
    X = M.solve(loads, cond, disps);
  }

  // SIFs along the front
  FILE* fs = out.empty() ? nullptr : std::fopen((out + "_sif.csv").c_str(), "w");
  if (fs) std::fprintf(fs, "mode,z,KI,KII,KIII\n");
  double avg[3][3] = {}, err[3][3] = {}, emax = 0;
  int ns = 0;
  const FrontPiece& F = M.front.pieces[0];
  const int nsm = 80;
  for (int i = 0; i < nsm; ++i) {
    const double u = F.u0 + (F.u1 - F.u0) * (i + 0.5) / nsm;
    Vec3 C, Cu;
    int n, s[16];
    double Rb[16];
    F.eval(u, C, Cu, n, s, Rb);
    for (int Mo = 0; Mo < 3; ++Mo) {
      const Vec3 K = M.front.sif(0, u, X[Mo]);
      if (fs) std::fprintf(fs, "%d,%.12g,%.12g,%.12g,%.12g\n", Mo + 1, C.dot(ft), K(0), K(1), K(2));
      for (int N = 0; N < 3; ++N) {
        avg[Mo][N] += std::abs(K(N));
        const double e = std::abs(K(N) - (N == Mo ? 1.0 : 0.0));
        err[Mo][N] += e;
        emax = std::max(emax, e);
      }
    }
    ++ns;
  }
  if (fs) std::fclose(fs);
  for (auto& a : avg) for (double& v : a) v /= ns;
  for (auto& a : err) for (double& v : a) v /= ns;

  // relative L2 error of the displacement on the z faces and the crack faces, per mode
  double l2[3];
  {
    const GaussRule g = gaussLegendre(4);
    double num[3] = {}, den[3] = {};
    for (const Element& e : M.mesh.elements) {
      if (e.patch < no && M.dirichlet[e.patch]) continue;
      for (const Rect& Rc : e.cells)
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) {
            const Vec2 uv(Rc.u0 + Rc.du() * g.x[i], Rc.v0 + Rc.dv() * g.x[j]);
            PatchEval Ev;
            M.mesh.patches[e.patch].evalSpan(e.su, e.sv, uv(0), uv(1), Ev);
            const double w = g.w[i] * g.w[j] * Rc.du() * Rc.dv() * Ev.normal().norm();
            for (int Mo = 0; Mo < 3; ++Mo) {
              const Vec3 ue = uExact(Mo, Ev.x, e.patch);
              num[Mo] += w * (M.displacement(e.patch, uv, X[Mo]) - ue).squaredNorm();
              den[Mo] += w * ue.squaredNorm();
            }
          }
    }
    for (int Mo = 0; Mo < 3; ++Mo) l2[Mo] = std::sqrt(num[Mo] / den[Mo]);
  }

  char line[4096];
  int c = std::snprintf(line, sizeof line, "{\"geometry\": \"%s\", \"unknowns\": %ld", dat.c_str(), M.unknowns);
  const char* nm[3] = {"I", "II", "III"};
  for (int Mo = 0; Mo < 3; ++Mo) {
    c += std::snprintf(line + c, sizeof line - c, ", \"mode%s\": {\"avg_abs\": [%.6e, %.6e, %.6e], "
                       "\"avg_err\": [%.6e, %.6e, %.6e], \"u_L2\": %.6e}", nm[Mo], avg[Mo][0], avg[Mo][1],
                       avg[Mo][2], err[Mo][0], err[Mo][1], err[Mo][2], l2[Mo]);
  }
  double cross = 0;
  for (int Mo = 0; Mo < 3; ++Mo)
    for (int N = 0; N < 3; ++N) if (N != Mo) cross = std::max(cross, avg[Mo][N]);
  c += std::snprintf(line + c, sizeof line - c, ", \"KI_err\": %.6e, \"KII_err\": %.6e, \"KIII_err\": %.6e, "
                     "\"cross_max\": %.6e", err[0][0], err[1][1], err[2][2], cross);
  std::snprintf(line + c, sizeof line - c, ", \"K_max_err\": %.6e, \"cond\": %.6e, \"residual\": %.3e, "
                "\"t_assemble\": %.3f, \"t_solve\": %.3f}", emax, M.condition, M.residual, M.t_assemble, M.t_solve);
  std::printf("%s\n", line);
  if (!out.empty()) std::ofstream(out + "_summary.json") << line << "\n";
  if (check_k > 0) {
    double worst = 0;
    for (int Mo = 0; Mo < 3; ++Mo) worst = std::max(worst, err[Mo][Mo]);
    if (!(worst <= check_k)) {
      std::fprintf(stderr, "pure-mode SIF error %.3e above %.3e\n", worst, check_k);
      return 1;
    }
  }
  return 0;
}
