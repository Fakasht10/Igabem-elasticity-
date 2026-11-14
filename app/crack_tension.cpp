// Prism with an edge crack under uniform tension (Rocha, Trevelyan & Leonel 2024, Sec. 4.4, Figs. 23 and 25).
//
//   crack_tension geo/crack/prismt_split_c1.dat [--out prefix] [--kappa-outer strain|stress] [--check tol]
//
// Block 5 x 8.75 x 7.5 (python/make_crack.py prism_tension), crack a = 2.5 at mid-height, sigma0 = 1 on the top and
// bottom faces, a band of the back face clamped, E = 1000, nu = 0.3. No closed form: the reference is the 2-D
// plane-strain K_I = sigma0 sqrt(pi a) F(a/b) (paper Eq. 30), meaningful near the middle of the front (z = 3.75),
// and the paper's own Fig. 25 values. The lateral faces carry the Williams enrichment with the plane-strain Kolosov
// constant or, --kappa-outer stress, the plane-stress one (the paper's models b and c).
//
// Output: prefix_sif.csv (z, K_I..K_III normalised by sigma0 sqrt(pi a)), one summary line (prefix_summary.json).
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
    std::fprintf(stderr, "usage: crack_tension geometry.dat [--out prefix] [--kappa-outer strain|stress]\n");
    return 2;
  }
  std::string dat = argv[1], out, kout = "strain";
  int ng = 12, grade = 6;
  double E = 1000.0, nu = 0.3, sig = 1.0, check = -1, shift = 0.02;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--kappa-outer") && i + 1 < argc) kout = argv[++i];
    else if (!std::strcmp(argv[i], "--ng") && i + 1 < argc) ng = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--check") && i + 1 < argc) check = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--shift") && i + 1 < argc) shift = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--grade") && i + 1 < argc) grade = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  const std::string meta = dat.substr(0, dat.size() - 4) + ".crack.json";
  Material mat;
  mat.mu = E / (2 * (1 + nu));
  mat.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

  CrackModel M;
  CrackOptions o;
  o.discontinuous = true;
  o.quad.n_gauss = ng;
  o.quad.far_tol = 1e-12;
  o.grade = grade;
  o.end_shift = shift;
  {
    const Json J = Json::load(meta);
    for (size_t i = 0; i < J["dirichlet"].size(); ++i) o.dirichlet.push_back(J["dirichlet"][i].asInt());
  }
  M.setup(dat, meta, mat, o);
  if (kout == "stress") M.front.kappa_outer = (3 - nu) / (1 + nu);
  const int top = M.meta["top"].asInt(), bot = M.meta["bottom"].asInt();
  const double a = M.meta["a"].asNumber(), W = M.meta["W"].asNumber();
  TractionFn load = [&](const Vec3&, const Vec3&, int p) -> Vec3 {
    if (p == top) return sig * Vec3::UnitY();
    if (p == bot) return -sig * Vec3::UnitY();
    return Vec3::Zero();
  };
  const std::vector<Eigen::VectorXd> X = M.solve({load});

  // 2-D plane-strain reference (Eq. 30)
  const double q = a / W;
  const double F = 1.122 - 0.231 * q + 10.550 * q * q - 21.710 * q * q * q + 30.382 * q * q * q * q;
  const double K0 = sig * std::sqrt(M_PI * a);
  FILE* fs = out.empty() ? nullptr : std::fopen((out + "_sif.csv").c_str(), "w");
  if (fs) std::fprintf(fs, "z,KI,KII,KIII\n");
  const FrontPiece& Fp = M.front.pieces[0];
  const int nsm = 120;
  double kmid = 0, kmax2 = 0, kmax3 = 0;
  for (int i = 0; i <= nsm; ++i) {
    const double u = Fp.u0 + (Fp.u1 - Fp.u0) * i / nsm;
    Vec3 C, Cu;
    int n, s[16];
    double Rb[16];
    Fp.eval(u, C, Cu, n, s, Rb);
    const Vec3 K = M.front.sif(0, u, X[0]) / K0;
    if (fs) std::fprintf(fs, "%.12g,%.12g,%.12g,%.12g\n", C(2), K(0), K(1), K(2));
    if (2 * i == nsm) kmid = K(0);
    kmax2 = std::max(kmax2, std::abs(K(1)));
    kmax3 = std::max(kmax3, std::abs(K(2)));
  }
  if (fs) std::fclose(fs);
  char line[1024];
  std::snprintf(line, sizeof line,
                "{\"geometry\": \"%s\", \"kappa_outer\": \"%s\", \"unknowns\": %ld, \"F_2d\": %.6f, \"KI_mid\": %.6f, "
                "\"KI_mid_diff\": %.6e, \"KII_max\": %.3e, \"KIII_max\": %.3e, \"residual\": %.3e, "
                "\"t_assemble\": %.3f, \"t_solve\": %.3f}",
                dat.c_str(), kout.c_str(), M.unknowns, F, kmid, std::abs(kmid / F - 1), kmax2, kmax3, M.residual,
                M.t_assemble, M.t_solve);
  std::printf("%s\n", line);
  if (!out.empty()) std::ofstream(out + "_summary.json") << line << "\n";
  if (check > 0 && !(std::abs(kmid / F - 1) <= check)) {
    std::fprintf(stderr, "mid-front K_I %.4f differs from the 2-D value %.4f by more than %.1e\n", kmid, F, check);
    return 1;
  }
  return 0;
}
