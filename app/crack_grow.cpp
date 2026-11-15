// Crack growth on a cached operator (src/crack/growth.hpp): SENB 3-point bend, crack a0 -> a0 + steps c, each
// step assembled from the previous one's raw rows. --verify also solves every step from scratch and reports the
// largest K difference along the front and the time ratio.
//
//   crack_grow step0.dat step1.dat ... [--nu NU] [--verify] [--check-k TOL] [--check-time R]
//
// Geometries from make_crack.py senb --growth (senb_grow); load P = 1 on the nose pad, P / 2 reactions on the
// support pads, the first end face clamped (as python/senb_k.py). One JSON line per step on stdout.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "crack/growth.hpp"
#include "json.hpp"

using namespace igabem;
using namespace igabem::crack;

int main(int argc, char** argv) {
  std::vector<std::string> dats;
  double nu = 0.0, check_k = -1, check_time = -1;
  bool verify = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--nu") && i + 1 < argc) nu = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify")) verify = true;
    else if (!std::strcmp(argv[i], "--check-k") && i + 1 < argc) { verify = true; check_k = std::atof(argv[++i]); }
    else if (!std::strcmp(argv[i], "--check-time") && i + 1 < argc) { verify = true; check_time = std::atof(argv[++i]); }
    else dats.push_back(argv[i]);
  }
  if (dats.empty()) {
    std::fprintf(stderr, "usage: crack_grow step0.dat step1.dat ... [--nu NU] [--verify] [--check-k TOL] [--check-time R]\n");
    return 2;
  }
  Material mat;
  mat.mu = 1.0 / (2 * (1 + nu));
  mat.lambda = nu / ((1 + nu) * (1 - 2 * nu));
  std::unique_ptr<CrackModel> prev;
  std::vector<TractionFn> prev_loads;
  std::unique_ptr<RawOperator> prev_raw;
  double worst_k = 0, worst_t = 0;
  for (size_t s = 0; s < dats.size(); ++s) {
    const std::string dat = dats[s], meta = dat.substr(0, dat.size() - 4) + ".crack.json";
    const Json J = Json::load(meta);
    const Json& g = J["patches"];
    CrackOptions o;
    o.discontinuous = true;
    o.quad.far_tol = 1e-12;
    o.quad.near_tol = 1e-8;
    o.dirichlet.push_back(g["ends"][0].asInt());
    auto M = std::make_unique<CrackModel>();
    M->setup(dat, meta, mat, o);
    const double B = J["B"].asNumber(), q = 0.5 / (J["pad"].asNumber() * B);
    std::vector<char> sup(M->mesh.patches.size(), 0), nose(M->mesh.patches.size(), 0);
    for (size_t i = 0; i < g["supports"].size(); ++i) sup[g["supports"][i].asInt()] = 1;
    for (size_t i = 0; i < g["nose"].size(); ++i) nose[g["nose"][i].asInt()] = 1;
    std::vector<TractionFn> loads = {[sup, nose, q](const Vec3&, const Vec3&, int p) -> Vec3 {
      if (sup[p]) return Vec3(0, q, 0);
      if (nose[p]) return Vec3(0, -2 * q, 0);
      return Vec3::Zero();
    }};
    auto raw = std::make_unique<RawOperator>();
    GrowthStats st;
    const auto t0 = std::chrono::steady_clock::now();
    const auto X = Growth::solve(*M, loads, {}, *raw, prev.get(), prev ? &prev_loads : nullptr, prev_raw.get(), st);
    const double t_step = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double t_asm = M->t_assemble;
    // K_I along the front (mid-thickness and mean)
    const FrontPiece& F = M->front.pieces[0];
    const int ns = 16;
    std::vector<Vec3> K(ns);
    double kmean = 0;
    for (int i = 0; i < ns; ++i) {
      K[i] = M->front.sif(0, F.u0 + (F.u1 - F.u0) * (i + 0.5) / ns, X[0]);
      kmean += K[i](0) / ns;
    }
    double dk = -1, t_full = -1;
    if (verify && prev) {
      const auto Xf = M->solve(loads);
      t_full = M->t_assemble;
      dk = 0;
      for (int i = 0; i < ns; ++i) {
        const Vec3 Kf = M->front.sif(0, F.u0 + (F.u1 - F.u0) * (i + 0.5) / ns, Xf[0]);
        dk = std::max(dk, (K[i] - Kf).norm() / std::abs(Kf(0)));
      }
      worst_k = std::max(worst_k, dk);
      worst_t = std::max(worst_t, t_asm / t_full);
    }
    std::printf("{\"step\": %zu, \"a\": %.6g, \"unknowns\": %ld, \"KI_mean\": %.10g, \"rows_reused\": %ld, \"rows_new\": %ld, "
                "\"elems\": %ld, \"elems_removed\": %ld, \"elems_added\": %ld, \"t_assemble\": %.3f, \"t_step\": %.3f, "
                "\"t_full_assemble\": %.3f, \"time_ratio\": %.4f, \"K_diff\": %.3e, \"t_quad\": %.3f, \"cpu_remove\": %.2f, "
                "\"cpu_add\": %.2f, \"cpu_finish\": %.2f}\n",
                s, J["a"].asNumber(), M->unknowns, kmean, st.rows_reused, st.rows_new, st.elems, st.elems_removed,
                st.elems_added, t_asm, t_step, t_full, t_full > 0 ? t_asm / t_full : -1.0, dk, st.t_quad, st.t_rem, st.t_add,
                st.t_finish);
    std::fflush(stdout);
    prev = std::move(M);
    prev_loads = loads;
    prev_raw = std::move(raw);
  }
  if (check_k > 0 && !(worst_k <= check_k)) { std::fprintf(stderr, "K differs from full reassembly by %.3e\n", worst_k); return 1; }
  if (check_time > 0 && !(worst_t <= check_time)) { std::fprintf(stderr, "step/assembly time %.3f above %.3f\n", worst_t, check_time); return 1; }
  return 0;
}
