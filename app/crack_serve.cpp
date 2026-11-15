// SIF server for client programs (nurb-fea-tcd's fracture extension): one JSON request per stdin line, one JSON
// reply per stdout line, schema igabem.sif/1. Operators are cached across requests.
//
//   crack_serve                    serve until EOF or {"cmd": "quit"}
//   crack_serve --case req.json    answer one request and exit (scripts, tests)
//
// Request (units are the caller's; K comes back in load units x sqrt(length)):
//   {"cmd": "solve", "id": "any",
//    "geometry": "/abs/geo/crack/penny_og_i0_n5.dat",          // sidecar <stem>.crack.json beside it
//    "material": {"nu": 0.3, "E": 1.0},                        // K does not depend on E under traction loads
//    "options": {"enrich": true, "ng": 16, "grade": 6, "shift": 0.1, "second": true, "far_tol": 1e-12, "near_tol": 1e-8, "discontinuous": false},
//    "clamp": [{"normal": [0, -1, 0]}, {"patches": [3, 4]}],   // outer patches held at zero displacement
//    "loads": [{"name": "far", "tractions": [{"normal": [0, 1, 0], "t": [0, 1, 0]}], "crack_pressure": 0}],
//    "samples": 40}                                            // front samples per front piece
//   Outer patches are picked by their outward normal at the parameter midpoint (n . sel > 0.99) or by index.
//   crack_pressure p puts t = -p n on both crack faces (n the face's outward normal).
// Reply:
//   {"schema": "igabem.sif/1", "id": ..., "ok": true, "loads": ["far"],
//    "front": [{"piece": 0, "u": .., "x": [..], "t": [..], "n": [..], "b": [..], "K": [[KI, KII, KIII], ...]}],
//    "info": {"unknowns": .., "cache_hit": false, "t_assemble": .., "t_factor": .., "t_rhs": .., "residual": ..,
//             "git": ".."}}
//   (t front tangent, b crack normal toward the upper face's solid, n = b x t pointing ahead of the front; K is
//   one row per load case.)  Errors: {"schema": "igabem.sif/1", "id": ..., "ok": false, "error": "..."}.
//
// Cache: the factored operator depends on geometry (file, size, mtime), nu, the options and the clamp set. A new
// load set on a cached operator costs a right-hand-side assembly and one back-substitution.
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <string>

#include "crack/dual_model.hpp"

#ifndef IGABEM_GIT
#define IGABEM_GIT "unknown"
#endif

using namespace igabem;
using namespace igabem::crack;

namespace {

struct Cached {
  std::string key;
  std::unique_ptr<CrackModel> model;
};
std::list<Cached> cache;  // most recent first
const size_t kCacheSize = 4;

std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  char b[32];
  std::snprintf(b, sizeof b, "%.15g", v);
  return b;
}
std::string jvec(const Vec3& v) { return "[" + jnum(v(0)) + ", " + jnum(v(1)) + ", " + jnum(v(2)) + "]"; }
std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}
Vec3 vec3(const Json& j) {
  if (!j.isArray() || j.size() != 3) throw std::runtime_error("expected a 3-vector");
  return Vec3(j[0].asNumber(), j[1].asNumber(), j[2].asNumber());
}

// outer patches chosen by {"patches": [...]} or {"normal": [...]}
std::vector<char> selectOuter(const Json& sel, int n_outer, const std::function<Vec3(int)>& normal) {
  std::vector<char> on(n_outer, 0);
  if (sel.has("patches")) {
    for (size_t i = 0; i < sel["patches"].size(); ++i) {
      const int p = sel["patches"][i].asInt();
      if (p < 0 || p >= n_outer) throw std::runtime_error("patch " + std::to_string(p) + " is not an outer patch");
      on[p] = 1;
    }
  } else if (sel.has("normal")) {
    const Vec3 d = vec3(sel["normal"]).normalized();
    for (int p = 0; p < n_outer; ++p) on[p] = normal(p).dot(d) > 0.99;
  } else {
    throw std::runtime_error("a selector needs \"patches\" or \"normal\"");
  }
  int k = 0;
  for (char c : on) k += c;
  if (!k) throw std::runtime_error("a selector matched no outer patch");
  return on;
}

std::string solve(const Json& rq) {
  const std::string dat = rq["geometry"].asString();
  if (dat.size() < 5 || dat.substr(dat.size() - 4) != ".dat") throw std::runtime_error("geometry must be a .dat file");
  const std::string meta = dat.substr(0, dat.size() - 4) + ".crack.json";
  struct stat st;
  if (stat(dat.c_str(), &st) != 0) throw std::runtime_error("cannot read " + dat);
  const Json& mj = rq["material"];
  const double nu = mj.get("nu", 0.0), E = mj.get("E", 1.0);
  if (!(nu >= 0.0 && nu < 0.5) || !(E > 0)) throw std::runtime_error("need 0 <= nu < 0.5 and E > 0");
  const Json& oj = rq["options"];
  CrackOptions o;
  o.enrich = oj.has("enrich") ? oj["enrich"].asBool() : true;
  o.quad.n_gauss = int(oj.get("ng", 16.0));
  o.grade = int(oj.get("grade", 6.0));
  o.end_shift = oj.get("shift", 0.1);
  o.second = !oj.has("second") || oj["second"].asBool();
  o.quad.far_tol = oj.get("far_tol", 1e-12);
  o.quad.near_tol = oj.get("near_tol", 1e-8);
  o.discontinuous = oj.has("discontinuous") && oj["discontinuous"].asBool();
  if (!o.enrich) throw std::runtime_error("SIFs need the enrichment (options.enrich = true)");

  // clamp set: resolved on the raw patches (before the model exists), as crack_case does
  const Json J = Json::load(meta);
  const int n_outer = J["outer"][1].asInt();
  std::vector<Patch> P0 = loadDat(dat);
  // outer patches are outward as written (make_crack.py); the same rule crack_case uses
  auto rawNormal = [&](int p) {
    PatchEval Ev;
    const Patch& Q = P0[p];
    Q.eval(0.5 * (Q.U.front() + Q.U.back()), 0.5 * (Q.V.front() + Q.V.back()), Ev);
    return Vec3(Ev.normal().normalized());
  };
  std::vector<char> clamp(n_outer, 0);
  for (size_t i = 0; i < rq["clamp"].size(); ++i) {
    const std::vector<char> s = selectOuter(rq["clamp"][i], n_outer, rawNormal);
    for (int p = 0; p < n_outer; ++p) clamp[p] |= s[p];
  }
  for (int p = 0; p < n_outer; ++p)
    if (clamp[p]) o.dirichlet.push_back(p);

  std::ostringstream key;
  key.precision(17);
  key << dat << '|' << st.st_size << '|' << st.st_mtime << '|' << nu << '|' << o.enrich << o.discontinuous << '|'
      << o.quad.n_gauss << '|' << o.grade << '|' << o.end_shift << '|' << o.second << '|' << o.quad.far_tol << ',' << o.quad.near_tol << '|';
  for (int p : o.dirichlet) key << p << ',';

  CrackModel* M = nullptr;
  bool hit = false;
  for (auto it = cache.begin(); it != cache.end(); ++it)
    if (it->key == key.str()) {
      cache.splice(cache.begin(), cache, it);
      M = cache.front().model.get();
      hit = true;
      break;
    }
  if (!M) {
    auto m = std::make_unique<CrackModel>();
    Material mat;
    mat.mu = E / (2 * (1 + nu));
    mat.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
    m->setup(dat, meta, mat, o);
    m->factor();
    cache.push_front({key.str(), std::move(m)});
    if (cache.size() > kCacheSize) cache.pop_back();
    M = cache.front().model.get();
  }

  // load cases
  const Json& lj = rq["loads"];
  if (!lj.isArray() || lj.size() == 0) throw std::runtime_error("need at least one load case");
  std::vector<TractionFn> loads;
  std::vector<std::string> names;
  for (size_t l = 0; l < lj.size(); ++l) {
    const Json& c = lj[l];
    names.push_back(c.get("name", "load" + std::to_string(l)));
    std::vector<Vec3> tp(M->n_outer, Vec3::Zero());
    for (size_t i = 0; i < c["tractions"].size(); ++i) {
      const Json& tr = c["tractions"][i];
      const std::vector<char> s = selectOuter(tr, M->n_outer, rawNormal);
      const Vec3 t = vec3(tr["t"]);
      for (int p = 0; p < M->n_outer; ++p)
        if (s[p]) {
          if (clamp[p]) throw std::runtime_error("load case " + names.back() + " puts a traction on a clamped patch");
          tp[p] += t;
        }
    }
    const double pr = c.get("crack_pressure", 0.0);
    const int no = M->n_outer;
    loads.push_back([tp, pr, no](const Vec3&, const Vec3& n, int p) -> Vec3 {
      if (p < no) return tp[p];
      return -pr * n;
    });
  }
  const std::vector<Eigen::VectorXd> X = M->solveFactored(loads);

  // front samples
  const int ns = int(rq.get("samples", 40.0));
  if (ns < 1 || ns > 10000) throw std::runtime_error("samples must be 1..10000");
  std::string fr;
  for (size_t k = 0; k < M->front.pieces.size(); ++k) {
    const FrontPiece& F = M->front.pieces[k];
    for (int i = 0; i < ns; ++i) {
      const double u = F.u0 + (F.u1 - F.u0) * (i + 0.5) / ns;
      Vec3 C, Cu;
      int n, s[16];
      double R[16];
      F.eval(u, C, Cu, n, s, R);
      // the enrichment's own frame (front.hpp shapes): n = m x t, flipped to point away from the crack
      const Vec3 b = M->m;
      Vec3 nn = b.cross(Cu.normalized());
      if (nn.dot(C - M->front.centroid) < 0) nn = -nn;
      nn.normalize();
      const Vec3 t = nn.cross(b);
      std::string K;
      for (size_t l = 0; l < X.size(); ++l) K += (l ? ", " : "") + jvec(M->front.sif(int(k), u, X[l]));
      if (!fr.empty()) fr += ", ";
      fr += "{\"piece\": " + std::to_string(k) + ", \"u\": " + jnum(u) + ", \"x\": " + jvec(C) + ", \"t\": " + jvec(t) +
            ", \"n\": " + jvec(nn) + ", \"b\": " + jvec(b) + ", \"K\": [" + K + "]}";
    }
  }
  std::string nm;
  for (size_t l = 0; l < names.size(); ++l) nm += (l ? ", " : "") + jstr(names[l]);
  return "\"ok\": true, \"loads\": [" + nm + "], \"front\": [" + fr + "], \"info\": {\"unknowns\": " +
         std::to_string(M->unknowns) + ", \"cache_hit\": " + (hit ? "true" : "false") +
         ", \"t_assemble\": " + jnum(hit ? 0.0 : M->t_assemble) + ", \"t_factor\": " + jnum(hit ? 0.0 : M->t_factor) +
         ", \"t_rhs\": " + jnum(M->t_rhs) + ", \"residual\": " + jnum(M->residual) + ", \"git\": " + jstr(IGABEM_GIT) + "}";
}

std::string answer(const std::string& line, bool& quit) {
  std::string id = "null";
  try {
    const Json rq = Json::parse(line);
    if (rq.has("id")) id = rq["id"].isString() ? jstr(rq["id"].asString()) : jnum(rq["id"].asNumber());
    const std::string cmd = rq.get("cmd", "solve");
    std::string body;
    if (cmd == "quit") { quit = true; body = "\"ok\": true"; }
    else if (cmd == "info") body = "\"ok\": true, \"git\": " + jstr(IGABEM_GIT) + ", \"cached\": " + std::to_string(cache.size());
    else if (cmd == "solve") body = solve(rq);
    else throw std::runtime_error("unknown cmd " + cmd);
    return "{\"schema\": \"igabem.sif/1\", \"id\": " + id + ", " + body + "}";
  } catch (const std::exception& e) {
    return "{\"schema\": \"igabem.sif/1\", \"id\": " + id + ", \"ok\": false, \"error\": " + jstr(e.what()) + "}";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && !std::strcmp(argv[1], "--case")) {
    std::ifstream f(argv[2]);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 2; }
    std::stringstream ss;
    ss << f.rdbuf();
    bool quit = false;
    const std::string r = answer(ss.str(), quit);
    std::printf("%s\n", r.c_str());
    return r.find("\"ok\": true") != std::string::npos ? 0 : 1;
  }
  if (argc != 1) { std::fprintf(stderr, "usage: crack_serve [--case request.json]\n"); return 2; }
  std::string line;
  bool quit = false;
  while (!quit && std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::cout << answer(line, quit) << std::endl;
  }
  return 0;
}
