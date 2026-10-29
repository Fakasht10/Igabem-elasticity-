// JSON case files for BcModel, in the terms of nurb-fea-tcd's BEM payload
// (feakit/bem_backend.py -> bemkit.solve): a support or load names where it
// acts by "face_id" (CAD face, python/face_ids.py), "face_ids", "patches" or a
// geometric "select"or, and what it does by bemkit's keys where they exist.
//
// {
//   "geometry": "geo/step/bracket_exact_p3_h0.7_c3.dat",   // .faces next to it
//   "material": {"E": 2000, "nu": 0.35},
//   "solver": {"n_gauss": 12, "far_tol": 1e-10},
//   "grade": {"faces": [3], "inner": [1, 13], "h0": 0.025},   // mesh graded towards supports
//   "supports": [
//     {"face_id": 3, "fix": ["ux", "uy", "uz"]},             // or "dofs": {"ux": true, ...}
//     {"select": {"box": [[x0,y0,z0], [x1,y1,z1]]}, "fix": ["uz"], "value": [0, 0, -0.01]},
//     {"symmetry": "y", "at": 0},
//     {"face_id": 7, "spring": {"k_n": 1e3, "k_t": 0}}         // or {"k": [kx,ky,kz]}, {"K_total": K}
//   ],
//   "cases": [{"name": "pull", "loads": [
//     {"face_id": 5, "force_N": 100, "direction": [1, 0, 0]},  // uniform, resultant F
//     {"select": {...}, "pressure": 2.0},
//     {"face_id": 5, "moment": [0, 0, 10], "about": [x, y, z]}, // + optional force_N/direction
//     {"face_id": 9, "bearing": {"force_N": 50, "direction": [1, 0, 0]}},
//     {"select": {...}, "traction": [tx, ty, tz]}]}],           // or top-level "loads": one case
//   "output": {"vtu": "out/bracket", "csv": "out/bracket.csv"}
// }
//
// Selectors: {"box": [lo, hi]}, {"plane": {"point", "normal", "tol"}},
// {"half": {"point", "normal"}}, {"cylinder": {"point", "axis", "radius", "tol"}},
// {"ball": {"center", "radius"}}, {"normal": {"direction", "angle_deg"}},
// {"faces": [...]}, {"patches": [...]}; keys in one object are ANDed, a list of
// objects is ORed. A support needs whole patches (it sets which traction
// components are unknown); "snap": true takes the patches covered at least half.
#pragma once
#ifndef IGABEM_SOURCE_DIR
#define IGABEM_SOURCE_DIR "."
#endif
#include <cstdlib>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bc_model.hpp"
#include "json.hpp"
#include "loads.hpp"
#include "regions.hpp"

namespace igabem {

inline Vec3 jvec(const Json& j) {
  const auto v = j.asNumbers();
  if (v.size() != 3) throw std::runtime_error("json: 3-vector expected");
  return Vec3(v[0], v[1], v[2]);
}

struct CaseFile {
  std::string geometry, faces, cache = "cache";
  Material mat;
  double E = 2.5, nu = 0.25;
  SolverOptions opt;
  std::vector<std::string> case_names;
  std::string vtu, csv;
  double size = 1.0;  // bounding-box diagonal (default tolerances)
  std::vector<int> face_of_patch;

  // Builds the mesh, regions, supports and load cases into model and cases.
  void load(const std::string& path, BcModel& model, std::vector<LoadCase>& cases) {
    const Json J = Json::load(path);
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    auto resolve = [&](const std::string& f) {
      if (std::filesystem::exists(f) || std::filesystem::path(f).is_absolute()) return f;
      return (dir / f).string();
    };
    geometry = resolve(J["geometry"].asString());
    faces = J.has("faces") ? resolve(J["faces"].asString()) : faceMapPath(geometry);
    if (J.has("grade")) gradeGeometry(J["grade"]);
    if (J.has("cache")) cache = J["cache"].asString();
    if (cache == "none") cache.clear();
    if (J.has("material")) {
      E = J["material"].get("E", E);
      nu = J["material"].get("nu", nu);
    }
    mat.mu = E / (2.0 * (1.0 + nu));
    mat.lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    opt.n_gauss = 12;
    opt.far_tol = 1e-10;
    if (J.has("solver")) {
      opt.n_gauss = int(J["solver"].get("n_gauss", double(opt.n_gauss)));
      opt.far_tol = J["solver"].get("far_tol", opt.far_tol);
    }
    if (J.has("output")) {
      vtu = J["output"].get("vtu", std::string());
      csv = J["output"].get("csv", std::string());
    }
    model.mat = mat;
    model.opt = opt;
    model.mesh.build(loadDat(geometry), model.popt);
    model.ts.build(model.mesh);
    face_of_patch = loadFaceMap(faces);
    Vec3 lo = Vec3::Constant(1e300), hi = Vec3::Constant(-1e300);
    for (const Vec3& P : model.mesh.dof_point) { lo = lo.cwiseMin(P); hi = hi.cwiseMax(P); }
    size = (hi - lo).norm();

    std::vector<FixedSupport> fixed;
    std::vector<SpringSupport> springs;
    int ns = 0;
    for (const Json& s : J["supports"].arr) {
      const std::string label = s.get("label", "support " + std::to_string(ns++));
      if (s.has("symmetry")) {
        const std::string ax = s["symmetry"].asString();
        const int c = ax == "x" ? 0 : ax == "y" ? 1 : ax == "z" ? 2 : -1;
        if (c < 0) throw std::runtime_error(label + ": symmetry must be x, y or z");
        Region R;
        R.label = label;
        Selector sel;
        sel.plane = true;
        sel.p0 = Vec3::Unit(c) * s.get("at", 0.0);
        sel.pn = Vec3::Unit(c);
        sel.ptol = s.get("tol", 1e-6 * size);
        R.any.push_back(sel);
        FixedSupport f;
        f.label = label;
        f.patches = wholePatches(model.mesh, R, label, false);
        f.fix = {c == 0, c == 1, c == 2};
        fixed.push_back(f);
        continue;
      }
      Region R = region(s, label);
      if (s.has("spring")) {
        const Json& k = s["spring"];
        auto RM = std::make_shared<RegionMesh>(model.mesh, R);
        SpringSupport sp;
        sp.label = label;
        sp.region = RM;
        if (k.has("k")) sp.k = springXYZ(jvec(k["k"]));
        else if (k.has("K_total")) sp.k = springNT(k["K_total"].asNumber() / RM->area, 0.0);
        else sp.k = springNT(k.get("k_n", 0.0), k.get("k_t", 0.0));
        springs.push_back(sp);
        continue;
      }
      FixedSupport f;
      f.label = label;
      f.patches = wholePatches(model.mesh, R, label, s.has("snap") && s["snap"].asBool());
      if (s.has("fix")) {
        f.fix = {false, false, false};
        for (const Json& c : s["fix"].arr) f.fix[c.asString() == "ux" ? 0 : c.asString() == "uy" ? 1 : 2] = true;
      } else if (s.has("dofs")) {
        const Json& d = s["dofs"];
        f.fix = {d.has("ux") && d["ux"].asBool(), d.has("uy") && d["uy"].asBool(), d.has("uz") && d["uz"].asBool()};
      }
      if (s.has("value")) {
        const Vec3 v = jvec(s["value"]);
        f.value = [v](const Vec3&) { return v; };
      }
      fixed.push_back(f);
    }

    std::vector<std::pair<std::string, const Json*>> specs;
    if (J.has("cases"))
      for (const Json& c : J["cases"].arr) specs.push_back({c.get("name", "case " + std::to_string(specs.size())), &c["loads"]});
    if (J.has("loads")) specs.push_back({"load", &J["loads"]});
    std::map<std::string, RegionPtr> regions;  // same selector text -> same region (shared RHS pass)
    for (const auto& [name, loads] : specs) {
      LoadCase lc;
      lc.name = name;
      int nl = 0;
      for (const Json& L : loads->arr) {
        const std::string label = L.get("label", name + " load " + std::to_string(nl++));
        const std::string key = regionKey(L);
        RegionPtr RM = regions.count(key) ? regions[key] : (regions[key] = std::make_shared<RegionMesh>(model.mesh, region(L, label)));
        if (RM->area <= 0.0) throw std::runtime_error(label + ": the region selects nothing");
        if (L.has("pressure")) lc.loads.push_back(pressureLoad(label, RM, L["pressure"].asNumber()));
        else if (L.has("traction")) lc.loads.push_back(tractionLoad(label, RM, jvec(L["traction"])));
        else if (L.has("bearing")) {
          const Json& b = L["bearing"];
          lc.loads.push_back(bearingLoad(label, RM, b["force_N"].asNumber() * jvec(b["direction"]).normalized()));
        } else {
          Vec3 F = Vec3::Zero();
          if (L.has("force_N")) {
            // bemkit: direction defaults to -(outward face normal), a push
            const Vec3 d = L.has("direction") ? jvec(L["direction"]) : Vec3(-RM->normal_sum);
            F = L["force_N"].asNumber() * d.normalized();
          }
          if (L.has("moment"))
            lc.loads.push_back(wrenchLoad(label, RM, F, jvec(L["moment"]), L.has("about") ? jvec(L["about"]) : RM->centroid));
          else if (L.has("force_N"))
            lc.loads.push_back(forceLoad(label, RM, F));
          else
            throw std::runtime_error(label + ": no load (force_N, moment, pressure, traction or bearing)");
        }
      }
      cases.push_back(std::move(lc));
      case_names.push_back(name);
    }
    model.setupOn(geometry, std::move(fixed), std::move(springs), cache);
  }

  Region region(const Json& e, const std::string& label) const {
    Region R;
    R.label = label;
    R.face_of_patch = face_of_patch;
    auto add = [&](const Json& s) { R.any.push_back(selector(s)); };
    if (e.has("face_id") || e.has("face_ids") || e.has("patches")) {
      Selector s;
      if (e.has("face_id")) s.faces.push_back(e["face_id"].asInt());
      if (e.has("face_ids")) for (double v : e["face_ids"].asNumbers()) s.faces.push_back(int(v));
      if (e.has("patches")) for (double v : e["patches"].asNumbers()) s.patches.push_back(int(v));
      if (e.has("select")) {  // face and selector together: AND
        Selector m = selector(e["select"]);
        m.faces = s.faces;
        m.patches = s.patches;
        R.any.push_back(m);
      } else {
        R.any.push_back(s);
      }
    } else if (e.has("select")) {
      if (e["select"].isArray()) for (const Json& s : e["select"].arr) add(s);
      else add(e["select"]);
    } else {
      throw std::runtime_error(label + ": no face_id, face_ids, patches or select");
    }
    if ((e.has("face_id") || e.has("face_ids")) && face_of_patch.empty())
      throw std::runtime_error(label + ": face ids need the face map " + faces + " (python/face_ids.py)");
    return R;
  }

  Selector selector(const Json& s) const {
    Selector g;
    if (s.has("box")) { g.box = true; g.lo = jvec(s["box"][0]); g.hi = jvec(s["box"][1]); }
    if (s.has("plane")) {
      g.plane = true;
      g.p0 = jvec(s["plane"]["point"]);
      g.pn = jvec(s["plane"]["normal"]).normalized();
      g.ptol = s["plane"].get("tol", 1e-6 * size);
    }
    if (s.has("half")) { g.half = true; g.h0 = jvec(s["half"]["point"]); g.hn = jvec(s["half"]["normal"]).normalized(); }
    if (s.has("cylinder")) {
      const Json& c = s["cylinder"];
      g.cyl = true;
      g.c0 = jvec(c["point"]);
      g.cd = jvec(c["axis"]).normalized();
      g.cr = c["radius"].asNumber();
      g.ctol = c.get("tol", 1e-6 * size);
    }
    if (s.has("ball")) { g.ball = true; g.bc = jvec(s["ball"]["center"]); g.br = s["ball"]["radius"].asNumber(); }
    if (s.has("normal")) {
      g.normal = true;
      g.nd = jvec(s["normal"]["direction"]).normalized();
      g.ncos = std::cos(s["normal"].get("angle_deg", 1.0) * M_PI / 180.0);
    }
    if (s.has("faces")) for (double v : s["faces"].asNumbers()) g.faces.push_back(int(v));
    if (s.has("patches")) for (double v : s["patches"].asNumbers()) g.patches.push_back(int(v));
    return g;
  }

  // Patches a support region covers. Supports set the unknowns patch by patch,
  // so a partial cover is refused unless snap (covered at least half).
  std::vector<int> wholePatches(const Mesh& m, const Region& R, const std::string& label, bool snap) const {
    std::vector<int> out;
    if (R.patchwise()) {
      for (size_t p = 0; p < m.patches.size(); ++p) {
        PatchEval E;
        const Patch& P = m.patches[p];
        P.eval(0.5 * (P.U.front() + P.U.back()), 0.5 * (P.V.front() + P.V.back()), E);
        if (R.contains(int(p), E.x, (E.normal() * m.normal_sign).normalized())) out.push_back(int(p));
      }
    } else {
      const RegionMesh RM(m, R, 3);
      const auto cov = RM.coverage();
      for (size_t p = 0; p < cov.size(); ++p) {
        if (cov[p] > 1.0 - 1e-6 || (snap && cov[p] >= 0.5)) out.push_back(int(p));
        else if (cov[p] > 1e-6 && !snap)
          throw std::runtime_error(label + ": covers " + std::to_string(100.0 * cov[p]) + "% of patch " +
                                   std::to_string(p) + "; supports need whole patches (select whole CAD faces, use a "
                                   "spring, or \"snap\": true)");
      }
    }
    if (out.empty()) throw std::runtime_error(label + ": the support selects no patch");
    return out;
  }

 private:
  // "grade": {"faces": [6], "inner": [1, 13], "h0": 0.025, "growth": 1.4}: refine the mesh
  // towards the given CAD faces (supports) by knot insertion, python/grade_dat.py; the graded
  // surface <stem>_g<faces>[_i<inner>]_h<h0>.dat (+ .faces) is made once and reused.
  void gradeGeometry(const Json& G) {
    auto ids = [&](const char* k) {
      std::string list, tag;
      if (G.has(k))
        for (double v : G[k].asNumbers()) {
          list += (list.empty() ? "" : ",") + std::to_string(int(v));
          tag += (tag.empty() ? "" : "-") + std::to_string(int(v));
        }
      return std::make_pair(list, tag);
    };
    const auto [fl, ft] = ids("faces");
    const auto [il, it] = ids("inner");
    if (fl.empty()) throw std::runtime_error("grade: needs \"faces\"");
    const double h0 = G.get("h0", 0.025), g = G.get("growth", 1.4);
    char hs[64];
    std::snprintf(hs, sizeof hs, "_h%g", h0);
    const std::filesystem::path src(geometry);
    const std::string out = (src.parent_path() / (src.stem().string() + "_g" + ft + (it.empty() ? "" : "_i" + it) +
                                                  hs + (g == 1.4 ? "" : "_r" + std::to_string(g)) + ".dat")).string();
    if (!std::filesystem::exists(out) ||
        std::filesystem::last_write_time(out) < std::filesystem::last_write_time(src)) {
      char cmd[2048];
      std::snprintf(cmd, sizeof cmd, "python3 \"%s/python/grade_dat.py\" \"%s\" \"%s\" --faces %s%s%s --h0 %.17g --growth %.17g",
                    IGABEM_SOURCE_DIR, geometry.c_str(), out.c_str(), fl.c_str(), il.empty() ? "" : " --inner ",
                    il.c_str(), h0, g);
      std::printf("# grading: %s\n", cmd);
      std::fflush(stdout);
      if (std::system(cmd) != 0) throw std::runtime_error("grade: python/grade_dat.py failed");
    }
    geometry = out;
    faces = faceMapPath(geometry);
  }

  static std::string regionKey(const Json& L) {
    std::string k;
    for (const char* f : {"face_id", "face_ids", "patches"})
      if (L.has(f)) for (double v : L[f].asNumbers()) k += std::string(f) + std::to_string(v) + ",";
    if (L.has("select")) k += "select@" + std::to_string(reinterpret_cast<std::uintptr_t>(&L["select"]));
    return k;
  }
};

}  // namespace igabem
