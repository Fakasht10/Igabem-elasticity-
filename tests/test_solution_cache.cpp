// Solution cache (src/case_io.hpp CaseFile::solve): a case solved once is read back with no operators and no
// factorisation, bit for bit; any change to an input the solution depends on misses; a new load case on the same
// geometry and supports reuses the stored operators and LU factors and matches a fresh solve.
//
// usage: test_solution_cache <repository root>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "case_io.hpp"
#include "interior.hpp"

using namespace igabem;
namespace fs = std::filesystem;

static int failures = 0;
static void expect(const char* what, bool ok) {
  std::printf("%-68s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

struct Run {
  CaseFile cf;
  BcModel model;
  std::vector<LoadCase> cases;
  std::vector<CaseReport> rep;
  std::vector<BoundarySolution> sol;
  explicit Run(const std::string& path) {
    cf.load(path, model, cases);
    sol = cf.solve(model, cases, &rep);
  }
};

static std::string caseText(const std::string& geo, const std::string& cache, double E = 3500, double F = 1.0,
                            int ng = 8, const std::string& csv = "a.csv",
                            const std::string& sup = R"({"label": "clamp", "patches": [0], "fix": ["ux", "uy", "uz"]})") {
  char b[2048];
  std::snprintf(b, sizeof b,
                "{\"geometry\": \"%s\", \"cache\": \"%s\", \"material\": {\"E\": %.17g, \"nu\": 0.36},"
                " \"solver\": {\"n_gauss\": %d},"
                " \"supports\": [%s],"
                " \"cases\": [{\"name\": \"bend\", \"loads\": [{\"patches\": [1], \"force_N\": %.17g, \"direction\": [0, 0, -1]}]},"
                "             {\"name\": \"pull\", \"loads\": [{\"patches\": [1], \"force_N\": 2, \"direction\": [1, 0, 0]}]}],"
                " \"output\": {\"csv\": \"%s\"}}",
                geo.c_str(), cache.c_str(), E, ng, sup.c_str(), F, csv.c_str());
  return b;
}

static std::string write(const fs::path& p, const std::string& s) {
  std::ofstream(p) << s;
  return p.string();
}

int main(int argc, char** argv) {
  const fs::path root = argc > 1 ? argv[1] : ".";
  const fs::path tmp = fs::temp_directory_path() / ("igabem_solcache_" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  const std::string cache = (tmp / "cache").string();
  const std::string geo = (tmp / "plate.dat").string();
  fs::copy_file(root / "geo/plate_t0.2_uniform.dat", geo);

  // 1. first call solves and stores
  const std::string base = write(tmp / "base.json", caseText(geo, cache));
  Run a(base);
  expect("first call: solved (not from cache), operators assembled and factored", !a.cf.solution_from_cache && a.model.prepared());

  // 2. second call: no assembly, no LU, identical numbers
  Run b(base);
  expect("second call: from cache", b.cf.solution_from_cache);
  expect("second call: no operators, no factorisation (model not prepared)", !b.model.prepared());
  bool same = a.sol.size() == b.sol.size();
  for (size_t k = 0; same && k < a.sol.size(); ++k)
    same = a.sol[k].u == b.sol[k].u && a.sol[k].t == b.sol[k].t && a.sol[k].region.size() == b.sol[k].region.size();
  expect("second call: u and t bit-identical, same region densities", same);
  bool rep_same = a.rep.size() == b.rep.size();
  for (size_t k = 0; rep_same && k < a.rep.size(); ++k)
    rep_same = a.rep[k].total.F == b.rep[k].total.F && a.rep[k].reactions[0].second.F == b.rep[k].reactions[0].second.F &&
               a.rep[k].applied.F == b.rep[k].applied.F;
  expect("second call: reports (applied, reaction, residual) bit-identical", rep_same);
  {
    const Vec3 x(1.5, 0.5, 0.1);  // inside the plate
    const auto fa = InteriorEvaluator(a.sol[0], a.cf.opt).eval(x), fb = InteriorEvaluator(b.sol[0], b.cf.opt).eval(x);
    expect("second call: interior stress and displacement bit-identical", fa.stress == fb.stress && fa.u == fb.u);
  }
  Run c(write(tmp / "out.json", caseText(geo, cache, 3500, 1.0, 8, "other.csv")));
  expect("output-only change: still a hit", c.cf.solution_from_cache);

  // 3. every input the solution depends on misses
  Run dE(write(tmp / "E.json", caseText(geo, cache, 3600)));
  expect("material changed: miss", !dE.cf.solution_from_cache && dE.model.prepared());
  Run dg(write(tmp / "ng.json", caseText(geo, cache, 3500, 1.0, 10)));
  expect("quadrature (n_gauss) changed: miss", !dg.cf.solution_from_cache && dg.model.prepared());
  Run dB(write(tmp / "bc.json", caseText(geo, cache, 3500, 1.0, 8, "a.csv",
                                         R"({"label": "clamp", "patches": [0], "fix": ["ux", "uy", "uz"]},)"
                                         R"({"label": "slide", "patches": [1], "fix": ["uy"]})")));
  expect("boundary conditions changed: miss", !dB.cf.solution_from_cache && dB.model.prepared());
  Run dF(write(tmp / "F.json", caseText(geo, cache, 3500, 1.5)));
  expect("load changed: miss", !dF.cf.solution_from_cache && dF.model.prepared());
  // 4. ... and a new load case on the same geometry and supports reuses operators and LU factors
  expect("load changed: operators and LU factors read from cache", dF.model.from_cache && dF.model.lu_from_cache);
  {
    Run fresh(write(tmp / "F_fresh.json", caseText(geo, "none", 3500, 1.5)));
    const double e = (dF.sol[0].u - fresh.sol[0].u).norm() / fresh.sol[0].u.norm();
    std::printf("    cached-LU solve vs fresh LU: relative difference %.2e\n", e);
    expect("load changed: cached-LU solution equals a fresh solve (1e-12)", e < 1e-12);
    const double s = (dF.sol[0].u - 1.5 * a.sol[0].u).norm() / dF.sol[0].u.norm();
    expect("load changed: 1.5 x force gives 1.5 x displacement (1e-12)", s < 1e-12);
  }
  std::ofstream(geo, std::ios::app) << "\n# mesh edited\n";
  Run dgeo(base);
  expect("geometry file bytes changed: miss", !dgeo.cf.solution_from_cache && dgeo.model.prepared());

  fs::remove_all(tmp);
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "all passed", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
