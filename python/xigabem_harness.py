"""Reproduction harness for Rocha, Trevelyan & Leonel, "An extended isogeometric boundary element formulation for
three-dimensional linear elastic fracture mechanics", CMAME 423 (2024) 116872 (XIGABEM).

Each example of the paper is a case: geometry levels (python/make_crack.py), one crack_case run per level and
formulation (XIGABEM, plain IgBEM), gates on the reference values, and the paper's own numbers next to ours.
Results go to results/xigabem/<case>/ (summary JSON per run, SIF and opening CSVs, case.json with the gates), and
python/plot_xigabem.py draws the figures from there. Runs are skipped when their summary exists and is newer than
the geometry and the binary (pass --force to redo), so re-plotting or adding a level never re-solves the rest.

usage: xigabem_harness.py [case ...] [--force] [--timeout S] [--list]
       cases: penny (Sec. 4.1, in-plane, mode I) -- more follow as the milestones land

Runtime (4 cores): penny levels n = 3..11, two layouts, both formulations, ~4 min cold; every run has a hard timeout.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(ROOT, "build", "crack_case")
OUT = os.path.join(ROOT, "results", "xigabem")

# Paper, Table 1 (in-plane penny, nu = 0): dof -> signed average relative error of K_I.
PAPER_PENNY_TABLE1 = {378: 4.689e-03, 600: 2.936e-03, 882: 2.267e-03, 1224: 1.776e-03, 1626: 1.432e-03}

CASES = {
    "penny": {
        "title": "Penny-shaped crack, in plane, remote tension (paper Sec. 4.1)",
        "make": ["penny", "--layout", "{layout}"],
        "geometry": "penny_{prefix}i0_n{n}.dat",
        "levels": [3, 4, 5, 6, 7, 9, 11],
        "layouts": ["disk", "ogrid"],
        "formulations": {"xigabem": [], "igabem": ["--plain"]},
        "extra": [],
        "est_s": {3: 2, 4: 4, 5: 5, 6: 6, 7: 8, 9: 17, 11: 40},
        # gates on the finest XIGABEM level (n = 11): mean |K_I error| (paper reaches 1.4e-3 at n = 7), |K_II|, |K_III|
        # (exactly zero by symmetry; 1e-5 of K_I = 1.13 is quadrature-level asymmetry)
        "gates": {"KI_avg_err": 2.0e-3, "KII_avg_abs": 1.0e-5, "KIII_avg_abs": 1.0e-5},
        "paper": {"table1_KI_signed_avg_err": PAPER_PENNY_TABLE1},
    },
}


def stale(target, deps):
    if not os.path.exists(target):
        return True
    t = os.path.getmtime(target)
    return any(os.path.getmtime(d) > t for d in deps if os.path.exists(d))


def run_case(name, force, timeout):
    c = CASES[name]
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    plan = [(lay, n, f) for lay in c["layouts"] for n in c["levels"] for f in c["formulations"]]
    est = sum(c["est_s"][n] * (0.25 if f == "igabem" else 1.0) for lay, n, f in plan)
    print("%s: %d runs, estimated %.0f s cold (hard timeout %d s per run)" % (name, len(plan), est, timeout))
    runs = []
    for lay, n, f in plan:
        prefix = "" if lay == "disk" else "og_"
        geo = os.path.join(ROOT, "geo", "crack", c["geometry"].format(prefix=prefix, n=n))
        if not os.path.exists(geo):
            subprocess.run([sys.executable, os.path.join(ROOT, "python", "make_crack.py")] +
                           [a.format(layout=lay) for a in c["make"]] + ["--n", str(n)], check=True,
                           stdout=subprocess.DEVNULL)
        tag = "%s_%s_n%d" % (f, lay, n)
        summ = os.path.join(out, tag + "_summary.json")
        if force or stale(summ, [geo, BIN]):
            t0 = time.time()
            cmd = [BIN, geo, "--out", os.path.join(out, tag)] + c["formulations"][f] + c["extra"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4")))
            if r.returncode != 0:
                raise RuntimeError("%s failed: %s" % (tag, r.stderr))
            print("  %-22s %6.1f s" % (tag, time.time() - t0))
        with open(summ) as fh:
            d = json.load(fh)
        d.update({"formulation": f, "layout": lay, "n": n, "tag": tag})
        runs.append(d)
    # gates on the finest XIGABEM run of each layout
    verdict = {}
    for lay in c["layouts"]:
        fin = max((r for r in runs if r["layout"] == lay and r["formulation"] == "xigabem"), key=lambda r: r["n"])
        verdict[lay] = {k: {"value": fin[k], "gate": g, "pass": fin[k] <= g} for k, g in c["gates"].items()}
    with open(os.path.join(out, "case.json"), "w") as fh:
        json.dump({"case": name, "title": c["title"], "runs": runs, "gates": verdict, "paper": c["paper"]}, fh, indent=1)
    ok = all(v["pass"] for g in verdict.values() for v in g.values())
    for lay, g in verdict.items():
        for k, v in g.items():
            print("  gate %-6s %-13s %.3e <= %.1e  %s" % (lay, k, v["value"], v["gate"], "ok" if v["pass"] else "FAIL"))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", default=list(CASES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for k, c in CASES.items():
            print("%-8s %s" % (k, c["title"]))
        return 0
    if not os.path.exists(BIN):
        print("build crack_case first: cmake --build build --target crack_case", file=sys.stderr)
        return 2
    ok = all([run_case(n, a.force, a.timeout) for n in a.cases])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
