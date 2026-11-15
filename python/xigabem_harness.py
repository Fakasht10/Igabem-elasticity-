"""Reproduction harness for Rocha, Trevelyan & Leonel, "An extended isogeometric boundary element formulation for
three-dimensional linear elastic fracture mechanics", CMAME 423 (2024) 116872 (XIGABEM).

Each example of the paper is a case: geometry levels (python/make_crack.py), one crack_case run per level and
formulation (XIGABEM, plain IgBEM), gates on the reference values, and the paper's own numbers next to ours.
Results go to results/xigabem/<case>/ (summary JSON per run, SIF and opening CSVs, case.json with the gates), and
python/plot_xigabem.py draws the figures from there. Runs are skipped when their summary exists and is newer than
the geometry and the binary (pass --force to redo), so re-plotting or adding a level never re-solves the rest.

usage: xigabem_harness.py [case ...] [--force] [--timeout S] [--list]
       cases: penny (Sec. 4.1, in plane, mode I), penny_inclined (Sec. 4.1, pi/4, mixed mode),
              ellipse, ellipse_inclined (Sec. 4.2, a = 2b, in plane and pi/6),
              prism (Sec. 4.3, edge crack, pure modes I, II, III), prism_tension (Sec. 4.4)

Collocation end shift 0.02 and fixed-zone front grading (layouts "<name>_graded", make_crack.py --grade d,q,L, the
case's "grade") follow the independent convergence review; prism_a3 is its test with exact K and an r^3/2 remainder.

Runtime (4 cores): all cases ~20 min cold (penny ~6, ellipse ~6, prism_tension ~6); every run has a hard timeout.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BIN = os.path.join(ROOT, "build", "crack_case")
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "results", "xigabem")

# Paper, Table 1 (in-plane penny, nu = 0): dof -> signed average relative error of K_I.
PAPER_PENNY_TABLE1 = {378: 4.689e-03, 600: 2.936e-03, 882: 2.267e-03, 1224: 1.776e-03, 1626: 1.432e-03}

CASES = {
    "penny": {
        "title": "Penny-shaped crack, in plane, remote tension (paper Sec. 4.1)",
        "make": ["penny", "--layout", "{layout}"],
        "geometry": "penny_{prefix}i0_n{n}.dat",
        "levels": [3, 4, 5, 6, 7, 9, 11],
        "layouts": ["disk", "ogrid", "disk_graded"],
        "grade": "0.05,0.3,2",
        "formulations": {"xigabem": [], "igabem": ["--plain"]},
        # independent K: interaction integral on front slices, annulus 0.1 R (crack/interaction.hpp)
        "extra": ["--mint", "0.1", "--indicator"],
        "est_s": {3: 4, 4: 6, 5: 8, 6: 10, 7: 14, 9: 25, 11: 55},
        # gates on the finest XIGABEM level (n = 11): mean |K_I error| (paper reaches 1.4e-3 at n = 7), |K_II|, |K_III|
        # (exactly zero by symmetry; 1e-5 of K_I = 1.13 is quadrature-level asymmetry), and the M-integral K_I
        "gates": {"KI_avg_err": 2.0e-3, "KII_avg_abs": 1.0e-5, "KIII_avg_abs": 1.0e-5, "KM_I_avg_err": 1.0e-3,
                  "effectivity": [0.3, 3.0]},
        "paper": {"table1_KI_signed_avg_err": PAPER_PENNY_TABLE1},
    },
    "penny_inclined": {
        "title": "Penny-shaped crack inclined by pi/4, remote tension: mixed mode I/II/III (paper Sec. 4.1, Fig. 12)",
        "make": ["penny", "--layout", "{layout}", "--incline", "45"],
        "geometry": "penny_{prefix}i45_n{n}.dat",
        "levels": [3, 4, 5, 6, 7, 9, 11],
        "layouts": ["ogrid"],
        "formulations": {"xigabem": []},
        "extra": [],
        "est_s": {3: 2, 4: 4, 5: 5, 6: 6, 7: 8, 9: 15, 11: 25},
        # paper: all SIFs within 2.03e-3 of the reference on its finest mesh (1626 dof); K_II, K_III errors are
        # relative to their amplitude 4 sin cos sigma sqrt(pi R) / (pi (2 - nu)) (times (1 - nu) for K_III)
        "gates": {"KI_avg_err": 2.0e-3, "KII_avg_err": 2.0e-3, "KIII_avg_err": 2.0e-3},
        "paper": {"max_rel_err_finest": 2.03e-3, "finest_dof": 1626},
    },
    "ellipse": {
        "title": "Elliptical crack a = 2b, nu = 0.3, remote tension via crack-face pressure (paper Sec. 4.2)",
        "make": ["ellipse", "--layout", "{layout}"],
        "geometry": "ellipse_{prefix}i0_n{n}.dat",
        "levels": [3, 4, 5, 6, 7, 9, 11],
        "layouts": ["disk", "disk_graded"],
        "grade": "0.05,0.3,2",
        "formulations": {"xigabem": [], "igabem": ["--plain"]},
        "extra": ["--nu", "0.3", "--load", "pressure"],
        "est_s": {3: 3, 4: 4, 5: 6, 6: 8, 7: 11, 9: 18, 11: 30},
        "gates": {"KI_avg_err": 5.0e-3},
        "paper": {},
    },
    "ellipse_inclined": {
        "title": "Elliptical crack a = 2b inclined by pi/6, nu = 0.3, mixed mode (paper Sec. 4.2, Table 2)",
        "make": ["ellipse", "--layout", "{layout}", "--incline", "30"],
        "geometry": "ellipse_{prefix}i30_n{n}.dat",
        "levels": [3, 4, 5, 6, 7, 9, 11],
        "layouts": ["disk"],
        "formulations": {"xigabem": []},
        "extra": ["--nu", "0.3", "--load", "pressure"],
        "est_s": {3: 3, 4: 4, 5: 6, 6: 8, 7: 11, 9: 18, 11: 30},
        "gates": {"t2_KI_0": 1.0e-2, "t2_KII_0": 1.0e-2, "t2_KI_90": 1.0e-2, "t2_KIII_90": 1.0e-2},
        # Table 2, relative errors (XIGABEM; VCCI and M-integral of Peng et al. for comparison); dof not given
        "paper": {"table2": {"t2_KI_0": [5.53e-4, 4.56e-2, 1.53e-2], "t2_KII_0": [4.83e-3, 4.14e-2, 1.28e-2],
                             "t2_KI_90": [1.89e-3, 8.28e-3, 2.21e-2], "t2_KIII_90": [1.77e-3, 6.88e-3, 5.96e-2]}},
    },
    "prism_tension": {
        "title": "Prism with edge crack under uniform tension, E = 1000, nu = 0.3 (paper Sec. 4.4, Fig. 25)",
        "bin": "crack_tension",
        "make": ["prism_tension", "--lateral", "{layout}"],
        "geometry": "prismt_{layout}_c{n}.dat",
        "levels": [1, 2, 3, 4],
        "layouts": ["split", "aligned", "enriched", "split_graded"],
        "grade": "0.25,0.3,2",
        "formulations": {"xigabem": []},
        "extra": [],
        "est_s": {1: 5, 2: 12, 3: 22, 4: 34},
        # no closed form: the converged mid-front K_I of the split layout must agree with the aligned one (the two
        # discretise the lateral faces independently) and sit within 2% of the 2-D plane-strain value 2.829 (Eq. 30)
        "gates": {"KI_mid_diff": 2.0e-2},
        "gate_layouts": ["split", "aligned", "split_graded"],  # one enriched lateral patch converges slowly (shown, not gated)
        "paper": {"F_2d": 2.829, "fig25_mid_diff": {"a: crack enrichment only, 3996 dof": 2.0e-2,
                                                    "b: fully enriched, plane-strain kappa, 2787 dof": 5.2e-3,
                                                    "c: fully enriched, plane-stress kappa, 2787 dof": 9.6e-3}},
    },
    "prism_a3": {
        "title": "Prism with edge crack, mode I plus the r^3/2 Williams term (a3 = 1, exact K_I = 1): front grading",
        "bin": "crack_prism",
        "make": ["prism", "--lateral", "split"],
        "geometry": "prism_split_k{n}.dat",
        "levels": [1, 2, 3, 4, 5],
        "layouts": ["split", "split_graded"],
        "grade": "0.1,0.3,2",
        "formulations": {"xigabem": []},
        # --mint: interaction-integral K at 5 stations, annulus 0.2 (0.4 of the crack depth); --indicator: the TBIE
        # residual estimate eta against the crack energy error (effectivity)
        "extra": ["--a3", "1", "--mint", "0.2", "--indicator"],
        # residual-driven front grading from k = 2 (python/adapt_front.py), shown against the hand grading
        "post": ["adapt_front.py"],
        "est_s": {1: 10, 2: 15, 3: 25, 4: 35, 5: 45},
        # the K unknown absorbs the r^3/2 remainder at O(h) (review: 9.4% at shift 0.1, k = 3); grading cuts it ~10x.
        # KM_track_I: the gap |K_dof - K_M| over the true K_dof error (or its inverse), at most 3
        "gates": {"KI_err": 5.0e-3, "KM_track_I": 3.0, "effectivity": [0.3, 3.0]},
        "gate_layouts": ["split_graded"],
        "paper": {},
    },
    "prism": {
        "title": "Prism with edge crack, pure modes I, II, III from the Williams field (paper Sec. 4.3, Table 3)",
        "bin": "crack_prism",
        "make": ["prism"],
        "geometry": "prism_k{n}.dat",
        "levels": [1, 2, 3, 4, 5, 6],
        "layouts": ["discontinuous"],
        "formulations": {"xigabem": []},
        "extra": [],
        "est_s": {1: 1, 2: 2, 3: 2, 4: 3, 5: 5, 6: 8},
        # mean |K_M - 1| of each pure mode on the finest level (paper: below 4e-4 at 2175 dof), and the largest
        # spurious SIF (mean |K_N|, N != M)
        "gates": {"KI_err": 1.0e-4, "KII_err": 1.0e-4, "KIII_err": 1.0e-4, "cross_max": 1.0e-4},
        "paper": {"table3": {
            "I": {639: [1.00057, 5.84e-07, 2.66e-06], 1275: [0.99979, 9.66e-06, 2.95e-05], 2175: [0.99991, 9.17e-06, 1.96e-05]},
            "II": {639: [7.06e-07, 1.00125, 1.70e-02], 1275: [7.55e-05, 1.00050, 2.21e-03], 2175: [8.36e-07, 1.00032, 1.39e-04]},
            "III": {639: [7.19e-07, 7.06e-04, 0.99579], 1275: [1.51e-03, 7.92e-04, 1.00228], 2175: [6.81e-06, 2.74e-04, 0.99991]},
        }},
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
    binary = os.path.join(BUILD, c.get("bin", "crack_case"))
    plan = [(lay, n, f) for lay in c["layouts"] for n in c["levels"] for f in c["formulations"]
            if f == "xigabem" or not lay.endswith("_graded")]
    est = sum(c["est_s"][n] * (0.25 if f == "igabem" else 1.0) for lay, n, f in plan)
    print("%s: %d runs, estimated %.0f s cold (hard timeout %d s per run)" % (name, len(plan), est, timeout))
    runs = []
    for lay, n, f in plan:
        base, graded = (lay[:-7], True) if lay.endswith("_graded") else (lay, False)
        prefix = {"disk": "", "ogrid": "og_"}.get(base, "")
        name = c["geometry"].format(prefix=prefix, n=n, layout=base)
        if graded:
            name = name[:-4] + "_g.dat"
        geo = os.path.join(ROOT, "geo", "crack", name)
        if not os.path.exists(geo):
            extra = ["--grade", c["grade"], "--suffix", "_g"] if graded else []
            subprocess.run([sys.executable, os.path.join(ROOT, "python", "make_crack.py")] +
                           [a.format(layout=base) for a in c["make"]] + ["--n", str(n)] + extra, check=True,
                           stdout=subprocess.DEVNULL)
        tag = "%s_%s_n%d" % (f, lay, n)
        summ = os.path.join(out, tag + "_summary.json")
        if force or stale(summ, [geo, binary]):
            t0 = time.time()
            cmd = [binary, geo, "--out", os.path.join(out, tag)] + c["formulations"][f] + c["extra"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env=dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4")))
            if r.returncode != 0:
                raise RuntimeError("%s failed: %s" % (tag, r.stderr))
            print("  %-22s %6.1f s" % (tag, time.time() - t0))
        with open(summ) as fh:
            d = json.load(fh)
        d.update({"formulation": f, "layout": lay, "n": n, "tag": tag, "graded": graded})
        runs.append(d)
    # gates on the finest XIGABEM run of each layout
    verdict = {}
    for lay in c.get("gate_layouts", c["layouts"]):
        fin = max((r for r in runs if r["layout"] == lay and r["formulation"] == "xigabem"), key=lambda r: r["n"])
        # a gate is an upper bound, or a [lo, hi] range
        verdict[lay] = {k: {"value": fin[k], "gate": g,
                            "pass": (g[0] <= fin[k] <= g[1]) if isinstance(g, list) else fin[k] <= g}
                        for k, g in c["gates"].items()}
    with open(os.path.join(out, "case.json"), "w") as fh:
        json.dump({"case": name, "title": c["title"], "runs": runs, "gates": verdict, "paper": c["paper"]}, fh, indent=1)
    for script in c.get("post", []):
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "python", script)], capture_output=True, text=True,
                           timeout=timeout)
        print("  %-22s %6.1f s%s" % (script, time.time() - t0, "" if r.returncode == 0 else "  (target not met)"))
        print("".join("    " + l + "\n" for l in r.stdout.splitlines()), end="")
    ok = all(v["pass"] for g in verdict.values() for v in g.values())
    for lay, g in verdict.items():
        for k, v in g.items():
            gs = ("in [%.1g, %.1g]" % tuple(v["gate"])) if isinstance(v["gate"], list) else "<= %.1e" % v["gate"]
            print("  gate %-6s %-13s %.3e %s  %s" % (lay, k, v["value"], gs, "ok" if v["pass"] else "FAIL"))
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
    for n in a.cases:
        b = CASES[n].get("bin", "crack_case")
        if not os.path.exists(os.path.join(BUILD, b)):
            print("build %s first: cmake --build build --target %s" % (b, b), file=sys.stderr)
            return 2
    ok = all([run_case(n, a.force, a.timeout) for n in a.cases])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
