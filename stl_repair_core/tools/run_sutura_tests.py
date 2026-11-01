"""Run the Sutura 0.9.8 tests that cover the extracted scope against the
pinned checkout and record exact pass/fail counts and timings.

usage: python tools/run_sutura_tests.py SUTURA_CHECKOUT OUT.json [--timeout 300]
Each suite is a plain script (Sutura has no test framework); exit code 0 = pass.
Needs the Sutura Rust extension (sutura_geom) importable in this interpreter.
"""
import json
import os
import subprocess
import sys
import time

RELEVANT = [
    # intake / IO
    "test_mesh_io.py", "test_units.py", "test_adversarial.py", "test_cli_hardening.py", "test_reload_n2m.py",
    # diagnosis
    "test_defects.py", "test_diagnostics.py", "test_diagnosis_v2.py", "test_diagnosis_determinism.py",
    "test_order_independence.py", "test_connected_components.py", "test_surface_metrics.py",
    "test_si_definition.py", "test_si_caps.py", "test_validate.py",
    # conservative repair kernels
    "test_topology_parity.py", "test_hole_parity.py", "test_closing.py", "test_pinched_vertices.py",
    "test_adopt_guard.py", "test_policy_adoption.py", "test_policy_stage1.py",
    # dependency guards
    "test_no_scipy.py", "test_no_trimesh.py", "test_no_runtime_pymeshlab.py", "test_engine_no_pymeshlab.py",
]


def main():
    repo, out = sys.argv[1], sys.argv[2]
    timeout = int(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 300
    tdir = os.path.join(repo, "tests")
    rows = []
    for name in RELEVANT:
        path = os.path.join(tdir, name)
        if not os.path.exists(path):
            rows.append({"test": name, "result": "missing"})
            continue
        t0 = time.perf_counter()
        try:
            r = subprocess.run([sys.executable, path], cwd=tdir, capture_output=True, text=True, timeout=timeout,
                               env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
            res = "pass" if r.returncode == 0 else "fail"
            tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            res, tail = "timeout", []
        dt = time.perf_counter() - t0
        rows.append({"test": name, "result": res, "seconds": round(dt, 2), "tail": tail})
        print("%-34s %-8s %7.2f s" % (name, res, dt), flush=True)
    summary = {k: sum(r["result"] == k for r in rows) for k in ("pass", "fail", "timeout", "missing")}
    rep = {"sutura_commit": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip(),
           "python": sys.version.split()[0], "summary": summary, "tests": rows}
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(summary)


if __name__ == "__main__":
    main()
