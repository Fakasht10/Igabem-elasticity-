"""Benchmark stl_repair_core against pinned Sutura 0.9.8 on identical inputs.

usage: python tools/compare_corpus.py SUTURA_CHECKOUT OUT.json [--jobs 3] [--timeout 240] [--only a,b]

For every STL in SUTURA_CHECKOUT/tests/real-world-samples (CC BY / CC0 per
ATTRIBUTION.md there; the meshes are read in place, never copied):
  ours      stl_repair_core.repair_stl with the recommended defaults and
            max_hole_diameter = 5 % of the bounding-box diagonal
  s1        Sutura Stage-1 chain (stage1.apply_chain(stage1_chain())), the
            part of Sutura this package extracts
  full      the Sutura 0.9.8 CLI in its default mode (Stage 1 + triage +
            Stage 2 solid rebuild), output written to a temp file
Every output is measured with the SAME yardstick (stl_repair_core.diagnose,
manifold3d status/volume, vertex-sampled two-sided distance to the input) so
no tool grades itself.
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HOLE_FRAC = 0.05


def _measure(V, F, mesh0, idx0, tag):
    from stl_repair_core import geometry as G
    from stl_repair_core.diagnose import diagnose
    from stl_repair_core.mesh import Mesh, canonical
    m = canonical(Mesh(np.asarray(V, float), np.asarray(F, np.int64), np.arange(len(F)),
                       np.zeros(len(F), np.uint16), np.zeros(len(F), np.int32)))
    d = diagnose(m)
    si = d.get("self_intersections", {})
    out = {k: d.get(k) for k in ("faces", "boundary_edges", "boundary_loops", "nonmanifold_edges",
                                 "nonmanifold_vertices", "components", "healthy_components", "watertight",
                                 "degenerate_faces", "duplicate_faces", "signed_volume", "surface_area")}
    out.update({"si_proper": si.get("proper"), "si_coplanar": si.get("coplanar_overlap"),
                "si_touching": si.get("touching")})
    if idx0 is not None and m.nf:
        # vertex samples both ways (exact point-to-triangle distances)
        d_out = G.SurfaceIndex(mesh0.V, mesh0.F).distance(np.unique(m.V[m.F].reshape(-1, 3), axis=0))
        d_in = G.SurfaceIndex(m.V, m.F).distance(np.unique(mesh0.V[mesh0.F].reshape(-1, 3), axis=0))
        out["dist_out_to_in_max"] = float(d_out.max())
        out["dist_in_to_out_max"] = float(d_in.max())
        out["dist_out_to_in_p99"] = float(np.percentile(d_out, 99))
        out["dist_in_to_out_p99"] = float(np.percentile(d_in, 99))
    return out


def _manifold(path):
    try:
        import manifold3d as m3
        import trimesh
        tm = trimesh.load(path, force="mesh", process=True)
        man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                                  tri_verts=np.asarray(tm.faces, np.uint32)))
        ok = man.status() == m3.Error.NoError
        return {"manifold_ok": bool(ok), "manifold_volume": float(man.volume()) if ok else None}
    except Exception as exc:  # noqa: BLE001
        return {"manifold_ok": False, "manifold_error": str(exc)[:100]}


def run_one(args):
    path, repo, timeout = args
    sys.path.insert(0, os.path.join(repo, "sutura"))
    from stl_repair_core import repair_stl, stlio
    from stl_repair_core import geometry as G
    from stl_repair_core.mesh import canonical, from_soup
    name = os.path.basename(path)
    row = {"mesh": name}
    tmp = tempfile.mkdtemp(prefix="cmp_")
    soup = stlio.read_stl(path)
    mesh0 = canonical(from_soup(soup))
    diag = G.bbox_diagonal(mesh0.V)
    row["input"] = _measure(mesh0.V, mesh0.F, mesh0, None, "input")
    row["bbox_diagonal"] = diag
    # ---- ours
    t0 = time.perf_counter()
    out = os.path.join(tmp, "ours.stl")
    rep = repair_stl(path, out, max_hole_diameter=HOLE_FRAC * diag)
    row["ours"] = {"status": rep["status"], "runtime_s": time.perf_counter() - t0,
                   "adopted": [o["operation"] for o in rep["adopted_operations"]],
                   "rejected": [o["operation"] for o in rep["rejected_operations"]],
                   "problems": rep["problems"][:6], "displacement_max": (rep.get("deviation_final") or {}).get("displacement_max"),
                   "cumulative_abs_dvolume": rep["cumulative_abs_dvolume"],
                   "output_geometry_hash": rep["output"]["geometry_hash"]}
    s2 = stlio.read_stl(out)
    m2 = canonical(from_soup(s2))
    row["ours"].update(_measure(m2.V, m2.F, mesh0, True, "ours"))
    row["ours"].update(_manifold(out))
    # ---- ours with a wide hole limit (25 % of the diagonal) for a like-for-like closing comparison
    out25 = os.path.join(tmp, "ours25.stl")
    t0 = time.perf_counter()
    rep25 = repair_stl(path, out25, max_hole_diameter=0.25 * diag)
    m25 = canonical(from_soup(stlio.read_stl(out25)))
    row["ours25"] = {"status": rep25["status"], "runtime_s": time.perf_counter() - t0,
                     "adopted": [o["operation"] for o in rep25["adopted_operations"]],
                     "problems": rep25["problems"][:6]}
    row["ours25"].update(_measure(m25.V, m25.F, mesh0, True, "ours25"))
    # ---- Sutura stage 1 (in-process, its own reader)
    try:
        from sutura_engine.pipeline import stage1, mesh_io
        t0 = time.perf_counter()
        v, t = mesh_io.read_stl(path)
        v2, t2, applied, skipped = stage1.apply_chain(v, t, stage1.stage1_chain())
        dt = time.perf_counter() - t0
        p1 = os.path.join(tmp, "s1.stl")
        mesh_io.stl_write_binary(p1, np.asarray(v2, np.float32), np.asarray(t2))
        row["s1"] = {"runtime_s": dt, "applied": applied, "skipped": list(skipped)}
        row["s1"].update(_measure(np.asarray(v2, float), t2, mesh0, True, "s1"))
        row["s1"].update(_manifold(p1))
    except Exception as exc:  # noqa: BLE001
        row["s1"] = {"error": repr(exc)[:200]}
    # ---- Sutura full CLI
    pf = os.path.join(tmp, "full.stl")
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, os.path.join(repo, "sutura", "repair.py"), path, "-o", pf, "--json",
                            "--no-history", "--no-cache", "--no-learning-triage", "-y"],
                           capture_output=True, text=True, timeout=timeout)
        dt = time.perf_counter() - t0
        try:
            js = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            js = {}
        row["full"] = {"runtime_s": dt, "exit": r.returncode, "category": js.get("category"),
                       "issues": js.get("issues"), "methods": js.get("method_applied") or js.get("method")}
        if os.path.exists(pf):
            s3 = stlio.read_stl(pf)
            m3_ = canonical(from_soup(s3))
            row["full"].update(_measure(m3_.V, m3_.F, mesh0, True, "full"))
            row["full"].update(_manifold(pf))
    except subprocess.TimeoutExpired:
        row["full"] = {"timeout_s": timeout}
    except Exception as exc:  # noqa: BLE001
        row["full"] = {"error": repr(exc)[:200]}
    print("%-26s ours=%s %.1fs  s1 wt=%s  full=%s" % (name, row["ours"]["status"], row["ours"]["runtime_s"],
                                                     row.get("s1", {}).get("watertight"),
                                                     row.get("full", {}).get("category")), flush=True)
    return row


def main():
    repo, out = sys.argv[1], sys.argv[2]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 3
    timeout = int(sys.argv[sys.argv.index("--timeout") + 1]) if "--timeout" in sys.argv else 240
    d = os.path.join(repo, "tests", "real-world-samples")
    files = sorted(f for f in os.listdir(d) if f.endswith(".stl"))
    if "--only" in sys.argv:
        keep = sys.argv[sys.argv.index("--only") + 1].split(",")
        files = [f for f in files if any(k in f for k in keep)]
    t0 = time.perf_counter()
    rows = []
    part = out + ".partial.jsonl"
    with mp.Pool(jobs) as pool, open(part, "w") as pf:
        for row in pool.imap_unordered(run_one, [(os.path.join(d, f), repo, timeout) for f in files], chunksize=1):
            rows.append(row)
            pf.write(json.dumps(row, default=float) + "\n")
            pf.flush()
    rows.sort(key=lambda r: r["mesh"])
    rep = {"sutura_commit": subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip(),
           "hole_fraction_of_bbox_diagonal": HOLE_FRAC, "wall_s": time.perf_counter() - t0, "rows": rows}
    with open(out, "w") as f:
        json.dump(rep, f, indent=1, default=float)


if __name__ == "__main__":
    main()
