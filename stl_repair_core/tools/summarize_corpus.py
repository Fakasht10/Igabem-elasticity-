"""Markdown tables from benchmarks/corpus_comparison.json.

usage: python tools/summarize_corpus.py benchmarks/corpus_comparison.json > benchmarks/corpus_comparison.md
"""
import json
import sys


def g(d, k, fmt="%s"):
    v = d.get(k)
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return fmt % v
    return str(v)


def main():
    rep = json.load(open(sys.argv[1]))
    rows = rep["rows"]
    print("Sutura commit `%s`; hole limit for `ours` = %.0f %% of bbox diagonal (`ours25`: 25 %%). Wall time %.0f s."
          % (rep["sutura_commit"][:7], 100 * rep["hole_fraction_of_bbox_diagonal"], rep["wall_s"]))
    print()
    print("Every column was measured with the same yardstick (stl_repair_core diagnosis + manifold3d), never the tool's own claim.")
    print("Distances are vertex-sampled maxima as a fraction of the bbox diagonal (out→in: invented geometry, in→out: lost geometry).")
    print()
    print("| mesh | faces | input defects (loops/NMv/SI) | ours | ours wt | ours25 | Sutura S1 wt | S1 healthy comps | S1 SI | S1 out→in | S1 in→out | full wt | full SI | full out→in | full in→out | full s | ours s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    agg = {"ours_pass": 0, "ours25_pass": 0, "s1_wt": 0, "full_wt": 0, "full_timeout": 0, "n": 0,
           "s1_si_left": 0, "full_si_left": 0, "s1_comp_loss": 0, "full_comp_change": 0, "full_dev_gt_1e3": 0,
           "s1_dev_gt_1e3": 0, "ours_dev_gt_0": 0}
    for r in rows:
        i, o, o25, s1, fu = r["input"], r["ours"], r.get("ours25", {}), r.get("s1", {}), r.get("full", {})
        D = r["bbox_diagonal"]

        def rel(d, k):
            v = d.get(k)
            return "–" if v is None else "%.1e" % (v / D)
        agg["n"] += 1
        agg["ours_pass"] += o["status"] == "PASS"
        agg["ours25_pass"] += o25.get("status") == "PASS"
        agg["s1_wt"] += bool(s1.get("watertight"))
        agg["full_wt"] += bool(fu.get("watertight"))
        agg["full_timeout"] += "timeout_s" in fu
        agg["s1_si_left"] += (s1.get("si_proper") or 0) > 0
        agg["full_si_left"] += (fu.get("si_proper") or 0) > 0
        agg["s1_comp_loss"] += (s1.get("components") or 0) < i["components"]
        agg["full_comp_change"] += fu.get("components") is not None and fu.get("components") != i["components"]
        agg["s1_dev_gt_1e3"] += (s1.get("dist_out_to_in_max") or 0) / D > 1e-3 or (s1.get("dist_in_to_out_max") or 0) / D > 1e-3
        agg["full_dev_gt_1e3"] += (fu.get("dist_out_to_in_max") or 0) / D > 1e-3 or (fu.get("dist_in_to_out_max") or 0) / D > 1e-3
        agg["ours_dev_gt_0"] += (o.get("dist_out_to_in_max") or 0) > 0 or (o.get("dist_in_to_out_max") or 0) > 0
        print("| %s | %d | %d/%d/%d | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f |" % (
            r["mesh"].replace(".stl", ""), i["faces"], i["boundary_loops"], i["nonmanifold_vertices"], i["si_proper"] or 0,
            o["status"], g(o, "watertight"), o25.get("status", "–"), g(s1, "watertight"),
            g(s1, "healthy_components"), i["healthy_components"], g(s1, "si_proper"), rel(s1, "dist_out_to_in_max"),
            rel(s1, "dist_in_to_out_max"), "timeout" if "timeout_s" in fu else g(fu, "watertight"), g(fu, "si_proper"),
            rel(fu, "dist_out_to_in_max"), rel(fu, "dist_in_to_out_max"), g(fu, "runtime_s", "%.0f"), o["runtime_s"]))
    print()
    print("**Totals (n=%d):** ours PASS %d (ours25 PASS %d); Sutura Stage 1 watertight %d; Sutura full watertight %d (timeouts %d)."
          % (agg["n"], agg["ours_pass"], agg["ours25_pass"], agg["s1_wt"], agg["full_wt"], agg["full_timeout"]))
    print("Proper self-intersections left in output: Stage 1 %d meshes, full %d meshes. Component count reduced by Stage 1: %d meshes; changed by full: %d meshes."
          % (agg["s1_si_left"], agg["full_si_left"], agg["s1_comp_loss"], agg["full_comp_change"]))
    print("Vertex-sampled deviation > 1e-3 x diagonal: Stage 1 %d meshes, full %d meshes; ours > 0 on %d meshes."
          % (agg["s1_dev_gt_1e3"], agg["full_dev_gt_1e3"], agg["ours_dev_gt_0"]))
    json.dump(agg, open(sys.argv[1].replace(".json", "_totals.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
