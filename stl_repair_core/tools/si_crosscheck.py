"""Cross-check the exact self-intersection diagnosis against two independent
implementations on the corpus inputs: PyMeshLab/VCG (float, per face) and
Sutura's own exact Rust scan (sutura_geom.si_faces_exact, same contract).

usage: python tools/si_crosscheck.py SUTURA_CHECKOUT OUT.json
"""
import json
import os
import sys
import time

import numpy as np


def main():
    repo, out = sys.argv[1], sys.argv[2]
    import pymeshlab
    from stl_repair_core import intersect as X
    from stl_repair_core import stlio
    from stl_repair_core.mesh import canonical, from_soup
    try:
        import sutura_geom
    except ImportError:
        sutura_geom = None
    d = os.path.join(repo, "tests", "real-world-samples")
    rows = []
    for f in sorted(x for x in os.listdir(d) if x.endswith(".stl")):
        p = os.path.join(d, f)
        m = canonical(from_soup(stlio.read_stl(p)))
        t0 = time.perf_counter()
        si = X.self_intersections(m.V, m.F)
        t_ours = time.perf_counter() - t0
        ours_proper = set(np.unique(si["proper"]).tolist())
        ours_any = ours_proper | set(np.unique(si["coplanar_overlap"]).tolist()) | set(np.unique(si["touching"]).tolist())
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(p)
        ms.compute_selection_by_self_intersections_per_face()
        ml = int(ms.current_mesh().selected_face_number())
        row = {"mesh": f, "ours_proper_faces": len(ours_proper), "ours_any_si_faces": len(ours_any),
               "pymeshlab_si_faces": ml, "ours_seconds": round(t_ours, 3)}
        if sutura_geom is not None:
            mask, cnt = sutura_geom.si_faces_exact(m.V, m.F.astype(np.int32))
            sf = set(np.flatnonzero(np.asarray(mask, bool)).tolist())
            row["sutura_exact_faces"] = len(sf)
            row["sutura_exact_equals_ours_proper"] = sf == ours_proper
            row["sym_diff"] = len(sf ^ ours_proper)
        rows.append(row)
        print(row, flush=True)
    agree = sum(r.get("sutura_exact_equals_ours_proper", False) for r in rows)
    both_zero = sum((r["ours_proper_faces"] == 0) == (r["pymeshlab_si_faces"] == 0) for r in rows)
    rep = {"rows": rows, "sutura_exact_equal_count": agree, "zero_vs_nonzero_agreement_with_pymeshlab": both_zero,
           "n": len(rows)}
    json.dump(rep, open(out, "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "rows"})


if __name__ == "__main__":
    main()
