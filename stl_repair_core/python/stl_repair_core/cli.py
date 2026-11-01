# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Command line:  stl-repair-core validate|repair ..."""
from __future__ import annotations

import argparse
import json
import sys

from .repair import DEFAULTS, repair_stl, validate_stl


def _summary(rep):
    lines = ["%s: %s" % (rep.get("command"), rep.get("status"))]
    if rep.get("error"):
        lines.append("  error: " + rep["error"])
    for p in rep.get("problems", []):
        lines.append("  problem: " + p)
    for op in rep.get("adopted_operations", []):
        lines.append("  adopted: %s" % op["operation"])
    for op in rep.get("rejected_operations", []):
        lines.append("  rejected: %s (%s)" % (op["operation"], "; ".join(op.get("rejected_because", []))))
    if "runtime_s" in rep:
        lines.append("  runtime: %.2f s" % rep["runtime_s"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stl-repair-core", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="diagnose an STL without modifying it")
    v.add_argument("input")
    v.add_argument("--json", dest="json_path")
    v.add_argument("--weld-tolerance", type=float, default=DEFAULTS["weld_tolerance"],
                   help="near-coincident vertex report radius, fraction of bbox diagonal")
    v.add_argument("--allow-open", action="store_true", help="do not fail on boundary edges")
    r = sub.add_parser("repair", help="conservative gated repair")
    r.add_argument("input")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--json", dest="json_path")
    r.add_argument("--max-hole-diameter", type=float, default=DEFAULTS["max_hole_diameter"],
                   help="model units; loops with a larger diameter are intentional openings (0 = no filling)")
    r.add_argument("--weld-tolerance", type=float, default=DEFAULTS["weld_tolerance"],
                   help="fraction of the bounding-box diagonal (0 = exact coincidence only)")
    r.add_argument("--debris-area-threshold", type=float, default=DEFAULTS["debris_area_threshold"],
                   help="model units^2; unhealthy components below it are removed (0 = keep all)")
    r.add_argument("--max-surface-deviation", type=float, default=None,
                   help="model units; default 1e-5 x bbox diagonal")
    r.add_argument("--max-volume-change", type=float, default=DEFAULTS["max_volume_change"],
                   help="fraction of |volume|")
    r.add_argument("--max-hole-vertices", type=int, default=DEFAULTS["max_hole_vertices"])
    r.add_argument("--require-watertight", action="store_true",
                   help="FAIL (and return the input) unless the output is closed and manifold")
    r.add_argument("--protected", help="JSON file {region_name: [input facet ids]}")
    r.add_argument("--protect-tags", action="store_true",
                   help="protect binary attribute-word regions (geomio face ids) / ASCII solids")
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        rep = validate_stl(a.input, a.json_path, weld_tolerance=a.weld_tolerance, allow_open=a.allow_open)
    else:
        prot = None
        if a.protected:
            with open(a.protected) as f:
                prot = json.load(f)
        rep = repair_stl(a.input, a.output, a.max_hole_diameter, a.weld_tolerance, a.debris_area_threshold,
                         a.max_surface_deviation, a.max_volume_change, protected_faces=prot,
                         protect_tags=a.protect_tags, json_path=a.json_path, max_hole_vertices=a.max_hole_vertices,
                         require_watertight=a.require_watertight)
    print(_summary(rep))
    return 0 if rep.get("status") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
