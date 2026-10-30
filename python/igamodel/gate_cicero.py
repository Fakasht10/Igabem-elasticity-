"""Phase-1 gate: the Cicero et al. (2020) SENB failure loads through the model tree
must equal python/failure/validate_cicero.py's.

Each radius is built as a tree by checked operations (cad, the case validate_cicero
writes, the material, one assessment at the notch root), saved as a model file,
loaded back, and run with igamodel.run.assess. The rows are compared with a
validate_cicero result file computed on the same machine.

usage (from python/):
  python3 -m failure.validate_cicero --work W --out D.json [--card ABS-M30-iso --sigma-uts 65]
  python3 -m igamodel.gate_cicero D.json --work W [--card ABS-M30-iso --sigma-uts 65] [--load-cases] [--submodel] [--rtol 1e-12]
"""
import argparse
import glob
import json
import os
import sys

from failure.validate_cicero import ROOT, ROOT_PT, STEP, card_props, case_for, paper_props

from . import Model, merge_case
from .run import assess


def tree(rho, props, card=None, sigma_uts=None):
    step = os.path.join(ROOT, "geo", "step", STEP[rho])
    m = Model()
    m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": step}})
    merge_case(m, case_for(step))
    m.apply({"op": "add", "parent": "root", "kind": "material", "props": {"E": props["E"], "nu": props["nu"]}})
    # validate_cicero's focus sphere is 1.5 L with the L it was given; L recomputed from K and sigma_0 differs
    # in the last bit (0.44000000000000006), which Gmsh turns into a different TET10 mesh at rho 0.25.
    a = {"label": "notch root", "point": list(ROOT_PT), "rho": rho, "focus_radius": 1.5 * props["L"]}
    a.update({"card": card, **({"sigma_uts": sigma_uts} if sigma_uts else {})} if card else
             {"K_Ic": props["K"], "sigma_uts": props["sigma_0"]})
    m.apply({"op": "add", "parent": "root", "kind": "assessment", "props": a})
    return m


def load_cases(rho, props, work):
    """Two load cases on one operator: 1 N and 3 N on the top pad. Linear elasticity, so the load factors differ
    by exactly 3, the failure loads in N agree, and bc_case solves once."""
    m = tree(rho, props)
    lc = m.apply({"op": "add", "parent": "root", "kind": "loadcase", "props": {"name": "bend3"}})
    top = m.of_kind("load")[0]
    m.apply({"op": "add", "parent": lc, "kind": "load", "region": top.region,
             "props": {**top.props, "force_N": 3.0}})
    r1, r3 = assess(m, ROOT, work)
    ratio = r1["factor_pm"] / r3["factor_pm"]
    same_N = abs(r3["failure_load_N"] / r1["failure_load_N"] - 1)
    print(f"load cases on one solve: factor ratio {ratio:.15f} (3), failure loads {r1['failure_load_N']:.4f} and "
          f"{r3['failure_load_N']:.4f} N, rel diff {same_N:.1e}; solve {r1['t_solve_s']} s, assess "
          f"{r1['t_assess_s']} + {r3['t_assess_s']} s", flush=True)
    return max(abs(ratio / 3 - 1), same_N), {"bend": r1, "bend3": r3}


def submodel(rho, props, work, half=2.0, h=1.0, curvature=12):
    """The same assessment on a box round the notch root, cut faces driven by the global solve. The box face
    nearest the root is `half` mm from it (> 4 L); the probe line runs along +y, so the box reaches 2 half above."""
    m = tree(rho, props)
    x, y, _ = ROOT_PT
    m.apply({"op": "add", "parent": "root", "kind": "submodel",
             "props": {"label": "notch", "lo": [x - half, y - half, -3.0], "hi": [x + half, y + half, 3.0], "h": h,
                       "curvature": curvature}})
    a = m.of_kind("assessment")[0]
    m.apply({"op": "set", "tag": a.tag, "key": "submodel", "value": "notch"})
    m.apply({"op": "set", "tag": a.tag, "key": "mesh_h", "value": min(2.5, half)})
    return assess(m, ROOT, work)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direct", help="validate_cicero result file (same machine)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--card")
    ap.add_argument("--sigma-uts", type=float)
    ap.add_argument("--rtol", type=float, default=1e-12)
    ap.add_argument("--load-cases", action="store_true", help="also check two load cases on one solve")
    ap.add_argument("--submodel", action="store_true",
                    help="also run the last radius on a submodel and compare with the whole part (0.2 %%)")
    ap.add_argument("--out")
    a = ap.parse_args()
    direct = json.load(open(a.direct))
    props = card_props(a.card, a.sigma_uts) if a.card else paper_props(direct["orientation"])
    geo_before = set(glob.glob(os.path.join(a.work, "geo_*.dat")))
    rows, worst = [], 0.0
    for d in direct["rows"]:
        rho = d["rho"]
        path = os.path.join(a.work, f"model_rho{rho}.json")
        tree(rho, props, a.card, a.sigma_uts).save(path)
        r = assess(Model.load(path), ROOT, a.work)[0]
        err = {k: abs(r[k] / d[dk] - 1) for k, dk in (("factor_pm", "predicted_N"),
                                                       ("factor_energy_pm", "energy_pm_predicted_N"),
                                                       ("s1_L2_per_case", "s1_L2_per_N"))}
        worst = max(worst, *err.values())
        print(f"rho {rho:<5} PM {r['factor_pm']:9.4f} N (direct {d['predicted_N']:9.4f})  "
              f"Energy PM {r['factor_energy_pm']:9.4f} N (direct {d['energy_pm_predicted_N']:9.4f})  "
              f"max rel diff {max(err.values()):.1e}  ({r['t_solve_s']} s solve, {r['t_assess_s']} s assess)",
              flush=True)
        rows.append({"rho": rho, "tree": r, "direct_predicted_N": d["predicted_N"],
                     "direct_energy_pm_N": d["energy_pm_predicted_N"], "rel_diff": err})
    cases = None
    if a.load_cases:
        e, cases = load_cases(direct["rows"][-1]["rho"], props, a.work)
        worst = max(worst, e)
    sub = None
    if a.submodel:
        d = direct["rows"][-1]
        sub = submodel(d["rho"], props, a.work)
        e = abs(sub["factor_pm"] / d["predicted_N"] - 1)
        print(f"submodel rho {d['rho']}: PM {sub['factor_pm']:.4f} N vs whole part {d['predicted_N']:.4f} N "
              f"({100 * e:+.3f} %), {sub['unknowns']} unknowns, {sub['t_submodel_solve_s']} s", flush=True)
        if e > 2e-3:
            worst = max(worst, e)
    new_geo = set(glob.glob(os.path.join(a.work, "geo_*.dat"))) - geo_before
    print(f"geometry cache: {'hit for every radius (same step_auto arguments)' if not new_geo else f'{len(new_geo)} new layouts'}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"props": props, "rtol": a.rtol, "worst": worst, "new_layouts": len(new_geo), "rows": rows,
                   "load_cases": cases, "submodel": sub},
                      f, indent=1)
    ok = worst <= a.rtol and not new_geo
    print(f"gate {'passed' if ok else 'FAILED'}: worst relative difference {worst:.1e} (rtol {a.rtol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
