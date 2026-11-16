"""Compare the re-run PLA specimen solves with the stored extracts (ES: es2020/data/igabem_bisector_profiles.json,
YS: yaren-susmel-2025/igabem_fields_3d.json) and collect DOFs / timings from each bc_case report.
Usage: python check_pla_solves.py <es_profiles.json> <ys_fields_3d.json>"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
es_ref = json.load(open(sys.argv[1]))["profiles"]
ys_ref = json.load(open(sys.argv[2]))["specimens"]
TAG = {"ES:sharp": "edge_WG25_WN6_T3_r0.15_a35_L86", "ES:intermediate": "edge_WG25_WN6_T3_r1_a0_L86",
       "ES:blunt": "edge_WG25_WN6_T3_r3_a0_L86", "YS:plain": "plain", "YS:V_r0.15": "V_r0.15", "YS:U_r1": "U_r1",
       "YS:U_r3": "U_r3"}
EARLIER = {"ES:sharp": "45.6 s (20 s operators, 16 s LU), 12528 unknowns", "ES:intermediate": "n/a, 8994 unknowns",
           "ES:blunt": "n/a, 7302 unknowns", "YS:plain": "10 s, 7092 unknowns", "YS:V_r0.15": "38 s, 12588 unknowns",
           "YS:U_r1": "19 s, 9258 unknowns", "YS:U_r3": "15 s, 7806 unknowns"}
out = {}
for key, tag in TAG.items():
    src, name = key.split(":")
    if src == "ES":
        f = np.load(f"{ROOT}/cache/es2020/{tag}_b1.5_line.npz")
        d, s1 = f["d"], f["s1_line"]
        ref = es_ref[name]
        rd, rs, unk_ref = np.array(ref["d"]), np.array(ref["s1"]), ref["unknowns"]
        rep = open(glob.glob(f"{ROOT}/cache/es2020/run_{tag}_h1.5/report.txt")[0]).read().splitlines()
    else:
        f = np.load(f"{ROOT}/cache/ys2025/profile_{name}_h0.npz")
        d, s1 = f["d"], f["s1"]
        ref = ys_ref[name]["bisector"]
        rd, rs, unk_ref = np.array(ref["d"]), np.array(ref["s1"]), ys_ref[name]["unknowns"]
        rep = open(glob.glob(f"{ROOT}/cache/ys2025/run_{name}_h0/report.txt")[0]).read().splitlines()
    new = np.array([np.interp(rd, d, row) for row in s1])
    rel = np.abs(new - rs) / np.maximum(np.abs(rs), 1e-12)
    unk = int(f["unknowns"])
    out[key] = dict(unknowns=unk, unknowns_earlier=unk_ref, Kt_root_mid=float(s1[s1.shape[0] // 2 if src == "ES" else -1, 0]),
                    max_rel_diff_profile=float(rel.max()), max_rel_diff_d_ge_0p1=float(rel[:, rd >= 0.1].max()),
                    earlier=EARLIER[key], report=[ln for ln in rep if any(w in ln.lower() for w in
                                                  ("unknowns", "assembl", "operator", "lu", "factor", "solve", "wall", "time", " s"))])
json.dump(out, open("check_pla_solves.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["unknowns"], v["unknowns_earlier"], f"Kt {v['Kt_root_mid']:.4f}", f"maxrel {v['max_rel_diff_profile']:.2e}")
