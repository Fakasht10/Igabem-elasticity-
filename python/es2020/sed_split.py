"""Split the stored Lazzarin-Berto grid (old/cache/fatigue/*_sed.npz, full tensor) into (tr s)^2 and s_vM^2,
thickness mean, same points and order as igabem_fields.json notches[k]["sed"]. Pure post-process, no solve."""
import json
import sys

import numpy as np
F = json.load(open("es2020/data/igabem_fields.json"))["notches"]
tag = {"sharp": "edge_WG25_WN6_T3_r0.15_a35_L86", "intermediate": "edge_WG25_WN6_T3_r1_a0_L86", "blunt": "edge_WG25_WN6_T3_r3_a0_L86"}
nu = 0.35
out = {"source": "Post-processed from the stored full stress tensor on the Lazzarin-Berto polar grid "
       "(the same evaluation that produced igabem_fields.json sed: rc, dA, r0 identical, same point order); unit nominal net "
       "stress, thickness mean over z/T 0.125..0.875 of each quantity; void points dropped as there. "
       "tr2 = (tr sigma)^2, vm2 = sigma_vM^2; check: (1+nu) s:s - nu tr2 = (2(1+nu)/3) vm2 + ((1-2nu)/3) tr2 = w2E (nu 0.35). "
       "W_vol = (1-2nu)/(6E) tr2, W_dev = (1+nu)/(3E) vm2.", "nu": nu, "notches": {}}
for k, t in tag.items():
    d = np.load(f"/home/user/old/cache/fatigue/{t}_sed.npz")
    s = d["sig"].astype(float)                       # (rc, phi, z, 3, 3)
    tr = np.trace(s, axis1=-2, axis2=-1)
    dev = s - tr[..., None, None] / 3 * np.eye(3)
    vm2 = 1.5 * np.sum(dev ** 2, axis=(-2, -1))
    keep = d["wind"].mean(axis=2) > 0.5              # (rc, phi)
    T2, V2 = (tr ** 2).mean(axis=2)[keep], vm2.mean(axis=2)[keep]
    w = 2 * (1 + nu) / 3 * V2 + (1 - 2 * nu) / 3 * T2
    g = F[k]["sed"]
    ref = np.asarray(g["w2E"])
    print(k, len(w), len(ref), "max rel diff vs stored w2E", np.max(np.abs(w - ref) / np.abs(ref).max()) if len(w) == len(ref) else "LENGTH MISMATCH")
    rc = np.repeat(d["rc"][:, None], d["phi"].size, 1)[keep]
    print("   rc match", np.allclose(rc, g["rc"]) if len(rc) == len(g["rc"]) else False)
    out["notches"][k] = dict(r0=g["r0"], rc=g["rc"], dA=g["dA"], w2E=g["w2E"], tr2=T2.round(10).tolist(), vm2=V2.round(10).tolist())
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "es2020/data/igabem_sed_split.json", "w"))
