"""K_I of an SENB 3-point bend coupon from XIGABEM, two-mesh extrapolated, against the ASTM D5045 handbook K.

Geometry from make_crack.senb (made on the fly for the coupon's W, B, a/W; S = 4W). Load P as pressure on the
nose pad, reactions P/2 on each support pad (self-equilibrated); one end face clamped only to remove rigid
motion. K comes from crack_serve (one request per mesh level; the operator is cached, so several loads cost one
solve each). K converges O(h) here (measured on the penny and on SENB, see fracture-layer notes), with h ~ 1/n,
so two levels n1 < n2 give K_inf = (n2 K2 - n1 K1) / (n2 - n1); the gap K2 - K_inf is reported as the error
estimate. When the paper thread's front grading / end shift land, rerun: the extrapolation should then move
K by less.

    python/senb_k.py --W 10 --B 4 --a 5 --P 1 [--nu 0.35] [--levels 2,3]       one coupon
    python/senb_k.py --check                                                  ctest gate: nu 0, a/W 0.5

K units: P in N and lengths in mm give MPa sqrt(mm). Values: K_I at mid-thickness and averaged over the front
(D5045 is a plane-strain, thickness-averaged formula; the BEM K drops toward the free surfaces).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_crack as mc  # noqa: E402
import make_geometries as mg  # noqa: E402

SERVE = os.environ.get("IGABEM_CRACK_SERVE", os.path.join(HERE, "..", "build", "crack_serve"))


def k_handbook(P, B, W, a, S=None):
    """ASTM D5045 / E399 SENB, S = 4W: K = P / (B sqrt W) f(a/W)."""
    x = a / W
    f = 6 * math.sqrt(x) * (1.99 - x * (1 - x) * (2.15 - 3.93 * x + 2.7 * x * x)) / ((1 + 2 * x) * (1 - x) ** 1.5)
    return P / (B * math.sqrt(W)) * f


def k_level(n, W, B, a, P, nu, samples=64, workdir=None):
    """(K_I mid-thickness, thickness mean, thickness RMS, unknowns, seconds, K_I(z) profile) at mesh level n.

    The server samples the straight front at cell midpoints uniform in z, so the plain mean is the midpoint rule
    for (1/B) int K dz (64 samples: within 1e-3 of the trapezoid rule)."""
    patches, meta = mc.senb(n, W=W, B=B, a_W=a / W)
    d = workdir or tempfile.mkdtemp(prefix="senb_")
    tag = "senb_W%g_B%g_a%g_n%d" % (W, B, a, n)
    geo = os.path.join(d, tag + ".dat")
    mg.write_dat(geo, patches, tag)
    with open(os.path.join(d, tag + ".crack.json"), "w") as f:
        json.dump(meta, f)
    g = meta["patches"]
    q = 0.5 * P / (meta["pad"] * B)  # reaction per support pad P/2 over pad x B
    rq = {
        "cmd": "solve", "geometry": geo, "material": {"nu": nu}, "samples": samples,
        "options": {"discontinuous": True},
        "clamp": [{"patches": g["ends"][:1]}],
        "loads": [{"name": "P", "tractions": [{"patches": g["supports"], "t": [0, q, 0]},
                                              {"patches": g["nose"], "t": [0, -2 * q, 0]}]}],
    }
    case = os.path.join(d, tag + ".json")
    with open(case, "w") as f:
        json.dump(rq, f)
    out = subprocess.run([SERVE, "--case", case], capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    if not r.get("ok"):
        raise RuntimeError(r.get("error"))
    fr = sorted(r["front"], key=lambda f: f["x"][2])
    KI = [f["K"][0][0] for f in fr]
    i = r["info"]
    mid = 0.5 * (KI[(len(KI) - 1) // 2] + KI[len(KI) // 2])
    return (mid, sum(KI) / len(KI), math.sqrt(sum(k * k for k in KI) / len(KI)), i["unknowns"],
            sum(i.get(k) or 0.0 for k in ("t_assemble", "t_factor", "t_rhs")), [(f["x"][2], f["K"][0][0]) for f in fr])


def coupon(W, B, a, P, nu=0.35, levels=(2, 3), verbose=True):
    """Two-level extrapolated K_I with the handbook value and the error estimate.

    K_mid      mid-thickness K_I
    K_mean     thickness average (1/B) int K_I dz
    K_rms      sqrt of the thickness average of K_I^2 (energy average: G per unit thickness ~ K^2)
    K_G        K_rms sqrt(1 - nu^2): the 2D plane-stress K with the same energy release per unit thickness. The 3D
               K(z) is plane strain inside the bar and the bar's compliance is near plane stress, so a thin bar's
               thickness-average K sits above the 2D handbook K by up to 1/sqrt(1 - nu^2) (1.067 at nu 0.35);
               K_G is the like-for-like number for D5045, whose formula is 2D.
    """
    n1, n2 = levels
    rows = {n: k_level(n, W, B, a, P, nu) for n in (n1, n2)}
    keys = ("K_mid", "K_mean", "K_rms")
    ext = {k: (n2 * rows[n2][j] - n1 * rows[n1][j]) / (n2 - n1) for j, k in enumerate(keys)}
    ext["K_G"] = ext["K_rms"] * math.sqrt(1 - nu * nu)
    K0 = k_handbook(P, B, W, a)
    res = {**ext, "K_D5045": K0, "err_est_mean": abs(rows[n2][1] - ext["K_mean"]),
           "err_est_mid": abs(rows[n2][0] - ext["K_mid"]), "profile": rows[n2][5],
           "levels": {n: {**{k: v[j] for j, k in enumerate(keys)}, "unknowns": v[3], "s": round(v[4], 2)}
                      for n, v in rows.items()}}
    if verbose:
        for n, v in rows.items():
            print(f"  n {n}: {v[3]:6d} unknowns {v[4]:6.1f} s   / K_D5045: mid {v[0] / K0:.4f}  mean {v[1] / K0:.4f}  rms {v[2] / K0:.4f}")
        print(f"  extrapolated / K_D5045: mid {ext['K_mid'] / K0:.4f}  mean {ext['K_mean'] / K0:.4f}  rms {ext['K_rms'] / K0:.4f}"
              f"  K_G {ext['K_G'] / K0:.4f}   (est. error of the mean {res['err_est_mean'] / K0:.2%})")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--W", type=float, default=10.0)
    ap.add_argument("--B", type=float, default=4.0)
    ap.add_argument("--a", type=float, default=5.0)
    ap.add_argument("--P", type=float, default=1.0)
    ap.add_argument("--nu", type=float, default=0.35)
    ap.add_argument("--levels", default="2,3")
    ap.add_argument("--check", action="store_true", help="gate: nu 0, a/W 0.5, levels 2,3; |K_mean/K_D5045 - 1| < 2 %% (thickness average)")
    a = ap.parse_args(argv)
    if a.check:
        r = coupon(10.0, 4.0, 5.0, 1.0, nu=0.0, levels=(2, 3))
        dev = r["K_mean"] / r["K_D5045"] - 1
        print("PASS" if abs(dev) < 0.02 else "FAIL", f"{dev:+.3%}")
        return 0 if abs(dev) < 0.02 else 1
    r = coupon(a.W, a.B, a.a, a.P, a.nu, tuple(map(int, a.levels.split(","))))
    print(json.dumps({k: v for k, v in r.items() if k not in ("levels", "profile")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
