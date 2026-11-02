"""M3: the machined-notch bead window (rve.notch) driven by the SENB part (Cicero et al. 2020, ABS, 0/90).

The window covers the notch root: cells of Cicero's 0.4 x 0.3 mm bead lattice, the notch (disk + slot, machined:
it cuts through the beads) trimmed out of them, notch centre at the window origin and root at (0, rho). It sits at
the part's root (ROOT_PT, bisector +y) with the slab at mid-thickness (z-periodic, the part's through-thickness
jumps applied as in rve.coupling). One-way drive u + chi : eps on the box faces (field+chi, the consistent drive);
drive points on the notch surface take the part's boundary recovery. Outputs: the weld tractions (lateral and
interlayer) and the filament-joint tractions (axial filament stress) around the root, per newton of SENB load.

usage (from python/): python3 -m rve.notch_coupling [--rho 1] [--cells 8,6] [--below 1] [--kappa k]
  runtime: part ~1-2 min (cached LU), window setup ~1 min first time (cached operators), solve ~1-2 min
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from failure.igabem_backend import ROOT  # noqa: E402
from rve import notch, window  # noqa: E402
from rve.coupling import E_PART, NU, ROOT_PT, part, strain  # noqa: E402


class NotchWindow:
    """Trimmed bead window around the notch root; global = o + window coordinates."""

    def __init__(self, rho, cells=(8, 6), below=1, refine=0, kappa=None, g=notch.CICERO, work="/tmp/rve_notch", phase=None):
        self.rho, self.g, self.work = rho, g, work
        w, h = g["w"], g["h"]
        self.info = notch.window(g, rho, cells, below, phase=phase, refine=refine, work=work)
        files = list(self.info["canon"])
        self.o = np.asarray(ROOT_PT, float) - np.array([0.0, rho, h / 2])  # notch centre; slab centre at mid-thickness
        # bead modulus: the periodic cell's in-plane E1 equals the part's (coupon) modulus
        Tc = os.path.join(work, "calib_r%d.json" % refine)
        subprocess.run([window.strength.BEAD_CELL] + files + ["--axes", "x,y", "--copy", "1:0:90", "--E", "1",
                        "--nu", str(NU), "--cache", os.path.join(work, "cache"), "--json", Tc]
                       + window.law_args(kappa, 1.0, NU, w, False), check=True, stdout=subprocess.DEVNULL)
        with open(Tc) as f:
            self.cell = json.load(f)
        self.E_bead = E_PART / self.cell["eng"]["E1"]
        self.base = [window.BEAD_WINDOW] + files + ["--pieces", self.info["manifest"], "--pitch", str(w),
                                                    "--E", "%.10g" % self.E_bead, "--nu", str(NU),
                                                    "--cache", os.path.join(work, "cache")]
        self.base += window.law_args(kappa, self.E_bead, NU, w, True) if kappa is not None else []
        pts = os.path.join(work, "drive_points.txt")
        subprocess.run(self.base + ["--points-out", pts], check=True, stdout=subprocess.DEVNULL)
        P = np.loadtxt(pts)
        d, t = P[:, 0] == 0, P[:, 0] == 1
        self.keys, self.x = P[d, 1:3].astype(int), P[d, 3:6]
        self.tkeys, self.tx = P[t, 1:3].astype(int), P[t, 3:6]
        self.Lz = 2 * h
        self.square = (w / g["b_int"]) ** 2

    def notch_distance(self, X):
        """Signed distance of window points to the notch surface (> 0 in the material)."""
        x, y = X[:, 0], X[:, 1]
        return np.where(y >= 0, np.hypot(x, y) - self.rho, np.abs(x) - self.rho)

    def solve(self, u, eps, tjump, tag="m3", probe=None):
        """Tractions (BeadWindow::tractions rows) of Q drives u + chi:eps (u (n, 3, Q), eps (n, 6, Q))."""
        Q = u.shape[-1]
        f = os.path.join(self.work, tag + "_field.txt")
        d = np.concatenate([u, eps], 1).transpose(0, 2, 1).reshape(len(u), 9 * Q)
        j = np.concatenate([tjump, np.zeros((len(tjump), 6, Q))], 1).transpose(0, 2, 1).reshape(len(tjump), 9 * Q)
        np.savetxt(f, np.vstack([np.column_stack([np.zeros(len(u)), self.keys, d]),
                                 np.column_stack([np.ones(len(tjump)), self.tkeys, j])]),
                   fmt=["%d", "%d", "%d"] + ["%.17g"] * (9 * Q))
        t = os.path.join(self.work, tag + "_tractions.txt")
        extra = []
        if probe is not None:                                           # interior stress at points (window frame)
            pin, self.probe_out = os.path.join(self.work, tag + "_probe_in.txt"), os.path.join(self.work, tag + "_probe.txt")
            np.savetxt(pin, probe, fmt="%.17g")
            extra = ["--probe", pin, self.probe_out]
        r = subprocess.run(self.base + ["--drive", "field+chi:" + f, "--tractions", t] + extra, check=True,
                           capture_output=True, text=True)
        self.report = r.stdout + r.stderr
        return np.loadtxt(t)


def run(rho=1.0, cells=(8, 6), below=1, refine=0, kappa=None, work="/tmp/rve_notch", phase=None, probe=None, log=print):
    t0 = time.time()
    tm = {}
    win = NotchWindow(rho, cells, below, refine, kappa, work=work, phase=phase)
    tm["window_setup"] = time.time() - t0
    X = win.x + win.o
    Xt = win.tx + win.o
    on = np.abs(win.notch_distance(win.x)) < 1e-7                         # drive points on the notch flank
    res = dict(rho=rho, cells=list(cells), below=below, refine=refine, kappa=kappa, E_bead=win.E_bead,
               phase=list(win.info["phase"]), n_drive=int(len(X)), n_on_notch=int(on.sum()), t=tm)
    with part(rho, os.path.abspath(os.path.join(ROOT, "cache", "failure")), log=log) as P:
        tm["part"] = time.time() - t0 - tm["window_setup"]
        t1 = time.time()
        Xs = np.vstack([X, Xt, Xt + [0, 0, win.Lz]])
        F = P.fields_all(Xs, on_surface=np.r_[on, np.zeros(2 * len(Xt), bool)])[P.cases[0]]
        if probe is not None:                                           # homogeneous part at the same points
            res["probe_part"] = P.fields_all(np.asarray(probe) + win.o)[P.cases[0]][1].reshape(-1, 9).tolist()
        tm["part_eval"] = time.time() - t1
    wind = F[2]
    inside = np.r_[wind[:len(X)][~on], wind[len(X):]]
    if np.any(inside < 0.99):
        raise RuntimeError("window drive points outside the part (min winding %.3f)" % inside.min())
    n1, nt = len(X), len(Xt)
    u, S = F[0][:n1], F[1]
    eps = strain(S[:n1, ..., None])[..., 0]
    tj = win.square * (S[n1 + nt:, :, 2] - S[n1:n1 + nt, :, 2])
    t1 = time.time()
    T = win.solve(u[..., None], eps[..., None], tj[..., None], probe=probe)
    if probe is not None:
        res["probe"] = np.loadtxt(win.probe_out).tolist()
    tm["window_solve"] = time.time() - t1
    np.save(os.path.join(work, "m3_tractions_rho%g_r%d_%dx%d.npy" % ((rho, refine) + tuple(cells))), T)
    res["faces"] = face_averages(T, win)
    # root summary: weld normal tractions and filament-joint axial stress, per N of SENB load
    kind, x, n, t = T[:, 0].astype(int), T[:, 3:6], T[:, 7:10], T[:, 10:13]
    tn = np.einsum("pc,pc->p", t, n)
    r = win.notch_distance(x)
    for k, name in ((0, "periodic_weld"), (1, "internal_weld"), (2, "filament_joint")):
        m = kind == k
        if m.any():
            i = np.argmax(tn[m])
            res[name] = dict(max_tn=float(tn[m][i]), at=x[m][i].tolist(), dist_to_notch=float(r[m][i]))
    tm["total"] = time.time() - t0
    log(win.report.strip())
    return res


def face_averages(T, win, top=5):
    """Mesh-robust weld and joint measures (the corner peaks where a weld neck meets the notch face are singular):
    per weld / joint face (kind, axis, + region, plane), the area-averaged normal traction (net bond stress) and the
    resultant normal force; the `top` faces of each kind by average, with their distance to the notch."""
    kind, ax, reg = T[:, 0].astype(int), T[:, 1].astype(int), T[:, 2].astype(int)
    x, W, n, t = T[:, 3:6], T[:, 6], T[:, 7:10], T[:, 10:13]
    tn = np.einsum("pc,pc->p", t, n)
    plane = np.round(x[np.arange(len(x)), ax], 6)
    keys = np.column_stack([kind, ax, reg, plane])
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    A = np.bincount(inv, W)
    Fn = np.bincount(inv, W * tn)
    xc = np.stack([np.bincount(inv, W * x[:, c]) for c in range(3)], 1) / A[:, None]
    d = win.notch_distance(xc)
    out = {}
    for k, name in ((0, "periodic_weld"), (1, "internal_weld"), (2, "filament_joint")):
        m = np.flatnonzero(u[:, 0] == k)
        o = m[np.argsort(-Fn[m] / A[m])][:top]
        out[name] = [dict(axis=int(u[i, 1]), avg_tn=float(Fn[i] / A[i]), area=float(A[i]), force=float(Fn[i]),
                          centre=xc[i].round(4).tolist(), dist_to_notch=float(d[i])) for i in o]
    return out


def tcd(rho=1.0, L=0.44, cells=(5, 5), below=-1, refine=0, nz=30, dd=0.005, work="/tmp/rve_notch", log=print):
    """TCD on the bisector from the bead window against the homogeneous part: hoop stress sxx (the bisector is +y)
    on x = 0 at distance d from the root, per layer centre line (A z = 0, B z = h) and averaged through one
    two-layer period (voids count zero: the homogenized stress the part sees). Point method at L/2, line method
    over 2L. Per N of SENB load."""
    h = notch.CICERO["h"]
    d = np.arange(dd / 2, 2 * L + 1e-12, dd)
    zc = -h / 2 + (np.arange(nz) + 0.5) * (2 * h / nz)
    zs = np.r_[zc, 0.0, h]
    D, Z = np.meshgrid(d, zs, indexing="ij")
    probe = np.column_stack([np.zeros(D.size), rho + D.ravel(), Z.ravel()])
    res = run(rho, cells, below, refine, None, work, probe=probe, log=log)
    W = np.array(res.pop("probe")).reshape(len(d), len(zs), -1)
    Sp = np.array(res.pop("probe_part")).reshape(len(d), len(zs), 3, 3)
    sxx = np.where(W[..., 3] >= 0, W[..., 4], 0.0)
    prof = dict(d=d, part=Sp[:, -1, 0, 0], homog=sxx[:, :nz].mean(1), A=sxx[:, nz], B=sxx[:, nz + 1],
                solid=(W[:, :nz, 3] >= 0).mean(1))
    out = {}
    for k in ("part", "homog", "A", "B"):
        y = prof[k]
        out[k] = dict(PM=float(np.interp(L / 2, d, y)), LM=float(y.mean()))
    res["tcd"] = dict(L=L, **out)
    np.savez(os.path.join(work, "tcd_profile_rho%g.npz" % rho), **prof)
    return res, prof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--cells", default="8,6")
    ap.add_argument("--below", type=int, default=1)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--work", default="/tmp/rve_notch")
    ap.add_argument("--json", default=None)
    ap.add_argument("--phase", default=None, help="lattice phase px,py (same for every refinement)")
    a = ap.parse_args(argv)
    res = run(a.rho, tuple(int(c) for c in a.cells.split(",")), a.below, a.refine, a.kappa, a.work,
              None if a.phase is None else tuple(float(p) for p in a.phase.split(",")))
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
