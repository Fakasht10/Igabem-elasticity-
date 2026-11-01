"""Global-local coupling of the bead window (rve.window) with the part IgBEM (failure.igabem_backend.Solve).

The part is solved once, homogenized (isotropic, as failure.cicero_orient), and kept alive (bc_case --serve). The
window (nx x ny 0/90 cells, z-periodic slab at mid-thickness, A filaments along the part x: layup 0/90, build
direction z) is placed at `centre` and driven on its cut faces by the part solution:
  field       u = u_part
  field+chi   u = u_part + chi : eps_part    (chi: the periodic cell's fluctuation; consistent, gate G2)
The window's reaction on each cut face is compared with the part's traction resultant over the same face (the full
w x 2h cell face, the part being homogeneous there): the force mismatch per face measures what a two-way coupling
would feed back (non-invasive Gendre et al. 2009: the mismatch is the extra load on the part at the window
boundary). Below a few percent the one-way drive is the converged two-way answer within that bound.

usage (from python/): python3 -m rve.coupling [--rho 1] [--d 1.5] [--cells 3] [--kappa k]
  runtime: ~1-2 min per radius for the part (once per run), ~10 s per window solve
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from failure.igabem_backend import ROOT, Solve, igabem_geometry  # noqa: E402
from failure.validate_cicero import STEP, case_for  # noqa: E402
from rve import window  # noqa: E402

E_PART, NU = 2241.0, 0.35
ROOT_PT = np.array([22.0, 5.0, 0.0])     # SENB notch root, mid-span, mid-thickness; bisector +y


def part(rho, work, log=print):
    step = os.path.join(ROOT, "geo", "step", STEP[rho])
    dat, _ = igabem_geometry(step, work, h=3.0, curvature=6, degree=2, log=log)
    return Solve(dat, case_for(step), E_PART, NU, workdir=os.path.join(work, "run_cpl_%g" % rho), log=log)


def strain(sig, E=E_PART, nu=NU):
    """Voigt engineering strain (n, 6, ...) of isotropic stress tensors (n, 3, 3, ...)."""
    eps = ((1 + nu) * sig - nu * np.einsum("pii...->p...", sig)[:, None, None] * np.eye(3).reshape((3, 3) + (1,) * (sig.ndim - 3))) / E
    return np.stack([eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2], 2 * eps[:, 1, 2], 2 * eps[:, 0, 2], 2 * eps[:, 0, 1]], 1)


class Window:
    """A bead window placed in the part: origin o (global = o + window coordinates)."""

    def __init__(self, cells, centre, g=window.BASE, refine=1, grade=0, kappa=None, work="/tmp/rve_coupling"):
        self.cells, self.g, self.kappa, self.work = cells, g, kappa, work
        self.files = window.pieces(g, refine, grade, work)
        w, h = g["w"], g["h"]
        # cell (i, j) centred at (i w, j w); slab z in [-h/2, 3h/2] (centre h/2)
        self.o = np.asarray(centre, float) - np.array([(cells[0] - 1) * w / 2, (cells[1] - 1) * w / 2, h / 2])
        # bead modulus calibrated so the cell's in-plane modulus E1 equals the part's (the part's E is the coupon's,
        # porosity and welds included): one periodic-cell solve at E = 1
        Tc = os.path.join(work, "calib.json")
        subprocess.run([window.strength.BEAD_CELL] + self.files + ["--axes", "x,y", "--copy", "1:0:90", "--E", "1",
                        "--nu", str(NU), "--cache", os.path.join(work, "cache"), "--json", Tc]
                       + window.law_args(kappa, 1.0, NU, w, False), check=True, stdout=subprocess.DEVNULL)
        with open(Tc) as f:
            self.cell = json.load(f)
        self.E_bead = E_PART / self.cell["eng"]["E1"]
        self.base = [window.BEAD_WINDOW] + self.files + ["--cells", "%d:%d" % tuple(cells), "--pitch", str(w),
                                                         "--E", "%.10g" % self.E_bead, "--nu", str(NU),
                                                         "--cache", os.path.join(work, "cache")]
        self.base += window.law_args(kappa, self.E_bead, NU, w, True) if kappa is not None else []
        pts = os.path.join(work, "drive_points.txt")
        subprocess.run(self.base + ["--points-out", pts], check=True, stdout=subprocess.DEVNULL)
        P = np.loadtxt(pts)
        d, t = P[:, 0] == 0, P[:, 0] == 1
        self.keys, self.x = P[d, 1:3].astype(int), P[d, 3:6]
        self.tkeys, self.tx = P[t, 1:3].astype(int), P[t, 3:6]       # - side of the z-periodic welds (A bottom)
        self.Lz = 2 * h
        self.square = (w / g["b_int"]) ** 2                          # cell face area / interlayer square area

    def solve_many(self, u, eps, tjump, tag="cpl"):
        """Window tractions (BeadWindow::tractions rows, 3 per drive) of Q drives u + chi:eps with the z-periodic
        traction jumps (u (n, 3, Q), eps (n, 6, Q) at the drive points, tjump (m, 3, Q)), one batched solve."""
        Q = u.shape[-1]
        f = os.path.join(self.work, tag + "_field.txt")
        d = np.concatenate([u, eps], 1).transpose(0, 2, 1).reshape(len(u), 9 * Q)
        j = np.concatenate([tjump, np.zeros((len(tjump), 6, Q))], 1).transpose(0, 2, 1).reshape(len(tjump), 9 * Q)
        np.savetxt(f, np.vstack([np.column_stack([np.zeros(len(u)), self.keys, d]),
                                 np.column_stack([np.ones(len(tjump)), self.tkeys, j])]),
                   fmt=["%d", "%d", "%d"] + ["%.17g"] * (9 * Q))
        t = os.path.join(self.work, tag + "_tractions.txt")
        subprocess.run(self.base + ["--drive", "field+chi:" + f, "--tractions", t], check=True, stdout=subprocess.DEVNULL)
        return np.loadtxt(t)

    def faces(self, n=8):
        """Cut faces of the window boundary, one per boundary cell face: (axis, sign, cell index, Gauss points (m, 3)
        and weights over the full w x 2h face, window frame)."""
        w, h = self.g["w"], self.g["h"]
        nx, ny = self.cells
        gx, gw = np.polynomial.legendre.leggauss(n)
        out = []
        for a, (na, nb) in ((0, (nx, ny)), (1, (ny, nx))):
            for s in (-1, 1):
                for k in range(nb):
                    c = (na - 1) * w * (s > 0) + s * w / 2           # face plane along axis a
                    t = k * w + 0.5 * w * gx                         # along the other in-plane axis
                    z = h / 2 + h * gx                               # z in [-h/2, 3h/2]
                    T, Z = np.meshgrid(t, z, indexing="ij")
                    X = np.zeros((T.size, 3))
                    X[:, a], X[:, 1 - a], X[:, 2] = c, T.ravel(), Z.ravel()
                    W = np.outer(gw * w / 2, gw * h).ravel()
                    out.append(dict(axis=a, sign=s, k=k, x=X, w=W, centre=X.mean(0)))
        return out


def window_face_forces(T, faces, w):
    """Resultant reaction force (faces, 3, drives) per boundary cell face (cut-face rows, kind 3), on the window."""
    kind, ax = T[:, 0].astype(int), T[:, 1].astype(int)
    x, W, n = T[:, 3:6], T[:, 6], T[:, 7:10]
    t = T[:, 10:].reshape(len(T), -1, 3).transpose(0, 2, 1)             # (rows, 3, drives)
    out = []
    for F in faces:
        a, s = F["axis"], F["sign"]
        m = (kind == 3) & (ax == a) & (np.sign(n[:, a]) == s) & (np.abs(x[:, a] - F["centre"][a]) < 1e-6)
        m &= np.abs(x[:, 1 - a] - F["centre"][1 - a]) <= w / 2 + 1e-9
        out.append(np.einsum("p,pcq->cq", W[m], t[m]))
    return np.array(out)                                                # (faces, 3, drives)


def kelvin(X, Y, F, E=E_PART, nu=NU, chunk=40):
    """Displacement (n, 3) and stress (n, 3, 3) at X of point forces F at Y in the infinite isotropic body."""
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    A = 1.0 / (16 * np.pi * mu * (1 - nu))
    u = np.zeros((len(X), 3))
    S = np.zeros((len(X), 3, 3))
    I = np.eye(3)
    for a in range(0, len(X), chunk):
        r = X[a:a + chunk, None, :] - Y[None, :, :]                    # (p, q, 3)
        rn = np.linalg.norm(r, axis=2)
        rf = np.einsum("pqk,qk->pq", r, F)
        u[a:a + chunk] = A * np.einsum("pq,pqi->pi", 1 / rn, (3 - 4 * nu) * F[None] + r * (rf / rn ** 2)[..., None])
        G = A * (-(3 - 4 * nu) * np.einsum("qi,pqj->pqij", F, r) + rf[..., None, None] * I
                 + np.einsum("pqi,qj->pqij", r, F) - 3 * np.einsum("pqi,pqj,pq->pqij", r, r, rf / rn ** 2)) / rn[..., None, None] ** 3
        G = G.sum(1)
        eps = 0.5 * (G + G.transpose(0, 2, 1))
        S[a:a + chunk] = lam * np.trace(eps, axis1=1, axis2=2)[:, None, None] * I + 2 * mu * eps
    return u, S


def kelvin_unit(X, Y, wq, E=E_PART, nu=NU, chunk=512, stress=True):
    """Displacement (n, 3, 3) and stress (n, 3, 3, 3) at X of the point forces wq e_c at Y (last axis: c), summed over
    Y: the field of a unit traction on the panel quadrature (Y, wq), all three components at once (batched matmuls)."""
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    A = 1.0 / (16 * np.pi * mu * (1 - nu))
    k = 3 - 4 * nu
    I = np.eye(3)
    u = np.empty((len(X), 3, 3))
    S = np.empty((len(X), 3, 3, 3)) if stress else None
    for a in range(0, len(X), chunk):
        r = X[a:a + chunk, None, :] - Y[None, :, :]                         # (p, q, 3)
        ir = 1 / np.linalg.norm(r, axis=2)
        e = r * ir[..., None]                                               # unit vectors
        w1 = wq * ir                                                        # weights / r
        we = (e * w1[..., None]).transpose(0, 2, 1)                         # (p, 3, q)
        u[a:a + chunk] = A * (k * w1.sum(1)[:, None, None] * I + we @ e)
        if not stress:
            continue
        # grad_j u_i for F = e_c:  A [-k d_ic r_j + r_c d_ij + r_i d_jc - 3 r_i r_j r_c / r^2] / r^3
        we *= ir[:, None, :]                                                # weights e / r^2
        g1 = we.sum(2)                                                      # (p, 3): sum w e_j / r^2
        ee = (e[:, :, :, None] * e[:, :, None, :]).reshape(len(r), -1, 9)   # e_i e_j
        g3 = ((ee * (wq * ir ** 2)[..., None]).transpose(0, 2, 1) @ e).reshape(-1, 3, 3, 3)
        G = A * (-k * I[None, :, None, :] * g1[:, None, :, None] + I[None, :, :, None] * g1[:, None, None, :]
                 + g1[:, :, None, None] * I[None, None, :, :] - 3 * g3)
        eps = 0.5 * (G + G.transpose(0, 2, 1, 3))
        S[a:a + chunk] = lam * np.einsum("piic->pc", eps)[:, None, None, :] * I[None, :, :, None] + 2 * mu * eps
    return u, S


def shell_panels(win, faces, delta, B=4.0, n=2):
    """Quadrature (Y (m, 3) global, wq (m,)) per window face of a unit traction on the face moved outward by delta and
    extended through the part thickness [-B/2, B/2] (the window is the mid-thickness slab of a column): panels of
    ~delta in-plane, graded in z (fine near the slab), n Gauss points each."""
    w = win.g["w"]
    gx, gw = np.polynomial.legendre.leggauss(n)
    zb = np.concatenate([-np.geomspace(B / 2, 0.3, 8), np.linspace(-0.3, 0.3, 9)[1:-1], np.geomspace(0.3, B / 2, 8)])
    zz = (0.5 * (zb[1:] + zb[:-1])[:, None] + 0.5 * np.diff(zb)[:, None] * gx).ravel()
    wz = (0.5 * np.diff(zb)[:, None] * gw).ravel()
    tb = np.linspace(-w / 2, w / 2, max(2, int(np.ceil(w / delta))) + 1)
    tt = (0.5 * (tb[1:] + tb[:-1])[:, None] + 0.5 * np.diff(tb)[:, None] * gx).ravel()
    wt = (0.5 * np.diff(tb)[:, None] * gw).ravel()
    T, Z = np.meshgrid(tt, zz, indexing="ij")
    out = []
    for F in faces:
        a, s = F["axis"], F["sign"]
        Y = np.zeros((T.size, 3))
        Y[:, a] = F["centre"][a] + s * delta
        Y[:, 1 - a] = F["centre"][1 - a] + T.ravel()
        Y[:, 2] = Z.ravel() - win.o[2]                                      # global z (window frame: z + o_z)
        out.append((Y + win.o, np.outer(wt, wz).ravel()))
    return out


def cheb_grid(lo, hi, n):
    """Tensor Chebyshev-Lobatto grid (prod n, 3) on the box and its interpolation: matrix (m, prod n) at points."""
    nodes = [0.5 * (l + h) - 0.5 * (h - l) * np.cos(np.pi * np.arange(k) / (k - 1)) for l, h, k in zip(lo, hi, n)]
    grid = np.stack(np.meshgrid(*nodes, indexing="ij"), -1).reshape(-1, 3)

    def interp(P):
        L = []
        for d, x in enumerate(nodes):                                       # barycentric Lagrange basis per axis
            k = len(x)
            bw = (-1.0) ** np.arange(k)
            bw[0] *= 0.5
            bw[-1] *= 0.5
            D = P[:, d, None] - x[None]
            hit = np.abs(D) < 1e-14
            D[hit] = 1.0
            Ld = bw / D
            Ld[hit.any(1)] = hit[hit.any(1)]
            L.append(Ld / Ld.sum(1, keepdims=True))
        return np.einsum("pi,pj,pk->pijk", *L).reshape(len(P), -1)
    return grid, interp


def run(rho=1.0, d=1.5, cells=(3, 3), kappa=None, work="/tmp/rve_coupling", delta=0.15, grid=(11, 11, 5),
        verify=False, log=print):
    """Two-way coupling (non-invasive) of the window with the part, condensed onto the interface: the unknowns are
    the shell tractions P (window face f, component c; 36 for 3 x 3 cells), loaded on the faces offset outward by
    delta and through the thickness. Both sides are linear in P and are computed once:
      part    Fp(P) = Fp0 + G P     one multi-rhs solve of the 36 unit sets on the part LU (bc_case forcesets),
                                    fields on a Chebyshev grid over the window (BEM part, smooth there) + Kelvin
      window  Fw(P) = Fw0 + W P     one batched GMRES of the 37 drives u + chi:eps, t_z jumps (bead_window)
    and the interface condition Fp - P area = Fw is a 36 x 36 solve; the weld tractions follow by superposition.
    verify: re-solve the part with the converged P, evaluate it directly at every point (no grid) and solve the
    window: the remaining face residual checks the condensation and the grid."""
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    tm = {}
    win = Window(cells, ROOT_PT + np.array([0.0, d, 0.0]), kappa=kappa, work=work)
    faces = win.faces()
    nf = len(faces)
    area = win.g["w"] * win.Lz
    tm["window_setup"] = time.time() - t0
    res = dict(rho=rho, d=d, cells=list(cells), kappa=kappa, E_bead=win.E_bead, delta=delta, grid=list(grid),
               cell_eng={k: v * win.E_bead if k[0] in "EG" else v for k, v in win.cell["eng"].items()}, t=tm)
    X, Xf = win.x + win.o, np.vstack([F["x"] + win.o for F in faces])
    Xt = win.tx + win.o
    Xs = np.vstack([X, Xf, Xt, Xt + [0, 0, win.Lz]])                       # stress points
    n1, n2, nt = len(X), len(Xf), len(Xt)
    nrm = np.zeros((n2, 3))
    fw = np.concatenate([F["w"] for F in faces])
    fid = np.repeat(np.arange(nf), [len(F["w"]) for F in faces])
    for i, F in enumerate(faces):
        nrm[fid == i, F["axis"]] = F["sign"]

    def reduce(u, S):
        """Drive (u, eps, tjump) and part face forces (nf, 3) of a field (u at X; S at Xs); trailing axis: sets."""
        Sd, Sf, Sb, St = S[:n1], S[n1:n1 + n2], S[n1 + n2:n1 + n2 + nt], S[n1 + n2 + nt:]
        Fpart = np.zeros((nf, 3) + S.shape[3:])
        np.add.at(Fpart, fid, np.einsum("p,pij...,pj->pi...", fw, Sf, nrm))
        return u, strain(Sd), win.square * (St[:, :, 2] - Sb[:, :, 2]), Fpart

    panels = shell_panels(win, faces, delta)
    with part(rho, os.path.abspath(os.path.join(ROOT, "cache", "failure")), log=log) as P:
        tm["part_lu"] = time.time() - t0 - tm["window_setup"]
        # part: 36 unit traction sets (face f, component c) -> set 3 f + c, one multi-rhs solve on the same LU
        t1 = time.time()
        rows = [np.column_stack([np.full(len(wq), 3 * f + c), Y, np.outer(wq, np.eye(3)[c])])
                for f, (Y, wq) in enumerate(panels) for c in range(3)]
        fpath = os.path.join(work, "forcesets.txt")
        np.savetxt(fpath, np.vstack(rows), fmt=["%d"] + ["%.17g"] * 6)
        P.forcesets(fpath)
        tm["part_sets"] = time.time() - t1
        # BEM fields on the grid (smooth inside the part), interpolated to every point
        t1 = time.time()
        lo = np.minimum(X.min(0), Xs.min(0)) - 1e-9
        hi = np.maximum(X.max(0), Xs.max(0)) + 1e-9
        Gp, interp = cheb_grid(lo, hi, grid)
        Fg = P.fields_all(Gp)
        if np.any(np.concatenate([Fg[c][2] for c in P.cases]) < 0.99):
            raise RuntimeError("window grid outside the part")
        base = P.cases[0]
        Ug = np.stack([Fg[base][0]] + [Fg[c][0] - Fg[base][0] for c in P.cases[1:]], -1)   # (g, 3, 37)
        Sg = np.stack([Fg[base][1]] + [Fg[c][1] - Fg[base][1] for c in P.cases[1:]], -1)   # (g, 3, 3, 37)
        tm["part_eval"] = time.time() - t1
        t1 = time.time()
        Mu, Ms = interp(X), interp(Xs)
        u = np.einsum("pg,g...->p...", Mu, Ug)
        S = np.einsum("pg,g...->p...", Ms, Sg)
        with ThreadPoolExecutor() as ex:                                    # particular (Kelvin) parts of the sets
            uks = list(ex.map(lambda p: kelvin_unit(X, *p, stress=False)[0], panels))
            Sks = list(ex.map(lambda p: kelvin_unit(Xs, *p)[1], panels))
        u[..., 1:] += np.concatenate(uks, -1)
        S[..., 1:] += np.concatenate(Sks, -1)
        ud, eps, tj, Fp = reduce(u, S)
        tm["interp_kelvin"] = time.time() - t1
        # window: the 37 drives in one batched solve
        t1 = time.time()
        T = win.solve_many(ud, eps, tj, tag="sets")
        Fw = window_face_forces(T, faces, win.g["w"])                      # (nf, 3, 37)
        tm["window_solve"] = time.time() - t1
        # interface: (G - area I - W) P = Fw0 - Fp0
        t1 = time.time()
        Gm = Fp[..., 1:].reshape(3 * nf, 3 * nf)
        Wm = Fw[..., 1:].reshape(3 * nf, 3 * nf)
        Pt = np.linalg.solve(Gm - area * np.eye(3 * nf) - Wm, (Fw[..., 0] - Fp[..., 0]).ravel())
        coef = np.concatenate([[1.0], Pt])
        Fp2, Fw2 = Fp @ coef, Fw @ coef
        tw = T[:, 10:].reshape(len(T), -1, 3)
        T2 = np.column_stack([T[:, :10], np.einsum("pqc,q->pc", tw, coef)])
        tm["interface"] = time.time() - t1
        scale = np.abs(Fp[..., 0]).max()
        mis = np.abs(Fw[..., 0] - Fp[..., 0]).max(1) / scale
        res["oneway"] = dict(max_mismatch=float(mis.max()), rms_mismatch=float(np.sqrt(np.mean(mis ** 2))))
        res["twoway"] = dict(residual=float(np.abs(Fp2 - Pt.reshape(nf, 3) * area - Fw2).max() / scale),
                             P_rel=float(np.abs(Pt).max() * area / scale), cond=float(np.linalg.cond(Gm - area * np.eye(3 * nf) - Wm)))
        # what the feedback changes: the weld tractions (kinds 0-2) against the one-way drive
        weld = T[:, 0] < 3
        t1w, t2w = tw[weld, 0], T2[weld, 10:13]
        res["twoway"]["weld_change"] = float(np.abs(t2w - t1w).max() / np.abs(t1w).max())
        res["P"] = Pt.reshape(nf, 3).tolist()
        np.save(os.path.join(work, "oneway_tractions.npy"), np.column_stack([T[:, :10], tw[:, 0]]))
        np.save(os.path.join(work, "twoway_tractions.npy"), T2)
        if verify:                                                          # direct check of the condensed answer
            # the part with the converged P as one load set, evaluated directly at every point (no grid)
            t1 = time.time()
            fpath = os.path.join(work, "forces.txt")
            np.savetxt(fpath, np.vstack([np.column_stack([np.zeros(len(wq)), Y, np.outer(wq, Pt[3 * f:3 * f + 3])])
                                         for f, (Y, wq) in enumerate(panels)]), fmt=["%d"] + ["%.17g"] * 6)
            P.forcesets(fpath)
            Fd = P.fields_all(np.vstack([X, Xs]))
            Fb = Fd[P.cases[0]], Fd[P.cases[1]]
            ub, Sb_ = Fb[0][0][:n1], Fb[0][1][n1:]
            ux = np.einsum("pg,gc->pc", Mu, Ug[..., 0])
            Sx = np.einsum("pg,gij->pij", Ms, Sg[..., 0])
            res["grid_error"] = dict(u=float(np.abs(ub - ux).max() / np.abs(ub).max()),
                                     stress=float(np.abs(Sb_ - Sx).max() / np.abs(Sb_).max()))
            ub, Sb_ = Fb[1][0][:n1], Fb[1][1][n1:]
            for f, (Yf, wq) in enumerate(panels):
                uk, _ = kelvin_unit(X, Yf, wq, stress=False)
                _, Sk = kelvin_unit(Xs, Yf, wq)
                ub, Sb_ = ub + uk @ Pt[3 * f:3 * f + 3], Sb_ + Sk @ Pt[3 * f:3 * f + 3]
            ud1, eps1, tj1, Fp1 = reduce(ub, Sb_)
            T1 = win.solve_many(ud1[..., None], eps1[..., None], tj1[..., None], tag="verify")
            Fw1 = window_face_forces(T1, faces, win.g["w"])[..., 0]
            res["verify"] = dict(residual=float(np.abs(Fp1 - Pt.reshape(nf, 3) * area - Fw1).max() / scale),
                                 weld_diff=float(np.abs(T1[weld, 10:13] - t2w).max() / np.abs(t1w).max()),
                                 t=time.time() - t1)
            log("grid interpolation error: u %.1e, stress %.1e (relative max)" % tuple(res["grid_error"].values()))
            log("verify: direct face residual %.2e, weld tractions vs condensed %.2e (%.0f s)" % (
                res["verify"]["residual"], res["verify"]["weld_diff"], res["verify"]["t"]))
    tm["total"] = time.time() - t0
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--d", type=float, default=1.5, help="window centre ahead of the notch root (mm)")
    ap.add_argument("--cells", type=int, default=3)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--grid", default="11,11,5", help="Chebyshev grid of the part fields over the window")
    ap.add_argument("--verify", action="store_true", help="direct re-solve and window check of the condensed answer")
    ap.add_argument("--work", default="/tmp/rve_coupling")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = run(a.rho, a.d, (a.cells, a.cells), a.kappa, a.work, grid=tuple(int(v) for v in a.grid.split(",")),
              verify=a.verify)
    o, t = res["oneway"], res["twoway"]
    print("one-way (u + chi:eps, t_z): face force mismatch max %.2f %%, rms %.2f %%" % (100 * o["max_mismatch"], 100 * o["rms_mismatch"]))
    print("two-way: interface residual %.1e, |P| area / |F| %.2f %%, weld tractions change %.2f %% (cond %.1f)" % (
        t["residual"], 100 * t["P_rel"], 100 * t["weld_change"], t["cond"]))
    print("timings: " + ", ".join("%s %.1f s" % kv for kv in res["t"].items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
