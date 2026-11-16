import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Tables for the combined model: weakest link over a field smoothed over the defect size a (disc of radius a, material only).
For each specimen and a: histogram (log bins) of the smoothed sigma_1/sigma_net with volume weights, so sum(w s^m) is available for any m.
ES: stored IgBEM polar grid around the root (thickness-mean), half-plane ahead of the root, mirrored at the bisector and the centre line;
    far volume from the stored whole-specimen histogram (total = stored - raster(0) + raster(a)). Flank material behind the root is not in the grid.
YS: existing 2D plane-stress FE (fe2d.py) re-evaluated on a 0.02 mm raster; same difference rule against the FE quadrature sum."""
import sys, math, json, numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
BINS = np.geomspace(1e-3, 12, 1501)
def disc(a, h):
    n = int(math.ceil(a / h)); y, x = np.mgrid[-n:n + 1, -n:n + 1]; return ((x * x + y * y) * h * h <= a * a + 1e-12).astype(float)
def smooth(s, mask, a, h, refl):
    """normalised disc average. s, mask on [x, y] grid; refl = (lo_x, hi_x, lo_y, hi_y) booleans: mirror symmetry at that edge."""
    if a < h: return s
    n = int(math.ceil(a / h)); S, M = s * mask, mask.astype(float)
    pad = lambda A: np.pad(np.pad(A, ((n if refl[0] else 0, n if refl[1] else 0), (0, 0)), mode="symmetric"), ((0, 0), (n if refl[2] else 0, n if refl[3] else 0)), mode="symmetric")
    S, M = pad(S), pad(M); k = disc(a, h)
    num, den = fftconvolve(S, k, mode="same"), fftconvolve(M, k, mode="same")
    sl = (slice(n if refl[0] else 0, S.shape[0] - (n if refl[1] else 0)), slice(n if refl[2] else 0, S.shape[1] - (n if refl[3] else 0)))
    return np.where(mask, num[sl] / np.maximum(den[sl], 1e-9), 0.0)
def hist(s, w): 
    s = np.clip(s, BINS[0] * 1.0001, BINS[-1] * 0.9999); i = np.digitize(s, BINS) - 1
    W = np.bincount(i, w, len(BINS) - 1); WS = np.bincount(i, w * s, len(BINS) - 1); return W, np.where(W > 0, WS / np.maximum(W, 1e-300), 1.0)
out = {}
def es():
    sys.path.insert(0, HERE + "../.."); from es2020 import methods as M
    A = np.array([0, .02, .05, .1, .15, .2, .3, .4, .5, .6, .8, 1.0, 1.25, 1.5, 2.0, 2.5, 2.9]); h, T = 0.01, 3.0
    x = (np.arange(300) + 0.5) * h; y = (np.arange(500) + 0.5) * h; X, Y = np.meshgrid(x, y, indexing="ij")
    for g, n in (("V", "sharp"), ("U1", "intermediate"), ("U3", "blunt")):
        a = M.fields()[n]["area"]; r, th, s1 = np.array(a["r"]), np.array(a["th"]), np.array(a["s1"])
        f = RegularGridInterpolator((np.log(r), th), s1, bounds_error=False, fill_value=None)
        s = f(np.stack([np.log(np.clip(np.hypot(X, Y), r[0], r[-1])), np.clip(np.arctan2(Y, X), th[0], th[-1])], -1)); s = np.maximum(s, 0)
        mask = np.ones_like(s, bool); w = np.full(s.shape, h * h * T * 4)
        v = M.fields()[n]["volume"]; out[f"ES_{g}_far_w"], out[f"ES_{g}_far_s"] = np.array(v["w"]), np.array(v["s1"])
        for i, aa in enumerate(A):
            W_, S_ = hist(smooth(s, mask, aa, h, (False, True, True, False)).ravel(), w.ravel()); out[f"ES_{g}_w{i}"], out[f"ES_{g}_s{i}"] = W_, S_
        print("ES", g, "raster vol", round(w.sum(), 1), "raster(0) sum m=5", round(float((out[f'ES_{g}_w0'] * out[f'ES_{g}_s0'] ** 5).sum()), 1), "stored whole-volume sum m=5", round(float((np.array(v['w']) * np.array(v['s1']) ** 5).sum()), 1),
              "smoothed a=0.4,1:", [round(float((out[f'ES_{g}_w{i}'] * out[f'ES_{g}_s{i}'] ** 5).sum()), 1) for i in (7, 11)], flush=True)
    out["ES_A"] = A
def ys():
    sys.path.insert(0, HERE); sys.path.insert(0, HERE + "ys/")
    import fe2d, ys_wl_fields as F
    from matplotlib.path import Path
    from skfem import Basis, ElementTriP2
    A = np.array([0, .02, .05, .1, .15, .2, .3, .4, .5, .6, .8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]); h, T = 0.02, 5.0
    x = (np.arange(625) + 0.5) * h; y = (np.arange(900) + 0.5) * h; X, Y = np.meshgrid(x, y, indexing="ij"); P = np.c_[X.ravel(), Y.ravel()]
    for name, kind, rho, hr in (("plain", "plain", 18.5, 0.15), ("V", "V", 0.15, 0.15 / 40), ("U1", "U", 1.0, 1 / 40), ("U3", "U", 3.0, 3 / 40)):
        w, s, pts = F.run(kind, rho, hr, coords=True)
        mask = Path(F.boundary(kind, rho, max(int(np.pi / 2 * rho / hr), 8) if kind != "plain" else 0)).contains_points(P).reshape(X.shape)
        lin = LinearNDInterpolator(pts, s); near = NearestNDInterpolator(pts, s); sr = lin(P); bad = ~np.isfinite(sr); sr[bad] = near(P[bad]); sr = np.where(mask, sr.reshape(X.shape), 0.0)
        win = (Y < 12.0) & mask; wr = np.full(X.shape, h * h * T * 4)
        out[f"YS_{name}_far_w"], out[f"YS_{name}_far_s"] = hist(s, w)
        for i, aa in enumerate(A):
            sm = smooth(sr, mask, aa, h, (True, False, True, False)); out[f"YS_{name}_w{i}"], out[f"YS_{name}_s{i}"] = hist(sm[win], wr[win])
        print("YS", name, "FE sum m=5", round(float((w * s ** 5).sum()), 1), "raster(0) window", round(float((out[f'YS_{name}_w0'] * out[f'YS_{name}_s0'] ** 5).sum()), 1),
              "a=0.4,1,3:", [round(float((out[f'YS_{name}_w{i}'] * out[f'YS_{name}_s{i}'] ** 5).sum()), 1) for i in (7, 11, 16)], flush=True)
    out["YS_A"] = A
if __name__ == "__main__":
    es(); ys(); np.savez_compressed(HERE + "combo_tables.npz", **out)
