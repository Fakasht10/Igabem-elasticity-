"""Paper's calibration (2020 Sec. 6, Fig. 9e) on OUR stress field: plain P50 sigma_max = 6.7 MPa at 2e6, k 5.1;
sharp-notch S-N curve = regression of log N on log sigma_n,max over all sharp specimens (R -1 and 0, failures);
at each N, L/2 is where sigma_y(r)/sigma_net equals plain(N)/sharp(N). Fit L = A N^B."""
import json, os, sys, numpy as np

from fatigue import calib as c
from fatigue.hsv import load
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 11, "font.family": "serif", "mathtext.fontset": "cm"})
UTS = 42.9
g, plain, spec = load()
no = {k: c.Notch(g[k]["f"], g[k]) for k in g}
sh = np.array([(N, s * 2 / (1 - R)) for k, p, R, s, N, ro in spec if k == "sharp" and not ro])
b, a = np.polyfit(np.log10(sh[:, 1]), np.log10(sh[:, 0]), 1)          # log N = a + b log S  (N on S, as in the paper)
kV = -b; SV = lambda N: 10 ** ((np.log10(N) - a) / b)
S0 = lambda N: np.minimum(6.7 * (N / 2e6) ** (-1 / 5.1), UTS)
n = no["sharp"]; d, s = np.asarray(n.d), np.asarray(n.s)
Ns = np.geomspace(1e3, 2e6, 40); L = []
for N in Ns:
    f = S0(N) / SV(N)
    L.append(2 * np.interp(-f, -s, d) if s.min() < f < s[0] else np.nan)
L = np.array(L); ok = np.isfinite(L) & (L / 2 < 0.99 * n.lig)
B, lA = np.polyfit(np.log10(Ns[ok]), np.log10(L[ok]), 1); A = 10 ** lA
print(f"sharp S-N (sigma_max net): k {kV:.2f}, S50(2e6) {SV(2e6):.2f} MPa; plain 6.7 MPa k 5.1")
print(f"OUR L_M(N) = {A:.2f} N^{B:.3f} mm  (L at 1e3 {A*1e3**B:.2f}, at 2e6 {A*2e6**B:.3f}); used N range {Ns[ok].min():.0f}-{Ns[ok].max():.0f}")
th = json.load(open(os.path.join(os.path.dirname(__file__), "data", "ezeh_susmel_2020_fig10_point_method.json")))["points"]
T = np.array([(q["x"], q["y"]) for q in th if not q["runout"]])
O = {k: np.array([(N, no[k].pm(A * N ** B) * s * 2 / (1 - R) / UTS) for kk, p, R, s, N, ro in spec if kk == k and not ro]) for k in no}
Oall = np.vstack(list(O.values()))
def drawn(y0, y1, N0=149.5):
    N = np.geomspace(10, 2e6, 300); sl = np.log(y1 / y0) / np.log(2e6 / N0)
    return N, np.where(N < N0, y0, y0 * (N / N0) ** sl)
fig, ax = plt.subplots(figsize=(9, 6.6), layout="constrained")
ax.plot(*drawn(0.9965, 0.1551), "k-", lw=1.4); ax.plot(*drawn(1.5591, 0.2427), "k--", lw=.9); ax.plot(*drawn(0.6369, 0.0991), "k--", lw=.9)
mk = {"blunt": "D", "intermediate": "o", "sharp": "^"}
for k in mk:
    t = np.array([(q["x"], q["y"]) for q in th if q["type"] == k and not q["runout"]])
    ax.scatter(t[:, 0], t[:, 1], marker=mk[k], s=50, facecolor="none", edgecolors="#d62728", lw=1.1)
    ax.scatter(O[k][:, 0], O[k][:, 1], marker=mk[k], s=18, color="#1f4e9c")
Nf = np.geomspace(1e2, 2e6, 50); h, lb = [], []
for P, colr, ls, who in ((T, "#d62728", "-", "authors: Fig. 10 points (hollow), their L_M = 16.4 N^-0.242"),
                         (Oall, "#1f4e9c", "--", f"ours: IgBEM + our L_M = {A:.1f} N^{B:.3f} (filled)")):
    bb, ii = np.polyfit(np.log10(P[:, 0]), np.log10(P[:, 1]), 1)
    ax.plot(Nf, 10 ** (ii + bb * np.log10(Nf)), color=colr, lw=2.6, ls=ls)
    h.append(Line2D([], [], color=colr, lw=2.6, ls=ls)); lb.append(f"trend, {who}\n   k {-1/bb:.2f}, value at 2e6 {10**(ii+bb*np.log10(2e6)):.3f}")
h += [Line2D([], [], marker=m, color="0.4", mfc="none", ls="", ms=7) for m in mk.values()]
lb += ["blunt U r3", "intermediate U r1", "sharp V r0.15"]
h += [Line2D([], [], color="k", lw=1.4), Line2D([], [], color="k", ls="--", lw=.9)]
lb += ["plain P50: 0.155 at 2e6, k 5.1 (as drawn)", "plain P10 / P90"]
ax.legend(h, lb, fontsize=8.6, loc="lower left")
ax.set(xscale="log", yscale="log", xlim=(10, 1e7), ylim=(0.01, 10), xlabel="$N_f$ [cycles to failure]", ylabel=r"$\sigma_{eff,max}/\sigma_{UTS}$")
ax.grid(alpha=.3, which="both")
ax.set_title("Point Method, R = -1 and 0, angles 0/30/45 (as Fig. 10): each with its own L_M(N),\n"
             "calibrated the paper's way (plain P50 + sharp V curve), UTS 42.9 MPa; failures only", fontsize=10.5)
for dd in ("/mnt/project-files/fatigue-data/", "/home/user/igabem/results/fatigue/"):
    fig.savefig(dd + "fig10_point_method_own_calibration.png", dpi=170)
