"""Mesh figure of the four Yaren-Susmel IgBEM quarters: top view, root zoom, and the y = 0 ligament face (thickness grading)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from fatigue import ys2025 as Y
from make_geometries import read_dat

def ev(P, u, v):
    Bu = BSpline(P.U, np.eye(len(P.U) - P.p - 1), P.p)(u); Bv = BSpline(P.V, np.eye(len(P.V) - P.q - 1), P.q)(v)
    H = np.einsum('ai,bj,jik->abk', Bu, Bv, P.Pw); return H[..., :3] / H[..., 3:4]

def lines(P, n=25):
    out = []
    for u in np.unique(P.U): out.append(ev(P, np.full(n, u).clip(P.U[0], P.U[-1] - 1e-12), np.linspace(P.V[0], P.V[-1] - 1e-12, n)).reshape(-1, 3) if False else ev(P, np.array([min(u, P.U[-1] - 1e-12)]), np.linspace(P.V[0], P.V[-1] - 1e-12, n))[0])
    for v in np.unique(P.V): out.append(ev(P, np.linspace(P.U[0], P.U[-1] - 1e-12, n), np.array([min(v, P.V[-1] - 1e-12)]))[:, 0])
    return out

names = ['plain', 'V_r0.15', 'U_r1', 'U_r3']; ttl = {'plain': 'plain hourglass R18.5', 'V_r0.15': 'V 35°, r 0.15', 'U_r1': 'U r1', 'U_r3': 'U r3'}
zoom = {'plain': 4.0, 'V_r0.15': 0.6, 'U_r1': 2.5, 'U_r3': 5.0}
fig, ax = plt.subplots(3, 4, figsize=(16, 13), gridspec_kw=dict(height_ratios=[2.2, 1.4, 1]))
for j, n in enumerate(names):
    dat, info = Y.geometry(n); P = read_dat(dat); L = []
    for Q in P:
        X = Q.euclid().reshape(-1, 3); L.append((Q, X))
    for Q, X in L:
        top = np.allclose(X[:, 2], Y.T, atol=1e-9); lig = np.allclose(X[:, 1], 0, atol=1e-9) and X[:, 0].max() <= Y.WN / 2 + 1e-9
        if top:
            for c in lines(Q):
                ax[0, j].plot(c[:, 0], c[:, 1], 'k', lw=.35); ax[1, j].plot(c[:, 0], c[:, 1], 'k', lw=.4)
        if lig:
            for c in lines(Q): ax[2, j].plot(c[:, 0], c[:, 2], 'k', lw=.4)
    unk = None
    ax[0, j].set_aspect('equal'); ax[0, j].set_title(f'{ttl[n]}: {len(P)} patches', fontsize=10)
    H = (Y.L_PLAIN if n == 'plain' else Y.SPECS[n].L) / 2
    ax[0, j].annotate('', (Y.WG / 4, H + 3), (Y.WG / 4, H), arrowprops=dict(arrowstyle='->', color='C3'))
    ax[0, j].text(Y.WG / 4 + .5, H + 1, 'traction = net stress 1\n(× WN/WG on gross)', color='C3', fontsize=7)
    ax[0, j].text(.3, H / 2, 'sym x', rotation=90, color='C0', fontsize=8); ax[0, j].text(1, .4, 'sym y', color='C0', fontsize=8)
    ax[0, j].set_ylim(-1, H + 5)
    z = zoom[n]; ax[1, j].set_xlim(Y.WN / 2 - z, Y.WN / 2 + z); ax[1, j].set_ylim(-.15 * z, 1.4 * z); ax[1, j].set_aspect('equal')
    ax[1, j].set_title(f'root zoom (top face z = T)', fontsize=9)
    ax[2, j].set_xlim(Y.WN / 2 - min(3, 4 * z), Y.WN / 2 + .05 * z); ax[2, j].set_title('ligament face y = 0 (x-z): thickness grading', fontsize=9)
    ax[2, j].set_xlabel('x [mm]'); ax[2, j].set_ylabel('z [mm]')
    h0 = info.get('h0'); ax[1, j].text(.02, .95, (f'h0 {h0:.3g} mm, n_arc {info["n_arc"]}, hz0 T/60' if h0 else f'step_auto h 3, focus at root, min span J {info["min_span_jacobian"]:.2f}'),
                                       transform=ax[1, j].transAxes, fontsize=7, va='top', bbox=dict(fc='w', ec='.7'))
fig.suptitle('Yaren & Susmel 2025 specimens, IgBEM quarter models (exact NURBS; notched = designed blocks, graded by knot insertion; plain = STEP import)\n'
             'E 3000 MPa, ν 0.35 (stress field independent of E); symmetry x = 0 and y = 0 (rollers), free faces z = 0, T; grip holes omitted (load ≥ 3.5 net widths from root)', fontsize=10)
plt.tight_layout(rect=(0, 0, 1, .95)); plt.savefig(sys.argv[1], dpi=130)
