"""Figure and ParaView file of the meshes behind the cube-cavity FEM/BEM comparison.

  (a) IGA-FEM octant: element (knot) lines on every boundary surface of the 3 exact NURBS
      volumes, p and n as used for the finest result
  (b) control net of the coarsest exact geometry (degree 4 x 4 x 1, one element per patch)
  (c) IgBEM surface mesh, geo/cube_cavity_r3.dat (768 elements), knot lines
Writes results/fig_fem_mesh.{png,pdf} and results/fields/fem_mesh_p{p}_n{n}.vtu (each NURBS
element sampled as s^3 linear hexes, cell data: patch, element).

usage: python plot_mesh.py [--p 5] [--n 8] [--sub 4]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import jax.numpy as jnp  # noqa: E402
import jaxiga as jx  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jaxiga.space.pointset import PointSet  # noqa: E402
from scipy.interpolate import BSpline  # noqa: E402

from make_geometries import read_dat  # noqa: E402
from octant_geometry import A, L, octant_patches, refined  # noqa: E402

RES = os.path.join(HERE, "..", "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a"]
TOL = 1e-9


def on_boundary(x):
    r = np.linalg.norm(x, axis=-1)
    return (np.abs(r - A) < TOL) | (np.abs(x.max(-1) - L) < TOL) | (np.abs(x).min(-1) < TOL)


def element_edges(space, m=12):
    """Physical polylines of the 12 edges of every element, kept if on the domain boundary."""
    s = np.linspace(-1, 1, m)
    refs = []
    for d in range(3):
        o = [k for k in range(3) if k != d]
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                r = np.zeros((m, 3))
                r[:, d], r[:, o[0]], r[:, o[1]] = s, a, b
                refs.append(r)
    ref = np.concatenate(refs)
    x = np.asarray(jx.evaluate(space, PointSet(np.arange(space.n_elems), jnp.asarray(ref))).x)
    x = x.reshape(space.n_elems, 12, m, 3).reshape(-1, m, 3)
    keep = on_boundary(x).all(axis=1)
    lines = x[keep]
    # drop duplicates shared by neighbouring elements
    key = np.round(np.sort(np.stack([lines[:, 0], lines[:, -1]], 1).reshape(len(lines), -1), 1), 9)
    _, idx = np.unique(np.round(lines[:, m // 2], 9), axis=0, return_index=True)
    del key
    return lines[np.sort(idx)]


def surface_lines(P, m=16):
    """Knot lines of one Bembel surface patch (homogeneous control points)."""
    nv, nu, _ = P.Pw.shape
    bu, bv = BSpline(np.asarray(P.U), np.eye(nu), P.p), BSpline(np.asarray(P.V), np.eye(nv), P.q)
    ev = lambda u, v: np.einsum("ki,kj,jic->kc", bu(u), bv(v), P.Pw)
    out = []
    t = np.linspace(0, 1, m * 4)
    for ku in np.unique(P.U):
        H = ev(np.full_like(t, ku), t)
        out.append(H[:, :3] / H[:, 3:])
    for kv in np.unique(P.V):
        H = ev(t, np.full_like(t, kv))
        out.append(H[:, :3] / H[:, 3:])
    return out


def write_vtu(path, space, sub):
    s = np.linspace(-1, 1, sub + 1)
    ref = np.array([[a, b, c] for c in s for b in s for a in s])
    x = np.asarray(jx.evaluate(space, PointSet(np.arange(space.n_elems), jnp.asarray(ref))).x)
    ne, npt = space.n_elems, (sub + 1) ** 3
    pts = x.reshape(-1, 3)
    idx = lambda i, j, k: i + (sub + 1) * (j + (sub + 1) * k)
    loc = []
    for k in range(sub):
        for j in range(sub):
            for i in range(sub):
                loc.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                            idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1),
                            idx(i, j + 1, k + 1)])
    loc = np.array(loc)
    conn = (np.arange(ne)[:, None, None] * npt + loc[None]).reshape(-1, 8)
    elem = np.repeat(np.arange(ne), len(loc))
    patch = np.repeat(np.asarray(space.elem_patch), len(loc))
    nc = len(conn)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1">\n'
                '<UnstructuredGrid>\n<Piece NumberOfPoints="%d" NumberOfCells="%d">\n' % (len(pts), nc))
        f.write('<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        np.savetxt(f, pts, fmt="%.9g")
        f.write('</DataArray></Points>\n<Cells>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        np.savetxt(f, conn, fmt="%d")
        f.write('</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        np.savetxt(f, 8 * np.arange(1, nc + 1), fmt="%d")
        f.write('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
        np.savetxt(f, np.full(nc, 12), fmt="%d")
        f.write('</DataArray>\n</Cells>\n<CellData Scalars="patch">\n')
        for name, v in (("patch", patch), ("element", elem)):
            f.write('<DataArray type="Int32" Name="%s" format="ascii">\n' % name)
            np.savetxt(f, v, fmt="%d")
            f.write("</DataArray>\n")
        f.write("</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")


def setup3d(ax, title, lim):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(*lim), ax.set_ylim(*lim), ax.set_zlim(*lim)
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel("$x$", labelpad=-8), ax.set_ylabel("$y$", labelpad=-8), ax.set_zlabel("$z$", labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2)
    ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sub", type=int, default=4)
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    base = octant_patches()
    space = jx.FunctionSpace(refined(base, args.p, args.n, 1.5), vec=3)

    fig = plt.figure(figsize=(12, 4.3), constrained_layout=True)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    patch_of = np.asarray(space.elem_patch)
    for k in range(3):
        sub = jx.FunctionSpace([refined(base, args.p, args.n, 1.5)[k]], vec=3)
        for ln in element_edges(sub):
            ax.plot(*ln.T, color=C[k], lw=0.45)
    del patch_of
    setup3d(ax, "(a) IGA-FEM octant, 3 patches\n     $p=%d$, %d$^3$ el./patch, %d el., %s dofs"
            % (args.p, args.n, space.n_elems, f"{space.n_dofs:,}"), (0, L))

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    for k, P in enumerate(base):
        X = np.asarray(P.ctrl_pts).reshape(2, 5, 5, 3)  # (w, v, u, 3)
        for g in (X, X.transpose(0, 2, 1, 3)):
            for w in range(2):
                for row in g[w]:
                    ax.plot(*row.T, color=C[k], lw=0.7)
        for v in range(5):
            for u in range(5):
                ax.plot(*X[:, v, u].T, color=C[k], lw=0.4, ls=":")
        ax.scatter(*X.reshape(-1, 3).T, s=5, color=C[k], depthshade=False)
    setup3d(ax, "(b) control net of the exact geometry\n     degree (4, 4, 1), before refinement", (0, L))

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    bem = read_dat(os.path.join(HERE, "..", "..", "geo", "cube_cavity_r3.dat"))
    for k, P in enumerate(bem):
        for ln in surface_lines(P):
            ax.plot(*ln.T, color=C[2] if k >= 6 else "#8a8a85", lw=0.35 if k >= 6 else 0.3)
    nel = sum((len(np.unique(P.U)) - 1) * (len(np.unique(P.V)) - 1) for P in bem)
    setup3d(ax, "(c) IgBEM surface mesh (full cube)\n     $p=4$ sphere, %d el., %s dofs" % (nel, "1,216"), (0, 1))
    os.makedirs(os.path.join(RES, "fields"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, "fig_fem_mesh." + ext), dpi=220, bbox_inches="tight", pad_inches=0.15)
    write_vtu(os.path.join(RES, "fields", "fem_mesh_p%d_n%d.vtu" % (args.p, args.n)), space, args.sub)


if __name__ == "__main__":
    main()
