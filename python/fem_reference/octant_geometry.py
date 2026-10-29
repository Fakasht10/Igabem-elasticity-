"""Exact 3-patch NURBS volume of one octant of the cube with a spherical cavity.

The BEM case (python/make_geometries.py, cube_cavity_r*.dat) is the unit cube with a
cavity of radius a = 0.2 at c = (1/2, 1/2, 1/2). Under the tension load it is symmetric
about the three planes through c, so the FEM reference meshes only the octant
[0, L]^3 \\ B(0, a), L = 1/2, in coordinates centred on c, with symmetry conditions on the
planes x = 0, y = 0, z = 0.

Construction (all exact):
  * The cavity surface is Bembel's 6-patch rational-quartic sphere (geo/sphere_r0.dat),
    the same surface the BEM uses. Each of the patches +x, +y, +z is split into Bezier
    quarters by knot insertion at 1/2, and the quarter inside the octant is kept.
  * The matching cube face is the central (gnomonic) projection of that quarter onto the
    plane x_n = L. If S^w = (X, Y, Z, W) are its homogeneous control points, the projection
    L S / S_n is a NURBS with the same knots and homogeneous points (L X, L Y, L Z, X_n),
    so the face is planar to round-off and is parametrized consistently with the sphere.
  * The volume is the ruled (degree-1 in w) blend of the two in homogeneous coordinates.
    Shared edges of neighbouring quarters lie in the planes x = y, y = z, x = z, where the
    two projections coincide, so the three patches are conforming; edges on the symmetry
    planes have a zero coordinate in every control point, so those faces are planar too.

Side w0 is the cavity, side w1 the cube face. Directions are oriented so det J > 0.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from make_geometries import read_dat  # noqa: E402

import jaxiga as jx  # noqa: E402
from jaxiga.geometry.nurbs import Patch  # noqa: E402

A, L = 0.2, 0.5
SPHERE = os.path.join(HERE, "..", "..", "geo", "sphere_r0.dat")
AXIS_NAMES = "xyz"


def _quarter(P, axis):
    """Bezier quarter of the Bembel patch facing +axis that lies in the positive octant."""
    nv, nu, _ = P.Pw.shape
    X = P.euclid().reshape(-1, 3)
    w = P.Pw[..., 3].ravel()
    srf = Patch.create([P.U, P.V], (P.p, P.q), X, w)
    srf = srf.insert_knots([0.5] * P.p, [0.5] * P.q)
    n0, n1 = srf.n_cp_per_dir
    H = np.asarray(srf.homogeneous()).reshape(n1, n0, 4)
    best = None
    for jv in (slice(0, P.q + 1), slice(P.q, n1)):
        for iu in (slice(0, P.p + 1), slice(P.p, n0)):
            Q = H[jv, iu]
            E = Q[..., :3] / Q[..., 3:]
            if E.min() > -1e-12:  # every control point in the closed positive octant
                best = Q
    assert best is not None, "no quarter of the patch lies in the octant"
    return best  # (p+1, p+1, 4) homogeneous, u fastest along axis 1


def octant_patches():
    """The three volume patches (+x, +y, +z regions), degree (4, 4, 1), one element each."""
    sph = read_dat(SPHERE)
    centres = [P.euclid().mean(axis=(0, 1)) for P in sph]
    patches = []
    for axis in range(3):
        k = int(np.argmax([c[axis] for c in centres]))
        Q = _quarter(sph[k], axis)  # unit-sphere quarter, homogeneous
        Qa = Q.copy()
        Qa[..., :3] *= A  # cavity of radius a
        Qc = np.empty_like(Q)
        Qc[..., :3] = L * Q[..., :3]
        Qc[..., 3] = Q[..., axis]  # weight X_n: central projection onto x_n = L
        assert Qc[..., 3].min() > 0
        Hw = np.stack([Qa, Qc], axis=0)  # (w, v, u, 4)
        E = (Hw[..., :3] / Hw[..., 3:]).reshape(-1, 3)
        W = Hw[..., 3].ravel()
        p = sph[k].p
        kv = [0.0] * (p + 1) + [1.0] * (p + 1)
        vol = Patch.create([kv, kv, [0, 0, 1, 1]], (p, p, 1), E, W)
        if _jac_sign(vol) < 0:
            vol = vol.reverse(0)
        name = AXIS_NAMES[axis]
        vol = vol.with_labels(w0="cavity", w1="face_" + name)
        patches.append(vol)
    return patches


def jacobian_range(vol, n=7):
    """(min, max) of det(dx/dxi) over an n^3 grid per element, by central differences."""
    space = jx.FunctionSpace([vol])
    ps = jx.grid(space, n)
    x = np.asarray(jx.evaluate(space, ps).x).reshape(-1, n, n, n, 3)  # (e, w, v, u, 3), u fastest
    h = 2.0 / (n - 1)  # reference element is [-1, 1]^3
    d0 = np.gradient(x, h, axis=3)
    d1 = np.gradient(x, h, axis=2)
    d2 = np.gradient(x, h, axis=1)
    det = np.einsum("...i,...i", d0, np.cross(d1, d2))
    return det.min(), det.max()


def _jac_sign(vol):
    lo, hi = jacobian_range(vol)
    return np.sign(lo) if lo * hi > 0 else 0


def refined(patches, p, n, grade=1.0):
    """Elevate to degree p everywhere; n elements per direction on each patch.

    Tangential knots are uniform. Radial knots follow w = s^grade (s uniform), which
    clusters elements at the cavity (w = 0) for grade > 1.
    """
    out = []
    s = np.arange(1, n) / n
    tang = list(s)
    rad = list(s**grade)
    for P in patches:
        Q = P.elevate((p, p, p))
        if n > 1:
            Q = Q.insert_knots(tang, tang, rad)
        out.append(Q)
    return out
