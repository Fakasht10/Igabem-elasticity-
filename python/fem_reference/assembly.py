"""Lean Galerkin elasticity assembly on a JAXIGA function space.

JAXIGA supplies the exact NURBS geometry, Bezier extraction, multipatch gluing and basis
evaluation; the stiffness, loads and solve are done here in NumPy/SciPy. Going through
jx.solve on the same space gives the same answer (cavity_fem.py --check-jaxiga), but it
spends almost all of its time in op-by-op JAX compilation, np.unique on the CSR pattern and
an autodiff right-hand side: 55 s at 4k dofs, of which SuperLU takes 0.6 s.

Dofs are interleaved, 3 i + c for component c of scalar basis function i (JAXIGA's order).
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

import jaxiga as jx
from jaxiga.space.function_space import expand_dofs


def stiffness(space, q, lam, mu, chunk=32):
    """K_ab,ij = int lam dRa_i dRb_j + mu (delta_ij grad Ra . grad Rb + dRa_j dRb_i)."""
    ps = jx.gauss(space, q)
    nd = space.n_dofs
    K = sps.csr_matrix((nd, nd))
    for basis in jx.evaluate(space, ps, chunk=chunk):
        dR = np.asarray(basis.dR)  # (e, q, 3, m)
        w = np.asarray(basis.w)  # (e, q)
        dofs = expand_dofs(np.asarray(basis.dofs), 3)  # (e, 3m)
        e, nq, _, m = dR.shape
        # M[e, k, a, l, b] = sum_q w dR_k,a dR_l,b, one batched GEMM per element
        D = dR.transpose(0, 2, 3, 1).reshape(e, 3 * m, nq)
        M = np.matmul(D * w[:, None, :], D.transpose(0, 2, 1)).reshape(e, 3, m, 3, m)
        tr = M[:, 0, :, 0, :] + M[:, 1, :, 1, :] + M[:, 2, :, 2, :]
        Ke = np.empty((e, m, 3, m, 3))
        for i in range(3):
            for j in range(3):
                Ke[:, :, i, :, j] = lam * M[:, i, :, j, :] + mu * M[:, j, :, i, :]
            Ke[:, :, i, :, i] += mu * tr
        Ke = Ke.reshape(e, 3 * m, 3 * m)
        rows = np.repeat(dofs, 3 * m, axis=1).ravel()
        cols = np.tile(dofs, (1, 3 * m)).ravel()
        K = K + sps.csr_matrix((Ke.ravel(), (rows, cols)), shape=(nd, nd))  # per chunk: bounded memory
    return K


def traction_load(space, label, tfun, q):
    """f_ai = int_side R_a t_i(x, n) dS, with tfun mapping (k, 3) points and normals to (k, 3)."""
    b = jx.evaluate(space, jx.boundary_gauss(space, label, q))
    R, w = np.asarray(b.R), np.asarray(b.w)  # (e, q, m), (e, q)
    x, nrm = np.asarray(b.x), np.asarray(b.normal)
    e, nq, m = R.shape
    t = tfun(x.reshape(-1, 3), nrm.reshape(-1, 3)).reshape(e, nq, 3)
    fe = np.einsum("eq,eqa,eqi->eai", w, R, t).reshape(e, 3 * m)
    f = np.zeros(space.n_dofs)
    np.add.at(f, expand_dofs(np.asarray(b.dofs), 3).ravel(), fe.ravel())
    return f


def symmetry_dofs(space, tol=1e-12):
    """Vector dofs fixed by u_d = 0 on the plane x_d = 0 (control points on the plane)."""
    cp = np.asarray(space.cpts)
    return np.unique(np.concatenate([3 * np.nonzero(np.abs(cp[:, d]) < tol)[0] + d
                                     for d in range(3)]))


class Solver:
    """Factor K once on the free dofs; solve any number of load cases."""

    def __init__(self, K, fixed):
        self.n = K.shape[0]
        self.free = np.setdiff1d(np.arange(self.n), fixed)
        Kff = K[self.free][:, self.free].tocsc()
        self.lu = spla.splu(Kff, permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True))

    def solve(self, f):
        u = np.zeros(self.n)
        u[self.free] = self.lu.solve(f[self.free])
        return u


def grad_at(space, u, ps):
    """Displacement gradient G[c, d] = du_c/dx_d at the points of ps, shape (k, 3, 3)."""
    b = jx.evaluate(space, ps)
    dR = np.asarray(b.dR)  # (e, q, 3, m)
    ue = u[expand_dofs(np.asarray(b.dofs), 3)].reshape(dR.shape[0], -1, 3)  # (e, m, 3)
    return np.einsum("eqda,eac->eqcd", dR, ue).reshape(-1, 3, 3), np.asarray(b.x).reshape(-1, 3)


def values_at(space, u, ps):
    b = jx.evaluate(space, ps)
    R = np.asarray(b.R)
    ue = u[expand_dofs(np.asarray(b.dofs), 3)].reshape(R.shape[0], -1, 3)
    return np.einsum("eqa,eac->eqc", R, ue).reshape(-1, 3), np.asarray(b.x).reshape(-1, 3), \
        np.asarray(b.w).ravel()
