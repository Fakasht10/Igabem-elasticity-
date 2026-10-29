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
    """Factor or precondition K once on the free dofs; solve any number of load cases.

    method="amg": conjugate gradients preconditioned by smoothed-aggregation AMG (pyamg)
    built on the six rigid-body modes. Memory is O(nnz(K)). Rigid modes are exact
    coefficient vectors here because the NURBS basis is a partition of unity that
    reproduces x, so u = a + w x P_i at the control points P_i.
    method="pardiso": MKL PARDISO sparse Cholesky (pypardiso), the default. At p = 5,
    18k dofs: 10 s and 1.9 GB, against 262 s and 6.5 GB for SuperLU.
    method="lu": SuperLU. Robust, but its fill grows like 5-6x nnz(K) at p = 5 in 3D.
    """

    def __init__(self, K, fixed, method="pardiso", cpts=None, tol=1e-10):
        self.n, self.method, self.tol = K.shape[0], method, tol
        self.free = np.setdiff1d(np.arange(self.n), fixed)
        Kff = K[self.free][:, self.free]
        self.iters = []
        if method == "pardiso":
            import pypardiso

            self.pardiso = pypardiso.PyPardisoSolver(mtype=2)  # real SPD, Cholesky
            self.Kup = sps.triu(Kff, format="csr")
            self.pardiso.factorize(self.Kup)
            return
        if method == "lu":
            self.lu = spla.splu(Kff.tocsc(), permc_spec="MMD_AT_PLUS_A",
                                options=dict(SymmetricMode=True))
            return
        import pyamg

        # Keep all dofs and put identity rows/columns on the fixed ones, so K keeps its
        # 3x3 nodal blocks and the aggregation never splits the components of a node.
        d = np.ones(self.n)
        d[fixed] = 0.0
        D = sps.diags(d)
        Kc = (D @ K @ D + sps.diags(1.0 - d)).tobsr(blocksize=(3, 3))
        B = rigid_modes(cpts) * d[:, None]
        self.ml = pyamg.smoothed_aggregation_solver(Kc, B=B, symmetry="symmetric",
                                                    max_coarse=500)
        self.fixed_mask = d == 0.0

    def solve(self, f):
        u = np.zeros(self.n)
        if self.method == "pardiso":
            u[self.free] = self.pardiso.solve(self.Kup, f[self.free])
        elif self.method == "lu":
            u[self.free] = self.lu.solve(f[self.free])
        else:
            res = []
            u = self.ml.solve(np.where(self.fixed_mask, 0.0, f), tol=self.tol, accel="cg",
                              maxiter=500, residuals=res)
            self.iters.append(len(res) - 1)
        return u


def rigid_modes(cpts):
    """(3 n_cp, 6) translations and rotations sampled at the control points, interleaved."""
    cp = np.asarray(cpts)
    B = np.zeros((3 * len(cp), 6))
    for c in range(3):
        B[c::3, c] = 1.0
    x, y, z = cp[:, 0], cp[:, 1], cp[:, 2]
    B[0::3, 3], B[1::3, 3] = -y, x  # rotation about z
    B[1::3, 4], B[2::3, 4] = -z, y  # about x
    B[0::3, 5], B[2::3, 5] = z, -x  # about y
    return B


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
