# Licensed under the PolyForm Noncommercial License 1.0.0 (see LICENSE).
# Required Notice: Copyright Arif Gökdaş (https://github.com/Krateian/Sutura)
"""Indexed mesh with per-face provenance, canonical ordering and hashing."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    V: np.ndarray       # (n,3) float64
    F: np.ndarray       # (m,3) int64
    src: np.ndarray     # (m,) int64 input facet id, -1 for faces created by repair
    attr: np.ndarray    # (m,) uint16 binary attribute word (carried with src)
    solid: np.ndarray   # (m,) int32 ASCII solid index (carried with src)

    @property
    def nf(self):
        return len(self.F)

    def copy(self):
        return Mesh(self.V.copy(), self.F.copy(), self.src.copy(), self.attr.copy(), self.solid.copy())

    def take(self, keep):
        """Face subset (mask or index array); vertices untouched."""
        return Mesh(self.V, self.F[keep], self.src[keep], self.attr[keep], self.solid[keep])

    def corners(self):
        return self.V[self.F]


def from_soup(soup) -> Mesh:
    """Bit-exact vertex merge only: same float64 value -> same index."""
    m = len(soup.tri)
    flat = soup.tri.reshape(-1, 3)
    if m == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64),
                    np.zeros(0, np.uint16), np.zeros(0, np.int32))
    V, inv = np.unique(flat, axis=0, return_inverse=True)
    F = np.asarray(inv, np.int64).reshape(m, 3)
    return Mesh(V, F, np.arange(m, dtype=np.int64), soup.attr.copy(), soup.solid.copy())


def compact(mesh: Mesh) -> Mesh:
    """Drop unreferenced vertices, keep relative vertex order."""
    if mesh.nf == 0:
        return Mesh(mesh.V[:0], mesh.F, mesh.src, mesh.attr, mesh.solid)
    used = np.unique(mesh.F)
    if len(used) == len(mesh.V):
        return mesh
    remap = np.full(len(mesh.V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(mesh.V[used], remap[mesh.F], mesh.src, mesh.attr, mesh.solid)


def canonical(mesh: Mesh) -> Mesh:
    """Order-independent representation.

    Vertices sorted lexicographically by coordinate, each face rotated so its
    smallest vertex index comes first (orientation kept), faces sorted by
    (vertex triple, src).  Two meshes that differ only in facet or vertex order
    map to identical V/F arrays.
    """
    mesh = compact(mesh)
    if mesh.nf == 0:
        return mesh
    order = np.lexsort((mesh.V[:, 2], mesh.V[:, 1], mesh.V[:, 0]))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    V = mesh.V[order]
    F = rank[mesh.F]
    r = np.argmin(F, axis=1)
    idx = (np.arange(3)[None, :] + r[:, None]) % 3
    F = np.take_along_axis(F, idx, axis=1)
    fo = np.lexsort((mesh.src, F[:, 2], F[:, 1], F[:, 0]))
    return Mesh(V, F[fo], mesh.src[fo], mesh.attr[fo], mesh.solid[fo])


def geometry_hash(mesh: Mesh) -> str:
    """SHA-256 of the canonical triangle set (coordinates + orientation),
    independent of facet order, vertex order and provenance."""
    c = canonical(mesh)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(c.V, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(c.F, dtype="<i8").tobytes())
    return h.hexdigest()
