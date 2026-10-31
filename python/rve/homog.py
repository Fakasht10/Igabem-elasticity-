"""Post-processing of app/rve_cell results: Voigt stiffness algebra, raster rotation, engineering constants.

Voigt order 11, 22, 33, 23, 13, 12 with engineering shear strains (the rve_cell / src/rve.hpp convention).
A cross-ply theta/(theta - 90) print is the 0/90 cell rotated by theta about the build axis z, so every raster
angle of one bead geometry is a rotation of one computed C (no new solve).
"""
import json

import numpy as np

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def to_tensor(C):
    """6x6 Voigt stiffness (engineering shear) -> C_ijkl."""
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for (p, q) in {(i, j), (j, i)}:
                for (r, s) in {(k, l), (l, k)}:
                    T[p, q, r, s] = C[a, b]
    return T


def to_voigt(T):
    return np.array([[T[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])


def rotate_z(C, theta_deg):
    """Stiffness of the cell rotated by theta about z (material axes turned by +theta)."""
    t = np.radians(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return to_voigt(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, to_tensor(np.asarray(C))))


def engineering(C):
    S = np.linalg.inv(np.asarray(C))
    return {"E1": 1 / S[0, 0], "E2": 1 / S[1, 1], "E3": 1 / S[2, 2], "G23": 1 / S[3, 3], "G13": 1 / S[4, 4],
            "G12": 1 / S[5, 5], "nu12": -S[0, 1] / S[0, 0], "nu13": -S[0, 2] / S[0, 0], "nu23": -S[1, 2] / S[1, 1]}


def cross_ply(C090, theta):
    """theta/(theta - 90) layup from the 0/90 cell: rotation by theta (0/90 -> theta = 0)."""
    return rotate_z(C090, theta)


def load(path):
    with open(path) as f:
        return json.load(f)


def isotropic_projection(C):
    """Closest isotropic stiffness in the Frobenius sense of the 4th-order tensor (bulk and shear moduli)."""
    T = to_tensor(np.asarray(C))
    I = np.eye(3)
    J = np.einsum("ij,kl->ijkl", I, I) / 3
    Is = 0.5 * (np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I))
    K = np.einsum("ijkl,ijkl", T, J) / 3          # <T, J> / <J, J> * (1/3) -> 3K J
    G = np.einsum("ijkl,ijkl", T, Is - J) / 10    # <T, K> / <K, K>, <K, K> = 5 -> 2G
    return K, G, 9 * K * G / (3 * K + G), (3 * K - 2 * G) / (2 * (3 * K + G))
