"""Laminate layer: the stiffness of an arbitrary stacking sequence of printed layers from one lamina C.

A layer of raster angle theta is the lamina (the 0/0 bead cell: every layer along x, one layer high, with its own
interlayer weld) turned by theta about the build axis z. A stack of such layers is a periodic laminate in z; its
exact homogenized 3D stiffness (perfect interfaces between the cells, every layer infinite in plane) follows from
the two continuity conditions of a layered medium:

    in-plane strains     e_i = (e11, e22, g12)  equal in every layer
    out-of-plane stress  s_o = (s33, s23, s13)  equal in every layer

so with each layer's C split into i / o blocks, s_i = (Cii - Cio Coo^-1 Coi) e_i + Cio Coo^-1 s_o and
e_o = Coo^-1 (s_o - Coi e_i), the thickness averages of these four blocks give the stack (stack_C). Under in-plane
load (s_o = 0) this is classical laminate theory's membrane stiffness A / t exactly (B = 0: a repeated sequence,
[...]_R with R -> infinity; a finite unsymmetric stack is softer by B A^-1 B, 0.5 % at R = 1 for the Agotnes 2026
layup). The interlayer welds sit inside each lamina, so they only add compliance to the out-of-plane blocks: the
in-plane laminate moduli do not depend on the interlayer weld stiffness (python/rve/clt_check.py measures it).

Approximation: a layer between two layers of other angles bonds to them on crossing contacts, not on the aligned
strip of the 0/0 cell. clt_check.py measures the error on the one stack with both geometries, +-45 (0/0 laminae
stacked here vs the 0/90 bead cell turned 45 deg).

    from rve.laminate import layup, stack_C, inplane
    C = stack_C(C_lamina, layup("[0, +-30, +-60, 0, 90]"))     # 6 x 6, Voigt 11 22 33 23 13 12, engineering shear
    inplane(C)   # Ex, Ey, Gxy, nuxy under plane stress (the coupon moduli)
"""
import re

import numpy as np

from rve.homog import rotate_z

I_, O_ = [0, 1, 5], [2, 3, 4]   # Voigt rows: in-plane (11, 22, 12), out-of-plane (33, 23, 13)


def layup(spec):
    """Angles (deg) of a stacking sequence: "[0, +-30, +-60, 0, 90]" (+- / ± give +a then -a; a trailing _R or
    _3 repeats; _s mirrors) or a list of angles."""
    if not isinstance(spec, str):
        return [float(a) for a in spec]
    m = re.fullmatch(r"\s*\[(.*)\]\s*(?:_?\{?(\w+)\}?)?\s*", spec)
    if not m:
        raise ValueError("bad layup %r" % spec)
    out = []
    for tok in m.group(1).split(","):
        tok = tok.strip().replace("±", "+-")
        if tok.startswith("+-") or tok.startswith("-+"):
            a = float(tok[2:])
            out += [a, -a] if tok[0] == "+" else [-a, a]
        else:
            out.append(float(tok))
    rep = m.group(2)
    if rep in (None, "R", "r"):
        return out
    if rep in ("s", "S"):
        return out + out[::-1]
    return out * int(rep)


def stack_C(C_lamina, angles, thickness=None):
    """Homogenized 3D stiffness of the periodic stack of the lamina turned by each angle (module docstring)."""
    angles = list(angles)
    t = np.ones(len(angles)) if thickness is None else np.asarray(thickness, float)
    t = t / t.sum()
    A = np.zeros((3, 3)); Bm = np.zeros((3, 3)); Cm = np.zeros((3, 3)); D = np.zeros((3, 3))
    for a, tk in zip(angles, t):
        C = rotate_z(C_lamina, a)
        Cii, Cio, Coi, Coo = C[np.ix_(I_, I_)], C[np.ix_(I_, O_)], C[np.ix_(O_, I_)], C[np.ix_(O_, O_)]
        Ki = np.linalg.inv(Coo)
        A += tk * (Cii - Cio @ Ki @ Coi)   # s_i = A e_i + B s_o
        Bm += tk * (Cio @ Ki)
        Cm += tk * (Ki @ Coi)              # e_o = D s_o - Cm e_i
        D += tk * Ki
    # invert the mixed form: s_o = D^-1 (e_o + Cm e_i), s_i = (A + B D^-1 Cm) e_i + B D^-1 e_o
    Di = np.linalg.inv(D)
    out = np.zeros((6, 6))
    out[np.ix_(I_, I_)] = A + Bm @ Di @ Cm
    out[np.ix_(I_, O_)] = Bm @ Di
    out[np.ix_(O_, I_)] = Di @ Cm
    out[np.ix_(O_, O_)] = Di
    return 0.5 * (out + out.T)


def inplane(C):
    """Plane-stress (s33 = s23 = s13 = 0) in-plane moduli of a 6 x 6 stiffness: Ex, Ey, Gxy, nuxy."""
    S = np.linalg.inv(np.asarray(C))
    return dict(Ex=1 / S[0, 0], Ey=1 / S[1, 1], Gxy=1 / S[5, 5], nuxy=-S[0, 1] / S[0, 0])


def lamina_from_engineering(E1, E2, G12, nu12, E3=None, G13=None, G23=None, nu13=None, nu23=0.4):
    """Orthotropic 6 x 6 stiffness from engineering constants (out-of-plane ones default to transverse isotropy
    about 1: E3 = E2, G13 = G12, nu13 = nu12, G23 = E2 / (2 (1 + nu23))); for classical-laminate checks, whose
    in-plane results do not depend on the out-of-plane constants."""
    E3 = E2 if E3 is None else E3
    G13 = G12 if G13 is None else G13
    nu13 = nu12 if nu13 is None else nu13
    G23 = E2 / (2 * (1 + nu23)) if G23 is None else G23
    S = np.zeros((6, 6))
    S[0, 0], S[1, 1], S[2, 2] = 1 / E1, 1 / E2, 1 / E3
    S[0, 1] = S[1, 0] = -nu12 / E1
    S[0, 2] = S[2, 0] = -nu13 / E1
    S[1, 2] = S[2, 1] = -nu23 / E2
    S[3, 3], S[4, 4], S[5, 5] = 1 / G23, 1 / G13, 1 / G12
    return np.linalg.inv(S)
