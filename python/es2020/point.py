"""TCD Point Method on IgBEM bisector profiles, as Ezeh & Susmel 2020 Sec. 5-6 apply it.

sigma_eff,max = sigma_n,max * K(L_M(N_f) / 2), K(d) = sigma_1(d) / sigma_n on the bisector, unit net stress (Eq. 2)
L_M(N_f) calibration (Fig. 9e): at each N*, L_M/2 is where K(d) = PLAIN(N*) / SHARP(N*); then fit A N^B.
"""
import json
import os

import numpy as np

from . import paper

ZMEAN = "mean"


class Profile:
    """K(d) along the bisector. `z` selects a thickness station index, or 'mean' (average over the stations,
    the closest 3D analogue of the paper's 2D model)."""

    def __init__(self, d, s):
        self.d, self.s = np.asarray(d, float), np.asarray(s, float)

    def __call__(self, r):
        return np.interp(np.clip(r, 0, self.d[-1]), self.d, self.s)

    def where(self, K):
        """Distance at which K(d) first drops to K (profile is monotone over the range used)."""
        s = np.minimum.accumulate(self.s)
        return np.interp(-np.asarray(K, float), -s, self.d)


def profiles(z=ZMEAN):
    js = json.load(open(os.path.join(paper.DATA, "igabem_bisector_profiles.json")))["profiles"]
    out = {}
    for k, v in js.items():
        s1 = np.asarray(v["s1"])
        out[k] = Profile(v["d"], s1.mean(0) if z == ZMEAN else s1[z])
    return out


def sigma_eff(prof, s_max, N, lm=paper.lm_paper):
    return s_max * prof(lm(N) / 2)


def fig10(prof, specs, lm=paper.lm_paper):
    """Our Fig. 10 ordinate for every specimen: list of (specimen, sigma_eff,max / UTS)."""
    return [(s, float(sigma_eff(prof[s.notch], s.s_max, s.N, lm)) / paper.UTS) for s in specs]


def calibrate(prof_sharp, plain=paper.PLAIN, notch=paper.SHARP_FIG8A, Ns=np.geomspace(1e3, 2e6, 60)):
    """Fig. 9e: L_M(N)/2 = d where K(d) = plain(N)/notch(N); returns (A, B, Ns, L)."""
    L = 2 * prof_sharp.where(plain(Ns) / notch(Ns))
    ok = (L > 0) & (L / 2 < 0.99 * prof_sharp.d[-1])
    B, lA = np.polyfit(np.log10(Ns[ok]), np.log10(L[ok]), 1)
    return 10 ** lA, B, Ns[ok], L[ok]
