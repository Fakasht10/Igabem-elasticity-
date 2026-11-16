"""Ezeh & Susmel 2020, Int J Fatigue 136:105583 (accepted manuscript, White Rose eprints 158161).

Every constant here is printed in the paper; the comment gives where. Nothing is refitted in this file.
"""
import json
import os
from dataclasses import dataclass

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")

UTS = 42.9                 # MPa, single average UTS (Sec. 2 and Sec. 6); the paper's normalisation
N_REF = 2e6                # cycles (Fig. 1, Sec. 6)


@dataclass(frozen=True)
class Basquin:
    """sigma(N) = s_ref (N / N_ref)^(-1/k), capped at `cap` (the plain curve is flat at UTS below its knee)."""
    s_ref: float
    k: float
    T: float = 1.0          # scatter ratio sigma_10% / sigma_90% at N_ref
    cap: float = np.inf

    def __call__(self, N, ps=0.5):
        f = {0.5: 1.0, 0.1: np.sqrt(self.T), 0.9: 1 / np.sqrt(self.T)}[ps]
        return np.minimum(f * self.s_ref * (np.asarray(N, float) / N_REF) ** (-1 / self.k), f * self.cap)

    def life(self, s):
        return N_REF * (self.s_ref / np.asarray(s, float)) ** self.k


# Sec. 6 (6.7 MPa, k 5.1) and Fig. 1 (T 2.451): plain P50 sigma_MAX,50% = 6.7 MPa at 2e6 (0.156 UTS, printed "0.16"), k 5.1, T_sigma 2.451
PLAIN = Basquin(6.7, 5.1, 2.451, cap=UTS)
# Fig. 1 / Sec. 1: unifying design curve sigma_MAX,Design = 0.1 UTS at 2e6, k 5.5 (P_S > 90 %); drawn flat below
# the same knee as the plain band (N where the plain P50 reaches UTS), as in Figs 1 and 10
N_KNEE = PLAIN.life(UTS)
DESIGN = Basquin(0.1 * UTS, 5.5, cap=0.1 * UTS * (N_KNEE / N_REF) ** (-1 / 5.5))
# Fig. 8a: sharp V notch, all angles and both R, nominal NET sigma_n,max: k 3.2, sigma_n,MAX-50% 3.9 MPa, T 2.229
SHARP_FIG8A = Basquin(3.9, 3.2, 2.229)
# Eq. (6): L_M(N_f) = 16.4 N_f^-0.242 mm, calibrated on PLAIN + SHARP_FIG8A by the Point Method (Fig. 9e)
LM_A, LM_B = 16.4, -0.242


def lm_paper(N):
    return LM_A * np.asarray(N, float) ** LM_B


# Table 1: n, k, sigma_n,A-50% (net amplitude, MPa) per notch / angle / R, for the data audit
TABLE1 = {("blunt", 45, -1): (13, 6.3, 9.2), ("blunt", 45, 0): (8, 8.3, 7.5), ("blunt", 30, -1): (9, 4.8, 6.7),
          ("blunt", 30, 0): (9, 7.5, 6.9), ("blunt", 0, -1): (10, 5.4, 8.2), ("blunt", 0, 0): (9, 7.3, 7.3),
          ("intermediate", 45, -1): (9, 4.3, 4.4), ("intermediate", 45, 0): (9, 5.0, 4.3),
          ("intermediate", 30, -1): (8, 4.8, 5.1), ("intermediate", 30, 0): (9, 5.4, 4.7),
          ("intermediate", 0, -1): (10, 4.1, 4.9), ("intermediate", 0, 0): (7, 4.6, 4.4),
          ("sharp", 45, -1): (8, 3.7, 4.1), ("sharp", 45, 0): (8, 4.5, 3.7), ("sharp", 30, -1): (8, 4.1, 4.8),
          ("sharp", 30, 0): (7, 4.7, 3.9), ("sharp", 0, -1): (8, 3.7, 4.3), ("sharp", 0, 0): (8, 5.0, 4.5)}

NOTCHES = ("sharp", "intermediate", "blunt")
LABEL = {"sharp": "Sharp (V, r 0.15)", "intermediate": "Intermediate (U, r 1)", "blunt": "Blunt (U, r 3)"}


@dataclass(frozen=True)
class Specimen:
    notch: str
    p: int                  # manufacturing angle (deg)
    R: float
    s_max: float            # nominal net sigma_n,max (MPa): Figs 5-7 sigma_n,a * 2/(1 - R), or as plotted in Fig. 8
    N: float
    runout: bool
    source: str             # "figs5-7" or "fig8": the two disagree for 73 of 157 specimens (README)


def specimens():
    """All 157 notched specimens (149 failures + 8 run-outs), vector-extracted from Figs 5-7."""
    js = json.load(open(os.path.join(DATA, "specimens_figs5to7.json")))
    return [Specimen(q["notch"], q["p"], float(q["R"]), t["sigma_na"] * 2 / (1 - q["R"]), t["N"], t["runout"],
                     "figs5-7") for q in js["series"] for t in q["specimens"]]


def fig10_markers():
    """Authors' Fig. 10 Point Method markers (vector-extracted): list of (notch, N, sigma_eff,max/UTS, runout)."""
    js = json.load(open(os.path.join(DATA, "fig10_point_markers.json")))
    return [(q["type"], q["x"], q["y"], q["runout"]) for q in js["points"]]


def fig8_specimens():
    """The 157 specimens as the authors plotted them in Fig. 8 (sigma_n,max, MPa) and post-processed into Fig. 10."""
    js = json.load(open(os.path.join(DATA, "fig8_markers.json")))
    return [Specimen(q["notch"], q["p"], float(q["R"]), q["s_max"], q["N"], q["runout"], "fig8") for q in js["points"]]


def pair(N1, v1, N2, v2, tol=0.004):
    """Match two plots of the same specimens: by N, and inside groups of equal N (|dlog N| < tol) by value rank.
    Returns index arrays (i, j) with N1[i] ~ N2[j]."""
    def order(N, v):
        o = np.lexsort((v, N))
        g = np.concatenate([[0], np.cumsum(np.diff(np.log10(np.asarray(N)[o])) > tol)])
        return o[np.lexsort((np.asarray(v)[o], g))]
    return order(N1, v1), order(N2, v2)
