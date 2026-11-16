"""pooled 195-plain Ezeh-Susmel dataset exactly as review reconcile.py builds it (copied so it can be imported without running reconcile)."""
import numpy as np, rank
def pooled():
    import sde_tcd as s
    d = rank.load_es(); nm = {"plain": "plain", "sharp": "V", "intermediate": "U1", "blunt": "U3"}
    d.g = np.array([nm[x] for x in s.notch]); d.R, d.S, d.lnN, d.ro = s.R, s.S, s.lnN, s.ro
    n = len(d.S); d.dv = np.zeros(n); d.phi = np.ones(n); d.lev = np.zeros(n, int); d.fit = np.isin(d.g, ("plain", "V")); return d
