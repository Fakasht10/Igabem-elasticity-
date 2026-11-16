import os; HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
"""Attribute the 195 plain points of Ezeh & Susmel 2020 Fig. 1 to their source.
Input (not redistributed): PDF of Ezeh & Susmel, Mat Design Process Comm 2019;1:e45 (doi 10.1002/mdp2.45). Its Fig. 3 (page 4, raster image)
plots ONLY the other-lab data: Letcher & Waytashek 2014 (R = -1) and Afrose et al. 2016 (R = 0).
A 2020 Fig. 1 point at R = -1 or 0 that sits on a Fig. 3 marker is other-lab; the rest (and all R = -0.5, 0.3 points) are the authors' own.
hit = fraction of a 3 px disc at the mapped position covered by marker pixels: > 0.8 other-lab, 0.3-0.8 ambiguous, <= 0.3 own-lab.
Run: PYTHONPATH=python python es2020/review/match_fig1_sources.py <mdp2.45.pdf>   -> fig1_source_match.json"""
import sys, json, numpy as np, pymupdf
from scipy import ndimage as ndi
from es2020 import paper
d = pymupdf.open(sys.argv[1]); p = d[3]; pm = pymupdf.Pixmap(d, p.get_images(full=True)[0][0])
a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, 0].astype(int)     # 667 x 405 grey image
m = a < 135; m[:33] = 0; m[344:] = 0; m[:, :92] = 0; m[:, 617:] = 0                                      # inside the plot frame
y, x = np.ogrid[-4:5, -4:5]; m = ndi.binary_opening(m, (x * x + y * y) <= 16)                            # keep filled markers, drop lines
px = lambda N: 89.5 + 88.17 * (np.log10(N) - 1); py = lambda s: 31.5 + 104.7 * (1 - np.log10(s))         # frame: N 10..1e7, s/UTS 10..0.01
yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]; out = []; cover = np.zeros_like(m)
for R, N, s in paper.fig1_plain():
    h = float(m[(xx - px(N)) ** 2 + (yy - py(s)) ** 2 <= 9].mean()); out.append(dict(R=float(R), N=float(N), s=float(s), hit=h))
    if h > 0.8 and R in (-1, 0): cover |= (xx - px(N)) ** 2 + (yy - py(s)) ** 2 <= 36
for R in (-1, -0.5, 0, 0.3):
    h = np.array([o["hit"] for o in out if o["R"] == R]); print("R", R, "n", len(h), "on marker", int((h > 0.8).sum()), "ambiguous", int(((h > 0.3) & (h <= 0.8)).sum()), "own", int((h <= 0.3).sum()))
print("Fig. 3 marker area explained by matched points:", round(float((m & cover).sum() / m.sum()), 3))
json.dump(out, open(HERE + "fig1_source_match.json", "w"))
