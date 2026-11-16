"""Vector extraction of Ezeh & Susmel 2020 Figs 8 and 10 from the accepted manuscript (White Rose eprints 158161).

    python -m es2020.extract paper.pdf      -> data/fig8_markers.json, data/fig10_point_markers.json

Marker centres come from the PDF drawing operators (PyMuPDF get_drawings), mapped with each panel frame and its
printed decades. No pixels. Fig. 8 encodes the series in the marker: shape = angle (diamond 45, triangle 30,
circle 0), fill = R (white -1, grey 0); the six legend markers sit at N = 168 inside each frame and are dropped.
"""
import json
import os
import sys

import numpy as np

from .paper import DATA

SHAPE = {("l", "l", "l", "l"): 45, ("l", "l"): 30, ("c", "c", "c", "c"): 0}
FIG8 = {"sharp": (176.323, 412.000, 105.913, 256.649), "intermediate": (175.998, 411.899, 314.195, 465.143),
        "blunt": (176.323, 412.000, 522.422, 673.159)}           # frame x0, x1, y0, y1 (pt); N 1e2-1e7, MPa 100-1


def _markers(page, frame, size):
    x0, x1, y0, y1 = frame
    for d in page.get_drawings():
        r, fill = d["rect"], d.get("fill")
        if fill is None or abs(r.width - size) > 0.3 or not (x0 - 3 < r.x0 < x1 + 3 and y0 - 3 < r.y0 < y1 + 3):
            continue
        yield tuple(i[0] for i in d["items"]), fill, (r.x0 + r.x1 - 2 * x0) / 2 / (x1 - x0), (r.y0 + r.y1 - 2 * y0) / 2 / (y1 - y0)


def fig8(doc):
    out = []
    for notch, fr in FIG8.items():
        for shape, fill, u, v in _markers(doc[27], fr, 3.9):
            N, s = 10 ** (2 + 5 * u), 10 ** (2 - 2 * v)
            if abs(np.log10(N / 168.0)) < 0.004:
                continue                                          # legend
            out.append({"notch": notch, "p": SHAPE[shape], "R": -1 if fill[0] > 0.9 else 0,
                        "N": round(N, 1), "s_max": round(s, 4), "runout": N > 1.8e6})
    return out


def fig10(doc, panel=0):
    """panel 0 = Point Method (upper frame), 1 = Area Method (lower frame); N 1e1-1e7, y 10-0.01."""
    page = doc[29]
    fr = sorted([d["rect"] for d in page.get_drawings() if len(d["items"]) == 1 and d["items"][0][0] == "re"
                 and abs(d["rect"].width - 351.7) < 0.5], key=lambda r: r.y0)[panel]
    legend = [d["rect"] for d in page.get_drawings() if len(d["items"]) == 1 and d["items"][0][0] == "re"
              and abs(d["rect"].width - 88.6) < 0.5]
    kind = {("l", "l", "l", "l"): ("blunt", "diamond"), ("c", "c", "c", "c"): ("intermediate", "circle"),
            ("l", "l"): ("sharp", "triangle")}
    out = []
    for shape, fill, u, v in _markers(page, (fr.x0, fr.x1, fr.y0, fr.y1), 5.8):
        x, y = fr.x0 + u * fr.width, fr.y0 + v * fr.height
        if shape not in kind or any(L.x0 <= x <= L.x1 and L.y0 <= y <= L.y1 for L in legend):
            continue
        N = 10 ** (1 + 6 * u)
        out.append({"x": N, "y": 10 ** (1 - 3 * v), "type": kind[shape][0], "marker": kind[shape][1], "runout": N > 1.8e6})
    return out


def main(pdf):
    import pymupdf
    doc = pymupdf.open(pdf)
    src = "Ezeh & Susmel 2020 IJF 136:105583, accepted manuscript (White Rose eprints 158161)"
    json.dump({"source": src + ", Fig. 8 (page 28), vector marker centres",
               "units": {"N": "cycles", "s_max": "MPa, nominal net sigma_n,max as plotted in Fig. 8"},
               "points": fig8(doc)}, open(os.path.join(DATA, "fig8_markers.json"), "w"), indent=0)
    json.dump({"source": src + ", Fig. 10 Point Method panel (page 30), vector marker centres; x = N_f, "
               "y = sigma_eff,max / sigma_UTS", "points": fig10(doc)},
              open(os.path.join(DATA, "fig10_point_markers.json"), "w"), indent=1)
    json.dump({"source": src + ", Fig. 10 Area Method panel (page 30), vector marker centres; 3 sharp markers "
               "(N 429-504) are absent from this panel in the PDF", "points": fig10(doc, 1)},
              open(os.path.join(DATA, "fig10_area_markers.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
