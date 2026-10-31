"""Interactive printed-material parameter card (one self-contained HTML page) from rve/data/grid.json.

Same numbers as rve.api: the grid, its trilinear interpolation in (h/w, b_int/w, b_lat/h), linear in nu, the 0/90
mesh correction, the raster rotation and the net-section bond strengths are ported to JavaScript; tests compare
the page's engine with rve.api (test_rve.TestApi.test_page_engine).

usage (from python/): python3 -m rve.card_page out.html
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rve import api  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_page.html")


def payload():
    g = api._grid()[0]
    with open(api.REGISTRY) as f:
        mats = [dict(id=r["id"], name=r.get("name", r["id"]), E=r["E_MPa"], nu=r["nu"],
                     uts=(r.get("tcd") or {}).get("sigma_uts_MPa")) for r in json.load(f) if r.get("E_MPa")]
    rnd = lambda a: json.loads(json.dumps(a), parse_float=lambda s: float("%.6g" % float(s)))  # noqa: E731
    return dict(axes=g["axes"], c00=rnd(g["c00"]), c090=rnd(g["c090_g1"]), corr=rnd(g.get("c090_corr")),
                porosity=rnd(g["porosity"]), ffm=api.FFM_FACTOR, materials=mats)


def main(argv=None):
    out = (argv or sys.argv[1:])[0]
    with open(TEMPLATE) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(payload(), separators=(",", ":")))
    with open(out, "w") as f:
        f.write(html)
    print("wrote %s (%.0f kB)" % (out, os.path.getsize(out) / 1024))


if __name__ == "__main__":
    main()
