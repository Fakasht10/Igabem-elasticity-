"""Bracket with intentional bolt holes and named, protected traction/contact
regions: repair must recover every region from provenance and keep the
attribute words (geomio face-id convention) on the reloaded output."""
import numpy as np

import fixtures as fx
import oracles
from stl_repair_core import repair_stl, stlio
from stl_repair_core.mesh import canonical, from_soup

CODES = {"bolt_hole_1": 11, "bolt_hole_2": 12, "contact": 20, "load": 30}
GEOMIO_MARK = b"IGABEM geomio: attribute word = CAD face id"


def _defective(drop_hole_wall=False):
    V, F, reg = fx.bracket()
    tag = np.zeros(len(F), np.int64)
    for nm, ids in reg.items():
        tag[ids] = CODES[nm]
    keep = np.ones(len(F), bool)
    keep[reg["load"][5]] = False   # small hole in the loaded face (diameter 0.559 < limit 0.58 < bolt 0.6)
    if drop_hole_wall:
        keep[reg["bolt_hole_2"]] = False                 # open bolt hole: two loops of diameter 0.6
    F2 = F.copy()
    F2[reg["bolt_hole_1"][3]] = F2[reg["bolt_hole_1"][3]][[0, 2, 1]]   # reversed protected face
    extra = [reg["contact"][7]]                          # duplicated contact face
    Fd = np.concatenate([F2[keep], F2[extra]])
    tags = np.concatenate([tag[keep], tag[extra]])
    regions = {nm: np.flatnonzero(tags == c).tolist() for nm, c in CODES.items() if (tags == c).any()}
    return V, Fd, tags, regions


def _clean_volume(tmp_path):
    V, F, _ = fx.bracket()
    p = fx.soup_write(tmp_path / "clean.stl", V, F)
    mf = oracles.manifold_check(p)
    assert mf["ok"] and mf["genus"] == 2
    return mf["volume"]


def test_bracket_repair_recovers_regions(tmp_path):
    vol = _clean_volume(tmp_path)
    V, F, tags, regions = _defective()
    p = tmp_path / "bracket.stl"
    stlio.write_stl(p, V, F, attr=tags, header=GEOMIO_MARK)
    out = tmp_path / "out.stl"
    rep = repair_stl(p, out, max_hole_diameter=0.58, protected_faces=regions)
    assert rep["status"] == "PASS", rep["problems"]
    ops = [o["operation"] for o in rep["adopted_operations"]]
    assert {"remove_duplicates", "reorient", "fill_hole"} <= set(ops)
    # independent verification of the reloaded file
    mf = oracles.manifold_check(out)
    assert mf["ok"] and mf["genus"] == 2                   # both bolt holes still through-holes
    assert abs(mf["volume"] - vol) <= 1e-6 * vol
    assert oracles.meshlab_check(out)["self_intersecting_faces"] == 0
    # provenance: every protected input facet is recovered with its full area
    from stl_repair_core.geometry import tri_areas
    for nm, r in rep["provenance"]["regions"].items():
        assert r["input_faces_recovered"] == r["input_faces"] - (1 if nm == "contact" else 0), (nm, r)
        Fr = np.unique(np.sort(F[regions[nm]], axis=1), axis=0)          # distinct input triangles
        exp = float(tri_areas(V, Fr).sum())
        assert abs(r["output_area"] - exp) <= 1e-12 * exp, (nm, r)
    # attribute words survive the round trip, new patch faces are untagged (0)
    s = stlio.read_stl(out)
    assert s.header.startswith(GEOMIO_MARK)
    m = canonical(from_soup(s))
    prov = np.load(str(out) + ".provenance.npz")["output_face_to_input_facet"]
    assert len(prov) == m.nf
    for nm, c in CODES.items():
        ids = regions[nm]
        assert (s.attr == c).sum() == len(ids) - (1 if nm == "contact" else 0)
    new_faces = prov < 0
    assert new_faces.sum() == rep["provenance"]["output_new"] > 0
    assert (s.attr[new_faces] == 0).all()                        # patch faces untagged
    assert np.array_equal(s.attr[~new_faces], tags[prov[~new_faces]])  # file order == provenance order


def test_open_bolt_hole_stays_open(tmp_path):
    V, F, tags, regions = _defective(drop_hole_wall=True)
    p = tmp_path / "open.stl"
    stlio.write_stl(p, V, F, attr=tags, header=GEOMIO_MARK)
    out = tmp_path / "out.stl"
    rep = repair_stl(p, out, max_hole_diameter=0.58, protected_faces=regions)
    assert rep["status"] == "PASS", rep["problems"]
    assert rep["output"]["intentional_openings"] == 2 and rep["after"]["boundary_loops"] == 2
    assert not rep["output"]["watertight"]
    loops = sorted(L["diameter"] for L in rep["after"]["boundary_loop_list"])
    assert all(d > 0.58 for d in loops)
    ml = oracles.meshlab_check(out)
    assert ml["holes"] == 2 and ml["self_intersecting_faces"] == 0


def test_protect_tags_flag(tmp_path):
    V, F, tags, regions = _defective()
    p = tmp_path / "bracket.stl"
    stlio.write_stl(p, V, F, attr=tags, header=GEOMIO_MARK)
    rep = repair_stl(p, tmp_path / "o.stl", max_hole_diameter=0.58, protect_tags=True)
    assert rep["status"] == "PASS"
    assert set(rep["provenance"]["regions"]) == {"attr:%d" % c for c in CODES.values()}
