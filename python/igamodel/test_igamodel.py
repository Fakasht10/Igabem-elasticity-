"""Fast checks of the model tree (no solve): the round trip with every case file in
cases/ and every form case_io.hpp documents, the refusals, undo/redo and replay.

usage: python3 -m unittest igamodel.test_igamodel      (from python/; < 1 s)

The solver-level gate (bc_case on a tree-written case gives the same CSV and VTU
as on the hand-written one) is in python/igamodel/README.md.
"""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from . import History, Model, ModelError, from_case, node_type, read_case, rename, resolve, to_case, write_case
from .casefile import strip_comments
from .model import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[2]
BOX = {"box": [[-2, -1, -1], [-1.1, 1, 1]]}
# Every support and load form documented at the top of src/case_io.hpp.
ALL_FORMS = {
    "geometry": "geo/step/bracket_exact_p3_h0.7_c3.dat", "faces": "geo/step/bracket.faces", "cache": "none",
    "material": {"E": 2000, "nu": 0.35}, "solver": {"n_gauss": 12, "far_tol": 1e-10},
    "grade": {"faces": [3], "inner": [1, 13], "h0": 0.025, "growth": 1.4},
    "supports": [
        {"face_id": 3, "fix": ["ux", "uy", "uz"]},
        {"label": "cut", "select": BOX, "fix": ["uz"], "value": [0, 0, -0.01]},
        {"symmetry": "y", "at": 0},
        {"face_id": 7, "spring": {"k_n": 1e3, "k_t": 0}},
        {"face_ids": [8, 9], "dofs": {"ux": True, "uy": False}, "snap": True},
        {"patches": [4], "spring": {"K_total": 50.0}},
        {"label": "pin", "face_id": 12, "rbe2": {"ref": [0, 0, 1], "hold": ["uz", "rx"], "k": [1, 2, 3, 4, 5, 6]}},
        {"select": [{"plane": {"point": [0, 0, 0], "normal": [0, 0, 1], "tol": 1e-3}},
                    {"cylinder": {"point": [0, 0, 0], "axis": [0, 0, 1], "radius": 0.2}}],
         "spring": {"k": [1, 1, 1]}},
    ],
    "cases": [
        {"name": "pull", "loads": [
            {"face_id": 5, "force_N": 100, "direction": [1, 0, 0]},
            {"select": {"half": {"point": [0, 0, 0], "normal": [1, 0, 0]}, "faces": [1]}, "pressure": 2.0},
            {"face_id": 5, "moment": [0, 0, 10], "about": [1, 2, 3]},
            {"face_id": 9, "bearing": {"force_N": 50, "direction": [1, 0, 0]}},
            {"select": {"ball": {"center": [0, 0, 0], "radius": 1}, "normal": {"direction": [0, 0, 1]}},
             "traction": [0.1, 0, 0]},
            {"rbe2": "pin", "force_N": 10, "direction": [0, 1, 0], "moment": [0, 0, 1]}]},
        {"loads": [{"face_id": 5, "force_N": 1}]},
    ],
    "loads": [{"face_id": 6, "pressure": 1.0}],
    "output": {"vtu": "out/bracket", "csv": "out/bracket.csv"},
}


def case_files():
    return sorted((ROOT / "cases").glob("*.json"))


class RoundTrip(unittest.TestCase):
    def test_every_case_file_reads_into_the_tree_and_writes_back_the_same_case(self):
        files = case_files()
        self.assertGreaterEqual(len(files), 6)
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                text, notes = strip_comments(f.read_text(encoding="utf-8"))
                m = read_case(f)
                self.assertEqual(to_case(m), json.loads(text), f.name)
                out = write_case(m, pathlib.Path(tmp) / f.name)
                again, notes2 = strip_comments(out.read_text(encoding="utf-8"))
                self.assertEqual(json.loads(again), json.loads(text), f.name)
                self.assertEqual(notes2, notes, f"{f.name}: the comment block survives")

    def test_a_hand_formatted_case_writes_back_byte_for_byte(self):
        f = ROOT / "cases" / "bracket_bolted.json"
        with tempfile.TemporaryDirectory() as tmp:
            out = write_case(read_case(f), pathlib.Path(tmp) / "x.json")
            self.assertEqual(out.read_bytes(), f.read_bytes())

    def test_every_documented_form_round_trips(self):
        self.assertEqual(to_case(from_case(copy.deepcopy(ALL_FORMS))), ALL_FORMS)

    def test_the_tree_survives_its_own_model_file(self):
        m = from_case(copy.deepcopy(ALL_FORMS), "notes\nline 2")
        with tempfile.TemporaryDirectory() as tmp:
            m.save(pathlib.Path(tmp) / "m.json")
            m2 = Model.load(pathlib.Path(tmp) / "m.json")
        self.assertEqual(m2.snapshot(), m.snapshot())
        self.assertEqual(to_case(m2), ALL_FORMS)

    def test_comments_inside_strings_are_text(self):
        text, notes = strip_comments('// head\n{"geometry": "a//b.dat", // tail\n "x": 1}')
        self.assertEqual(notes, "head")
        self.assertEqual(json.loads(text), {"geometry": "a//b.dat", "x": 1})


def refused(case_edit):
    c = copy.deepcopy(ALL_FORMS)
    case_edit(c)
    return c


class Refusals(unittest.TestCase):
    def test_a_case_bc_case_would_misread_is_refused(self):
        bad = {
            "unknown top key": lambda c: c.update(meshing={}),
            "unknown support key": lambda c: c["supports"][0].update(fixx=["ux"]),
            "two support behaviours": lambda c: c["supports"][0].update(spring={"k_n": 1}),
            "symmetry with a region": lambda c: c["supports"][2].update(face_id=3),
            "support with nothing": lambda c: c["supports"][0].pop("fix"),
            "support without a region": lambda c: c["supports"][0].pop("face_id"),
            "bad fix component": lambda c: c["supports"][0].update(fix=["rx"]),
            "bad spring key": lambda c: c["supports"][3].update(spring={"kn": 1}),
            "unknown selector": lambda c: c["supports"][1].update(select={"sphere": {}}),
            "box with one corner": lambda c: c["supports"][1].update(select={"box": [[0, 0, 0]]}),
            "selector list with a face": lambda c: c["supports"][7].update(face_id=1),
            "two load kinds": lambda c: c["cases"][0]["loads"][1].update(force_N=1),
            "load with nothing": lambda c: c["cases"][0]["loads"][0].pop("force_N"),
            "rbe2 load on a missing master": lambda c: c["cases"][0]["loads"][5].update(rbe2="bolt"),
            "rbe2 load with a region": lambda c: c["cases"][0]["loads"][5].update(face_id=1),
            "non-finite number": lambda c: c["cases"][0]["loads"][0].update(force_N=float("inf")),
            "load case extra key": lambda c: c["cases"][0].update(scale=2),
            "no geometry": lambda c: c.pop("geometry"),
        }
        for why, edit in bad.items():
            with self.subTest(why), self.assertRaises(ModelError):
                from_case(refused(edit))


class Operations(unittest.TestCase):
    def setUp(self):
        self.m = from_case(copy.deepcopy(ALL_FORMS))
        self.h = History(self.m)

    def test_a_refused_operation_changes_nothing(self):
        sup = self.m.of_kind("support")[0].tag
        for op in ({"op": "set", "tag": sup, "key": "spring", "value": {"k_n": 1}},
                   {"op": "set", "tag": sup, "key": "fix", "value": ["ux", "rz"]},
                   {"op": "select", "tag": sup, "region": {"face_id": 1.5}},
                   {"op": "add", "parent": "root", "kind": "material"},
                   {"op": "add", "parent": "root", "kind": "load", "region": {"face_id": 1}, "props": {"pressure": 1}},
                   {"op": "remove", "tag": "root"}, {"op": "remove", "tag": "nope"}, {"op": "rename", "tag": sup}):
            before = self.m.snapshot()
            with self.subTest(op=op), self.assertRaises(ModelError):
                self.h.do(op)
            self.assertEqual(self.m.snapshot(), before)
        self.assertEqual(self.h.log, [])

    def test_switching_a_support_goes_through_an_unfinished_state_that_cannot_be_written(self):
        sup = self.m.of_kind("support")[0].tag
        self.h.do({"op": "set", "tag": sup, "key": "fix", "value": None})
        with self.assertRaisesRegex(ModelError, "has no fix"):
            to_case(self.m)
        self.h.do({"op": "set", "tag": sup, "key": "spring", "value": {"k_n": 10.0}})
        self.assertEqual(to_case(self.m)["supports"][0], {"face_id": 3, "spring": {"k_n": 10.0}})

    def test_undo_redo_and_replay(self):
        s0 = self.m.snapshot()
        lc = self.h.do({"op": "add", "parent": "root", "kind": "loadcase", "props": {"name": "new"}})
        self.h.do({"op": "add", "parent": lc, "kind": "load", "region": {"face_id": 7}, "props": {"force_N": 2.0}})
        s2 = self.m.snapshot()
        self.assertTrue(self.h.undo() and self.h.undo())
        self.assertFalse(self.h.undo())
        self.assertEqual(self.m.snapshot(), s0)
        self.assertTrue(self.h.redo() and self.h.redo())
        self.assertEqual(self.m.snapshot(), s2)
        fresh = from_case(copy.deepcopy(ALL_FORMS))
        fresh.replay(copy.deepcopy(self.h.log))
        self.assertEqual(fresh.snapshot(), s2, "tags are deterministic, so a journal replays to the same tree")
        self.assertEqual(to_case(fresh)["cases"][-1], {"name": "new", "loads": [{"face_id": 7, "force_N": 2.0}]})


NAMED = {
    "geometry": "g.dat", "names": {"clamp": {"face_id": 3}, "slot": {"face_ids": [8, 9], "select": BOX}},
    "supports": [{"label": "fixed", "name": "clamp", "fix": ["ux", "uy", "uz"]}],
    "cases": [{"name": "a", "loads": [{"name": "slot", "pressure": 1.0}, {"name": "clamp", "force_N": 2.0}]}],
}


class Names(unittest.TestCase):
    def setUp(self):
        self.m = from_case(copy.deepcopy(NAMED))
        self.h = History(self.m)

    def test_named_surfaces_round_trip(self):
        self.assertEqual(to_case(self.m), NAMED)
        self.assertEqual(resolve(self.m, {"name": "slot"}), NAMED["names"]["slot"])

    def test_rename_updates_every_reference_in_one_undo_step(self):
        s0 = self.m.snapshot()
        self.h.do(rename(self.m, "clamp", "base"))
        c = to_case(self.m)
        self.assertEqual(list(c["names"]), ["base", "slot"])
        self.assertEqual(c["supports"][0]["name"], "base")
        self.assertEqual(c["cases"][0]["loads"][1]["name"], "base")
        self.assertEqual(c["cases"][0]["loads"][0]["name"], "slot")
        self.assertTrue(self.h.undo())
        self.assertEqual(self.m.snapshot(), s0)
        self.assertFalse(self.h.undo(), "a rename is one step")

    def test_name_refusals(self):
        bad = {
            "unknown name": lambda c: c["supports"][0].update(name="clmp"),
            "name with a face": lambda c: c["supports"][0].update(face_id=3),
            "nested name": lambda c: c["names"].update(other={"name": "clamp"}),
            "empty name": lambda c: c["supports"][0].update(name=""),
            "names not an object": lambda c: c.update(names=[]),
        }
        for why, edit in bad.items():
            c = copy.deepcopy(NAMED)
            edit(c)
            with self.subTest(why), self.assertRaises(ModelError):
                from_case(c)
        for args in (("nope", "x"), ("clamp", "slot")):
            with self.subTest(args), self.assertRaises(ModelError):
                rename(self.m, *args)

    def test_a_refused_batch_changes_nothing(self):
        sel = self.m.of_kind("selection")[0].tag
        before = self.m.snapshot()
        with self.assertRaises(ModelError):
            self.h.do({"op": "batch", "ops": [{"op": "set", "tag": sel, "key": "name", "value": "x"},
                                              {"op": "remove", "tag": "nope"}]})
        self.assertEqual(self.m.snapshot(), before)
        sup = self.m.of_kind("support")[0].tag
        self.h.do({"op": "set", "tag": sel, "key": "name", "value": "x"})   # leaves a dangling reference
        with self.assertRaisesRegex(ModelError, f"{sup} refers to 'clamp'"):
            to_case(self.m)
        self.m.apply({"op": "add", "parent": "root", "kind": "selection", "props": {"name": "x"},
                      "region": {"face_id": 1}})
        with self.assertRaisesRegex(ModelError, "two surfaces are named 'x'"):
            to_case(self.m)


class Core(unittest.TestCase):
    def test_every_node_type_has_a_json_schema(self):
        for kind in REGISTRY:
            json.dumps(node_type(kind).schema())
        s = node_type("support").schema()["properties"]
        self.assertEqual(s["fix"]["items"]["enum"], ["ux", "uy", "uz"])
        self.assertEqual(s["rbe2"]["propertyNames"]["enum"], ["ref", "hold", "k"])

    def test_the_core_imports_nothing_heavy(self):
        code = ("import sys, igamodel; heavy = {'numpy', 'scipy', 'gmsh', 'OCP', 'failure', 'geomio'};"
                "print(sorted(heavy & set(sys.modules)))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             cwd=ROOT / "python")
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
