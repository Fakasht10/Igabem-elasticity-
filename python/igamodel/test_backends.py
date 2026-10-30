"""Fast checks of the backend layer and the assessment nodes (no solve, < 1 s).

usage: python3 -m unittest igamodel.test_backends      (from python/)

The solver-level gate (failure loads through the tree equal validate_cicero's)
is igamodel/gate_cicero.py; see python/igamodel/README.md.
"""
import copy
import pathlib
import tempfile
import unittest

import numpy as np

from . import Model, ModelError, from_case, to_case
from . import backends, run
from .test_igamodel import ALL_FORMS


def senb_like():
    m = from_case(copy.deepcopy(ALL_FORMS))
    m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": "geo/step/senb_rho1.step", "h": 3.0}})
    m.apply({"op": "add", "parent": "root", "kind": "assessment",
             "props": {"label": "root", "point": [22, 5, 0], "K_Ic": 111.9, "sigma_uts": 95.2, "rho": 1.0}})
    return m


class Nodes(unittest.TestCase):
    def test_cad_and_assessment_live_in_the_model_file_not_the_case_file(self):
        m = senb_like()
        self.assertEqual(to_case(m), ALL_FORMS)
        with tempfile.TemporaryDirectory() as tmp:
            m.save(pathlib.Path(tmp) / "m.json")
            m2 = Model.load(pathlib.Path(tmp) / "m.json")
        self.assertEqual(m2.snapshot(), m.snapshot())
        self.assertEqual(m2.of_kind("assessment")[0].props["point"], [22, 5, 0])

    def test_assessment_refusals(self):
        m = senb_like()
        tag = m.of_kind("assessment")[0].tag
        for op in ({"op": "set", "tag": tag, "key": "card", "value": "ABS-M30-iso"},     # card and K_Ic
                   {"op": "set", "tag": tag, "key": "state", "value": "plane"},
                   {"op": "set", "tag": tag, "key": "point", "value": [1, 2]},
                   {"op": "add", "parent": "root", "kind": "cad"}):                      # one cad only
            before = m.snapshot()
            with self.subTest(op=op), self.assertRaises(ModelError):
                m.apply(op)
            self.assertEqual(m.snapshot(), before)

    def test_strength_from_a_card_or_from_the_node(self):
        from failure import materials
        from failure.fracture import L_tcd
        c = materials.card("ABS-M30-iso")
        mat = {"E": c["E_MPa"], "nu": c["nu"]}
        m = senb_like()
        a = m.of_kind("assessment")[0]
        self.assertEqual(run.strength(a, mat)[:2], (111.9, 95.2))
        m.apply({"op": "set", "tag": a.tag, "key": "K_Ic", "value": None})
        with self.assertRaisesRegex(ModelError, "needs a card"):
            run.strength(m.find(a.tag), {"E": 1, "nu": 0.3})
        m.apply({"op": "set", "tag": a.tag, "key": "card", "value": "ABS-M30-iso"})
        K, s, _ = run.strength(m.find(a.tag), mat)
        self.assertEqual((K, s), (c["tcd"]["K_Ic_MPa_sqrt_mm"], 95.2), "sigma_uts on the node overrides the card")
        self.assertGreater(L_tcd(K, s), 0)
        with self.assertRaisesRegex(ModelError, "card ABS-M30-iso has E"):
            run.strength(m.find(a.tag), {"E": 2241.0, "nu": c["nu"]})


class Weibull(unittest.TestCase):
    def test_weibull_parameters_from_a_card_or_the_node(self):
        from failure import materials
        c = materials.card("pla-raster-0-weibull")
        mat = {"E": c["E_MPa"], "nu": c["nu"]}
        m = senb_like()
        tag = m.apply({"op": "add", "parent": "root", "kind": "weibull", "props": {"card": "pla-raster-0-weibull"}})
        self.assertEqual(run.weibull_params(m.find(tag), mat)[:3],
                         (c["weibull"]["m"], c["weibull"]["S0_MPa"], c["weibull"]["V0_mm3"]))
        with self.assertRaisesRegex(ModelError, "card pla-raster-0-weibull has E"):
            run.weibull_params(m.find(tag), {"E": 1.0, "nu": c["nu"]})
        with self.assertRaisesRegex(ModelError, "not both"):
            m.apply({"op": "set", "tag": tag, "key": "m", "value": 5.0})
        m.apply({"op": "set", "tag": tag, "key": "card", "value": None})
        with self.assertRaisesRegex(ModelError, "needs a card, or m and S0"):
            run.weibull_params(m.find(tag), mat)
        m.apply({"op": "batch", "ops": [{"op": "set", "tag": tag, "key": "m", "value": 5.0},
                                        {"op": "set", "tag": tag, "key": "S0", "value": 100.0}]})
        self.assertEqual(run.weibull_params(m.find(tag), mat), (5.0, 100.0, 1.0, "m, S0 set"))

    def test_weibull_refusals_before_any_work(self):
        m = from_case(copy.deepcopy(ALL_FORMS))
        m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": "x.step"}})
        tag = m.apply({"op": "add", "parent": "root", "kind": "weibull",
                       "props": {"m": 5.0, "S0": 100.0, "probability": 1.0}})
        with self.assertRaisesRegex(ModelError, "strictly between 0 and 1"):
            run.assess(m, ".", "/nonexistent")
        m.apply({"op": "batch", "ops": [{"op": "set", "tag": tag, "key": "probability", "value": 0.1},
                                        {"op": "set", "tag": tag, "key": "focus_point", "value": [0, 0, 0]}]})
        with self.assertRaisesRegex(ModelError, "focus_point, focus_radius and focus_h"):
            run.assess(m, ".", "/nonexistent")


class Capabilities(unittest.TestCase):
    def test_features_name_what_the_model_uses(self):
        f = backends.features(senb_like())
        self.assertTrue({"fixed", "prescribed_displacement", "symmetry", "spring", "rbe2", "pressure", "traction",
                         "bearing", "rbe2_load", "force", "moment", "graded_mesh", "cad_geometry",
                         "stress_at_points"} <= f)
        self.assertLessEqual(f, backends.get("igabem").capabilities)

    def test_a_backend_refuses_what_it_cannot_solve(self):
        class FixedOnly:
            name, capabilities = "toy", frozenset({"fixed", "force"})
        with self.assertRaisesRegex(ModelError, r"toy backend cannot solve \[.*'rbe2'"):
            backends.check(FixedOnly(), senb_like())
        with self.assertRaises(ModelError):
            backends.get("nope")

    def test_assess_refuses_before_any_work(self):
        m = from_case(copy.deepcopy(ALL_FORMS))
        with self.assertRaisesRegex(ModelError, "no assessment"):
            run.assess(m, ".", "/nonexistent")
        m.apply({"op": "add", "parent": "root", "kind": "assessment", "props": {"K_Ic": 1.0, "sigma_uts": 1.0}})
        with self.assertRaisesRegex(ModelError, "no cad"):
            run.assess(m, ".", "/nonexistent")
        m.apply({"op": "add", "parent": "root", "kind": "cad", "props": {"step": "x.step"}})
        with self.assertRaisesRegex(ModelError, "needs a point"):
            run.assess(m, ".", "/nonexistent")


class FakeSolve:
    cases, seconds, report = ["a", "b"], 0.0, []

    def __init__(self):
        self.calls = 0

    def stress_all(self, P, on):
        self.calls += 1
        s = np.repeat(np.eye(3)[None], len(P), 0) * P[:, :1, None]
        return {"a": (s, np.ones(len(P))), "b": (2 * s, np.ones(len(P)))}

    def close(self):
        pass


class Runs(unittest.TestCase):
    def test_load_cases_share_one_evaluation_and_callers_cannot_spoil_it(self):
        fake = FakeSolve()
        r = backends.Run(fake)
        P = np.arange(12.0).reshape(4, 3)
        sa, _ = r.view("a").stress(P, True)
        sb, _ = r.view("b").stress(P, True)
        self.assertEqual(fake.calls, 1)
        np.testing.assert_array_equal(sb, 2 * sa)
        sa[:] = -1
        np.testing.assert_array_equal(r.view("a").stress(P, True)[0][:, 0, 0], P[:, 0])
        r.view("a").stress(P, False)            # a different on-surface flag is a different query
        self.assertEqual(fake.calls, 2)
        with self.assertRaisesRegex(ModelError, "no load case 'c'"):
            r.view("c")


SENB = str(pathlib.Path(__file__).resolve().parents[2] / "geo" / "step" / "senb_rho1.step")
ROOT_PT = (22.0, 5.0, 0.0)


class Submodels(unittest.TestCase):
    def test_ill_posed_boxes_are_refused(self):
        from .submodel import check_box
        from failure.validate_cicero import case_for
        c = case_for(SENB)
        loaded = [s["face_id"] for s in c["supports"]] + [c["cases"][0]["loads"][0]["face_id"]]
        check_box(SENB, [20, 3, -3], [24, 7, 3], ROOT_PT, loaded, 1e-6)              # the gate's box
        for why, lo, hi in (("lies in face", [20, 3, -2], [24, 7, 3]),              # z = -2 is a face of the bar
                            ("not inside", [20, 6, -3], [24, 9, 3]),                 # the root is below the box
                            ("carries a support or load", [20, 3, -3], [24, 12, 3]),  # reaches the top pad
                            ("lo < hi", [24, 3, -3], [20, 7, 3])):
            with self.subTest(why), self.assertRaisesRegex(ModelError, why):
                check_box(SENB, lo, hi, ROOT_PT, loaded, 1e-6)

    def test_cut_faces_are_the_box_faces_inside_the_part(self):
        from .submodel import cut_faces, cut_step
        with tempfile.TemporaryDirectory() as tmp:
            out = str(pathlib.Path(tmp) / "cut.step")
            cut_step(SENB, [20, 3, -3], [24, 7, 3], out)
            # x = 20, x = 24, y = 7 and y = 3 on both sides of the notch; z = -3, 3 lie off the bar
            self.assertEqual(len(cut_faces(out, [20, 3, -3], [24, 7, 3], 1e-6)), 5)
            with self.assertRaisesRegex(ModelError, "misses the part"):
                cut_step(SENB, [100, 100, 100], [101, 101, 101], str(pathlib.Path(tmp) / "none.step"))

    def test_submodel_nodes_and_the_regions_a_box_can_be_checked_against(self):
        m = senb_like()
        with self.assertRaisesRegex(ModelError, "lo < hi"):
            m.apply({"op": "add", "parent": "root", "kind": "submodel", "props": {"lo": [1, 0, 0], "hi": [0, 1, 1]}})
        m.apply({"op": "add", "parent": "root", "kind": "submodel", "props": {"label": "notch", "lo": [0, 0, 0],
                                                                              "hi": [1, 1, 1]}})
        self.assertIn("submodel", backends.features(m))
        with self.assertRaisesRegex(ModelError, "acts on a selection"):
            run._region_faces(m)                      # ALL_FORMS has select- and patch-based regions
        clean = Model()
        clean.apply({"op": "add", "parent": "root", "kind": "support", "region": {"face_ids": [3, 4]},
                     "props": {"fix": ["ux"]}})
        self.assertEqual(run._region_faces(clean), [3, 4])


if __name__ == "__main__":
    unittest.main()
