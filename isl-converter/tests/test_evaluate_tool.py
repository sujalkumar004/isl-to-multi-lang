"""
Test of tools/evaluate_model.py with a FAKE in-memory dataset and the colour-classifier stand-in model.
Colour-coded fake pictures make exactly three characters learnable (red = अ, green = आ, blue = इ), so the
tool must report the other 40 as never predicted, and it must catch a class-order mismatch with the dataset.
Run:  python -m unittest tests.test_evaluate_tool -v     (needs tensorflow and scikit-learn)
"""
import glob
import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import sklearn  # noqa: F401
    import tensorflow  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

from PIL import Image  # noqa: E402


def make_fake_dataset(names):
    class FakeDS:
        def __init__(self):
            rng = np.random.default_rng(0)
            self.features = {"label": types.SimpleNamespace(names=names)}
            self.rows = []
            for c in range(43):
                for _ in range(30):
                    if c < 3:
                        color = [(230, 20, 20), (20, 230, 20), (20, 20, 230)][c]
                        arr = np.full((128, 128, 3), color, np.uint8)
                    else:
                        arr = rng.integers(0, 40, (128, 128, 3), dtype=np.uint8)
                    self.rows.append({"image": Image.fromarray(arr), "label": c})

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return [r["label"] for r in self.rows] if i == "label" else self.rows[i]

    return FakeDS()


@unittest.skipUnless(HAVE, "needs tensorflow and scikit-learn")
class TestEvaluateTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from make_test_model import build
        cls.tmp = tempfile.mkdtemp()
        cls.model = os.path.join(cls.tmp, "m.keras")
        build(cls.model)
        with open(os.path.join(ROOT, "model", "labels.json"), encoding="utf-8") as f:
            cls.names = json.load(f)

    def run_tool(self, names, extra=()):
        import importlib
        import evaluate_model
        importlib.reload(evaluate_model)
        fake = types.ModuleType("datasets")
        fake.load_dataset = lambda name: {"train": make_fake_dataset(names)}
        sys.modules["datasets"] = fake
        out = os.path.join(self.tmp, "report.json")
        ref = os.path.join(self.tmp, "reference")
        code = evaluate_model.main(["--model", self.model, "--out", out, "--reference-dir", ref,
                                    "--references", "2", *extra])
        with open(out, encoding="utf-8") as f:
            return code, json.load(f), ref

    def test_report_finds_learnable_characters_and_the_dead_ones(self):
        code, rep, ref = self.run_tool(self.names, ["--robustness", "--robustness-n", "120"])
        self.assertEqual(code, 0)
        by = {c["label"]: c for c in rep["classes"]}
        self.assertEqual(len(rep["classes"]), 43)
        self.assertEqual(by["आ"]["recall"], 1.0)
        self.assertEqual(by["इ"]["recall"], 1.0)
        self.assertEqual(by["क"]["status"], "not_detected")
        self.assertIn("क", rep["overall"]["classes_never_predicted"])
        self.assertNotIn("आ", rep["overall"]["classes_never_predicted"])
        self.assertTrue(rep["dataset"]["labels_match"])
        self.assertIn("mirror", rep["robustness"])
        self.assertIn("baseline", rep["robustness"])
        self.assertEqual(len(glob.glob(os.path.join(ref, "*.jpg"))), 86)          # 2 pictures x 43 characters
        self.assertIn("held-out test split", rep["split"])

    def test_the_split_is_the_notebooks_split(self):
        # 15 % of 30 pictures per class -> 4 or 5 per class, stratified, like train_test_split in the Colab notebook
        _, rep, _ = self.run_tool(self.names)
        supports = {c["support"] for c in rep["classes"]}
        self.assertTrue(supports <= {4, 5}, supports)
        self.assertEqual(rep["overall"]["samples"], sum(c["support"] for c in rep["classes"]))

    def test_class_order_mismatch_with_the_dataset_is_reported(self):
        _, rep, _ = self.run_tool(list(reversed(self.names)))
        self.assertFalse(rep["dataset"]["labels_match"])

    def test_inspect_only_needs_no_dataset(self):
        import importlib
        import evaluate_model
        importlib.reload(evaluate_model)
        sys.modules.pop("datasets", None)
        self.assertEqual(evaluate_model.main(["--model", self.model, "--inspect-only"]), 0)


    # ------------------------------------------------------------------ your own webcam frames
    def make_collected(self, per_class=12):
        """Frames as the website records them: <set>/<class index>/<stamp>.jpg + .json with the hand box."""
        root = os.path.join(tempfile.mkdtemp(), "test")
        for cls, color in enumerate([(230, 20, 20), (20, 230, 20), (20, 20, 230)]):
            folder = os.path.join(root, str(cls))
            os.makedirs(folder)
            for k in range(per_class):
                Image.new("RGB", (720, 540), color).save(os.path.join(folder, f"f{k:03d}.jpg"), quality=90)
                with open(os.path.join(folder, f"f{k:03d}.json"), "w", encoding="utf-8") as f:
                    json.dump({"index": cls, "label": self.names[cls], "bbox": [0.3, 0.3, 0.5, 0.7], "frame_size": [720, 540]}, f)
        return root

    def run_webcam(self, folder, out=None):
        import importlib
        import evaluate_model
        importlib.reload(evaluate_model)
        out = out or os.path.join(self.tmp, "webcam_report.json")
        code = evaluate_model.main(["--model", self.model, "--out", out, "--no-dataset", "--collected", folder])
        with open(out, encoding="utf-8") as f:
            return code, json.load(f)

    def test_webcam_frames_are_scored_without_any_dataset(self):
        sys.modules.pop("datasets", None)                       # proves the dataset is not needed
        code, rep = self.run_webcam(self.make_collected())
        self.assertEqual(code, 0)
        w = rep["webcam"]
        self.assertEqual((w["frames"], w["recorded_classes"], w["set"]), (36, 3, "test"))
        self.assertEqual(set(w["modes"]), {"full", "hand", "multi", "multi_mirrored"})
        for mode in w["modes"].values():
            self.assertEqual(mode["overall"]["accuracy"], 1.0)
        multi = {c["label"]: c for c in w["modes"]["multi"]["classes"]}
        self.assertEqual(multi["अ"]["status"], "good")
        self.assertEqual(multi["क"]["status"], "untested")     # never recorded
        self.assertEqual(w["gated"]["pass_rate"], 1.0)          # confident, no gap problem, all three crops agree
        self.assertEqual(w["gated"]["wrong_among_passed"], 0.0)
        self.assertEqual(multi["आ"]["pass_rate"], 1.0)
        self.assertIn(w["best_mode"], w["modes"])

    def test_webcam_result_is_merged_into_an_existing_dataset_report(self):
        _, ds_rep, _ = self.run_tool(self.names)                # writes report.json (dataset part)
        out = os.path.join(self.tmp, "report.json")
        _, rep = self.run_webcam(self.make_collected(), out=out)
        self.assertEqual(len(rep["classes"]), 43)              # dataset results kept
        self.assertIn("webcam", rep)                           # webcam results added

    def test_missing_frames_give_a_clear_message(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_webcam(tempfile.mkdtemp())
        self.assertIn("No recorded frames", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
