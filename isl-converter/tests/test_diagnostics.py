"""
Tests for backend/diagnostics.py and backend/preprocessing.py (no TensorFlow needed).
Run:  python -m unittest tests.test_diagnostics -v
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import diagnostics as dg  # noqa: E402
import preprocessing as pp  # noqa: E402


def probs_for(pred, conf, n=5):
    p = np.full((len(pred), n), 0.0, dtype=np.float32)
    for i, (c, cf) in enumerate(zip(pred, conf)):
        p[i] += (1 - cf) / (n - 1)
        p[i, c] = cf
    return p


class TestLabelChecks(unittest.TestCase):
    def test_clean_list_has_no_issues(self):
        self.assertEqual(dg.check_labels(["अ", "आ", "क्ष"], 3), [])

    def test_real_project_labels_are_clean_and_43(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "model", "labels.json"), encoding="utf-8") as f:
            labels = json.load(f)
        self.assertEqual(len(labels), 43)
        self.assertEqual(dg.check_labels(labels, 43), [])

    def test_duplicates_count_mismatch_and_empty_are_errors(self):
        msgs = " | ".join(i["message"] for i in dg.check_labels(["अ", "अ", " "], 5))
        self.assertIn("appears twice", msgs)
        self.assertIn("5 outputs but the label list has 3", msgs)
        self.assertIn("is empty", msgs)

    def test_encoding_problems(self):
        import unicodedata
        nfd = unicodedata.normalize("NFD", "ऩ")
        issues = dg.check_labels(["अ", nfd, "आ\u200b"], 3)
        text = " | ".join(i["message"] for i in issues)
        self.assertIn("NFC", text)
        self.assertIn("control character", text)          # zero-width space is invisible

    def test_dataset_order_mismatch_is_an_error(self):
        issues = dg.check_labels(["अ", "आ", "इ"], 3, dataset_labels=["अ", "इ", "आ"])
        self.assertTrue(any("does not match the dataset" in i["message"] for i in issues))
        self.assertEqual(dg.check_labels(["अ", "आ"], 2, dataset_labels=["अ", "आ"]), [])


class TestMetricsAndStatus(unittest.TestCase):
    def setUp(self):
        # class 0: perfect. class 1: never predicted (always seen as 0). class 2: 60 % right. class 3: never in the test data.
        y = [0] * 20 + [1] * 20 + [2] * 20
        pred = [0] * 20 + [0] * 20 + [2] * 12 + [4] * 8
        conf = [0.95] * 20 + [0.80] * 20 + [0.9] * 12 + [0.5] * 8
        self.probs = probs_for(pred, conf)
        self.y = np.array(y)
        self.labels = ["a", "b", "c", "d", "e"]

    def test_status_per_class(self):
        classes, overall, conf = dg.per_class_metrics(self.probs, self.y, self.labels)
        by = {c["label"]: c for c in classes}
        self.assertEqual(by["b"]["status"], "not_detected")     # recall 0, never predicted
        self.assertEqual(by["c"]["status"], "poor")             # recall 0.6
        self.assertEqual(by["d"]["status"], "untested")         # no test samples
        self.assertEqual(by["a"]["recall"], 1.0)
        self.assertLess(by["a"]["precision"], 1.0)              # b's images were called a
        self.assertEqual(by["b"]["confused_with"][0][0], "a")
        self.assertEqual(overall["classes_never_predicted"], ["b"])
        self.assertAlmostEqual(overall["accuracy"], 32 / 60, places=3)
        self.assertEqual(conf[0], ["b", "a", 20])

    def test_status_thresholds(self):
        s = dg.class_status
        self.assertEqual(s({"support": 10, "recall": 0.95, "precision": 0.9, "predicted_count": 10}), "good")
        self.assertEqual(s({"support": 10, "recall": 0.80, "precision": 0.9, "predicted_count": 10}), "fair")
        self.assertEqual(s({"support": 10, "recall": 0.50, "precision": 0.9, "predicted_count": 10}), "poor")
        self.assertEqual(s({"support": 10, "recall": 0.01, "precision": 0.9, "predicted_count": 10}), "not_detected")
        self.assertEqual(s({"support": 0}), "untested")
        self.assertEqual(s(None), "untested")


class TestCalibration(unittest.TestCase):
    def test_threshold_is_the_lowest_confidence_that_is_reliable(self):
        # predictions of class 0: confident ones (>= .6) are all right, the weak ones (.4) are wrong
        pred = [0] * 40 + [0] * 20
        y = [0] * 40 + [1] * 20
        conf = [0.65] * 40 + [0.40] * 20
        t = dg.calibrate_thresholds(probs_for(pred, conf), y, 5)
        self.assertEqual(t[0], 0.45)          # 0.30/0.35/0.40 include the wrong ones; 0.45 is the first clean level
        self.assertIsNone(t[1])               # class 1 never predicted

    def test_no_threshold_when_the_class_is_never_reliable_or_data_is_scarce(self):
        pred = [0] * 30
        y = [0] * 15 + [1] * 15               # only 50 % precise at every confidence
        self.assertIsNone(dg.calibrate_thresholds(probs_for(pred, [0.9] * 30), y, 5)[0])
        self.assertIsNone(dg.calibrate_thresholds(probs_for([0] * 5, [0.9] * 5), [0] * 5, 5)[0])   # < 10 samples

    def test_class_thresholds_from_report(self):
        rep = {"classes": [{"label": "a", "suggested_threshold": 0.5}, {"label": "b", "suggested_threshold": None},
                           {"label": "zz", "suggested_threshold": 0.4}]}
        self.assertEqual(dg.class_thresholds(rep, ["a", "b"]), {"a": 0.5})
        self.assertEqual(dg.class_thresholds(None, ["a"]), {})


class TestReportAndSimilarity(unittest.TestCase):
    def test_load_report_handles_missing_and_corrupt_files(self):
        d = tempfile.mkdtemp()
        self.assertIsNone(dg.load_report(os.path.join(d, "none.json")))
        bad = os.path.join(d, "bad.json")
        with open(bad, "w") as f:
            f.write("{not json")
        self.assertIsNone(dg.load_report(bad))
        good = os.path.join(d, "ok.json")
        with open(good, "w") as f:
            json.dump({"classes": [{"status": "good"}, {"status": "poor"}]}, f)
        self.assertEqual(dg.summarize_report(dg.load_report(good))["poor"], 1)

    def test_identical_class_pictures_are_reported_as_most_similar(self):
        rng = np.random.default_rng(0)
        means = rng.random((4, 8, 8, 3))
        means[3] = means[1] + 0.001 * rng.random((8, 8, 3))          # class 3 looks like class 1
        top = dg.similar_class_pairs(means, ["a", "b", "c", "d"], top=1)[0]
        self.assertEqual(set(top[:2]), {"b", "d"})
        self.assertGreater(top[2], 0.99)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (720, 540), (120, 120, 120))
        self.bbox = (0.40, 0.30, 0.60, 0.60)                        # a hand in the middle

    def test_full_is_the_untouched_frame(self):
        views, info = pp.build_views(self.img, self.bbox, "full")
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].size, (720, 540))
        self.assertEqual(info["mode_used"], "full")

    def test_hand_is_a_smaller_square_and_multi_gives_three_growing_crops(self):
        (hand,), info = pp.build_views(self.img, self.bbox, "hand")
        self.assertEqual(hand.size[0], hand.size[1])
        self.assertLess(hand.size[0], 540)
        views, info = pp.build_views(self.img, self.bbox, "multi")
        sizes = [v.size[0] for v in views]
        self.assertEqual(len(views), 3)
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])
        self.assertEqual(info["n_views"], 3)

    def test_without_a_hand_box_crops_fall_back_to_the_full_frame(self):
        for mode in ("hand", "multi"):
            views, info = pp.build_views(self.img, None, mode)
            self.assertEqual((len(views), info["mode_used"]), (1, "full"))

    def test_mirror_flips_the_pixels_and_the_box(self):
        img = Image.new("RGB", (100, 50), (0, 0, 0))
        img.putpixel((0, 0), (255, 0, 0))
        views, info = pp.build_views(img, None, "full", mirror=True)
        self.assertEqual(views[0].getpixel((99, 0)), (255, 0, 0))
        self.assertTrue(info["mirrored"])
        self.assertEqual(pp.mirror_bbox((0.1, 0.2, 0.3, 0.6)), (0.7, 0.2, 0.9, 0.6))

    def test_compare_views_lists_every_variant(self):
        self.assertEqual(set(pp.compare_views(self.img, self.bbox)), {"full", "mirror_full", "hand", "multi", "mirror_hand"})
        self.assertEqual(set(pp.compare_views(self.img, None)), {"full", "mirror_full"})

    def test_frame_quality_explains_dark_images_and_tiny_hands(self):
        dark = pp.frame_quality(Image.new("RGB", (720, 540), (10, 10, 10)), (0.5, 0.5, 0.52, 0.52))
        self.assertLess(dark["brightness"], 0.1)
        text = " ".join(dark["notes"])
        self.assertIn("dark", text)
        self.assertIn("very small", text)
        fine = pp.frame_quality(self.img, self.bbox)
        self.assertEqual(fine["notes"], [])
        self.assertGreater(fine["hand_fraction"], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
