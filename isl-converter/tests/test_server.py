"""
Integration tests of the Flask API with real JPEG frames and a STAND-IN colour-classifier model
(tests/make_test_model.py). They test the plumbing: JPEG decode -> preprocessing -> Keras -> recognizer
-> HTTP. They say nothing about how accurate YOUR model is.
Run:  python -m unittest tests.test_server -v     (needs tensorflow)
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import tensorflow  # noqa: F401
    HAVE_TF = True
except ImportError:
    HAVE_TF = False

from PIL import Image  # noqa: E402

RED, GREEN, BLUE, BLACK = (230, 20, 20), (20, 230, 20), (20, 20, 230), (0, 0, 0)
DIM_RED = (25, 0, 0)   # stand-in model gives this only moderate confidence (~0.5 to 0.7)


def jpeg(color, size=(720, 540)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=85)
    buf.seek(0)
    return buf


@unittest.skipUnless(HAVE_TF, "tensorflow not installed")
class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from make_test_model import build
        cls.tmp = tempfile.mkdtemp()
        cls.model_path = os.path.join(cls.tmp, "stand_in.keras")
        build(cls.model_path)
        os.environ["ISL_MODEL_PATH"] = cls.model_path
        cls.report_path = os.path.join(cls.tmp, "model_report.json")           # does not exist yet
        cls.reference_dir = os.path.join(cls.tmp, "reference")
        os.makedirs(cls.reference_dir)
        cls.data_dir = os.path.join(cls.tmp, "data")
        os.environ["ISL_DATA_DIR"] = cls.data_dir
        os.environ["ISL_REPORT_PATH"] = cls.report_path
        os.environ["ISL_REFERENCE_DIR"] = cls.reference_dir
        import importlib
        import app as app_module
        cls.app = importlib.reload(app_module)
        assert cls.app.MODEL.wait(120), "model did not finish loading"
        cls.client = cls.app.app.test_client()

    def setUp(self):
        self.sid = "t" + str(time.time_ns())
        self.post("/api/config", json={"config": {"hand_absent_seconds": 0.1}})

    # helpers ---------------------------------------------------------
    def post(self, url, json=None):
        return self.client.post(url, json=dict(json or {}, session=self.sid))

    def frame(self, color, hand="1", send_image=True):
        data = {"session": self.sid, "hand": hand}
        if send_image:
            data["frame"] = (jpeg(color), "frame.jpg")
        r = self.client.post("/api/frame", data=data, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        return r.get_json()

    def hold(self, color, n):
        out = None
        for _ in range(n):
            out = self.frame(color)
        return out

    # model / info ----------------------------------------------------
    def test_info_reports_model_loaded(self):
        info = self.client.get("/api/info").get_json()
        self.assertTrue(info["model"]["ready"], info["model"]["error"])
        self.assertEqual(info["model"]["state"], "ready")
        self.assertEqual(info["model"]["classes"], 43)
        self.assertEqual(len(info["labels"]), 43)
        self.assertEqual(set(info["languages"]), {"en", "hi", "mr"})

    def test_frame_while_model_is_loading_is_a_503_with_the_loading_message(self):
        original = self.app.MODEL.state
        self.app.MODEL.state = "loading"
        try:
            r = self.client.post("/api/frame", data={"session": self.sid, "hand": "0"},
                                 content_type="multipart/form-data")
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.get_json()["error"], "Loading recognition model...")
            self.assertEqual(r.get_json()["model_state"], "loading")
        finally:
            self.app.MODEL.state = original

    # continuous recognition through the real API -----------------------
    def test_holding_one_sign_adds_one_character(self):
        out = self.hold(RED, 40)
        self.assertEqual(out["text"], "अ")
        self.assertEqual(out["state"], "LOCKED")
        self.assertEqual(out["current_sign"], "अ")
        self.assertGreater(out["confidence"], 0.9)
        self.assertEqual(out["agreement"], out["window_size"])
        accepted = [s for s in [out] if s["event"] == "accepted"]
        self.assertEqual(accepted, [])          # only the frame that accepts carries the event

    def test_accepted_event_is_reported_exactly_once(self):
        events = [self.frame(RED)["event"] for _ in range(30)]
        self.assertEqual(events.count("accepted"), 1)

    def test_sequence_release_repeat_space_backspace_clear(self):
        self.hold(RED, 10)
        self.assertEqual(self.hold(GREEN, 10)["text"], "अआ")
        self.hold(BLUE, 10)
        self.assertEqual(self.hold(BLUE, 30)["text"], "अआइ")       # held: no duplicates

        time.sleep(0.15)                                           # hand lowered: no image needed
        for _ in range(2):
            self.frame(None, hand="0", send_image=False)
            time.sleep(0.06)
        self.assertEqual(self.frame(None, hand="0", send_image=False)["state"], "WAITING")
        self.assertEqual(self.hold(BLUE, 10)["text"], "अआइइ")

        self.assertEqual(self.post("/api/space").get_json()["text"], "अआइइ ")
        self.assertEqual(self.post("/api/space").get_json()["text"], "अआइइ ")   # no double space
        self.assertEqual(self.post("/api/backspace").get_json()["text"], "अआइइ")
        self.assertEqual(self.post("/api/clear").get_json()["text"], "")

    def test_pause_resume_keeps_text_and_rereads_the_sign(self):
        self.hold(RED, 10)
        snap = self.post("/api/resume").get_json()
        self.assertEqual(snap["text"], "अ")                       # text is kept
        self.assertIsNone(snap["locked_sign"])
        self.assertEqual(snap["window_fill"], 0)
        self.assertEqual(self.hold(RED, 10)["text"], "अअ")         # sign read afresh after resuming

    def test_black_image_is_never_accepted(self):
        out = self.hold(BLACK, 30)
        self.assertEqual(out["text"], "")
        self.assertLess(out["confidence"], 0.2)

    def test_threshold_change_takes_effect(self):
        out = self.hold(DIM_RED, 20)
        self.assertTrue(0.45 < out["confidence"] < 0.70, out["confidence"])   # fixture sanity
        self.assertEqual(out["text"], "")                                      # below the 0.70 default
        self.assertEqual(out["state"], "STABLE")                               # steady but not confident
        self.post("/api/config", json={"config": {"confidence_threshold": 0.4}})
        self.assertEqual(self.hold(DIM_RED, 3)["text"], "अ")

    def test_hand_absent_frames_need_no_image(self):
        self.assertEqual(self.frame(None, hand="0", send_image=False)["state"], "WAITING")

    def test_missing_or_invalid_frame_is_a_400(self):
        r = self.client.post("/api/frame", data={"session": self.sid, "hand": "1"}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/frame", data={"session": self.sid, "hand": "1", "frame": (io.BytesIO(b"nope"), "f.jpg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_config_is_per_session_and_clamped(self):
        r = self.post("/api/config", json={"config": {"confidence_threshold": 5, "window_size": 4}}).get_json()
        self.assertEqual(r["config"]["confidence_threshold"], 0.99)
        self.assertEqual(r["config"]["window_size"], 4)
        other = self.client.post("/api/config", json={"session": "other", "config": {}}).get_json()
        self.assertEqual(other["config"]["window_size"], 6)

    def test_every_view_mode_works_with_and_without_a_hand_box(self):
        for mode in ("full", "hand", "multi"):
            self.post("/api/config", json={"config": {"view_mode": mode, "mirror_input": mode == "hand"}})
            for bbox in ("0.2,0.2,0.6,0.7", "garbage", ""):
                data = {"session": self.sid, "hand": "1", "bbox": bbox, "frame": (jpeg(RED), "f.jpg")}
                r = self.client.post("/api/frame", data=data, content_type="multipart/form-data")
                self.assertEqual(r.status_code, 200, (mode, bbox))
                self.assertEqual(r.get_json()["current_sign"], "अ")

    def test_unknown_view_mode_is_ignored(self):
        cfg = self.post("/api/config", json={"config": {"view_mode": "hand"}}).get_json()["config"]
        self.assertEqual(cfg["view_mode"], "hand")
        cfg = self.post("/api/config", json={"config": {"view_mode": "wizard"}}).get_json()["config"]
        self.assertEqual(cfg["view_mode"], "hand")

    def test_debug_frames_explain_the_prediction(self):
        data = {"session": self.sid, "hand": "1", "debug": "1", "bbox": "0.3,0.3,0.6,0.7", "frame": (jpeg(GREEN), "f.jpg")}
        r = self.client.post("/api/frame", data=data, content_type="multipart/form-data").get_json()
        d = r["debug"]
        self.assertEqual(len(d["probs"] if "probs" in d else r["probs"]), 43)
        self.assertEqual(r["instant_top"][0][0], "आ")                     # green -> class 1
        self.assertEqual(len(r["top"]), 5)
        self.assertEqual(set(d["compare"]), {"full", "mirror_full", "hand", "multi", "mirror_hand"})
        self.assertEqual(d["compare"]["hand"]["top"][0][0], "आ")
        self.assertEqual(d["selected"], "multi")                         # the default view mode
        self.assertEqual(d["views"]["n_views"], 3)
        self.assertEqual(len(d["view_tops"]), 3)
        self.assertEqual(d["view_agreement"], 1.0)
        self.assertEqual(d["quality"]["notes"], [])
        self.assertEqual(d["model_input_size"], 96)
        raw = __import__("base64").b64decode(d["model_input"].split(",", 1)[1])
        self.assertEqual(Image.open(io.BytesIO(raw)).size, (96, 96))     # exactly what the model receives
        self.assertIn("stable_frames", r)
        self.assertIn("counters", r)

    def test_debug_output_is_only_sent_when_asked_for(self):
        r = self.frame(RED)
        self.assertNotIn("debug", r)
        self.assertNotIn("probs", r)

    # diagnostics --------------------------------------------------------
    def test_diagnostics_without_a_report_says_untested_and_labels_are_clean(self):
        d = self.client.get("/api/diagnostics").get_json()
        self.assertIsNone(d["report"])
        self.assertEqual(d["label_issues"], [])
        self.assertEqual(len(d["classes"]), 43)
        self.assertTrue(all(c["status"] == "untested" for c in d["classes"]))
        self.assertEqual(d["model"]["params"] > 0, True)
        self.assertEqual(d["model"]["input_size"], 96)
        self.assertEqual(d["calibrated_classes"], 0)

    def test_diagnostics_with_a_report_shows_statuses_and_calibration_reaches_new_sessions(self):
        import json as _json
        report = {"generated": "2026-01-01", "split": "test", "overall": {"accuracy": 0.5},
                  "dataset": {"labels": self.app.LABELS, "images": 10},
                  "classes": [{"index": 0, "label": "अ", "support": 20, "predicted_count": 20, "recall": 0.95,
                               "precision": 0.95, "status": "good", "suggested_threshold": 0.45},
                              {"index": 1, "label": "आ", "support": 20, "predicted_count": 0, "recall": 0.0,
                               "precision": 0.0, "status": "not_detected", "suggested_threshold": None}]}
        with open(self.report_path, "w", encoding="utf-8") as f:
            _json.dump(report, f, ensure_ascii=False)
        try:
            d = self.client.get("/api/diagnostics").get_json()
            self.assertEqual(d["report"]["overall"]["accuracy"], 0.5)
            self.assertEqual(d["classes"][0]["status"], "good")
            self.assertEqual(d["classes"][1]["status"], "not_detected")
            self.assertEqual(d["classes"][2]["status"], "untested")
            self.assertEqual(d["label_issues"], [])
            self.assertEqual(d["calibrated_classes"], 1)
            # a NEW session now accepts the dim red sign (conf ~0.55) because 'अ' is calibrated to 0.45
            fresh = "cal" + str(time.time_ns())
            self.client.post("/api/config", json={"session": fresh, "config": {"hand_absent_seconds": 0.1}})
            out = None
            for _ in range(10):
                out = self.client.post("/api/frame", data={"session": fresh, "hand": "1", "frame": (jpeg(DIM_RED), "f.jpg")},
                                       content_type="multipart/form-data").get_json()
            self.assertEqual(out["text"], "अ")
            self.assertLess(out["threshold_used"], out["threshold"])
            # ...and a report that disagrees with labels.json is flagged
            report["dataset"]["labels"] = list(reversed(self.app.LABELS))
            with open(self.report_path, "w", encoding="utf-8") as f:
                _json.dump(report, f, ensure_ascii=False)
            d = self.client.get("/api/diagnostics").get_json()
            self.assertTrue(any("does not match the dataset" in i["message"] for i in d["label_issues"]))
        finally:
            os.remove(self.report_path)

    # crop agreement + recording your own frames --------------------------
    def test_snapshot_reports_crop_agreement(self):
        r = self.frame(RED)
        self.assertEqual(r["view_agreement"], 1.0)
        self.assertEqual(r["min_view_agreement"], 0.6)

    def test_frames_can_be_recorded_for_a_character(self):
        def post(**extra):
            data = {"session": self.sid, "hand": "1", "bbox": "0.3,0.3,0.5,0.7", "frame": (jpeg(GREEN), "f.jpg"), **extra}
            return self.client.post("/api/frame", data=data, content_type="multipart/form-data").get_json()
        self.assertEqual(post(collect="3", collect_set="train")["collected"], 1)
        self.assertEqual(post(collect="3", collect_set="train")["collected"], 2)
        folder = os.path.join(self.data_dir, "collected", "train", "3")
        jpgs = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
        self.assertEqual(len(jpgs), 2)
        meta = json.load(open(os.path.join(folder, jpgs[0][:-4] + ".json"), encoding="utf-8"))
        self.assertEqual((meta["index"], meta["label"], meta["bbox"], meta["frame_size"]), (3, "ई", [0.3, 0.3, 0.5, 0.7], [720, 540]))
        self.assertEqual(Image.open(os.path.join(folder, jpgs[0])).size, (720, 540))        # the exact frame, not a copy of a copy
        counts = self.client.get("/api/collected").get_json()
        self.assertEqual(counts["sets"]["train"]["3"], 2)

    def test_nothing_is_recorded_without_a_hand_box_or_with_bad_requests(self):
        def post(**extra):
            data = {"session": self.sid, "hand": "1", "frame": (jpeg(GREEN), "f.jpg"), **extra}
            return self.client.post("/api/frame", data=data, content_type="multipart/form-data").get_json()
        before = self.client.get("/api/collected").get_json()["sets"]
        self.assertNotIn("collected", post(collect="5", collect_set="train"))                         # no bbox
        self.assertNotIn("collected", post(bbox="0.3,0.3,0.5,0.7", collect="99", collect_set="train"))   # no such class
        self.assertNotIn("collected", post(bbox="0.3,0.3,0.5,0.7", collect="abc", collect_set="train"))
        for bad in ("../evil", "a/b", "", "x" * 40, "sp ace"):
            self.assertNotIn("collected", post(bbox="0.3,0.3,0.5,0.7", collect="5", collect_set=bad) if bad else {"x": 1}, bad)
        data = {"session": self.sid, "hand": "0"}                                                       # no hand: frame not even read
        out = self.client.post("/api/frame", data=data, content_type="multipart/form-data").get_json()
        self.assertNotIn("collected", out)
        self.assertEqual(self.client.get("/api/collected").get_json()["sets"], before)
        self.assertFalse(os.path.exists(os.path.join(self.data_dir, "collected", "..", "evil")))

    def test_webcam_evaluation_appears_in_diagnostics_and_its_thresholds_are_used(self):
        report = {"generated": "2026-01-01", "overall": None, "classes": [], "dataset": None,
                  "webcam": {"set": "test", "frames": 10, "recorded_classes": 1, "best_mode": "multi",
                             "gated": {"rules": {}, "pass_rate": 0.5, "wrong_among_passed": 0.1},
                             "modes": {"multi": {"overall": {"accuracy": 0.9}, "classes": [
                                 {"index": 0, "label": "अ", "support": 10, "recall": 0.9, "precision": 0.9, "status": "good",
                                  "confused_with": [["आ", 1]], "pass_rate": 0.5, "correct_when_passed": 1.0}]}},
                             "thresholds": {"अ": 0.5}}}
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        try:
            d = self.client.get("/api/diagnostics").get_json()
            self.assertEqual(d["report"]["webcam"]["best_mode"], "multi")
            self.assertEqual(d["report"]["webcam"]["mode_accuracy"], {"multi": 0.9})
            self.assertEqual(d["classes"][0]["webcam"]["recall"], 0.9)
            self.assertEqual(d["classes"][0]["webcam"]["confused_with"], [["आ", 1]])
            self.assertNotIn("webcam", d["classes"][1])
            self.assertEqual(d["calibrated_classes"], 1)
        finally:
            os.remove(self.report_path)

    def test_reference_pictures_are_served(self):
        Image.new("RGB", (128, 128), (9, 9, 9)).save(os.path.join(self.reference_dir, "0_0.jpg"))
        r = self.client.get("/reference/0_0.jpg")
        self.assertEqual((r.status_code, r.mimetype), (200, "image/jpeg"))
        r.close()
        self.assertEqual(self.client.get("/reference/nope.jpg").status_code, 404)
        self.assertEqual(self.client.get("/api/diagnostics").get_json()["references_per_class"], 1)
        os.remove(os.path.join(self.reference_dir, "0_0.jpg"))

    # translation endpoint (Google call replaced by a stub) ------------------
    def test_translate_endpoint_translates_the_whole_text_into_all_three_languages(self):
        import translation
        seen = []
        original = translation._backend
        translation._cache.clear()
        translation._backend = lambda text, target: seen.append((text, target)) or f"<{target}>{text}"
        try:
            r = self.client.post("/api/translate", json={"text": "आई उ", "targets": ["en", "hi", "mr"]}).get_json()
        finally:
            translation._backend = original
        self.assertEqual(r["translations"]["en"]["text"], "<en>आई उ")
        self.assertEqual(r["translations"]["hi"]["text"], "<hi>आई उ")
        self.assertEqual(r["translations"]["mr"]["text"], "आई उ")
        self.assertEqual(sorted(seen), [("आई उ", "en"), ("आई उ", "hi")])   # one call per language, full text

    def test_translate_errors_are_graceful(self):
        import translation
        original = translation._backend
        translation._cache.clear()

        def offline(text, target):
            raise ConnectionError("offline")

        translation._backend = offline
        try:
            r = self.client.post("/api/translate", json={"text": "आई", "targets": ["en", "mr"]})
        finally:
            translation._backend = original
        self.assertEqual(r.status_code, 200)
        body = r.get_json()["translations"]
        self.assertIn("Translation is unavailable", body["en"]["error"])
        self.assertEqual(body["mr"]["text"], "आई")
        bad = self.client.post("/api/translate", json={"text": "आई", "targets": ["xx"]})
        self.assertEqual(bad.status_code, 400)

    # website -----------------------------------------------------------
    def test_website_files_are_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Indian Sign Language Converter".encode(), r.data)
        for path in ("/static/style.css", "/static/script.js"):
            rr = self.client.get(path)
            self.assertEqual(rr.status_code, 200, path)
            rr.close()
        r.close()


def run_in_subprocess(code, env_extra):
    env = dict(os.environ, **env_extra)
    return subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "backend"),
                          capture_output=True, text=True, timeout=180, env=env)


@unittest.skipUnless(HAVE_TF, "tensorflow not installed")
class TestModelProblems(unittest.TestCase):
    """The server must say what is wrong instead of faking recognition."""

    CODE = (
        "import app; app.MODEL.wait(120); c=app.app.test_client();"
        "i=c.get('/api/info').get_json()['model'];"
        "r=c.post('/api/frame', data={'session':'x','hand':'1'}, content_type='multipart/form-data');"
        "print(i['state'], r.status_code, i['error'])"
    )

    def test_missing_model_file(self):
        out = run_in_subprocess(self.CODE, {"ISL_MODEL_PATH": "/nonexistent/model.keras"})
        self.assertIn("error 503 Model file not found", out.stdout, out.stderr[-600:])

    def test_model_with_wrong_number_of_classes(self):
        from make_test_model import build
        path = os.path.join(tempfile.mkdtemp(), "ten.keras")
        build(path, num_classes=10)
        out = run_in_subprocess(self.CODE, {"ISL_MODEL_PATH": path})
        self.assertIn("error 503 The model has 10 outputs", out.stdout, out.stderr[-600:])

    def test_model_with_non_square_or_grayscale_input_is_rejected(self):
        import tensorflow as tf
        d = tempfile.mkdtemp()
        rect = os.path.join(d, "rect.keras")
        tf.keras.Sequential([tf.keras.layers.Input((64, 96, 3)), tf.keras.layers.GlobalAveragePooling2D(),
                             tf.keras.layers.Dense(43, activation="softmax")]).save(rect)
        out = run_in_subprocess(self.CODE, {"ISL_MODEL_PATH": rect})
        self.assertIn("error 503 The model expects input (64, 96, 3)", out.stdout, out.stderr[-600:])
        from make_test_model import build
        gray = os.path.join(d, "gray.keras")
        build(gray, channels=1)
        out = run_in_subprocess(self.CODE, {"ISL_MODEL_PATH": gray})
        self.assertIn("error 503 The model expects input (96, 96, 1)", out.stdout, out.stderr[-600:])

    def test_a_model_trained_at_128_pixels_is_used_at_128_pixels(self):
        from make_test_model import build
        path = os.path.join(tempfile.mkdtemp(), "m128.keras")
        build(path, img_size=128)
        code = (
            "import app, io; from PIL import Image; app.MODEL.wait(120); c=app.app.test_client();"
            "b=io.BytesIO(); Image.new('RGB',(720,540),(230,20,20)).save(b,'JPEG'); b.seek(0);"
            "r=c.post('/api/frame', data={'session':'x','hand':'1','debug':'1','frame':(b,'f.jpg')}, content_type='multipart/form-data').get_json();"
            "print('SIZE', app.MODEL.input_size, r['current_sign'], r['debug']['model_input_size'])"
        )
        out = run_in_subprocess(code, {"ISL_MODEL_PATH": path})
        self.assertIn("SIZE 128 अ 128", out.stdout, out.stderr[-600:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
