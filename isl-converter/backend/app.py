"""
app.py : web server for the Indian Sign Language Converter.

    python backend/app.py        ->  http://127.0.0.1:5000

The same server hosts the website (frontend/) and the API (/api/...).

    browser camera -> JPEG frame + "hand in view" flag -> POST /api/frame
        -> your CNN (model_service.py) -> ContinuousRecognizer (recognition.py)
        -> snapshot (state, sign, confidence, text) -> browser
    recognised text -> POST /api/translate -> English / Hindi / Marathi (translation.py)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import diagnostics
import translation
from model_service import ModelService
from preprocessing import build_views, compare_views, frame_quality
from recognition import ContinuousRecognizer, RecognizerConfig

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_DIR = ROOT / "model"
MODEL_PATH = Path(os.environ.get("ISL_MODEL_PATH", MODEL_DIR / "marathi_sign_model.keras"))
LABELS_PATH = Path(os.environ.get("ISL_LABELS_PATH", MODEL_DIR / "labels.json"))
REPORT_PATH = Path(os.environ.get("ISL_REPORT_PATH", MODEL_DIR / "model_report.json"))   # from tools/evaluate_model.py
REFERENCE_DIR = Path(os.environ.get("ISL_REFERENCE_DIR", MODEL_DIR / "reference"))       # dataset sample pictures
DATA_DIR = Path(os.environ.get("ISL_DATA_DIR", ROOT / "data"))                           # frames you record yourself
SET_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_SAMPLES_PER_CLASS = 3000

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")

with open(LABELS_PATH, encoding="utf-8") as f:
    LABELS = json.load(f)

# The model is loaded once, in the background, so the page can show "Loading recognition model..."
MODEL = ModelService(MODEL_PATH, len(LABELS))
MODEL.start()


# ---------------------------------------------------------------------------
# Sessions: one recognizer (text buffer + lock state) per browser tab
# ---------------------------------------------------------------------------
_threshold_cache = {"mtime": None, "value": {}}


def current_thresholds():
    """Calibrated per-class thresholds from model/model_report.json (empty if there is no report)."""
    try:
        mtime = REPORT_PATH.stat().st_mtime
    except OSError:
        _threshold_cache.update(mtime=None, value={})
        return {}
    if _threshold_cache["mtime"] != mtime:
        _threshold_cache.update(mtime=mtime, value=diagnostics.class_thresholds(diagnostics.load_report(REPORT_PATH), LABELS))
    return _threshold_cache["value"]


class Session:
    def __init__(self):
        self.recognizer = ContinuousRecognizer(LABELS, RecognizerConfig(), class_thresholds=current_thresholds())
        self.lock = threading.Lock()
        self.last_used = time.monotonic()


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()


def get_session(session_id) -> Session:
    session_id = str(session_id or "default")[:64]
    with SESSIONS_LOCK:
        now = time.monotonic()
        for sid in [s for s, v in SESSIONS.items() if now - v.last_used > 3600]:
            del SESSIONS[sid]
        session = SESSIONS.setdefault(session_id, Session())
        session.last_used = now
        return session


def parse_hand(value):
    if value in ("1", "true", "True"):
        return True
    if value in ("0", "false", "False"):
        return False
    return None  # unknown: the hand gate is ignored for this frame


def parse_bbox(value):
    try:
        x0, y0, x1, y1 = (float(v) for v in value.split(","))
        if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
            return x0, y0, x1, y1
    except (AttributeError, ValueError):
        pass
    return None


def parse_collect(index, set_name):
    """(class index, set name) if the browser asked to record this frame, else None. Bad values are ignored."""
    try:
        i = int(index)
    except (TypeError, ValueError):
        return None
    name = (set_name or "train").strip()
    if 0 <= i < len(LABELS) and SET_NAME.match(name):
        return i, name
    return None


def session_and_body():
    data = request.get_json(silent=True) or {}
    return get_session(request.form.get("session") or data.get("session")), data



# ---------------------------------------------------------------------------
# Running the model on one frame
# ---------------------------------------------------------------------------
def _data_uri(image, size):
    """The picture the model actually receives (resized like prepare_image) as a small JPEG."""
    buf = io.BytesIO()
    image.convert("RGB").resize((size, size)).save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _combine(scores):
    """Average the probabilities of the views; agreement = share of views whose own top guess is the final answer."""
    avg = scores.mean(axis=0)
    agreement = float(np.mean(scores.argmax(axis=1) == int(avg.argmax())))
    return avg, agreement


def run_model(image, bbox, cfg, debug):
    """
    Returns (probabilities used for recognition, crop agreement, debug info or None).
    The probabilities always come from the trained CNN. In debug mode the SAME batch also scores every
    preprocessing variant, so the Developer panel can show which one the model likes best for this frame.
    """
    views, info = build_views(image, bbox, cfg.view_mode, cfg.mirror_input)
    if not debug:
        probs, agreement = _combine(MODEL.predict_views(views))
        return probs, agreement, None

    variants = compare_views(image, bbox)
    flat = list(views) + [im for ims in variants.values() for im in ims]
    scores = MODEL.predict_views(flat)
    probs, agreement = _combine(scores[: len(views)])
    compare, offset = {}, len(views)
    for name, ims in variants.items():
        avg = scores[offset: offset + len(ims)].mean(axis=0)
        offset += len(ims)
        order = avg.argsort()[::-1][:3]
        compare[name] = {"views": len(ims), "top": [[LABELS[int(i)], round(float(avg[int(i)]), 4)] for i in order]}
    selected = {("full", False): "full", ("full", True): "mirror_full", ("hand", False): "hand",
                ("hand", True): "mirror_hand", ("multi", False): "multi"}.get((info["mode_used"], info["mirrored"]))
    view_scores = scores[: len(views)]
    return probs, agreement, {
        "compare": compare,
        "selected": selected,
        "views": info,
        "view_agreement": agreement,
        "view_tops": [[LABELS[int(v.argmax())], round(float(v.max()), 4)] for v in view_scores],
        "quality": frame_quality(image, bbox),
        "bbox": list(bbox) if bbox else None,
        "frame_size": list(image.size),
        "model_input": _data_uri(views[0], MODEL.input_size),
        "model_input_size": MODEL.input_size,
    }


# ---------------------------------------------------------------------------
# Recording your own webcam frames (for testing and retraining on YOUR camera)
# ---------------------------------------------------------------------------
def save_sample(raw, image, bbox, index, set_name):
    """
    Store the exact frame the model was given, plus the hand box, under data/collected/<set>/<class index>/.
    Everything stays on this computer. Returns the number of samples now stored for that class, or None if full.
    """
    folder = DATA_DIR / "collected" / set_name / str(index)
    folder.mkdir(parents=True, exist_ok=True)
    count = sum(1 for _ in folder.glob("*.jpg"))
    if count >= MAX_SAMPLES_PER_CLASS:
        return None
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**9:09d}"
    (folder / f"{stem}.jpg").write_bytes(raw)
    meta = {"index": index, "label": LABELS[index], "bbox": list(bbox), "frame_size": list(image.size)}
    (folder / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return count + 1


def collected_counts():
    """{set: {class index: number of stored frames}}"""
    out = {}
    root = DATA_DIR / "collected"
    if root.is_dir():
        for set_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            counts = {}
            for cls in set_dir.iterdir():
                if cls.is_dir() and cls.name.isdigit():
                    counts[int(cls.name)] = sum(1 for _ in cls.glob("*.jpg"))
            out[set_dir.name] = counts
    return out


# ---------------------------------------------------------------------------
# Website
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/api/info")
def api_info():
    return jsonify(
        model=MODEL.status(),
        labels=LABELS,
        config=RecognizerConfig().as_dict(),
        limits=RecognizerConfig.LIMITS,
        choices=RecognizerConfig.CHOICES,
        languages=translation.UI_LANGUAGES,
    )


@app.get("/api/collected")
def api_collected():
    """How many of your own webcam frames are stored per set and character."""
    return jsonify(sets=collected_counts(), folder=str(DATA_DIR / "collected"), max_per_class=MAX_SAMPLES_PER_CLASS)


@app.get("/api/diagnostics")
def api_diagnostics():
    """
    Everything the Developer panel needs to answer "which characters can this model recognise?":
    label integrity, the model's structure and the offline per-character evaluation report (if any).
    """
    report = diagnostics.load_report(REPORT_PATH)
    status = MODEL.status()
    outputs = MODEL.num_classes if not MODEL.ready else status["classes"]
    dataset_labels = ((report or {}).get("dataset") or {}).get("labels")
    issues = diagnostics.check_labels(LABELS, outputs if MODEL.ready else None, dataset_labels)
    classes = []
    by_index = {c["index"]: c for c in (report or {}).get("classes", [])}
    for i, label in enumerate(LABELS):
        entry = dict(by_index.get(i) or {})
        entry.update(index=i, label=label)
        entry["status"] = entry.get("status") or diagnostics.class_status(by_index.get(i))
        classes.append(entry)
    webcam = (report or {}).get("webcam")
    if webcam:
        by_mode = {mode: {c["index"]: c for c in m["classes"]} for mode, m in webcam["modes"].items()}
        for entry in classes:
            main = by_mode.get("multi", {}).get(entry["index"])
            if main and main.get("support"):
                entry["webcam"] = {
                    "support": main["support"], "recall": main["recall"], "precision": main["precision"],
                    "status": main["status"], "confused_with": main.get("confused_with", []),
                    "pass_rate": main.get("pass_rate"), "correct_when_passed": main.get("correct_when_passed"),
                    "recall_by_mode": {m: by_mode[m][entry["index"]]["recall"] for m in by_mode if entry["index"] in by_mode[m]},
                }
    n_ref = 0
    if REFERENCE_DIR.is_dir():
        n_ref = len(list(REFERENCE_DIR.glob("0_*.jpg")))
    return jsonify(
        labels=LABELS,
        label_issues=issues,
        model=dict(status, **MODEL.describe()),
        report=None if report is None else {
            "generated": report.get("generated"),
            "split": report.get("split"),
            "overall": report.get("overall"),
            "dataset": {k: v for k, v in (report.get("dataset") or {}).items() if k != "labels"},
            "confusions": report.get("confusions", []),
            "robustness": report.get("robustness"),
            "summary": diagnostics.summarize_report(report),
            "webcam": None if not webcam else {
                "set": webcam["set"], "frames": webcam["frames"], "recorded_classes": webcam["recorded_classes"],
                "best_mode": webcam["best_mode"], "gated": {k: webcam["gated"][k] for k in ("rules", "pass_rate", "wrong_among_passed")},
                "mode_accuracy": {m: v["overall"]["accuracy"] for m, v in webcam["modes"].items()},
                "webcam_summary": diagnostics.summarize_report({"classes": webcam["modes"]["multi"]["classes"]}),
            },
        },
        classes=classes,
        references_per_class=n_ref,
        calibrated_classes=len(current_thresholds()),
        legend={
            "not_detected": f"recall below {diagnostics.NOT_DETECTED_RECALL:.0%} or never predicted",
            "poor": f"recall below {diagnostics.POOR_RECALL:.0%} or precision below {diagnostics.POOR_PRECISION:.0%}",
            "fair": f"recall below {diagnostics.GOOD_RECALL:.0%} or precision below {diagnostics.GOOD_PRECISION:.0%}",
            "good": "recall and precision are high",
            "untested": "no offline test data for this character",
        },
    )


@app.get("/reference/<path:name>")
def reference_image(name):
    """Dataset sample pictures exported by tools/evaluate_model.py (what the model was trained on)."""
    return send_from_directory(REFERENCE_DIR, name)


@app.post("/api/frame")
def api_frame():
    """One camera frame (JPEG) + hand flag -> recognizer snapshot."""
    if not MODEL.ready:
        status = MODEL.status()
        message = "Loading recognition model..." if status["state"] == "loading" else status["error"]
        return jsonify(error=message, model_state=status["state"]), 503

    session = get_session(request.form.get("session"))
    hand = parse_hand(request.form.get("hand"))
    bbox = parse_bbox(request.form.get("bbox"))
    debug = request.form.get("debug") == "1"
    started = time.perf_counter()

    collect = parse_collect(request.form.get("collect"), request.form.get("collect_set"))

    with session.lock:
        rec = session.recognizer
        probs, agreement, extra, saved = None, 1.0, None, None
        if rec.needs_prediction(hand):
            upload = request.files.get("frame")
            if upload is None:
                return jsonify(error="frame missing"), 400
            raw = upload.read()
            try:
                image = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:  # noqa: BLE001
                return jsonify(error="frame is not a valid image"), 400
            probs, agreement, extra = run_model(image, bbox, rec.config, debug)
            if collect and hand and bbox:
                saved = save_sample(raw, image, bbox, collect[0], collect[1])
        result = rec.process(probs, hand_present=hand, view_agreement=agreement, debug=debug)
        if debug:
            result["debug"] = extra or {"bbox": list(bbox) if bbox else None}
        if saved is not None:
            result["collected"] = saved

    result["server_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return jsonify(result)


def _manual(action: str):
    session, _ = session_and_body()
    with session.lock:
        rec = session.recognizer
        changed = getattr(rec, action)()
        event = {"add_space": "space", "backspace": "backspace", "clear": "clear"}[action]
        snap = rec.snapshot(event=event if changed or action == "clear" else None)
    return jsonify(snap)


@app.post("/api/space")
def api_space():
    return _manual("add_space")


@app.post("/api/backspace")
def api_backspace():
    return _manual("backspace")


@app.post("/api/clear")
def api_clear():
    return _manual("clear")


@app.post("/api/resume")
def api_resume():
    """Recognition (re)starts: forget old frames and any lock, keep the text."""
    session, _ = session_and_body()
    with session.lock:
        session.recognizer.reset_recognition()
        return jsonify(session.recognizer.snapshot(event="resume"))


@app.post("/api/config")
def api_config():
    session, data = session_and_body()
    with session.lock:
        session.recognizer.config.update(**(data.get("config") or {}))
        return jsonify(config=session.recognizer.config.as_dict(), snapshot=session.recognizer.snapshot())


@app.post("/api/translate")
def api_translate():
    """Translate the COMPLETE recognised text into one or more languages."""
    data = request.get_json(silent=True) or {}
    requested = data.get("targets") or data.get("target") or ["en"]
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, list) or not requested:
        return jsonify(error="targets must be a list of language codes"), 400
    codes = []
    for value in requested:
        code = translation.resolve_language(value)
        if code is None:
            return jsonify(error=f"Unsupported language: {value}"), 400
        codes.append(code)
    results = translation.translate_many(data.get("text", ""), codes)
    return jsonify(translations=results, source=translation.SOURCE_LANGUAGE)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nIndian Sign Language Converter: http://127.0.0.1:{port}\n")
    # threaded: the model loads in the background and several requests can be in flight.
    app.run(host="127.0.0.1", port=port, threaded=True)
