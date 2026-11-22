"""
diagnostics.py : checks and metrics that explain WHY some characters are not recognised.

Pure numpy / json (no TensorFlow), so everything here is unit-tested.

* check_labels()          integrity of the class list (duplicates, encoding, count vs model outputs)
* per_class_metrics()     precision / recall / confusions / suitable confidence per character
* calibrate_thresholds()  the lowest confidence at which a letter was still >= 95% precise
* class_status()          good / fair / poor / not_detected from measured recall and precision
* load_report()           reads model/model_report.json written by tools/evaluate_model.py
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np

# Status rules (shown in the Developer panel legend)
NOT_DETECTED_RECALL = 0.05
POOR_RECALL, POOR_PRECISION = 0.70, 0.60
GOOD_RECALL, GOOD_PRECISION = 0.90, 0.85
TARGET_PRECISION = 0.95
THRESHOLD_GRID = tuple(round(v, 2) for v in np.arange(0.30, 0.951, 0.05))
MIN_CALIBRATION_SAMPLES = 10


# --------------------------------------------------------------------------- labels
def check_labels(labels, num_outputs=None, dataset_labels=None):
    """List of {"level": "error"|"warning", "message": str}. Empty list = the label list is clean."""
    issues = []

    def add(level, message):
        issues.append({"level": level, "message": message})

    if not isinstance(labels, list) or not labels:
        add("error", "The label list is empty or is not a list.")
        return issues
    if num_outputs is not None and len(labels) != num_outputs:
        add("error", f"The model has {num_outputs} outputs but the label list has {len(labels)} entries, "
                     "so the labels cannot be mapped to the model's output indexes correctly.")
    seen = {}
    for i, lab in enumerate(labels):
        if not isinstance(lab, str) or not lab.strip():
            add("error", f"Label {i} is empty.")
            continue
        if lab in seen:
            add("error", f"Label {lab!r} appears twice (indexes {seen[lab]} and {i}).")
        seen.setdefault(lab, i)
        if unicodedata.normalize("NFC", lab) != lab:
            add("warning", f"Label {i} ({lab!r}) is not Unicode-NFC normalised; translation normalises it, "
                           "but comparisons with other files may fail.")
        if any(unicodedata.category(ch).startswith("C") for ch in lab):
            add("error", f"Label {i} contains an invisible control character.")
        if lab != lab.strip():
            add("warning", f"Label {i} ({lab!r}) has leading or trailing whitespace.")
    if dataset_labels is not None:
        if list(dataset_labels) != list(labels):
            diff = [i for i, (a, b) in enumerate(zip(labels, dataset_labels)) if a != b]
            add("error", "model/labels.json does not match the dataset's class order"
                         + (f" (first difference at index {diff[0]}: {labels[diff[0]]!r} vs {dataset_labels[diff[0]]!r})" if diff else
                            f" (different length: {len(labels)} vs {len(dataset_labels)})")
                         + ". Predictions would be shown as the wrong character.")
    return issues


# --------------------------------------------------------------------------- metrics
def class_status(entry):
    """good / fair / poor / not_detected / untested from one class entry of the report."""
    if entry is None or entry.get("support", 0) == 0:
        return "untested"
    recall, precision = entry.get("recall", 0.0), entry.get("precision", 0.0)
    if entry.get("predicted_count", 0) == 0 or recall < NOT_DETECTED_RECALL:
        return "not_detected"
    if recall < POOR_RECALL or precision < POOR_PRECISION:
        return "poor"
    if recall < GOOD_RECALL or precision < GOOD_PRECISION:
        return "fair"
    return "good"


def calibrate_thresholds(probs, y_true, num_classes, target_precision=TARGET_PRECISION):
    """
    For each class, the LOWEST confidence at which predictions of that class were still
    >= target_precision correct on held-out data (None if it never gets there, or there is too little data).
    """
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    out = []
    for c in range(num_classes):
        rows = y_pred == c
        conf = probs[rows, c]
        correct = (y_true[rows] == c)
        chosen = None
        for t in THRESHOLD_GRID:
            sel = conf >= t
            if sel.sum() >= MIN_CALIBRATION_SAMPLES and correct[sel].mean() >= target_precision:
                chosen = float(t)
                break
        out.append(chosen)
    return out


def per_class_metrics(probs, y_true, labels):
    """
    probs (n, classes), y_true (n,) -> (classes: list of dict, overall: dict, confusions: list)
    """
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true)
    n_classes = len(labels)
    y_pred = probs.argmax(axis=1)
    conf_top = probs.max(axis=1)
    thresholds = calibrate_thresholds(probs, y_true, n_classes)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    classes = []
    for c in range(n_classes):
        support = int(cm[c].sum())
        predicted = int(cm[:, c].sum())
        tp = int(cm[c, c])
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        right = (y_true == c) & (y_pred == c)
        wrong = (y_true == c) & (y_pred != c)
        row = cm[c].copy()
        row[c] = 0
        top_conf = [[labels[int(j)], int(row[j])] for j in np.argsort(row)[::-1][:3] if row[j] > 0]
        entry = {
            "index": c, "label": labels[c], "support": support, "predicted_count": predicted,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "mean_conf_correct": round(float(conf_top[right].mean()), 4) if right.any() else None,
            "mean_conf_wrong": round(float(conf_top[wrong].mean()), 4) if wrong.any() else None,
            "confused_with": top_conf,
            "suggested_threshold": thresholds[c],
        }
        entry["status"] = class_status(entry)
        classes.append(entry)

    top3 = float(np.mean([y_true[i] in np.argsort(probs[i])[::-1][:3] for i in range(len(y_true))])) if len(y_true) else 0.0
    top5 = float(np.mean([y_true[i] in np.argsort(probs[i])[::-1][:5] for i in range(len(y_true))])) if len(y_true) else 0.0
    supports = np.array([e["support"] for e in classes])
    f1s = np.array([e["f1"] for e in classes])
    overall = {
        "samples": int(len(y_true)),
        "accuracy": round(float((y_pred == y_true).mean()), 4) if len(y_true) else 0.0,
        "top3": round(top3, 4), "top5": round(top5, 4),
        "macro_f1": round(float(f1s[supports > 0].mean()), 4) if (supports > 0).any() else 0.0,
        "classes_never_predicted": [e["label"] for e in classes if e["support"] and e["predicted_count"] == 0],
        "classes_with_zero_recall": [e["label"] for e in classes if e["support"] and e["recall"] == 0],
    }
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:30], off.shape))[0]
    confusions = [[labels[int(t)], labels[int(p)], int(off[t, p])] for t, p in pairs if off[t, p] > 0]
    return classes, overall, confusions


def similar_class_pairs(class_means, labels, top=10):
    """
    Pairs of classes whose average training picture is most alike (cosine similarity).
    If two different letters look alike on average, the DATASET (not the model) makes them hard to tell apart.
    """
    m = np.asarray(class_means, dtype=np.float32).reshape(len(labels), -1)
    m = m - m.mean(axis=0, keepdims=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
    sim = m @ m.T
    np.fill_diagonal(sim, -1)
    iu = np.triu_indices(len(labels), 1)
    order = np.argsort(sim[iu])[::-1][:top]
    return [[labels[int(iu[0][k])], labels[int(iu[1][k])], round(float(sim[iu][k]), 3)] for k in order]


# --------------------------------------------------------------------------- report file
def load_report(path):
    """The offline evaluation report, or None if it does not exist or cannot be read."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def class_thresholds(report, labels):
    """
    {label: calibrated threshold}. Thresholds measured on YOUR webcam frames (tools/evaluate_model.py --collected)
    win over the ones measured on the dataset, because they describe the conditions you really use.
    """
    if not report:
        return {}
    out = {}
    for entry in report.get("classes", []):
        t = entry.get("suggested_threshold")
        if t is not None and entry.get("label") in labels:
            out[entry["label"]] = float(t)
    for label, t in ((report.get("webcam") or {}).get("thresholds") or {}).items():
        if label in labels and t is not None:
            out[label] = float(t)
    return out


def summarize_report(report):
    """Counts per status for the Developer panel header."""
    counts = {"good": 0, "fair": 0, "poor": 0, "not_detected": 0, "untested": 0}
    for entry in (report or {}).get("classes", []):
        counts[entry.get("status") or class_status(entry)] += 1
    return counts
