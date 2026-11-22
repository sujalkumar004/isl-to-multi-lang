#!/usr/bin/env python
"""
evaluate_model.py : find out, character by character, what your trained model can and cannot recognise.

It loads YOUR model (model/marathi_sign_model.keras) and the Marathi Sign Language dataset, then reports:

  * whether the class list matches the dataset and the model's outputs (label mapping check)
  * accuracy per character: precision, recall, which characters it is confused with
  * characters that are NEVER predicted (untrained / dead outputs)
  * a suitable confidence threshold per character (used by the website's adaptive threshold)
  * optional robustness tests (mirror image, hand smaller in the frame, rotation, dark, bright, blur)
  * pairs of characters whose training pictures already look alike (a DATASET limit, not a model bug)
  * sample dataset pictures per character, shown next to your webcam in the Developer panel

The result is written to model/model_report.json (+ model/reference/*.jpg). The website reads it
automatically: refresh the Developer panel after running this.

Setup (once):   pip install -r tools/requirements-tools.txt
Run:            python tools/evaluate_model.py
Options:
  --split test|all   test = the same 15 % held-out images the Colab notebook used (default).
                     Use "all" for a model trained with the ORIGINAL notebook (it trained on the first
                     8,000 images only, so nearly every image of the other 35 letters is unseen).
  --robustness       also run the perturbation tests (takes a few minutes)
  --limit N          use at most N images (quick check)
  --inspect-only     no dataset download: only the model structure and the label checks

YOUR OWN WEBCAM FRAMES (the most reliable test, needs no dataset and no internet):
  In the website's Developer mode, tick "Save frames" in the Character tester and record each sign. Then:
      python tools/evaluate_model.py --no-dataset --collected data/collected/test
  scores the model on those frames (ground truth = the character you were signing) with every preprocessing
  mode, tells you which mode works best on YOUR camera, and which characters pass the website's accept rules.
  Record a second, separate set for testing than the one you train on ("test" vs "train"), otherwise the
  result is too optimistic.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import diagnostics as dg  # noqa: E402
from preprocessing import build_views  # noqa: E402

SEVERITY = {"not_detected": 0, "poor": 1, "fair": 2, "good": 3, "untested": 4}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(ROOT / "model" / "marathi_sign_model.keras"))
    p.add_argument("--labels", default=str(ROOT / "model" / "labels.json"))
    p.add_argument("--out", default=str(ROOT / "model" / "model_report.json"))
    p.add_argument("--reference-dir", default=str(ROOT / "model" / "reference"))
    p.add_argument("--dataset", default="VinayHajare/Marathi-Sign-Language")
    p.add_argument("--split", choices=("test", "all"), default="test")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42, help="must equal SEED in the Colab notebook (42)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--robustness", action="store_true")
    p.add_argument("--robustness-n", type=int, default=3000)
    p.add_argument("--references", type=int, default=3, help="sample pictures per character")
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--collected", default=None, help="folder with frames recorded in the website, e.g. data/collected/test")
    p.add_argument("--no-dataset", action="store_true", help="skip the Hugging Face dataset (use with --collected)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- helpers
def heading(text):
    print("\n" + text + "\n" + "-" * len(text))


def load_model_and_labels(args):
    import tensorflow as tf
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)
    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}")
    model = tf.keras.models.load_model(args.model)
    return model, labels


def inspect(model, labels):
    heading("1. MODEL STRUCTURE")
    print(f"input  : {model.input_shape}")
    print(f"output : {model.output_shape}")
    print(f"params : {model.count_params():,}")
    for layer in model.layers:
        print(f"  {layer.name:28s} {type(layer).__name__:22s} {layer.count_params():>9,}")
    dims = model.input_shape[1:]
    problems = []
    if len(dims) != 3 or dims[0] != dims[1] or dims[2] != 3:
        problems.append(f"input {dims} is not a square colour image, which the website requires")
    heading("2. CLASS LIST (label mapping)")
    issues = dg.check_labels(labels, model.output_shape[-1])
    if not issues:
        print(f"OK: {len(labels)} unique labels, in the model's output order, no encoding problems.")
    for i in issues:
        print(f"{i['level'].upper()}: {i['message']}")
    return int(dims[0]), issues, problems


def load_images(ds, indices, size, label=""):
    from PIL import Image  # noqa: F401
    X = np.empty((len(indices), size, size, 3), dtype=np.uint8)
    for k, i in enumerate(indices):
        # identical to the notebook and to the website: RGB, resize((size, size)), /255 later
        X[k] = np.asarray(ds[int(i)]["image"].convert("RGB").resize((size, size)), dtype=np.uint8)
        if (k + 1) % 2000 == 0:
            print(f"  {label}{k + 1}/{len(indices)} images prepared")
    return X


def predict_all(model, X, batch=256):
    out = []
    for s in range(0, len(X), batch):
        out.append(np.asarray(model(X[s:s + batch].astype("float32") / 255.0, training=False)))
    return np.concatenate(out)


def stratified_split_indices(y, test_size, seed):
    """The exact split the Colab notebook makes (sklearn, stratified). Falls back to numpy if sklearn is absent."""
    try:
        from sklearn.model_selection import train_test_split
        _, test_idx = train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)
        return np.sort(test_idx), True
    except ImportError:
        print("WARNING: scikit-learn is not installed, so the split is NOT identical to the notebook's. "
              "Run: pip install -r tools/requirements-tools.txt")
        rng = np.random.default_rng(seed)
        chosen = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            chosen.extend(rng.choice(idx, size=max(1, int(round(len(idx) * test_size))), replace=False))
        return np.sort(np.array(chosen)), False


# --------------------------------------------------------------------------- robustness
def perturbations(size, seed=0):
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed)

    def per_image(fn):
        def run(X):
            return np.stack([np.asarray(fn(Image.fromarray(x)), dtype=np.uint8) for x in X])
        return run

    def zoom_out(im):
        small = im.resize((int(size * 0.7), int(size * 0.7)))
        canvas = Image.new("RGB", (size, size), (128, 128, 128))
        canvas.paste(small, ((size - small.size[0]) // 2, (size - small.size[1]) // 2))
        return canvas

    def rotate(im):
        return im.rotate(float(rng.uniform(-15, 15)), fillcolor=(128, 128, 128))

    return {
        "mirror": lambda X: X[:, :, ::-1, :].copy(),
        "zoom_out_70pct": per_image(zoom_out),
        "rotate_15deg": per_image(rotate),
        "darker": lambda X: (X.astype(np.float32) * 0.55).astype(np.uint8),
        "brighter": lambda X: np.clip(X.astype(np.float32) * 1.45, 0, 255).astype(np.uint8),
        "blur": per_image(lambda im: im.filter(ImageFilter.GaussianBlur(1.2))),
    }


def run_robustness(model, X, y, n_classes, n, seed):
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(X), size=min(n, len(X)), replace=False))
    Xs, ys = X[pick], y[pick]
    base = predict_all(model, Xs).argmax(1)
    overall = {"baseline": round(float((base == ys).mean()), 4)}
    per_class = {c: {} for c in range(n_classes)}
    for name, fn in perturbations(X.shape[1], seed).items():
        pred = predict_all(model, fn(Xs)).argmax(1)
        overall[name] = round(float((pred == ys).mean()), 4)
        for c in range(n_classes):
            m = ys == c
            if m.sum() >= 5:
                per_class[c][name] = round(float((pred[m] == c).mean()), 3)
        print(f"  {name:16s} accuracy {overall[name]:.1%}   (unperturbed {overall['baseline']:.1%})")
    return overall, per_class


# --------------------------------------------------------------------------- your own webcam frames
WEBCAM_MODES = {                      # name -> (view mode, mirrored), the same options the website has
    "full": ("full", False),
    "hand": ("hand", False),
    "multi": ("multi", False),
    "multi_mirrored": ("multi", True),
}
GATE = {"confidence": 0.70, "margin": 0.10, "view_agreement": 0.60}      # the website's default accept rules


def load_collected(folder):
    from PIL import Image  # noqa: F401
    items = []
    for jpg in sorted(Path(folder).glob("*/*.jpg")):
        meta = jpg.with_suffix(".json")
        if not jpg.parent.name.isdigit() or not meta.exists():
            continue
        info = json.loads(meta.read_text(encoding="utf-8"))
        items.append((jpg, int(jpg.parent.name), tuple(info["bbox"])))
    return items


def score_webcam(model, items, size, mode, mirrored, chunk=96):
    """probabilities (n, classes), crop agreement (n,) : exactly what the website computes for a frame."""
    from PIL import Image
    probs, agree = [], []
    for s in range(0, len(items), chunk):
        views_per_frame, flat = [], []
        for path, _, bbox in items[s:s + chunk]:
            views, _ = build_views(Image.open(path).convert("RGB"), bbox, mode, mirrored)
            views_per_frame.append(len(views))
            flat.extend(np.asarray(v.convert("RGB").resize((size, size)), dtype=np.uint8) for v in views)
        out = predict_all(model, np.stack(flat))
        pos = 0
        for n in views_per_frame:
            block = out[pos:pos + n]
            pos += n
            avg = block.mean(0)
            probs.append(avg)
            agree.append(float(np.mean(block.argmax(1) == avg.argmax())))
        print(f"    {mode}{' (mirrored)' if mirrored else ''}: {min(s + chunk, len(items))}/{len(items)} frames")
    return np.stack(probs), np.array(agree)


def evaluate_webcam(model, labels, size, folder):
    heading("WEBCAM FRAMES YOU RECORDED")
    items = load_collected(folder)
    if not items:
        sys.exit(f"No recorded frames found in {folder}. Use Developer mode > Character tester > Save frames.")
    y = np.array([c for _, c, _ in items])
    per_class = np.bincount(y, minlength=len(labels))
    print(f"{len(items)} frames, {int((per_class > 0).sum())} of {len(labels)} characters recorded "
          f"({per_class[per_class > 0].min()} to {per_class.max()} frames each)")
    modes, best, best_acc, multi = {}, None, -1.0, None
    for name, (mode, mirrored) in WEBCAM_MODES.items():
        probs, agree = score_webcam(model, items, size, mode, mirrored)
        classes, overall, confusions = dg.per_class_metrics(probs, y, labels)
        modes[name] = {"overall": overall, "classes": classes, "confusions": confusions[:15]}
        if name == "multi":
            multi = (probs, agree)
        if overall["accuracy"] > best_acc:
            best, best_acc = name, overall["accuracy"]
    # what the website would actually accept (confidence + gap to runner-up + crop agreement)
    probs, agree = multi
    order = np.argsort(probs, axis=1)
    top1, top2 = probs[np.arange(len(y)), order[:, -1]], probs[np.arange(len(y)), order[:, -2]]
    passed = (top1 >= GATE["confidence"]) & (top1 - top2 >= GATE["margin"]) & (agree >= GATE["view_agreement"])
    correct = order[:, -1] == y
    gated = {
        "rules": GATE,
        "pass_rate": round(float(passed.mean()), 4),
        "wrong_among_passed": round(float((~correct[passed]).mean()), 4) if passed.any() else None,
        "per_class": {labels[c]: {"pass_rate": round(float(passed[y == c].mean()), 3),
                                  "correct_when_passed": (round(float(correct[(y == c) & passed].mean()), 3) if ((y == c) & passed).any() else None)}
                      for c in range(len(labels)) if (y == c).any()},
    }
    for c in modes["multi"]["classes"]:
        g = gated["per_class"].get(c["label"])
        if g:
            c.update(pass_rate=g["pass_rate"], correct_when_passed=g["correct_when_passed"])
    heading("WEBCAM RESULT BY PREPROCESSING MODE")
    for name, m in modes.items():
        o = m["overall"]
        print(f"  {name:16s} accuracy {o['accuracy']:6.1%}   top-3 {o['top3']:6.1%}   never predicted: {len(o['classes_never_predicted'])}")
    print(f"  best mode on your camera: {best}")
    print(f"  with the website's accept rules (multi): {gated['pass_rate']:.0%} of frames pass, "
          + (f"{gated['wrong_among_passed']:.0%} of those are wrong" if gated["wrong_among_passed"] is not None else "none pass"))
    heading("WEBCAM RESULT PER CHARACTER (multi mode, worst first)")
    print(f"{'char':6s}{'status':14s}{'recall':>8s}{'passes':>8s}{'frames':>8s}  most confused with")
    for c in sorted((c for c in modes["multi"]["classes"] if c["support"]), key=lambda e: (SEVERITY[e["status"]], e["recall"])):
        conf = ", ".join(f"{a} x{n}" for a, n in c["confused_with"])
        print(f"{c['label']:6s}{c['status']:14s}{c['recall']:8.0%}{c.get('pass_rate', 0):8.0%}{c['support']:8d}  {conf}")
    missing = [labels[c] for c in range(len(labels)) if per_class[c] == 0]
    if missing:
        print(f"\nNot recorded yet: {' '.join(missing)}")
    return {
        "set": Path(folder).name, "frames": len(items), "recorded_classes": int((per_class > 0).sum()),
        "best_mode": best, "modes": modes, "gated": gated,
        "thresholds": {c["label"]: c["suggested_threshold"] for c in modes["multi"]["classes"] if c["suggested_threshold"] is not None},
    }


# --------------------------------------------------------------------------- dataset
def evaluate_dataset(args, model, labels, size, label_issues):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("The 'datasets' package is needed:  pip install -r tools/requirements-tools.txt\n"
                 "(or skip the dataset: --no-dataset --collected data/collected/test)")

    heading("3. DATASET")
    ds = load_dataset(args.dataset)["train"]
    ds_labels = list(ds.features["label"].names)
    y_all = np.array(ds["label"], dtype=np.int64)
    counts = np.bincount(y_all, minlength=len(ds_labels))
    print(f"{len(ds):,} images, {len(ds_labels)} classes, {counts.min()} to {counts.max()} images per class "
          f"(imbalance {counts.max() / max(1, counts.min()):.2f}x)")
    ds_issues = dg.check_labels(labels, model.output_shape[-1], ds_labels)
    mismatch = [i for i in ds_issues if "does not match the dataset" in i["message"]]
    print("Class order in model/labels.json matches the dataset: " + ("YES" if not mismatch else "NO"))
    for i in mismatch:
        print("ERROR:", i["message"])

    if args.split == "test":
        idx, exact = stratified_split_indices(y_all, args.test_size, args.seed)
        split_name = f"held-out test split ({args.test_size:.0%}, seed {args.seed})" + ("" if exact else ", approximate")
    else:
        idx = np.arange(len(ds))
        split_name = "all images"
    if args.limit and args.limit < len(idx):
        idx = np.sort(np.random.default_rng(args.seed).choice(idx, size=args.limit, replace=False))
        split_name += f", limited to {args.limit}"
    print(f"Evaluating on: {split_name} ({len(idx):,} images)")

    heading("4. PREDICTIONS")
    X = load_images(ds, idx, size, "test ")
    y = y_all[idx]
    probs = predict_all(model, X)
    classes, overall, confusions = dg.per_class_metrics(probs, y, labels)
    print(f"accuracy {overall['accuracy']:.1%} | top-3 {overall['top3']:.1%} | top-5 {overall['top5']:.1%} | macro F1 {overall['macro_f1']:.3f}")

    robustness = None
    if args.robustness:
        heading("5. ROBUSTNESS (what happens when the picture is not perfect)")
        overall_rob, per_class_rob = run_robustness(model, X, y, len(labels), args.robustness_n, args.seed)
        robustness = overall_rob
        for c in classes:
            c["robustness"] = per_class_rob.get(c["index"], {})

    heading("6. DATASET LOOK-ALIKES")
    means = np.stack([X[y == c].astype(np.float32).mean(0) if (y == c).any() else np.zeros_like(X[0], dtype=np.float32)
                      for c in range(len(labels))])
    similar = dg.similar_class_pairs(means, labels, top=10)
    for a, b, s in similar[:6]:
        print(f"  {a}  ~  {b}   similarity {s:.2f}")

    heading("7. SAMPLE PICTURES")
    ref_dir = Path(args.reference_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)
    for old in ref_dir.glob("*.jpg"):
        old.unlink()
    for c in range(len(labels)):
        pool = idx[y == c] if (y == c).any() else np.where(y_all == c)[0]
        for k, i in enumerate(pool[: args.references]):
            ds[int(i)]["image"].convert("RGB").save(ref_dir / f"{c}_{k}.jpg", quality=92)
    print(f"saved {args.references} pictures per character in {ref_dir}")

    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "split": split_name,
        "model": {"file": os.path.basename(args.model), "input_size": size, "params": int(model.count_params())},
        "dataset": {"name": args.dataset, "images": int(len(ds)), "labels": ds_labels, "labels_match": not mismatch,
                    "per_class_counts": counts.tolist(),
                    "imbalance_ratio": round(float(counts.max() / max(1, counts.min())), 3),
                    "similar_pairs": similar},
        "overall": overall,
        "confusions": confusions,
        "robustness": robustness,
        "classes": classes,
    }

    heading("RESULT PER CHARACTER (worst first)")
    print(f"{'char':6s}{'status':14s}{'recall':>8s}{'precision':>11s}{'tests':>7s}  most confused with")
    for c in sorted(classes, key=lambda e: (SEVERITY[e["status"]], e["recall"])):
        conf = ", ".join(f"{a} x{n}" for a, n in c["confused_with"])
        print(f"{c['label']:6s}{c['status']:14s}{c['recall']:8.0%}{c['precision']:11.0%}{c['support']:7d}  {conf}")
    summary = dg.summarize_report(report)
    print("\nSUMMARY:", ", ".join(f"{k} {v}" for k, v in summary.items() if v))
    if overall["classes_never_predicted"]:
        print(f"NEVER PREDICTED ({len(overall['classes_never_predicted'])}): {' '.join(overall['classes_never_predicted'])}")
        print("  -> the model cannot output these characters. That is a training problem, not a website problem: retrain it.")
    n_cal = sum(1 for c in classes if c["suggested_threshold"] is not None)
    print(f"Calibrated thresholds found for {n_cal} of {len(labels)} characters (used by the website's adaptive threshold).")
    if args.split == "all":
        print("NOTE: 'all' includes images the model may have trained on, so its results are optimistic for those.")
    return report


# --------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    model, labels = load_model_and_labels(args)
    size, label_issues, problems = inspect(model, labels)
    if problems:
        sys.exit("\n".join(problems))
    if args.inspect_only:
        print("\n(--inspect-only) Nothing was evaluated. Run without it to measure every character.")
        return 1 if any(i["level"] == "error" for i in label_issues) else 0
    if args.no_dataset and not args.collected:
        sys.exit("--no-dataset needs --collected <folder>.")

    out = Path(args.out)
    if args.no_dataset:
        report = dg.load_report(out) or {"model": {"file": os.path.basename(args.model), "input_size": size,
                                                     "params": int(model.count_params())},
                                         "split": "webcam frames only", "overall": None, "classes": [], "dataset": None}
    else:
        report = evaluate_dataset(args, model, labels, size, label_issues)
    report["generated"] = datetime.datetime.now().isoformat(timespec="seconds")
    if args.collected:
        report["webcam"] = evaluate_webcam(model, labels, size, args.collected)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport written to {out}. Refresh the website's Developer panel to see it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
