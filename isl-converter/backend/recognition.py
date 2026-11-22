"""
recognition.py
==============
Continuous character recognition built around the EXISTING CNN from CAPSTONE.ipynb.

What is kept from the notebook (same behaviour)
-----------------------------------------------
* prepare_image()          RGB, resize, /255.0                     (unchanged; size = model input)
* predict_single_frame()   one image -> 43 class probabilities     (model(x) instead of
                                                                    model.predict(x), same numbers,
                                                                    much lower per-call overhead)
* predict_multiple_frames() average probabilities, majority label,
                           frame agreement, top-3                  (now a thin wrapper)
* CONFIDENCE_THRESHOLD / frame agreement / text_buffer / space / backspace / reset

What is new
-----------
* summarize_predictions()  the aggregation half of predict_multiple_frames(), split out so the
                           SAME maths can run on a rolling window of live frames.
* RecognizerConfig         all tunable numbers in one place.
* ContinuousRecognizer     the state machine that turns a stream of frames into text
                           without repeating a character while a sign is held.

This file has no TensorFlow import, so the logic can be unit-tested anywhere.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, fields

import numpy as np
from PIL import Image

IMG_SIZE = 96  # must match the size the CNN was trained with

# Recognition states shown in the UI
WAITING = "WAITING"      # no hand in view (or nothing analysed yet)
DETECTING = "DETECTING"  # hand present, window still filling / sign not steady
STABLE = "STABLE"        # sign is steady but confidence is below the threshold
ACCEPTED = "ACCEPTED"    # a character was appended on this frame
LOCKED = "LOCKED"        # that character is locked until the sign changes / is released


# =====================================================================
# 1. Functions carried over from the notebook
# =====================================================================
def prepare_image(pil_image, img_size=IMG_SIZE):
    """
    Identical to the notebook (training cell 1 and inference cells 10/11):
    RGB -> resize((size, size)) -> float32 / 255.  Verified numerically: max difference 0.0.
    `img_size` is the model's own input size (96 for the original notebook model).
    """
    image = pil_image.convert("RGB").resize((img_size, img_size))
    return np.array(image, dtype=np.float32) / 255.0


def model_input_size(model, default=IMG_SIZE):
    """The square input size the model expects (falls back to the notebook's 96)."""
    try:
        return int(model.input_shape[1])
    except Exception:  # noqa: BLE001  (test doubles have no input_shape)
        return default


def predict_views(model, images, img_size=None):
    """Several images in ONE model call -> (n, num_classes) probabilities."""
    size = img_size or model_input_size(model)
    batch = np.stack([prepare_image(im, size) for im in images])
    return np.asarray(model(batch, training=False), dtype=np.float32)


def predict_single_frame(model, pil_image, img_size=None):
    """
    One image -> probability vector (num_classes,).

    CHANGED: the notebook used model.predict(image, verbose=0). predict() builds a
    dataset/progress loop on every call (tens of ms). For a live stream we call the
    model directly, which gives the same probabilities in a few ms.
    `model` is now a parameter instead of a global so the function is testable.
    """
    return predict_views(model, [pil_image], img_size)[0]


def summarize_predictions(prob_list, labels, top_k=3):
    """
    The aggregation half of the notebook's predict_multiple_frames(), operating on
    probability vectors instead of images.

    Returns the same 5-tuple the notebook returned:
        (final_character, final_confidence, frame_agreement (0..1),
         top_predictions [(label, prob), ...], frame_characters [label per frame])
    """
    all_predictions = np.asarray(prob_list, dtype=np.float32)
    frame_characters = [labels[int(np.argmax(p))] for p in all_predictions]

    average_prediction = np.mean(all_predictions, axis=0)
    final_index = int(np.argmax(average_prediction))
    final_character = labels[final_index]
    final_confidence = float(average_prediction[final_index])

    frame_agreement = frame_characters.count(final_character) / len(frame_characters)

    top_indices = np.argsort(average_prediction)[-top_k:][::-1]
    top_predictions = [(labels[int(i)], float(average_prediction[int(i)])) for i in top_indices]

    return final_character, final_confidence, frame_agreement, top_predictions, frame_characters


def predict_multiple_frames(model, labels, images):
    """Notebook behaviour preserved: predict every image, then summarise."""
    return summarize_predictions([predict_single_frame(model, im) for im in images], labels)


def crop_to_hand(pil_image, bbox, margin=0.35):
    """
    Preprocessing used by the "hand" and "multi" view modes (multi is the default).
    bbox = (x0, y0, x1, y1) in 0..1 image coordinates from the MediaPipe hand landmarks.
    Returns a square crop around the hand. The Colab notebook repeats this exact geometry when it trains
    on frames you recorded, so training and live use see the same kind of picture.
    """
    width, height = pil_image.size
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
    side = max((x1 - x0) * width, (y1 - y0) * height) * (1 + 2 * margin)
    size = int(round(min(max(side, 64.0), float(min(width, height)))))
    left = int(round(min(max(cx - size / 2, 0.0), width - size)))
    top = int(round(min(max(cy - size / 2, 0.0), height - size)))
    return pil_image.crop((left, top, left + size, top + size))


# =====================================================================
# 2. Configuration
# =====================================================================
@dataclass
class RecognizerConfig:
    # Same meaning as the notebook. The notebook used 0.70 in one cell and 0.80 in the next.
    confidence_threshold: float = 0.70

    # A character is accepted only if the last `window_size` frames agree.
    # 6 frames at ~10 fps is ~0.6 s of holding the sign.
    window_size: int = 6
    # Fraction of the window whose top prediction must equal the winning character.
    # 0.80 with 6 frames = at least 5 of 6, so a single flickering frame is tolerated.
    min_agreement: float = 0.80

    # AMBIGUITY REJECTION. The winner's average probability must beat the runner-up by at least
    # this much. "क 45% vs ख 44%" is a coin flip, not a recognition, whatever the threshold says.
    min_margin: float = 0.10

    # CALIBRATED PER-CLASS THRESHOLDS. If model/model_report.json exists (written by
    # tools/evaluate_model.py), a letter that was found to be reliable at a lower confidence on
    # held-out data may be accepted below the global threshold, but never below `threshold_floor`
    # and never above the global threshold. Without a report this has no effect.
    adaptive_threshold: bool = True
    threshold_floor: float = 0.40

    # FAST ACCEPT (off by default). For very clear signs: `fast_frames` identical top-1 frames in a
    # row, each averaging >= `fast_confidence`, are accepted without waiting for the full window.
    fast_accept: bool = False
    fast_frames: int = 3
    fast_confidence: float = 0.90

    # To type the SAME letter twice the sign must be released first:
    # `release_frames` consecutive frames in which the locked sign is not the top prediction
    # (~0.8 s at 10 fps). Deliberately larger than window_size: a held sign that wobbles for a
    # few frames must never be mistaken for a new one. Lowering the hand out of view
    # releases the lock much sooner (hand_absent_seconds).
    release_frames: int = 8

    # MediaPipe hand gate (hand_present comes from the browser).
    require_hand: bool = True
    hand_absent_seconds: float = 0.30  # short dropouts of the detector are ignored

    # Optional automatic space: hand lowered for this long adds ONE space. Off by default.
    auto_space: bool = False
    auto_space_seconds: float = 2.0

    # PREPROCESSING options handled by the server before the CNN (not by the state machine).
    #   full  = the whole camera frame squashed to the model size (what the notebook did)
    #   hand  = a square crop around the hand found by MediaPipe
    #   multi = the average of three hand crops of different sizes
    #
    # Default "multi". Measured with the real model on two real webcam captures (see README): the same
    # frame gave four different letters depending on framing, and the whole-frame mode was 99% sure of a
    # letter the hand crops disagreed with. Crops around the hand match the centred close-ups the model
    # was trained on; averaging three of them makes the model honestly uncertain when it is unsure.
    view_mode: str = "multi"
    mirror_input: bool = False   # flip the frame left-right before recognition (left-handed signers)

    # CROP AGREEMENT. In "multi" mode the model looks at three crops of the same hand. This is the average
    # fraction of those crops whose own top guess equals the final answer, over the frame window. 0.60 means
    # that on average at least 2 of 3 crops must agree. With one view (full / hand) it is always 1.0.
    min_view_agreement: float = 0.60

    LIMITS = {
        "confidence_threshold": (0.10, 0.99),
        "window_size": (3, 15),
        "min_agreement": (0.50, 1.00),
        "min_margin": (0.0, 0.60),
        "min_view_agreement": (0.0, 1.0),
        "threshold_floor": (0.20, 0.90),
        "fast_frames": (2, 6),
        "fast_confidence": (0.70, 0.99),
        "release_frames": (2, 30),
        "hand_absent_seconds": (0.10, 2.00),
        "auto_space_seconds": (1.0, 10.0),
    }
    CHOICES = {"view_mode": ("full", "hand", "multi")}

    def update(self, **changes):
        """Apply a partial update with type coercion and clamping. Unknown keys / values are ignored."""
        names = {f.name for f in fields(self)}
        for key, value in changes.items():
            if key not in names:
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, str):
                if value not in self.CHOICES.get(key, ()):
                    continue
            elif isinstance(current, int):
                value = int(round(float(value)))
            else:
                value = float(value)
            if key in self.LIMITS:
                lo, hi = self.LIMITS[key]
                value = min(max(value, lo), hi)
            setattr(self, key, value)
        return self

    def as_dict(self):
        return asdict(self)


# =====================================================================
# 3. The continuous recognizer
# =====================================================================
class ContinuousRecognizer:
    """
    Turns a stream of per-frame probability vectors into a text buffer.

        WAITING -> DETECTING -> STABLE -> ACCEPTED -> LOCKED -> DETECTING ...

    Rules
    -----
    1. A frame only counts if a hand is present (when the hand gate is on).
    2. The last `window_size` frames are summarised with the notebook's own maths
       (average probability, majority label, frame agreement).
    3. ACCEPT when the window is full, agreement >= min_agreement, the average confidence reaches
       the threshold for that letter, AND the winner beats the runner-up by min_margin.
       The character is appended ONCE and LOCKED.
    4. While LOCKED nothing is appended for that sign, however long it is held.
    5. The lock is released by
         a) a different sign becoming stable (it is accepted straight away), or
         b) `release_frames` consecutive frames where the locked sign is not the top
            prediction (hand relaxed / moved away), or
         c) the hand leaving the camera for `hand_absent_seconds`.
       Cases b) and c) also clear the frame window so stale frames of the old sign can
       never re-trigger it. This is what makes a deliberate double letter possible:
       lower the hand between the two.
    """

    SPACE = " "

    def __init__(self, labels, config=None, class_thresholds=None):
        self.labels = list(labels)
        self.config = config or RecognizerConfig()
        self.class_thresholds = dict(class_thresholds or {})   # label -> calibrated threshold
        self.tokens = []          # accepted characters / spaces (one entry per sign)
        self._reset_tracking()

    # ---------------------------------------------------------------- state
    def _reset_tracking(self):
        self.window = deque()     # per-frame probability vectors
        self.view_window = deque()  # per-frame crop agreement (1.0 when there is a single view)
        self.locked = None        # character currently locked (or None)
        self._release_count = 0
        self._absent_since = None
        self._auto_space_done = False
        self.state = WAITING
        self.message = "Show a sign to the camera."
        self._hand = None
        self.frames_processed = 0
        self.frames_skipped_no_hand = 0

    @property
    def text_buffer(self):
        """Same name and meaning as the notebook's text_buffer (a string)."""
        return "".join(self.tokens)

    def needs_prediction(self, hand_present):
        """False when the frame can be skipped (hand gate on and no hand): saves CNN calls."""
        return not (self.config.require_hand and hand_present is False)

    def threshold_for(self, label):
        """Global threshold, or the calibrated per-class one (only ever LOWER, never below the floor)."""
        cfg = self.config
        if cfg.adaptive_threshold and label in self.class_thresholds:
            return min(cfg.confidence_threshold, max(cfg.threshold_floor, self.class_thresholds[label]))
        return cfg.confidence_threshold

    def reset_recognition(self):
        """
        Forget the frame window and any lock but KEEP the text.
        Used when recognition is resumed after a pause and when the camera is restarted, so old
        frames from before the pause can never be mixed with new ones.
        """
        self._reset_tracking()

    # -------------------------------------------------------------- manual keys
    def add_space(self):
        """Manual SPACE. No leading space and no double space."""
        if self.tokens and self.tokens[-1] != self.SPACE:
            self.tokens.append(self.SPACE)
            return True
        return False

    def backspace(self):
        """
        Remove the last sign. Works on whole tokens, so a conjunct such as 'क्ष'
        (three Unicode code points) is removed as one unit; the notebook's
        text_buffer[:-1] would have left a broken 'क्'.
        The lock is kept, so a sign that is still being held is not re-added.
        """
        if self.tokens:
            self.tokens.pop()
            return True
        return False

    def clear(self):
        """Reset: empty text AND fresh recognition state."""
        self.tokens = []
        self._reset_tracking()

    # -------------------------------------------------------------- assessment
    def _assess(self):
        """Everything the decision (and the debug panel) needs, from the current window."""
        cfg = self.config
        if not self.window:
            return {"empty": True, "char": None, "conf": 0.0, "agree_fraction": 0.0, "agreement": 0,
                    "top": [], "frame_chars": [], "margin": 0.0, "second": None, "stable": False,
                    "confident": False, "threshold": cfg.confidence_threshold, "stable_frames": 0,
                    "view_agreement": 1.0, "consistent": True}
        char, conf, agree_fraction, top, frame_chars = summarize_predictions(
            list(self.window), self.labels, top_k=5
        )
        margin = top[0][1] - top[1][1] if len(top) > 1 else top[0][1]
        second = top[1][0] if len(top) > 1 else None
        run = 1
        while run < len(frame_chars) and frame_chars[-1 - run] == frame_chars[-1]:
            run += 1
        threshold = self.threshold_for(char)
        stable = len(self.window) >= cfg.window_size and agree_fraction >= cfg.min_agreement - 1e-9
        view_agreement = float(np.mean(self.view_window)) if self.view_window else 1.0
        consistent = view_agreement >= cfg.min_view_agreement - 1e-9
        confident = conf >= threshold and margin >= cfg.min_margin - 1e-9 and consistent
        return {"empty": False, "view_agreement": view_agreement, "consistent": consistent, "char": char, "conf": conf, "agree_fraction": agree_fraction,
                "agreement": frame_chars.count(char), "top": top, "frame_chars": frame_chars,
                "margin": margin, "second": second, "stable": stable, "confident": confident,
                "threshold": threshold, "stable_frames": run}

    def _fast_candidate(self, frame_chars):
        """A very clear sign: fast_frames identical top-1 frames in a row, each strongly confident."""
        cfg = self.config
        n = cfg.fast_frames
        if not cfg.fast_accept or len(self.window) < n or len(set(frame_chars[-n:])) != 1:
            return None
        if self.view_window and float(np.mean(list(self.view_window)[-n:])) < cfg.min_view_agreement - 1e-9:
            return None
        recent = np.mean(np.stack(list(self.window)[-n:]), axis=0)
        order = np.argsort(recent)[::-1]
        label = self.labels[int(order[0])]
        if label != frame_chars[-1]:
            return None
        if recent[order[0]] >= cfg.fast_confidence and recent[order[0]] - recent[order[1]] >= cfg.min_margin:
            return label
        return None

    # -------------------------------------------------------------- main step
    def process(self, probs, hand_present=None, now=None, debug=False, view_agreement=1.0):
        """
        Feed one frame. Returns a snapshot dict for the UI.
        probs        probability vector, or None if the frame was skipped
                     (see needs_prediction()).
        hand_present True / False from MediaPipe, or None if unknown (gate ignored).
        now          timestamp in seconds (defaults to time.monotonic()); injectable for tests.
        debug        add the raw per-frame probabilities to the snapshot.
        view_agreement fraction of the frame's crops whose own top guess matches the final answer (1.0 = one view).
        """
        cfg = self.config
        now = time.monotonic() if now is None else now
        self._trim_window()
        self._hand = hand_present
        event = None
        accepted = None

        gate_on = cfg.require_hand and hand_present is not None

        # ---- 1. hand gate ------------------------------------------------
        if gate_on and not hand_present:
            self.frames_skipped_no_hand += 1
            if self._absent_since is None:
                self._absent_since = now
            absent_for = now - self._absent_since

            if absent_for >= cfg.hand_absent_seconds:
                if self.locked is not None or self.window:
                    event = "released"
                self.locked = None
                self._release_count = 0
                self._clear_window()
                self.state = WAITING
                self.message = "Waiting for a hand in view."
                if cfg.auto_space and not self._auto_space_done and absent_for >= cfg.auto_space_seconds:
                    if self.add_space():
                        event = "auto_space"
                    self._auto_space_done = True
            # shorter than the grace period: detector dropout, keep everything as it is
            return self._snapshot(event=event, debug=debug)

        self._absent_since = None
        self._auto_space_done = False

        if probs is None:
            raise ValueError("probs is required when a hand is present or the gate is off")

        # ---- 2. rolling window (uses the notebook's aggregation) -------------
        self.frames_processed += 1
        self.window.append(np.asarray(probs, dtype=np.float32))
        self.view_window.append(float(min(max(view_agreement, 0.0), 1.0)))
        self._trim_window()
        a = self._assess()
        char, stable, confident = a["char"], a["stable"], a["confident"]

        fast = self._fast_candidate(a["frame_chars"])
        if fast is not None:
            char, stable, confident = fast, True, True

        # ---- 3. lock handling ---------------------------------------------
        if self.locked is not None:
            if stable and confident and char != self.locked:
                accepted = self._accept(char)            # sign changed -> next character
            else:
                if a["frame_chars"][-1] == self.locked:
                    self._release_count = 0
                else:
                    self._release_count += 1
                if self._release_count >= cfg.release_frames:
                    self.locked = None                   # sign released (neutral / relaxed hand)
                    self._release_count = 0
                    self._clear_window()                  # forget stale frames of the old sign
                    return self._finish(None, "released", debug)

        # ---- 4. accept a new stable character -------------------------------------
        elif stable and confident:
            accepted = self._accept(char)

        return self._finish(accepted, event, debug)

    # -------------------------------------------------------------- internals
    def _trim_window(self):
        while len(self.window) > self.config.window_size:
            self.window.popleft()
        while len(self.view_window) > self.config.window_size:
            self.view_window.popleft()

    def _clear_window(self):
        self.window.clear()
        self.view_window.clear()

    def _accept(self, char):
        self.tokens.append(char)
        self.locked = char
        self._release_count = 0
        return char

    def _finish(self, accepted, event, debug=False):
        cfg = self.config
        a = self._assess()
        char = a["char"]

        # a steady sign that differs from the locked one (it was not accepted, so it is not confident)
        new_sign_pending = self.locked is not None and a["stable"] and char != self.locked

        if accepted is not None:
            self.state = ACCEPTED
            self.message = f"Added {accepted}. Change the sign to continue."
            event = "accepted"
        elif self.locked is not None and not new_sign_pending:
            self.state = LOCKED
            self.message = f"{self.locked} is locked. Change the sign or lower your hand for the next one."
        elif a["stable"]:
            self.state = STABLE
            if a["conf"] < a["threshold"]:
                self.message = (
                    f"Steady sign {char}, but confidence {a['conf']:.0%} is below the "
                    f"{a['threshold']:.0%} threshold, so it is not added."
                )
            elif a["margin"] < cfg.min_margin - 1e-9:
                self.message = (
                    f"{char} is too close to {a['second']} ({a['margin']:.0%} apart, "
                    f"{cfg.min_margin:.0%} needed), so it is not added."
                )
            else:
                self.message = (
                    f"Different crops of your hand give different answers (only {a['view_agreement']:.0%} agree, "
                    f"{cfg.min_view_agreement:.0%} needed), so {char} is not added."
                )
        else:
            self.state = DETECTING
            self.message = (
                f"Reading sign: {len(self.window)}/{cfg.window_size} frames."
                if len(self.window) < cfg.window_size
                else "Sign is not steady yet."
            )
        return self._snapshot(event=event, accepted=accepted, debug=debug)

    def snapshot(self, event=None, debug=False):
        """Current view of the recognizer without feeding a frame (used after manual keys)."""
        return self._snapshot(event=event, debug=debug)

    def _snapshot(self, event=None, accepted=None, debug=False):
        cfg = self.config
        a = self._assess()
        snap = {
            "state": self.state,
            "message": self.message,
            "event": event,
            "accepted": accepted,
            "current_sign": a["char"],
            "confidence": a["conf"],
            "margin": a["margin"],
            "agreement": a["agreement"],
            "agreement_fraction": a["agree_fraction"],
            "stable_frames": a["stable_frames"],
            "window_size": cfg.window_size,
            "window_fill": len(self.window),
            "window_chars": a["frame_chars"],
            "locked_sign": self.locked,
            "release_count": self._release_count,
            "hand": self._hand,
            "text": self.text_buffer,
            "tokens": list(self.tokens),
            "top": [[label, prob] for label, prob in a["top"]],
            "threshold": cfg.confidence_threshold,
            "threshold_used": a["threshold"],
            "min_margin": cfg.min_margin,
            "view_agreement": a["view_agreement"],
            "min_view_agreement": cfg.min_view_agreement,
            "counters": {"processed": self.frames_processed, "skipped_no_hand": self.frames_skipped_no_hand},
        }
        if debug:
            last = self.window[-1] if self.window else None
            avg = np.mean(np.stack(list(self.window)), axis=0) if self.window else None
            snap["instant_top"] = (
                [[self.labels[int(i)], round(float(last[int(i)]), 4)] for i in np.argsort(last)[::-1][:5]]
                if last is not None else []
            )
            snap["probs"] = [round(float(v), 4) for v in last] if last is not None else []
            snap["avg_probs"] = [round(float(v), 4) for v in avg] if avg is not None else []
        return snap
