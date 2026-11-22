"""
Logic tests for backend/recognition.py (ContinuousRecognizer).

IMPORTANT: these feed *synthetic* probability vectors into the state machine. They test the
temporal logic (debounce, lock, release, space, backspace). They say nothing about how
accurate the CNN is on real camera frames.

The H/E/L/O tests use a stand-in vocabulary because the trained model's classes are Marathi
letters (see labels.json). The state machine is label-agnostic, so the logic is identical.

Run:  python -m unittest discover -s tests -v
"""
import json
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from recognition import (  # noqa: E402
    ACCEPTED, DETECTING, LOCKED, STABLE, WAITING,
    ContinuousRecognizer, RecognizerConfig, crop_to_hand,
    predict_multiple_frames, summarize_predictions,
)

FPS = 10
DT = 1.0 / FPS

LATIN = ["H", "E", "L", "O", "W", "R", "D", "X", "Y", "Z"]  # stand-in vocabulary
LABELS_FILE = os.path.join(os.path.dirname(__file__), "..", "model", "labels.json")


def load_labels():
    with open(LABELS_FILE, encoding="utf-8") as f:
        return json.load(f)


def prob(labels, label, conf=0.92):
    """Probability vector whose top class is `label` with probability `conf`."""
    p = np.full(len(labels), (1 - conf) / (len(labels) - 1), dtype=np.float32)
    p[labels.index(label)] = conf
    return p


class Rig:
    """Drives a recognizer with a fake 10 fps clock."""

    def __init__(self, labels, **cfg):
        self.labels = labels
        self.rec = ContinuousRecognizer(labels, RecognizerConfig(**cfg))
        self.t = 0.0
        self.snaps = []

    def _step(self, probs, hand, view_agreement=1.0):
        self.t += DT
        snap = self.rec.process(probs, hand_present=hand, now=self.t, view_agreement=view_agreement)
        self.snaps.append(snap)
        return snap

    def sign(self, label, frames, conf=0.92, hand=True):
        for _ in range(frames):
            snap = self._step(prob(self.labels, label, conf), hand)
        return snap

    def no_hand(self, seconds):
        snap = None
        for _ in range(int(round(seconds * FPS))):
            snap = self._step(None, False)
        return snap

    @property
    def text(self):
        return self.rec.text_buffer


class TestDuplicatePrevention(unittest.TestCase):
    def test_holding_H_for_3_seconds_gives_single_H(self):
        r = Rig(LATIN)
        r.sign("H", 3 * FPS)                 # 30 frames of H
        self.assertEqual(r.text, "H")
        self.assertNotEqual(r.text, "H" * 30)

    def test_holding_H_for_a_minute_still_single_H(self):
        r = Rig(LATIN)
        r.sign("H", 600)
        self.assertEqual(r.text, "H")

    def test_acceptance_happens_exactly_once(self):
        r = Rig(LATIN)
        r.sign("H", 40)
        accepted = [s["accepted"] for s in r.snaps if s["accepted"]]
        self.assertEqual(accepted, ["H"])

    def test_state_sequence_for_one_held_sign(self):
        r = Rig(LATIN)
        r.sign("H", 40)
        states = [s["state"] for s in r.snaps]
        self.assertEqual(states[0], DETECTING)                 # window filling
        self.assertEqual(states[5], ACCEPTED)                  # 6th frame fills the window
        self.assertTrue(all(s == LOCKED for s in states[6:]))  # locked while held
        self.assertEqual(r.snaps[-1]["current_sign"], "H")     # still displayed while locked
        self.assertEqual(r.snaps[-1]["locked_sign"], "H")

    def test_isolated_flicker_frames_do_not_duplicate_or_add_letters(self):
        for seed in range(200):
            rng = random.Random(seed)
            r = Rig(LATIN)
            for i in range(120):
                # ~10 % of frames flicker to another class, never two in a row
                if i > 6 and rng.random() < 0.10 and r.snaps[-1].get("_flick") is None:
                    other = rng.choice([c for c in LATIN if c != "H"])
                    snap = r._step(prob(LATIN, other, 0.9), True)
                    snap["_flick"] = True
                else:
                    r.sign("H", 1)
            self.assertEqual(r.text, "H", f"seed {seed}")

    def test_wobble_shorter_than_release_window_does_not_duplicate(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        for label in ["E", "L", "O", "W", "R", "D", "X"]:  # 7 different frames, none 5/6-stable
            r._step(prob(LATIN, label, 0.6), True)
        r.sign("H", 30)
        self.assertEqual(r.text, "H")

    def test_low_confidence_is_never_accepted(self):
        r = Rig(LATIN)
        snap = r.sign("H", 100, conf=0.55)    # threshold is 0.70
        self.assertEqual(r.text, "")
        self.assertEqual(snap["state"], STABLE)  # steady but not confident

    def test_new_sign_with_low_confidence_reports_stable_not_locked(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        snap = r.sign("E", 7, conf=0.55)       # steady new sign, below the 0.70 threshold
        self.assertEqual(snap["state"], STABLE)
        self.assertIn("E", snap["message"])
        self.assertEqual(r.text, "H")          # nothing added
        self.assertEqual(snap["locked_sign"], "H")
        r.sign("E", 40, conf=0.55)             # however long it stays unconfident, nothing is added
        self.assertEqual(r.text, "H")
        r.sign("E", 10, conf=0.92)             # user signs it clearly
        self.assertEqual(r.text, "HE")

    def test_partial_hold_is_not_accepted(self):
        r = Rig(LATIN)
        r.sign("H", 4)                         # window (6) is not full yet
        self.assertEqual(r.text, "")


class TestWordConstruction(unittest.TestCase):
    def test_H_E_L_L_O(self):
        """H -> E -> L -> L -> O, lowering the hand between the two L's."""
        r = Rig(LATIN)
        r.sign("H", 15)
        self.assertEqual(r.text, "H")
        r.sign("E", 15)
        self.assertEqual(r.text, "HE")
        r.sign("L", 15)
        self.assertEqual(r.text, "HEL")
        r.no_hand(0.6)                         # hand lowered: releases the lock
        r.sign("L", 15)
        self.assertEqual(r.text, "HELL")
        r.sign("O", 15)
        self.assertEqual(r.text, "HELLO")

    def test_H_E_L_L_O_does_not_double_when_L_is_held_and_no_hand_gap(self):
        r = Rig(LATIN)
        for label in ["H", "E", "L", "O"]:
            r.sign(label, 20)
        self.assertEqual(r.text, "HELO")        # LL needs a release; one held L is one L

    def test_double_letter_by_relaxed_hand_still_in_view(self):
        """No hand gate signal: the sign must leave the top prediction for release_frames."""
        r = Rig(LATIN, require_hand=False)
        r.sign("L", 15)
        for label in ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"]:   # 9 frames, all different
            r._step(prob(LATIN, label, 0.5), None)
        r.sign("L", 15)
        self.assertEqual(r.text, "LL")

    def test_fast_changes_are_all_captured(self):
        r = Rig(LATIN)
        for label in ["H", "E", "L", "O"]:
            r.sign(label, 8)                   # only ~0.8 s per sign
        self.assertEqual(r.text, "HELO")

    def test_HELLO_space_WORLD(self):
        r = Rig(LATIN)
        for i, label in enumerate("HELLO"):
            if i == 3:
                r.no_hand(0.6)
            r.sign(label, 15)
        r.no_hand(0.5)
        self.assertTrue(r.rec.add_space())     # manual SPACE
        for i, label in enumerate("WORLD"):
            r.sign(label, 15)
            r.no_hand(0.5)
        self.assertEqual(r.text, "HELLO WORLD")

    def test_same_sign_after_hand_removed_is_accepted_again(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.no_hand(0.5)
        r.sign("H", 15)
        self.assertEqual(r.text, "HH")

    def test_short_detector_dropout_does_not_release(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r._step(None, False)                   # 0.1 s < hand_absent_seconds
        r._step(None, False)
        r.sign("H", 20)
        self.assertEqual(r.text, "H")

    def test_hand_removal_between_letters_changes_state_to_waiting(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        snap = r.no_hand(0.6)
        self.assertEqual(snap["state"], WAITING)
        self.assertIsNone(snap["locked_sign"])
        self.assertEqual(snap["window_fill"], 0)   # stale frames are forgotten

    def test_a_transition_through_hand_removal_cannot_reaccept_stale_frames(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.no_hand(0.5)
        # hand returns, but only 3 frames: stale H frames must not count
        r.sign("H", 3)
        self.assertEqual(r.text, "H")


class TestManualControls(unittest.TestCase):
    def test_space_backspace_clear(self):
        r = Rig(LATIN)
        self.assertFalse(r.rec.add_space())    # no leading space
        r.sign("H", 15)
        self.assertTrue(r.rec.add_space())
        self.assertFalse(r.rec.add_space())    # no double space
        self.assertEqual(r.text, "H ")
        self.assertTrue(r.rec.backspace())
        self.assertEqual(r.text, "H")
        r.rec.clear()
        self.assertEqual(r.text, "")
        self.assertIsNone(r.rec.snapshot()["locked_sign"])

    def test_backspace_while_sign_still_held_does_not_readd(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.rec.backspace()
        r.sign("H", 30)
        self.assertEqual(r.text, "")

    def test_backspace_on_empty_is_safe(self):
        r = Rig(LATIN)
        self.assertFalse(r.rec.backspace())

    def test_backspace_removes_whole_conjunct(self):
        labels = load_labels()
        r = Rig(labels)
        r.sign("क्ष", 10)
        r.sign("ल", 10)
        self.assertEqual(r.text, "क्षल")
        r.rec.backspace()
        r.rec.backspace()
        self.assertEqual(r.text, "")           # notebook's text_buffer[:-1] would leave 'क्'

    def test_reset_recognition_keeps_text_but_forgets_lock_and_window(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.rec.reset_recognition()
        snap = r.rec.snapshot()
        self.assertEqual(r.text, "H")               # text kept (pause / resume must not lose it)
        self.assertIsNone(snap["locked_sign"])
        self.assertEqual(snap["window_fill"], 0)
        r.sign("H", 3)
        self.assertEqual(r.text, "H")               # 3 frames are not enough after a reset
        r.sign("H", 10)
        self.assertEqual(r.text, "HH")              # sign is read afresh after resuming

    def test_clear_allows_sign_to_be_recognised_again(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.rec.clear()
        r.sign("H", 15)
        self.assertEqual(r.text, "H")


class TestAutoSpace(unittest.TestCase):
    def test_off_by_default(self):
        r = Rig(LATIN)
        r.sign("H", 15)
        r.no_hand(5)
        self.assertEqual(r.text, "H")

    def test_adds_exactly_one_space_per_long_pause(self):
        r = Rig(LATIN, auto_space=True, auto_space_seconds=2.0)
        r.sign("H", 15)
        r.no_hand(1.5)
        self.assertEqual(r.text, "H")           # not long enough
        r.no_hand(1.0)
        self.assertEqual(r.text, "H ")
        r.no_hand(5)
        self.assertEqual(r.text, "H ")          # never a second space
        r.sign("E", 15)
        self.assertEqual(r.text, "H E")

    def test_no_space_before_first_letter(self):
        r = Rig(LATIN, auto_space=True)
        r.no_hand(6)
        self.assertEqual(r.text, "")


class TestGateAndConfig(unittest.TestCase):
    def test_needs_prediction_skips_frames_without_hand(self):
        rec = ContinuousRecognizer(LATIN)
        self.assertFalse(rec.needs_prediction(False))
        self.assertTrue(rec.needs_prediction(True))
        self.assertTrue(rec.needs_prediction(None))    # detector unavailable
        rec.config.update(require_hand=False)
        self.assertTrue(rec.needs_prediction(False))

    def test_unknown_hand_status_disables_gate_but_recognition_works(self):
        r = Rig(LATIN)
        for _ in range(15):
            r._step(prob(LATIN, "H"), None)
        self.assertEqual(r.text, "H")

    def test_config_update_clamps_and_ignores_unknown(self):
        cfg = RecognizerConfig()
        cfg.update(confidence_threshold=5, window_size=100, min_agreement=0.0, bogus=1)
        self.assertEqual(cfg.confidence_threshold, 0.99)
        self.assertEqual(cfg.window_size, 15)
        self.assertEqual(cfg.min_agreement, 0.5)
        self.assertFalse(hasattr(cfg, "bogus"))

    def test_lowering_threshold_at_runtime_takes_effect(self):
        r = Rig(LATIN)
        r.sign("H", 20, conf=0.55)
        self.assertEqual(r.text, "")
        r.rec.config.update(confidence_threshold=0.5)
        r.sign("H", 2, conf=0.55)
        self.assertEqual(r.text, "H")

    def test_process_requires_probs_when_hand_present(self):
        rec = ContinuousRecognizer(LATIN)
        with self.assertRaises(ValueError):
            rec.process(None, hand_present=True)


class TestNotebookFunctions(unittest.TestCase):
    def test_summarize_matches_notebook_definition(self):
        labels = ["a", "b", "c"]
        frames = [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.8, 0.1, 0.1]]
        char, conf, agree, top, fc = summarize_predictions(frames, labels)
        self.assertEqual(char, "a")
        self.assertAlmostEqual(conf, np.mean([f[0] for f in frames]), places=5)
        self.assertAlmostEqual(agree, 3 / 4)
        self.assertEqual(fc, ["a", "a", "b", "a"])
        self.assertEqual([t[0] for t in top], ["a", "b", "c"])

    def test_predict_multiple_frames_wrapper(self):
        from PIL import Image
        labels = ["a", "b"]
        model = lambda x, training=False: np.array([[0.9, 0.1]] * len(x), dtype=np.float32)
        imgs = [Image.new("RGB", (200, 150)) for _ in range(3)]
        char, conf, agree, top, fc = predict_multiple_frames(model, labels, imgs)
        self.assertEqual((char, agree), ("a", 1.0))

    def test_crop_to_hand_is_square_and_inside_image(self):
        from PIL import Image
        img = Image.new("RGB", (720, 540))
        crop = crop_to_hand(img, (0.02, 0.5, 0.2, 0.9))     # hand near the corner
        self.assertEqual(crop.size[0], crop.size[1])
        self.assertGreater(crop.size[0], 0)


def two_way(labels, first, second, p1, p2):
    """Probability vector with two competing classes and the rest sharing what is left."""
    v = np.full(len(labels), (1 - p1 - p2) / (len(labels) - 2), dtype=np.float32)
    v[labels.index(first)] = p1
    v[labels.index(second)] = p2
    return v


class TestAmbiguityAndCalibration(unittest.TestCase):
    def test_a_coin_flip_between_two_letters_is_not_accepted_even_above_threshold(self):
        r = Rig(LATIN, confidence_threshold=0.40)
        for _ in range(20):
            snap = r._step(two_way(LATIN, "H", "E", 0.46, 0.44), True)     # 2 points apart
        self.assertEqual(r.text, "")
        self.assertEqual(snap["state"], STABLE)
        self.assertIn("too close to E", snap["message"])
        self.assertAlmostEqual(snap["margin"], 0.02, places=2)

    def test_same_scores_with_a_clear_margin_are_accepted(self):
        r = Rig(LATIN, confidence_threshold=0.40)
        for _ in range(8):
            r._step(two_way(LATIN, "H", "E", 0.60, 0.20), True)
        self.assertEqual(r.text, "H")

    def test_margin_can_be_switched_off(self):
        r = Rig(LATIN, confidence_threshold=0.40, min_margin=0.0)
        for _ in range(8):
            r._step(two_way(LATIN, "H", "E", 0.46, 0.44), True)
        self.assertEqual(r.text, "H")

    def test_calibrated_threshold_lowers_the_bar_only_for_that_letter(self):
        rec = ContinuousRecognizer(LATIN, RecognizerConfig(), class_thresholds={"H": 0.50})
        self.assertEqual(rec.threshold_for("H"), 0.50)
        self.assertEqual(rec.threshold_for("E"), 0.70)          # no calibration: global
        r = Rig(LATIN)
        r.rec.class_thresholds = {"H": 0.50}
        r.sign("H", 12, conf=0.62)
        r.sign("E", 12, conf=0.62)
        self.assertEqual(r.text, "H")                            # H accepted at 62 %, E is not

    def test_calibration_can_never_go_below_the_floor_or_above_the_global_threshold(self):
        rec = ContinuousRecognizer(LATIN, RecognizerConfig(), class_thresholds={"H": 0.05, "E": 0.95})
        self.assertEqual(rec.threshold_for("H"), 0.40)           # floor
        self.assertEqual(rec.threshold_for("E"), 0.70)           # never stricter than the slider
        rec.config.update(adaptive_threshold=False)
        self.assertEqual(rec.threshold_for("H"), 0.70)

    def test_snapshot_reports_the_threshold_that_was_used(self):
        r = Rig(LATIN)
        r.rec.class_thresholds = {"H": 0.55}
        snap = r.sign("H", 8, conf=0.60)
        self.assertEqual(snap["threshold"], 0.70)
        self.assertEqual(snap["threshold_used"], 0.55)


class TestDebugOutput(unittest.TestCase):
    def test_top5_stable_frames_and_margin(self):
        r = Rig(LATIN)
        snap = r.sign("H", 4)
        self.assertEqual(len(snap["top"]), 5)
        self.assertEqual(snap["top"][0][0], "H")
        self.assertEqual(snap["stable_frames"], 4)
        self.assertGreater(snap["margin"], 0.8)
        r._step(prob(LATIN, "E"), True)
        self.assertEqual(r.snaps[-1]["stable_frames"], 1)       # the run restarts when the top-1 changes

    def test_debug_snapshot_contains_raw_probabilities(self):
        rec = ContinuousRecognizer(LATIN)
        snap = rec.process(prob(LATIN, "H"), True, now=0.1, debug=True)
        self.assertEqual(len(snap["probs"]), len(LATIN))
        self.assertEqual(snap["instant_top"][0][0], "H")
        self.assertNotIn("probs", rec.process(prob(LATIN, "H"), True, now=0.2))   # not sent unless asked

    def test_counters_show_frames_lost_to_the_hand_gate(self):
        r = Rig(LATIN)
        r.sign("H", 5)
        r.no_hand(0.5)
        c = r.snaps[-1]["counters"]
        self.assertEqual(c["processed"], 5)
        self.assertEqual(c["skipped_no_hand"], 5)


class TestFastAccept(unittest.TestCase):
    def test_off_by_default(self):
        r = Rig(LATIN)
        r.sign("H", 4, conf=0.97)
        self.assertEqual(r.text, "")

    def test_clear_sign_is_accepted_after_three_frames(self):
        r = Rig(LATIN, fast_accept=True)
        r.sign("H", 2, conf=0.96)
        self.assertEqual(r.text, "")
        r.sign("H", 1, conf=0.96)
        self.assertEqual(r.text, "H")

    def test_needs_real_confidence_not_just_three_frames(self):
        r = Rig(LATIN, fast_accept=True)
        r.sign("H", 4, conf=0.80)                                # above threshold but below fast_confidence
        self.assertEqual(r.text, "")

    def test_holding_still_gives_one_character(self):
        r = Rig(LATIN, fast_accept=True)
        r.sign("H", 300, conf=0.96)
        self.assertEqual(r.text, "H")

    def test_fast_sign_changes_are_all_captured(self):
        r = Rig(LATIN, fast_accept=True)
        for label in ["H", "E", "L", "O"]:
            r.sign(label, 4, conf=0.96)
        self.assertEqual(r.text, "HELO")

    def test_random_flicker_cannot_trigger_fast_accept(self):
        for seed in range(100):
            rng = random.Random(seed)
            r = Rig(LATIN, fast_accept=True)
            for _ in range(80):
                r._step(prob(LATIN, rng.choice(LATIN), 0.96), True)   # confident but random
            self.assertLessEqual(len(r.text), 8, f"seed {seed}: {r.text}")


class TestPreprocessingOptions(unittest.TestCase):
    def test_view_mode_accepts_only_known_values(self):
        cfg = RecognizerConfig()
        cfg.update(view_mode="hand")
        self.assertEqual(cfg.view_mode, "hand")
        cfg.update(view_mode="bogus")
        self.assertEqual(cfg.view_mode, "hand")
        cfg.update(view_mode="multi", mirror_input=1, min_margin=9)
        self.assertEqual((cfg.view_mode, cfg.mirror_input, cfg.min_margin), ("multi", True, 0.6))


class TestCropAgreement(unittest.TestCase):
    """The model looked at several crops of the hand; if they disagree the frame is not trustworthy."""

    def feed(self, r, label, n, agreement, conf=0.95):
        for _ in range(n):
            snap = r._step(prob(LATIN, label, conf), True, view_agreement=agreement)
        return snap

    def test_confident_but_disagreeing_crops_are_not_accepted(self):
        r = Rig(LATIN)
        snap = self.feed(r, "H", 20, agreement=1 / 3)        # each frame: only 1 of 3 crops agrees
        self.assertEqual(r.text, "")
        self.assertEqual(snap["state"], STABLE)
        self.assertIn("Different crops", snap["message"])
        self.assertAlmostEqual(snap["view_agreement"], 1 / 3, places=2)

    def test_two_of_three_crops_is_enough(self):
        r = Rig(LATIN)
        self.feed(r, "H", 8, agreement=2 / 3)
        self.assertEqual(r.text, "H")

    def test_single_view_modes_are_unaffected(self):
        r = Rig(LATIN)
        r.sign("H", 8)                                       # default view_agreement = 1.0
        self.assertEqual(r.text, "H")

    def test_the_gate_can_be_switched_off(self):
        r = Rig(LATIN, min_view_agreement=0.0)
        self.feed(r, "H", 8, agreement=0.0)
        self.assertEqual(r.text, "H")

    def test_agreement_is_averaged_over_the_window_so_one_bad_frame_is_tolerated(self):
        r = Rig(LATIN)
        self.feed(r, "H", 5, agreement=1.0)
        self.feed(r, "H", 1, agreement=0.0)                  # one frame where the crops disagreed
        self.assertEqual(r.text, "H")                         # mean 0.83 >= 0.60

    def test_fast_accept_also_needs_the_crops_to_agree(self):
        r = Rig(LATIN, fast_accept=True)
        self.feed(r, "H", 3, agreement=1 / 3)
        self.assertEqual(r.text, "")
        r2 = Rig(LATIN, fast_accept=True)
        self.feed(r2, "H", 3, agreement=1.0)
        self.assertEqual(r2.text, "H")

    def test_window_and_agreement_history_are_cleared_together(self):
        r = Rig(LATIN)
        self.feed(r, "H", 8, agreement=1.0)
        r.no_hand(0.5)
        self.assertEqual(len(r.rec.window), len(r.rec.view_window))
        self.assertEqual(len(r.rec.view_window), 0)

    def test_default_view_mode_is_multi(self):
        self.assertEqual(RecognizerConfig().view_mode, "multi")


class TestRealLabelsWord(unittest.TestCase):
    def test_marathi_word_aai_with_real_classes(self):
        """'आई' (mother) uses two classes the model was actually trained on: आ then ई."""
        labels = load_labels()
        self.assertEqual(len(labels), 43)
        r = Rig(labels)
        r.sign("आ", 15)
        r.sign("ई", 15)
        self.assertEqual(r.text, "आई")


if __name__ == "__main__":
    unittest.main(verbosity=2)
