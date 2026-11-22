"""
preprocessing.py : how a camera frame becomes the image(s) the CNN sees.

The CNN was trained on centred, close-up 128x128 hand pictures (per the dataset card) that the
notebook squashed to 96x96. A raw webcam frame is a whole upper body, so the hand is small and the
frame is squashed by a different amount in x and y. These helpers offer three views of the same frame:

    full   the whole frame, exactly as the notebook did (default)
    hand   a square crop around the hand (MediaPipe bounding box)
    multi  three hand crops of different sizes; their probabilities are averaged

`compare_views()` builds every variant at once so the Developer panel can show, for the SAME frame,
what each preprocessing option would predict. Nothing here decides a character: it only prepares
pixels for the trained model.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from recognition import crop_to_hand

VIEW_MODES = ("full", "hand", "multi")
HAND_MARGIN = 0.35                      # the margin used for the single "hand" crop
MULTI_MARGINS = (0.15, 0.35, 0.60)      # tight, medium and loose crops around the hand


def mirror_bbox(bbox):
    """Bounding box (x0, y0, x1, y1) after the image is flipped left-right."""
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (1 - x1, y0, 1 - x0, y1)


def build_views(image, bbox, mode="full", mirror=False):
    """
    The image(s) to feed the model for the chosen mode.
    Returns (views, info). Without a hand box, "hand" and "multi" fall back to the full frame
    and `info["mode_used"]` says so.
    """
    if mirror:
        image = ImageOps.mirror(image)
        bbox = mirror_bbox(bbox)
    if mode == "full" or bbox is None:
        used = "full"
        views = [image]
    elif mode == "hand":
        used = "hand"
        views = [crop_to_hand(image, bbox, HAND_MARGIN)]
    else:
        used = "multi"
        views = [crop_to_hand(image, bbox, m) for m in MULTI_MARGINS]
    return views, {"mode_requested": mode, "mode_used": used, "mirrored": bool(mirror), "n_views": len(views)}


def compare_views(image, bbox):
    """
    Every preprocessing variant of one frame, for the Developer panel.
    {name: [PIL images]}  (the probabilities of a name's images are averaged).
    Without a hand box only the two full-frame variants exist.
    """
    out = {"full": [image], "mirror_full": [ImageOps.mirror(image)]}
    if bbox is not None:
        out["hand"] = [crop_to_hand(image, bbox, HAND_MARGIN)]
        out["multi"] = [crop_to_hand(image, bbox, m) for m in MULTI_MARGINS]
        out["mirror_hand"] = [crop_to_hand(ImageOps.mirror(image), mirror_bbox(bbox), HAND_MARGIN)]
    return out


def frame_quality(image, bbox=None):
    """
    Cheap measurements that explain many "the model gives nothing" cases:
    brightness (0..1), contrast (0..1) and how much of the frame the hand covers.
    """
    small = image.convert("L")
    small.thumbnail((160, 160))
    g = np.asarray(small, dtype=np.float32) / 255.0
    brightness = float(g.mean())
    contrast = float(g.std())
    hand_fraction = None
    hand_pixels = None
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        hand_fraction = float(max(0.0, (x1 - x0) * (y1 - y0)))
        hand_pixels = int(round(max(x1 - x0, y1 - y0) * max(image.size)))
    notes = []
    if brightness < 0.20:
        notes.append("The image is dark. Add light in front of you.")
    elif brightness > 0.88:
        notes.append("The image is overexposed. Reduce the light behind or on the hand.")
    if hand_fraction is not None and hand_fraction < 0.03:
        notes.append("The hand is very small in the frame. Move closer or use the hand crop.")
    return {
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "hand_fraction": None if hand_fraction is None else round(hand_fraction, 4),
        "hand_pixels": hand_pixels,
        "notes": notes,
    }
