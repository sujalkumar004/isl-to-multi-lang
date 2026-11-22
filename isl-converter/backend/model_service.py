"""
model_service.py : loads the trained Keras CNN ONCE, in the background, and serialises predictions.

The web server starts immediately; the page shows "Loading recognition model..." until
`state` becomes "ready". If the file is missing or does not fit, state becomes "error" with a
message that says what to fix. Nothing here ever fabricates a prediction.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from recognition import IMG_SIZE, predict_views

LOADING, READY, ERROR = "loading", "ready", "error"


class ModelService:
    def __init__(self, model_path, num_classes: int):
        self.path = Path(model_path)
        self.num_classes = num_classes
        self.state = LOADING
        self.error = None
        self.input_size = IMG_SIZE       # replaced by the model's real input size once loaded
        self._model = None
        self._predict_lock = threading.Lock()
        self._thread = None
        self._done = threading.Event()

    # ------------------------------------------------------------------ loading
    def start(self):
        """Begin loading in a background thread (idempotent)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self.load, name="model-loader", daemon=True)
            self._thread.start()

    def wait(self, timeout=None) -> bool:
        return self._done.wait(timeout)

    def load(self):
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Could not load the model: {exc}")
        finally:
            self._done.set()

    def _fail(self, message: str):
        self.error = message
        self.state = ERROR

    def _load(self):
        if not self.path.exists():
            return self._fail(
                f"Model file not found: {self.path.name}. Copy marathi_sign_model.keras "
                f"(saved by your notebook) into the model/ folder and restart the server."
            )
        try:
            import tensorflow as tf
        except ImportError:
            return self._fail("TensorFlow is not installed. Run: pip install -r backend/requirements.txt")

        model = tf.keras.models.load_model(self.path)
        in_shape, out_shape = model.input_shape, model.output_shape
        dims = tuple(in_shape[1:]) if len(in_shape) == 4 else None
        if dims is None or dims[0] != dims[1] or dims[2] != 3 or dims[0] is None or dims[0] < 32:
            return self._fail(
                f"The model expects input {tuple(in_shape[1:])}. The website prepares square colour "
                f"images (for example {IMG_SIZE}x{IMG_SIZE}x3 like your notebook), so it cannot use this model."
            )
        if out_shape[-1] != self.num_classes:
            return self._fail(
                f"The model has {out_shape[-1]} outputs but model/labels.json lists {self.num_classes} classes."
            )
        self.input_size = int(dims[0])
        model(np.zeros((1, self.input_size, self.input_size, 3), dtype="float32"), training=False)  # warm-up
        self._model = model
        self.state = READY  # set last: readers only see READY once the model is usable

    # ---------------------------------------------------------------- prediction
    @property
    def ready(self) -> bool:
        return self.state == READY

    def predict_views(self, images):
        """Several PIL images in ONE model call -> (n, classes) probabilities."""
        if not self.ready:
            raise RuntimeError("model is not ready")
        with self._predict_lock:
            return predict_views(self._model, images, self.input_size)

    def predict(self, pil_image):
        """One PIL image -> probability vector. Raises RuntimeError if the model is not ready."""
        return self.predict_views([pil_image])[0]

    def describe(self) -> dict:
        """Architecture summary for the Developer panel."""
        if self._model is None:
            return {}
        layers = []
        for layer in self._model.layers:
            try:
                shape = [d for d in layer.output.shape]
            except Exception:  # noqa: BLE001
                shape = None
            layers.append({"name": layer.name, "type": type(layer).__name__, "params": int(layer.count_params()),
                           "output_shape": shape})
        return {
            "input_shape": [d for d in self._model.input_shape],
            "output_shape": [d for d in self._model.output_shape],
            "params": int(self._model.count_params()),
            "layers": layers,
        }

    def status(self) -> dict:
        return {
            "state": self.state,
            "ready": self.ready,
            "error": self.error,
            "classes": self.num_classes,
            "input_size": self.input_size,
        }
