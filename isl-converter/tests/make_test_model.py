"""
TEST FIXTURE ONLY. Builds a stand-in Keras model that classifies by IMAGE COLOUR so the server
pipeline (JPEG -> preprocessing -> model -> recognizer -> HTTP) can be tested with images whose
"sign" we control:  red -> class 0 (अ),  green -> class 1 (आ),  blue -> class 2 (इ),  black -> no class.
It is NOT your model and is never used by the website unless you point ISL_MODEL_PATH at it.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build(path, num_classes=43, img_size=96, channels=3):
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, channels)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation="softmax"),
    ])
    w = np.zeros((channels, num_classes), dtype="float32")
    for cls, ch in enumerate(range(min(3, channels))):
        w[:, cls] = -12.0
        w[ch, cls] = 40.0
    model.layers[-1].set_weights([w, np.zeros(num_classes, dtype="float32")])
    model.save(path)
    return model


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
