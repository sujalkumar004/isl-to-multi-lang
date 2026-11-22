# Marathi Sign Language: corrected training. Paste this whole file into ONE Colab cell and run it.
# First: Runtime > Change runtime type > T4 GPU.

# ---- Cell 1: setup (run this first) -----------------------------------------
!pip -q install datasets

import tensorflow as tf
print("TensorFlow", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus if gpus else "NONE. Use Runtime > Change runtime type > T4 GPU, then run this cell again.")


# ---- Cell 2: load the dataset and prepare the images ------------------------
import gc, json
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
try:
    from tqdm.auto import tqdm
except ImportError:            # progress bar is optional
    tqdm = lambda it, **kwargs: it

# Settings
IMG_SIZE = 128       # the dataset's own size. The website reads the size from the model, so 96 or 128 both work.
MODEL_SIZE = "large" # "large" (BatchNorm, ~1.2M parameters, recommended) or "standard" (your original small CNN)
FLIP_AUGMENT = False # True only if you will sign with the OTHER hand than the people in the dataset
MAX_IMAGES = None    # None = use the whole dataset (about 50,000 images). Or a number, e.g. 20000
COLLECTED_REPEAT = 3 # frames you recorded in the website (Developer mode > Character tester > Save frames):
                     # how many times each is used in training. 0 = ignore them. See below.
COLLECTED_DIR = None # a folder of recorded frames already in this runtime; None = you are asked to upload a .zip
BATCH_SIZE = 64
EPOCHS = 40          # early stopping normally ends training sooner
SEED = 42

ds = load_dataset("VinayHajare/Marathi-Sign-Language")["train"]
labels = ds.features["label"].names
NUM_CLASSES = len(labels)
print("Images in dataset:", len(ds), "| classes:", NUM_CLASSES)

# The dataset is SORTED by class. Your old notebook took the first 8000 rows, so it only saw
# 8 letters. If a subset is requested, shuffle BEFORE selecting.
if MAX_IMAGES is not None and MAX_IMAGES < len(ds):
    ds = ds.shuffle(seed=SEED).select(range(MAX_IMAGES))

# uint8 arrays use 4x less memory than float32; they are scaled to 0..1 per batch below.
X = np.empty((len(ds), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
y = np.empty(len(ds), dtype=np.int64)
for i, row in enumerate(tqdm(ds, desc="Preparing images")):
    # same preprocessing as the website: RGB, resize to IMG_SIZE x IMG_SIZE (the /255 comes later)
    X[i] = np.asarray(row["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    y[i] = row["label"]

counts = np.bincount(y, minlength=NUM_CLASSES)
print(f"Classes present: {(counts > 0).sum()} of {NUM_CLASSES} | images per class: min {counts.min()}, max {counts.max()}")
assert (counts > 0).all(), "Some classes are missing. Set MAX_IMAGES = None."

# 70 / 15 / 15 split, stratified so every letter appears in every part.
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=SEED, stratify=y_trainval)
del X, y, X_trainval, y_trainval
gc.collect()
print("Train:", len(X_train), "| validation:", len(X_val), "| test:", len(X_test))

# ---- YOUR OWN WEBCAM FRAMES (the most effective fix for "works on the dataset, fails on my camera") ----------
# The dataset shows centred close-ups; your camera shows a room. Frames recorded in the website's Developer mode
# (set "train") are stored with the hand box, so we cut out exactly the three crops the website feeds the model.
# They are added to the TRAINING data only. Measure the result on a different recording (set "test") with
# python tools/evaluate_model.py --no-dataset --collected data/collected/test
import glob, os, zipfile
from PIL import Image

def find_collected():
    if COLLECTED_REPEAT <= 0:
        return None
    if COLLECTED_DIR:
        return COLLECTED_DIR
    try:
        from google.colab import files
    except ImportError:
        return None
    print("\nOptional: upload collected_train.zip (zip of data/collected/train). Cancel to train on the dataset only.")
    uploaded = files.upload()
    for name in uploaded:
        if name.lower().endswith(".zip"):
            zipfile.ZipFile(name).extractall("collected")
            return "collected"
    return None

def hand_crop(image, bbox, margin):      # identical to crop_to_hand in the website's backend/recognition.py
    width, height = image.size
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = (x0 + x1) / 2 * width, (y0 + y1) / 2 * height
    side = max((x1 - x0) * width, (y1 - y0) * height) * (1 + 2 * margin)
    size = int(round(min(max(side, 64.0), float(min(width, height)))))
    left = int(round(min(max(cx - size / 2, 0.0), width - size)))
    top = int(round(min(max(cy - size / 2, 0.0), height - size)))
    return image.crop((left, top, left + size, top + size))

collected_root = find_collected()
if collected_root:
    X_extra, y_extra = [], []
    for jpg in sorted(glob.glob(os.path.join(collected_root, "**", "*.jpg"), recursive=True)):
        meta_path = jpg[:-4] + ".json"
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        idx = int(meta["index"])
        assert labels[idx] == meta["label"], f"Class order changed since recording: {meta['label']} vs {labels[idx]}"
        frame = Image.open(jpg).convert("RGB")
        for margin in (0.15, 0.35, 0.60):
            crop = hand_crop(frame, meta["bbox"], margin).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X_extra.append(np.asarray(crop, dtype=np.uint8))
            y_extra.append(idx)
    if X_extra:
        X_extra, y_extra = np.stack(X_extra), np.array(y_extra, dtype=np.int64)
        seen = np.bincount(y_extra, minlength=NUM_CLASSES)
        print(f"Your webcam frames: {len(X_extra)} crops for {(seen > 0).sum()} of {NUM_CLASSES} letters "
              f"(min {seen[seen > 0].min()}, max {seen.max()} per letter), each used {COLLECTED_REPEAT}x")
        X_train = np.concatenate([X_train] + [X_extra] * COLLECTED_REPEAT)
        y_train = np.concatenate([y_train] + [y_extra] * COLLECTED_REPEAT)
        print("Train is now:", len(X_train))
    else:
        print("No usable frames found (each .jpg needs its .json). Training on the dataset only.")


# ---- Cell 3: build the model and train ---------------------------------------
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Augmentation that imitates the gap between the dataset (centred close-ups) and a webcam:
# the hand smaller or off-centre, slightly rotated, different light. This is what most helps live use.
aug_layers = [
    layers.RandomRotation(0.05),                                        # about +/- 18 degrees
    layers.RandomZoom(height_factor=(-0.15, 0.40), width_factor=(-0.15, 0.40)),   # hand up to 40% smaller in the frame
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25, value_range=(0.0, 1.0)),
]
if FLIP_AUGMENT:
    aug_layers.insert(0, layers.RandomFlip("horizontal"))
augment = tf.keras.Sequential(aug_layers)

def to_float(x, label):
    return tf.cast(x, tf.float32) / 255.0, label      # same scaling as the website

def make_train_data():
    return (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(5000, seed=SEED)
            .batch(BATCH_SIZE)
            .map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x, label: (tf.clip_by_value(augment(x, training=True), 0.0, 1.0), label),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

def make_eval_data(X_, y_):
    return (tf.data.Dataset.from_tensor_slices((X_, y_))
            .batch(BATCH_SIZE)
            .map(to_float, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

train_data = make_train_data()
val_data = make_eval_data(X_val, y_val)
test_data = make_eval_data(X_test, y_test)

def conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)

if MODEL_SIZE == "large":
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    for filters in (32, 64, 128, 256):
        x = conv_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
else:  # your original notebook's CNN
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),  # your old cell never passed this
    ],
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["accuracy"], label="train"); ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train"); ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.show()


# ---- Cell 4: test, save and download -------------------------------------------
from sklearn.metrics import classification_report, confusion_matrix

test_loss, test_acc = model.evaluate(test_data, verbose=0)
print(f"\nTEST ACCURACY: {test_acc * 100:.2f} %   (images the model never saw or tuned on)")

y_pred = np.concatenate([np.argmax(model.predict(xb, verbose=0), axis=1) for xb, _ in test_data])
print(classification_report(y_test, y_pred, labels=np.arange(NUM_CLASSES),
                            target_names=labels, zero_division=0, digits=3))

# Which letters does the model mix up?  These are the ones to watch in the live demo.
cm = confusion_matrix(y_test, y_pred, labels=np.arange(NUM_CLASSES))
np.fill_diagonal(cm, 0)
worst = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:10], cm.shape))[0]
print("Most confused pairs (true -> predicted, how many test images):")
for t, p in worst:
    if cm[t, p] > 0:
        print(f"  {labels[t]} -> {labels[p]}   {cm[t, p]}")

# Save, then reload the saved file to prove it works exactly as the website will load it.
model.save("marathi_sign_model.keras")
reloaded = tf.keras.models.load_model("marathi_sign_model.keras")
assert reloaded.input_shape == (None, IMG_SIZE, IMG_SIZE, 3) and reloaded.output_shape == (None, NUM_CLASSES)
sample = X_test[:64].astype("float32") / 255.0
assert np.allclose(model.predict(sample, verbose=0), reloaded.predict(sample, verbose=0), atol=1e-5)
print("\nSaved marathi_sign_model.keras and reloaded it successfully:", reloaded.input_shape, "->", reloaded.output_shape)

# The website needs the class names in the SAME ORDER as the model's outputs.
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)
print("First labels:", labels[:8], "... total", len(labels))

try:
    from google.colab import files
    files.download("marathi_sign_model.keras")
    files.download("labels.json")
except ImportError:
    print("Not running in Colab: the files are in the current folder.")
