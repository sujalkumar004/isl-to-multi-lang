# 🤟 Indian Sign Language Converter

**Real-Time ISL → Text → Multilingual Translation**

A website that reads hand signs from the camera with **your trained TensorFlow CNN**, adds each
recognized character to the text once (no duplicates while a sign is held), translates the finished
text into English, Hindi and Marathi, and can speak it aloud. There is no simulated recognition
anywhere: if the model is missing, the page says so and recognition stays off.

---

## Read this first: what your real model does

Findings from inspecting your uploaded `marathi_sign_model.keras` (input 96x96x3, 43 outputs, 115,307 parameters,
the notebook's small CNN) and `labels.json` (identical to `model/labels.json`):

1. **It is not the original 8-class model.** It outputs many different letters, so it was retrained on (most of) the
   43 classes. The notebook's "only 8 letters were trained" problem does not apply to this file.
2. **Labels and preprocessing code are correct.** 43 unique labels in the model's order, no encoding problems;
   RGB -> resize -> /255 is identical to training (maximum difference 0.0).
3. **The real problem is that the model is confidently inconsistent.** It gives 90 %+ confidence to pure noise,
   colour blobs and room crops (noise: always ऐ; blobs: ज्ञ 276 of 300 times, at 96 %), and on your two real
   webcam captures from the notebook it answered differently depending only on how the frame was cropped:

   | Same frame | Whole frame | Hand crop | Three hand crops (each shown) | Averaged |
   |---|---|---|---|---|
   | Capture A | श 98 % | ण 100 % | ण 100 %, ण 100 %, ज्ञ 100 % | ण 67 % |
   | Capture B | फ 99 % | ल 100 % | ण 100 %, ल 100 %, ज्ञ 99 % | ल 33 % |

   (Measured in the real website with your real model. I do not know which sign you were making, so I cannot say which
   answer is right, but contradictory 100 % answers prove that some are wrong and that confidence alone cannot be
   trusted for this model. Only two unlabeled frames, with landmark drawings on them: an illustration, not a benchmark.)
4. **Why it happens:** the training pictures are centred close-ups on a plain background. A webcam frame is a whole
   room with the hand at 15 % of the width. The model never saw that, so it has no idea it is out of its depth, and
   softmax always produces some confident-looking letter.

What that changed in the website:

* **Default preprocessing is now the three-crop view around the hand** (it was the whole frame), because the whole
  frame is what the model is worst at. You can still choose the whole frame under Recognition settings.
* **A letter is accepted only if the crops agree** (2 of 3 by default). Contradictory crops now give "no letter"
  instead of a confident wrong one. This makes the site *quieter*: it will refuse more. That is the honest price of
  a model that cannot tell when it is wrong.
* **The lasting fix is training on your own camera.** The Developer mode can record labelled frames from your webcam,
  the evaluation tool scores the model on them (real per-letter accuracy without needing the dataset), and the Colab
  notebook trains on them. See "Finding out which characters work".

---

## Project structure

```text
isl-converter/
├── backend/
│   ├── app.py              Flask server: website + API, sessions, wiring
│   ├── recognition.py      Continuous recognition (refactored notebook logic + state machine)
│   ├── preprocessing.py    Camera frame -> what the CNN sees (whole frame / hand crop / multi-crop / mirror)
│   ├── diagnostics.py      Label checks, per-character metrics, calibrated thresholds
│   ├── model_service.py    Loads the .keras model once, in the background; serialises predictions
│   ├── translation.py      Translation of the complete text (deep-translator, source = Marathi)
│   └── requirements.txt
├── model/
│   ├── marathi_sign_model.keras   <- YOU PLACE YOUR MODEL HERE
│   ├── labels.json                The 43 class names in model order
│   ├── model_report.json          (created by tools/evaluate_model.py) per-character results
│   └── reference/                 (created by tools/evaluate_model.py) dataset sample pictures
├── frontend/
│   ├── index.html  style.css  script.js
│   └── developer.js               Developer mode: diagnostics, character grid, tester
├── tools/
│   ├── evaluate_model.py          Measure every character with YOUR model and the dataset
│   └── requirements-tools.txt
├── data/collected/                (created when you record frames) <set>/<character number>/*.jpg + *.json
├── training/
│   ├── train_marathi_sign_colab.ipynb   Upload to Google Colab (recommended)
│   └── train_marathi_sign_colab.py      Same code as one pasteable cell
├── tests/
└── README.md
```

The frontend is plain HTML/CSS/JavaScript served by the Flask server. There is **no separate frontend
server and no build step**.

---

## Install

Python 3.10 to 3.12 is recommended (TensorFlow must support your version).

```bash
cd isl-converter
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Place the model

Copy the file your notebook saved, **`marathi_sign_model.keras`**, into the **`model/`** folder:

```text
isl-converter/model/marathi_sign_model.keras
```

(To keep it elsewhere, set the environment variable `ISL_MODEL_PATH` to its full path.) It must take
96 × 96 × 3 input and output 43 classes; the server checks this and tells you if it does not fit.

## Run

```bash
python backend/app.py
```

Open **http://127.0.0.1:5000** in Chrome or Edge. That single command runs both the backend (API and
model) and the frontend (the website). The model loads in the background: the page shows *Loading
recognition model...* until the Model status says *Loaded*.

The camera works only on `localhost` / `127.0.0.1` or over HTTPS. MediaPipe (hand detection) and the
fonts load from a CDN, and translation calls an online service, so the machine needs internet.

For anything beyond a local demo: `pip install gunicorn`, then
`gunicorn --chdir backend -w 1 --threads 4 app:app` (one worker, because each browser tab's text is
kept in memory and the model is loaded once).

---

## How to test

### 1. Camera recognition
1. Open the site. Nothing asks for the camera yet. The status panel shows *Camera: Off*, *Model: Loaded*.
2. Click **Start Camera**, allow the permission. The status becomes *Camera: Active*, the hand overlay
   appears when your hand is in view, and *Hand: Detected* turns on.
3. Show a sign. **Current Sign**, **Confidence** (number and bar) and **Recognition Status** update live.
   When a letter is accepted it appears in **Recognized Text** and **Recent Recognitions**.

### 2. Continuous recognition: "HELLO"
With the current Marathi model, test the same behaviour with letters it knows.

| You do | Result |
|---|---|
| Show आ and hold it for 3+ seconds | `आ` (one letter, never `आआआआ`); status *Stable, locked* |
| Change to ई | `आई` |
| Lower your hand for about half a second, then show ई again | `आईई` (a repeated letter needs a release) |
| Press **Space**, sign more letters | a new word starts |

The exact **H → E → L → L → O** sequence is verified by an automated test that feeds the recognizer
frame-by-frame (with a stand-in vocabulary, because the model has no H/E/L/O classes):

```bash
python -m unittest tests.test_recognition -v -k H_E_L_L_O
```

The `HELLO` test lowers the hand between the two L's. `-k holding` shows that holding one sign for 3 s or
a whole minute gives a single character.

### 3. SPACE, BACKSPACE, CLEAR
Click the buttons under **Recognized Text**, or press **Space**, **Backspace**, **Shift + Delete**.
Space never creates a leading or double space. Backspace removes one whole sign (a conjunct such as क्ष is
one unit). Clear also empties Recent Recognitions.

### 4. Pause / resume / stop
* **Pause Recognition**: the camera stays visible, no frames are sent, the text is kept.
* **Start Recognition**: continues; the sign is read afresh (old frames and the lock are forgotten).
* **Stop Camera**: releases the camera; the text stays.

### 5. Hindi translation
Sign some letters (for example आई), choose **हिंदी**. After a moment the **Hindi** row in
*Multilingual Translation* shows the translation (आई should give a word meaning "mother"). The whole
text is translated at once. Or test the module directly:

```bash
python -c "import sys; sys.path.insert(0,'backend'); import translation as t; print(t.translate_text('आई','hi'))"
```

### 6. Marathi translation
Choose **मराठी**. Marathi is the recognized text itself (source language), so it appears instantly and
does not need the internet. English and Hindi are translated *from* Marathi.

### 7. Text-to-Speech
Choose a language and click **Speak** (click again to stop). The site speaks the selected row using
`en-IN`, `hi-IN` or `mr-IN`. If the device has no voice for the language you get a message. For Marathi,
if only a Hindi voice exists, it reads the Marathi text with the Hindi voice and says so. Voices depend on
the browser and operating system.

### 8. Errors (all handled, the page never crashes)
| Situation | What you see |
|---|---|
| Permission denied | *Camera permission denied.* and *Camera permission is required for sign recognition. Please allow camera access and try again.* |
| No / busy camera | *Camera unavailable. Please check your camera connection.* |
| Model still loading | *Loading recognition model...* |
| No hand in view | *No hand detected* |
| Steady sign, low confidence | *Waiting for a clearer sign...* |
| Server stopped / network error | *Recognition service unavailable. Please try again.* (recovers by itself when the server is back) |

---

## Finding out which characters work (and fixing the ones that do not)

Work through this in order. Each step tells you something the previous one cannot.

**Step 1: record your own webcam frames (recommended; needs no dataset and no internet).**
Open **Developer mode** (toggle in the header), start the camera, select a character in the grid, tick
**Save the frames of this test**, set the name to `train`, press **Start test**, hold the sign about 5 seconds while
moving the hand a little (closer, farther, slightly rotated, different light), press **Stop**. Repeat for all 43.
Then use a different set name, `test`, and record every character again on another day or with a different background.
The frames go to `data/collected/<set>/<character number>/` on this computer. They include your face and room;
nothing is uploaded anywhere unless you upload the zip yourself in step 4.

**Step 2: measure the model on those frames** (real ground truth: the character you were signing):

```powershell
python tools\evaluate_model.py --no-dataset --collected data\collected\test
```

It scores four preprocessing modes (whole frame, hand crop, three crops, three crops mirrored), tells you which is
best on YOUR camera, prints recall per character and what it is mistaken for, and shows what share of frames the
website's accept rules would let through and how many of those are wrong. Results appear in Developer mode
(Reload report). Record and test on separate sets: measuring on the frames you trained on is too optimistic.

**Step 3 (optional, needs internet once): measure on the dataset.**

```powershell
python -m pip install -r tools\requirements-tools.txt
python tools\evaluate_model.py            # a model trained with the Colab notebook (held-out test images)
python tools\evaluate_model.py --split all   # a model trained with the ORIGINAL notebook
python tools\evaluate_model.py --robustness  # also mirrored / smaller / rotated / dark / blurred pictures
```

It lists characters the model **never predicts**, checks `labels.json` against the dataset's class order, finds
characters whose training pictures already look alike, and calibrates a confidence level per character. Running
both step 2 and step 3 keeps both results in `model/model_report.json`.

**Developer mode panels**

| Panel | What it answers |
|---|---|
| Live diagnostics | Top prediction, confidence, alternatives, gap to the runner-up, whether the hand crops agree, stable frames, state, what is still needed to accept, hand size, frames skipped, picture brightness |
| What the model sees | The exact 96/128 px picture the CNN receives, next to dataset pictures of the selected character |
| Same frame, different preprocessing | Whole frame / hand crop / three crops / mirrored, each scored on the same frame |
| Supported Marathi characters | All 43, colour-coded: dataset result, your recorded-webcam result and your live test result |
| Character tester | Select a character, make the sign, Start test: how often the model's top guess was that character, what it was mistaken for, which preprocessing works best |

**Step 4: retrain on your own camera** (the reliable fix for "works on the dataset, fails on my camera").
Zip `data/collected/train` as `collected_train.zip`, open `training/train_marathi_sign_colab.ipynb` in Google Colab
with a T4 GPU, **Run all**, upload the zip when Cell 2 asks (Cancel to skip), then copy the downloaded
`marathi_sign_model.keras` and `labels.json` into `model/`. The notebook cuts out exactly the three crops the website
feeds the model, adds them to the training data only, shuffles before sampling (all 43 letters seen), splits 70/15/15,
uses early stopping and the learning-rate callback, and augments with smaller / shifted / rotated hands and changed
light. It refuses recordings made with a different class list. Then repeat step 2 on the `test` set and compare.
Aim for at least 100 frames per letter across different distances, angles and backgrounds.

**Step 5: settings that help without retraining** (Recognition settings):

* *What the model sees*: three crops (default), one hand crop, or the whole frame. Step 2 shows which is best for you.
* *Hand crops that must agree*: raise it to be stricter, lower it (or 0) to accept more.
* *Mirror the image*: try it if signs made with one hand work and the other fails; the mirrored row in step 2 shows it.
* *Minimum gap to the runner-up* rejects coin flips (क 45 % vs ख 44 %).
* *Calibrated per-character thresholds* (used when `model_report.json` exists, webcam-measured ones win) accept a
  letter below the global threshold only if held-out data showed it at least 95 % precise there, never below 40 %.
* *Accept very clear signs after 3 frames* helps fast spelling.
* Plain background, light in front of you, hand large and centred, like the dataset pictures.

Continuous recognition is unchanged in spirit: a character is added once, then locked; holding a sign never repeats
it. Nothing here lowers the confidence threshold drastically, and nothing here invents a prediction.

**Characters that stay unreliable after retraining** are usually one of three things, which the tools tell apart:
two letters' training pictures already look alike (`Look-alike training pictures`), the sign involves movement or a
second hand that a still picture cannot capture, or there is too little variety for that letter. More recorded frames
for those letters is the fix.

---

## How it works

```text
Browser camera ─▶ MediaPipe: is a hand in view? ─▶ JPEG frame (max ~10/s) + hand flag
        │                                                   │  POST /api/frame
        │                                                   ▼
        │                        model_service.py: your CNN (loaded once)  ─▶  probabilities
        │                                                   ▼
        │                        recognition.py: ContinuousRecognizer
        │                        rolling window ─▶ confidence + frame agreement ─▶ accept once ─▶ lock
        ▼                                                   │
   UI updates  ◀──────── snapshot: state, sign, confidence, agreement, text ┘
        │
        └─▶ POST /api/translate (complete text) ─▶ English / Hindi / Marathi ─▶ Speak (Web Speech API)
```

**Accept once, lock, release.** A character is added when the last 6 frames agree (5 of 6), the average
confidence reaches the threshold (70%), and a hand is in view. It is then locked: holding the sign adds
nothing more. The lock is released when a different sign becomes stable, when the locked sign has not been
the top prediction for 8 frames in a row, or when the hand leaves the camera for 0.3 s. Releasing also
clears the frame window so old frames cannot retrigger the letter.

The values are reasoned defaults, not tuned to your model. Adjust them live under **Recognition
settings** (confidence threshold, frames, agreement, minimum gap to the runner-up, release frames, hand gate,
preprocessing view, mirror, calibrated thresholds, fast accept, optional auto-space).

**Performance.** At most ~10 frames per second are sent, frames without a hand are not uploaded or
predicted, the model is loaded once, and the tab pauses when hidden.

### API
| Endpoint | Purpose |
|---|---|
| `GET /api/info` | Model state (`loading` / `ready` / `error`), labels, default settings |
| `POST /api/frame` | multipart: `session`, `hand` (1/0/unknown), `frame` (JPEG; optional when `hand=0`), `bbox` (hand box), `debug` (1 = add probabilities, model input picture, preprocessing comparison) |
| `POST /api/frame` with `collect=<character number>` and `collect_set=<name>` | also stores the frame and hand box under `data/collected/` (only when a hand box exists) |
| `GET /api/collected` | Number of stored frames per set and character |
| `GET /api/diagnostics` | Label check, model structure, offline report and per-character status |
| `GET /reference/<file>` | Dataset sample pictures exported by `tools/evaluate_model.py` |
| `POST /api/space`, `/api/backspace`, `/api/clear` | Manual keys (JSON `{session}`) |
| `POST /api/resume` | Forget old frames and the lock, keep the text |
| `POST /api/config` | Update settings (values are clamped) |
| `POST /api/translate` | `{text, targets:["en","hi","mr"]}` → per-language `{text, error}` |

Notebook-only code is gone: no `google.colab`, `cv2_imshow`, `input()` or Colab JavaScript.
Preprocessing (96×96 RGB, /255), the labels, the confidence threshold, frame agreement, `text_buffer`
and the translation call are the notebook's; `predict_single_frame` now calls the model directly instead
of `model.predict()`, and backspace works on whole signs.

---

## Automated tests

```bash
python -m unittest tests.test_recognition -v     # 60 logic tests (no TensorFlow needed)
python -m unittest tests.test_diagnostics -v     # 18 tests: labels, metrics, calibration, preprocessing (no TensorFlow)
python -m unittest tests.test_translation -v     # 8 tests, translation service stubbed
python -m unittest tests.test_server -v          # 29 API tests, needs TensorFlow
python -m unittest tests.test_evaluate_tool -v   # 7 tests of tools/evaluate_model.py, needs TensorFlow + scikit-learn
```

Covered: no duplicates on long holds or flicker, HELLO and HELLO WORLD sequences, double letters, low-confidence
rejection, ambiguity (gap) rejection, crop-agreement rejection, calibrated thresholds that can only lower the bar
within limits, fast accept, hand gate, pause/resume, space / backspace / clear, label integrity checks, per-character
metrics, hand-crop / multi-crop / mirror preprocessing, model missing / non-square / wrong class count, 96 px and
128 px models, real JPEG -> Keras -> recognizer through HTTP, the debug payload, recording of frames (and refusal of
bad set names, missing hand boxes, bad class numbers), diagnostics with and without a report, and the evaluation
tool on a fake dataset and on recorded webcam frames.

**Not covered by automatic tests:** how accurate *your* model is on real frames. The tests use synthetic
probabilities, a test-only colour-classifier stand-in (`tests/make_test_model.py`, never used by the site) and fake
data. The real-model results in "Read this first" were measured separately with your uploaded model. Step 2 above
measures it properly on your camera.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Banner: *Model file not found* | Put `marathi_sign_model.keras` in `model/` (or set `ISL_MODEL_PATH`) and restart. |
| *The model has N outputs but labels.json lists 43* / *expects input …* | The model must be the notebook's 96×96×3 → 43-class CNN. |
| *TensorFlow is not installed* | Activate the virtual environment and run `pip install -r backend/requirements.txt`. |
| Camera never starts / permission denied | Use `http://127.0.0.1:5000` (not a LAN address). Click the camera icon in the address bar, allow, then Start Camera again. Close other apps using the camera. |
| Works on the laptop but not on a phone | Browsers only allow the camera on `localhost` or HTTPS. Serve the site over HTTPS (for example through a tunnel). |
| Hand status says *Unavailable* | MediaPipe could not load (no internet). Recognition still works, but lowering the hand no longer releases a letter; change the sign instead. |
| Letters are wrong or never accepted | Open **Developer mode** and follow "Finding out which characters work". First `python tools\evaluate_model.py`: characters listed as *never predicted* need retraining (`training/train_marathi_sign_colab.ipynb`). Lowering the threshold only accepts wrong letters more often. |
| Status stays "Stable" but nothing is added | The Developer panel's *Needed to accept* line shows which condition fails: confidence below the threshold, or the top two letters too close together. |
| Nothing is added; status says "Different crops of your hand give different answers" | The model disagrees with itself between crops of the same hand. Improve framing and light, lower *Hand crops that must agree* to try, and above all retrain with frames recorded on your camera (step 4). |
| Some characters only work with one hand | Try **Mirror the image** in Recognition settings, or check the *Same frame, different preprocessing* table. |
| Developer mode makes the video slower | It scores several picture versions per frame. Turn it off for normal use. |
| A repeated letter is missing | Release between identical letters: lower your hand out of view for about half a second. |
| Translation shows *unavailable* | It needs internet. Click **Retry** in the row. Marathi always works. |
| No sound / *No … voice installed* | Install a voice for that language in the system settings, or try Chrome or Edge. Marathi voices are rare on desktops. |
| Very slow | The first frames after start are slower while MediaPipe downloads. CPU-only TensorFlow handles this small CNN in about 10 ms per frame. |
