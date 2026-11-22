/* Indian Sign Language Converter: browser client.
 *
 *   camera -> (MediaPipe: is a hand in view?) -> JPEG frame + hand flag -> POST /api/frame
 *          <- snapshot (state, sign, confidence, text) <- Python ContinuousRecognizer + CNN
 *
 * Every recognition decision (confidence, frame agreement, lock / duplicate prevention) is made
 * on the server in backend/recognition.py. This file captures frames, sends them at a limited
 * rate, and displays whatever the server reports. It never invents a prediction.
 */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* ------------------------------------------------------------ constants */
  const MSG = {
    cameraUnavailable: "Camera unavailable. Please check your camera connection.",
    permissionDenied: "Camera permission denied.",
    permissionHelp:
      "Camera permission is required for sign recognition. Please allow camera access and try again.",
    loadingModel: "Loading recognition model...",
    noHand: "No hand detected",
    lowConfidence: "Waiting for a clearer sign...",
    service: "Recognition service unavailable. Please try again.",
    translation: "Translation is unavailable right now. Check the internet connection.",
  };

  const FRAME_INTERVAL_MS = 100;     // at most ~10 frames per second are sent for recognition
  const MAX_UPLOAD_WIDTH = 720;      // the notebook captured 720x540 frames
  const HAND_FRESH_MS = 1500;        // hand status older than this counts as "unknown"
  const HAND_START_TIMEOUT_MS = 25000;
  const ACCEPT_FLASH_MS = 900;
  const TRANSLATE_DEBOUNCE_MS = 700;
  const HISTORY_MAX = 14;

  const LANGS = {
    en: { name: "English", speech: "en-IN" },
    hi: { name: "Hindi", speech: "hi-IN" },
    mr: { name: "Marathi", speech: "mr-IN" },
  };
  const STATE_INFO = {
    WAITING: { label: "Waiting", tone: "idle" },
    DETECTING: { label: "Detecting", tone: "info" },
    STABLE: { label: "Stable", tone: "warn" },
    ACCEPTED: { label: "Sign accepted", tone: "ok" },
    LOCKED: { label: "Stable, locked", tone: "ok" },
  };
  const SLIDERS = {
    confidence_threshold: (v) => Math.round(v * 100) + "%",
    window_size: (v) => v + " frames",
    min_agreement: (v) => Math.round(v * 100) + "%",
    release_frames: (v) => v + " frames",
    auto_space_seconds: (v) => v + " s",
    min_margin: (v) => Math.round(v * 100) + "%",
    min_view_agreement: (v) => Math.round(v * 100) + "%",
  };
  const CHECKS = ["require_hand", "auto_space", "mirror_input", "adaptive_threshold", "fast_accept"];
  const SELECTS = ["view_mode"];

  /* ------------------------------------------------------------ state */
  const S = {
    session: window.crypto && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2) + Date.now().toString(36),
    config: { require_hand: true, view_mode: "full", window_size: 6, confidence_threshold: 0.7 },
    collect: null,               // {index, set}: also store the frames the model is given (Developer mode tester)
    collectTick: 0,
    debug: false,                // Developer mode: ask the server for the full diagnostics with every frame
    model: { state: "loading", error: null, classes: 43, input_size: 96 },
    modelReachable: true,
    cam: "off",                  // off | starting | on | denied | unavailable
    streaming: false,
    recognizing: false,
    stream: null,
    startToken: 0,
    looping: false,
    serviceError: false,
    holistic: null,
    holisticState: "idle",       // idle | loading | ready | unavailable
    holisticRunning: false,
    hand: null,                  // { present, at, bbox }
    snap: null,
    lang: "en",
    flashUntil: 0,
    flashTimer: null,
    tokensKey: "",
    tokensCount: 0,
    history: [],
    tr: { id: 0, timer: null, doneFor: null, results: {} },
    speaking: false,
    pushTimer: null,
    pendingPatch: null,
    fps: 0,
    lastFrameAt: 0,
  };

  const video = $("video");
  const overlay = $("overlay");
  const grab = document.createElement("canvas");

  /* ------------------------------------------------------------ small helpers */
  const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };
  const setTone = (el, tone) => { if (el.dataset.tone !== tone) el.dataset.tone = tone; };

  function showBanner(message, kind) {
    const el = $("banner");
    el.textContent = message;
    el.dataset.kind = kind || "error";
    el.hidden = false;
  }
  function clearBanner(onlyIfText) {
    const el = $("banner");
    if (!onlyIfText || el.textContent === onlyIfText) el.hidden = true;
  }
  let toastTimer = null;
  function toast(message) {
    const el = $("toast");
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.hidden = true), 5000);
  }
  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.assign({ session: S.session }, body || {})),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || "HTTP " + res.status);
      err.status = res.status;
      throw err;
    }
    return data;
  }
  const getInfo = async () => {
    const res = await fetch("/api/info");
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  };

  /* ------------------------------------------------------------ derived view + painting */
  const flashActive = () => performance.now() < S.flashUntil;

  function recognitionView() {
    if (S.cam !== "on") return { tone: "idle", label: "Idle" };
    if (!S.recognizing) return { tone: "warn", label: "Paused" };
    if (S.serviceError) return { tone: "err", label: "Service unavailable" };
    if (S.model.state === "loading") return { tone: "warn", label: "Loading model" };
    const r = S.snap;
    if (flashActive()) return STATE_INFO.ACCEPTED;
    return (r && STATE_INFO[r.state]) || { tone: "info", label: "Starting…" };
  }

  function handView() {
    if (S.cam !== "on") return { tone: "idle", label: "–" };
    if (S.holisticState === "loading" || S.holisticState === "idle") return { tone: "warn", label: "Starting…" };
    if (S.holisticState === "unavailable") return { tone: "warn", label: "Unavailable" };
    if (!S.hand) return { tone: "idle", label: "Checking…" };
    return S.hand.present ? { tone: "ok", label: "Detected" } : { tone: "warn", label: "Not detected" };
  }

  function modelView() {
    if (!S.modelReachable) return { tone: "err", label: "Unreachable" };
    if (S.model.state === "ready") return { tone: "ok", label: "Loaded" };
    if (S.model.state === "error") return { tone: "err", label: "Not loaded" };
    return { tone: "warn", label: "Loading…" };
  }

  function cameraView() {
    return {
      off: { tone: "idle", label: "Off" },
      starting: { tone: "warn", label: "Starting…" },
      on: { tone: "ok", label: "Active" },
      denied: { tone: "err", label: "Permission denied" },
      unavailable: { tone: "err", label: "Unavailable" },
    }[S.cam];
  }

  function hintView() {
    const r = S.snap;
    if (S.cam === "denied") return { tone: "err", text: MSG.permissionDenied };
    if (S.cam === "unavailable") return { tone: "err", text: MSG.cameraUnavailable };
    if (S.cam === "off") return { tone: "idle", text: "Press Start Camera to begin." };
    if (S.cam === "starting") {
      return { tone: "warn", text: S.model.state === "loading" ? MSG.loadingModel : "Starting the camera…" };
    }
    if (!S.recognizing) return { tone: "warn", text: "Recognition is paused. Your text is kept." };
    if (S.serviceError) return { tone: "err", text: MSG.service };
    if (S.model.state === "loading") return { tone: "warn", text: MSG.loadingModel };
    if (S.holisticState === "loading" && S.config.require_hand) return { tone: "warn", text: "Starting hand detection…" };
    const gate = S.config.require_hand && S.hand && !S.hand.present && S.holisticState === "ready";
    if (gate) return { tone: "warn", text: MSG.noHand };
    if (!r) return { tone: "idle", text: "Show a sign to the camera." };
    if (r.state === "STABLE" || (r.window_fill >= r.window_size && r.confidence < r.threshold && r.state !== "LOCKED")) {
      return { tone: "warn", text: MSG.lowConfidence };
    }
    if (r.state === "ACCEPTED" || flashActive()) return { tone: "ok", text: "Added " + (r.accepted || r.locked_sign || "") + ". Change the sign to continue." };
    return { tone: "idle", text: r.message || "" };
  }

  function setRow(id, view) {
    const li = $("st-" + id);
    setTone(li.querySelector(".dot"), view.tone);
    setText(li.querySelector(".st-value"), view.label);
  }

  function paint() {
    const r = S.snap;
    const cam = cameraView();
    const rec = recognitionView();
    const hand = handView();
    const model = modelView();
    const hint = hintView();

    // camera chip + viewport
    setTone($("camChip"), cam.tone);
    setText($("camChipText"), S.cam === "denied" ? "Camera permission denied" : "Camera " + cam.label.toLowerCase());
    $("viewport").classList.toggle("live", S.streaming);
    const empty = {
      off: "The camera is off. Press Start Camera to begin.",
      starting: "Starting the camera…",
      denied: MSG.permissionDenied,
      unavailable: MSG.cameraUnavailable,
      on: "",
    }[S.cam];
    setText($("camEmptyText"), empty);
    setTone($("camEmpty"), cam.tone === "err" ? "err" : "idle");
    $("statePill").hidden = S.cam !== "on";
    setTone($("statePill"), rec.tone);
    setText($("statePill"), rec.label);
    $("pausedVeil").hidden = !(S.cam === "on" && !S.recognizing);

    // status panel
    setRow("camera", cam);
    setRow("model", model);
    setRow("hand", hand);
    setRow("recognition", rec);
    setRow("language", { tone: "ok", label: LANGS[S.lang].name });

    // metrics
    const live = S.cam === "on" && r && r.window_fill > 0;
    setText($("mSign"), live && r.current_sign ? r.current_sign : "–");
    setText($("mState"), rec.label);
    setTone($("mState"), rec.tone);
    setText($("mHand"), hand.label);
    setTone($("mHand"), hand.tone);

    // confidence bar (the real value from the model)
    const conf = live ? Math.min(1, r.confidence || 0) : 0;
    $("confFill").style.width = (conf * 100).toFixed(1) + "%";
    setText($("confValue"), live ? (conf * 100).toFixed(1) + "%" : "–");
    $("confBar").setAttribute("aria-valuenow", String(Math.round(conf * 100)));
    setTone($("confBar"), live ? (rec.tone === "idle" ? "info" : rec.tone) : "idle");
    $("confTick").style.left = ((r && r.threshold) || S.config.confidence_threshold) * 100 + "%";

    // hint
    setText($("hint"), hint.text);
    setTone($("hint"), hint.tone);

    // buttons
    $("btnStart").disabled = !["off", "denied", "unavailable"].includes(S.cam);
    $("btnStop").disabled = !["starting", "on"].includes(S.cam);
    $("btnRecStart").disabled = !(S.cam === "on" && !S.recognizing);
    $("btnRecPause").disabled = !(S.cam === "on" && S.recognizing);
  }

  const snapshotListeners = [];

  function applySnapshot(r) {
    S.snap = r;
    if (r.event === "accepted") {
      S.flashUntil = performance.now() + ACCEPT_FLASH_MS;
      clearTimeout(S.flashTimer);
      S.flashTimer = setTimeout(paint, ACCEPT_FLASH_MS + 40);
      addHistory(r.accepted);
    } else if (r.event === "clear") {
      S.history = [];
      renderHistory();
    }
    renderText(r.tokens || [], r.event === "accepted");
    paint();
    snapshotListeners.forEach((fn) => { try { fn(r); } catch (err) { console.warn("listener:", err); } });
  }

  /* ------------------------------------------------------------ recognized text + history */
  function currentText() {
    return (S.snap && S.snap.tokens ? S.snap.tokens.join("") : "").trim();
  }

  function renderText(tokens, fresh) {
    const key = tokens.join("\u0001");
    if (key === S.tokensKey) return;
    const grew = tokens.length > S.tokensCount;
    S.tokensKey = key;
    S.tokensCount = tokens.length;
    const nodes = tokens.map((t, i) => {
      const span = document.createElement("span");
      span.className = "ch";
      span.textContent = t;
      if (fresh && grew && i === tokens.length - 1 && t !== " ") span.classList.add("fresh");
      return span;
    });
    $("textOut").replaceChildren(...nodes);
    onTextChanged();
  }

  function addHistory(ch) {
    if (!ch) return;
    S.history.push(ch);
    if (S.history.length > HISTORY_MAX) S.history.shift();
    renderHistory();
  }
  function renderHistory() {
    const items = S.history.map((ch) => {
      const li = document.createElement("li");
      li.textContent = ch;
      return li;
    });
    $("history").replaceChildren(...items);
  }

  /* ------------------------------------------------------------ translation */
  function onTextChanged() {
    S.tr.doneFor = null;
    renderTranslations();
    clearTimeout(S.tr.timer);
    if (currentText()) S.tr.timer = setTimeout(translateNow, TRANSLATE_DEBOUNCE_MS);
  }

  async function translateNow() {
    const text = currentText();
    const id = ++S.tr.id;
    if (!text) return;
    try {
      const res = await postJSON("/api/translate", { text, targets: ["en", "hi", "mr"] });
      if (id !== S.tr.id) return;              // a newer request superseded this one
      S.tr.results = res.translations || {};
    } catch (err) {
      if (id !== S.tr.id) return;
      S.tr.results = { en: { error: MSG.translation }, hi: { error: MSG.translation }, mr: { text } };
    }
    S.tr.doneFor = text;
    renderTranslations();
  }

  function renderTranslations() {
    const text = currentText();
    const put = (id, value, state) => {
      const el = $(id);
      el.textContent = value;
      if (state) el.dataset.state = state; else delete el.dataset.state;
    };
    put("trSrc", text || "–");
    put("trMr", text || "–");
    for (const [code, id] of [["en", "trEn"], ["hi", "trHi"]]) {
      if (!text) { put(id, "–"); continue; }
      const res = S.tr.doneFor === text ? S.tr.results[code] : null;
      if (!res) put(id, "Translating…", "loading");
      else if (res.error) {
        put(id, res.error + " ", "error");
        const retry = document.createElement("button");
        retry.type = "button";
        retry.className = "linklike";
        retry.textContent = "Retry";
        retry.addEventListener("click", () => { S.tr.doneFor = null; renderTranslations(); translateNow(); });
        $(id).appendChild(retry);
      } else put(id, res.text || "–");
    }
    document.querySelectorAll(".tr-row[data-row]").forEach((row) => {
      if (row.dataset.row !== "src") row.dataset.selected = String(row.dataset.row === S.lang);
    });
  }

  function setLanguage(code) {
    S.lang = code;
    document.querySelectorAll(".lang-btn").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === code)));
    renderTranslations();
    paint();
  }

  /* ------------------------------------------------------------ text to speech */
  function loadVoices() {
    return new Promise((resolve) => {
      const synth = window.speechSynthesis;
      if (!synth) return resolve([]);
      const now = synth.getVoices();
      if (now.length) return resolve(now);
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(synth.getVoices()); } };
      if (synth.addEventListener) synth.addEventListener("voiceschanged", finish, { once: true });
      setTimeout(finish, 1200);
    });
  }

  function findVoice(voices, tag) {
    const norm = (s) => String(s || "").toLowerCase().replace("_", "-");
    const want = norm(tag);
    const primary = want.split("-")[0];
    return (
      voices.find((v) => norm(v.lang) === want) ||
      voices.find((v) => norm(v.lang).split("-")[0] === primary) ||
      null
    );
  }

  function outputForSpeech() {
    const text = currentText();
    if (!text) return { error: "Nothing to speak yet. Sign something first." };
    if (S.lang === "mr") return { text };
    const res = S.tr.doneFor === text ? S.tr.results[S.lang] : null;
    if (!res) return { error: "The translation is still loading. Try again in a moment." };
    if (res.error || !res.text) return { error: "There is no " + LANGS[S.lang].name + " translation to speak. " + MSG.translation };
    return { text: res.text };
  }

  async function speak() {
    const synth = window.speechSynthesis;
    if (!synth || !window.SpeechSynthesisUtterance) {
      return toast("Text-to-speech is not supported in this browser. Try Chrome, Edge or Safari.");
    }
    if (S.speaking) { synth.cancel(); return; }          // the button doubles as a stop button

    const out = outputForSpeech();
    if (out.error) return toast(out.error);

    const lang = LANGS[S.lang].speech;
    const voices = await loadVoices();
    let voice = findVoice(voices, lang);
    if (!voice && voices.length) {
      if (S.lang === "mr") {
        voice = findVoice(voices, "hi-IN");
        if (voice) toast("No Marathi voice is installed, so the Hindi voice reads the Marathi text.");
      }
      if (!voice) {
        return toast(
          "No " + LANGS[S.lang].name + " voice is installed on this device. Choose another language, or add a voice in your system settings."
        );
      }
    }
    const u = new SpeechSynthesisUtterance(out.text);
    u.lang = voice ? voice.lang : lang;
    if (voice) u.voice = voice;
    const setSpeaking = (on) => {
      S.speaking = on;
      const btn = $("btnSpeak");
      btn.setAttribute("aria-pressed", String(on));
      setText(btn.querySelector("span"), on ? "Stop" : "Speak");
    };
    u.onstart = () => setSpeaking(true);
    u.onend = () => setSpeaking(false);
    u.onerror = (e) => {
      setSpeaking(false);
      if (e && e.error && !["canceled", "interrupted"].includes(e.error)) toast("Speech could not be played (" + e.error + ").");
    };
    synth.cancel();
    synth.speak(u);
  }

  /* ------------------------------------------------------------ MediaPipe hand detection */
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = resolve;
      s.onerror = () => reject(new Error("could not load " + src));
      document.head.appendChild(s);
    });
  }

  function handDetectorFailed() {
    S.holistic = null;
    S.holisticState = "unavailable";
    paint();
    toast(
      "Hand detection could not load (it needs internet). Recognition still runs, " +
        "but lowering your hand will not release a letter; change the sign instead."
    );
  }

  async function loadHandDetector() {
    if (S.holistic) { holisticLoop(); return; }
    S.holisticState = "loading";
    paint();
    try {
      if (!window.Holistic) await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js");
      if (!window.drawConnectors) await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js");
      // The notebook's detector. Lighter settings: only hand presence is needed here.
      const h = new window.Holistic({ locateFile: (f) => "https://cdn.jsdelivr.net/npm/@mediapipe/holistic/" + f });
      h.setOptions({
        modelComplexity: 0,
        smoothLandmarks: true,
        enableSegmentation: false,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      h.onResults(onHolisticResults);
      S.holistic = h;
      holisticLoop();
      setTimeout(() => { if (S.holisticState === "loading") handDetectorFailed(); }, HAND_START_TIMEOUT_MS);
    } catch (err) {
      handDetectorFailed();
    }
  }

  async function holisticLoop() {
    if (S.holisticRunning) return;
    S.holisticRunning = true;
    while (S.cam === "on" && S.holistic) {
      try {
        if (video.readyState >= 2) await S.holistic.send({ image: video });
      } catch (err) {
        console.warn("hand detector:", err);
        await sleep(300);
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
    S.holisticRunning = false;
  }

  function bboxOf(hands) {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const lm of hands) for (const p of lm) {
      x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
      x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
    }
    const c = (v) => Math.min(1, Math.max(0, v));
    return [c(x0), c(y0), c(x1), c(y1)].map((v) => +v.toFixed(4));
  }

  function onHolisticResults(res) {
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const hands = [res.leftHandLandmarks, res.rightHandLandmarks].filter(Boolean);
    if (window.drawConnectors && window.HAND_CONNECTIONS) {
      for (const lm of hands) {
        window.drawConnectors(ctx, lm, window.HAND_CONNECTIONS, { color: "#ffffff", lineWidth: 3 });
        window.drawLandmarks(ctx, lm, { color: "#f0b232", lineWidth: 1, radius: 4 });
      }
    }
    const before = S.hand && S.hand.present;
    S.hand = { present: hands.length > 0, at: performance.now(), bbox: hands.length ? bboxOf(hands) : null };
    if (S.holisticState !== "ready") { S.holisticState = "ready"; paint(); }
    else if (before !== S.hand.present) paint();
  }

  function handFlag() {
    if (!S.holistic || S.holisticState !== "ready" || !S.hand) return null;   // unknown: server ignores the gate
    if (performance.now() - S.hand.at > HAND_FRESH_MS) return null;
    return S.hand.present;
  }

  /* ------------------------------------------------------------ camera */
  function setCam(state) {
    S.cam = state;
    paint();
  }

  function teardown() {
    S.startToken += 1;
    S.streaming = false;
    S.recognizing = false;
    if (S.stream) S.stream.getTracks().forEach((t) => { t.onended = null; t.stop(); });
    S.stream = null;
    video.srcObject = null;
    overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
    S.hand = null;
    if (S.snap) S.snap = Object.assign({}, S.snap, { current_sign: null, confidence: 0, window_fill: 0, window_chars: [], state: "WAITING", event: null });
  }

  function stopCamera() {
    teardown();
    setCam("off");
    clearBanner(MSG.service);
  }

  function cameraFailed(err) {
    teardown();
    const name = err && err.name;
    if (["NotAllowedError", "SecurityError", "PermissionDeniedError"].includes(name)) {
      setCam("denied");
      showBanner(MSG.permissionHelp);
    } else {
      setCam("unavailable");
      showBanner(MSG.cameraUnavailable);
    }
  }

  async function ensureModelReady(token) {
    for (;;) {
      let info;
      try {
        info = await getInfo();
        S.modelReachable = true;
      } catch (err) {
        S.modelReachable = false;
        showBanner(MSG.service);
        paint();
        return false;
      }
      S.model = info.model;
      clearBanner(MSG.service);
      paint();
      if (info.model.state === "ready") return true;
      if (info.model.state === "error") {
        showBanner(info.model.error || "The recognition model could not be loaded.");
        return false;
      }
      await sleep(1000);
      if (token !== S.startToken) return false;
    }
  }

  async function startCamera() {
    if (!["off", "denied", "unavailable"].includes(S.cam)) return;
    clearBanner();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCam("unavailable");
      showBanner("This page cannot use the camera. Open it at http://127.0.0.1:5000 or over HTTPS. " + MSG.cameraUnavailable);
      return;
    }
    const token = ++S.startToken;
    setCam("starting");

    // 1. camera permission + initialisation
    try {
      S.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 720, height: 540, facingMode: "user" },
        audio: false,
      });
      if (token !== S.startToken) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; return; }
      video.srcObject = S.stream;
      await video.play();
    } catch (err) {
      if (token === S.startToken) cameraFailed(err);
      return;
    }
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    S.streaming = true;
    S.stream.getVideoTracks().forEach((t) => {
      t.onended = () => { if (S.cam === "on" || S.cam === "starting") cameraFailed({ name: "NotFoundError" }); };
    });
    paint();

    // 2. load / verify the model
    if (!(await ensureModelReady(token))) {
      if (token === S.startToken) stopCamera();
      return;
    }
    if (token !== S.startToken) return;

    // 3. begin recognition
    try {
      applySnapshot(await postJSON("/api/resume"));
    } catch (err) {
      showBanner(MSG.service);
      stopCamera();
      return;
    }
    S.recognizing = true;
    setCam("on");
    loadHandDetector();          // not awaited: frames wait until the detector has started
    sendLoop();
  }

  async function startRecognition() {
    if (S.cam !== "on" || S.recognizing) return;
    try {
      applySnapshot(await postJSON("/api/resume"));   // forget old frames and any lock, keep the text
    } catch (err) {
      return showBanner(MSG.service);
    }
    S.recognizing = true;
    paint();
  }

  function pauseRecognition() {
    if (S.cam !== "on") return;
    S.recognizing = false;            // the loop stops sending frames: prediction stops, text stays
    paint();
  }

  /* ------------------------------------------------------------ frame streaming */
  function grabFrame() {
    const scale = Math.min(1, MAX_UPLOAD_WIDTH / video.videoWidth);
    grab.width = Math.round(video.videoWidth * scale);
    grab.height = Math.round(video.videoHeight * scale);
    grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);   // raw, unmirrored, no overlay
    return new Promise((resolve, reject) =>
      grab.toBlob((b) => (b ? resolve(b) : reject(new Error("frame capture failed"))), "image/jpeg", 0.85)
    );
  }

  async function sendFrame() {
    const hand = handFlag();
    const form = new FormData();
    form.append("session", S.session);
    form.append("hand", hand === null ? "unknown" : hand ? "1" : "0");
    if (!(hand === false && S.config.require_hand)) {          // no hand: nothing to read, skip the upload
      form.append("frame", await grabFrame(), "frame.jpg");
      // the hand box lets the server crop around the hand (view modes "hand"/"multi") and measure hand size
      if (S.hand && S.hand.present && S.hand.bbox) form.append("bbox", S.hand.bbox.join(","));
    }
    if (S.debug) form.append("debug", "1");
    if (S.collect && S.hand && S.hand.present && S.hand.bbox && S.collectTick++ % 2 === 0) {
      form.append("collect", String(S.collect.index));
      form.append("collect_set", S.collect.set);
    }
    const res = await fetch("/api/frame", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (res.status === 503) {
      const err = new Error(data.error || "The model is not ready.");
      err.modelState = data.model_state;
      throw err;
    }
    if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
    return data;
  }

  async function sendLoop() {
    if (S.looping) return;
    S.looping = true;
    let failures = 0;
    while (S.cam === "on") {
      const waitingForDetector = S.config.require_hand && (S.holisticState === "loading" || S.holisticState === "idle");
      if (!S.recognizing || document.hidden || waitingForDetector) {
        await sleep(150);
        continue;
      }
      const t0 = performance.now();
      try {
        const snap = await sendFrame();
        if (!S.recognizing || S.cam !== "on") continue;         // paused or stopped while the request was in flight
        failures = 0;
        if (S.serviceError) { S.serviceError = false; clearBanner(MSG.service); }
        S.model.state = "ready";
        applySnapshot(snap);
        if (S.lastFrameAt) S.fps = S.fps ? S.fps * 0.85 + (1000 / Math.max(1, t0 - S.lastFrameAt)) * 0.15 : 1000 / Math.max(1, t0 - S.lastFrameAt);
        S.lastFrameAt = t0;
      } catch (err) {
        if (err.modelState === "loading") {
          S.model.state = "loading";
          paint();
          await sleep(1000);
        } else if (err.modelState === "error") {
          S.model.state = "error";
          showBanner(err.message);
          stopCamera();
          break;
        } else {
          failures += 1;
          S.serviceError = true;
          showBanner(MSG.service);
          paint();
          await sleep(Math.min(3000, 500 * failures));
        }
      }
      const elapsed = performance.now() - t0;
      if (elapsed < FRAME_INTERVAL_MS) await sleep(FRAME_INTERVAL_MS - elapsed);
    }
    S.looping = false;
  }

  /* ------------------------------------------------------------ manual keys */
  async function manual(action) {
    try {
      applySnapshot(await postJSON("/api/" + action));
    } catch (err) {
      showBanner(MSG.service);
      toast(MSG.service);
    }
  }

  /* ------------------------------------------------------------ settings */
  function applyConfig(cfg) {
    Object.assign(S.config, cfg);
    for (const key of Object.keys(SLIDERS)) {
      if (cfg[key] === undefined) continue;
      $(key).value = cfg[key];
      $("o_" + key).textContent = SLIDERS[key](cfg[key]);
    }
    for (const key of CHECKS) if (cfg[key] !== undefined) $(key).checked = !!cfg[key];
    for (const key of SELECTS) if (cfg[key] !== undefined) $(key).value = cfg[key];
    paint();
  }

  function pushConfig(patch) {
    Object.assign(S.config, patch);
    S.pendingPatch = Object.assign(S.pendingPatch || {}, patch);
    clearTimeout(S.pushTimer);
    S.pushTimer = setTimeout(async () => {
      const body = S.pendingPatch;
      S.pendingPatch = null;
      try {
        applyConfig((await postJSON("/api/config", { config: body })).config);   // server clamps values
      } catch (err) {
        toast("Could not save the settings. " + MSG.service);
      }
    }, 150);
  }

  /* ------------------------------------------------------------ navigation highlight */
  function setActiveNav(section) {
    document.querySelectorAll(".nav-link").forEach((a) => {
      if (a.dataset.section === section) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }

  function watchSections() {
    if (!("IntersectionObserver" in window)) return;
    const map = { dashboard: "home", developer: "developer", how: "how", about: "about" };
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && setActiveNav(map[e.target.id])),
      { rootMargin: "-35% 0px -60% 0px" }
    );
    Object.keys(map).forEach((id) => io.observe($(id)));
  }

  /* ------------------------------------------------------------ startup */
  function wire() {
    $("btnStart").addEventListener("click", startCamera);
    $("btnStop").addEventListener("click", stopCamera);
    $("btnRecStart").addEventListener("click", startRecognition);
    $("btnRecPause").addEventListener("click", pauseRecognition);
    $("btnSpace").addEventListener("click", () => manual("space"));
    $("btnBack").addEventListener("click", () => manual("backspace"));
    $("btnClear").addEventListener("click", () => manual("clear"));
    $("btnSpeak").addEventListener("click", speak);
    document.querySelectorAll(".lang-btn").forEach((b) => b.addEventListener("click", () => setLanguage(b.dataset.lang)));

    for (const key of Object.keys(SLIDERS)) {
      $(key).addEventListener("input", (e) => {
        const v = parseFloat(e.target.value);
        $("o_" + key).textContent = SLIDERS[key](v);
        pushConfig({ [key]: v });
      });
    }
    for (const key of CHECKS) $(key).addEventListener("change", (e) => pushConfig({ [key]: e.target.checked }));
    for (const key of SELECTS) $(key).addEventListener("change", (e) => pushConfig({ [key]: e.target.value }));

    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target;
      if (["INPUT", "SELECT", "TEXTAREA"].includes(t.tagName) || t.isContentEditable) return;   // typing / sliders
      const activatesOnSpace = ["BUTTON", "SUMMARY", "A"].includes(t.tagName);                    // Space already clicks these
      if (e.code === "Space") { if (!activatesOnSpace) { e.preventDefault(); manual("space"); } }
      else if (e.key === "Backspace") { e.preventDefault(); manual("backspace"); }
      else if (e.key === "Delete" && e.shiftKey) { e.preventDefault(); manual("clear"); }
    });

    window.addEventListener("beforeunload", () => { if (window.speechSynthesis) speechSynthesis.cancel(); });
  }

  /* Poll the server until the model has finished loading (or failed). */
  async function watchModel() {
    for (;;) {
      try {
        const info = await getInfo();
        S.modelReachable = true;
        S.model = info.model;
        clearBanner(MSG.service);
        if (!S.configLoaded) { S.configLoaded = true; applyConfig(info.config); }
        setText($("factClasses"), info.model.classes + " Marathi (Devanagari) letters");
        setText($("factInput"), info.model.input_size + " × " + info.model.input_size + " colour image");
        paint();
        if (info.model.state === "error" && S.cam === "off") showBanner(info.model.error);
        if (info.model.state !== "loading") return;
        await sleep(1200);
      } catch (err) {
        S.modelReachable = false;
        showBanner(MSG.service);
        paint();
        await sleep(3000);
      }
    }
  }

  // Small public surface for developer.js. Nothing here can change a prediction.
  window.ISL = {
    S,
    onSnapshot: (fn) => snapshotListeners.push(fn),
    setDebug: (on) => { S.debug = !!on; },
    setCollect: (c) => { S.collect = c || null; S.collectTick = 0; },
    postJSON,
    toast,
  };

  wire();
  watchSections();
  renderTranslations();
  paint();
  watchModel();
})();
