/* Developer mode.
 *
 * Shows what the trained model sees and predicts, and lets you test every character. It only DISPLAYS
 * what the server returns (the CNN's probabilities and the recognizer's decisions); it never decides,
 * changes or fakes a prediction.
 *
 *   /api/diagnostics  -> label check, model structure, offline per-character report (tools/evaluate_model.py)
 *   /api/frame (debug) -> per-frame probabilities, the picture the model received, preprocessing comparison
 */
"use strict";

(() => {
  const ISL = window.ISL;
  if (!ISL) return;
  const $ = (id) => document.getElementById(id);

  const STATUS = { good: "Good", fair: "Fair", poor: "Poor", not_detected: "Not detected", untested: "Untested" };
  const TONE = { good: "ok", fair: "info", poor: "warn", not_detected: "err", untested: "idle" };
  const VIEW_NAMES = {
    full: "Whole frame",
    hand: "Hand crop",
    multi: "Three hand crops, averaged",
    mirror_full: "Mirrored whole frame",
    mirror_hand: "Mirrored hand crop",
  };
  const MIN_TEST_FRAMES = 25;
  const MAX_TEST_FRAMES = 200;
  const LIVE_KEY = "isl_live_results";
  const DEV_KEY = "isl_dev_mode";

  const D = {
    on: false,
    diag: null,
    selected: null,       // class index
    live: readStore(LIVE_KEY) || {},
    collected: {},        // {set: {class index: frames stored on this computer}}
    test: null,
    lastProcessed: -1,
  };

  /* ------------------------------------------------------------ helpers */
  function readStore(key) {
    try { return JSON.parse(localStorage.getItem(key) || "null"); } catch (err) { return null; }
  }
  function writeStore(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (err) { /* private mode: keep in memory only */ }
  }
  const pct = (v, digits = 0) => (v === null || v === undefined ? "–" : (v * 100).toFixed(digits) + "%");
  const setText = (id, text) => { const el = $(id); if (el.textContent !== text) el.textContent = text; };

  function el(tag, props, children) {
    const node = document.createElement(tag);
    Object.assign(node, props || {});
    (children || []).forEach((c) => node.append(c));
    return node;
  }
  function dlRows(rows) {
    const dl = el("dl", { className: "dev-readout" });
    rows.forEach(([k, v]) => dl.append(el("div", {}, [el("dt", { textContent: k }), el("dd", { textContent: v })])));
    return dl;
  }

  /* ------------------------------------------------------------ switching developer mode on / off */
  function setDev(on) {
    D.on = on;
    $("devToggle").checked = on;
    $("developer").hidden = !on;
    $("navDev").hidden = !on;
    ISL.setDebug(on);                       // ask the server for the full diagnostics with every frame
    writeStore(DEV_KEY, on);
    if (!on) stopTest(false);
    if (on && !D.diag) loadDiagnostics();
    updateButtons();
  }

  async function loadDiagnostics() {
    setText("devSummary", "Loading…");
    try {
      const res = await fetch("/api/diagnostics");
      if (!res.ok) throw new Error("HTTP " + res.status);
      D.diag = await res.json();
    } catch (err) {
      setText("devSummary", "Could not load the diagnostics. Is the server still running?");
      return;
    }
    await loadCollected();
    renderSummary();
    renderIssues();
    renderLegend();
    renderGrid();
    renderModel();
    if (D.selected !== null) select(D.selected);
  }

  /* ------------------------------------------------------------ character grid */
  function combinedTone(c) {
    const live = D.live[c.label];
    return live ? live.verdict : c.webcam ? c.webcam.status : c.status;
  }

  function renderSummary() {
    const d = D.diag;
    const parts = [];
    const r = d.report;
    if (!r) {
      parts.push("No evaluation yet, so every character shows as Untested. On your computer run "
        + "python tools/evaluate_model.py (dataset) and/or record webcam frames below and run it with --collected, then press Reload. "
        + "The live tester works without either.");
    } else {
      if (r.overall) {
        const s = r.summary;
        parts.push(`Dataset test (${r.split}): ` + ["good", "fair", "poor", "not_detected", "untested"]
          .filter((k) => s[k]).map((k) => `${STATUS[k]} ${s[k]}`).join(", ")
          + `. Overall accuracy ${pct(r.overall.accuracy, 1)}.`);
        const never = (r.overall.classes_never_predicted || []).length;
        if (never) parts.push(`${never} character(s) are never predicted: the model cannot output them, so it must be retrained.`);
      }
      if (r.webcam) {
        const w = r.webcam;
        parts.push(`Your webcam frames (set "${w.set}", ${w.frames} frames, ${w.recorded_classes} characters): accuracy by view `
          + Object.entries(w.mode_accuracy).map(([k, v]) => `${k.replace(/_/g, " ")} ${pct(v)}`).join(", ")
          + `; best ${w.best_mode.replace(/_/g, " ")}. With the accept rules ${pct(w.gated.pass_rate)} of frames pass`
          + (w.gated.wrong_among_passed !== null ? ` and ${pct(w.gated.wrong_among_passed)} of those are wrong.` : "."));
      }
    }
    const tested = Object.values(D.live);
    if (tested.length) {
      const c = { good: 0, poor: 0, not_detected: 0 };
      tested.forEach((t) => { c[t.verdict] = (c[t.verdict] || 0) + 1; });
      parts.push(`Live tests on this camera: ${tested.length} character(s), good ${c.good || 0}, poor ${c.poor || 0}, not detected ${c.not_detected || 0}.`);
    }
    setText("devSummary", parts.join(" "));
  }

  function renderIssues() {
    const list = $("devIssues");
    list.replaceChildren();
    const issues = D.diag.label_issues || [];
    if (!D.diag.model.ready) issues.push({ level: "warning", message: "The model is not loaded yet: " + (D.diag.model.error || D.diag.model.state) });
    if (!issues.length) {
      list.append(el("li", { className: "ok", textContent: `Label check passed: ${D.diag.labels.length} unique characters, in the model's output order.` }));
    }
    issues.forEach((i) => list.append(el("li", { className: i.level, textContent: (i.level === "error" ? "Error: " : "Warning: ") + i.message })));
  }

  function renderLegend() {
    const ul = $("devLegend");
    ul.replaceChildren();
    Object.entries(D.diag.legend).forEach(([k, text]) => {
      ul.append(el("li", {}, [el("i", { className: "swatch" }), el("span", { textContent: `${STATUS[k]}: ${text}` })]));
      ul.lastChild.firstChild.dataset.tone = TONE[k];
    });
  }

  function renderGrid() {
    const grid = $("charGrid");
    grid.replaceChildren();
    D.diag.classes.forEach((c) => {
      const live = D.live[c.label];
      const b = el("button", { type: "button", className: "char" });
      b.dataset.index = c.index;
      b.dataset.tone = TONE[combinedTone(c)];
      b.setAttribute("aria-pressed", String(D.selected === c.index));
      b.setAttribute("aria-label", `${c.label}: dataset ${STATUS[c.status]}` + (c.webcam ? `, webcam ${STATUS[c.webcam.status]}` : "") + (live ? `, live ${STATUS[live.verdict]}` : ""));
      b.append(
        el("span", { className: "char-glyph", textContent: c.label, lang: "mr" }),
        el("span", { className: "char-cap", textContent: "Dataset: " + STATUS[c.status] }),
        el("span", { className: "char-live", textContent: live ? `Live: ${STATUS[live.verdict]} ${pct(live.rate)}`
          : c.webcam ? `Webcam: ${STATUS[c.webcam.status]} ${pct(c.webcam.recall)}` : "Live: not tested" })
      );
      b.addEventListener("click", () => select(c.index));
      grid.append(b);
    });
  }

  function select(index) {
    D.selected = index;
    const c = D.diag && D.diag.classes[index];
    if (!c) return;
    document.querySelectorAll("#charGrid .char").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.index) === index)));
    setText("tTarget", c.label);

    const detail = $("charDetail");
    detail.hidden = false;
    detail.replaceChildren();
    const rows = [];
    if (c.support) {
      rows.push(["Dataset test", `${STATUS[c.status]}: found ${pct(c.recall)} of ${c.support} test pictures; ${pct(c.precision)} of its predictions were right`]);
      if (c.confused_with && c.confused_with.length) rows.push(["Often mistaken for", c.confused_with.map((x) => `${x[0]} (${x[1]}x)`).join(", ")]);
      if (c.suggested_threshold !== null && c.suggested_threshold !== undefined) rows.push(["Calibrated confidence", pct(c.suggested_threshold) + " (accepted from this level, never below the floor)"]);
      else if (c.predicted_count) rows.push(["Calibrated confidence", "none found: this character is never reliable enough at any level"]);
      if (c.robustness && Object.keys(c.robustness).length) rows.push(["Robustness (offline)", Object.entries(c.robustness).map(([k, v]) => `${k.replace(/_/g, " ")} ${pct(v)}`).join(", ")]);
    } else {
      rows.push(["Dataset test", "no test data (run python tools/evaluate_model.py)"]);
    }
    if (c.webcam) {
      const w = c.webcam;
      rows.push(["Recorded webcam frames", `${STATUS[w.status]}: right in ${pct(w.recall)} of ${w.support} frames; passes the accept rules in ${pct(w.pass_rate)}`
        + (w.confused_with && w.confused_with.length ? `; mistaken for ${w.confused_with.map((x) => `${x[0]} (${x[1]}x)`).join(", ")}` : "")]);
      rows.push(["Webcam by view", Object.entries(w.recall_by_mode).map(([k, v]) => `${k.replace(/_/g, " ")} ${pct(v)}`).join(", ")]);
    }
    const stored = Object.entries(D.collected).map(([set, m]) => [set, m[c.index] || 0]).filter(([, n]) => n);
    if (stored.length) rows.push(["Frames saved on this computer", stored.map(([set, n]) => `${set}: ${n}`).join(", ")]);
    const live = D.live[c.label];
    rows.push(["Live test", live ? `${STATUS[live.verdict]}: top guess was ${c.label} in ${pct(live.rate)} of ${live.frames} frames`
      + (live.confused && live.confused.length ? `; mistaken for ${live.confused.map((x) => `${x[0]} (${x[1]}x)`).join(", ")}` : "") : "not tested yet"]);
    detail.append(dlRows(rows));

    renderRefs(c);
    updateButtons();
  }

  function renderRefs(c) {
    const box = $("dRefs");
    box.replaceChildren();
    const n = D.diag.references_per_class;
    if (!n) {
      setText("dRefCap", "No dataset pictures yet: python tools/evaluate_model.py exports them.");
      return;
    }
    for (let k = 0; k < n; k++) {
      const img = el("img", { src: `/reference/${c.index}_${k}.jpg`, alt: `Dataset picture ${k + 1} of ${c.label}`, width: 92, height: 92, loading: "lazy" });
      img.onerror = () => img.remove();
      box.append(img);
    }
    setText("dRefCap", `Dataset pictures of ${c.label}: what the model was trained on. Copy this pose and framing.`);
  }

  function renderModel() {
    const d = D.diag;
    const box = $("devModel");
    box.replaceChildren();
    const m = d.model;
    const rows = [
      ["Model status", m.state + (m.error ? ": " + m.error : "")],
      ["Input", m.input_shape ? `${m.input_shape.slice(1).join(" × ")} (${m.input_size} px square)` : "–"],
      ["Outputs", m.output_shape ? `${m.output_shape[m.output_shape.length - 1]} classes` : "–"],
      ["Parameters", m.params ? m.params.toLocaleString() : "–"],
      ["Layers", m.layers ? m.layers.map((l) => l.type).join(", ") : "–"],
      ["Calibrated characters", `${d.calibrated_classes} of ${d.labels.length}`],
    ];
    const r = d.report;
    if (r) {
      rows.push(["Report created", r.generated || "–"]);
      rows.push(["Pictures in dataset", r.dataset && r.dataset.images ? String(r.dataset.images) : "–"]);
      if (r.dataset && r.dataset.imbalance_ratio) rows.push(["Class balance", `largest class is ${r.dataset.imbalance_ratio}x the smallest`]);
      if (r.overall) rows.push(["Top-3 / top-5 accuracy", `${pct(r.overall.top3, 1)} / ${pct(r.overall.top5, 1)}`]);
      if (r.robustness) rows.push(["Robustness (accuracy)", Object.entries(r.robustness).map(([k, v]) => `${k.replace(/_/g, " ")} ${pct(v)}`).join(", ")]);
      if (r.confusions && r.confusions.length) rows.push(["Most confused pairs", r.confusions.slice(0, 6).map((x) => `${x[0]}→${x[1]} (${x[2]})`).join(", ")]);
      if (r.dataset && r.dataset.similar_pairs) rows.push(["Look-alike training pictures", r.dataset.similar_pairs.slice(0, 5).map((x) => `${x[0]} ~ ${x[1]}`).join(", ")]);
    }
    box.append(dlRows(rows));
    const reload = el("button", { type: "button", className: "btn outline small", textContent: "Reload report" });
    reload.addEventListener("click", loadDiagnostics);
    box.append(reload);
  }

  /* ------------------------------------------------------------ live readout */
  function renderBars(id, list) {
    const ol = $(id);
    ol.replaceChildren(...(list || []).map(([label, p]) => {
      const bar = el("span", { className: "bar-i" });
      bar.style.width = (p * 100).toFixed(1) + "%";
      return el("li", {}, [el("span", { className: "b-label", textContent: label, lang: "mr" }), el("span", { className: "b-track" }, [bar]), el("span", { className: "b-val", textContent: pct(p, 1) })]);
    }));
  }

  function renderLive(r) {
    if (!D.on) return;
    const d = r.debug || {};
    const top = r.top || [];
    const live = r.window_fill > 0;
    setText("dPred", live && r.current_sign ? r.current_sign : "–");
    setText("dConf", live ? pct(r.confidence, 1) : "–");
    setText("dAlt", live && top.length > 1 ? top.slice(1, 4).map((t) => `${t[0]} ${pct(t[1])}`).join(", ") : "–");
    setText("dMargin", live ? pct(r.margin, 1) : "–");
    setText("dAgree", live ? `${pct(r.view_agreement)} (needs ${pct(r.min_view_agreement)})` + (d.view_tops && d.view_tops.length > 1 ? `: ${d.view_tops.map((v) => `${v[0]} ${pct(v[1])}`).join(", ")}` : "") : "–");
    setText("dStable", live ? `${r.stable_frames} in a row; ${r.agreement} of ${r.window_size} agree` : "–");
    setText("dStatus", r.state + (r.locked_sign ? ` (locked on ${r.locked_sign})` : ""));
    const need = Math.ceil((ISL.S.config.min_agreement ?? 0.8) * r.window_size - 1e-9);
    setText("dNeeded", `confidence ≥ ${pct(r.threshold_used)}, gap ≥ ${pct(r.min_margin)}, ${need} of ${r.window_size} frames agree, crops agree ≥ ${pct(r.min_view_agreement)}`);

    const q = d.quality;
    let hand = r.hand === true ? "Detected" : r.hand === false ? "Not detected" : "Unknown (detector not running)";
    if (r.hand === true && q && q.hand_pixels) hand += `, about ${q.hand_pixels}px wide (${pct(q.hand_fraction, 1)} of the frame)`;
    setText("dHand", hand);
    const c = r.counters || {};
    setText("dCounters", `${c.processed || 0} read / ${c.skipped_no_hand || 0} skipped (no hand)`);
    setText("dQuality", q ? `brightness ${pct(q.brightness)}, contrast ${pct(q.contrast)}` + (q.notes.length ? `. ${q.notes.join(" ")}` : "") : "–");

    renderBars("dTop", top);
    renderBars("dInstant", r.instant_top);

    if (d.model_input) {
      $("dModelInput").src = d.model_input;
      const v = d.views || {};
      setText("dModelInputCap", `Model input ${d.model_input_size}×${d.model_input_size}px, from the ${v.mode_used === "full" ? "whole frame" : v.mode_used === "hand" ? "hand crop" : "first of " + v.n_views + " hand crops"}${v.mirrored ? ", mirrored" : ""}. ` +
        (v.mode_requested !== v.mode_used ? `(${v.mode_requested} was requested but no hand box was available.)` : ""));
    }
    if (d.compare) renderCompare(d);
    feedTest(r);
  }

  function renderCompare(d) {
    const rows = Object.keys(VIEW_NAMES).filter((k) => d.compare[k]);
    const best = rows.reduce((b, k) => (d.compare[k].top[0][1] > (b ? d.compare[b].top[0][1] : -1) ? k : b), null);
    const target = D.selected !== null && D.diag ? D.diag.classes[D.selected].label : null;
    const body = document.querySelector("#dCompare tbody");
    body.replaceChildren(...rows.map((k) => {
      const top = d.compare[k].top;
      const tr = el("tr", { className: k === best ? "best" : "" });
      const name = el("th", { scope: "row", textContent: VIEW_NAMES[k] });
      if (k === d.selected) name.append(el("span", { className: "tag", textContent: "in use" }));
      const guess = el("td", { lang: "mr", textContent: `${top[0][0]} ${pct(top[0][1])}` + (target && top[0][0] === target ? "  ✓" : "") });
      const alt = el("td", { lang: "mr", textContent: top.slice(1).map((t) => `${t[0]} ${pct(t[1])}`).join(", ") });
      tr.append(name, guess, alt);
      return tr;
    }));
  }

  /* ------------------------------------------------------------ character tester */
  function updateButtons() {
    const camera = ISL.S.cam === "on";
    $("tStart").disabled = !(D.on && D.selected !== null && camera && !D.test);
    $("tStop").disabled = !D.test;
  }

  const SET_OK = /^[A-Za-z0-9_-]{1,32}$/;

  function applyCollect() {
    const wanted = $("tCollect").checked;
    if (wanted && !SET_OK.test($("tSet").value.trim())) {
      $("tCollect").checked = false;
      ISL.toast("Set name: 1 to 32 letters, digits, - or _ .");
      ISL.setCollect(null);
      return;
    }
    ISL.setCollect(wanted && D.test ? { index: D.test.index, set: $("tSet").value.trim() } : null);
  }

  async function loadCollected() {
    try {
      const res = await fetch("/api/collected");
      if (!res.ok) return;
      const data = await res.json();
      D.collected = data.sets || {};
      const parts = Object.entries(D.collected).map(([set, m]) => `${set}: ${Object.values(m).reduce((a, b) => a + b, 0)} frames, ${Object.keys(m).length} characters`);
      setText("tCollectInfo", parts.length ? "Stored so far: " + parts.join("; ") + "." : "Nothing stored yet.");
      if (D.selected !== null && D.diag) select(D.selected);
    } catch (err) { /* offline: not important */ }
  }

  function startTest() {
    if (D.selected === null) return ISL.toast("Pick a character in the grid first.");
    if (ISL.S.cam !== "on") return ISL.toast("Start the camera first.");
    if (!ISL.S.recognizing) return ISL.toast("Recognition is paused. Press Start Recognition.");
    const c = D.diag.classes[D.selected];
    D.test = { index: c.index, label: c.label, frames: 0, hits: 0, sumProb: 0, ok: 0, wrong: 0, confusion: {}, views: {} };
    D.lastProcessed = ISL.S.snap && ISL.S.snap.counters ? ISL.S.snap.counters.processed : -1;
    applyCollect();
    renderTest();
    updateButtons();
  }

  function feedTest(r) {
    const t = D.test;
    if (!t) return;
    const processed = r.counters ? r.counters.processed : 0;
    if (processed === D.lastProcessed) return;           // not a new camera frame (manual key, hand lowered, ...)
    D.lastProcessed = processed;
    if (r.hand === false || !r.probs || !r.probs.length || !r.instant_top || !r.instant_top.length) return;

    t.frames += 1;
    const top1 = r.instant_top[0][0];
    if (top1 === t.label) t.hits += 1; else t.confusion[top1] = (t.confusion[top1] || 0) + 1;
    t.sumProb += r.probs[t.index] || 0;
    if (r.event === "accepted") { if (r.accepted === t.label) t.ok += 1; else t.wrong += 1; }
    if (typeof r.collected === "number") t.saved = r.collected;
    const cmp = r.debug && r.debug.compare;
    if (cmp) {
      Object.entries(cmp).forEach(([name, v]) => {
        const s = t.views[name] || (t.views[name] = { frames: 0, hits: 0 });
        s.frames += 1;
        if (v.top[0][0] === t.label) s.hits += 1;
      });
    }
    renderTest();
    if (t.frames >= MAX_TEST_FRAMES) stopTest(true);
  }

  function verdictFor(t) {
    if (t.frames < MIN_TEST_FRAMES) return null;
    const rate = t.hits / t.frames;
    return rate >= 0.7 ? "good" : rate >= 0.3 ? "poor" : "not_detected";
  }

  function topConfusions(t) {
    return Object.entries(t.confusion).sort((a, b) => b[1] - a[1]).slice(0, 3);
  }

  function bestView(t) {
    const rows = Object.entries(t.views).filter(([, s]) => s.frames >= 10).map(([k, s]) => [k, s.hits / s.frames]);
    rows.sort((a, b) => b[1] - a[1]);
    return rows[0] || null;
  }

  function renderTest() {
    const t = D.test;
    const stats = $("tStats");
    const verdictEl = $("tVerdict");
    if (!t) { stats.replaceChildren(); return; }
    const rate = t.frames ? t.hits / t.frames : 0;
    const rows = [
      ["Frames tested", `${t.frames} (needs ${MIN_TEST_FRAMES}+)`],
      [`Top guess was ${t.label}`, `${t.hits} frames (${pct(rate)})`],
      [`Average probability of ${t.label}`, pct(t.frames ? t.sumProb / t.frames : 0, 1)],
      ["Accepted as the letter", `${t.ok} correct, ${t.wrong} as another character`],
    ];
    if (t.saved) rows.push(["Frames saved", `${t.saved} for ${t.label} in set "${$("tSet").value.trim()}"`]);
    const conf = topConfusions(t);
    if (conf.length) rows.push(["Mistaken for", conf.map(([k, n]) => `${k} (${n} frames)`).join(", ")]);
    const bv = bestView(t);
    if (bv) rows.push(["Best preprocessing so far", `${VIEW_NAMES[bv[0]]} (${pct(bv[1])} of frames)`]);
    stats.replaceChildren(...dlRows(rows).children);
    const v = verdictFor(t);
    verdictEl.dataset.tone = v ? TONE[v] : "idle";
    verdictEl.textContent = v ? `${STATUS[v]}: the model's top guess was ${t.label} in ${pct(rate)} of the frames.`
      : `Keep the sign in view: ${t.frames} of ${MIN_TEST_FRAMES} frames.`;
  }

  function stopTest(save) {
    const t = D.test;
    if (!t) return;
    D.test = null;
    ISL.setCollect(null);
    if (t.saved) loadCollected();
    const verdict = verdictFor(t);
    if (save !== false && verdict) {
      const bv = bestView(t);
      D.live[t.label] = {
        verdict, rate: +(t.hits / t.frames).toFixed(3), frames: t.frames,
        mean_prob: +(t.sumProb / t.frames).toFixed(3), accepted_ok: t.ok, accepted_wrong: t.wrong,
        confused: topConfusions(t), best_view: bv ? bv[0] : null, at: new Date().toISOString(),
      };
      writeStore(LIVE_KEY, D.live);
      renderSummary();
      renderGrid();
      select(t.index);
    } else if (save !== false) {
      ISL.toast(`Only ${t.frames} frames were tested. Keep the sign in view for at least ${MIN_TEST_FRAMES} frames to save a result.`);
    }
    updateButtons();
  }

  function exportResults() {
    const payload = {
      generated: new Date().toISOString(),
      live_results: D.live,
      offline_summary: D.diag && D.diag.report ? D.diag.report.summary : null,
      settings: ISL.S.config,
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 1)], { type: "application/json" }));
    const a = el("a", { href: url, download: "isl-character-test-results.json" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ------------------------------------------------------------ wiring */
  $("devToggle").addEventListener("change", (e) => setDev(e.target.checked));
  $("tStart").addEventListener("click", startTest);
  $("tStop").addEventListener("click", () => stopTest(true));
  $("tReset").addEventListener("click", () => {
    D.live = {};
    writeStore(LIVE_KEY, D.live);
    if (D.diag) { renderSummary(); renderGrid(); if (D.selected !== null) select(D.selected); }
    ISL.toast("Saved live results cleared.");
  });
  $("tExport").addEventListener("click", exportResults);
  $("tCollect").addEventListener("change", applyCollect);
  $("tSet").addEventListener("change", applyCollect);
  ISL.onSnapshot((r) => { renderLive(r); updateButtons(); });

  if (readStore(DEV_KEY)) setDev(true);
})();
