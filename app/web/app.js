/* Crop Disease Detection: frontend (no build step). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const svgNS = "http://www.w3.org/2000/svg";
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { model: "efficientnet_b0", file: null, result: null, cam: "gradcam", stats: null, facing: "environment", stream: null, req: 0, showAllClasses: false };
const MODEL_KEY = { custom_cnn: "custom", efficientnet_b0: "effb0" };
const MODEL_LABEL = { custom_cnn: "Custom CNN", efficientnet_b0: "EfficientNet-B0" };

const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const int = (n) => Number(n).toLocaleString("en-US");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ------------------------------------------------------------ theme */
function currentTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function syncThemeButton() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  $("#theme-toggle").setAttribute("aria-label", `Switch to ${next} theme`);
}
$("#theme-toggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
  syncThemeButton();
});
syncThemeButton();

/* ------------------------------------------------------------ stage states */
const stage = $("#stage");
function setStage(s) { stage.dataset.state = s; }
function showError(title, text) {
  $("#error-title").textContent = title;
  $("#error-text").textContent = text;
  setStage("error");
}

/* ------------------------------------------------------------ models */
async function loadModels() {
  const box = $("#model-options");
  try {
    const res = await fetch("/api/models");
    const models = await res.json();
    if (!models.length) throw new Error("none");
    if (!models.find((m) => m.name === state.model)) state.model = models[0].name;
    box.innerHTML = models.map((m) => `
      <label><input type="radio" name="model" value="${m.name}" ${m.name === state.model ? "checked" : ""}>
      <span><i class="dot dot-${MODEL_KEY[m.name]}" aria-hidden="true"></i>${esc(m.display)}<small data-lat="${m.name}"></small></span></label>`).join("");
    box.addEventListener("change", (e) => {
      if (e.target.name !== "model") return;
      state.model = e.target.value;
      if (state.file) run(state.file);
    });
    fillLatency();
  } catch {
    box.innerHTML = `<span class="foot">No trained models installed. Run <code>python scripts/fetch_models.py</code>.</span>`;
  }
}
function fillLatency() {
  if (!state.stats) return;
  document.querySelectorAll("[data-lat]").forEach((el) => {
    const m = state.stats.models[el.dataset.lat];
    if (m?.latency_ms?.onnx) el.textContent = `${m.latency_ms.onnx} ms`;
  });
}

/* ------------------------------------------------------------ samples */
async function loadSamples() {
  try {
    const list = await (await fetch("/api/samples")).json();
    $("#sample-row").innerHTML = list.map((s, i) => `
      <button class="sample" type="button" role="listitem" data-i="${i}" aria-label="Try sample: ${esc(s.crop)}, ${esc(s.disease)}">
        <img src="${s.url}" alt="" width="52" height="52" loading="lazy"></button>`).join("");
    const runSample = async (s) => {
      const blob = await (await fetch(s.url)).blob();
      run(new File([blob], s.file, { type: blob.type || "image/jpeg" }));
    };
    $("#sample-row").addEventListener("click", (e) => {
      const btn = e.target.closest(".sample");
      if (btn) runSample(list[+btn.dataset.i]);
    });
    // deep link: /?sample=tomato_early_blight.jpg runs that sample on load
    const linked = new URLSearchParams(location.search).get("sample");
    const hit = linked && list.find((s) => s.file === linked);
    if (hit) runSample(hit);
  } catch {
    $(".samples").hidden = true;
  }
}

/* ------------------------------------------------------------ upload, drop, paste */
const fileInput = $("#file-input");
$("#upload-btn").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) run(fileInput.files[0]); fileInput.value = ""; });
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) run(f); });
window.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
  if (f) run(f);
});
$("#error-retry").addEventListener("click", () => setStage("empty"));
$("#again-btn").addEventListener("click", () => {
  state.file = null;
  $("#diagnosis").hidden = true;
  setStage("empty");
  $("#diagnose").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  $("#upload-btn").focus({ preventScroll: true });
});

/* ------------------------------------------------------------ camera */
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Camera unavailable here", "Browsers only allow the camera on https or localhost. Upload a photo instead.");
    return;
  }
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 1280 } }, audio: false });
    const v = $("#video");
    v.srcObject = state.stream;
    await v.play();
    setStage("camera");
    $("#shutter").focus();
  } catch (err) {
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    showError(denied ? "Camera permission denied" : "No camera found",
      denied ? "Allow camera access in the browser's address bar, or upload a photo instead." : "Connect a camera or upload a photo instead.");
  }
}
function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
}
$("#camera-btn").addEventListener("click", startCamera);
$("#camera-cancel").addEventListener("click", () => { stopCamera(); setStage("empty"); });
$("#camera-flip").addEventListener("click", () => { state.facing = state.facing === "environment" ? "user" : "environment"; startCamera(); });
$("#shutter").addEventListener("click", () => {
  const v = $("#video");
  const side = Math.min(v.videoWidth, v.videoHeight);
  const c = document.createElement("canvas");
  c.width = c.height = side;
  c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side);
  stopCamera();
  c.toBlob((b) => run(new File([b], "camera.jpg", { type: "image/jpeg" })), "image/jpeg", 0.92);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && stage.dataset.state === "camera") { stopCamera(); setStage("empty"); }
});

/* ------------------------------------------------------------ predict */
const photo = $("#specimen-photo");
const heat = $("#specimen-heat");

async function run(file) {
  state.file = file;
  const id = ++state.req;
  const preview = URL.createObjectURL(file);
  photo.src = preview;
  heat.removeAttribute("src");
  $("#stage-status").textContent = `Examining the leaf with ${MODEL_LABEL[state.model] || "the model"}…`;
  setStage("loading");

  const body = new FormData();
  body.append("file", file);
  body.append("model", state.model);
  try {
    const res = await fetch("/api/predict", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (id !== state.req) return; // a newer request replaced this one
    if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
    state.result = data;
    photo.src = data.image;
    heat.src = data.heatmaps[state.cam];
    setStage("result");
    renderDiagnosis(data);
  } catch (err) {
    if (id !== state.req) return;
    showError("That didn't work", err.message || "Check that the server is running, then try again.");
  } finally {
    URL.revokeObjectURL(preview);
  }
}

/* heatmap controls */
const heatAmount = $("#heat-amount");
function syncHeat() {
  $(".specimen-img").style.setProperty("--heat", heatAmount.value / 100);
  $("#heat-out").textContent = `${heatAmount.value}%`;
}
heatAmount.addEventListener("input", syncHeat);
syncHeat();
document.querySelectorAll('input[name="cam"]').forEach((r) => r.addEventListener("change", () => {
  state.cam = r.value;
  if (state.result) { heat.src = state.result.heatmaps[state.cam]; renderMeta(state.result); }
}));

function renderDiagnosis(d) {
  const top = d.top[0];
  const healthy = top.disease === "Healthy";
  $("#dx-crop").textContent = top.crop;
  $("#dx-title").textContent = healthy ? "Healthy leaf" : top.disease;
  $("#dx-conf").textContent = pct(top.prob, 0);
  const meter = $("#dx-meter");
  meter.parentElement.classList.toggle("unsure", !!d.unsure);
  meter.style.width = "0";
  requestAnimationFrame(() => { meter.style.width = `${top.prob * 100}%`; });
  $("#dx-tau").style.left = `${d.model.unsure_threshold * 100}%`;
  $("#dx-tau").title = `Unsure below ${pct(d.model.unsure_threshold, 0)}`;
  $("#dx-unsure").hidden = !d.unsure;
  $("#dx-unsure-reason").textContent = d.unsure ? `(${d.unsure}).` : "";
  $("#dx-noleaf").hidden = !d.no_leaf;

  $("#dx-top3").innerHTML = d.top.map((t) => `
    <li><span class="name">${esc(t.disease === "Healthy" ? "Healthy" : t.disease)} <small>· ${esc(t.crop)}</small></span>
    <span class="pct">${pct(t.prob, 1)}</span>
    <span class="bar"><span style="width:0" data-w="${(t.prob * 100).toFixed(1)}%"></span></span></li>`).join("");
  requestAnimationFrame(() => document.querySelectorAll("#dx-top3 .bar span").forEach((s) => { s.style.width = s.dataset.w; }));
  renderMeta(d);
  renderInfo(d);

  const sec = $("#diagnosis");
  const first = sec.hidden;
  sec.hidden = false;
  if (first) sec.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

function renderMeta(d) {
  $("#dx-meta").innerHTML = `
    <div><dt>Model</dt><dd>${esc(d.model.display)}</dd></div>
    <div><dt>Runtime</dt><dd>${esc(d.model.runtime)}</dd></div>
    <div><dt>Prediction</dt><dd>${d.timing_ms.predict} ms</dd></div>
    <div><dt>Both heatmaps</dt><dd>${d.timing_ms.gradcam} ms</dd></div>`;
}

function renderInfo(d) {
  const box = $("#dx-info");
  const info = d.info || {};
  const top = d.top[0];
  if (d.unsure) {
    box.innerHTML = `<p class="eyebrow">Disease notes</p><h3 id="info-title">Retake the photo first</h3>
      <p class="chart-sub">The model isn't confident enough to show treatment notes. Try one leaf filling the frame, in daylight, on a plain background.</p>`;
    return;
  }
  if (info.healthy) {
    box.innerHTML = `<p class="eyebrow">Disease notes</p><h3 id="info-title">No disease found</h3>
      <span class="badge good">Healthy ${esc(top.crop.toLowerCase())} leaf</span>
      <section><h4>Keep it that way</h4><p>${esc(info.management)}</p></section>`;
    return;
  }
  const src = info.source_url ? `<a href="${esc(info.source_url)}" target="_blank" rel="noopener">Source</a>` : "";
  box.innerHTML = `<p class="eyebrow">Disease notes</p>
    <h3 id="info-title">${esc(info.disease || top.disease)}</h3>
    ${info.verified ? "" : `<span class="badge" title="Not yet checked against an agricultural extension source">Unverified draft</span>`}
    <section><h4>Cause</h4><p>${esc(info.cause)}</p></section>
    <section><h4>Symptoms</h4><p>${esc(info.symptoms)}</p></section>
    <section><h4>What to do</h4><p>${esc(info.management)}</p></section>
    <p class="foot">Check with a local agricultural extension officer before treating plants. ${src}</p>`;
}

/* ------------------------------------------------------------ stats */
async function loadStats() {
  $("#report-error").hidden = true;
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) throw new Error();
    state.stats = await res.json();
    renderStats(state.stats);
    fillLatency();
  } catch {
    $("#report-error").hidden = false;
    $("#kpis").hidden = true;
  }
}
$("#report-retry").addEventListener("click", () => { $("#kpis").hidden = false; loadStats(); });

function renderStats(s) {
  const e = s.models.efficientnet_b0;
  const c = s.models.custom_cnn;
  const ds = s.dataset;
  const pm = (m, d = 2) => `${(m.mean * 100).toFixed(d)}%`;
  const sd = (m) => `± ${(m.std * 100).toFixed(2)}`;

  $("#proof").innerHTML = `
    <li><span class="num">${pm(e.test_accuracy)}</span><span class="lbl">test accuracy</span></li>
    <li><span class="num">${int(ds.split.test)}</span><span class="lbl">unseen test leaves</span></li>
    <li><span class="num">${e.latency_ms.onnx} ms</span><span class="lbl">per leaf on a laptop CPU</span></li>`;
  $("#r-test-n").textContent = int(ds.split.test);

  $("#kpis").innerHTML = `
    <div class="kpi card"><span class="lbl">Test accuracy</span><span class="num">${pm(e.test_accuracy)}</span><span class="sub">EfficientNet-B0, ${sd(e.test_accuracy)} over 3 seeds</span></div>
    <div class="kpi card"><span class="lbl">Macro-F1</span><span class="num">${pm(e.test_macro_f1)}</span><span class="sub">every class weighted equally, ${sd(e.test_macro_f1)}</span></div>
    <div class="kpi card"><span class="lbl">Misclassified test leaves</span><span class="num">${e.test_errors}<span class="sub"> / ${int(ds.split.test)}</span></span><span class="sub">best-validation run (${e.app_run})</span></div>
    <div class="kpi card"><span class="lbl">Latency per leaf</span><span class="num">${e.latency_ms.onnx} ms</span><span class="sub">laptop CPU, ONNX Runtime, batch 1</span></div>`;

  const rows = [
    ["Test accuracy", (m) => m.test_accuracy.mean, (m) => `${pm(m.test_accuracy)} <small>${sd(m.test_accuracy)}</small>`, "max"],
    ["Macro-F1", (m) => m.test_macro_f1.mean, (m) => `${pm(m.test_macro_f1)} <small>${sd(m.test_macro_f1)}</small>`, "max"],
    ["Parameters", (m) => m.params_m, (m) => `${m.params_m.toFixed(2)}M`, "min"],
    ["Compute per image", (m) => m.gmacs, (m) => `${m.gmacs.toFixed(2)} GMACs`, "min"],
    ["CPU latency", (m) => m.latency_ms.onnx, (m) => `${m.latency_ms.onnx} ms`, "min"],
    ["Epochs trained", (m) => m.epochs_run.mean, (m) => `${m.epochs_run.mean.toFixed(1)}`, "min"],
    ["Training time", (m) => m.train_minutes.mean, (m) => `${m.train_minutes.mean.toFixed(1)} min`, "min"],
    ["From", null, (m) => (m.display === "Custom CNN" ? "scratch" : "ImageNet"), null],
  ];
  $("#compare-table tbody").innerHTML = rows.map(([label, val, fmt, best]) => {
    let wc = "", we = "";
    if (best) {
      const a = val(c), b = val(e);
      if (a !== b) { const cWins = best === "max" ? a > b : a < b; wc = cWins ? "win" : ""; we = cWins ? "" : "win"; }
    }
    return `<tr><td>${label}</td><td class="${wc}">${fmt(c)}</td><td class="${we}">${fmt(e)}</td></tr>`;
  }).join("");

  const ex = { c: c.explain, e: e.explain };
  if (ex.c && ex.e) {
    $("#x-leaf").textContent = pct(ex.e.leaf_fraction, 0);
    const pair = (k, d = 0) => `
      <span><i class="dot dot-custom" aria-hidden="true"></i>Custom ${pct(ex.c[k], d)}</span>
      <span><i class="dot dot-effb0" aria-hidden="true"></i>EffNet ${pct(ex.e[k], d)}</span>`;
    $("#xstats").innerHTML = `
      <div class="xstat"><span class="lbl">Heatmap mass on the leaf</span><div class="pair">${pair("gradcam_energy_ratio")}</div></div>
      <div class="xstat"><span class="lbl">Heatmap peak on the leaf</span><div class="pair">${pair("gradcam_pointing_acc")}</div></div>
      <div class="xstat"><span class="lbl">Accuracy with the background blacked out</span><div class="pair">${pair("accuracy_segmented")}</div></div>`;
  }

  $("#c-errors").textContent = `${e.test_errors} mistakes out of ${int(ds.split.test)} leaves`;
  $("#confusions").innerHTML = e.top_confusions.slice(0, 4).map((x) => `
    <li><span>${esc(x.true.disease)} <span class="arrow" aria-label="mistaken for">→</span> ${esc(x.pred.disease)}<br>
    <small>${esc(x.true.crop)}${x.true.crop !== x.pred.crop ? ` → ${esc(x.pred.crop)}` : ""}</small></span>
    <span class="n">${x.count}×</span></li>`).join("");

  const total = ds.split.train + ds.split.val + ds.split.test;
  $("#data-nums").innerHTML = `
    <div><span class="num">${int(ds.images)}</span><span class="lbl">lab photos</span></div>
    <div><span class="num">${ds.classes}</span><span class="lbl">classes</span></div>
    <div><span class="num">${ds.crops}</span><span class="lbl">crops</span></div>`;
  const parts = [["Train", ds.split.train, "var(--split-1)"], ["Validation", ds.split.val, "var(--split-2)"], ["Test", ds.split.test, "var(--split-3)"]];
  $("#splitbar").innerHTML = parts.map(([, n, col]) => `<span style="flex:${n};background:${col}"></span>`).join("");
  $("#split-legend").innerHTML = parts.map(([l, n, col]) => `<li><i style="background:${col}"></i>${l} <span class="num mono">${int(n)}</span> (${((n / total) * 100).toFixed(0)}%)</li>`).join("");
  const lk = s.leakage_check;
  $("#leak-note").textContent = `${int(ds.linked_images)} photos (${((ds.linked_images / ds.images) * 100).toFixed(1)}%) share a near-duplicate group and stay on one side of the split.`
    + (lk.random ? ` A plain random split scored ${pct(lk.random.accuracy, 2)} vs ${pct(lk.grouped.accuracy, 2)} grouped: no measurable leakage.` : "");

  drawCurves();
  drawClasses();
  tables(s);
}

/* ------------------------------------------------------------ chart helpers */
const tip = $("#tooltip");
function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 16, top = y + 16;
  if (left + r.width > innerWidth - 8) left = x - r.width - 16;
  if (top + r.height > innerHeight - 8) top = y - r.height - 16;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
function hideTip() { tip.hidden = true; }
function el(name, attrs = {}, parent) {
  const n = document.createElementNS(svgNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
const scale = (d0, d1, r0, r1) => { const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0); f.inv = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0); return f; };
const rowDot = (key) => `<i class="dot dot-${key}" aria-hidden="true"></i>`;

/* validation curves: line chart, 3 thin seed lines + a mean line per model */
function seriesFor(model) {
  const curves = state.stats.models[model].curves;
  const maxE = Math.max(...curves.map((c) => c.points.length));
  const mean = [];
  for (let i = 0; i < maxE; i++) {
    const vals = curves.map((c) => c.points[i]).filter(Boolean);
    if (vals.length === curves.length) {
      const v = vals.map((p) => p.val_macro_f1);
      mean.push({ epoch: i + 1, v: v.reduce((a, b) => a + b, 0) / v.length, lo: Math.min(...v), hi: Math.max(...v), phase: vals[0].phase });
    }
  }
  return { curves, mean };
}

function drawCurves() {
  const host = $("#chart-curves");
  const w = host.clientWidth || 600, h = 300;
  const m = { l: 40, r: 112, t: 28, b: 34 };
  host.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, height: h }, host);
  const data = { custom_cnn: seriesFor("custom_cnn"), efficientnet_b0: seriesFor("efficientnet_b0") };
  const maxE = Math.max(...Object.values(data).flatMap((d) => d.curves.map((c) => c.points.length)));
  const x = scale(1, maxE, m.l, w - m.r);
  const y = scale(50, 100, h - m.b, m.t);

  const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
  for (let v = 50; v <= 100; v += 10) {
    el("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v) }, grid);
    el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, axis).textContent = v;
  }
  for (let ep = 1; ep <= maxE; ep++) {
    if (ep === 1 || ep % (w < 520 ? 5 : 2) === 0) el("text", { x: x(ep), y: h - m.b + 18, "text-anchor": "middle" }, axis).textContent = ep;
  }
  el("text", { x: w - m.r, y: h - 4, "text-anchor": "end", class: "lbl" }, svg).textContent = "epoch";
  el("text", { x: m.l, y: m.t - 8, class: "lbl" }, svg).textContent = "val macro-F1 (%)";

  // phase marker: EfficientNet-B0 backbone unfrozen after epoch 2
  const ph = el("g", { class: "phase" }, svg);
  el("line", { x1: x(2.5), x2: x(2.5), y1: m.t, y2: h - m.b }, ph);
  el("text", { x: x(2.5) + 6, y: y(57) }, ph).textContent = "EfficientNet-B0 unfreezes";
  el("text", { x: x(2.5) + 6, y: y(57) + 14 }, ph).textContent = "its backbone";

  const line = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.epoch).toFixed(1)},${y(p.v * 100).toFixed(1)}`).join("");
  for (const [model, d] of Object.entries(data)) {
    const k = MODEL_KEY[model];
    d.curves.forEach((c) => el("path", { class: `series seed s-${k}`, d: line(c.points.map((p) => ({ epoch: p.epoch, v: p.val_macro_f1 }))) }, svg));
    el("path", { class: `series s-${k}`, d: line(d.mean) }, svg);
    const last = d.mean[d.mean.length - 1];
    el("text", { class: "end-lbl", x: x(last.epoch) + 8, y: y(last.v * 100) + (model === "efficientnet_b0" ? -8 : 14) }, svg).textContent = MODEL_LABEL[model];
  }

  // hover layer: crosshair + dots + tooltip
  const cross = el("line", { class: "crosshair", y1: m.t, y2: h - m.b, visibility: "hidden" }, svg);
  const dots = Object.keys(data).map((model) => el("circle", { r: 5, class: `hover-dot f-${MODEL_KEY[model]}`, visibility: "hidden" }, svg));
  const hit = el("rect", { class: "hit", x: m.l, y: m.t, width: w - m.l - m.r + 20, height: h - m.t - m.b }, svg);
  const move = (ev) => {
    const pt = svg.getBoundingClientRect();
    const sx = ((ev.clientX - pt.left) / pt.width) * w;
    const ep = Math.max(1, Math.min(maxE, Math.round(x.inv(sx))));
    cross.setAttribute("x1", x(ep)); cross.setAttribute("x2", x(ep)); cross.setAttribute("visibility", "visible");
    let rows = "";
    Object.entries(data).forEach(([model, d], i) => {
      const p = d.mean.find((q) => q.epoch === ep);
      if (p) {
        dots[i].setAttribute("cx", x(ep)); dots[i].setAttribute("cy", y(p.v * 100)); dots[i].setAttribute("visibility", "visible");
        rows += `<div class="t-row"><span>${rowDot(MODEL_KEY[model])}${MODEL_LABEL[model]}${model === "efficientnet_b0" ? ` (${p.phase})` : ""}</span><span>${(p.v * 100).toFixed(2)}%</span></div>`;
      } else {
        dots[i].setAttribute("visibility", "hidden");
        rows += `<div class="t-row"><span>${rowDot(MODEL_KEY[model])}${MODEL_LABEL[model]}</span><span>done</span></div>`;
      }
    });
    showTip(`<div class="t-title">Epoch ${ep} · mean of 3 seeds</div>${rows}`, ev.clientX, ev.clientY);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
}

/* per-class F1: dot plot, two dots per class joined by a connector */
function classRows() {
  const c = state.stats.models.custom_cnn.per_class;
  const e = Object.fromEntries(state.stats.models.efficientnet_b0.per_class.map((r) => [r.class, r]));
  return c.map((r) => ({ name: r.disease === "Healthy" ? `${r.crop}: healthy` : `${r.crop}: ${r.disease}`, c: r, e: e[r.class] }))
    .sort((a, b) => Math.min(a.c.f1, a.e.f1) - Math.min(b.c.f1, b.e.f1) || a.name.localeCompare(b.name));
}

function drawClasses() {
  const host = $("#chart-classes");
  const all = classRows();
  const rows = state.showAllClasses ? all : all.slice(0, 10);
  const w = host.clientWidth || 600;
  const rowH = 30;
  const labelW = Math.min(300, Math.max(150, w * 0.38));
  const m = { l: labelW, r: 24, t: 8, b: 30 };
  const h = m.t + rows.length * rowH + m.b;
  host.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, height: h }, host);
  const lo = Math.floor((Math.min(...all.map((r) => Math.min(r.c.f1, r.e.f1))) * 100 - 1) / 5) * 5;
  const x = scale(lo, 100, m.l, w - m.r);
  const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
  for (let v = lo; v <= 100; v += (100 - lo) > 10 ? 5 : 2.5) {
    el("line", { x1: x(v), x2: x(v), y1: m.t, y2: h - m.b }, grid);
    el("text", { x: x(v), y: h - m.b + 18, "text-anchor": "middle" }, axis).textContent = `${v}%`;
  }
  const maxChars = Math.floor(labelW / 6.6);
  rows.forEach((r, i) => {
    const cy = m.t + i * rowH + rowH / 2;
    const t = el("text", { class: "row-lbl", x: m.l - 12, y: cy + 4, "text-anchor": "end" }, svg);
    t.textContent = r.name.length > maxChars ? `${r.name.slice(0, maxChars - 1)}…` : r.name;
    const xc = x(r.c.f1 * 100), xe = x(r.e.f1 * 100);
    if (Math.abs(xc - xe) > 1) el("line", { class: "connector", x1: Math.min(xc, xe), x2: Math.max(xc, xe), y1: cy, y2: cy }, svg);
    el("circle", { class: "dotmark f-custom", cx: xc, cy: cy - (Math.abs(xc - xe) < 4 ? 3 : 0), r: 5 }, svg);
    el("circle", { class: "dotmark f-effb0", cx: xe, cy: cy + (Math.abs(xc - xe) < 4 ? 3 : 0), r: 5 }, svg);
    const hit = el("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: w, height: rowH }, svg);
    const show = (ev) => showTip(`<div class="t-title">${esc(r.name)}</div>
      <div class="t-row"><span>${rowDot("custom")}Custom CNN F1</span><span>${(r.c.f1 * 100).toFixed(1)}%</span></div>
      <div class="t-row"><span>${rowDot("effb0")}EfficientNet-B0 F1</span><span>${(r.e.f1 * 100).toFixed(1)}%</span></div>
      <div class="t-row"><span>Test leaves</span><span>${r.c.support}</span></div>`, ev.clientX, ev.clientY);
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerdown", show);
    hit.addEventListener("pointerleave", hideTip);
  });
}
$("#classes-toggle").addEventListener("click", (e) => {
  state.showAllClasses = !state.showAllClasses;
  e.currentTarget.textContent = state.showAllClasses ? "Show the 10 hardest" : "Show all 38 classes";
  e.currentTarget.setAttribute("aria-expanded", state.showAllClasses);
  drawClasses();
});

function tables(s) {
  const c = seriesFor("custom_cnn").mean, e = seriesFor("efficientnet_b0").mean;
  const n = Math.max(c.length, e.length);
  let t = "<table><thead><tr><th>Epoch</th><th>Custom CNN</th><th>EfficientNet-B0</th></tr></thead><tbody>";
  for (let i = 0; i < n; i++) t += `<tr><td>${i + 1}</td><td>${c[i] ? (c[i].v * 100).toFixed(2) : "–"}</td><td>${e[i] ? (e[i].v * 100).toFixed(2) : "–"}</td></tr>`;
  $("#table-curves").innerHTML = t + "</tbody></table>";
  t = "<table><thead><tr><th>Class</th><th>Custom CNN F1</th><th>EfficientNet-B0 F1</th><th>Test leaves</th></tr></thead><tbody>";
  for (const r of classRows()) t += `<tr><td>${esc(r.name)}</td><td>${(r.c.f1 * 100).toFixed(1)}</td><td>${(r.e.f1 * 100).toFixed(1)}</td><td>${r.c.support}</td></tr>`;
  $("#table-classes").innerHTML = t + "</tbody></table>";
}

/* redraw charts on resize */
let rz;
new ResizeObserver(() => { clearTimeout(rz); rz = setTimeout(() => { if (state.stats) { drawCurves(); drawClasses(); } }, 120); }).observe($("#report"));

loadModels();
loadSamples();
loadStats();
