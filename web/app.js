// Nawa web demo. Plain JavaScript, no framework and no build step.
//
// The browser only draws and displays. Every number shown comes from the C++ engine:
// the drawing is sent as raw grayscale pixels to POST /api/predict, which runs the same
// preprocessing and model as `nawa predict`, and returns the network's internals.

"use strict";

const API = {
  model: "/api/model",
  predict: "/api/predict",
  neuron: (id, variant) => `/api/neuron/${id}?variant=${variant}`,
};
const UNSURE_BELOW = 0.7;  // show the "not sure" warning below this confidence

// ---------------------------------------------------------------------------
// Elements and state
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const pad = $("pad");
const padCtx = pad.getContext("2d", { willReadFrequently: true });
const svg = $("net");
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  model: null,          // /api/model response
  variant: "fp32",
  last: null,           // last /api/predict response
  hasInk: false,        // anything drawn since the last clear?
  inFlight: false,      // a prediction request is running
  queued: false,        // another prediction was requested meanwhile
  neuronCache: new Map(),
  selectedNeuron: null,
};

// ---------------------------------------------------------------------------
// Drawing (Pointer Events: mouse, pen and touch with one code path)
// ---------------------------------------------------------------------------

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function clearPad() {
  padCtx.fillStyle = cssVar("--pad-bg") || "#fff";
  padCtx.fillRect(0, 0, pad.width, pad.height);
  state.hasInk = false;
}

function padPoint(event) {
  // The canvas may be displayed smaller than 280 px (narrow screens): map to canvas pixels.
  const rect = pad.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * pad.width) / rect.width,
    y: ((event.clientY - rect.top) * pad.height) / rect.height,
  };
}

let drawing = false;
let lastPoint = null;

function strokeTo(point) {
  padCtx.strokeStyle = cssVar("--pad-ink") || "#111";
  padCtx.lineWidth = 20;          // thick, like MNIST strokes after scaling to 28 x 28
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(lastPoint.x, lastPoint.y);
  padCtx.lineTo(point.x, point.y);
  padCtx.stroke();
  lastPoint = point;
  state.hasInk = true;
}

pad.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  pad.setPointerCapture(event.pointerId);  // keep receiving moves outside the canvas
  drawing = true;
  lastPoint = padPoint(event);
  strokeTo(lastPoint);  // a single tap leaves a dot
});
pad.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  // Coalesced events give the full-resolution path of fast strokes.
  const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
  for (const e of events.length ? events : [event]) strokeTo(padPoint(e));
});
function endStroke() {
  if (!drawing) return;
  drawing = false;
  if ($("live-toggle").checked) requestPrediction();
}
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

$("clear-btn").addEventListener("click", () => {
  clearPad();
  resetResult();
});
$("predict-btn").addEventListener("click", () => requestPrediction());

// Hand-drawn sample digits as stroke paths in pad coordinates (280 x 280), for trying the demo
// without drawing, and for opening the page with #example (e.g. for screenshots).
const EXAMPLES = [
  [[[80, 72], [204, 68], [188, 104], [150, 164], [128, 222]]],                           // 7
  [[[86, 88], [120, 62], [168, 64], [190, 96], [172, 128], [132, 146], [176, 160], [194, 196],
    [168, 226], [118, 228], [86, 206]]],                                                  // 3
  [[[168, 58], [92, 164], [178, 164]], [[152, 110], [152, 230]]],                         // 4
  [[[82, 96], [110, 64], [160, 60], [186, 92], [176, 132], [96, 222], [200, 220]]],       // 2
  [[[140, 58], [100, 76], [82, 130], [92, 190], [136, 222], [180, 196], [194, 136], [180, 80],
    [140, 58]]],                                                                          // 0
];
let exampleIndex = 0;

function drawExample() {
  clearPad();
  const strokes = EXAMPLES[exampleIndex++ % EXAMPLES.length];
  for (const stroke of strokes) {
    lastPoint = { x: stroke[0][0], y: stroke[0][1] };
    for (const [x, y] of stroke) strokeTo({ x, y });
  }
  requestPrediction();
}
$("example-btn").addEventListener("click", drawExample);

// ---------------------------------------------------------------------------
// Talking to the server
// ---------------------------------------------------------------------------

function showError(message) {
  $("error-text").textContent = message;
  $("error-banner").hidden = false;
  const status = $("status");
  status.textContent = "Server unreachable";
  status.className = "status down";
}

function hideError() {
  $("error-banner").hidden = true;
}

async function fetchJson(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    throw new Error("Can't reach the Nawa server. Is `nawa serve` running?");
  }
  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    body = null;
  }
  if (!response.ok) {
    const error = new Error((body && body.error) || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

async function loadModelInfo() {
  try {
    state.model = await fetchJson(API.model);
  } catch (err) {
    showError(err.message);
    return;
  }
  hideError();
  const m = state.model;
  const status = $("status");
  status.textContent = `Connected · ${m.kernel.toUpperCase()} · ${m.threads} threads`;
  status.className = "status ok";

  const int8 = $("int8-radio");
  int8.disabled = !m.variants.includes("int8");
  int8.parentElement.title = int8.disabled
    ? "Start the server with --int8 models/mnist_mlp_int8.nawa to enable"
    : "8-bit weights, 4x smaller";

  renderModelInfo();
  buildNetwork(m.hidden_size, m.classes);
}

function readPixels() {
  // Grayscale 0..255 per pixel (the pad is opaque, so alpha is always 255).
  const { data } = padCtx.getImageData(0, 0, pad.width, pad.height);
  const pixels = new Array(pad.width * pad.height);
  for (let i = 0, p = 0; i < pixels.length; i++, p += 4) {
    pixels[i] = Math.round(0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]);
  }
  return pixels;
}

// Only one request at a time; if the user keeps drawing, the latest drawing is sent next.
async function requestPrediction() {
  if (!state.model) return;
  if (!state.hasInk) {
    resetResult("Draw a digit first");
    return;
  }
  if (state.inFlight) {
    state.queued = true;
    return;
  }
  state.inFlight = true;
  $("predict-btn").disabled = true;
  const started = performance.now();
  try {
    const result = await fetchJson(API.predict, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        width: pad.width, height: pad.height, pixels: readPixels(), variant: state.variant,
      }),
    });
    hideError();
    result.roundTripMs = performance.now() - started;
    state.last = result;
    render(result);
  } catch (err) {
    if (err.status === 422) {
      resetResult("Draw a digit first");  // empty canvas
    } else {
      showError(err.message);
    }
  } finally {
    state.inFlight = false;
    $("predict-btn").disabled = false;
    if (state.queued) {
      state.queued = false;
      requestPrediction();
    }
  }
}

for (const radio of document.querySelectorAll('input[name="variant"]')) {
  radio.addEventListener("change", () => {
    state.variant = radio.value;
    state.neuronCache.clear();  // int8 weight maps differ (dequantized)
    if (state.hasInk) requestPrediction();
    if (state.selectedNeuron !== null) showNeuron(state.selectedNeuron);
  });
}
$("retry-btn").addEventListener("click", loadModelInfo);

// ---------------------------------------------------------------------------
// Rendering: result panel
// ---------------------------------------------------------------------------

function formatMicros(us) {
  return us >= 1000 ? `${(us / 1000).toFixed(2)} ms` : `${us.toFixed(1)} µs`;
}

function buildBars(classes) {
  const bars = $("bars");
  bars.replaceChildren();
  for (let d = 0; d < classes; d++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.id = `bar-${d}`;
    row.innerHTML =
      `<span class="bar-digit">${d}</span>` +
      `<span class="bar-track"><span class="bar-fill"></span></span>` +
      `<span class="bar-value">–</span>`;
    bars.appendChild(row);
  }
}

function resetResult(message = "Draw a digit to start") {
  $("digit").textContent = "–";
  $("confidence").textContent = message;
  $("unsure").hidden = true;
  for (const id of ["active", "t-pre", "t-fwd", "t-total", "t-rtt"]) $(id).textContent = "–";
  if (state.model) {
    for (let d = 0; d < state.model.classes; d++) {
      const row = $(`bar-${d}`);
      row.classList.remove("winner");
      row.querySelector(".bar-fill").style.width = "0%";
      row.querySelector(".bar-value").textContent = "–";
    }
    renderNetwork(null);
  }
  const seen = $("seen").getContext("2d");
  seen.fillStyle = "#000";
  seen.fillRect(0, 0, 28, 28);
}

function renderModelInfo() {
  const m = state.model;
  const fp32 = m.models.fp32;
  const rows = [
    ["Layers", fp32.plan.join(" → ")],
    ["Parameters", fp32.parameters.toLocaleString("en-US")],
    ["Input", `${m.input.image.join(" × ")} = ${m.input.shape[0]} values`],
    // The values are float32 on the server; round them for display (0.1307, not 0.13070000708).
    ["Normalization", `(x · ${m.input.pixel_scale.toPrecision(3)} − ${Number(m.input.mean[0].toPrecision(4))}) / ${Number(m.input.std[0].toPrecision(4))}`],
    ["GEMM kernel", m.kernel],
    ["Threads", m.threads_note],
    ["Variants", m.variants.join(", ")],
  ];
  const dl = $("model-info");
  dl.replaceChildren();
  for (const [term, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.append(dt, dd);
  }
}

function drawSeen(input28) {
  // The 28 x 28 input as the model received it (white digit on black, like MNIST).
  const ctx = $("seen").getContext("2d");
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = input28[i];
    image.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);
}

function render(r) {
  const digit = $("digit");
  digit.textContent = r.prediction;
  digit.classList.remove("bump");
  void digit.offsetWidth;  // restart the small "bump" animation
  digit.classList.add("bump");
  setTimeout(() => digit.classList.remove("bump"), 200);

  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  $("confidence").textContent = `${pct(r.confidence)} confident`;
  $("unsure").hidden = r.confidence >= UNSURE_BELOW;

  r.probabilities.forEach((p, d) => {
    const row = $(`bar-${d}`);
    row.classList.toggle("winner", d === r.prediction);
    row.querySelector(".bar-fill").style.width = `${(p * 100).toFixed(2)}%`;
    row.querySelector(".bar-value").textContent = pct(p);
  });

  $("active").textContent = `${r.active_neurons} / ${r.hidden_size}`;
  $("t-pre").textContent = formatMicros(r.timing_us.preprocess);
  $("t-fwd").textContent = formatMicros(r.timing_us.forward);
  $("t-total").textContent = formatMicros(r.timing_us.total);
  $("t-rtt").textContent = `${r.roundTripMs.toFixed(1)} ms`;

  drawSeen(r.input28);
  const note = [];
  if (r.preprocess.inverted) note.push("inverted (dark on light → light on dark)");
  const b = r.preprocess.box;
  note.push(`cropped to ${b.width} × ${b.height}, scaled to 20 px, centered by mass`);
  $("prep-note").textContent = note.join("; ") + ".";

  renderNetwork(r);
  if (state.selectedNeuron !== null) showNeuron(state.selectedNeuron);
}

// ---------------------------------------------------------------------------
// Rendering: the network (SVG, built once, updated per prediction)
// ---------------------------------------------------------------------------

const layout = {
  input: { x: 16, y: 140, cell: 6.5 },           // 28 x 28 heatmap
  hidden: { x: 272, y: 34, cols: 8, dx: 17, dy: 30, r: 6.5 },
  output: { x: 575, y: 52, dy: 46, r: 15 },
};

function hiddenPos(i) {
  const h = layout.hidden;
  return { x: h.x + (i % h.cols) * h.dx, y: h.y + Math.floor(i / h.cols) * h.dy };
}
function outputPos(d) {
  return { x: layout.output.x, y: layout.output.y + d * layout.output.dy };
}

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function buildNetwork(hiddenSize, classes) {
  buildBars(classes);
  const inputGroup = $("input-map");
  const hiddenGroup = $("hidden");
  const outputGroup = $("outputs");
  const labels = $("labels");
  for (const g of [inputGroup, hiddenGroup, outputGroup, labels, $("links")]) g.replaceChildren();

  const { x, y, cell } = layout.input;
  for (let i = 0; i < 784; i++) {
    el("rect", {
      class: "input-cell", x: x + (i % 28) * cell, y: y + Math.floor(i / 28) * cell,
      width: cell + 0.3, height: cell + 0.3, fill: "#000",
    }, inputGroup);
  }
  for (let i = 0; i < hiddenSize; i++) {
    const p = hiddenPos(i);
    const node = el("circle", { class: "hidden-node", cx: p.x, cy: p.y, r: layout.hidden.r }, hiddenGroup);
    node.dataset.id = i;
    const title = el("title", {}, node);
    title.textContent = `Hidden neuron ${i}`;
    node.addEventListener("pointerenter", () => showNeuron(i));
    node.addEventListener("click", () => showNeuron(i));  // touch screens: tap
  }
  for (let d = 0; d < classes; d++) {
    const p = outputPos(d);
    el("circle", { class: "output-node", id: `out-${d}`, cx: p.x, cy: p.y, r: layout.output.r }, outputGroup);
    const label = el("text", { class: "output-label", x: p.x, y: p.y + 4.5, "text-anchor": "middle" }, labels);
    label.textContent = d;
    const prob = el("text", { class: "output-prob", id: `out-prob-${d}`, x: p.x + 22, y: p.y + 4 }, labels);
    prob.textContent = "";
  }
  const title = (text, xPos, anchor = "middle") => {
    const t = el("text", { class: "col-title", x: xPos, y: 16, "text-anchor": anchor }, labels);
    t.textContent = text;
  };
  title("Input 28 × 28 (784)", x + 14 * cell);
  title("Hidden 128 · ReLU", layout.hidden.x + 3.5 * layout.hidden.dx);
  title("Output 10", layout.output.x);
  renderNetwork(null);
}

function mix(value) {
  // Accent color with opacity proportional to value (0..1), via fill-opacity.
  return Math.max(0.12, Math.min(1, value));
}

function renderNetwork(r) {
  const accent = cssVar("--accent");
  const cells = $("input-map").children;
  for (let i = 0; i < cells.length; i++) {
    const v = r ? r.input28[i] : 0;
    cells[i].setAttribute("fill", `rgb(${v},${v},${v})`);
  }

  const hidden = $("hidden").children;
  const maxAct = r ? Math.max(...r.hidden, 1e-6) : 1;
  for (let i = 0; i < hidden.length; i++) {
    const a = r ? r.hidden[i] : 0;
    const node = hidden[i];
    if (a > 0) {
      node.style.fill = accent;
      node.style.fillOpacity = mix(a / maxAct);
      node.setAttribute("r", layout.hidden.r * (0.8 + 0.4 * (a / maxAct)));
    } else {
      node.style.fill = "";           // falls back to the "off" color from CSS
      node.style.fillOpacity = "";
      node.setAttribute("r", layout.hidden.r * 0.8);
    }
    node.querySelector("title").textContent = r
      ? `Hidden neuron ${i}: activation ${a.toFixed(3)}${a > 0 ? "" : " (off: ReLU)"}`
      : `Hidden neuron ${i}`;
  }

  if (!state.model) return;
  for (let d = 0; d < state.model.classes; d++) {
    const node = $(`out-${d}`);
    const p = r ? r.probabilities[d] : 0;
    node.classList.toggle("winner", !!r && d === r.prediction);
    node.style.fill = p > 0.001 ? accent : "";
    node.style.fillOpacity = p > 0.001 ? mix(p) : "";
    $(`out-prob-${d}`).textContent = r ? `${(p * 100).toFixed(p >= 0.1 ? 0 : 1)}%` : "";
  }

  // The strongest hidden -> winner connections: thickness and opacity ~ |contribution|.
  const links = $("links");
  links.replaceChildren();
  if (!r) return;
  const target = outputPos(r.prediction);
  const maxAbs = Math.max(...r.contributions.map((c) => Math.abs(c.contribution)), 1e-6);
  for (const c of r.contributions) {
    const from = hiddenPos(c.neuron);
    const strength = Math.abs(c.contribution) / maxAbs;
    const midX = (from.x + target.x) / 2;
    const link = el("path", {
      class: `link ${c.contribution >= 0 ? "pos" : "neg"}`,
      d: `M ${from.x} ${from.y} C ${midX} ${from.y}, ${midX} ${target.y}, ${target.x - layout.output.r} ${target.y}`,
      "stroke-width": (0.75 + 4.5 * strength).toFixed(2),
    }, links);
    link.style.opacity = 0;
    const title = el("title", {}, link);
    title.textContent =
      `neuron ${c.neuron}: activation ${c.activation.toFixed(2)} × weight ${c.weight.toFixed(3)} = ${c.contribution.toFixed(3)}`;
    requestAnimationFrame(() => { link.style.opacity = (0.2 + 0.8 * strength).toFixed(2); });
  }
}

// ---------------------------------------------------------------------------
// Hidden neuron details: weight map from GET /api/neuron/<id>
// ---------------------------------------------------------------------------

function divergingColor(value, maxAbs) {
  // Negative -> blue, 0 -> neutral, positive -> red.
  const t = Math.max(-1, Math.min(1, value / maxAbs));
  const neutral = [245, 245, 245];
  const target = t >= 0 ? [217, 72, 15] : [28, 126, 214];
  const k = Math.abs(t);
  return neutral.map((n, i) => Math.round(n + (target[i] - n) * k));
}

async function showNeuron(id) {
  state.selectedNeuron = id;
  for (const node of $("hidden").children) {
    node.classList.toggle("selected", Number(node.dataset.id) === id);
  }
  const key = `${state.variant}:${id}`;
  let info = state.neuronCache.get(key);
  if (!info) {
    try {
      info = await fetchJson(API.neuron(id, state.variant));
    } catch (err) {
      showError(err.message);
      return;
    }
    state.neuronCache.set(key, info);
  }
  if (state.selectedNeuron !== id) return;  // the user moved on while we were loading

  const maxAbs = Math.max(Math.abs(info.min), Math.abs(info.max), 1e-6);
  const ctx = $("neuron-map").getContext("2d");
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    image.data.set([...divergingColor(info.weights[i], maxAbs), 255], i * 4);
  }
  ctx.putImageData(image, 0, 0);

  const r = state.last;
  const lines = [];
  if (r) {
    const a = r.hidden[id];
    lines.push(`Activation: <b>${a.toFixed(3)}</b>${a > 0 ? "" : " (switched off by ReLU)"}`);
    lines.push(`Weight to “${r.prediction}”: ${info.outgoing[r.prediction].toFixed(3)} · contribution ${(a * info.outgoing[r.prediction]).toFixed(3)}`);
  }
  lines.push(`Bias: ${info.bias.toFixed(3)} · weights ${info.min.toFixed(3)} … ${info.max.toFixed(3)}`);
  $("neuron-title").textContent = `Hidden neuron ${id}`;
  $("neuron-stats").innerHTML = lines.join("<br>");
  $("neuron-card").hidden = false;
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

clearPad();
buildBars(10);
resetResult();
loadModelInfo().then(() => {
  if (location.hash === "#example" && state.model) drawExample();
});

// Keep the pad's colors in sync if the system switches between light and dark mode.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  clearPad();
  resetResult();
});
