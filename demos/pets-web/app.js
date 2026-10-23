// In-browser inference for the pet breed demo: ONNX Runtime Web, no server.
// Mirrors demos/pets-gradcam/inference.py: square centre crop, ImageNet normalisation,
// temperature-scaled softmax, and the class-activation map blended over the crop.
"use strict";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const $ = (id) => document.getElementById(id);
const RAMP = [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]];
let session = null;
let cfg = null;
let busy = false;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function prettify(file) {
  const hard = file.endsWith("_hard.jpg");
  const name = file.replace(/(_hard)?\.jpg$/, "").split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  return hard ? `${name} (a known mistake)` : name;
}

// Resize the shorter side to `size` and centre-crop a square, as in training.
function cropToCanvas(img) {
  const size = cfg.image_size;
  const canvas = $("crop");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const side = Math.min(w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(img, (w - side) / 2, (h - side) / 2, side, side, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

function toTensor(pixels) {
  const size = cfg.image_size;
  const plane = size * size;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      data[c * plane + i] = (pixels.data[4 * i + c] / 255 - cfg.mean[c]) / cfg.std[c];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

function softmax(logits, temperature) {
  const scaled = Array.from(logits, (v) => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Bilinear upsampling with half-pixel centres (PyTorch's align_corners=False).
function upsample(map, h, w, size) {
  const out = new Float32Array(size * size);
  const coord = (dst, n) => Math.min(Math.max((dst + 0.5) * (n / size) - 0.5, 0), n - 1);
  for (let y = 0; y < size; y++) {
    const sy = coord(y, h), y0 = Math.floor(sy), y1 = Math.min(y0 + 1, h - 1), fy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = coord(x, w), x0 = Math.floor(sx), x1 = Math.min(x0 + 1, w - 1), fx = sx - x0;
      const top = map[y0 * w + x0] * (1 - fx) + map[y0 * w + x1] * fx;
      const bottom = map[y1 * w + x0] * (1 - fx) + map[y1 * w + x1] * fx;
      out[y * size + x] = top * (1 - fy) + bottom * fy;
    }
  }
  return out;
}

function drawHeat(pixels, cams, dims, k) {
  const [, , h, w] = dims;
  const size = cfg.image_size;
  const map = Array.from(cams.subarray(k * h * w, (k + 1) * h * w), (v) => Math.max(v, 0));
  const heat = upsample(map, h, w, size);
  const sorted = Float32Array.from(heat).sort();
  const scale = sorted[Math.floor(0.99 * (sorted.length - 1))] + 1e-12;  // 99th percentile
  const out = new ImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.min(Math.max(heat[i] / scale, 0), 1);
    const pos = v * (RAMP.length - 1);
    const lo = Math.min(Math.floor(pos), RAMP.length - 2);
    const frac = pos - lo;
    const alpha = 0.25 + 0.5 * v;
    for (let c = 0; c < 3; c++) {
      const colour = RAMP[lo][c] * (1 - frac) + RAMP[lo + 1][c] * frac;
      out.data[4 * i + c] = pixels.data[4 * i + c] * (1 - alpha) + colour * alpha;
    }
    out.data[4 * i + 3] = 255;
  }
  $("heat").getContext("2d").putImageData(out, 0, 0);
}

function showProbabilities(probs) {
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 5);
  const [pTop, kTop] = order[0];
  const name = cfg.classes[kTop];
  const verdict = $("verdict");
  verdict.classList.toggle("unsure", pTop < 0.5);
  verdict.innerHTML = pTop >= 0.5
    ? `<strong>${name}</strong> with ${(100 * pTop).toFixed(0)}% probability.`
    : `Unsure: the best guess, <strong>${name}</strong>, has only ${(100 * pTop).toFixed(0)}%.`;
  const bars = $("bars");
  bars.replaceChildren(...order.map(([p, k]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
    li.querySelector(".name").textContent = cfg.classes[k];
    li.querySelector(".pct").textContent = `${(100 * p).toFixed(1)}%`;
    li.querySelector(".fill").style.width = `${(100 * p).toFixed(1)}%`;
    return li;
  }));
  return kTop;
}

async function classify(img) {
  if (!session || busy) return;
  busy = true;
  try {
    const pixels = cropToCanvas(img);
    const start = performance.now();
    const outputs = await session.run({ image: toTensor(pixels) });
    const ms = performance.now() - start;
    const probs = softmax(outputs.logits.data, cfg.temperature);
    const k = showProbabilities(probs);
    drawHeat(pixels, outputs.cams.data, outputs.cams.dims, k);
    setStatus(`Computed in ${ms.toFixed(0)} ms in your browser.`);
  } catch (err) {
    console.error(err);
    setStatus(`Could not classify this image: ${err.message || err}`, true);
  } finally {
    busy = false;
  }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("the file is not an image the browser can read"));
    img.src = src;
  });
}

async function classifyFile(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    await classify(await loadImage(url));
  } catch (err) {
    setStatus(`Could not read this file: ${err.message}`, true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function wireInputs() {
  const drop = $("drop");
  $("file").addEventListener("change", (e) => classifyFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    classifyFile(e.dataTransfer.files[0]);
  });
}

async function showExamples() {
  const names = await (await fetch("examples.json")).json();
  $("examples").replaceChildren(...names.map((file) => {
    const button = document.createElement("button");
    button.type = "button";
    button.title = prettify(file);
    button.setAttribute("aria-label", `Classify the example: ${prettify(file)}`);
    const img = document.createElement("img");
    img.src = `examples/${file}`;
    img.alt = prettify(file);
    button.append(img);
    button.addEventListener("click", async () => classify(await loadImage(img.src)));
    return button;
  }));
  return names;
}

async function main() {
  wireInputs();
  try {
    cfg = await (await fetch("config.json")).json();
    $("facts").insertAdjacentHTML("beforeend",
      `<li>${(100 * cfg.test_accuracy).toFixed(1)}% on ${cfg.test_images.toLocaleString("en")} test images</li>` +
      `<li>Temperature-scaled probabilities (T = ${cfg.temperature.toFixed(2)})</li>` +
      `<li>Runs in your browser</li>`);
    const [names] = await Promise.all([
      showExamples(),
      ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] }).then((s) => { session = s; }),
    ]);
    setStatus("Model ready. Pick a photo or a test image.");
    if (names.length) classify(await loadImage(`examples/${names[0]}`));
  } catch (err) {
    console.error(err);
    setStatus(`The model could not be loaded: ${err.message || err}`, true);
  }
}

main();
