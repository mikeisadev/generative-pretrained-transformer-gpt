// Test della logica pura delle azioni del tab Transformer (js/tabs/transformer/actions/):
// matematica (entropia, cross-entropy, gradiente sui logits, ruota), batch, dati del training,
// e l'esecuzione vera di un prompt con il motore JS sui pesi addestrati.
//   node --test webapp/tests/transformer-actions.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { loadEngine, mulberry32, sample } from "../js/engine/gpt.js";
import {
  crossEntropy,
  cumulative,
  entropy,
  layoutText,
  lerpInto,
  logSoftmaxRow,
  logitsGradient,
  makeBatch,
  quantizeSigned,
  rowNorms,
  segmentIndex,
  softmaxWithTemperature,
  visibleChar,
} from "../js/tabs/transformer/actions/math.js";
import { TRAINING_EXCERPT } from "../js/tabs/transformer/actions/excerpt.js";
import { DEFAULT_PROMPT, checkPrompt, runPrompt } from "../js/tabs/transformer/actions/promptRun.js";
import { batchLoss, groupOfStation, metricIndex, sampleBatch, summarizeByKind } from "../js/tabs/transformer/actions/trainingData.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const HAS_DATA = existsSync(`${DATA_DIR}weights.json`);
const fsLoaders = {
  loadJson: async (file) => JSON.parse(await readFile(file, "utf8")),
  loadBuffer: async (file) => {
    const bytes = await readFile(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  },
};
let engine = null;
const realEngine = async () => (engine ??= await loadEngine(DATA_DIR, fsLoaders));
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} ≠ ${b}`);

test("entropia: 0 per una certezza, ln V per la distribuzione uniforme", () => {
  close(entropy([1, 0, 0]), 0);
  close(entropy(new Array(65).fill(1 / 65)), Math.log(65), 1e-9);
});

test("log-softmax e softmax con temperatura sono coerenti e stabili con logits grandi", () => {
  const logits = Float32Array.from([1000, 1001, 999]);
  const lp = logSoftmaxRow(logits, 0, 3);
  close(lp.reduce((a, v) => a + Math.exp(v), 0), 1, 1e-6);
  const p1 = softmaxWithTemperature([1, 2, 3], 1);
  const pCold = softmaxWithTemperature([1, 2, 3], 0.5);
  const pHot = softmaxWithTemperature([1, 2, 3], 2);
  close(p1.reduce((a, v) => a + v, 0), 1);
  assert.ok(pCold[2] > p1[2] && p1[2] > pHot[2], "τ < 1 concentra, τ > 1 appiattisce");
});

test("cross-entropy: logits uniformi danno ln V; il gradiente sui logits è (p − onehot)/N", () => {
  const V = 5;
  const T = 3;
  const logits = new Float32Array(T * V); // tutti zero = uniforme
  const ce = crossEntropy(logits, [0, 3, 4], V);
  close(ce.mean, Math.log(V), 1e-6);
  close(ce.targetProb[1], 1 / V, 1e-6);
  const g = logitsGradient(Float32Array.from([2, 0, -1, 0.5, 0]), 0, V, 2, 4);
  close(g.reduce((a, v) => a + v, 0), 0, 1e-7); // somma zero: alza il target, abbassa gli altri
  assert.ok(g[2] < 0 && g[0] > 0);
  // confronto con la derivata numerica
  const f = (z) => -logSoftmaxRow(Float32Array.from(z), 0, V)[2] / 4;
  const z = [2, 0, -1, 0.5, 0];
  const h = 1e-3;
  for (let i = 0; i < V; i++) {
    const zp = [...z];
    const zm = [...z];
    zp[i] += h;
    zm[i] -= h;
    close(g[i], (f(zp) - f(zm)) / (2 * h), 1e-4);
  }
});

test("la ruota (cumulativa) indica lo stesso token di sample() del motore", () => {
  const probs = Float32Array.from([0.1, 0.0, 0.5, 0.25, 0.15]);
  const cum = cumulative(probs);
  close(cum.at(-1), 1, 1e-7);
  const rng = mulberry32(7);
  for (let k = 0; k < 200; k++) {
    const u = mulberry32(1000 + k)();
    const fromWheel = segmentIndex(cum, u);
    const fromEngine = sample(probs, () => u);
    assert.equal(fromWheel, fromEngine, `u = ${u}`);
  }
  assert.ok(rng() >= 0);
});

test("layoutText va a capo dopo \\n e a fine riga; visibleChar mostra spazio e a capo", () => {
  const cells = layoutText([..."ab\ncdef"], 3);
  assert.deepEqual(cells.map((c) => [c.col, c.row]), [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2]]);
  assert.equal(visibleChar(" "), "␣");
  assert.equal(visibleChar("\n"), "↵");
});

test("quantizzazione come gpt/export.py e piccole utility numeriche", () => {
  assert.equal(quantizeSigned(0, 2), 128);
  assert.equal(quantizeSigned(2, 2), 255);
  assert.equal(quantizeSigned(-2, 2), 1);
  assert.equal(quantizeSigned(-9, 2), 0);
  const out = new Float32Array(3);
  assert.deepEqual([...lerpInto(out, [0, 2, 4], [2, 2, 0], 0.5)], [1, 2, 2]);
  assert.deepEqual([...rowNorms(Float32Array.from([3, 4, 0, 0]), 2, 2)], [5, 0]);
});

test("makeBatch: y è x spostato di uno, come get_batch", () => {
  const ids = Array.from({ length: 100 }, (_, i) => i);
  const batch = makeBatch(ids, 8, 4, mulberry32(3));
  assert.equal(batch.length, 4);
  for (const w of batch) {
    assert.equal(w.x.length, 8);
    assert.deepEqual(w.y, w.x.map((v) => v + 1));
  }
  assert.throws(() => makeBatch([1, 2], 8, 1, Math.random));
});

test("riassunto delle metriche per famiglia di gruppi e mappa stazione → gruppo", () => {
  const byGroup = { "blocks.0.attention": [3, 1], "blocks.1.attention": [4, 1], lm_head: [2, 2] };
  const rss = summarizeByKind(byGroup, "rss");
  assert.deepEqual(rss.find((k) => k.id === "attention").values, [5, Math.SQRT2]);
  const mean = summarizeByKind(byGroup, "mean");
  assert.deepEqual(mean.find((k) => k.id === "attention").values, [3.5, 1]);
  assert.equal(groupOfStation("blocks.2.heads"), "blocks.2.attention");
  assert.equal(groupOfStation("blocks.0.ln2"), "blocks.0.layernorm");
  assert.equal(groupOfStation("blocks.3.ffwd"), "blocks.3.ffwd");
  assert.equal(groupOfStation("lm_head"), "lm_head");
  assert.equal(metricIndex([0, 25, 50, 75], 60), 2);
  assert.equal(metricIndex([0, 25, 50, 75], 9999), 3);
});

test("l'estratto di training è nel vocabolario e il prompt di default è valido", { skip: !HAS_DATA && "mancano i dati esportati" }, async () => {
  const e = await realEngine();
  assert.ok(TRAINING_EXCERPT.length > 1000);
  assert.equal(e.encode(TRAINING_EXCERPT).length, TRAINING_EXCERPT.length);
  assert.equal(checkPrompt(e, DEFAULT_PROMPT).ok, true);
  const bad = checkPrompt(e, "Però 1 è");
  assert.equal(bad.ok, false);
  assert.deepEqual(bad.unknown, ["1", "è", "ò"]);
  assert.equal(checkPrompt(e, "").ok, false);
});

test("runPrompt: i dati raccolti coincidono con il motore (stesso seed → stessi token)", { skip: !HAS_DATA && "mancano i dati esportati" }, async () => {
  const e = await realEngine();
  const options = { temperature: 0.8, tokens: 6, seed: 1337 };
  const run = runPrompt(e, DEFAULT_PROMPT, options);
  const expected = [...e.generate(e.encode(DEFAULT_PROMPT), 6, { temperature: 0.8, seed: 1337 })].map((s) => s.id);
  assert.deepEqual(run.steps.map((s) => s.id), expected);
  const s0 = run.steps[0];
  assert.equal(s0.T, 32, "il prompt di default è più lungo di block_size: il contesto è tagliato");
  assert.equal(s0.contextStart, DEFAULT_PROMPT.length - 32);
  // la ruota con il numero estratto indica proprio il token scelto
  assert.equal(segmentIndex(cumulative(s0.probs), s0.u), s0.id);
  // l'ultima riga della logit lens coincide con la distribuzione a temperatura 1
  const { probs } = e.nextTokenDistribution(s0.contextIds, { temperature: 1 });
  s0.lens.at(-1).probs.forEach((p, i) => close(p, probs[i], 1e-5));
  assert.equal(s0.attention.length, e.config.n_layer);
  assert.equal(s0.attention[0][0].length, 32 * 32);
  close(entropy(s0.probs), s0.entropy, 1e-9);
  assert.equal(run.generated.length, 6);
});

test("loss di un batch: pesi casuali ≈ ln 65, pesi addestrati molto meno", { skip: !HAS_DATA && "mancano i dati esportati" }, async () => {
  const e = await realEngine();
  const windows = sampleBatch(e.encode(TRAINING_EXCERPT), 32, 4, 42);
  const trained = batchLoss(e, windows).mean;
  await e.loadSnapshot(0);
  const initial = batchLoss(e, windows).mean;
  e.restoreWeights();
  assert.ok(Math.abs(initial - Math.log(65)) < 0.5, `loss iniziale ${initial}`);
  assert.ok(trained < 2.2, `loss finale ${trained}`);
});
