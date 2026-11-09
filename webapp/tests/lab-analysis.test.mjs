// Test della fase 2 del Laboratorio pesi (js/tabs/lab/analysis/): node --test webapp/tests/*.test.mjs
// Confronto A/B contro calcoli indipendenti in float64, ordinamento della tabella, modalità differenza,
// protocollo "finestre-contigue-v1" con il motore vero contro builtin.json (loss entro 1e-5, conteggi
// identici, testo greedy uguale), lettura dei benchmark (anche rotti), serie dei grafici dalla traccia
// vera e da tracce rotte, confronti pronti, prossimo carattere su un prompt, voci della ricerca.
import assert from "node:assert/strict";
import { existsSync, readdirSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { GPTEngine, expectedParameters } from "../js/engine/gpt.js";
import { loadBuiltinRun, runsFromFiles } from "../js/shared/weights/run.js";
import { BENCH_LIMITS, METRIC_ROWS, comparisonRows, comparisonWarnings, environmentText, readBenchmark, referenceAt } from "../js/tabs/lab/analysis/benchmarkModel.js";
import { COLUMNS, DEFAULT_SORT, compareAB, differenceMap, matchSummary, nextSort, onlyHalfRounding, sortRows } from "../js/tabs/lab/analysis/compareModel.js";
import { createComparison } from "../js/tabs/lab/analysis/comparison.js";
import { DIFF_LABEL, createDiffMode } from "../js/tabs/lab/analysis/diffMode.js";
import { EXAMPLE_CONTINUATIONS } from "../js/tabs/lab/analysis/content.js";
import { fixed, int, num, pct, seconds, shape, signedFixed, signedInt, signedNum, signedPoints } from "../js/tabs/lab/analysis/format.js";
import { PRESET_IDS, earlyStep, presets } from "../js/tabs/lab/analysis/presets.js";
import { barData, logitPairs, nextTokenProbe, sharedIds } from "../js/tabs/lab/analysis/probe.js";
import { ExcerptScore, GreedyRun, TOLERANCES, argmax, checkConsistency, contiguousWindows, firstDifference, greedyText, scoreText } from "../js/tabs/lab/analysis/protocol.js";
import { ANALYSIS_VIEWS, searchEntries } from "../js/tabs/lab/analysis/search.js";
import {
  abSeries,
  attentionEntropy,
  batchLossSeries,
  exampleProbSeries,
  finalStep,
  groupFamilies,
  groupMetricSeries,
  isConstant,
  lossSeries,
  metricSeries,
  movingAverage,
  pairs,
  snapshotSeries,
} from "../js/tabs/lab/analysis/series.js";
import { VIEW_IDS, resolveTarget } from "../js/tabs/lab/links.js";
import { searchIndex } from "../js/tabs/lab/search.js";
import { COURSE_RUN_ID, LabState } from "../js/tabs/lab/state.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const bytes = await readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const builtin = await loadBuiltinRun({ baseUrl: DATA_DIR, loadJson, loadBuffer });
const builtinJson = await loadJson(`${DATA_DIR}benchmark/builtin.json`);
const excerptText = await readFile(`${DATA_DIR}benchmark/val_excerpt.txt`, "utf8");

/** I run esportati dal Python in questo round (facoltativi: i test che li usano si saltano se mancano). */
const PY_RUNS = "/tmp/claude-0/-home-user-generative-pretrained-transformer-gpt/f3c7aefd-3f8d-51c4-8dc8-31c9c2f5b2c5/scratchpad/py-runs/";
const RUN_DIRS = { ffn3: `${PY_RUNS}runs/ffn3`, gelu: `${PY_RUNS}copia/runs/gpt-con-gelu` };

function fileLike(path, bytes) {
  return { name: path.split("/").pop(), size: bytes.byteLength, webkitRelativePath: path, text: async () => new TextDecoder().decode(bytes), arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}
async function folderFiles(dir) {
  const out = [];
  const base = dir.split("/").filter(Boolean).pop();
  const walk = async (abs, rel) => {
    for (const name of readdirSync(abs)) {
      const p = `${abs}/${name}`;
      if (statSync(p).isDirectory()) await walk(p, `${rel}/${name}`);
      else out.push(fileLike(`${rel}/${name}`, new Uint8Array(await readFile(p))));
    }
  };
  await walk(dir, base);
  return out;
}

/** Il confronto rifatto da capo in float64, senza la libreria: ‖A‖, ‖B‖, ‖A − B‖, ‖A − B‖/‖B‖, coseno, max |Δ|. */
function independent(a, b) {
  let aa = 0;
  let bb = 0;
  let dd = 0;
  let dot = 0;
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    aa += a[i] * a[i];
    bb += b[i] * b[i];
    dd += (a[i] - b[i]) ** 2;
    dot += a[i] * b[i];
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  // un vettore tutto zero non ha direzione: coseno indefinito (null, come la libreria); ‖B‖ = 0 → cambio relativo ∞
  return { normA: Math.sqrt(aa), normB: Math.sqrt(bb), diffNorm: Math.sqrt(dd), relChange: Math.sqrt(dd) / Math.sqrt(bb), cosine: aa > 0 && bb > 0 ? dot / Math.sqrt(aa * bb) : null, maxAbsDiff: max };
}

const close = (actual, expected, rel = 1e-9, label = "") => assert.ok(actual === expected || Math.abs(actual - expected) <= rel * Math.max(1, Math.abs(expected)), `${label}: ${actual} invece di ${expected}`);
const tensor = (name, shapeOf, values) => [name, { name, shape: shapeOf, numel: values.length, dtype: "float32", data: Float32Array.from(values) }];

// ---------------------------------------------------------------- confronto

test("confronto: pesi finali (A) contro step 0 (B) del GPT del corso, ogni misura contro un calcolo indipendente", async () => {
  const step0 = await builtin.loadSnapshot(0);
  const c = compareAB(builtin, step0, { trace: builtin.trace });
  assert.equal(c.rows.length, 94);
  assert.equal(c.matchedParams, 209729);
  assert.deepEqual([c.onlyA.length, c.onlyB.length, c.shapeMismatch.length], [0, 0, 0]);
  for (const r of c.rows) {
    const expected = independent(builtin.tensor(r.name).data, step0.tensors.get(r.name).data);
    for (const k of ["normA", "normB", "diffNorm", "relChange", "cosine", "maxAbsDiff"]) close(r[k], expected[k], 1e-9, `${r.name}.${k}`);
  }
  // i bias delle LayerNorm partono da zero: rispetto a B = step 0 il cambio relativo è infinito (lo si spiega
  // nella tabella e lo si mette in fondo: in cima i cambiamenti misurabili)
  assert.equal(c.rows.find((r) => r.name === "blocks.0.ln1.bias").relChange, Infinity);
  const sorted = sortRows(c.rows);
  assert.ok(Number.isFinite(sorted[0].relChange));
  assert.equal(sorted.at(-1).relChange, Infinity);
  assert.equal(sorted.filter((r) => r.relChange === Infinity).length, 9, "8 bias delle LayerNorm dei blocchi + ln_f.bias");
  // il totale è la norma di tutti i pesi in fila (come i gruppi di gpt/recording.py), non la media delle righe
  const all = independent(builtin.flat(), step0.flat);
  close(c.totals.relChange, all.relChange, 1e-9, "totale");
  close(c.totals.normA, all.normA, 1e-9, "‖A‖ totale");
  close(c.totals.cosine, all.cosine, 1e-9, "coseno totale");
  // i gruppi seguono trace.groups (16 per il corso) e sommano i pesi abbinati
  assert.deepEqual(c.groups.map((g) => g.group), builtin.trace.groups);
  assert.equal(c.groups.reduce((s, g) => s + g.params, 0), 209729);
  assert.equal(c.identical, false);
});

test("confronto: A − B ha il verso giusto (A è la scena, B il termine di paragone) e i nomi tornano al loro posto", () => {
  const A = new Map([tensor("w", [2, 2], [1, 2, 3, 4]), tensor("solo_a", [1], [5]), tensor("forma", [2], [1, 1]), tensor("module.x", [2], [3, 4])]);
  const B = new Map([tensor("w", [2, 2], [1, 2, 3, 2]), tensor("solo_b", [3], [1, 2, 3]), tensor("forma", [3], [1, 1, 1]), tensor("x", [2], [3, 0])]);
  const c = compareAB(A, B);
  const w = c.rows.find((r) => r.name === "w");
  assert.equal(w.diffNorm, 2);
  close(w.normA, Math.sqrt(30));
  close(w.normB, Math.sqrt(18));
  close(w.relChange, 2 / Math.sqrt(18)); // rispetto a B
  assert.equal(w.maxAbsDiff, 2);
  const x = c.rows.find((r) => r.name === "module.x"); // abbinato togliendo il prefisso: il nome è quello di A
  assert.equal(x.nameB, "x");
  assert.equal(x.relChange, 4 / 3);
  assert.deepEqual(c.onlyA.map((t) => t.name), ["solo_a"]);
  assert.deepEqual(c.onlyB.map((t) => t.name), ["solo_b"]);
  assert.deepEqual(c.shapeMismatch, [{ name: "forma", nameB: null, shapeA: [2], shapeB: [3] }]);
  const summary = matchSummary(c);
  assert.equal(summary.complete, false);
  assert.equal(summary.matched, 2);
  assert.ok(summary.lines.some((l) => l.includes("forma (2 in A, 3 in B)")));
  assert.ok(summary.lines.some((l) => l.startsWith("Solo in A (1): solo_a")));
  assert.ok(summary.lines.some((l) => l.startsWith("Solo in B (1): solo_b")));
  assert.ok(summary.lines.some((l) => l.startsWith("1 abbinato togliendo il prefisso") && l.includes("module.x in A ↔ x in B")));
  // la differenza per la scena: A − B, con gli array riusati la volta dopo
  const d = differenceMap(c, A, B);
  assert.deepEqual([...d.values.get("w")], [0, 0, 0, 2]);
  assert.deepEqual([...d.values.get("module.x")], [0, 4]);
  assert.equal(d.maxAbs, 4);
  assert.deepEqual(d.unmatched.sort(), ["forma", "solo_a"]);
  const again = differenceMap(c, A, B, d.values);
  assert.equal(again.values.get("w"), d.values.get("w"), "stesso array: niente allocazioni scorrendo gli step");
});

test("confronto: tensori identici, tutto zero e con NaN non rompono le misure", () => {
  const A = new Map([tensor("z", [3], [0, 0, 0]), tensor("same", [2], [1, -1]), tensor("nan", [2], [NaN, 2])]);
  const B = new Map([tensor("z", [3], [0, 0, 0]), tensor("same", [2], [1, -1]), tensor("nan", [2], [1, 2])]);
  const c = compareAB(A, B);
  const z = c.rows.find((r) => r.name === "z");
  assert.equal(z.relChange, 0);
  assert.equal(z.cosine, null); // nessuna direzione per un vettore nullo: va in fondo all'ordinamento
  assert.equal(c.rows.find((r) => r.name === "nan").skipped, 1);
  assert.equal(compareAB(new Map([tensor("same", [2], [1, -1])]), new Map([tensor("same", [2], [1, -1])])).identical, true);
});

test("ordinamento della tabella: i più cambiati in alto, pari merito per nome, valori mancanti in fondo, verso che si inverte", () => {
  const rows = [
    { name: "blocks.10.a", shape: [2, 2], relChange: 0.5, cosine: 0.9 },
    { name: "blocks.2.a", shape: [8], relChange: 0.5, cosine: null },
    { name: "lm_head.weight", shape: [65, 64], relChange: 1.8, cosine: 0.2 },
    { name: "ln_f.bias", shape: [64], relChange: NaN, cosine: 0.99 },
  ];
  assert.deepEqual(DEFAULT_SORT, { key: "relChange", dir: "desc" });
  assert.deepEqual(sortRows(rows).map((r) => r.name), ["lm_head.weight", "blocks.2.a", "blocks.10.a", "ln_f.bias"]);
  assert.deepEqual(sortRows(rows, "relChange", "asc").map((r) => r.name), ["blocks.2.a", "blocks.10.a", "lm_head.weight", "ln_f.bias"]);
  assert.deepEqual(sortRows(rows, "cosine", "asc").map((r) => r.name), ["lm_head.weight", "blocks.10.a", "ln_f.bias", "blocks.2.a"]);
  assert.deepEqual(sortRows(rows, "name", "asc").map((r) => r.name), ["blocks.2.a", "blocks.10.a", "lm_head.weight", "ln_f.bias"]);
  assert.deepEqual(sortRows(rows, "shape", "desc").map((r) => r.name), ["lm_head.weight", "ln_f.bias", "blocks.2.a", "blocks.10.a"]);
  assert.equal(rows[0].name, "blocks.10.a", "sortRows non cambia l'array originale");
  assert.deepEqual(nextSort(DEFAULT_SORT, "relChange"), { key: "relChange", dir: "asc" });
  assert.deepEqual(nextSort(DEFAULT_SORT, "cosine"), { key: "cosine", dir: "asc" });
  assert.deepEqual(nextSort(DEFAULT_SORT, "normA"), { key: "normA", dir: "desc" });
  assert.deepEqual(nextSort(DEFAULT_SORT, "boh"), DEFAULT_SORT);
  assert.deepEqual(COLUMNS.map((c) => c.id), ["name", "relChange", "cosine", "diffNorm", "maxAbsDiff", "normA", "normB", "shape"]);
  // i cambi relativi infiniti (B tutto zero) vanno in fondo come i mancanti: in cima i cambiamenti misurabili
  const withInf = [...rows, { name: "a.bias", shape: [4], relChange: Infinity, cosine: 0.5 }];
  assert.deepEqual(sortRows(withInf).map((r) => r.name), ["lm_head.weight", "blocks.2.a", "blocks.10.a", "a.bias", "ln_f.bias"]);
});

test("float16 contro float32: pesi finali e snapshot dello step 5.000 sono gli stessi pesi, a meno dell'arrotondamento", async () => {
  const last = await builtin.loadSnapshot(5000);
  const c = compareAB(builtin, last);
  assert.equal(onlyHalfRounding(c, "float32", "float16"), true);
  assert.ok(c.totals.relChange < 5e-4);
  const step0 = await builtin.loadSnapshot(0);
  assert.equal(onlyHalfRounding(compareAB(builtin, step0), "float32", "float16"), false);
  assert.equal(onlyHalfRounding(c, "float32", "float32"), false, "senza un lato float16 non è arrotondamento");
});

test("confronto con un'architettura diversa (FFN 3C esportata dal Python): si confronta ciò che si può e si dice il resto", { skip: !existsSync(RUN_DIRS.ffn3) && "manca il run esportato ffn3" }, async () => {
  const { runs } = await runsFromFiles(await folderFiles(RUN_DIRS.ffn3), { defaultVocab: builtin.vocab });
  const c = compareAB(runs[0], builtin);
  assert.equal(c.shapeMismatch.length, 12); // net.0.weight, net.0.bias, net.2.weight di 4 blocchi
  assert.ok(c.shapeMismatch.every((s) => /ffwd\.net\.[02]\./.test(s.name)));
  assert.deepEqual(c.shapeMismatch.find((s) => s.name === "blocks.0.ffwd.net.0.weight"), { name: "blocks.0.ffwd.net.0.weight", nameB: null, shapeA: [192, 64], shapeB: [256, 64] });
  assert.equal(c.rows.length, 82);
  assert.ok(c.rows.some((r) => r.name === "blocks.0.ffwd.net.2.bias"), "il bias da 64 ha la stessa forma: si confronta");
  const d = differenceMap(c, runs[0], builtin);
  assert.equal(d.unmatched.length, 12);
  assert.equal(d.values.size, 82);
});

// ---------------------------------------------------------------- benchmark nel browser

const course = builtin.createEngine();
const excerptIds = course.encode(excerptText);

test("protocollo finestre-contigue-v1: il motore JS sul GPT del corso dà i numeri di builtin.json (loss entro 1e-5, conteggi identici)", () => {
  const r = scoreText(course, excerptIds);
  const py = readBenchmark(builtinJson).excerpt;
  assert.equal(r.windows, py.counts.windows);
  assert.equal(r.predictions, py.counts.predictions);
  assert.equal(r.windows, 126);
  assert.equal(r.predictions, 4014);
  assert.ok(Math.abs(r.loss - py.loss) <= 1e-5, `loss ${r.loss} contro ${py.loss}`);
  assert.equal(r.top1, py.counts.top1);
  assert.equal(r.top5, py.counts.top5);
  assert.ok(Math.abs(r.entropy - py.entropy) <= 1e-5);
  assert.ok(Math.abs(r.bpc - py.bpc) <= 2e-5);
  assert.equal(r.bits.length, py.surprisalBits.length);
  assert.ok(r.bits.every((b, i) => Math.abs(b - py.surprisalBits[i]) <= TOLERANCES.python.bits));
  const check = checkConsistency(r, py, { greedy: greedyText(course, builtinJson.generation.prompt, builtinJson.generation.max_new_tokens), pythonGreedy: builtinJson.generation.greedy });
  assert.equal(check.verdict, "ok", JSON.stringify(check.rows.filter((x) => x.level !== "ok")));
  assert.equal(check.rows.find((x) => x.id === "greedy").delta, -1, "testo greedy identico a quello di PyTorch");
});

test("protocollo a pezzi = protocollo intero, e anche sugli snapshot (step 0 e 1.000, float16)", async () => {
  const s = new ExcerptScore(excerptIds, course.config, { bits: false });
  assert.equal(s.total, 126);
  let calls = 0;
  while (s.next(course)) calls++;
  assert.equal(calls, 125);
  assert.equal(s.finished, true);
  assert.equal(s.next(course), false);
  const whole = scoreText(course, excerptIds, { bits: false });
  assert.deepEqual({ ...s.result(), bits: null }, { ...whole, bits: null });
  const bench = readBenchmark(builtinJson);
  for (const step of [0, 1000]) {
    const engine = builtin.createEngine();
    engine.setWeights(builtin.engineFlat((await builtin.loadSnapshot(step)).tensors));
    const r = scoreText(engine, excerptIds, { bits: false });
    const ref = referenceAt(bench, step);
    assert.equal(ref.source, "snapshot");
    const check = checkConsistency(r, ref.excerpt);
    assert.equal(check.verdict, "ok", `step ${step}: ${JSON.stringify(check.rows)}`);
    assert.ok(Math.abs(r.loss - ref.excerpt.loss) <= 1e-5);
  }
});

test("finestre contigue, argmax, greedy un token alla volta, prima differenza tra testi", () => {
  assert.deepEqual(contiguousWindows(5, 2), [{ start: 0, len: 2 }, { start: 2, len: 2 }]);
  assert.deepEqual(contiguousWindows(6, 2), [{ start: 0, len: 2 }, { start: 2, len: 2 }, { start: 4, len: 1 }]);
  assert.deepEqual(contiguousWindows(1, 32), []);
  assert.equal(contiguousWindows(4015, 32).length, 126);
  assert.throws(() => contiguousWindows(10, 0), RangeError);
  assert.equal(argmax([1, 3, 3, 2]), 1, "a pari merito il primo indice, come PyTorch");
  const run = new GreedyRun(course.encode("ROMEO:\n"), 5);
  while (run.next(course));
  assert.equal(course.decode(run.ids()), builtinJson.generation.greedy.slice(0, 5));
  assert.equal(firstDifference("abc", "abc"), -1);
  assert.equal(firstDifference("abc", "abd"), 2);
  assert.equal(firstDifference("ab", "abc"), 2);
});

test("controllo di coerenza: ✓ entro 1e-5 e conteggi identici, ≈ entro le tolleranze del Python, ✗ oltre", () => {
  const python = { loss: 2, entropy: 1, bpc: 2 / Math.LN2, counts: { predictions: 10, windows: 2, top1: 5, top5: 8 } };
  const browser = (o = {}) => ({ predictions: 10, windows: 2, loss: 2, entropy: 1, bpc: 2 / Math.LN2, top1: 5, top5: 8, bits: null, ...o });
  assert.equal(checkConsistency(browser(), python).verdict, "ok");
  assert.equal(checkConsistency(browser({ loss: 2 + 9e-6 }), python).verdict, "ok");
  assert.equal(checkConsistency(browser({ loss: 2 + 5e-5, bpc: (2 + 5e-5) / Math.LN2 }), python).verdict, "near");
  assert.equal(checkConsistency(browser({ top1: 6 }), python).verdict, "near", "un pari merito rotto in modo diverso");
  assert.equal(checkConsistency(browser({ top5: 11 }), python).verdict, "bad");
  assert.equal(checkConsistency(browser({ loss: 2.1, bpc: 2.1 / Math.LN2 }), python).verdict, "bad");
  assert.equal(checkConsistency(browser({ windows: 3 }), python).verdict, "bad");
  assert.equal(checkConsistency(browser(), python, { greedy: "ciao", pythonGreedy: "ciaO" }).verdict, "bad");
  assert.equal(checkConsistency(browser(), python, { greedy: "ciao", pythonGreedy: "ciaO" }).rows.at(-1).delta, 3);
  assert.equal(checkConsistency(browser(), null).verdict, "none");
  // un excerpt senza conteggi: le righe dei conteggi sono solo da leggere
  const noCounts = checkConsistency(browser(), { loss: 2, entropy: 1, bpc: 2 / Math.LN2 });
  assert.equal(noCounts.verdict, "ok");
  assert.equal(noCounts.rows.find((r) => r.id === "top1").level, "info");
});

// ---------------------------------------------------------------- benchmark.json

test("benchmark.json del corso: metriche, estratto, testi, snapshot e riferimento letti tutti", () => {
  const b = readBenchmark(builtinJson);
  assert.equal(b.ok, true);
  assert.deepEqual(b.problems, []);
  assert.equal(b.name, "GPT del corso");
  assert.equal(b.protocol.id, "finestre-contigue-v1");
  assert.equal(b.protocol.blockSize, 32);
  assert.equal(b.metrics.loss, builtinJson.metrics.loss);
  assert.equal(b.metrics.params, 209729);
  assert.equal(b.counts.top1, 50887);
  assert.equal(b.excerpt.counts.predictions, 4014);
  assert.equal(b.excerpt.surprisalBits.length, 4014);
  assert.equal(b.generation.greedy, builtinJson.generation.greedy);
  assert.deepEqual(b.snapshots.map((s) => s.step), [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.equal(b.snapshots[0].excerpt.counts.top1, 39);
  assert.equal(b.reference.uniformLoss, builtinJson.reference.uniform_loss);
  assert.match(environmentText(b), /cpu · Linux · x86_64 · 1 thread/);
  const final = referenceAt(b, null);
  assert.equal(final.source, "final");
  assert.equal(final.greedy, builtinJson.generation.greedy);
  assert.equal(referenceAt(b, 500).metrics.loss, builtinJson.snapshots[4].metrics.loss);
  assert.equal(referenceAt(b, 750), null, "uno step interpolato non è mai stato misurato");
  assert.equal(referenceAt(null, null), null);
});

test("benchmark.json rotti: niente eccezioni, i campi sbagliati diventano null e finiscono nei problemi", () => {
  for (const raw of [null, undefined, 42, "ciao", [], [1, 2]]) {
    const b = readBenchmark(raw);
    assert.equal(b.ok, false);
    assert.ok(b.problems.length);
    assert.equal(b.metrics.loss, null);
    assert.equal(referenceAt(b, null), null);
  }
  const broken = JSON.parse(
    JSON.stringify({
      name: "x".repeat(5000),
      protocol: { id: "altro-protocollo", block_size: -3 },
      metrics: { loss: "1.8", perplexity: null, top1: 45.6, top5: 0.8, entropy: 1e400, predictions: 3.5 },
      counts: { top1: -1, top5: "8" },
      excerpt: { loss: 1.7, counts: { predictions: 10 }, surprisal_bits: [1, "due", 3] },
      generation: { greedy: 12, sampled: "ok", seed: 1.5 },
      snapshots: [{ step: 5, metrics: { loss: 2 } }, { step: 5, metrics: { loss: 3 } }, { step: "dieci" }, null],
      notes: ["vero", 3, { a: 1 }],
      environment: "pc",
    }),
  );
  broken.__proto__ = { loss: 0 }; // una chiave del prototipo è solo un nome
  const b = readBenchmark(broken);
  assert.equal(b.ok, true, "la top-5 e l'estratto bastano per mostrare qualcosa");
  assert.equal(b.name.length, BENCH_LIMITS.name);
  assert.equal(b.metrics.loss, null);
  assert.equal(b.metrics.top1, null, "45,6 non è una quota tra 0 e 1");
  assert.equal(b.metrics.top5, 0.8);
  assert.equal(b.metrics.entropy, null);
  assert.equal(b.metrics.predictions, null);
  assert.equal(b.protocol.blockSize, null);
  assert.equal(b.counts, null, "conteggi tutti rotti: come se mancassero");
  assert.equal(b.excerpt.surprisalBits, null);
  assert.equal(b.generation.greedy, null);
  assert.equal(b.generation.sampled, "ok");
  assert.equal(b.generation.seed, null);
  assert.deepEqual(b.snapshots.map((s) => s.step), [5]);
  assert.deepEqual(b.notes, ["vero"]);
  assert.equal(b.environment, null);
  for (const p of ["metrics.loss", "metrics.top1", "surprisal_bits", "snapshots[1]", "protocollo altro-protocollo"]) assert.ok(b.problems.some((x) => x.includes(p)), `manca il problema ${p}: ${b.problems.join(" | ")}`);
  assert.equal(readBenchmark({ metrics: {}, notes: "x" }).ok, false);
  // un benchmark scritto a mano con le metriche al primo livello (come parseBenchmark)
  assert.equal(readBenchmark({ loss: 1.5, top1: 0.4 }).metrics.loss, 1.5);
});

test("confronto dei benchmark A/B: A − B, colonna «meglio» con il verso giusto, nessun verdetto per entropia e velocità", () => {
  const a = referenceAt(readBenchmark({ metrics: { loss: 1.7, perplexity: 5.5, bpc: 2.45, top1: 0.5, top5: 0.82, entropy: 1.6, params: 209729, tokens_per_s: 500 } }));
  const b = referenceAt(readBenchmark(builtinJson));
  const rows = comparisonRows(a, b);
  assert.deepEqual(rows.map((r) => r.key), METRIC_ROWS.map((r) => r.key));
  const row = (k) => rows.find((r) => r.key === k);
  close(row("loss").delta, 1.7 - builtinJson.metrics.loss);
  assert.equal(row("loss").better, "A");
  assert.equal(row("top1").better, "A");
  assert.ok(row("top1").delta > 0);
  assert.equal(row("entropy").better, null);
  assert.equal(row("tokens_per_s").better, null);
  assert.equal(row("tokens_per_s").machine, true);
  assert.equal(row("params").better, null);
  assert.equal(row("params").delta, 0);
  const worse = comparisonRows(b, a);
  assert.equal(worse.find((r) => r.key === "loss").better, "B");
  assert.equal(comparisonRows(b, b).find((r) => r.key === "loss").better, "=");
  const missing = comparisonRows(a, null);
  assert.ok(missing.every((r) => r.delta === null && r.better === null));
});

test("avvisi di confrontabilità (come comparison_warnings): protocollo, testo, contesto, vocabolario, stessi pesi, codice, macchina", () => {
  const base = readBenchmark(builtinJson);
  assert.deepEqual(comparisonWarnings(base, base).map((w) => w.text), ["stessi pesi, bit per bit"]);
  const other = readBenchmark({ ...builtinJson, protocol: { id: "v2", block_size: 64 }, data: { sha256: "x" }, model: { ...builtinJson.model, vocab_size: 80, weights_sha256: "aaa", code_sha256: "bbb" }, environment: { ...builtinJson.environment, machine: "arm64" } });
  const texts = comparisonWarnings(base, other).map((w) => `${w.level}: ${w.text}`);
  assert.equal(texts.length, 6, texts.join("\n"));
  assert.ok(texts[0].startsWith("attenzione: protocolli diversi"));
  assert.ok(texts.some((t) => t.startsWith("nota: contesti diversi (block_size 32 contro 64)")));
  assert.ok(texts.some((t) => t.startsWith("attenzione: vocabolari diversi (65 contro 80")));
  assert.ok(texts.some((t) => t.startsWith("nota: sorgenti del modello diversi")));
  assert.ok(texts.some((t) => t.startsWith("nota: velocità misurate in condizioni diverse")));
  assert.deepEqual(comparisonWarnings(base, readBenchmark(null)), []);
});

test("benchmark.json di un run esportato dal Python (GELU, non eseguibile dal motore): letto come quello del corso", { skip: !existsSync(RUN_DIRS.gelu) && "manca il run esportato gpt-con-gelu" }, async () => {
  const { runs } = await runsFromFiles(await folderFiles(RUN_DIRS.gelu), { defaultVocab: builtin.vocab });
  const run = runs[0];
  assert.equal(run.compatibleWithEngine, false);
  const b = readBenchmark(run.benchmark.raw);
  assert.equal(b.ok, true);
  assert.deepEqual(b.problems, []);
  assert.deepEqual(b.snapshots.map((s) => s.step), [0, 50, 100, 200]);
  const rows = comparisonRows(referenceAt(b), referenceAt(readBenchmark(builtinJson)));
  assert.equal(rows.find((r) => r.key === "loss").better, "B", "200 step di prova contro 5.000: vince il corso");
});

// ---------------------------------------------------------------- serie dei grafici

test("serie dei grafici dalla traccia vera del corso: loss, batch, lr costante, gradiente, update ratio, gruppi, snapshot", () => {
  const trace = builtin.trace;
  const loss = lossSeries(trace);
  assert.equal(loss.train.length, 51);
  assert.equal(loss.val.length, 51);
  assert.deepEqual(loss.val[0], [0, trace.evals.val[0]]);
  assert.deepEqual(loss.val.at(-1), [4999, trace.evals.val.at(-1)]);
  const batch = batchLossSeries(trace);
  assert.equal(batch.length, 5000);
  assert.deepEqual(batch[10], [10, trace.train_loss[10]]);
  const lr = metricSeries(trace, "lr");
  assert.equal(lr.length, 201);
  assert.equal(isConstant(lr), true, "nessuno scheduler nel training del corso");
  assert.equal(isConstant(metricSeries(trace, "grad_norm")), false);
  assert.equal(metricSeries(trace, "update_ratio", { positive: true }).length, 201);
  assert.deepEqual(metricSeries(trace, "toString"), [], "un nome del prototipo non è una metrica");
  assert.equal(finalStep(trace), 5000);

  const families = groupFamilies(trace);
  assert.deepEqual(families.map((f) => f.id), ["attention", "ffwd", "layernorm", "outside"]);
  assert.deepEqual(families[0].members.map((m) => m.label), ["blocco 0", "blocco 1", "blocco 2", "blocco 3"]);
  assert.deepEqual(families[3].members.map((m) => m.group), ["token_embedding", "position_embedding", "ln_f", "lm_head"]);
  const std = groupMetricSeries(trace, "weight_std_by_group", "token_embedding");
  assert.equal(std.length, 201);
  assert.ok(Math.abs(std[0][1] - 1) < 0.01, "nn.Embedding parte da N(0, 1): deviazione standard ≈ 1");
  assert.deepEqual(groupMetricSeries(trace, "weight_std_by_group", "__proto__"), []);

  assert.deepEqual(snapshotSeries(trace, "loss").map((p) => p[0]), [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.equal(snapshotSeries(trace, "top5")[4][1], trace.snapshots[4].top5);
  const heat = attentionEntropy(trace);
  assert.equal(heat.rows, 16);
  assert.equal(heat.cols, 11);
  assert.equal(heat.rowLabels[5], "b1h1");
  assert.equal(heat.colLabels[4], "500");
  close(heat.values[5 * 11 + 4], trace.snapshots[4].attention_entropy[1][1], 1e-6, "entropia b1·h1 allo step 500");
  // la frase d'esempio del training e il suo carattere giusto (verificato nel testo vero)
  const example = trace.recorder.example_prompt;
  assert.equal(EXAMPLE_CONTINUATIONS[example].next, "o");
  const p = exampleProbSeries(trace, builtin.vocab.indexOf("o"));
  assert.equal(p.length, 11);
  assert.ok(p.at(-1)[1] > p[0][1] * 10, "il training impara la «o» di «through»");
  assert.deepEqual(abSeries(loss.train, [], "train").map((s) => s.name), ["A · train"]);
  assert.deepEqual(abSeries(loss.train, loss.val).map((s) => s.name), ["A", "B"]);
});

test("serie da tracce rotte o malevole: solo numeri finiti e step interi, lunghezze diverse tagliate, niente eccezioni", () => {
  for (const t of [null, undefined, 3, "x", [], { evals: "no", metrics: [], snapshots: {} }]) {
    assert.deepEqual(lossSeries(t), { train: [], val: [] });
    assert.deepEqual(batchLossSeries(t), []);
    assert.deepEqual(metricSeries(t, "lr"), []);
    assert.deepEqual(groupFamilies(t), []);
    assert.deepEqual(snapshotSeries(t, "loss"), []);
    assert.equal(attentionEntropy(t), null);
    assert.equal(finalStep(t), null);
  }
  const t = {
    evals: { steps: [0, 1.5, -1, 3, "4", 5], val: [1, 2, 3, NaN, 5] },
    train_loss: [1, null, "2", 3, Infinity],
    metrics: { steps: [0, 25, 50], lr: [0.1, -0.1, 0], update_ratio: [1e-3, 0, -1], weight_std_by_group: { a: [1] } },
    groups: ["blocks.0.attention", "transformer.h.0.attn", "transformer.h.11.attn", "wte", 7, "x".repeat(500)],
    snapshots: [{ step: 2, loss: 1, attention_entropy: [[1, "x"], "no"] }, { step: 1, loss: "2" }, { step: -1, loss: 3 }],
  };
  assert.deepEqual(lossSeries(t).val, [[0, 1]]);
  assert.deepEqual(batchLossSeries(t), [[0, 1], [3, 3]]);
  assert.deepEqual(metricSeries(t, "update_ratio", { positive: true }), [[0, 1e-3]]);
  assert.deepEqual(groupMetricSeries(t, "weight_std_by_group", "a"), [[0, 1]]);
  const fam = groupFamilies(t);
  assert.deepEqual(fam.map((f) => f.id), ["attention", "transformer.h.*.attn", "outside"]);
  assert.deepEqual(fam[1].members.map((m) => m.label), ["blocco 0", "blocco 11"]);
  assert.deepEqual(snapshotSeries(t, "loss"), [[2, 1]]);
  const heat = attentionEntropy(t);
  assert.deepEqual([heat.rows, heat.cols], [4, 1], "2 blocchi × 2 head, un solo snapshot con l'entropia");
  assert.equal(heat.values[0], 1);
  assert.ok(Number.isNaN(heat.values[1]), "un valore rotto resta un buco, non uno zero");
  assert.ok(Number.isNaN(heat.values[2]), "un blocco rotto resta un buco");
  assert.deepEqual(pairs([1, 2], [3]), [[1, 3]]);
  // un trace enorme: niente stack degli argomenti traboccato (nessun Math.max(...lista)), heatmap limitata a 64 × 64
  const huge = { snapshots: Array.from({ length: 150_000 }, (_, i) => ({ step: i, loss: 1 })), metrics: { steps: Array.from({ length: 150_000 }, (_, i) => i), lr: new Array(150_000).fill(0.1) } };
  assert.equal(finalStep(huge), 149_999);
  assert.equal(metricSeries(huge, "lr").length, 150_000);
  const wide = attentionEntropy({ snapshots: [{ step: 0, attention_entropy: Array.from({ length: 200 }, () => new Array(200_000).fill(1)) }] });
  assert.deepEqual([wide.rows, wide.cols], [64 * 64, 1]);
  // famiglie con troppi gruppi: al più 8 serie (primi e ultimi blocchi), il resto contato
  const many = groupFamilies({ groups: Array.from({ length: 12 }, (_, i) => `blocks.${i}.ffwd`) });
  assert.equal(many[0].members.length, 8);
  assert.equal(many[0].truncated, 4);
  assert.deepEqual(many[0].members.slice(-2).map((m) => m.label), ["blocco 10", "blocco 11"]);
});

test("media mobile dei bit di sorpresa: finestra a sinistra, non finiti ignorati", () => {
  assert.deepEqual(movingAverage([1, 2, 3, 4], 2), [[0, 1], [1, 1.5], [2, 2.5], [3, 3.5]]);
  assert.deepEqual(movingAverage([1, NaN, 3], 2), [[0, 1], [1, 1], [2, 3]]);
  assert.deepEqual(movingAverage([], 4), []);
  const bits = readBenchmark(builtinJson).excerpt.surprisalBits;
  const avg = movingAverage(bits, 4014);
  close(avg.at(-1)[1], builtinJson.excerpt.bpc, 1e-3, "la media di tutti i bit è il bit per carattere");
});

// ---------------------------------------------------------------- confronti pronti, prompt, formati, ricerca

test("confronti pronti: training (finali contro step 0), fasi (5.000 contro 500), il tuo run contro il corso", () => {
  const stepsOf = (id) => (id === "c" ? [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000] : id === "u2" ? [0, 50, 100, 200] : []);
  const entries = [{ id: "c", name: "GPT del corso", origin: "builtin" }, { id: "u", name: "FFN 3C", origin: "upload" }];
  const list = presets({ runA: "c", courseId: "c", stepsOf, entries });
  assert.deepEqual(list.map((p) => p.id), ["training", "fasi", "corso"]);
  assert.deepEqual([list[0].a, list[0].b], [{ run: "c", step: null }, { run: "c", step: 0 }]);
  assert.deepEqual([list[1].a, list[1].b], [{ run: "c", step: 5000 }, { run: "c", step: 500 }]);
  assert.deepEqual([list[2].a, list[2].b], [{ run: "u", step: null }, { run: "c", step: null }]);
  // un run senza storia come A: la storia del training è quella del corso
  const fromUpload = presets({ runA: "u", courseId: "c", stepsOf, entries });
  assert.equal(fromUpload[0].a.run, "c");
  assert.equal(fromUpload.at(-1).a.run, "u");
  // un run con la sua storia (200 step): "fasi" sparisce se coincide con "training"
  const own = presets({ runA: "u2", courseId: "c", stepsOf, entries: [...entries, { id: "u2", name: "200 step", origin: "upload" }] });
  assert.equal(own[0].a.run, "u2");
  assert.equal(earlyStep([0, 50, 100, 200]), 50);
  assert.ok(!own.some((p) => p.id === "fasi" && p.b.step === 0));
  assert.deepEqual(presets({ runA: "c", courseId: "c", stepsOf: () => [], entries: entries.slice(0, 1) }), []);
  assert.deepEqual([...PRESET_IDS], ["training", "fasi", "corso"]);
});

test("il prossimo carattere su un prompt: probabilità del motore, stessi token per A e B, caratteri sconosciuti e prompt lunghi", async () => {
  const example = builtin.trace.recorder.example_prompt;
  const a = nextTokenProbe(course, example);
  assert.equal(a.ok, true);
  assert.equal(a.top[0].id, argmax(a.probs));
  // lo stesso numero che il registratore del training ha scritto con PyTorch allo step 5.000 (pesi finali)
  const recorded = builtin.trace.snapshots.at(-1).example_next_token_probs;
  for (const t of a.top) close(t.prob, recorded[t.id], 2e-5, `p(${t.char})`);
  close([...a.probs].reduce((s, p) => s + p, 0), 1, 1e-5);
  assert.ok(a.top.every((t, i) => i === 0 || t.prob <= a.top[i - 1].prob));
  const early = builtin.createEngine();
  early.setWeights(builtin.engineFlat((await builtin.loadSnapshot(0)).tensors));
  const b = nextTokenProbe(early, example);
  const ids = sharedIds(a, b, 5);
  assert.ok(ids.length >= 5 && ids.length <= 10);
  assert.equal(ids[0], a.top[0].id, "nell'ordine delle probabilità di A");
  const bars = barData(b, ids);
  assert.equal(bars.labels.length, ids.length);
  assert.equal(bars.values[0], b.probs[ids[0]]);
  assert.equal(logitPairs(a, b).length, 65);
  assert.equal(logitPairs(a, { ok: false }).length, 0);
  const unknown = nextTokenProbe(course, "città");
  assert.equal(unknown.ok, false);
  assert.match(unknown.reason, /«à»/);
  assert.equal(nextTokenProbe(course, "").ok, false);
  const long = nextTokenProbe(course, "a".repeat(100));
  assert.equal(long.cropped, true);
  assert.equal(long.used, 32);
  assert.deepEqual([...long.probs], [...nextTokenProbe(course, "a".repeat(32)).probs], "oltre block_size contano solo gli ultimi caratteri");
});

test("numeri all'italiana: cifre significative, percentuali, differenze con il segno, notazione scientifica leggibile", () => {
  assert.equal(num(0.898123), "0,898");
  assert.equal(num(-21.3456), "−21,3");
  assert.equal(num(209729), "209.729");
  assert.equal(num(999.96), "1.000");
  assert.equal(num(3.2e-7), "3,2 · 10⁻⁷");
  assert.equal(num(0), "0");
  assert.equal(num(null), "—");
  assert.equal(num(NaN), "NaN");
  assert.equal(num(-Infinity), "−∞");
  assert.equal(pct(0.473), "47,3%");
  assert.equal(pct(1.853), "185%");
  assert.equal(pct(0.0031), "0,31%");
  assert.equal(pct(0.000031), "0,003%");
  assert.equal(pct(0), "0%");
  assert.equal(pct(Infinity), "∞");
  assert.equal(fixed(1.814528), "1,8145");
  assert.equal(fixed(-0.00001), "0,0000", "niente «−0»");
  assert.equal(signedFixed(-0.6687, 4), "−0,6687");
  assert.equal(signedFixed(0.12, 2), "+0,12");
  assert.equal(signedFixed(0.00001, 4), "0,0000");
  assert.equal(signedNum(4.2e-8), "+4,2 · 10⁻⁸");
  assert.equal(signedInt(-1), "−1");
  assert.equal(signedInt(0), "0");
  assert.equal(signedPoints(0.1685), "+16,85 pp");
  assert.equal(signedPoints(-0.0001), "−0,01 pp");
  assert.equal(signedPoints(0.00001), "0 pp");
  assert.equal(int(111539), "111.539");
  assert.equal(shape([16, 64]), "16 × 64");
  assert.equal(seconds(0.82), "820 ms");
  assert.equal(seconds(12.34), "12,3 s");
});

test("ricerca e link della fase 2: ogni vista ha una voce, si risolve con links.js e non duplica le altre", () => {
  const ids = new Set(searchEntries.map((e) => e.id));
  assert.deepEqual([...ids].sort(), Object.keys(ANALYSIS_VIEWS).sort());
  for (const e of searchEntries) {
    assert.equal(e.kind, "view");
    assert.ok(e.title && e.text && e.keywords.length, e.id);
    assert.deepEqual(resolveTarget(e), { type: "view", id: e.id });
    assert.ok(Object.hasOwn(VIEW_IDS, e.id));
  }
  const keys = searchIndex.map((e) => `${e.kind}:${e.id}`);
  assert.equal(new Set(keys).size, keys.length, "nessun doppione con le voci delle fondamenta");
  for (const id of PRESET_IDS) assert.ok(ids.has(`confronto:${id}`));
});

// Un motore sintetico piccolo (pesi casuali con seme) per provare il protocollo su un'altra config.
test("protocollo su un'altra config (block_size 4, vocabolario di 11): finestre e conteggi coerenti", () => {
  const config = { vocab_size: 11, block_size: 4, n_embd: 8, n_head: 2, n_layer: 1, dropout: 0 };
  let offset = 0;
  const params = expectedParameters(config).map(({ name, shape: s }) => {
    const numel = s.reduce((x, y) => x * y, 1);
    const p = { name, shape: s, offset, numel };
    offset += numel;
    return p;
  });
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.4;
  const weights = Float32Array.from({ length: offset }, rnd);
  const vocab = "abcdefghij ".split("");
  const engine = GPTEngine.fromBuffers({ format_version: 1, dtype: "float32", byte_order: "little", file: "w.bin", config, vocab, total_params: offset, params }, weights.buffer);
  const ids = engine.encode("abcabcabcab hijij");
  const r = scoreText(engine, ids);
  assert.equal(r.windows, contiguousWindows(ids.length, 4).length);
  assert.equal(r.predictions, ids.length - 1);
  assert.ok(r.top1 <= r.top5 && r.top5 <= r.predictions);
  close(r.perplexity, Math.exp(r.loss));
  close(r.bpc, r.loss / Math.LN2);
  assert.ok(r.entropy > 0 && r.entropy <= Math.log(11) + 1e-9);
});

// ---------------------------------------------------------------- confronto vivo e modalità differenza

/** Una scena finta con le stesse regole di LabScene.showValues / matrixOf / highlightCells. */
function fakeScene(tensors) {
  const matrices = new Map([...tensors].map(([name, t]) => [name, { weights: t.data, data: t.data, custom: false }]));
  const scene = {
    customValues: new Set(),
    valuesLabel: "",
    veils: new Map(),
    calls: 0,
    matrixOf: (name) => (matrices.has(name) ? { data: matrices.get(name).data } : null),
    setWeights(map) {
      for (const [name, t] of map) {
        const e = matrices.get(name);
        if (!e) continue;
        e.weights = t.data ?? t;
        if (!e.custom) e.data = e.weights;
      }
    },
    showValues(map, { label = "" } = {}) {
      scene.calls++;
      const shown = [];
      const missing = [];
      for (const [name, values] of map) {
        const e = matrices.get(name);
        if (!e) {
          missing.push(name);
          continue;
        }
        if (values === null) {
          e.custom = false;
          e.data = e.weights;
          scene.customValues.delete(name);
          continue;
        }
        e.custom = true;
        e.data = values;
        scene.customValues.add(name);
        shown.push(name);
      }
      scene.valuesLabel = scene.customValues.size ? label : "";
      return { shown, missing, label: scene.valuesLabel };
    },
    highlightCells(name, cells) {
      if (cells === null) scene.veils.delete(name);
      else scene.veils.set(name, cells);
      return true;
    },
  };
  return scene;
}

const fsLoaders = { baseUrl: DATA_DIR, loadJson, loadBuffer };
const nextReady = (cmp, predicate = () => true) =>
  new Promise((resolve) => {
    const off = cmp.on((s) => {
      if (s.status === "ready" && predicate(s)) {
        off();
        resolve(s);
      }
    });
  });

test("confronto vivo con lo stato vero: niente B = niente confronto, poi A e B, step che cambiano, richieste superate", async () => {
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store: null });
  await state.init();
  const cmp = createComparison(state);
  assert.equal((await cmp.refresh()).status, "none");
  let ready = nextReady(cmp);
  await state.setRunB(COURSE_RUN_ID, { step: 0 });
  let s = await ready;
  assert.equal(s.result.rows.length, 94);
  assert.deepEqual([s.a.step, s.b.step, s.a.dtype, s.b.dtype], [null, 0, "float32", "float16"]);
  assert.equal(s.b.name, "GPT del corso");
  // tre step di fila: vince l'ultimo, nessun risultato vecchio dopo
  ready = nextReady(cmp, (x) => x.a.step === 2000);
  void state.setStep(500);
  void state.setStep(1000);
  await state.setStep(2000);
  s = await ready;
  assert.equal(s.a.step, 2000);
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(cmp.current.a.step, 2000, "nessuna richiesta superata annuncia i suoi pesi dopo");
  await state.setRunB(null);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(cmp.current.status, "none");
  cmp.dispose();
});

test("modalità differenza: A − B nella scena, resta accesa durante i ricalcoli, si spegne senza B e lascia i valori degli altri", async () => {
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store: null });
  await state.init();
  const scene = fakeScene(builtin.tensors);
  let action = false;
  const diff = createDiffMode({ scene, state, actionRunning: () => action });
  const cmp = createComparison(state);
  cmp.on((x) => diff.update(x));
  assert.equal(diff.setEnabled(true), false, "senza un confronto non si accende");
  const ready = nextReady(cmp);
  await state.setRunB(COURSE_RUN_ID, { step: 0 });
  const s = await ready;
  assert.equal(diff.setEnabled(true), true);
  assert.equal(diff.enabled, true);
  assert.equal(state.mode, "differenza");
  assert.equal(state.modeLabel, DIFF_LABEL);
  assert.equal(scene.customValues.size, 94);
  const name = "blocks.0.sa.heads.0.key.weight";
  const shown = scene.matrixOf(name).data;
  const a = builtin.tensor(name).data;
  const b = s.b.view.tensors.get(name).data;
  assert.ok(shown.every((v, i) => Math.abs(v - (a[i] - b[i])) < 1e-7), "la scena mostra A − B");
  close(diff.legend.maxAbs, 0.9008457660675049, 1e-6, "colore pieno della scala comune");
  // un ricalcolo in corso non spegne niente (era un difetto: la tabella e la scena lampeggiavano)
  diff.update({ status: "busy" });
  assert.equal(diff.enabled, true);
  // scala dei pesi: stessi valori, scala diversa
  diff.setScale("weights");
  assert.equal(diff.scale, "weights");
  assert.equal(scene.matrixOf(name).data, shown, "stessi array riusati");
  // un altro modulo mostra i suoi valori su un tensore e cambia modalità: ci spegniamo, i suoi valori restano
  const theirs = new Float32Array(a.length).fill(7);
  scene.showValues(new Map([[name, theirs]]), { label: "attivazioni" });
  state.setMode("valori", { label: "attivazioni" });
  assert.equal(diff.enabled, false);
  assert.equal(diff.state.reason, "other");
  assert.equal(scene.matrixOf(name).data, theirs, "i valori dell'altro modulo non si toccano");
  assert.equal(scene.valuesLabel, "attivazioni", "neanche la sua etichetta");
  assert.equal(scene.customValues.size, 1, "le nostre 93 matrici sono tornate ai pesi");
  scene.showValues(new Map([[name, null]]));
  state.setMode("pesi");
  // riaccesa, poi parte un'azione: torna ai pesi veri, con il motivo giusto
  diff.setEnabled(true);
  assert.equal(scene.customValues.size, 94);
  action = true;
  diff.stopForAction();
  assert.equal(diff.enabled, false);
  assert.equal(diff.state.reason, "action");
  assert.equal(scene.customValues.size, 0);
  assert.equal(state.mode, "pesi");
  action = false;
  // senza B si spegne da sola
  diff.setEnabled(true);
  await state.setRunB(null);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(diff.enabled, false);
  assert.equal(scene.customValues.size, 0);
  assert.equal(state.mode, "pesi");
  diff.dispose();
  cmp.dispose();
});

test("modalità differenza con un'architettura diversa: i tensori senza gemello si velano e il velo se ne va spegnendo", () => {
  const A = new Map([tensor("w", [2, 2], [1, 2, 3, 4]), tensor("ffn", [3], [1, 1, 1]), tensor("solo_a", [1], [5])]);
  const B = new Map([tensor("w", [2, 2], [0, 2, 3, 4]), tensor("ffn", [4], [1, 1, 1, 1])]);
  const listeners = new Map();
  const state = { mode: "pesi", modeLabel: "", on: (type, fn) => (listeners.set(type, fn), () => listeners.delete(type)), setMode(mode, { label = "" } = {}) { this.mode = mode; this.modeLabel = label; listeners.get("mode")?.({ mode }); } };
  const scene = fakeScene(A);
  const diff = createDiffMode({ scene, state });
  const result = compareAB(A, B);
  diff.update({ status: "ready", result, a: { view: { tensors: A } }, b: { view: { tensors: B } } });
  diff.setEnabled(true);
  assert.deepEqual([...scene.customValues], ["w"]);
  assert.deepEqual([...scene.veils.keys()].sort(), ["ffn", "solo_a"]);
  assert.equal(diff.legend.veiled, 2);
  assert.deepEqual([...scene.matrixOf("w").data], [1, 0, 0, 0]);
  diff.setEnabled(false);
  assert.equal(scene.veils.size, 0);
  assert.equal(scene.customValues.size, 0);
  assert.equal(state.mode, "pesi");
  diff.dispose();
});

