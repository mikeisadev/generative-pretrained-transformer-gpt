// Test della logica pura delle viste del tab Transformer (LOD, "Pesi", "Rete neurale"):
//   node --test webapp/tests/*.test.mjs
// Niente DOM e niente three.js: si testano conteggi, layout dei neuroni, archi e spiegazioni,
// sia su un modello sintetico piccolo sia (se presenti) sui pesi veri di data/.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { GPTEngine, expectedParameters, loadEngine } from "../js/engine/gpt.js";
import { buildEdges, buildNeuronLayout, explainNeuron, neuronValues, parameterBreakdown, slotPositions, softmax } from "../js/tabs/transformer/views/neuronModel.js";
import { componentOf, componentTotals, countParams, partsSummary, shapeText } from "../js/tabs/transformer/views/paramGroups.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const HAS_DATA = existsSync(`${DATA_DIR}weights.json`) && existsSync(`${DATA_DIR}weights.f32.bin`);
const fsLoaders = {
  loadJson: async (file) => JSON.parse(await readFile(file, "utf8")),
  loadBuffer: async (file) => {
    const bytes = await readFile(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  },
};

const SMALL = { vocab_size: 11, block_size: 8, n_embd: 16, n_head: 4, n_layer: 2, dropout: 0 };
const SMALL_VOCAB = ["\n", " ", "a", "b", "c", "d", "e", "f", "g", "h", "!"];

function smallEngine(seed = 3) {
  let offset = 0;
  const params = expectedParameters(SMALL).map(({ name, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const entry = { name, shape, offset, numel };
    offset += numel;
    return entry;
  });
  const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "w.bin", config: SMALL, vocab: SMALL_VOCAB, total_params: offset, params, training: null };
  let s = seed;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
  const data = Float32Array.from({ length: offset }, () => rnd() * 0.5);
  return GPTEngine.fromBuffers(manifest, data.buffer);
}
const paramOf = (engine) => (name) => engine.paramViews.get(name).data;

test("componentOf: ogni parametro ha un componente e i totali tornano", () => {
  const params = expectedParameters(SMALL).map(({ name, shape }) => ({ name, shape, numel: shape.reduce((a, b) => a * b, 1) }));
  const total = params.reduce((a, p) => a + p.numel, 0);
  const totals = componentTotals(params);
  assert.equal([...totals.values()].reduce((a, c) => a + c.count, 0), total);
  assert.equal(componentOf("blocks.1.sa.heads.3.key.weight"), "blocks.1.heads");
  assert.equal(componentOf("blocks.0.ffwd.net.2.bias"), "blocks.0.ffwd");
  assert.equal(componentOf("qualcosa.di.strano"), null);
  assert.equal(partsSummary(params, SMALL.n_layer).reduce((a, p) => a + p.count, 0), total);
  assert.equal(countParams(params, "lm_head."), SMALL.vocab_size * SMALL.n_embd + SMALL.vocab_size);
  assert.equal(shapeText([64]), "(64,)");
  assert.equal(shapeText([65, 64]), "(65, 64)");
});

test("slotPositions: estremi, ordine e spazi tra i gruppi", () => {
  const { y } = slotPositions(8, 2, -2, [{ size: 4, gap: 2 }]);
  assert.equal(y[0], 2);
  assert.equal(y.at(-1), -2);
  for (let i = 1; i < y.length; i++) assert.ok(y[i] < y[i - 1]);
  assert.ok(y[3] - y[4] > y[0] - y[1] * 1.5); // salto tra i due gruppi
});

test("layout dei neuroni: conteggi per colonna e totale", () => {
  const layout = buildNeuronLayout(SMALL);
  const C = SMALL.n_embd;
  const perBlock = C /*x*/ + C /*ln1*/ + 3 * C + C + C + C + C + 4 * C + C + C;
  const expected = SMALL.vocab_size + SMALL.block_size + 3 * C + SMALL.n_layer * perBlock + C + C + 2 * SMALL.vocab_size;
  assert.equal(layout.count, expected);
  assert.equal(layout.positions.length, 3 * expected);
  for (let k = 0; k < layout.count; k++) assert.ok(Number.isFinite(layout.positions[3 * k + 1]));
  assert.equal(layout.byId.get("b1.hidden").n, 4 * C);
});

test("archi: uno per peso delle matrici più i γ; con i bias si ritrova il totale", () => {
  const engine = smallEngine();
  const layout = buildNeuronLayout(SMALL);
  const edges = buildEdges(layout, paramOf(engine));
  const breakdown = parameterBreakdown(edges, [...engine.paramViews.values()]);
  assert.equal(breakdown.total, engine.paramCount);
  // ordine: permutazione completa, e all'inizio solo archi "forti" della loro matrice
  const seen = new Uint8Array(edges.count);
  for (const k of edges.order) seen[k]++;
  assert.ok(seen.every((v) => v === 1));
  const head = edges.order.slice(0, Math.floor(edges.count * 0.05));
  const avgHead = head.reduce((a, k) => a + edges.strength[k], 0) / head.length;
  const avgAll = edges.strength.reduce((a, v) => a + v, 0) / edges.count;
  assert.ok(avgHead > avgAll);
  // ogni matrice è rappresentata tra i primi archi (rango normalizzato per matrice)
  const firstTenth = new Set(Array.from(edges.order.slice(0, Math.ceil(edges.count * 0.1)), (k) => edges.matrix[k]));
  assert.equal(firstTenth.size, edges.matrices.length);
});

test("attivazioni e spiegazioni coincidono con il forward del motore (modello sintetico)", () => {
  const engine = smallEngine(11);
  const layout = buildNeuronLayout(SMALL);
  const ids = engine.encode("abc! hag");
  const { trace } = engine.forward(ids, { trace: true });
  const param = paramOf(engine);
  for (const t of [0, 3, ids.length - 1]) {
    const values = neuronValues(layout, trace, t);
    const probs = values.subarray(layout.byId.get("probs").offset, layout.byId.get("probs").offset + SMALL.vocab_size);
    assert.ok(Math.abs(probs.reduce((a, v) => a + v, 0) - 1) < 1e-5);
    const hidden = layout.byId.get("b0.hidden");
    for (let i = 0; i < hidden.n; i++) assert.ok(values[hidden.offset + i] >= 0, "ReLU non negativa");
    // ogni neurone: il calcolo spiegato restituisce il valore del motore
    for (let k = 0; k < layout.count; k += 7) {
      const info = explainNeuron(layout, k, trace, t, param, SMALL_VOCAB);
      const shown = info.relu ? info.post : info.kind === "onehot" ? (info.active ? 1 : 0) : info.sum;
      assert.ok(Math.abs(shown - values[k]) < 1e-4, `${info.column.id}[${info.index}] t=${t}: ${shown} vs ${values[k]}`);
    }
  }
  assert.deepEqual(Array.from(softmax(new Float32Array([0, 0]))), [0.5, 0.5]);
});

test("pesi veri: 209.729 parametri = archi + γ + bias; spiegazioni fedeli", { skip: !HAS_DATA && "mancano i pesi in webapp/data" }, async () => {
  const engine = await loadEngine(DATA_DIR, fsLoaders);
  const layout = buildNeuronLayout(engine.config);
  const edges = buildEdges(layout, paramOf(engine));
  const breakdown = parameterBreakdown(edges, [...engine.paramViews.values()]);
  assert.deepEqual(breakdown, { matrix: 206976, gamma: 576, neuron: 2177, total: 209729 });
  assert.equal(layout.count, 4387);
  const ids = engine.encode("ROMEO:\nBut, soft! what light thr");
  const { trace } = engine.forward(ids, { trace: true });
  const t = ids.length - 1;
  const values = neuronValues(layout, trace, t);
  for (let k = 0; k < layout.count; k += 13) {
    const info = explainNeuron(layout, k, trace, t, paramOf(engine), engine.vocab);
    const shown = info.relu ? info.post : info.kind === "onehot" ? (info.active ? 1 : 0) : info.sum;
    assert.ok(Math.abs(shown - values[k]) < 1e-3, `${info.column.id}[${info.index}]`);
  }
});
