// Test della disposizione "Orizzontale con profondità" della vista Rete neurale (tab Transformer):
//   node --test webapp/tests/*.test.mjs
// Logica pura (niente DOM né three.js): dalla traccia vera del motore alla networkSpec del renderer del
// nucleo, mappatura unità → neurone della disposizione a righe, inquadrature. Su un modello sintetico
// piccolo e, se presenti, sui pesi veri di data/.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { ARC_KIND } from "../js/core/neuralNet/links.js";
import { decodeUnit, headOf } from "../js/core/neuralNet/layout.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { GPTEngine, expectedParameters, loadEngine } from "../js/engine/gpt.js";
import { DEPTH_DIRECTIONS, cameraBasis, framePose } from "../js/tabs/transformer/views/neuronDepthCamera.js";
import {
  ATTENTION_SLABS,
  DEPTH_FRACTIONS,
  DEPTH_VIEW_ID,
  attentionWeights,
  buildDepthSpec,
  depthBuildPlan,
  depthCounts,
  depthSectionIds,
  depthSectionValues,
  depthSections,
  depthUnitNeuron,
  linearTensors,
  neuronDepthUnit,
  sectionTitle,
  shownValue,
  shownWeights,
  unitDescription,
  weightOrders,
} from "../js/tabs/transformer/views/neuronDepthSpec.js";
import { buildNeuronLayout, neuronValues } from "../js/tabs/transformer/views/neuronModel.js";

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

/** Un GPT minuscolo con pesi pseudo-casuali deterministici: stesso formato del modello vero. */
function smallEngine(seed = 5) {
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

/** Specifica, modello e traccia per un testo. */
function setup(engine, text, fraction = 0.02) {
  const ids = engine.encode(text).slice(-engine.config.block_size);
  const { trace } = engine.forward(ids, { trace: true });
  const param = (name) => engine.paramViews.get(name).data;
  const spec = buildDepthSpec({ config: engine.config, vocab: engine.vocab, trace, param, fraction });
  return { ids, trace, param, spec };
}

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps * (1 + Math.abs(b)), `${a} ≉ ${b}`);

test("sezioni: ordine del flusso (ingresso, ogni blocco, uscita) e id dei link", () => {
  const ids = depthSectionIds(SMALL);
  const block = (l) => [`b${l}.ln1`, `b${l}.attn`, `b${l}.proj`, `b${l}.resid1`, `b${l}.ln2`, `b${l}.hidden`, `b${l}.ffnOut`, `b${l}.resid2`];
  assert.deepEqual(ids, ["tok_onehot", "pos_onehot", "emb", "x0", ...block(0), ...block(1), "ln_f", "logits", "probs"]);
  const engine = smallEngine();
  const { spec } = setup(engine, "abc de");
  assert.deepEqual(spec.sections.map((s) => s.id), ids, "la specifica segue lo stesso ordine");
  assert.deepEqual(spec.groups.map((g) => g.label), ["Ingresso", "Blocco 0", "Blocco 1", "Uscita"], "tutti i blocchi, nessun × N");
  assert.ok(!spec.sections.some((s) => s.repeat), "nessuna ripetizione compressa");
  assert.deepEqual(ATTENTION_SLABS.map((s) => s.name), ["key", "value", "query", "uscita (concat)"]);
  assert.equal(DEPTH_VIEW_ID, "neuroni-profondita");
});

test("conteggi: neuroni per sezione dal config; per posizione = disposizione a righe senza le copie di x", () => {
  const C = SMALL.n_embd;
  const sizes = Object.fromEntries(depthSections(SMALL).map((s) => [s.id, s.slabs * s.neurons]));
  assert.equal(sizes.tok_onehot, SMALL.vocab_size);
  assert.equal(sizes.pos_onehot, SMALL.block_size);
  assert.equal(sizes.emb, 2 * C);
  assert.equal(sizes["b0.attn"], 4 * C, "key, value, query e uscita");
  assert.equal(sizes["b1.hidden"], 4 * C);
  assert.equal(sizes.logits, SMALL.vocab_size);
  const counts = depthCounts(SMALL, 5);
  assert.equal(Object.values(sizes).reduce((a, b) => a + b, 0), counts.perPosition);
  assert.equal(counts.units, 5 * counts.perPosition);
  assert.equal(counts.rowsCount, buildNeuronLayout(SMALL).count, "con le copie di x si torna al conto della disposizione a righe");
  const engine = smallEngine();
  const { spec, trace } = setup(engine, "abc de");
  const model = buildNetworkModel(spec);
  assert.equal(model.net.units, depthCounts(SMALL, trace.T).units, "il renderer disegna tutti i neuroni di tutte le posizioni");
  assert.equal(model.net.slices.count, trace.T);
  assert.equal(model.values.simulated.size, 0, "nessuna sezione simulata: valori tutti veri");
  assert.equal(model.net.simulated, false);
});

test("valori: ogni sezione coincide con la traccia del motore (indice [lastra][neurone][posizione])", () => {
  const engine = smallEngine();
  const { spec, trace } = setup(engine, "ab!c\nde");
  const layout = buildNeuronLayout(SMALL);
  const T = trace.T;
  const perT = Array.from({ length: T }, (_, t) => neuronValues(layout, trace, t));
  let checked = 0;
  for (const s of spec.sections) {
    assert.equal(s.values.length, s.slabs * s.neurons * T, `${s.id}: forma dei valori`);
    for (let slab = 0; slab < s.slabs; slab++) {
      for (let n = 0; n < s.neurons; n++) {
        const k = depthUnitNeuron(layout, s.id, slab, n);
        assert.ok(k >= 0, `${s.id}[${slab}][${n}] senza neurone corrispondente`);
        for (let t = 0; t < T; t++) {
          near(s.values[(slab * s.neurons + n) * T + t], perT[t][k]);
          checked++;
        }
      }
    }
  }
  assert.equal(checked, depthCounts(SMALL, T).units);
  // e direttamente dalla traccia, senza passare da neuronModel: la query della head 2 al token 3
  const attn = depthSectionValues(SMALL, trace).get("b1.attn");
  const hs = SMALL.n_embd / SMALL.n_head;
  near(attn[(2 * SMALL.n_embd + 2 * hs + 1) * T + 3], trace.blocks[1].heads[2].q[3 * hs + 1]);
  // le probabilità di ogni posizione sommano a 1
  const probs = depthSectionValues(SMALL, trace).get("probs");
  for (let t = 0; t < T; t++) {
    let sum = 0;
    for (let v = 0; v < SMALL.vocab_size; v++) sum += probs[v * T + t];
    near(sum, 1, 1e-5);
  }
});

test("attention: pesi veri del motore, maschera causale rispettata (nessun arco verso il futuro)", () => {
  const engine = smallEngine(9);
  const { spec, trace } = setup(engine, "abcdefgh");
  const T = trace.T;
  const links = spec.links.filter((l) => l.pattern === "attention");
  assert.equal(links.length, SMALL.n_layer, "una self-attention per blocco");
  for (const [l, link] of links.entries()) {
    assert.equal(link.within, `b${l}.attn`);
    assert.equal(link.causal, true);
    assert.equal(link.heads, SMALL.n_head);
    for (let h = 0; h < SMALL.n_head; h++) {
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          const w = link.weights[(h * T + i) * T + j];
          assert.equal(w, trace.blocks[l].heads[h].att[i * T + j]);
          if (j > i) assert.equal(w, 0, "futuro a zero");
        }
      }
    }
  }
  assert.deepEqual(attentionWeights(trace, 0), links[0].weights);
  // gli archi disegnati dal renderer: dalla key (lastra 0) del token j all'uscita (ultima lastra) del token i ≤ … mai j > i
  const model = buildNetworkModel(spec);
  const { arcs } = model.links;
  let attention = 0;
  for (let a = 0; a < arcs.count; a++) {
    if (arcs.kind[a] !== ARC_KIND.attention) continue;
    attention++;
    const from = model.net.sections[model.layout.sectionOf[arcs.from[a]]];
    const to = model.net.sections[model.layout.sectionOf[arcs.to[a]]];
    const key = decodeUnit(from, arcs.from[a] - from.offset);
    const query = decodeUnit(to, arcs.to[a] - to.offset);
    assert.equal(from, to);
    assert.equal(key.slab, 0, "parte dalla key");
    assert.equal(query.slab, ATTENTION_SLABS.length - 1, "arriva all'uscita");
    assert.ok(key.depth <= query.depth, `arco dal token ${key.depth} al token ${query.depth}: verso il futuro`);
  }
  assert.ok(attention > 0);
  // residui: uno per token e per connessione (2 per blocco)
  const residual = Array.from({ length: arcs.count }, (_, a) => arcs.kind[a]).filter((k) => k === ARC_KIND.residual).length;
  assert.equal(residual, 2 * SMALL.n_layer * T);
});

test("pesi: le coppie explicit di ogni nn.Linear sono i pesi veri del motore, i più forti per |w|", () => {
  const engine = smallEngine(11);
  const fraction = 0.1;
  const { spec, param } = setup(engine, "hello".replace(/[lo]/g, "a"), fraction);
  const tensors = linearTensors(SMALL);
  const explicit = spec.links.filter((l) => l.pattern === "explicit" && !l.label.startsWith("γ"));
  assert.equal(explicit.length, tensors.length, "un collegamento per tensore");
  const bySlice = (id) => {
    const s = spec.sections.find((x) => x.id === id);
    return s.slabs * s.neurons;
  };
  for (const t of tensors) {
    const link = explicit.find((l) => l.label.endsWith(`(${t.name})`));
    assert.ok(link, `manca ${t.name}`);
    assert.equal(link.from, t.from);
    assert.equal(link.to, t.to);
    const W = param(t.name);
    assert.equal(W.length, t.rows * t.cols, `${t.name}: forma`);
    const { from, to, weight } = link.pairs;
    assert.equal(weight.length, shownWeights(W.length, fraction));
    // inversa delle mappe del tensore: indice nella fetta → riga (uscita) e colonna (ingresso)
    const rowOf = new Map(Array.from({ length: t.rows }, (_, r) => [t.toIndex(r), r]));
    const colOf = new Map(Array.from({ length: t.cols }, (_, c) => [t.fromIndex(c), c]));
    let minShown = Infinity;
    const shown = new Set();
    for (let i = 0; i < weight.length; i++) {
      assert.ok(from[i] < bySlice(t.from) && to[i] < bySlice(t.to), `${t.name}: indice fuori dalla fetta`);
      const e = rowOf.get(to[i]) * t.cols + colOf.get(from[i]);
      assert.equal(weight[i], W[e], `${t.name}: peso della coppia ${i}`);
      minShown = Math.min(minShown, Math.abs(W[e]));
      shown.add(e);
    }
    for (let e = 0; e < W.length; e++) if (!shown.has(e)) assert.ok(Math.abs(W[e]) <= minShown, `${t.name}: un peso escluso è più forte di uno disegnato`);
  }
  // γ delle LayerNorm: tutti, uno a uno, con il loro valore
  const gammas = spec.links.filter((l) => l.label.startsWith("γ"));
  assert.equal(gammas.length, 2 * SMALL.n_layer + 1);
  const g = gammas.find((l) => l.label.includes("blocks.1.ln2.weight"));
  assert.deepEqual(Array.from(g.pairs.weight), Array.from(param("blocks.1.ln2.weight")));
  assert.deepEqual(Array.from(g.pairs.from), Array.from(g.pairs.to));
  // l'ordine per |w| è decrescente
  const orders = weightOrders(SMALL, param);
  const W = param("blocks.0.ffwd.net.0.weight");
  const order = orders.get("blocks.0.ffwd.net.0.weight");
  for (let i = 1; i < order.length; i++) assert.ok(Math.abs(W[order[i - 1]]) >= Math.abs(W[order[i]]));
});

test("mappatura: ogni unità va a un neurone diverso della disposizione a righe, e ritorno", () => {
  const layout = buildNeuronLayout(SMALL);
  const seen = new Set();
  for (const s of depthSections(SMALL)) {
    for (let slab = 0; slab < s.slabs; slab++) {
      for (let n = 0; n < s.neurons; n++) {
        const k = depthUnitNeuron(layout, s.id, slab, n);
        assert.ok(k >= 0 && !seen.has(k), `${s.id}[${slab}][${n}] → ${k}`);
        seen.add(k);
        assert.deepEqual(neuronDepthUnit(layout, k), { section: s.id, slab, neuron: n }, `ritorno da ${k}`);
      }
    }
  }
  // tutti i neuroni a righe tranne le copie di x sono coperti; le copie tornano al vettore copiato
  const copies = layout.columns.filter((c) => c.key === "x");
  assert.equal(seen.size, layout.count - copies.reduce((a, c) => a + c.n, 0));
  assert.deepEqual(neuronDepthUnit(layout, layout.byId.get("b0.x").offset + 3), { section: "x0", slab: 0, neuron: 3 });
  assert.deepEqual(neuronDepthUnit(layout, layout.byId.get("b1.x").offset + 3), { section: "b0.resid2", slab: 0, neuron: 3 });
  assert.deepEqual(neuronDepthUnit(layout, layout.byId.get("out.x").offset), { section: "b1.resid2", slab: 0, neuron: 0 });
  // query, key e value finiscono nelle lastre giuste (qkvIndex: part × C + head × hs + d)
  const C = SMALL.n_embd;
  const qkv = layout.byId.get("b1.qkv").offset;
  assert.equal(depthUnitNeuron(layout, "b1.attn", 2, 5), qkv + 0 * C + 5, "lastra 2 = query");
  assert.equal(depthUnitNeuron(layout, "b1.attn", 0, 5), qkv + 1 * C + 5, "lastra 0 = key");
  assert.equal(depthUnitNeuron(layout, "b1.attn", 1, 5), qkv + 2 * C + 5, "lastra 1 = value");
  assert.equal(depthUnitNeuron(layout, "b1.attn", 3, 5), layout.byId.get("b1.concat").offset + 5, "lastra 3 = concat");
  assert.equal(depthUnitNeuron(layout, "nessuna", 0, 0), -1);
});

test("prompt di un solo token: niente attention da disegnare, specifica valida", () => {
  const engine = smallEngine();
  const { spec } = setup(engine, "a");
  assert.equal(spec.depth.count, 1);
  assert.ok(!spec.links.some((l) => l.pattern === "attention"));
  const model = buildNetworkModel(spec);
  assert.equal(model.net.units, depthCounts(SMALL, 1).units);
});

/** Come NetworkScene.describe (che vive con three.js): sezione, lastra, neurone, posizione, valore, head. */
function describe(model, u) {
  const s = model.net.sections[model.layout.sectionOf[u]];
  const info = decodeUnit(s, u - s.offset);
  return { section: s, ...info, value: model.values.raw[u], head: s.heads > 1 ? headOf(s, info.neuron) : undefined };
}

test("etichetta: sezione, lastra, head, token (anche con un solo token) e valori minuscoli leggibili", () => {
  const engine = smallEngine();
  const { spec, trace } = setup(engine, "ab!c");
  const model = buildNetworkModel(spec);
  // la key della head 2, neurone 1 della fascia, al token 3 "c"
  const s = model.net.byId.get("b1.attn");
  const hs = SMALL.n_embd / SMALL.n_head;
  const u = s.offset + (0 * s.neurons + 2 * hs + 1) * s.depth.count + 3;
  const d = unitDescription(describe(model, u), spec.depth.names);
  assert.equal(d.title, "Self-attention · blocco 1");
  assert.equal(d.neuron, `key · neurone ${2 * hs + 1} di ${SMALL.n_embd}`);
  assert.equal(d.head, 2);
  assert.equal(d.token, `token 3 «c» (di ${trace.T})`);
  assert.equal(d.value, `valore ${shownValue(trace.blocks[1].heads[2].k[3 * hs + 1])}`);
  // logits: il nome del carattere del neurone
  const logits = model.net.byId.get("logits");
  assert.match(unitDescription(describe(model, logits.offset + 2 * logits.depth.count), spec.depth.names).neuron, /^neurone 2 «a» di 11$/);
  // un solo token: la profondità normalizzata dal nucleo perde i nomi, l'etichetta li prende dalla specifica
  const one = setup(engine, "b");
  const m1 = buildNetworkModel(one.spec);
  assert.equal(m1.net.slices.count, 1);
  assert.equal(unitDescription(describe(m1, m1.net.byId.get("x0").offset), one.spec.depth.names).token, "token 0 «b» (di 1)");
  assert.equal(sectionTitle({ id: "ln_f", label: "LN finale" }), "LN finale");
  // valori: tre decimali, sotto 0,001 notazione scientifica (mai "0,0000" per una probabilità minuscola)
  assert.equal(shownValue(0), "0,000");
  assert.equal(shownValue(1.2015), "1,202");
  assert.equal(shownValue(-0.0042), "-0,0042");
  assert.equal(shownValue(3.2e-7), "3,2 · 10⁻⁷");
  assert.equal(shownValue(-9.96e-5), "-1,0 · 10⁻⁴");
  assert.equal(shownValue(0.0001234), "1,2 · 10⁻⁴");
});

test("testi: niente simulazioni dichiarate per sbaglio, niente copie (qui non ci sono), legenda onesta sui collegamenti fissi", () => {
  const engine = smallEngine();
  const { spec } = setup(engine, "abc");
  assert.match(spec.intro, /non una simulazione/);
  assert.doesNotMatch(spec.note, /copie/, "le copie del vettore x esistono solo nella disposizione a righe");
  assert.match(spec.note, /LayerNorm e softmax/);
  const texts = depthSections(SMALL).map((s) => s.text).join(" ");
  assert.doesNotMatch(texts, /simulat/);
});

test("quote: si ricostruisce solo con un prompt o pesi nuovi, o oltre la quota costruita (poi alla massima)", () => {
  const max = Math.max(...DEPTH_FRACTIONS);
  const key = "trained:|1,2,3";
  // nessun modello o un prompt diverso: costruzione alla quota chiesta
  assert.deepEqual(depthBuildPlan({ key, builtKey: undefined, builtFraction: 0, T: 3, fraction: 0.02 }), { rebuild: true, fraction: 0.02 });
  assert.deepEqual(depthBuildPlan({ key, builtKey: "trained:|1,2", builtFraction: max, T: 3, fraction: 0.02 }), { rebuild: true, fraction: 0.02 });
  // stesso prompt, quota già costruita o più bassa: niente ricostruzione (setSliceDensity)
  for (const fraction of [0.005, 0.01, 0.02]) assert.equal(depthBuildPlan({ key, builtKey: key, builtFraction: 0.02, T: 3, fraction }).rebuild, false);
  // stesso prompt, quota più alta: una volta sola, alla massima; poi tutte le quote senza ricostruire
  assert.deepEqual(depthBuildPlan({ key, builtKey: key, builtFraction: 0.02, T: 3, fraction: 0.05 }), { rebuild: true, fraction: max });
  for (const fraction of DEPTH_FRACTIONS) assert.equal(depthBuildPlan({ key, builtKey: key, builtFraction: max, T: 3, fraction }).rebuild, false);
  // pesi personalizzati (chiave null): sempre nuovi; un solo token: vale solo la quota esatta
  assert.equal(depthBuildPlan({ key: null, builtKey: null, builtFraction: max, T: 3, fraction: 0.02 }).rebuild, true);
  assert.equal(depthBuildPlan({ key, builtKey: key, builtFraction: 0.02, T: 1, fraction: 0.02 }).rebuild, false);
  assert.deepEqual(depthBuildPlan({ key, builtKey: key, builtFraction: 0.02, T: 1, fraction: 0.01 }), { rebuild: true, fraction: 0.01 });
});

test("quote: anche la più alta resta sotto il tetto dei pesi della fetta del nucleo", () => {
  const engine = smallEngine();
  for (const f of DEPTH_FRACTIONS) {
    const { spec } = setup(engine, "abcd", f);
    const model = buildNetworkModel(spec);
    assert.equal(model.links.sliceSegments.dropped, 0, `quota ${f}: nessun peso tagliato dal tetto`);
  }
});

test("camera: i punti entrano nella parte libera dello stage, il centro scende sotto i comandi", () => {
  const points = [];
  for (const x of [-50, 50]) for (const y of [-8, 12]) for (const z of [-9, 9]) points.push([x, y, z]);
  const fov = 45;
  const aspect = 1.44;
  for (const [name, dir] of Object.entries(DEPTH_DIRECTIONS)) {
    const top = 0.12;
    const bottom = 0.05;
    const pose = framePose(points, dir, { fov, aspect, top, bottom, margin: 1 });
    const { right, up, d } = cameraBasis(dir);
    const tanV = Math.tan((fov * Math.PI) / 360);
    for (const p of points) {
      const rel = p.map((v, k) => v - pose.position[k]);
      const depth = -(rel[0] * d[0] + rel[1] * d[1] + rel[2] * d[2]);
      assert.ok(depth > 0, `${name}: punto dietro la camera`);
      const ny = (rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2]) / (depth * tanV);
      const nx = (rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2]) / (depth * tanV * aspect);
      assert.ok(Math.abs(nx) <= 1 + 1e-9, `${name}: fuori in larghezza (${nx})`);
      assert.ok(ny <= 1 - 2 * top + 1e-9 && ny >= -1 + 2 * bottom - 1e-9, `${name}: fuori dalla fascia libera (${ny})`);
    }
  }
  // senza comandi il punto guardato è il centro dei punti; con i comandi in alto si sposta in su
  const plain = framePose(points, DEPTH_DIRECTIONS.side, { fov, aspect });
  const shifted = framePose(points, DEPTH_DIRECTIONS.side, { fov, aspect, top: 0.2 });
  assert.ok(shifted.target[1] > plain.target[1]);
  assert.deepEqual(framePose([], DEPTH_DIRECTIONS.side, { fov, aspect }).target, [0, 0, 0]);
});

test("GPT del corso (pesi veri): 4.067 neuroni distinti per token, valori e pesi del motore", { skip: !HAS_DATA && "data/ non presente" }, async () => {
  const engine = await loadEngine(DATA_DIR, fsLoaders);
  const config = engine.config;
  const counts = depthCounts(config, config.block_size);
  assert.equal(counts.perPosition, 4067);
  assert.equal(counts.rowsCount, 4387, "come la disposizione a righe (con le 320 copie di x)");
  assert.equal(counts.units, 130144, "32 posizioni: sotto il limite di 250.000 del nucleo");
  const text = "ROMEO:\nBut soft, what light thro"; // 32 caratteri: block_size, il caso più grande
  const { spec, trace, param } = setup(engine, text);
  assert.equal(trace.T, config.block_size);
  const model = buildNetworkModel(spec);
  assert.equal(model.net.units, 4067 * trace.T);
  assert.equal(model.links.capped, false, "nessun collegamento tagliato dai tetti");
  // valori di una sezione profonda contro la traccia: uscita della feed-forward del blocco 3
  const layout = buildNeuronLayout(config);
  const s = spec.sections.find((x) => x.id === "b3.ffnOut");
  for (const t of [0, 13, trace.T - 1]) {
    const values = neuronValues(layout, trace, t);
    for (let n = 0; n < s.neurons; n++) near(s.values[n * trace.T + t], values[depthUnitNeuron(layout, s.id, 0, n)]);
  }
  // lm_head: pesi veri
  const lm = spec.links.find((l) => l.label.endsWith("(lm_head.weight)"));
  const W = param("lm_head.weight");
  for (let i = 0; i < lm.pairs.weight.length; i++) assert.equal(lm.pairs.weight[i], W[lm.pairs.to[i] * config.n_embd + lm.pairs.from[i]]);
});
