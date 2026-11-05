// Test del motore GPT in JavaScript (modulo puro, niente DOM): node --test webapp/tests/
// La fedeltà numerica rispetto a PyTorch è verificata da tests/test_js_engine.py; qui si
// controllano proprietà che devono valere per qualunque peso: forme, probabilità,
// causalità, determinismo, tokenizer, gestione degli errori, snapshot.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_SEED,
  GPTEngine,
  VocabularyError,
  expectedParameters,
  loadEngine,
  mulberry32,
  sample,
} from "../js/engine/gpt.js";

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const HAS_DATA = existsSync(`${DATA_DIR}weights.json`) && existsSync(`${DATA_DIR}weights.f32.bin`);
const HAS_TRACE = existsSync(`${DATA_DIR}training/trace.json`);

// Modello sintetico piccolo, con la stessa architettura e gli stessi nomi di PyTorch.
const SMALL = { vocab_size: 11, block_size: 8, n_embd: 16, n_head: 4, n_layer: 2, dropout: 0 };
const SMALL_VOCAB = ["\n", " ", "a", "b", "c", "d", "e", "f", "g", "h", "!"];

function syntheticManifest(config = SMALL, vocab = SMALL_VOCAB) {
  let offset = 0;
  const params = expectedParameters(config).map(({ name, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const entry = { name, shape, offset, numel };
    offset += numel;
    return entry;
  });
  return { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", config, vocab, total_params: offset, params, training: null };
}

function randomWeights(manifest, seed = 7) {
  const rng = mulberry32(seed);
  const values = new Float32Array(manifest.total_params);
  for (const p of manifest.params) {
    const isNormWeight = /ln(1|2|_f)\.weight$/.test(p.name);
    for (let i = 0; i < p.numel; i++) values[p.offset + i] = isNormWeight ? 1 + 0.1 * (rng() - 0.5) : rng() - 0.5;
  }
  return values;
}

function smallEngine(seed = 7) {
  const manifest = syntheticManifest();
  return GPTEngine.fromBuffers(manifest, randomWeights(manifest, seed).buffer);
}

const sum = (values) => values.reduce((a, b) => a + b, 0);
const near = (a, b, eps, what = "") => assert.ok(Math.abs(a - b) <= eps, `${what} ${a} ≉ ${b}`);

const fsLoaders = {
  loadJson: async (file) => JSON.parse(await readFile(file, "utf8")),
  loadBuffer: async (file) => {
    const bytes = await readFile(file);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  },
};
let realEngine = null;
const loadReal = async () => (realEngine ??= await loadEngine(DATA_DIR, fsLoaders));

test("expectedParameters: 209.729 parametri nella configurazione del corso", () => {
  const params = expectedParameters({ vocab_size: 65, block_size: 32, n_embd: 64, n_head: 4, n_layer: 4 });
  assert.equal(params.length, 94);
  assert.equal(sum(params.map((p) => p.shape.reduce((a, b) => a * b, 1))), 209729);
  assert.equal(params[0].name, "token_embedding_table.weight");
  assert.deepEqual(params.at(-1), { name: "lm_head.bias", shape: [65] });
});

test("forward: forme dei logits e di tutte le attivazioni del trace", () => {
  const engine = smallEngine();
  const { vocab_size: V, n_embd: C, head_size: hs, n_layer: L, n_head: H } = engine.config;
  const ids = engine.encode("abc de");
  const T = ids.length;
  const { logits, trace } = engine.forward(ids, { trace: true });
  assert.equal(logits.length, T * V);
  assert.deepEqual(trace.ids, ids);
  for (const key of ["tokEmb", "posEmb", "x0", "lnF"]) assert.equal(trace[key].length, T * C, key);
  assert.equal(trace.logits, logits);
  assert.equal(trace.blocks.length, L);
  for (const block of trace.blocks) {
    for (const key of ["ln1", "concat", "proj", "resid1", "ln2", "ffnOut", "resid2"]) assert.equal(block[key].length, T * C, key);
    for (const key of ["ffnPre", "ffnPost"]) assert.equal(block[key].length, T * 4 * C, key);
    assert.ok(block.ffnPost.every((v) => v >= 0), "dopo la ReLU niente valori negativi");
    assert.equal(block.heads.length, H);
    for (const head of block.heads) {
      for (const key of ["q", "k", "v", "out"]) assert.equal(head[key].length, T * hs, key);
      assert.equal(head.scores.length, T * T);
      assert.equal(head.att.length, T * T);
    }
  }
  // Senza trace: stessi logits, nessun trace.
  const plain = engine.forward(ids);
  assert.deepEqual(plain.logits, logits);
  assert.equal(plain.trace, undefined);
});

test("attention: righe che sommano a 1, futuro mascherato (−∞ prima, 0 dopo la softmax)", () => {
  const engine = smallEngine();
  const ids = engine.encode("abcdefg!");
  const T = ids.length;
  const { trace } = engine.forward(ids, { trace: true });
  for (const head of trace.blocks.flatMap((b) => b.heads)) {
    for (let i = 0; i < T; i++) {
      near(sum(head.att.subarray(i * T, (i + 1) * T)), 1, 1e-6, `riga ${i}`);
      for (let j = i + 1; j < T; j++) {
        assert.equal(head.scores[i * T + j], -Infinity);
        assert.equal(head.att[i * T + j], 0);
      }
      assert.ok(Number.isFinite(head.scores[i * T + i]));
    }
  }
});

test("causalità: cambiare un token futuro non cambia i logits delle posizioni passate", () => {
  const engine = smallEngine();
  const V = engine.config.vocab_size;
  const a = engine.encode("abcdefgh");
  const b = engine.encode("abcd!!!!");
  const la = engine.forward(a).logits;
  const lb = engine.forward(b).logits;
  assert.deepEqual(la.subarray(0, 4 * V), lb.subarray(0, 4 * V)); // identici, non solo vicini
  assert.notDeepEqual(la.subarray(4 * V), lb.subarray(4 * V));
});

test("nextTokenDistribution: probabilità che sommano a 1, temperatura e top-k", () => {
  const engine = smallEngine();
  const ids = engine.encode("ab c");
  const base = engine.nextTokenDistribution(ids);
  near(sum(base.probs), 1, 1e-6, "somma");
  // È la softmax dell'ultima riga dei logits del forward.
  const V = engine.config.vocab_size;
  const last = engine.forward(ids).logits.subarray((ids.length - 1) * V);
  assert.deepEqual(base.logits, last);

  const cold = engine.nextTokenDistribution(ids, { temperature: 0.3 });
  const hot = engine.nextTokenDistribution(ids, { temperature: 3 });
  assert.ok(Math.max(...cold.probs) > Math.max(...base.probs));
  assert.ok(Math.max(...hot.probs) < Math.max(...base.probs));
  near(sum(cold.probs), 1, 1e-6);

  const top3 = engine.nextTokenDistribution(ids, { topK: 3 });
  assert.equal(top3.probs.filter((p) => p > 0).length, 3);
  near(sum(top3.probs), 1, 1e-6);
  const best = engine.topTokens(base.probs, 3).map((t) => t.id).sort((x, y) => x - y);
  assert.deepEqual(top3.probs.reduce((acc, p, i) => (p > 0 ? [...acc, i] : acc), []), best);

  assert.throws(() => engine.nextTokenDistribution(ids, { temperature: 0 }), RangeError);
  assert.throws(() => engine.nextTokenDistribution(ids, { topK: 0 }), RangeError);
  assert.throws(() => engine.nextTokenDistribution(ids, { topK: V + 1 }), RangeError);
});

test("contesto più lungo di block_size: si usano gli ultimi block_size token", () => {
  const engine = smallEngine();
  const long = engine.encode("abcdefgh!abc");
  const tail = long.slice(-engine.config.block_size);
  assert.deepEqual(engine.nextTokenDistribution(long).probs, engine.nextTokenDistribution(tail).probs);
  assert.throws(() => engine.forward(long), /block_size=8/);
});

test("generate: deterministico con lo stesso seed, passo per passo, contesto tagliato", () => {
  const engine = smallEngine();
  const prompt = engine.encode("ab");
  const run = (options) => [...engine.generate(prompt, 20, options)].map((s) => s.id);
  const a = run({ seed: 42 });
  assert.equal(a.length, 20);
  assert.deepEqual(run({ seed: 42 }), a);
  assert.notDeepEqual(run({ seed: 43 }), a);
  assert.deepEqual(run({}), run({ seed: DEFAULT_SEED }));

  // Ogni passo usa la distribuzione degli ultimi block_size token di prompt + generati.
  const context = [...prompt];
  for (const step of engine.generate(prompt, 12, { seed: 5, temperature: 0.9, topK: 4 })) {
    const expected = engine.nextTokenDistribution(context, { temperature: 0.9, topK: 4 });
    assert.deepEqual(step.probs, expected.probs);
    assert.equal(step.contextStart, Math.max(0, context.length - engine.config.block_size));
    assert.ok(step.probs[step.id] > 0, "estratto un token con probabilità 0");
    assert.equal(step.token, engine.vocab[step.id]);
    context.push(step.id);
  }

  // Il generatore è pigro: si può fermare dopo un passo; con trace c'è il forward del contesto.
  const first = engine.generate(prompt, 5, { trace: true }).next().value;
  assert.equal(first.trace.T, prompt.length);
  assert.equal(first.trace.logits.length, prompt.length * engine.config.vocab_size);
  assert.deepEqual([...engine.generate(prompt, 0)], []);
  assert.throws(() => engine.generate(prompt, -1), RangeError); // errore subito, non al primo next()
  assert.throws(() => engine.generate([], 3), RangeError);
});

test("mulberry32 e sample: riproducibili, in [0, 1), rispettano le probabilità", () => {
  const a = mulberry32(123);
  const b = mulberry32(123);
  for (let i = 0; i < 100; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x >= 0 && x < 1);
  }
  assert.equal(sample([0, 1, 0], mulberry32(1)), 1);
  assert.equal(sample([0.5, 0.5, 0], () => 0.999999), 1); // mai l'indice a probabilità 0
  const rng = mulberry32(9);
  const counts = [0, 0, 0];
  for (let i = 0; i < 20000; i++) counts[sample([0.2, 0.5, 0.3], rng)]++;
  near(counts[1] / 20000, 0.5, 0.02, "frequenza");
  assert.throws(() => sample([0, 0], rng), /nessuna probabilità/);
});

test("encode/decode: andata e ritorno, errori chiari", () => {
  const engine = smallEngine();
  const text = "abc\n hg!";
  assert.equal(engine.decode(engine.encode(text)), text);
  assert.deepEqual(engine.encode(""), []);
  assert.throws(
    () => engine.encode("abZ€a"),
    (error) => error instanceof VocabularyError && error.unknown.includes("Z") && error.unknown.includes("€") && /vocabolario/.test(error.message),
  );
  assert.throws(() => engine.decode([0, 99]), /token id non valido: 99/);
  assert.throws(() => engine.forward([0, 1.5]), RangeError);
  assert.throws(() => engine.forward([]), /almeno un token/);
});

test("logitLens: uno strato per il residual stream, l'ultimo coincide con il modello", () => {
  const engine = smallEngine();
  const ids = engine.encode("bad cafe");
  const { T, V, layers } = engine.logitLens(ids, { topK: 3 });
  assert.equal(layers.length, engine.config.n_layer + 1);
  assert.deepEqual(layers.map((l) => l.label), ["embedding", "blocco 1", "blocco 2"]);
  const { logits } = engine.forward(ids);
  const last = layers.at(-1);
  for (let t = 0; t < T; t++) {
    const row = last.probs.subarray(t * V, (t + 1) * V);
    near(sum(row), 1, 1e-6);
    // softmax dei logits veri del modello alla posizione t
    const lrow = logits.subarray(t * V, (t + 1) * V);
    const max = Math.max(...lrow);
    const z = sum(Array.from(lrow, (v) => Math.exp(v - max)));
    for (let i = 0; i < V; i++) near(row[i], Math.exp(lrow[i] - max) / z, 1e-6, `p[${t},${i}]`);
    assert.equal(last.top[t].length, 3);
    assert.ok(last.top[t][0].prob >= last.top[t][1].prob && last.top[t][1].prob >= last.top[t][2].prob);
  }
});

test("pesi: paramViews vive, setWeights, restoreWeights, controllo del manifest", () => {
  const engine = smallEngine(1);
  const manifest = syntheticManifest();
  assert.equal(engine.paramCount, manifest.total_params);
  assert.equal(engine.paramViews.size, manifest.params.length);
  const view = engine.paramViews.get("lm_head.bias");
  assert.deepEqual(view.shape, [SMALL.vocab_size]);
  const ids = engine.encode("abc");
  const before = engine.forward(ids).logits;

  const other = randomWeights(manifest, 2);
  engine.setWeights(other);
  assert.equal(engine.weightsSource, "custom");
  assert.equal(view.data[0], other[view.offset]); // la vista mostra i pesi nuovi
  assert.notDeepEqual(engine.forward(ids).logits, before);
  engine.restoreWeights();
  assert.equal(engine.weightsSource, "trained");
  assert.deepEqual(engine.forward(ids).logits, before);

  assert.throws(() => engine.setWeights(new Float32Array(3)), /servono/);
  assert.throws(() => GPTEngine.fromBuffers(manifest, new ArrayBuffer(8)), /attesi/);
  const missing = { ...manifest, params: manifest.params.slice(1) };
  assert.throws(() => GPTEngine.fromBuffers(missing, randomWeights(manifest).buffer), /mancante.*token_embedding_table/);
  const wrongShape = { ...manifest, params: manifest.params.map((p, i) => (i === 0 ? { ...p, shape: [p.numel] } : p)) };
  assert.throws(() => GPTEngine.fromBuffers(wrongShape, randomWeights(manifest).buffer), /forma/);
  // Senza sorgenti (fromBuffers nudo) loadSnapshot spiega cosa manca.
  return assert.rejects(engine.loadSnapshot(0), /storia del training non disponibile/);
});

test("fromBuffers accetta anche un Buffer di Node non allineato", () => {
  const manifest = syntheticManifest();
  const weights = randomWeights(manifest);
  const padded = Buffer.alloc(weights.byteLength + 1);
  Buffer.from(weights.buffer).copy(padded, 1);
  const engine = GPTEngine.fromBuffers(manifest, padded.subarray(1));
  assert.deepEqual(engine.weights, weights);
});

test("dati veri: 209.729 parametri, vocabolario di 65 caratteri, logits 32 × 65", { skip: !HAS_DATA && "mancano webapp/data/weights.*" }, async () => {
  const engine = await loadReal();
  assert.equal(engine.paramCount, 209729);
  assert.equal(engine.vocab.length, 65);
  const ids = engine.encode("ROMEO:\nBut, soft! what light thr");
  assert.equal(ids.length, 32);
  const { logits } = engine.forward(ids);
  assert.equal(logits.length, 32 * 65);
  assert.ok(logits.every(Number.isFinite));
  // Il modello addestrato ha imparato qualcosa: dopo "thr" una vocale è molto probabile.
  const top = engine.topTokens(engine.nextTokenDistribution(ids).probs, 2).map((t) => t.token);
  assert.ok(top.every((ch) => "aeiou".includes(ch)), `top-2: ${top}`);
});

test("dati veri: loadSnapshot cambia i pesi, restoreWeights li ripristina", { skip: !(HAS_DATA && HAS_TRACE) && "manca la storia del training" }, async () => {
  const engine = await loadReal();
  const ids = engine.encode("ROMEO:");
  const trained = engine.nextTokenDistribution(ids).probs;

  const init = await engine.loadSnapshot(0);
  assert.equal(init.step, 0);
  assert.equal(engine.weightsSource, "snapshot");
  assert.equal(engine.snapshotStep, 0);
  const random = engine.nextTokenDistribution(ids).probs;
  assert.ok(Math.max(...random) < 0.2, "all'inizializzazione la distribuzione è quasi uniforme");

  const trace = await engine.loadTrace();
  const lastStep = trace.snapshots.at(-1).step;
  await engine.loadSnapshot(lastStep); // l'ultimo snapshot è il checkpoint in float16
  const final = engine.nextTokenDistribution(ids).probs;
  for (let i = 0; i < final.length; i++) near(final[i], trained[i], 5e-3, `p[${i}]`);

  // Due richieste in fila: vince l'ultima, la prima restituisce null.
  const [first, second] = await Promise.all([engine.loadSnapshot(0), engine.loadSnapshot(lastStep)]);
  assert.equal(first, null);
  assert.equal(second.step, lastStep);
  assert.equal(engine.snapshotStep, lastStep);

  engine.restoreWeights();
  assert.equal(engine.snapshotStep, null);
  assert.deepEqual(engine.nextTokenDistribution(ids).probs, trained);
  await assert.rejects(engine.loadSnapshot(123), /non disponibile.*0, 50/);
});
