// Test delle azioni animate del Laboratorio pesi (js/tabs/lab/actions/): node --test webapp/tests/*.test.mjs
// - ogni passo dei calcoli ricalcolato qui (x · Wᵀ, punteggi con la maschera, softmax, ·V, concat, LayerNorm,
//   feed-forward con ReLU, logits) coincide con il motore JS, bit per bit, anche su uno snapshot e su un
//   modello sintetico di forma diversa (un run compatibile caricato dallo studente);
// - il gradiente di lm_head (p − y, (p − y)·hᵀ), ∂L/∂h e i gradienti di γ e β di ln_f contro le differenze
//   finite (in double e sul motore vero); SGD e il primo passo di AdamW (come PyTorch), la loss che scende;
// - impostazioni, catalogo, ricerca, righe di codice vere, geometria e inquadrature (puri);
// - tutti i passi di tutte le azioni costruiti su una regia finta con la scena vera (three.js da vendor e
//   un canvas finto): carte con i numeri giusti, veli sulle matrici vere, lavagne senza "undefined"/"NaN".
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      if (specifier.startsWith("three/addons/")) return { url: ${JSON.stringify(`${VENDOR}addons/`)} + specifier.slice(13), shortCircuit: true };
      return next(specifier, context);
    }`),
);

const { GPTEngine, expectedParameters, loadEngine, mulberry32 } = await import("../js/engine/gpt.js");
const M = await import("../js/tabs/lab/actions/math.js");
const D = await import("../js/tabs/lab/actions/data.js");
const TR = await import("../js/tabs/lab/actions/training.js");
const SET = await import("../js/tabs/lab/actions/settings.js");
const P = await import("../js/tabs/lab/actions/plan.js");
const F = await import("../js/tabs/lab/actions/format.js");
const { CODE, codeFor } = await import("../js/tabs/lab/actions/code.js");
const { ACTIONS, ACTION_IDS, CATEGORIES, actionCard } = await import("../js/tabs/lab/actions/catalog.js");
const { searchEntries } = await import("../js/tabs/lab/actions/search.js");
const { searchIndex } = await import("../js/tabs/lab/search.js");
const { makeSteps, onChange, progressCount } = await import("../js/tabs/lab/actions/stepKit.js");
const { Card, TokenStrip, LABEL_PX } = await import("../js/tabs/lab/actions/cards.js");
const { linearSteps } = await import("../js/tabs/lab/actions/linearAction.js");
const { attentionSteps, attentionValueSteps } = await import("../js/tabs/lab/actions/attentionAction.js");
const { layerNormSteps } = await import("../js/tabs/lab/actions/layerNormAction.js");
const { ffnSteps } = await import("../js/tabs/lab/actions/ffnAction.js");
const { logitsSteps } = await import("../js/tabs/lab/actions/logitsAction.js");
const { trainingStepSteps } = await import("../js/tabs/lab/actions/trainingAction.js");
const { historySteps } = await import("../js/tabs/lab/actions/historyAction.js");
const { LabScene } = await import("../js/tabs/lab/scene/labScene.js");
const { layoutModel } = await import("../js/tabs/lab/scene/layout.js");
const { NumbersGroup } = await import("../js/core/matrixNumbers.js");
const { padBottom } = await import("../js/core/navMath.js");
const { buildModuleTree } = await import("../js/shared/weights/moduleTree.js");
const { loadBuiltinRun } = await import("../js/shared/weights/run.js");
const { LabState } = await import("../js/tabs/lab/state.js");

const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const ACTIONS_DIR = fileURLToPath(new URL("../js/tabs/lab/actions/", import.meta.url));
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const b = await readFile(file);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const fsLoaders = { loadJson, loadBuffer };
const spec = await loadJson(`${DATA_DIR}model.json`);
const engine = await loadEngine(DATA_DIR, fsLoaders);
const trace = await engine.loadTrace();
const PROMPT = SET.DEFAULT_PROMPT;
const ids = engine.encode(PROMPT);

// ---------------------------------------------------------------- aiuti

/** Uguali bit per bit (anche i −∞ dei punteggi mascherati). */
function sameBits(a, b, what) {
  assert.equal(a.length, b.length, `${what}: lunghezze diverse`);
  for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) assert.fail(`${what}[${i}]: ${a[i]} ≠ ${b[i]}`);
}

/** La loss di una posizione in DOUBLE (per le differenze finite): logits = h · Wᵀ + b, −log softmax. */
function lossDouble(h, W, b, target, V, C) {
  const z = new Float64Array(V);
  for (let c = 0; c < V; c++) {
    let acc = b[c];
    for (let j = 0; j < C; j++) acc += h[j] * W[c * C + j];
    z[c] = acc;
  }
  let max = -Infinity;
  for (const v of z) max = Math.max(max, v);
  let sum = 0;
  for (const v of z) sum += Math.exp(v - max);
  return max + Math.log(sum) - z[target];
}

/** Un motore sintetico con una forma diversa dal corso (come un run compatibile caricato dallo studente). */
function syntheticEngine(config, seed = 7) {
  const rng = mulberry32(seed);
  const params = expectedParameters(config);
  let offset = 0;
  const layout = params.map(({ name, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const entry = { name, shape, offset, numel };
    offset += numel;
    return entry;
  });
  const flat = new Float32Array(offset);
  for (const p of layout) {
    const ln = /\.(ln\d|ln_f)\.weight$/.test(p.name);
    for (let i = 0; i < p.numel; i++) flat[p.offset + i] = ln ? 1 + 0.2 * (rng() - 0.5) : (rng() - 0.5) * 0.6;
  }
  const vocab = Array.from({ length: config.vocab_size }, (_, i) => String.fromCharCode(97 + i));
  const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "w.bin", config, vocab, total_params: offset, params: layout };
  return GPTEngine.fromBuffers(manifest, flat.buffer);
}

/** Tutti i calcoli delle azioni contro la traccia del motore, per un motore e un prompt. */
function checkAgainstEngine(eng, promptIds, label) {
  const cfg = eng.config;
  for (let b = 0; b < cfg.n_layer; b++) {
    for (let h = 0; h < cfg.n_head; h++) {
      for (const t of [0, Math.floor(promptIds.length / 2), promptIds.length - 1]) {
        const f = D.forwardData(eng, promptIds, { token: t, block: b, head: h });
        for (const choice of SET.LINEAR_CHOICES.map((c) => c.id)) {
          const L = D.linearData(f, choice);
          sameBits(L.recomputed, L.output, `${label} ${choice} b${b} h${h} t${t}`);
          for (const d of [0, L.outDim - 1]) {
            const r = M.linearRow(L.input, L.W, L.b, L.inDim, d);
            assert.equal(r.value, L.output[d], `${label} ${choice}: riga ${d}`);
            assert.ok(Math.abs(r.products.reduce((s, p) => s + p, 0) - r.sum) < 1e-5, "i prodotti sommano alla somma");
          }
        }
        const A = D.attentionData(f);
        sameBits(A.masked, A.engine.scores, `${label} punteggi b${b} h${h}`);
        sameBits(A.att, A.engine.att, `${label} attention b${b} h${h}`);
        sameBits(A.out, A.engine.out, `${label} att·V b${b} h${h}`);
        sameBits(A.concat, f.trace.blocks[b].concat, `${label} concat b${b}`);
        for (let i = 0; i < f.T; i++) {
          for (let j = 0; j < f.T; j++) {
            if (j > i) assert.equal(A.masked[i * f.T + j], -Infinity);
            else assert.equal(A.scaled[i * f.T + j], A.engine.scores[i * f.T + j], "prima della maschera: gli stessi punteggi");
          }
          const steps = M.softmaxRowSteps(A.masked, f.T, i);
          for (let j = 0; j < f.T; j++) assert.ok(Math.abs(steps.probs[j] - A.att[i * f.T + j]) < 1e-7);
        }
        for (const which of ["ln1", "ln2", "ln_f"]) {
          const LN = D.layerNormData(f, which);
          sameBits(LN.steps.out, LN.output, `${label} ${which}`);
          const mv = M.meanVariance(LN.steps.normalized);
          assert.ok(Math.abs(mv.mean) < 1e-6, `${which}: media 0`);
          assert.ok(Math.abs(mv.variance - LN.steps.variance / (LN.steps.variance + LN.steps.eps)) < 1e-5, `${which}: varianza 1 a meno di ε`);
        }
        const FF = D.ffnData(f);
        sameBits(FF.pre, FF.engine.pre, `${label} ffn pre`);
        sameBits(FF.post, FF.engine.post, `${label} ReLU`);
        sameBits(FF.out, FF.engine.out, `${label} ffn out`);
        assert.equal(FF.off, FF.pre.filter((v) => !(v > 0)).length);
        assert.equal(FF.off + FF.on, f.hidden);
        sameBits(M.addVectors(FF.residIn, FF.out), FF.resid, `${label} residuo della feed-forward`);
        const blk = f.trace.blocks[b];
        sameBits(M.addVectors(D.rowOf(f.blockInput, t, f.C), D.rowOf(blk.proj, t, f.C)), D.rowOf(blk.resid1, t, f.C), `${label} residuo dell'attention`);
        const LG = D.logitsData(f);
        const sum = LG.probs.reduce((s, p) => s + p, 0);
        assert.ok(Math.abs(sum - 1) < 1e-6, "le probabilità sommano a 1");
      }
    }
  }
}

// ---------------------------------------------------------------- catalogo, ricerca, codice

test("catalogo e ricerca: id unici, ogni azione ha la sua voce, nell'indice del tab", () => {
  assert.equal(new Set(ACTION_IDS).size, ACTIONS.length);
  assert.deepEqual(ACTION_IDS, ["linear", "attention", "attention-v", "layernorm", "ffn", "logits", "training-step", "training-history"]);
  for (const a of ACTIONS) {
    assert.ok(CATEGORIES[a.category], `${a.id}: categoria`);
    assert.ok(a.title && a.summary && a.keywords.length >= 3, `${a.id}: testi`);
    assert.equal(actionCard(a.id), a);
  }
  assert.equal(searchEntries.length, ACTIONS.length);
  const keys = new Set(searchIndex.map((e) => `${e.kind}:${e.id}`));
  assert.equal(keys.size, searchIndex.length, "nessun doppione kind:id nell'indice del Laboratorio");
  for (const e of searchEntries) {
    assert.equal(e.kind, "action");
    assert.ok(keys.has(`action:${e.id}`), `${e.id} nell'indice del tab`);
  }
});

test("le righe di codice delle azioni esistono davvero nei loro snippet di model.json", () => {
  for (const [id, [snippet, line]] of Object.entries(CODE)) {
    assert.ok(spec.snippets[snippet], `${id}: snippet ${snippet}`);
    assert.ok(spec.snippets[snippet].code.includes(line), `${id}: «${line}» non è in ${snippet}`);
  }
  assert.deepEqual(codeFor("relu", "blocco 0"), ["ffwd", "nn.ReLU()  ·  blocco 0", ["nn.ReLU(),"]]);
});

test("niente innerHTML nei moduli delle azioni (prompt e nomi caricati sono testo non fidato)", async () => {
  for (const file of await readdir(ACTIONS_DIR)) {
    const src = await readFile(`${ACTIONS_DIR}${file}`, "utf8");
    assert.ok(!/innerHTML|insertAdjacentHTML|outerHTML/.test(src), `${file} usa HTML grezzo`);
  }
});

// ---------------------------------------------------------------- impostazioni

test("prompt: vocabolario, troncamento agli ultimi block_size, messaggi", () => {
  const vocab = engine.vocab;
  const ok = SET.checkPrompt("ROMEO:\nBut", vocab, 32);
  assert.equal(ok.ok, true);
  assert.deepEqual(ok.ids, engine.encode("ROMEO:\nBut"));
  assert.match(ok.message, /10 caratteri = 10 token/);
  const bad = SET.checkPrompt("Città è", vocab, 32);
  assert.equal(bad.ok, false);
  assert.deepEqual(bad.unknown, ["à", "è"]);
  assert.match(bad.message, /«à» «è»/);
  const long = SET.checkPrompt("a".repeat(40), vocab, 32);
  assert.equal(long.ok, true);
  assert.equal(long.ids.length, 32);
  assert.equal(long.truncated, 8);
  assert.equal(SET.checkPrompt("", vocab, 32).ok, false);
  assert.equal(SET.checkPrompt("abc", null, 32).ok, false);
  assert.equal(SET.checkPrompt("a\r\nb", vocab, 32).ids.length, 3, "\\r\\n diventa un a capo");
});

test("token, esempio di training e scelte", () => {
  assert.equal(SET.resolveToken(null, 17), 16);
  assert.equal(SET.resolveToken(40, 17), 16);
  assert.equal(SET.resolveToken(-3, 17), 0);
  assert.equal(SET.resolveToken(5, 17), 5);
  assert.deepEqual(SET.trainingExample([1, 2, 3], null), { context: [1, 2], target: 3, position: 1 });
  assert.deepEqual(SET.trainingExample([1, 2, 3], 0), { context: [1], target: 2, position: 0 });
  assert.equal(SET.trainingExample([1], null), null);
  const s = new SET.ActionSettings();
  const seen = [];
  s.on((c) => seen.push(c.key));
  assert.equal(s.set({ block: 3, head: 0 }), true, "head era già 0: cambia solo block");
  assert.deepEqual(seen, ["block"]);
  assert.equal(s.set({ block: 3 }), false);
  assert.deepEqual(s.clamp({ n_layer: 2, n_head: 2 }), { block: 1, head: 0 });
  assert.equal(s.set({ _listeners: null, nonEsiste: 1 }), false, "solo le scelte note");
  assert.equal(SET.visibleChar(" "), "␣");
  assert.equal(SET.visibleChar("\n"), "↵");
});

// ---------------------------------------------------------------- calcoli contro il motore

test("ogni passo dei calcoli coincide con il motore JS bit per bit (pesi finali, prompt d'esempio)", () => {
  checkAgainstEngine(engine, ids, "finali");
});

test("anche su uno snapshot del training (step 500) e su un altro prompt", async () => {
  const snap = await loadEngine(DATA_DIR, fsLoaders);
  await snap.loadSnapshot(500);
  checkAgainstEngine(snap, snap.encode("First Citizen:\nBefore we proceed"), "snapshot 500");
});

test("anche su un modello di forma diversa (V 11, T 6, C 12, 3 head, 2 blocchi): run compatibili caricati", () => {
  const cfg = { vocab_size: 11, block_size: 6, n_embd: 12, n_head: 3, n_layer: 2, dropout: 0 };
  const eng = syntheticEngine(cfg);
  checkAgainstEngine(eng, [3, 1, 4, 1, 5, 9], "sintetico");
  const f = D.forwardData(eng, [3, 1, 4], { block: 1, head: 2, token: 99 });
  assert.equal(f.t, 2, "token oltre la fine: l'ultimo");
  assert.equal(f.names.key, "blocks.1.sa.heads.2.key.weight");
  assert.equal(D.linearData(f, "ffn1").outDim, 48, "4 · n_embd neuroni nascosti");
  assert.equal(D.linearData(f, "lm_head").outDim, 11);
});

test("la riga di una nn.Linear: prodotti, somme parziali, bias", () => {
  const x = Float32Array.from([1, -2, 0.5]);
  const W = Float32Array.from([0.1, 0.2, 0.3, -1, 0, 2]);
  const b = Float32Array.from([0.05, -0.5]);
  const r = M.linearRow(x, W, b, 3, 1);
  assert.deepEqual(Array.from(r.products), [-1, -0, 1]);
  assert.equal(r.sum, 0);
  assert.equal(r.bias, -0.5);
  assert.equal(r.value, -0.5);
  assert.deepEqual(Array.from(M.runningSums(r.products)), [0, -1, -1, 0]);
  assert.equal(M.linearRow(x, W, null, 3, 0).bias, null);
  assert.deepEqual(Array.from(M.linearForward(x, W, b, 3, 2)), [Math.fround(x[0] * W[0] + x[1] * W[1] + x[2] * W[2] + b[0]), -0.5]);
});

test("ReLU, topK, softmax e cross-entropy su numeri scritti a mano", () => {
  const r = M.relu(Float32Array.from([-1, 0, 2, -0.5, 3]));
  assert.deepEqual(Array.from(r.post), [0, 0, 2, 0, 3]);
  assert.equal(r.off, 3);
  assert.equal(r.on, 2);
  assert.deepEqual(M.topK([0.1, 0.5, 0.5, 0.2], 2), [{ id: 1, value: 0.5 }, { id: 2, value: 0.5 }]);
  const ce = M.crossEntropy(Float32Array.from([0, 0, Math.log(2)]), 2);
  assert.ok(Math.abs(ce.p - 0.5) < 1e-7);
  assert.ok(Math.abs(ce.loss - Math.log(2)) < 1e-7);
  const masked = M.causalMask(Float32Array.from([1, 2, 3, 4]), 2);
  assert.deepEqual(Array.from(masked), [1, -Infinity, 3, 4]);
  assert.deepEqual(Array.from(M.softmaxRows(masked, 2).subarray(0, 2)), [1, 0], "il futuro diventa 0");
});

// ---------------------------------------------------------------- gradiente e aggiornamento

test("gradiente di lm_head contro le differenze finite (in double): ∂L/∂W, ∂L/∂b, ∂L/∂h", () => {
  const rng = mulberry32(3);
  const V = 7;
  const C = 5;
  const h = Float64Array.from({ length: C }, () => rng() * 2 - 1);
  const W = Float64Array.from({ length: V * C }, () => rng() - 0.5);
  const b = Float64Array.from({ length: V }, () => rng() - 0.5);
  const target = 4;
  const logits = M.linearForward(h, W, b, C, V);
  const { probs } = M.crossEntropy(logits, target);
  const g = M.lmHeadGradient(probs, target, h);
  const dh = M.lmHeadInputGradient(g.dLogits, W, C);
  const eps = 1e-6;
  const fd = (arr, k) => {
    const keep = arr[k];
    arr[k] = keep + eps;
    const up = lossDouble(h, W, b, target, V, C);
    arr[k] = keep - eps;
    const down = lossDouble(h, W, b, target, V, C);
    arr[k] = keep;
    return (up - down) / (2 * eps);
  };
  for (let k = 0; k < V * C; k++) assert.ok(Math.abs(fd(W, k) - g.dW[k]) < 2e-6, `dW[${k}]: ${fd(W, k)} vs ${g.dW[k]}`);
  for (let c = 0; c < V; c++) assert.ok(Math.abs(fd(b, c) - g.db[c]) < 2e-6, `db[${c}]`);
  for (let j = 0; j < C; j++) assert.ok(Math.abs(fd(h, j) - dh[j]) < 2e-6, `dh[${j}]`);
  assert.ok(g.dLogits[target] < 0 && g.dLogits.every((v, c) => c === target || v > 0), "negativo solo sul carattere giusto");
  assert.ok(Math.abs(g.dLogits.reduce((s, v) => s + v, 0)) < 1e-6, "p − y somma a 0");
});

test("passo di training sul motore vero: gradienti contro le differenze finite, la loss scende", () => {
  const lr = TR.learningRateAt(trace).lr;
  const T = TR.trainingStepData(engine, ids, { token: null, lr });
  assert.equal(T.targetChar, "!");
  assert.deepEqual(T.context, ids.slice(0, -1));
  assert.ok(T.loss > 0 && T.p > 0 && T.p < 1);
  assert.ok(Math.abs(T.loss + Math.log(T.p)) < 1e-9);
  assert.ok(T.sgd.loss < T.loss, `SGD: ${T.sgd.loss} < ${T.loss}`);
  assert.ok(T.adamw.loss < T.loss, `AdamW: ${T.adamw.loss} < ${T.loss}`);
  assert.ok(T.sgd.p > T.p);
  // la loss dopo SGD in forma chiusa (h non cambia: si è aggiornata solo lm_head) = quella del motore
  const closed = M.crossEntropy(M.linearForward(T.h, T.sgd.W, T.sgd.b, T.C, T.V), T.target).loss;
  assert.ok(Math.abs(closed - T.sgd.loss) < 1e-12, "motore e forma chiusa danno la stessa loss");
  // differenze finite sul motore vero: lm_head, e un passo più indietro γ e β di ln_f
  const copy = TR.privateEngine(engine);
  const base = Float32Array.from(engine.weights);
  const lossWith = (name, k, delta) => {
    const flat = Float32Array.from(base);
    flat[engine.paramViews.get(name).offset + k] += delta;
    copy.setWeights(flat);
    return TR.engineLoss(copy, T.context, T.target).loss;
  };
  const fd = (name, k, eps = 2e-3) => (lossWith(name, k, eps) - lossWith(name, k, -eps)) / (2 * eps);
  for (const j of [0, 17, 40, 63]) {
    const k = T.target * T.C + j;
    assert.ok(Math.abs(fd("lm_head.weight", k) - T.grad.dW[k]) < 2e-3 * (1 + Math.abs(T.grad.dW[k])), `lm_head.weight[!, ${j}]`);
    assert.ok(Math.abs(fd("ln_f.bias", j) - T.lnF.dBeta[j]) < 2e-3 * (1 + Math.abs(T.lnF.dBeta[j])), `ln_f.bias[${j}] = ∂L/∂h[${j}]`);
    assert.ok(Math.abs(fd("ln_f.weight", j) - T.lnF.dGamma[j]) < 2e-3 * (1 + Math.abs(T.lnF.dGamma[j])), `ln_f.weight[${j}]`);
  }
  for (const c of [0, T.target]) assert.ok(Math.abs(fd("lm_head.bias", c) - T.grad.db[c]) < 2e-3, `lm_head.bias[${c}]`);
  // il motore dello stato non è stato toccato
  assert.deepEqual(Array.from(engine.weights.subarray(0, 64)), Array.from(base.subarray(0, 64)));
  assert.equal(engine.weightsSource, "trained");
});

test("SGD e primo passo di AdamW: come PyTorch (torch.optim.AdamW, step 1)", () => {
  const w = Float32Array.from([0.5, -0.25, 1, 0, 2]);
  const g = Float32Array.from([0.1, -3, 1e-9, 0, -0.02]);
  const lr = 1e-3;
  assert.deepEqual(Array.from(M.sgdStep(w, g, lr)), Array.from(w, (v, i) => Math.fround(v - lr * g[i])));
  // _single_tensor_adamw di PyTorch in float32: decadimento, momenti, correzioni, addcdiv
  const f = Math.fround;
  const { beta1, beta2, eps, weightDecay } = M.ADAMW_DEFAULTS;
  const expected = Array.from(w, (p0, i) => {
    let p = f(p0 * f(1 - lr * weightDecay));
    const m = f((1 - beta1) * g[i]);
    const v = f((1 - beta2) * f(g[i] * g[i]));
    const denom = f(f(Math.sqrt(v)) / f(Math.sqrt(1 - beta2)) + eps);
    p = f(p - f(lr / (1 - beta1)) * f(m / denom));
    return p;
  });
  const got = M.adamwFirstStep(w, g, { lr });
  for (let i = 0; i < w.length; i++) assert.ok(Math.abs(got[i] - expected[i]) < 1e-7, `AdamW[${i}]: ${got[i]} vs ${expected[i]}`);
  // ≈ lr nella direzione giusta per i gradienti non nulli, indipendente dalla loro grandezza
  assert.ok(Math.abs(w[1] * (1 - lr * weightDecay) + lr - got[1]) < 1e-6);
  assert.ok(Math.abs(w[0] * (1 - lr * weightDecay) - lr - got[0]) < 1e-6);
  assert.equal(got[3], 0, "gradiente nullo: si muove solo il weight decay (0 resta 0)");
  const st = M.updateStats(w, got);
  assert.equal(st.count, 5);
  assert.ok(st.maxAbs <= lr * 1.01 + 2 * lr * weightDecay);
});

test("learning rate e storia del training da trace.json", () => {
  assert.deepEqual(TR.learningRateAt(trace), { lr: 0.001, source: `trace.json, metrics.lr allo step ${trace.metrics.steps.at(-1)}` });
  assert.equal(TR.learningRateAt(trace, 500).lr, 0.001);
  assert.deepEqual(TR.learningRateAt({ train_config: { learning_rate: 3e-4 } }), { lr: 3e-4, source: "trace.json, train_config.learning_rate" });
  assert.equal(TR.learningRateAt(null), null);
  const h = TR.historySeries(trace);
  assert.ok(h.train.length >= 100 && h.train.length <= 170, `${h.train.length} punti di train`);
  assert.equal(h.val.length, trace.evals.steps.length);
  assert.equal(h.snapshots.length, 11);
  assert.deepEqual(h.snapshots.map((s) => s.step), [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.equal(h.maxStep, 5000);
  const first = trace.train_loss.slice(0, h.group);
  assert.ok(Math.abs(h.train[0][1] - first.reduce((s, v) => s + v, 0) / first.length) < 1e-12, "media del primo gruppo di step");
  assert.ok(h.yMin < 2 && h.yMax > 4, "la loss va da ~4,4 a ~1,6");
  assert.deepEqual(TR.valLossNear(trace, 480), { step: 500, val: trace.evals.val[5], train: trace.evals.train[5] });
});

// ---------------------------------------------------------------- geometria e inquadrature

test("dove stanno le carte: x sopra le colonne di W, l'uscita accanto alle righe, dopo la bias", () => {
  const W = { x0: 1, x1: 3.56, y0: 0, y1: 0.64, rows: 16, cols: 64, cellW: 0.04, cellH: 0.04 };
  const x = P.rowAbove(W);
  assert.equal(x.x0, W.x0);
  assert.ok(Math.abs(x.x1 - W.x1) < 1e-12, "stesse colonne di W");
  assert.ok(x.y0 > W.y1);
  const bias = { x0: 3.6, x1: 3.72, y0: 0, y1: 0.64 };
  const y = P.columnRight(W, { after: bias });
  assert.ok(y.x0 > bias.x1);
  assert.equal(y.rows, 16);
  assert.equal(y.y1, W.y1);
  assert.ok(Math.abs(y.y0 - W.y0) < 1e-12, "stesse righe di W");
  const c = P.cellCenter(W, 2, 3);
  assert.ok(Math.abs(c.x - 1.14) < 1e-12 && Math.abs(c.y - 0.54) < 1e-12);
  assert.deepEqual(P.cellsRect(W, 0, 1, 0, 2), { x0: 1, x1: 1.08, y0: 0.6, y1: 0.64 });
  assert.deepEqual(P.unionRect(null, { x0: 0, x1: 1, y0: 0, y1: 1 }, { x0: 2, x1: 3, y0: -1, y1: 0 }), { x0: 0, x1: 3, y0: -1, y1: 1 });
  assert.equal(P.unionRect(null), null);
  const tril = { x0: 0, x1: 1.28, y0: 0, y1: 1.28, rows: 32, cols: 32, cellW: 0.04, cellH: 0.04 };
  const scores = P.subRect(tril, 17, 17);
  assert.equal(scores.x0, 0);
  assert.equal(scores.y1, 1.28);
  assert.ok(Math.abs(scores.y0 - (1.28 - 0.68)) < 1e-12);
  assert.ok(P.blockLeft(scores, { cols: 16 }).x1 < scores.x0, "Q a sinistra dei punteggi");
  assert.ok(P.blockAbove(scores, { rows: 16 }).y0 > scores.y1, "Kᵀ sopra");
});

test("le carte stanno appena davanti a ciò che coprono (zAbove)", () => {
  const boxes = [
    { kind: "module", x0: 0, x1: 10, y0: 0, y1: 10, z: 0.1 },
    { kind: "param", x0: 1, x1: 3, y0: 1, y1: 2, z: 0.2, matrix: { x0: 1, x1: 3, y0: 1, y1: 1.9 } },
    { kind: "module", x0: 20, x1: 30, y0: 0, y1: 1, z: 0.9 },
  ];
  assert.ok(Math.abs(P.zAbove(boxes, { x0: 2, x1: 2.5, y0: 1.2, y1: 1.5 }) - (0.2 + 0.006 + P.CARD_LIFT)) < 1e-12);
  assert.ok(Math.abs(P.zAbove(boxes, { x0: 5, x1: 6, y0: 5, y1: 6 }) - (0.1 + 0.006 + P.CARD_LIFT)) < 1e-12);
  assert.equal(P.zAbove([], { x0: 0, x1: 1, y0: 0, y1: 1 }), P.CARD_LIFT);
});

test("inquadratura: il rettangolo entra nella parte libera, sotto i comandi e sopra lavagna e player", () => {
  const rect = { x0: -1, x1: 1, y0: -0.5, y1: 0.5 };
  const o = { rect, fovDeg: 45, width: 800, height: 900, top: 100, bottom: 360, margin: 1.1 };
  const pose = P.framePose(o);
  const free = 900 - 100 - 360;
  assert.equal(pose.free, free);
  assert.ok(pose.scale * 1 * 1.1 <= free + 1e-9 && pose.scale * 2 * 1.1 <= 800 + 1e-9, "entra in altezza e in larghezza");
  // dove cade il centro del rettangolo sullo schermo: (target − centro) · scala px sopra il centro del canvas
  const centerY = 450 + (pose.target.y - 0) * pose.scale;
  assert.ok(Math.abs(centerY - (100 + free / 2)) < 1e-6, "al centro della parte libera");
  const tanHalf = Math.tan((45 * Math.PI) / 360);
  assert.ok(Math.abs(pose.distance - 900 / (2 * pose.scale * tanHalf)) < 1e-9);
  const left = P.framePose({ ...o, leftPx: 120 });
  assert.ok(left.target.x < pose.target.x, "con le etichette a sinistra il rettangolo si sposta a destra");
  assert.ok(left.scale <= pose.scale);
  const squeezed = P.framePose({ ...o, top: 600, bottom: 600 });
  assert.ok(Math.abs(squeezed.free - 900 * 0.3) < 1e-9, "resta almeno il 30% dell'altezza");
  const tiny = P.framePose({ ...o, rect: { x0: 0, x1: 1e-4, y0: 0, y1: 1e-4 } });
  assert.equal(tiny.scale, 2600, "mai più vicino di così");
  assert.equal(P.readableColumns(766, { minCellPx: 54, labelPx: 60, min: 4, max: 14 }), 11);
  assert.equal(P.readableColumns(200, { minCellPx: 54, min: 4 }), 4);
});

test("lavagna, codice e pad su schermo largo: il pad resta nell'angolo, niente inseguimenti", () => {
  // 1280 × 800 con la sidebar aperta (misurato): stage 900 × 748, barra del player, codice a destra
  const sw = 900;
  const sh = 748;
  const bar = { left: 90, right: 810, top: 535, bottom: 704 };
  const code = { left: 476, right: 886, top: 56, bottom: 507 };
  const padSize = { w: 190, h: 146 };
  const corner = { left: 696, right: 886, top: 379, bottom: 525 };
  const beside = { left: 274, right: 464, top: 379, bottom: 525 }; // accanto al codice (dove il nucleo lo sposta se il codice lo tocca)
  const o = { sw, sh, bar, code, padSize, padBottomFn: padBottom };
  const a = P.deskLayout({ ...o, pad: corner });
  const b = P.deskLayout({ ...o, pad: beside });
  // la disposizione non dipende da dove sta il pad VERO in questo momento: niente giri a vuoto
  assert.deepEqual(a, b);
  assert.equal(a.fallback, false);
  assert.deepEqual(a.padSpot, corner, "il posto nell'angolo, calcolato come il nucleo (padBottom)");
  assert.equal(a.codeMax, corner.top - 10 - code.top);
  // con il codice alto codeMax il nucleo sceglie l'angolo: le stesse condizioni di navPad.layout()
  const codeBottom = code.top + a.codeMax;
  const right = 14;
  const clashes = sw - right > code.left && corner.top < codeBottom + 8 && sh - (sh - corner.bottom) > code.top;
  assert.equal(clashes, false, "il codice non tocca il pad nell'angolo");
  assert.equal(a.left, bar.left, "allineata alla barra");
  assert.equal(a.maxW, corner.left - 10 - bar.left, "finisce 10 px prima del pad");
  // il codice finisce 8 px sopra la lavagna se le sta sopra, mai sotto CODE_FLOOR_PX
  const board = { left: a.left, right: a.left + a.maxW, top: 372, bottom: 527 };
  assert.equal(P.codeAboveBoard(a.codeMax, code, board), 372 - 8 - code.top);
  assert.equal(P.codeAboveBoard(a.codeMax, code, { ...board, top: 120 }), P.CODE_FLOOR_PX);
  assert.equal(P.codeAboveBoard(a.codeMax, code, { ...board, right: 470 }), a.codeMax, "lavagna non sotto il codice");
  assert.equal(P.codeAboveBoard(null, code, board), null);
  // la barra più alta (didascalia su tre righe) alza il pad e accorcia il codice, la lavagna resta larga
  const tall = P.deskLayout({ ...o, bar: { ...bar, top: 500 }, pad: corner });
  assert.equal(tall.codeMax, a.codeMax - 35);
  assert.equal(tall.maxW, a.maxW);

  // finestra bassa: un codice di CODE_MIN_PX non sta sopra il pad nell'angolo → codice costante, il pad va
  // dove lo mette il nucleo, la lavagna evita il pad vero e il codice (e parte dal bordo se resta poco posto)
  const low = { ...o, sh: 548, bar: { ...bar, top: 335, bottom: 504 } };
  const f1 = P.deskLayout({ ...low, pad: { ...beside, top: 179, bottom: 325 } });
  assert.equal(f1.fallback, true);
  assert.equal(f1.codeMax, P.CODE_MIN_PX);
  assert.equal(f1.left, 16, "poco posto fra la barra e il pad: dal bordo sinistro");
  assert.equal(f1.maxW, beside.left - 10 - 16);
  assert.equal(P.codeAboveBoard(f1.codeMax, null, board), P.CODE_MIN_PX);

  // senza codice: il pad nell'angolo, nessun limite per il codice
  const noCode = P.deskLayout({ ...o, code: null, pad: corner });
  assert.equal(noCode.codeMax, null);
  assert.equal(noCode.maxW, a.maxW);
  // schermo largo: il pad non tocca la barra (sta in basso, accanto): la lavagna è larga quanto la barra
  const wide = P.deskLayout({ sw: 1220, sh: 848, bar: { left: 250, right: 970, top: 635, bottom: 804 }, code: { left: 766, right: 1206, top: 56, bottom: 500 }, pad: { left: 1016, right: 1206, top: 688, bottom: 834 }, padSize, padBottomFn: padBottom });
  assert.equal(wide.maxW, 720);
  assert.equal(wide.codeMax, 848 - 14 - 146 - 10 - 56);
  // grafici aperti: il nucleo non guarda il codice, conta il pad vero; la lavagna finisce prima dei grafici
  const charts = P.deskLayout({ ...o, chartsLeft: 440, bar: { left: 16, right: 424, top: 535, bottom: 704 }, pad: { left: 238, right: 428, top: 379, bottom: 525 } });
  assert.equal(charts.codeMax, null);
  assert.equal(charts.maxW, 238 - 10 - 16);
  assert.ok(charts.left + charts.maxW <= 440 - 16);
  // mai più stretta di 200 px
  assert.equal(P.deskLayout({ ...o, pad: corner, bar: { ...bar, left: 16 }, sw: 300 }).maxW, 200);
});

test("numeri all'italiana per la lavagna", () => {
  assert.equal(F.num(-0.12345), "−0,1235");
  assert.equal(F.num(0.00002), "2,0 · 10⁻⁵");
  assert.equal(F.signed(0.5, 2), "+0,50");
  assert.equal(F.signed(-0.5, 2), "−0,50");
  assert.equal(F.sci(0.000034), "3,4 · 10⁻⁵");
  assert.equal(F.sci(0), "0");
  assert.equal(F.pct(0.4123), "41,2%");
  assert.equal(F.pct(0.0004), "0,04%");
  assert.equal(F.int(209729), "209.729");
  assert.equal(F.factor(-0.5, 1), "(−0,5)");
  assert.equal(F.factor(0.5, 1), "0,5");
  assert.equal(F.vectorPreview(Float32Array.from([0.81, -1.2, 0.33, 4, 5]), { count: 3 }), "[0,81; −1,20; 0,33; …]");
});

test("stepKit: contatori che cambiano solo quando serve, passi con le misure", () => {
  assert.equal(progressCount(0, 64), 0);
  assert.equal(progressCount(1, 64), 64);
  assert.equal(progressCount(0.85, 64), 64, "una pausa alla fine");
  const seen = [];
  const f = onChange((v) => seen.push(v));
  [0, 0, 1, 1, 2].forEach(f);
  assert.deepEqual(seen, [0, 1, 2]);
  const calls = [];
  const stage = { reduced: false, step: () => calls.push("step"), board: { show: () => calls.push("board") }, code: () => calls.push("code"), frameSoon: () => calls.push("frame"), syncLayout: () => calls.push("sync"), noteEnter: () => calls.push("note"), noteUpdate: () => calls.push("noteUpdate") };
  const [s] = makeSteps(stage, [{ title: "t", caption: "c", scene: () => calls.push("scene"), board: () => [], code: ["head", "t", []], update: (p) => calls.push(`u${p}`), frame: () => ({ x0: 0, x1: 1, y0: 0, y1: 1 }) }], { id: "x" });
  s.enter();
  s.update(0.5);
  assert.deepEqual(calls, ["step", "scene", "board", "code", "u0", "frame", "note", "u0.5", "noteUpdate"]);
  calls.length = 0;
  stage.reduced = true;
  s.enter();
  s.update(0.3);
  assert.deepEqual(calls, ["step", "scene", "board", "code", "u1", "frame", "note"], "riduci il movimento: subito lo stato finale");
});

// ---------------------------------------------------------------- tutte le azioni su una regia finta

/** Canvas finto per three.js e le etichette (misura il testo: 0,6 em per carattere). */
function fakeCanvas() {
  const canvas = { width: 0, height: 0, texts: [] };
  let font = "10px sans-serif";
  const ctx = new Proxy(
    {
      get font() {
        return font;
      },
      set font(v) {
        font = v;
      },
      measureText: (t) => ({ width: String(t).length * 0.6 * Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? 10) }),
      fillText: (t) => canvas.texts.push(String(t)),
    },
    { get: (target, key) => (key in target ? target[key] : () => {}), set: (target, key, value) => ((target[key] = value), true) },
  );
  canvas.getContext = () => ctx;
  return canvas;
}

const run = await loadBuiltinRun({ baseUrl: DATA_DIR, loadJson, loadBuffer });
const tree = buildModuleTree(run.params, { snippets: spec.snippets, config: run.config });
const layout = layoutModel(tree, { config: run.config, snippets: spec.snippets });
const scene = new LabScene({ createCanvas: fakeCanvas, numbers: new NumbersGroup({ createCanvas: fakeCanvas }) });
scene.build({ run, layout, view: { tensors: run.tensors } });

const BAD_TEXT = /undefined|NaN|null|\[object/;

/** La regia finta: le stesse operazioni di ActionStage, con controlli su ogni carta e ogni testo. */
function fakeStage(problems) {
  const cards = new Map();
  const check = (cond, msg) => cond || problems.push(msg);
  const text = (t, where) => check(typeof t === "string" && !BAD_TEXT.test(t), `${where}: testo «${t}»`);
  const finiteRect = (r, where) => check(r && [r.x0, r.x1, r.y0, r.y1].every(Number.isFinite) && r.x1 > r.x0 && r.y1 > r.y0, `${where}: rettangolo ${JSON.stringify(r)}`);
  const blocks = (list, where) => {
    for (const b of list ?? []) {
      if (!b) continue;
      for (const v of Object.values(b)) {
        if (typeof v === "string") text(v, where);
        else if (Array.isArray(v)) for (const s of v) typeof s === "string" ? text(s, where) : s && typeof s === "object" && typeof s.text === "string" && text(s.text, where);
      }
      if (b.vector) check(b.vector.values.length > 0 && Array.from(b.vector.values).every(Number.isFinite), `${where}: vettore con numeri non finiti`);
      if (b.bars) for (const bar of b.bars) check(Number.isFinite(bar.value), `${where}: barra ${bar.label}`);
      if (b.table) for (const row of b.table.rows) for (const c of row) text(String(c), where);
    }
  };
  const stage = {
    scene,
    cards,
    reduced: false,
    touchedStep: false,
    ownStepChange: false,
    freeArea: () => ({ width: 900, height: 820, top: 100, bottom: 330 }),
    step() {
      for (const c of cards.values()) c.visible = false;
      scene.showValues(null);
      for (const name of scene.matrices.keys()) scene.highlightCells(name, null);
    },
    card(key, s) {
      const r = P.cardRect(s.rect);
      finiteRect(r, `carta ${key}`);
      check(s.data && s.data.length >= r.rows * r.cols, `carta ${key}: ${s.data?.length} valori per ${r.rows} × ${r.cols}`);
      if (s.data) {
        for (let i = 0; i < r.rows * r.cols; i++) {
          const masked = s.mask?.(Math.floor(i / r.cols), i % r.cols);
          if (!masked && !Number.isFinite(s.data[i])) {
            problems.push(`carta ${key}: valore ${s.data[i]} in ${i}`);
            break;
          }
        }
      }
      if (s.label) text(s.label, `etichetta ${key}`);
      const card = {
        rows: r.rows,
        cols: r.cols,
        rect: r,
        box: r,
        z: 0.3,
        visible: true,
        label: s.label,
        labelSide: s.labelSide ?? "top",
        data: s.data,
        reveal: () => {},
        mark: (rects) => (rects ?? []).forEach((q) => check(q.r0 >= 0 && q.c0 >= 0 && q.r1 <= r.rows && q.c1 <= r.cols && q.r1 > q.r0 && q.c1 > q.c0, `carta ${key}: riquadro fuori ${JSON.stringify(q)}`)),
        setData: (d) => check(d.length >= r.rows * r.cols, `carta ${key}: setData corto`),
        cellCenter: (i, j) => ({ x: r.x0 + (j + 0.5) * r.cellW, y: r.y1 - (i + 0.5) * r.cellH, z: 0.3 }),
        labelReservePx: () => 60,
      };
      cards.set(key, card);
      return card;
    },
    strip(key, texts, positions) {
      check(texts.length === positions.length && positions.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)), `fila ${key}`);
      texts.forEach((t) => text(t, `fila ${key}`));
    },
    highlight(name, cells, options) {
      check(Boolean(scene.matrixOf(name)), `velo su una matrice che la scena non ha: ${name}`);
      check(scene.highlightCells(name, cells, options), `velo rifiutato su ${name}`);
    },
    link: (segments) => segments.forEach(([a, b]) => check([a.x, a.y, b.x, b.y].every(Number.isFinite), "linea con punti non finiti")),
    clearLinks() {},
    outline: (rects) => (Array.isArray(rects) ? rects : [rects]).forEach((r) => finiteRect(r, "riquadro")),
    showValues(map, label) {
      const res = scene.showValues(map, { colormap: "pesi", label });
      check(res.missing.length === 0, `valori su tensori assenti: ${res.missing}`);
      text(label, "etichetta dei valori");
    },
    clearValues: () => scene.showValues(null),
    board: { show: (b) => blocks(b, "lavagna"), setLive: (k, t) => text(t, `lavagna viva ${k}`) },
    code(snippet, title, highlight) {
      check(Boolean(spec.snippets[snippet]), `snippet ${snippet}`);
      text(title, "titolo del codice");
      for (const line of highlight) check(spec.snippets[snippet]?.code.includes(line), `riga «${line}» non in ${snippet}`);
    },
  };
  return stage;
}

/**
 * Esegue tutti i passi come il player: enter (scene, lavagna, codice), update da 0 a 1, inquadratura.
 * `between`: attesa dopo ogni passo (la storia del training cambia lo step: i pesi arrivano dopo).
 */
async function runSteps(stage, specs, problems, label, { between = null } = {}) {
  assert.ok(Array.isArray(specs) && specs.length >= 1, `${label}: nessun passo`);
  for (const [k, s] of specs.entries()) {
    const where = `${label} passo ${k + 1} «${s.title}»`;
    if (BAD_TEXT.test(s.title ?? "") || BAD_TEXT.test(s.caption ?? "") || !s.caption) problems.push(`${where}: titolo o didascalia`);
    stage.step();
    s.scene?.();
    stage.board.show(s.board?.());
    if (s.code) stage.code(...s.code);
    for (const p of [0, 0.1, 0.35, 0.6, 0.9, 1]) s.update?.(p);
    const rect = s.frame?.();
    if (rect) {
      if (![rect.x0, rect.x1, rect.y0, rect.y1].every(Number.isFinite) || rect.x1 <= rect.x0 || rect.y1 <= rect.y0) problems.push(`${where}: inquadratura ${JSON.stringify(rect)}`);
    }
    if (between) await between();
  }
}

test("tutti i passi delle azioni del passaggio in avanti, per ogni nn.Linear e due token", async () => {
  const problems = [];
  const stage = fakeStage(problems);
  for (const token of [null, 3]) {
    const f = D.forwardData(engine, ids, { token, block: 2, head: 1 });
    for (const choice of SET.LINEAR_CHOICES.map((c) => c.id)) await runSteps(stage, linearSteps(stage, f, choice), problems, `linear/${choice}/t${token}`);
    await runSteps(stage, attentionSteps(stage, f), problems, `attention/t${token}`);
    await runSteps(stage, attentionValueSteps(stage, f), problems, `attention-v/t${token}`);
    for (const which of ["ln1", "ln2", "ln_f"]) await runSteps(stage, layerNormSteps(stage, f, which), problems, `layernorm/${which}/t${token}`);
    await runSteps(stage, ffnSteps(stage, f), problems, `ffn/t${token}`);
    await runSteps(stage, logitsSteps(stage, f), problems, `logits/t${token}`);
  }
  assert.deepEqual(problems, []);
  stage.step();
  assert.equal(scene.customValues.size, 0, "a fine passo niente valori lasciati sulla scena");
});

test("«Un passo di training»: tutti i passi, i pesi della scena tornano quelli veri", async () => {
  const problems = [];
  const stage = fakeStage(problems);
  const state = { step: null, builtin: run };
  const specs = trainingStepSteps(stage, { tab: { state }, engine, ids, token: null, run });
  await runSteps(stage, specs, problems, "training-step");
  assert.deepEqual(problems, []);
  assert.equal(specs.length, 9);
  stage.step();
  assert.equal(scene.matrixOf("lm_head.weight").data, run.tensor("lm_head.weight").data, "lm_head torna ai pesi veri");
  assert.equal(trainingStepSteps(stage, { tab: { state }, engine, ids: ids.slice(0, 1), token: null, run }), null, "un carattere solo non è un esempio");
});

test("«I pesi durante il training»: gli 11 snapshot sulla scena, poi lo step di prima", async () => {
  const problems = [];
  const stage = fakeStage(problems);
  const state = new LabState({ loadBuiltin: async () => run });
  await state.init();
  const seen = [];
  state.on("weights", ({ step, view }) => {
    seen.push(step);
    scene.setWeights(view.tensors);
  });
  const settings = new SET.ActionSettings({ block: 1, head: 3 });
  const specs = await historySteps(stage, { state }, settings);
  assert.equal(specs.length, 12, "11 snapshot e il riepilogo");
  await runSteps(stage, specs, problems, "training-history", { between: () => new Promise((r) => setTimeout(r, 5)) });
  assert.deepEqual(problems, []);
  assert.equal(stage.touchedStep, true);
  assert.deepEqual([...new Set(seen)], [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.equal(scene.matrixOf("blocks.0.ffwd.net.0.weight").data, (await run.loadSnapshot(5000)).tensors.get("blocks.0.ffwd.net.0.weight").data);
  await state.setStep(null);
  scene.setWeights(run.tensors);
  // un run senza storia: un passo solo che spiega
  const bare = new LabState({ loadBuiltin: async () => ({ ...run, id: run.id, name: run.name, trace: null, snapshots: [], steps: [], tensors: run.tensors, summary: () => run.summary() }) });
  await bare.init();
  const none = await historySteps(stage, { state: bare }, settings);
  assert.equal(none.length, 1);
  assert.match(none[0].caption, /solo i pesi finali/);
});

// ---------------------------------------------------------------- carte e file di token (three.js, canvas finto)

test("carta: celle scoperte poco a poco, maschera, riquadri, etichetta a grandezza costante, pulizia", () => {
  const group = new NumbersGroup({ createCanvas: fakeCanvas });
  const data = Float32Array.from({ length: 12 }, (_, i) => i - 6);
  const card = new Card({ rect: { x0: 0, y1: 1, rows: 3, cols: 4 }, data, label: "x · W[0]", labelSide: "left", group, z: 0.3, createCanvas: fakeCanvas });
  assert.deepEqual([card.rows, card.cols], [3, 4]);
  assert.ok(Math.abs(card.rect.x1 - 0.16) < 1e-12 && Math.abs(card.rect.y0 - 0.88) < 1e-12);
  assert.equal(card.m.data, data);
  card.reveal(5);
  assert.equal(card.m.masked(1, 0), false);
  assert.equal(card.m.masked(1, 1), true, "dalla sesta cella: da calcolare");
  card.configure({ reveal: Infinity, mask: (r, c) => c > r });
  assert.equal(card.m.masked(0, 1), true);
  assert.equal(card.m.masked(1, 0), false);
  card.configure({ reveal: (r) => r === 2 });
  assert.equal(card.m.masked(2, 0), false);
  assert.equal(card.m.masked(0, 0), true);
  card.mark([{ r0: 0, r1: 1, c0: 0, c1: 4 }]);
  assert.equal(card.marks.visible, true);
  card.mark(null);
  assert.equal(card.marks.visible, false);
  card.scaleLabel(0.05);
  assert.ok(card.label.position.x < card.rect.x0, "l'etichetta a sinistra finisce prima delle celle");
  assert.ok(Math.abs(card.label.scale.y - 0.05) < 1e-12);
  assert.ok(card.labelReservePx() > LABEL_PX, "lo spazio per l'etichetta in pixel");
  const c = card.cellCenter(2, 3);
  assert.ok(Math.abs(c.x - 0.14) < 1e-12 && Math.abs(c.y - 0.9) < 1e-12);
  card.dispose();
  assert.equal(card.object3d.parent, null);
  const strip = new TokenStrip(["R", "O", "↵"], [{ x: 0, y: 0 }, { x: 0.04, y: 0 }, { x: 0.08, y: 0 }], { createCanvas: fakeCanvas, marked: 2 });
  assert.ok(strip.object3d.children.length >= 1);
  strip.dispose();
  group.dispose();
});
