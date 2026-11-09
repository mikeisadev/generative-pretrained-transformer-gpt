// Test della vista "Pesi numerici" del Transformer (js/tabs/transformer/views/numbersModel.js):
// aggancio agli snapshot e stima lineare con l'etichetta giusta, scelta della sorgente, verifica
// "snapshot finale = pesi finali arrotondati a float16" sui file veri, righe e colonne di ogni tensore,
// elenco come il codice Python, link tensor:<nome>@<step>, calendario dei "pesi che nascono", carte delle
// head e, con three.js della cartella vendor, la mappatura nome del tensore ↔ heatmap della scena.
//   node --test webapp/tests/*.test.mjs
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { decodeFloat16, decodeFloat32 } from "../js/engine/weights.js";
import {
  BIRTH_VIEW_ID,
  FINAL_SOURCE,
  NUMBERS_VIEW_ID,
  birthSchedule,
  cardDepths,
  cellMeaning,
  float16RoundingCheck,
  formatTensorTarget,
  headCard,
  labTarget,
  parseTensorTarget,
  sameSource,
  shownChar,
  snapToSnapshot,
  sourceLabel,
  tensorAxes,
  tensorGroups,
  tensorKind,
  trainingSource,
} from "../js/tabs/transformer/views/numbersModel.js";

const DATA = fileURLToPath(new URL("../data/", import.meta.url));
const HAS_DATA = existsSync(`${DATA}model.json`) && existsSync(`${DATA}weights.f32.bin`) && existsSync(`${DATA}training/snap_05000.f16.bin`);
const STEPS = [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000];
const readJson = async (f) => JSON.parse(await readFile(f, "utf8"));
const readBuffer = async (f) => {
  const b = await readFile(f);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

// ---------------------------------------------------------------- sorgente dei pesi

test("aggancio allo snapshot più vicino (a metà strada vince il precedente)", () => {
  assert.equal(snapToSnapshot(0, STEPS), 0);
  assert.equal(snapToSnapshot(24, STEPS), 0);
  assert.equal(snapToSnapshot(25, STEPS), 0);
  assert.equal(snapToSnapshot(26, STEPS), 50);
  assert.equal(snapToSnapshot(760, STEPS), 1000);
  assert.equal(snapToSnapshot(740, STEPS), 500);
  assert.equal(snapToSnapshot(4499, STEPS), 4000);
  assert.equal(snapToSnapshot(99999, STEPS), 5000);
  assert.equal(snapToSnapshot(10, []), null);
});

test("sorgente del training: snapshot vero senza interpolazione, stima lineare con", () => {
  assert.deepEqual(trainingSource(760, STEPS), { kind: "snapshot", step: 1000 });
  assert.deepEqual(trainingSource(-5, STEPS), { kind: "snapshot", step: 0 });
  assert.deepEqual(trainingSource(7000, STEPS), { kind: "snapshot", step: 5000 });
  assert.deepEqual(trainingSource(750, STEPS, { interpolate: true }), { kind: "interpolated", step: 750, from: [500, 1000], t: 0.5 });
  assert.deepEqual(trainingSource(1000, STEPS, { interpolate: true }), { kind: "snapshot", step: 1000 }, "uno step di snapshot resta un dato vero");
  const s = trainingSource(2500.4, STEPS, { interpolate: true });
  assert.equal(s.step, 2500);
  assert.deepEqual(s.from, [2000, 3000]);
  assert.ok(Math.abs(s.t - 0.5) < 1e-12);
  assert.throws(() => trainingSource(10, []));
  assert.ok(sameSource(FINAL_SOURCE, { kind: "final" }));
  assert.ok(sameSource({ kind: "snapshot", step: 50 }, { kind: "snapshot", step: 50 }));
  assert.ok(!sameSource({ kind: "snapshot", step: 50 }, { kind: "snapshot", step: 100 }));
  assert.ok(!sameSource({ kind: "snapshot", step: 50 }, FINAL_SOURCE));
  assert.ok(sameSource(trainingSource(750, STEPS, { interpolate: true }), trainingSource(750, STEPS, { interpolate: true })));
  assert.ok(!sameSource(trainingSource(750, STEPS, { interpolate: true }), trainingSource(800, STEPS, { interpolate: true })));
});

test("etichette oneste: pesi finali, step 0, snapshot intermedio, stima lineare dichiarata", () => {
  const final = sourceLabel(FINAL_SOURCE, { steps: STEPS });
  assert.equal(final.title, "Pesi finali (float32)");
  assert.match(final.detail, /weights\.f32\.bin/);
  assert.equal(final.estimate, false);
  const zero = sourceLabel({ kind: "snapshot", step: 0 }, { steps: STEPS });
  assert.equal(zero.title, "Step 0: pesi iniziali (float16)");
  assert.match(zero.detail, /casuali/);
  const mid = sourceLabel({ kind: "snapshot", step: 1000 }, { steps: STEPS });
  assert.equal(mid.title, "Step 1000 (float16)");
  assert.match(mid.detail, /Snapshot vero del training/);
  assert.match(mid.detail, /float16/);
  const est = sourceLabel(trainingSource(750, STEPS, { interpolate: true }), { steps: STEPS });
  assert.equal(est.title, "Step 750: stima lineare tra 500 e 1000");
  assert.equal(est.estimate, true);
  assert.match(est.detail, /Non è un vero step del training/);
  assert.match(est.detail, /500 \(50%\) e 1000 \(50%\)/);
  // ultimo snapshot: prima della verifica non si promette niente, dopo dipende dai dati
  assert.equal(sourceLabel({ kind: "snapshot", step: 5000 }, { steps: STEPS }).title, "Step 5000 (float16)");
  const ok = sourceLabel({ kind: "snapshot", step: 5000 }, { steps: STEPS, lastCheck: { matches: true, maxAbsDiff: 0.00097, count: 209729, beyond: 0 }, format: (v) => v.toFixed(5) });
  assert.equal(ok.title, "Step 5000 (float16: i pesi finali arrotondati)");
  assert.match(ok.detail, /209\.729 valori/);
  assert.match(ok.detail, /0\.00097/);
  const ko = sourceLabel({ kind: "snapshot", step: 5000 }, { steps: STEPS, lastCheck: { matches: false, maxAbsDiff: 0.3, count: 209729, beyond: 1200 }, format: (v) => String(v) });
  assert.equal(ko.title, "Step 5000 (float16)");
  assert.match(ko.detail, /NON coincide/);
  assert.match(ko.detail, /1\.200 valori/);
});

test("cartellino dello step nella barra dello stage: corto, e la stima si dichiara anche lì", () => {
  const check = { matches: true, maxAbsDiff: 0.00097, count: 209729, beyond: 0 };
  const tags = [
    sourceLabel({ kind: "snapshot", step: 0 }, { steps: STEPS }),
    sourceLabel({ kind: "snapshot", step: 1000 }, { steps: STEPS }),
    sourceLabel({ kind: "snapshot", step: 5000 }, { steps: STEPS, lastCheck: check }),
    sourceLabel({ kind: "snapshot", step: 5000 }, { steps: STEPS, lastCheck: { ...check, matches: false, beyond: 3 } }),
  ];
  assert.deepEqual(tags.map((l) => l.tag), ["Step 0", "Step 1000", "Step 5000", "Step 5000"]);
  assert.ok(tags.every((l) => !l.estimate));
  const est = sourceLabel(trainingSource(2600, STEPS, { interpolate: true }), { steps: STEPS });
  assert.equal(est.tag, "≈ 2600 stima");
  assert.equal(est.estimate, true);
  // sta nella barra accanto al cursore (gruppo largo 330 px): al più 13 caratteri
  for (const l of [...tags, est, sourceLabel(FINAL_SOURCE, { steps: STEPS })]) assert.ok(l.tag.length <= 13, l.tag);
});

test("verifica float16: un valore entro mezzo passo è arrotondamento, oltre no", () => {
  // valori esatti in float16: differenza zero
  const exact = Float32Array.from([0, 0.5, -0.25, 1.5, 2 ** -20]);
  assert.deepEqual(float16RoundingCheck(exact, exact.slice()), { matches: true, maxAbsDiff: 0, count: 5, worstIndex: -1, beyond: 0 });
  // 1 + 2^-12 sta a mezzo passo (il passo del float16 tra 1 e 2 è 2^-10): ancora arrotondamento
  const w = Float32Array.from([1 + 2 ** -12, 0.3]);
  const snap = Float32Array.from([1, 0.2999267578125]); // 0,3 in float16
  assert.equal(float16RoundingCheck(w, snap).matches, true);
  // un millesimo di troppo su 0,3: non è più arrotondamento
  const off = Float32Array.from([1, 0.3 + 0.001]);
  const r = float16RoundingCheck(w, off);
  assert.equal(r.matches, false);
  assert.equal(r.beyond, 1);
  assert.equal(r.worstIndex, 1);
  assert.equal(float16RoundingCheck(w, new Float32Array(3)).matches, false, "lunghezze diverse");
});

test("dati veri: lo snapshot dello step 5000 è il checkpoint della web app arrotondato a float16", { skip: !HAS_DATA && "dati di webapp/data assenti" }, async () => {
  const manifest = await readJson(`${DATA}weights.json`);
  const final = decodeFloat32(await readBuffer(`${DATA}weights.f32.bin`), manifest.total_params);
  const trace = await readJson(`${DATA}training/trace.json`);
  assert.deepEqual(trace.snapshots.map((s) => s.step), STEPS);
  const last = decodeFloat16(await readBuffer(`${DATA}training/snap_05000.f16.bin`), manifest.total_params);
  const check = float16RoundingCheck(final, last);
  assert.equal(check.matches, true);
  assert.equal(check.beyond, 0);
  assert.ok(check.maxAbsDiff > 0 && check.maxAbsDiff < 1e-3, `scarto massimo ${check.maxAbsDiff}`);
  // lo step 4000 è un altro punto del training: la verifica lo distingue
  const before = decodeFloat16(await readBuffer(`${DATA}training/snap_04000.f16.bin`), manifest.total_params);
  assert.equal(float16RoundingCheck(final, before).matches, false);
});

// ---------------------------------------------------------------- righe e colonne

const CONFIG = { vocab_size: 65, block_size: 32, n_embd: 64, n_head: 4, n_layer: 4 };
const VOCAB = [..."\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"];

test("righe e colonne con un significato, per ogni tipo di tensore", () => {
  const ax = (name, shape) => tensorAxes(name, shape, { vocab: VOCAB, config: CONFIG });
  const emb = ax("token_embedding_table.weight", [65, 64]);
  assert.equal(emb.kind, "token_embedding");
  assert.equal(emb.rowTitle, "carattere");
  assert.equal(emb.rowLabel(39), "'a'");
  assert.equal(emb.rowLabel(0), "'↵'");
  assert.equal(emb.rowLabel(1), "'␣'");
  assert.equal(emb.colTitle, "dimensione dell'embedding");
  assert.equal(emb.colLabel(63), "63");
  assert.match(emb.meaning, /65 caratteri/);
  const pos = ax("position_embedding_table.weight", [32, 64]);
  assert.equal(pos.rowTitle, "posizione nel contesto");
  assert.equal(pos.rowLabel(31), "31");
  const key = ax("blocks.2.sa.heads.3.key.weight", [16, 64]);
  assert.equal(key.kind, "head");
  assert.equal(key.rowLabel(5), "k5");
  assert.match(key.rowTitle, /key/);
  assert.match(key.colTitle, /dopo ln1/);
  assert.match(key.meaning, /Head 3 del blocco 2/);
  assert.equal(ax("blocks.0.sa.heads.0.query.weight", [16, 64]).rowLabel(0), "q0");
  assert.equal(ax("blocks.0.sa.heads.0.value.weight", [16, 64]).rowLabel(15), "v15");
  const proj = ax("blocks.1.sa.proj.weight", [64, 64]);
  assert.equal(proj.colLabel(21), "h1·5");
  assert.equal(proj.colLabel(63), "h3·15");
  const ffIn = ax("blocks.0.ffwd.net.0.weight", [256, 64]);
  assert.equal(ffIn.rowTitle, "neurone nascosto");
  assert.match(ffIn.colTitle, /dopo ln2/);
  const ffOut = ax("blocks.0.ffwd.net.2.weight", [64, 256]);
  assert.equal(ffOut.colTitle, "neurone nascosto");
  const head = ax("lm_head.weight", [65, 64]);
  assert.equal(head.rowLabel(39), "'a'");
  assert.match(head.rowTitle, /carattere in uscita/);
  // vettori: una colonna, nessun titolo di colonna
  for (const [name, title] of [["blocks.0.ln1.weight", /γ/], ["blocks.3.ln2.bias", /β/], ["ln_f.weight", /γ/], ["ln_f.bias", /β/], ["blocks.0.sa.proj.bias", /uscita/], ["blocks.0.ffwd.net.0.bias", /neurone nascosto/], ["blocks.0.ffwd.net.2.bias", /uscita/], ["lm_head.bias", /carattere/]]) {
    const v = ax(name, [name.startsWith("lm_head") ? 65 : name.includes("net.0") ? 256 : 64]);
    assert.equal(v.vector, true, name);
    assert.equal(v.colTitle, "", name);
    assert.match(v.rowTitle, title, name);
  }
  assert.match(ax("ln_f.weight", [64]).meaning, /ln_f/);
  assert.match(ax("blocks.2.ln1.weight", [64]).meaning, /ln1 del blocco 2/);
  assert.equal(ax("lm_head.bias", [65]).rowLabel(39), "'a'");
  assert.equal(tensorKind("transformer.h.0.attn.c_attn.weight"), null);
  assert.equal(ax("strano.weight", [3, 4]).kind, null);
});

test("significato di una cella (suggerimento nella scena)", () => {
  const emb = tensorAxes("token_embedding_table.weight", [65, 64], { vocab: VOCAB, config: CONFIG });
  assert.equal(cellMeaning(emb, 39, 3), "riga 39 = carattere 'a' · colonna 3 = dimensione dell'embedding 3");
  const gamma = tensorAxes("blocks.0.ln1.weight", [64], { vocab: VOCAB, config: CONFIG });
  assert.equal(cellMeaning(gamma, 12, 0), "riga 12 = dimensione (γ: quanto la scala) 12");
  const key = tensorAxes("blocks.0.sa.heads.1.key.weight", [16, 64], { vocab: VOCAB, config: CONFIG });
  assert.equal(cellMeaning(key, 2, 40), "riga 2 = dimensione della key (uscita della head) k2 · colonna 40 = dimensione d'ingresso (x dopo ln1) 40");
  assert.equal(shownChar("\n"), "↵");
  assert.equal(shownChar("a"), "a");
});

// ---------------------------------------------------------------- elenco come il codice

test("tensori raggruppati come il codice Python, nell'ordine del forward", { skip: !HAS_DATA && "dati assenti" }, async () => {
  const spec = await readJson(`${DATA}model.json`);
  const groups = tensorGroups(spec.params);
  assert.deepEqual(groups.map((g) => g.id), ["embedding", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "output"]);
  assert.equal(groups.reduce((a, g) => a + g.count, 0), spec.total_params);
  assert.equal(groups.reduce((a, g) => a + g.items.length, 0), spec.params.length);
  assert.equal(groups[1].title, "blocks[0] · Block");
  assert.equal(groups[1].items.length, 22);
  const key = groups[2].items.find((i) => i.name === "blocks.1.sa.heads.2.key.weight");
  assert.equal(key.path, "blocks[1].sa.heads[2].key.weight");
  assert.equal(key.local, "sa.heads[2].key.weight");
  assert.deepEqual(key.shape, [16, 64]);
  assert.equal(key.numel, 1024);
  assert.equal(groups[1].items.find((i) => i.name.endsWith("ffwd.net.0.weight")).local, "ffwd.net[0].weight");
  assert.deepEqual(groups[5].items.map((i) => i.name), ["ln_f.weight", "ln_f.bias", "lm_head.weight", "lm_head.bias"]);
  // ogni tensore del modello ha un tipo noto (righe e colonne con un significato)
  for (const p of spec.params) assert.ok(tensorKind(p.name), p.name);
});

// ---------------------------------------------------------------- link diretti

test("link tensor:<nome>[@<step>]: pesi finali, snapshot vero, stima, errori", () => {
  const names = new Set(["lm_head.weight", "blocks.0.ln1.weight"]);
  assert.deepEqual(parseTensorTarget("lm_head.weight", { names, steps: STEPS }), { name: "lm_head.weight", step: null, interpolate: false });
  assert.deepEqual(parseTensorTarget("lm_head.weight@1000", { names, steps: STEPS }), { name: "lm_head.weight", step: 1000, interpolate: false });
  assert.deepEqual(parseTensorTarget("blocks.0.ln1.weight@750", { names, steps: STEPS }), { name: "blocks.0.ln1.weight", step: 750, interpolate: true });
  assert.deepEqual(parseTensorTarget("lm_head.weight@0", { names, steps: STEPS }), { name: "lm_head.weight", step: 0, interpolate: false });
  for (const bad of ["", "nessuno.weight", "lm_head.weight@", "lm_head.weight@abc", "lm_head.weight@-5", "lm_head.weight@5001", "lm_head.weight@1e3", "lm_head.weight@1000.5"]) {
    assert.equal(parseTensorTarget(bad, { names, steps: STEPS }), null, bad);
  }
  assert.equal(parseTensorTarget("lm_head.weight@100", { names, steps: [] }), null, "senza snapshot niente step");
  assert.equal(parseTensorTarget(42, { names, steps: STEPS }), null);
  assert.equal(formatTensorTarget("lm_head.weight"), "lm_head.weight");
  assert.equal(formatTensorTarget("lm_head.weight", 1000), "lm_head.weight@1000");
  assert.deepEqual(labTarget("lm_head.weight", FINAL_SOURCE), { kind: "tensor", id: "lm_head.weight" });
  assert.deepEqual(labTarget("lm_head.weight", { kind: "snapshot", step: 250 }), { kind: "tensor", id: "lm_head.weight@250" });
  assert.deepEqual(labTarget("lm_head.weight", trainingSource(750, STEPS, { interpolate: true })), { kind: "tensor", id: "lm_head.weight@750" });
  // andata e ritorno
  const t = parseTensorTarget(labTarget("blocks.0.ln1.weight", { kind: "snapshot", step: 4000 }).id, { names, steps: STEPS });
  assert.deepEqual(t, { name: "blocks.0.ln1.weight", step: 4000, interpolate: false });
  assert.equal(NUMBERS_VIEW_ID, "numeri");
  assert.equal(BIRTH_VIEW_ID, "numeri.training");
});

// ---------------------------------------------------------------- "Guarda i pesi nascere"

test("calendario dei pesi che nascono: un tratto per coppia di snapshot, poi la pausa", () => {
  const s = birthSchedule(STEPS, { segment: 1, hold: 2 });
  assert.equal(s.duration, 10 + 2);
  assert.equal(s.stepAt(0), 0);
  assert.equal(s.stepAt(0.99), 0, "senza interpolazione si resta sullo snapshot");
  assert.equal(s.stepAt(1), 50);
  assert.equal(s.stepAt(4.5), 500);
  assert.equal(s.stepAt(9.99), 4000);
  assert.equal(s.stepAt(10), 5000);
  assert.equal(s.stepAt(100), 5000);
  assert.equal(s.stepAt(-1), 0);
  // solo step di snapshot veri
  for (let t = 0; t < s.duration; t += 0.05) assert.ok(STEPS.includes(s.stepAt(t)), `t = ${t}`);
  const lin = birthSchedule(STEPS, { interpolate: true, segment: 1 });
  assert.equal(lin.stepAt(5.5), 1250);
  assert.equal(lin.stepAt(8.25), 3250);
  // il momento in cui si arriva a uno step (per riprendere l'animazione)
  assert.equal(s.timeOf(0), 0);
  assert.equal(s.timeOf(1000), 5);
  assert.equal(s.timeOf(1200), 5, "senza interpolazione: l'inizio del tratto");
  assert.equal(lin.timeOf(1250), 5.5);
  assert.equal(s.timeOf(5000), 10);
  for (const step of STEPS) assert.equal(s.stepAt(s.timeOf(step)), step);
});

// ---------------------------------------------------------------- carte delle head

test("carte di una head: quella scelta passa davanti, le altre restano in ordine", () => {
  const original = { query: 0.25, key: 0, value: -0.25 };
  assert.deepEqual(cardDepths(original, "query"), original);
  assert.deepEqual(cardDepths(original, "key"), { key: 0.25, query: 0, value: -0.25 });
  assert.deepEqual(cardDepths(original, "value"), { value: 0.25, query: 0, key: -0.25 });
  assert.deepEqual(headCard("blocks.3.sa.heads.1.value.weight"), { layer: 3, head: 1, kind: "value" });
  assert.equal(headCard("blocks.3.sa.proj.weight"), null);
});

// ---------------------------------------------------------------- nome del tensore ↔ heatmap della scena (three.js)

// three.js sta in webapp/vendor (nel browser lo trova l'importmap): qui lo trova un hook di risoluzione.
// Le etichette della scena disegnano su un canvas 2D: un canvas finto basta (si contano solo i mesh).
const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      if (specifier.startsWith("three/addons/")) return { url: ${JSON.stringify(`${VENDOR}addons/`)} + specifier.slice(13), shortCircuit: true };
      return next(specifier, context);
    }`),
);

test("ogni tensore del modello ha la sua heatmap nella scena, con righe e colonne della tabella", { skip: !HAS_DATA && "dati assenti" }, async () => {
  const fakeContext = { font: "", fillStyle: "", textBaseline: "", measureText: (t) => ({ width: String(t).length * 30 }), fillText() {} };
  globalThis.document ??= { createElement: () => ({ width: 0, height: 0, getContext: () => fakeContext }) };
  const { buildArchitecture } = await import("../js/tabs/transformer/architecture.js");
  const { tableShape } = await import("../js/core/ui/tensorTableMath.js");
  const spec = await readJson(`${DATA}model.json`);
  const arch = buildArchitecture(spec);
  assert.deepEqual([...arch.tensorMeshes.keys()].sort(), spec.params.map((p) => p.name).sort(), "un mesh per ogni parametro, nessuno in più");
  for (const p of spec.params) {
    const meshes = arch.tensorMeshes.get(p.name);
    assert.equal(meshes.length, 1, p.name);
    const [mesh] = meshes;
    assert.equal(mesh.userData.tensorName, p.name);
    const { width: cols, height: rows } = mesh.material.map.image;
    // la heatmap (e quindi attachNumbers) e la tabella leggono la stessa forma: un vettore è una colonna
    assert.deepEqual([rows, cols], tableShape(p.shape), p.name);
    assert.equal(rows * cols, p.numel, p.name);
    const axes = tensorAxes(p.name, p.shape, { vocab: spec.vocab, config: spec.config });
    assert.equal(axes.vector, cols === 1 && p.shape.length === 1, p.name);
  }
  // le carte di ogni head sono davvero sovrapposte (per questo quella scelta passa davanti)
  const z = (k) => arch.tensorMeshes.get(`blocks.0.sa.heads.0.${k}.weight`)[0].position.z;
  assert.deepEqual({ query: z("query"), key: z("key"), value: z("value") }, { query: 0.25, key: 0, value: -0.25 });
});
