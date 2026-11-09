// Test della logica pura del Laboratorio pesi (js/tabs/lab/): node --test webapp/tests/*.test.mjs
// Albero e disposizione del GPT del corso (94 tensori, 209.729 parametri, righe Python vere lette da
// gpt/*.py), architetture modificate (run esportati dal Python, se ci sono, e run costruiti qui),
// stato (transizioni ed eventi, step agganciati, cache, motore), link diretti, caricamenti rotti,
// specifiche della rete neurale e del cervello, indice della ricerca.
import assert from "node:assert/strict";
import { readdirSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { missingRuns, runDir } from "./runFixtures.mjs";

import { normalizeBrainSpec } from "../js/core/brain/spec.js";
import { buildNetworkModel } from "../js/core/neuralNet/model.js";
import { expectedParameters } from "../js/engine/gpt.js";
import { buildModuleTree, findNode, walkTree } from "../js/shared/weights/moduleTree.js";
import { loadBuiltinRun, runsFromFiles } from "../js/shared/weights/run.js";
import { brainSpec } from "../js/tabs/lab/brain.js";
import { tensorMeaning } from "../js/tabs/lab/inspect/meaning.js";
import { moduleArgs, outlineRows } from "../js/tabs/lab/inspect/outline.js";
import { RUN_ALIASES, VIEW_IDS, formatTensorTarget, parseRunTarget, parseTensorTarget, resolveTarget } from "../js/tabs/lab/links.js";
import { NETWORK_PROMPT, courseDims, encodeWith, structureSpec, trueSpec } from "../js/tabs/lab/network.js";
import { architectureDiff, compatibilityText, configDiff, formatBytes, formatDate, trainingText } from "../js/tabs/lab/runs/describe.js";
import { CELL, VECTOR_STRETCH, classesFromRunInfo, codeLine, coveredBy, hitTest, layoutModel, macroItems, moduleItems, subtractRects } from "../js/tabs/lab/scene/layout.js";
import { packShelves } from "../js/tabs/lab/scene/pack.js";
import { roleOf } from "../js/tabs/lab/scene/roles.js";
import { searchIndex } from "../js/tabs/lab/search.js";
import { COURSE_RUN_ID, LabState, SNAPSHOT_BUDGET } from "../js/tabs/lab/state.js";
import { snapStep, sliderModel, stepLabel } from "../js/tabs/lab/steps.js";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const bytes = await readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const fsLoaders = { baseUrl: DATA_DIR, loadJson, loadBuffer };
const spec = await loadJson(`${DATA_DIR}model.json`);
const builtin = await loadBuiltinRun(fsLoaders);

/** Run esportati dal Python (facoltativi, fuori dal repository: vedi runFixtures.mjs e LAB_RUNS_DIR). */
const RUN_DIRS = { gpt200: runDir("gpt200"), ffn3: runDir("ffn3"), gelu: runDir("gelu") };

function fileLike(path, content) {
  const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content instanceof Uint8Array ? content : new Uint8Array(content);
  return { name: path.split("/").pop(), size: bytes.byteLength, webkitRelativePath: path, text: async () => new TextDecoder().decode(bytes), arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
}

/** I file di una cartella del disco come li darebbe un input con webkitdirectory. */
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

/** Un run con l'architettura del corso a una config qualsiasi (pesi con seme), più tensori in più o in meno. */
function syntheticRun(config, { extra = [], drop = [], name = "sintetico" } = {}) {
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
  const params = [...expectedParameters(config).filter((p) => !drop.includes(p.name)), ...extra];
  let offset = 0;
  const layout = params.map(({ name: n, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const p = { name: n, shape, offset, numel };
    offset += numel;
    return p;
  });
  const flat = Float32Array.from({ length: offset }, () => rand() * 0.2);
  const vocab = Array.from({ length: config.vocab_size }, (_, i) => (i === 0 ? "\n" : i === 1 ? " " : String.fromCharCode(64 + i)));
  const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", source: "test", trained: true, config, vocab, total_params: offset, params: layout };
  return [fileLike(`${name}/weights.json`, JSON.stringify(manifest)), fileLike(`${name}/weights.f32.bin`, new Uint8Array(flat.buffer))];
}

/** Un RunStore finto in memoria (stessa interfaccia). */
function memoryStore() {
  const runs = new Map();
  return {
    runs,
    async list() {
      return [...runs.values()].map((r) => ({ id: r.id, name: r.name, description: r.description, savedAt: 1, bytes: r.totalParams * 4 }));
    },
    async load(id) {
      return runs.get(id) ?? null;
    },
    async save(run) {
      runs.set(run.id, run);
      return { ok: true, id: run.id, bytes: run.totalParams * 4 };
    },
    async remove(id) {
      return runs.delete(id) || true;
    },
    async estimate() {
      return { usage: 1000, quota: 1e9 };
    },
  };
}

/** Le righe vere dei file gpt/*.py (per controllare le righe Python della scena). */
const sourceLines = new Map();
async function pyLine(file, line) {
  if (!sourceLines.has(file)) sourceLines.set(file, (await readFile(`${ROOT}${file}`, "utf8")).split("\n"));
  return sourceLines.get(file)[line - 1];
}

const courseTree = () => buildModuleTree(builtin.params, { snippets: spec.snippets, config: builtin.config });
const courseLayout = () => layoutModel(courseTree(), { config: builtin.config, snippets: spec.snippets });

// ---------------------------------------------------------------- albero e disposizione del GPT del corso

test("GPT del corso: 94 tensori, 209.729 parametri, albero come il codice, righe Python vere", async () => {
  const tree = courseTree();
  assert.equal(tree.label, "GPTLanguageModel");
  assert.equal(tree.architecture, "corso");
  assert.equal(tree.params, 209729);
  assert.equal(tree.tensors, 94);
  assert.deepEqual(tree.children.map((c) => c.key), ["token_embedding_table", "position_embedding_table", "blocks", "ln_f", "lm_head"]);
  const L = courseLayout();
  assert.equal(L.params.length, 94);
  assert.equal(L.params.reduce((s, b) => s + b.params, 0), 209729);
  // ogni contenitore e ogni peso annotato porta la riga Python che lo crea: è davvero quella riga del file
  let checked = 0;
  for (const b of L.boxes) {
    if (!b.code?.file || !Number.isInteger(b.code.line)) continue;
    const line = await pyLine(b.code.file, b.code.line);
    const first = b.code.first ?? codeLine(b.code);
    assert.ok(line?.includes(first.split("\n")[0].trim()), `${b.path}: «${first}» non è alla riga ${b.code.file}:${b.code.line} («${line}»)`);
    if (b.header?.tag) {
      const shown = b.code.createdBy && /^class\s/.test(b.code.text) && b.kind === "module" ? b.code.createdBy : b.code;
      assert.equal(b.header.tag, `${shown.file.split("/").pop()}:${shown.line}`, `${b.path}: file:riga dell'intestazione`);
      assert.ok((await pyLine(shown.file, shown.line)).includes(b.header.code), `${b.path}: la riga mostrata è quella del file:riga`);
    }
    checked++;
  }
  assert.ok(checked >= 200, `righe controllate: ${checked}`);
  const key = L.nodes.get("blocks.0.sa.heads.0.key");
  assert.equal(key.header.code, "self.key = nn.Linear(config.n_embd, self.head_size, bias=False)");
  assert.equal(key.header.tag, "attention.py:29");
  assert.equal(L.nodes.get("blocks.2").header.code, "self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])");
});

test("disposizione: annidamento, ordine da sinistra a destra, matrici come in PyTorch, nodi senza pesi", () => {
  const L = courseLayout();
  const n = (p) => L.nodes.get(p);
  // il flusso: embedding → blocks[0..3] → ln_f → lm_head, da sinistra a destra
  const xs = ["token_embedding_table", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "ln_f", "lm_head"].map((p) => n(p).x0);
  assert.deepEqual([...xs].sort((a, b) => a - b), xs);
  assert.ok(n("blocks.0.ln1").x1 < n("blocks.0.sa").x0 && n("blocks.0.sa").x1 < n("blocks.0.ln2").x0 && n("blocks.0.ln2").x1 < n("blocks.0.ffwd").x0);
  // le tabelle di embedding in colonna (si sommano); le head in colonna; key, query, value in colonna
  assert.ok(Math.abs(n("token_embedding_table").x0 - n("position_embedding_table").x0) < 1e-9);
  assert.ok(n("token_embedding_table").y0 > n("position_embedding_table").y1);
  for (let h = 1; h < 4; h++) assert.ok(n(`blocks.0.sa.heads.${h}`).y1 < n(`blocks.0.sa.heads.${h - 1}`).y0);
  assert.ok(n("blocks.0.sa.heads.0.query").y1 < n("blocks.0.sa.heads.0.key").y0);
  // annidamento: ogni figlio dentro il genitore, un poco più avanti in z
  walkTree(courseTree(), (t) => {
    const b = n(t.path);
    const parent = t.path ? n(t.path.split(".").slice(0, -1).join(".")) ?? L.root : null;
    if (!b || !parent) return;
    assert.ok(b.x0 >= parent.x0 - 1e-9 && b.x1 <= parent.x1 + 1e-9 && b.y0 >= parent.y0 - 1e-9 && b.y1 <= parent.y1 + 1e-9, `${t.path} dentro il genitore`);
    assert.ok(b.z > parent.z, `${t.path} davanti al genitore`);
  });
  // matrici: 1 cella = 1 peso, nn.Linear (out, in) alta out; bias in colonna accanto alle righe di W
  const w = n("blocks.0.ffwd.net.0.weight").matrix;
  assert.deepEqual([w.rows, w.cols, w.w, w.h], [256, 64, 64 * CELL, 256 * CELL]);
  const bias = n("blocks.0.ffwd.net.0.bias").matrix;
  assert.deepEqual([bias.rows, bias.cols, bias.vector], [256, 1, "column"]);
  assert.ok(Math.abs(bias.y1 - w.y1) < 1e-9 && bias.x0 > w.x1, "il bias parte dalla stessa riga di W");
  assert.equal(bias.cellW, CELL * VECTOR_STRETCH);
  const gamma = n("blocks.0.ln1.weight");
  assert.equal(gamma.label, "weight = γ (64)");
  assert.deepEqual([gamma.matrix.rows, gamma.matrix.cols, gamma.matrix.vector], [1, 64, "row"]);
  // il buffer tril (non è un peso) e i moduli senza pesi del codice
  const tril = n("blocks.3.sa.heads.2.tril");
  assert.equal(tril.kind, "buffer");
  assert.deepEqual(tril.shape, [32, 32]);
  assert.match(tril.label, /buffer, non è un peso/);
  assert.equal(tril.code.file, "gpt/attention.py");
  assert.equal(n("blocks.0.ffwd.net.1").type, "nn.ReLU");
  assert.equal(n("blocks.0.ffwd.net.3").type, "nn.Dropout");
  assert.equal(n("blocks.0.sa.dropout").header.tag, "attention.py:75", "il Dropout di MultiHeadAttention sta in attention.py, non in block.py");
  assert.equal(L.boxes.filter((b) => b.kind === "buffer").length, 16);
});

test("disposizione: isole della Panoramica, moduli del livello Moduli, scelta con il puntatore", () => {
  const L = courseLayout();
  const macro = macroItems(L);
  assert.deepEqual(macro.map((m) => m.title), ["Embedding", "blocks[0] · Block", "blocks[1] · Block", "blocks[2] · Block", "blocks[3] · Block", "ln_f · nn.LayerNorm", "lm_head · nn.Linear"]);
  assert.equal(macro.reduce((s, m) => s + m.params, 0), 209729);
  assert.ok(Math.abs(macro.reduce((s, m) => s + m.share, 0) - 1) < 1e-12);
  const mods = moduleItems(L).map((m) => m.path);
  assert.equal(mods.length, 2 + 4 * 4 + 2);
  assert.ok(mods.includes("blocks.3.ffwd") && mods.includes("lm_head"));
  const key = L.nodes.get("blocks.1.sa.heads.3.value.weight");
  assert.equal(hitTest(L, key.matrix.cx, key.matrix.cy).path, "blocks.1.sa.heads.3.value.weight");
  const sa = L.nodes.get("blocks.1.sa");
  assert.equal(hitTest(L, sa.x0 + 0.01, sa.y1 - 0.01).path, "blocks.1.sa");
  assert.equal(hitTest(L, L.bounds.x1 + 5, 0), null);
  const rows = outlineRows(L);
  assert.equal(rows[0].label, "GPTLanguageModel");
  assert.ok(rows.some((r) => r.label === "key: nn.Linear(in=64, out=16, bias=False)"));
  assert.ok(rows.some((r) => r.label === "token_embedding_table: nn.Embedding(65, 64)"));
  assert.equal(moduleArgs(L.nodes.get("lm_head")), "(in=64, out=65)");
});

// ---------------------------------------------------------------- architetture modificate

test("architetture modificate costruite qui: più blocchi, FFN più larga, modulo nuovo, nomi con HTML", async () => {
  const cfg = { vocab_size: 65, block_size: 16, n_embd: 32, n_head: 4, n_layer: 6, dropout: 0 };
  // 6 blocchi e un modulo nuovo (es. una "porta" appresa) che il codice del corso non crea
  const extra = [{ name: "blocks.0.sa.gate.weight", shape: [32] }, { name: "<img src=x onerror=alert(1)>.weight", shape: [4, 4] }];
  const { runs, errors } = await runsFromFiles(syntheticRun(cfg, { extra, name: "sei-blocchi" }), { defaultVocab: builtin.vocab });
  assert.deepEqual(errors, []);
  const run = runs[0];
  const tree = buildModuleTree(run.params, { snippets: spec.snippets, config: run.config });
  assert.equal(tree.architecture, "modificata");
  const L = layoutModel(tree, { config: run.config, snippets: spec.snippets });
  assert.equal(L.params.length, run.tensors.size);
  assert.deepEqual(macroItems(L).filter((m) => /^blocks\[/.test(m.title)).length, 6);
  const gate = L.nodes.get("blocks.0.sa.gate");
  assert.equal(gate.code, null);
  assert.match(gate.header.sub, /non nel codice del corso/);
  const html = L.nodes.get("<img src=x onerror=alert(1)>");
  assert.equal(html.header.title, "<img src=x onerror=alert(1)>", "il nome resta testo: chi lo mostra usa textContent o fillText");
  const diff = architectureDiff(run.tensors, builtin.tensors);
  assert.ok(diff.notes.some((n) => /6 blocchi transformer invece di 4/.test(n)), diff.notes.join(" | "));
  assert.ok(diff.notes.some((n) => /tensori che il codice del corso non crea/.test(n)));
  assert.equal(courseDims(run.tensors).L, 6);
  // FFN più larga (6 × n_embd): stessa struttura, forme diverse, il motore JS non la esegue
  const wide = expectedParameters(cfg).map((p) => ({ ...p, shape: /ffwd\.net\.0\.weight$/.test(p.name) ? [6 * 32, 32] : /ffwd\.net\.0\.bias$/.test(p.name) ? [6 * 32] : /ffwd\.net\.2\.weight$/.test(p.name) ? [32, 6 * 32] : p.shape }));
  const { runs: wr } = await runsFromFiles(syntheticRun(cfg, { drop: wide.map((p) => p.name), extra: wide, name: "ffn-larga" }), { defaultVocab: builtin.vocab });
  assert.equal(wr[0].compatibleWithEngine, false);
  assert.equal(courseDims(wr[0].tensors).F, 192);
  assert.ok(architectureDiff(wr[0].tensors, builtin.tensors).notes.some((n) => /192 neuroni nascosti invece di 256/.test(n)));
  assert.equal(compatibilityText(wr[0]).ok, false);
});

test("run esportati dal Python (gpt-200-step, ffn3, gpt-con-gelu): albero, note, compatibilità", { skip: missingRuns("gpt200", "ffn3", "gelu") }, async () => {
  const load = async (dir) => {
    const { runs, errors } = await runsFromFiles(await folderFiles(dir), { defaultVocab: builtin.vocab });
    assert.deepEqual(errors, [], dir);
    return runs[0];
  };
  const g200 = await load(RUN_DIRS.gpt200);
  assert.equal(g200.compatibleWithEngine, true);
  assert.deepEqual(g200.steps, [0, 50, 100, 200]);
  assert.equal(buildModuleTree(g200.params, { snippets: spec.snippets, config: g200.config }).architecture, "corso");
  assert.ok(architectureDiff(g200.tensors, builtin.tensors).same);
  assert.match(trainingText(g200), /4 snapshot/);

  const ffn3 = await load(RUN_DIRS.ffn3);
  assert.equal(ffn3.totalParams, 176705);
  assert.equal(ffn3.compatibleWithEngine, false);
  const tree = buildModuleTree(ffn3.params, { snippets: spec.snippets, config: ffn3.config });
  assert.equal(tree.architecture, "modificata");
  assert.ok(tree.mismatched.includes("blocks.0.ffwd.net.0.weight"));
  const L = layoutModel(tree, { config: ffn3.config, snippets: spec.snippets, classes: classesFromRunInfo(ffn3.info?.modules) });
  assert.equal(L.nodes.get("blocks.0.ffwd.net.0.weight").matrix.rows, 192);
  assert.ok(architectureDiff(ffn3.tensors, builtin.tensors).notes.some((n) => /192 neuroni nascosti invece di 256 \(3 × n_embd invece di 4 ×\)/.test(n)));

  const gelu = await load(RUN_DIRS.gelu);
  assert.equal(gelu.compatibleWithEngine, false, "run.json dichiara engine_compatible: false");
  assert.ok(gelu.benchmark?.metrics?.loss > 0);
  const notes = architectureDiff(gelu.tensors, builtin.tensors, { engineDeclared: gelu.info.engineCompatible }).notes;
  assert.ok(notes.some((n) => /engine_compatible: false/.test(n)));
  assert.ok(compatibilityText(gelu).reasons.some((r) => /engine_compatible: false/.test(r)));
  assert.ok(classesFromRunInfo(gelu.info.modules).get("blocks.0.sa") === "MultiHeadAttention");
});

// ---------------------------------------------------------------- stato

test("stato: init, step agganciati agli snapshot, interpolazione dichiarata, eventi in ordine", async () => {
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store: memoryStore() });
  const events = [];
  for (const type of ["runs", "runA", "step", "weights", "select", "mode", "interpolate"]) state.on(type, (d) => events.push([type, d.step ?? d.id ?? d.path ?? d.mode ?? d.interpolate ?? d.reason]));
  await state.init();
  assert.equal(state.runA, COURSE_RUN_ID);
  assert.deepEqual(events.map((e) => e[0]), ["runs", "runA", "weights"]);
  assert.equal(state.current.view.source, "final");
  assert.equal(state.current.view.label, "pesi finali (float32)");
  assert.deepEqual(state.stepsOf(COURSE_RUN_ID), [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  events.length = 0;
  await state.setStep(700); // agganciato allo snapshot vero più vicino
  assert.equal(state.step, 500);
  assert.deepEqual(events, [["step", 500], ["weights", 500]]);
  assert.equal(state.current.view.source, "snapshot");
  assert.equal(state.current.view.exact, true);
  assert.notEqual(state.current.view.tensors.get("lm_head.weight").data[0], builtin.tensor("lm_head.weight").data[0]);
  await state.setInterpolate(true);
  await state.setStep(750);
  assert.equal(state.current.view.source, "interpolated");
  assert.deepEqual(state.current.view.from, [500, 1000]);
  assert.match(state.current.view.label, /approssimato/);
  const a = (await state.weightsAt(COURSE_RUN_ID, 500)).tensors.get("lm_head.weight").data;
  const z = (await state.weightsAt(COURSE_RUN_ID, 1000)).tensors.get("lm_head.weight").data;
  const mid = state.current.view.tensors.get("lm_head.weight").data;
  assert.ok(Math.abs(mid[7] - (a[7] + (z[7] - a[7]) * 0.5)) < 1e-6);
  assert.ok(state.cacheInfo.bytes > 0, "le interpolazioni contano nel budget");
  events.length = 0;
  await state.setInterpolate(false); // torna su uno snapshot vero
  assert.equal(state.step, 500);
  assert.deepEqual(events.map((e) => e[0]), ["interpolate", "step", "weights"]);
  await state.setStep(null);
  assert.equal(state.current.view.source, "final");
  state.select("blocks.0.sa");
  assert.equal(state.kindOf("blocks.0.sa"), "module");
  assert.equal(state.tensor, null);
  state.select("lm_head.weight");
  assert.equal(state.tensor, "lm_head.weight");
  state.setMode("differenza", { label: "A − B" });
  assert.throws(() => state.setMode("boh"), /modalità sconosciuta/);
  assert.equal(stepLabel(500, { source: "snapshot", step: 500 }), "snapshot dello step 500");
});

test("stato: le richieste superate non annunciano pesi vecchi; registro, salvataggio, Dimentica, motore", async () => {
  const store = memoryStore();
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store });
  await state.init();
  const seen = [];
  state.on("weights", ({ step }) => seen.push(step));
  await Promise.all([state.setStep(0), state.setStep(50), state.setStep(1000)]);
  assert.deepEqual(seen, [1000], "solo l'ultima richiesta arriva alla scena");
  // un run caricato: aggiunto una volta sola, diventa A, si salva, si dimentica
  const cfg = { vocab_size: 65, block_size: 8, n_embd: 16, n_head: 2, n_layer: 1, dropout: 0 };
  const { runs } = await runsFromFiles(syntheticRun(cfg, { name: "piccolo" }));
  assert.deepEqual(state.addRuns(runs).added, [runs[0].id]);
  assert.deepEqual(state.addRuns(runs).duplicates, [runs[0].id]);
  assert.equal(state.entries[0].origin, "builtin");
  const order = [];
  state.on("runA", ({ id }) => order.push(["runA", id]));
  state.on("step", ({ step }) => order.push(["step", step]));
  state.select("blocks.0.sa.heads.3.key.weight"); // il run piccolo ha 2 head: la scelta non vale più
  assert.equal(await state.setRunA(runs[0].id, { step: 500 }), true);
  assert.equal(state.step, null, "un run senza storia ha solo i pesi finali");
  assert.equal(state.selected, null);
  assert.deepEqual(order[0], ["runA", runs[0].id]);
  const engine = await state.engineFor(runs[0].id);
  assert.ok(engine.engine, "compatibile: il motore JS lo esegue");
  assert.equal(engine.engine.forward([1, 2, 3]).logits.length, 3 * 65);
  const course = await state.engineFor(COURSE_RUN_ID, 250);
  assert.equal(course.step, 250);
  assert.equal(course.engine.paramViews.get("lm_head.bias").data[3], (await state.weightsAt(COURSE_RUN_ID, 250)).tensors.get("lm_head.bias").data[3]);
  assert.equal(await state.engineFor(COURSE_RUN_ID, 250), await state.engineFor(COURSE_RUN_ID, 250), "in cache");
  const saved = await state.saveRun(runs[0].id);
  assert.equal(saved.ok, true);
  assert.equal(state.entry(runs[0].id).saved, true);
  assert.equal((await state.storageUsage()).saved, saved.bytes);
  assert.equal(await state.forget(COURSE_RUN_ID), false, "il GPT del corso non si dimentica");
  assert.equal(await state.forget(runs[0].id), true);
  assert.equal(state.runA, COURSE_RUN_ID);
  assert.equal(store.runs.size, 0);
  // un run salvato in una visita precedente: nel registro senza pesi, letto quando serve
  await store.save(runs[0]);
  await state.refreshSaved();
  assert.equal(state.entry(runs[0].id).loaded, false);
  delete (await store.load(runs[0].id)).__x;
  assert.equal(await state.setRunA(runs[0].id), true);
  assert.equal(state.entry(runs[0].id).loaded, true);
  // un'architettura che il motore non esegue: niente motore, il motivo in italiano
  const wide = { vocab_size: 65, block_size: 8, n_embd: 16, n_head: 2, n_layer: 1, dropout: 0 };
  const bad = expectedParameters(wide).map((p) => (p.name === "blocks.0.ffwd.net.0.weight" ? { ...p, shape: [48, 16] } : p.name === "blocks.0.ffwd.net.0.bias" ? { ...p, shape: [48] } : p.name === "blocks.0.ffwd.net.2.weight" ? { ...p, shape: [16, 48] } : p));
  const { runs: badRuns } = await runsFromFiles(syntheticRun(wide, { drop: bad.map((p) => p.name), extra: bad, name: "ffn-48" }));
  state.addRuns(badRuns);
  const r = await state.engineFor(badRuns[0].id);
  assert.equal(r.engine, null);
  assert.match(r.reason, /forme diverse/);
  await assert.rejects(state.weightsAt(badRuns[0].id, 100), /non ha snapshot del training/);
  state.release(badRuns[0].id);
});

test("stato: nei run grandi gli snapshot degli step lasciati escono dalla memoria, quelli di A e B restano", async () => {
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store: memoryStore() });
  await state.init();
  const run = state.builtin;
  const loaded = () => run.snapshots.filter((s) => s.loaded).map((s) => s.step);
  const cell = async (step) => (await state.weightsAt(COURSE_RUN_ID, step)).tensors.get("lm_head.weight").data[3];
  await state.setStep(250);
  const at250 = await cell(250);
  await state.setStep(500);
  // il GPT del corso è piccolo (11 snapshot = 9 MB decodificati): con il budget vero resta tutto in cache
  assert.ok(run.steps.length * run.totalParams * 4 < SNAPSHOT_BUDGET);
  assert.equal(state.releaseSnapshots(), 0);
  assert.deepEqual(loaded(), [250, 500]);
  // come se fosse grande (budget 0): resta lo snapshot mostrato; vista e motore dello step lasciato escono
  await state.engineFor(COURSE_RUN_ID, 250);
  const views = state.cacheInfo.views;
  assert.equal(state.releaseSnapshots({ budget: 0 }), 1);
  assert.deepEqual(loaded(), [500]);
  assert.equal(state.cacheInfo.views, views - 1);
  assert.equal(state.cacheInfo.engines, 0);
  assert.equal(state.current.view.step, 500, "la vista di A resta quella di prima");
  // B sullo stesso run, interpolato tra due snapshot: restano lo snapshot di A e i due attorno a B
  await state.setInterpolate(true);
  await state.setRunB(COURSE_RUN_ID, { step: 1250 });
  await state.weightsAt(state.runB, state.stepB); // come il confronto
  await state.setStep(3000);
  state.releaseSnapshots({ budget: 0 });
  assert.deepEqual(loaded(), [1000, 1500, 3000]);
  assert.equal((await state.weightsAt(COURSE_RUN_ID, 1250)).source, "interpolated", "l'interpolazione è una copia: resta in cache");
  // senza B e con i pesi finali in A non serve più nessuno snapshot; tornando a uno step il file si rilegge
  await state.setRunB(null);
  await state.setStep(null);
  state.releaseSnapshots({ budget: 0 });
  assert.deepEqual(loaded(), []);
  assert.equal(await cell(250), at250, "gli stessi valori dal file riletto");
  assert.deepEqual(loaded(), [250]);
});

test("stato: un run salvato che il browser non legge più dà un errore leggibile e lo stato resta", async () => {
  const store = memoryStore();
  store.list = async () => [{ id: "upload:perso", name: "Run perso", description: "", savedAt: 5, bytes: 10 }];
  const state = new LabState({ loadBuiltin: () => loadBuiltinRun(fsLoaders), store });
  await state.init();
  const errors = [];
  state.on("error", ({ message }) => errors.push(message));
  assert.equal(await state.setRunA("upload:perso"), false);
  assert.equal(state.runA, COURSE_RUN_ID);
  assert.match(errors[0], /non è più leggibile nel browser/);
  assert.equal(await state.setRunA("upload:inesistente"), false);
});

test("step: cursore agganciato agli snapshot o interpolato (dichiarato), etichette", () => {
  const steps = [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000];
  const m = sliderModel(steps);
  assert.deepEqual([m.min, m.max, m.step], [0, 11, 1]);
  assert.equal(m.toStep(4), 500);
  assert.equal(m.toStep(11), null);
  assert.equal(m.toValue(null), 11);
  assert.equal(m.toValue(260), 3);
  const i = sliderModel(steps, { interpolate: true });
  assert.equal(i.toStep(753), 753);
  assert.equal(i.toStep(497), 500, "vicino a uno snapshot si aggancia allo snapshot vero");
  assert.equal(i.toStep(i.max), null);
  assert.equal(snapStep(steps, 75), 50);
  assert.equal(snapStep([], 75), null);
  assert.equal(sliderModel([]).toStep(0), null);
});

// ---------------------------------------------------------------- link diretti

test("link diretti: tensor:<nome>[@<step>], run:, view:, action: (testo non fidato)", () => {
  assert.deepEqual(parseTensorTarget("blocks.0.sa.heads.0.key.weight@500"), { name: "blocks.0.sa.heads.0.key.weight", step: 500 });
  assert.deepEqual(parseTensorTarget("lm_head.weight"), { name: "lm_head.weight", step: null });
  assert.deepEqual(parseTensorTarget("lm_head.weight@finali"), { name: "lm_head.weight", step: null });
  for (const bad of ["", "@500", "lm_head.weight@abc", "lm head", "a\u0000b", "x".repeat(700), null, 42]) assert.equal(parseTensorTarget(bad), null, String(bad).slice(0, 20));
  assert.equal(formatTensorTarget("lm_head.weight", 500), "lm_head.weight@500");
  assert.equal(formatTensorTarget("lm_head.weight", null), "lm_head.weight");
  assert.equal(parseRunTarget("corso"), COURSE_RUN_ID);
  assert.equal(RUN_ALIASES.builtin, COURSE_RUN_ID);
  assert.deepEqual(resolveTarget({ kind: "tensor", id: "ln_f.bias@0" }), { type: "tensor", name: "ln_f.bias", step: 0 });
  assert.deepEqual(resolveTarget({ kind: "view", id: "tensor:ln_f.bias@0" }), { type: "tensor", name: "ln_f.bias", step: 0 });
  assert.deepEqual(resolveTarget({ kind: "view", id: "step:0" }), { type: "step", step: 0 });
  assert.deepEqual(resolveTarget({ kind: "view", id: "step:finali" }), { type: "step", step: null });
  assert.deepEqual(resolveTarget({ kind: "view", id: "rete" }), { type: "rep", id: "rete" });
  assert.deepEqual(resolveTarget({ kind: "run", id: "upload:abc" }), { type: "run", id: "upload:abc" });
  assert.deepEqual(resolveTarget({ kind: "action", id: "attention" }), { type: "action", id: "attention" });
  assert.deepEqual(resolveTarget({ kind: "view", id: "esporta" }), { type: "view", id: "esporta" });
  assert.equal(resolveTarget({ kind: "view", id: "inesistente" }), null);
  assert.equal(resolveTarget({ kind: "boh", id: "x" }), null);
  assert.equal(resolveTarget({}), null);
});

test("indice della ricerca: voci valide, id unici, ogni vista si apre con un link", () => {
  const keys = new Set();
  for (const e of searchIndex) {
    assert.ok(["step", "action", "view"].includes(e.kind));
    assert.ok(!keys.has(`${e.kind}:${e.id}`), e.id);
    keys.add(`${e.kind}:${e.id}`);
    if (e.kind === "view") assert.ok(resolveTarget(e), `link non risolto: ${e.id}`);
    const t = resolveTarget(e);
    if (t?.type === "tensor" && !t.name.endsWith(".tril")) assert.ok(builtin.tensor(t.name), `tensore inesistente: ${t.name}`);
  }
  for (const id of Object.keys(VIEW_IDS)) assert.ok(keys.has(`view:${id}`) || ["centro"].includes(id), `vista senza voce: ${id}`);
  // i testi che citano una riga del vocabolario devono dire il carattere vero (era «la riga 12 è «e»»: è «?»)
  for (const e of searchIndex) {
    for (const m of String(e.text ?? "").matchAll(/riga (\d+) è il carattere «(.)»/g)) assert.equal(builtin.vocab[Number(m[1])], m[2], `${e.id}: ${m[0]}`);
  }
});

// ---------------------------------------------------------------- caricamenti rotti

test("caricamenti rotti: JSON rotto, offset sbagliati, file mancanti, nomi con HTML: errori in italiano, niente eccezioni", async () => {
  const cfg = { vocab_size: 11, block_size: 4, n_embd: 8, n_head: 2, n_layer: 1, dropout: 0 };
  const [manifest, bin] = syntheticRun(cfg, { name: "ok" });
  const good = JSON.parse(await manifest.text());
  const shifted = { ...good, params: good.params.map((p, i) => (i === 1 ? { ...p, offset: p.offset + 3 } : p)) };
  const html = { ...good, params: good.params.map((p, i) => (i === 0 ? { ...p, name: "<script>alert(1)</script>" } : p)) };
  const result = await runsFromFiles([
    manifest,
    bin,
    fileLike("rotto/weights.json", "{non è json"),
    fileLike("offset/weights.json", JSON.stringify(shifted)),
    fileLike("offset/weights.f32.bin", new Uint8Array(await bin.arrayBuffer())),
    fileLike("senzabin/weights.json", JSON.stringify(good)),
    fileLike("html/weights.json", JSON.stringify(html)),
    fileLike("html/weights.f32.bin", new Uint8Array(await bin.arrayBuffer())),
    fileLike("strano/leggimi.txt", "ciao"),
  ]);
  const byRun = new Map(result.errors.map((e) => [e.run, e.message]));
  assert.equal(result.runs.length, 2, "il run buono e quello con un nome strano si caricano");
  assert.match(byRun.get("rotto"), /JSON/i);
  assert.match(byRun.get("offset"), /offset|sovrappo|buco|intervall/i);
  assert.match(byRun.get("senzabin"), /manca il file dei pesi weights\.f32\.bin/);
  assert.ok(result.ignored.some((i) => i.path.endsWith("leggimi.txt")));
  const weird = result.runs.find((r) => r.tensors.has("<script>alert(1)</script>"));
  const tree = buildModuleTree(weird.params, { snippets: spec.snippets, config: weird.config });
  const L = layoutModel(tree, { config: weird.config, snippets: spec.snippets });
  assert.equal(L.nodes.get("<script>alert(1)</script>").label, "<script>alert(1)</script> (11, 8)");
  // oltre il limite totale: un errore prima di leggere qualunque cosa
  await assert.rejects(runsFromFiles([manifest, bin], { limits: { maxTotalBytes: 10 } }), /limite/i);
});

// ---------------------------------------------------------------- significato, descrizioni, impacchettamento

test("significato di righe, colonne e celle; ruoli dei moduli", () => {
  const ctx = { vocab: builtin.vocab, config: builtin.config };
  const tok = tensorMeaning("token_embedding_table.weight", [65, 64], ctx);
  assert.equal(tok.rows.describe(builtin.vocab.indexOf("e")), `riga ${builtin.vocab.indexOf("e")}: il carattere «e» (token ${builtin.vocab.indexOf("e")})`);
  assert.equal(tok.rows.label(1), "␣");
  const key = tensorMeaning("blocks.2.sa.heads.3.key.weight", [16, 64], ctx);
  assert.match(key.rows.describe(5), /uscita 5 della key della head 3/);
  assert.match(key.cols.describe(7), /dimensione 7 di x dopo ln1/);
  assert.match(tensorMeaning("blocks.0.sa.proj.weight", [64, 64], ctx).cols.describe(37), /head 2, dimensione 5/);
  assert.match(tensorMeaning("blocks.0.ffwd.net.2.weight", [64, 256], ctx).cols.describe(9), /neurone nascosto 9/);
  assert.match(tensorMeaning("blocks.0.ln2.bias", [64], ctx).title, /β/);
  assert.equal(tensorMeaning("blocks.0.ln2.bias", [64], ctx).cols, null);
  assert.match(tensorMeaning("blocks.0.sa.heads.0.tril", [32, 32], ctx).cell(2, 5), /futuro/);
  assert.match(tensorMeaning("strano.peso", [3, 3], ctx).rows.describe(1), /riga 1/);
  assert.equal(roleOf("blocks.0.ln1"), "norm");
  assert.equal(roleOf("blocks.0.sa.heads.0.key"), "attention");
  assert.equal(roleOf("blocks.0.ffwd.net.0"), "mlp");
  assert.equal(roleOf("lm_head"), "output");
  assert.equal(roleOf("token_embedding_table"), "embedding");
  assert.equal(roleOf("transformer.h.0.attn.c_attn"), "attention");
});

test("descrizioni del run: config con differenze, byte, date", () => {
  const diff = configDiff({ ...builtin.config, n_layer: 6, rope: true }, builtin.config);
  assert.deepEqual(diff.find((d) => d.key === "n_layer"), { key: "n_layer", label: "blocchi transformer", value: 6, course: 4, changed: true, extra: false });
  assert.equal(diff.find((d) => d.key === "rope").extra, true);
  assert.equal(formatBytes(838916), "839 kB");
  assert.equal(formatBytes(2_600_000), "2,6 MB");
  assert.equal(formatDate("non una data"), "non una data");
  assert.match(formatDate("2026-11-09T01:14:26+00:00"), /2026/);
  assert.match(trainingText(builtin), /11 snapshot/);
});

test("velature senza sovrapposizioni: rettangolo meno i figli = la parte scoperta, esatta", () => {
  const area = (r) => (r.x1 - r.x0) * (r.y1 - r.y0);
  const r = { x0: 0, x1: 10, y0: 0, y1: 6 };
  const holes = [{ x0: 1, x1: 3, y0: 1, y1: 5 }, { x0: 4, x1: 9, y0: 2, y1: 3 }, { x0: 8, x1: 12, y0: 4, y1: 7 }];
  const strips = subtractRects(r, holes);
  const clipped = holes.reduce((s, h) => s + Math.max(0, Math.min(r.x1, h.x1) - Math.max(r.x0, h.x0)) * Math.max(0, Math.min(r.y1, h.y1) - Math.max(r.y0, h.y0)), 0);
  assert.ok(Math.abs(strips.reduce((s, x) => s + area(x), 0) - (area(r) - clipped)) < 1e-9);
  for (const a of strips) for (const h of holes) assert.ok(!(a.x0 < h.x1 && h.x0 < a.x1 && a.y0 < h.y1 && h.y0 < a.y1), "una striscia non entra in un figlio");
  assert.deepEqual(subtractRects(r, []), [r]);
  // su tutto il modello: per ogni contenitore, strisce + figli = il contenitore (ogni pixel colorato una volta)
  const L = courseLayout();
  for (const b of L.boxes.filter((x) => x.kind === "module")) {
    const kids = coveredBy(b);
    const covered = subtractRects(b, kids).reduce((s, x) => s + area(x), 0) + kids.reduce((s, k) => s + area(k), 0);
    assert.ok(Math.abs(covered - area(b)) < 1e-6 * area(b), b.path);
  }
});

test("atlante delle etichette: scaffali senza sovrapposizioni, pagine a potenze di 2", () => {
  const sizes = Array.from({ length: 300 }, (_, i) => ({ w: 100 + ((i * 37) % 900), h: 20 + ((i * 13) % 90) }));
  const { pages, places } = packShelves(sizes, { pageWidth: 2048, maxPageHeight: 1024, padding: 8 });
  assert.ok(pages.length >= 2);
  for (const p of pages) assert.ok(Number.isInteger(Math.log2(p.height)));
  for (let a = 0; a < sizes.length; a++) {
    const pa = places[a];
    assert.ok(pa.x + sizes[a].w <= 2048 && pa.y + sizes[a].h <= pages[pa.page].height);
    for (let b = a + 1; b < sizes.length; b++) {
      const pb = places[b];
      if (pa.page !== pb.page) continue;
      const overlap = pa.x < pb.x + sizes[b].w && pb.x < pa.x + sizes[a].w && pa.y < pb.y + sizes[b].h && pb.y < pa.y + sizes[a].h;
      assert.ok(!overlap, `${a} e ${b} si sovrappongono`);
    }
  }
});

// ---------------------------------------------------------------- rete neurale e cervello

test("rete neurale: la rete vera del run A (motore) e quella di sola struttura (dichiarata)", async () => {
  const engine = builtin.createEngine();
  const spec1 = trueSpec({ run: builtin, engine, label: "pesi finali (float32)" });
  const model = buildNetworkModel(spec1);
  assert.equal(model.values.simulated.size, 0, "valori veri: nessuna sezione simulata");
  assert.equal(spec1.depth.count, encodeWith(builtin.vocab, NETWORK_PROMPT).length);
  assert.match(spec1.title, /vera/);
  // GELU dichiarata o FFN diversa: struttura e pesi veri, attivazioni simulate e dichiarate
  const cfg = { vocab_size: 65, block_size: 16, n_embd: 32, n_head: 4, n_layer: 2, dropout: 0 };
  const { runs } = await runsFromFiles(syntheticRun(cfg, { name: "struttura" }), { defaultVocab: builtin.vocab });
  const s = structureSpec({ run: runs[0], label: "pesi finali (float32)", reason: "run.json dichiara engine_compatible: false" });
  assert.match(s.intro, /Valori non disponibili: architettura diversa dal motore JS/);
  const m2 = buildNetworkModel(s);
  assert.ok(m2.values.simulated.size > 0, "il renderer dichiara la simulazione");
  const explicit = s.links.filter((l) => l.pattern === "explicit");
  assert.ok(explicit.length >= 2 * (3 * 4 + 3) + 1);
  const w = runs[0].tensor("lm_head.weight").data;
  const lm = explicit.find((l) => /lm_head\.weight/.test(l.label));
  const e = lm.pairs.to[0] * 32 + lm.pairs.from[0];
  assert.equal(lm.pairs.weight[0], w[e], "i collegamenti hanno il peso vero");
  // architettura sconosciuta (nomi diversi dal corso): moduli in fila nell'ordine dei pesi
  const odd = new Map([["enc.lin.weight", { name: "enc.lin.weight", shape: [8, 4], data: new Float32Array(32).fill(0.1) }], ["enc.out.weight", { name: "enc.out.weight", shape: [3, 8], data: new Float32Array(24).fill(-0.2) }]]);
  const chain = structureSpec({ run: { name: "strano", tensors: odd, vocab: null }, label: "pesi finali" });
  assert.equal(buildNetworkModel(chain).net.sections.length, 2);
});

test("cervello: brainSpec dalla ricerca, valido con strict e senza avvisi", () => {
  const s = normalizeBrainSpec(brainSpec, { strict: true });
  assert.deepEqual(s.warnings, []);
  assert.equal(s.analogies.length, 10);
  assert.ok(s.analogies.every((a) => a.different && a.sources.length));
  assert.equal(s.circuit.groups.length, 8);
  assert.ok(s.regions.includes("cerebellum") && s.regions.includes("hippocampus"));
  assert.ok(brainSpec.facts.some((f) => /209\.729 parametri/.test(f.value)));
  assert.ok(!/<[a-z]/i.test(JSON.stringify(brainSpec)), "niente HTML nei testi");
});

test("albero: findNode e righe di run.json per i moduli che il codice del corso non spiega", () => {
  const tree = courseTree();
  assert.equal(findNode(tree, "blocks.0.sa.heads.0.key.weight").code.role, "matrice W (uscite × ingressi): y = x · Wᵀ + b");
  const classes = classesFromRunInfo({ path: "", class: "GPTLanguageModel", children: [{ path: "blocks", class: "nn.Sequential", children: [{ path: "blocks.0", class: "Block" }] }] });
  assert.equal(classes.get("blocks.0"), "Block");
  assert.equal(classesFromRunInfo(null).size, 0);
  assert.equal(classesFromRunInfo({ path: "x", class: 5 }).size, 0);
  // run.json è un file dello studente: la classe va nelle etichette, senza controlli bidirezionali (prima restavano)
  const strange = classesFromRunInfo({ path: "", class: "GPT\u202eledoM", children: [{ path: "a", class: "\u2066\u202e" }] });
  assert.equal(strange.get(""), "GPTledoM");
  assert.equal(strange.has("a"), false, "una classe fatta solo di controlli non c'è");
});
