// Test della libreria dei pesi (js/shared/weights/) e delle matrici con i numeri (js/core/matrixNumbers.js):
//   node --test webapp/tests/*.test.mjs
// Formati (manifest, trace, snapshot, safetensors costruiti byte per byte), riconoscimento dei file
// caricati, run del corso e run caricati, compatibilità con il motore JS, statistiche, confronto,
// albero dei moduli (righe Python controllate sui file gpt/*.py veri), salvataggio, formato dei
// numeri e, con three.js caricato dalla cartella vendor, NumericMatrix / attachNumbers con un
// canvas finto che registra cosa viene scritto.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import { GPTEngine, expectedParameters, loadEngine } from "../js/engine/gpt.js";
import { compareTensors, difference, tensorDiff } from "../js/shared/weights/compare.js";
import {
  capWindow,
  chooseCellFormat,
  contrastRatio,
  expandWindow,
  formatCell,
  formatPrecise,
  matrixShape,
  quantizeSigned,
  relativeLuminance,
  textColorFor,
  textContrastFor,
  windowCells,
  windowContains,
  windowFromPoints,
} from "../js/shared/weights/display.js";
import { filePath, recognizeFiles } from "../js/shared/weights/files.js";
import { decodeSnapshot, decodeWeights, parseBenchmark, parseRunInfo, parseTrace, parseWeightsManifest } from "../js/shared/weights/formats.js";
import { groupOf, groupsFromTrace } from "../js/shared/weights/groups.js";
import { buildModuleTree, findNode, pythonPath, walkTree } from "../js/shared/weights/moduleTree.js";
import { parseModuleSource, splitComment } from "../js/shared/weights/pythonSource.js";
import { engineCompatibility, fingerprint, inferGptConfig, loadBuiltinRun, runsFromFiles, snapshotBracket } from "../js/shared/weights/run.js";
import { parseSafetensors, safetensorsInfo } from "../js/shared/weights/safetensors.js";
import { histogram, percentiles, range, tensorStats } from "../js/shared/weights/stats.js";
import { RunStore, deserializeRun, serializeRun } from "../js/shared/weights/store.js";
import { WeightsFormatError, canonicalName, checkLayout, naturalCompare } from "../js/shared/weights/validate.js";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const DATA_DIR = fileURLToPath(new URL("../data/", import.meta.url));
const HAS_DATA = existsSync(`${DATA_DIR}weights.json`) && existsSync(`${DATA_DIR}weights.f32.bin`);
const HAS_TRACE = existsSync(`${DATA_DIR}training/trace.json`);
const loadJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const loadBuffer = async (file) => {
  const bytes = await readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};
const fsLoaders = { baseUrl: DATA_DIR, loadJson, loadBuffer };

// ---------------------------------------------------------------- aiuti

const SMALL = { vocab_size: 11, block_size: 8, n_embd: 16, n_head: 4, n_layer: 2, dropout: 0 };
const SMALL_VOCAB = ["\n", " ", "a", "b", "c", "d", "e", "f", "g", "h", "!"];

/** Manifest + pesi casuali (seme fisso) con l'architettura del corso, a una config qualsiasi. */
function synthetic(config = SMALL, { vocab = SMALL_VOCAB, extra = [], seed = 7, rename = (n) => n } = {}) {
  let offset = 0;
  const params = [...expectedParameters(config), ...extra].map(({ name, shape }) => {
    const numel = shape.reduce((a, b) => a * b, 1);
    const entry = { name: rename(name), shape, offset, numel };
    offset += numel;
    return entry;
  });
  let s = seed;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32) * 2 - 1;
  const values = Float32Array.from({ length: offset }, () => rnd() * 0.5);
  const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", config, vocab, total_params: offset, params, training: null };
  return { manifest, values, bytes: new Uint8Array(values.buffer) };
}

/** Oggetto "tipo File" con percorso relativo (come quelli di una cartella con webkitdirectory). */
function fileLike(path, content) {
  const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content instanceof Uint8Array ? content : new Uint8Array(content);
  return {
    name: path.split("/").pop(),
    size: bytes.byteLength,
    webkitRelativePath: path,
    text: async () => new TextDecoder().decode(bytes),
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
}
const sized = (path, size) => ({ name: path.split("/").pop(), size, webkitRelativePath: path, text: async () => "", arrayBuffer: async () => new ArrayBuffer(0) });

/** Un .safetensors scritto a mano: 8 byte di lunghezza (u64 little-endian), intestazione JSON, dati. */
function safetensors(header, data = new Uint8Array(0), { pad = 0, declaredLength = null } = {}) {
  const json = new TextEncoder().encode(JSON.stringify(header) + " ".repeat(pad));
  const out = new Uint8Array(8 + json.length + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, declaredLength ?? json.length, true);
  view.setUint32(4, 0, true);
  out.set(json, 8);
  out.set(data, 8 + json.length);
  return out;
}
const f32 = (...values) => new Uint8Array(new Float32Array(values).buffer);
const u16 = (...values) => new Uint8Array(new Uint16Array(values).buffer);
const concat = (...parts) => {
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};
const throwsFormat = (fn, pattern) => assert.throws(fn, (error) => error instanceof WeightsFormatError && pattern.test(error.message));

// ---------------------------------------------------------------- manifest, trace, snapshot

test("weights.json del corso: layout controllato e tensori come viste sul file float32", { skip: !HAS_DATA }, async () => {
  const raw = await loadJson(`${DATA_DIR}weights.json`);
  const manifest = parseWeightsManifest(raw);
  assert.equal(manifest.params.length, 94);
  assert.equal(manifest.totalParams, 209729);
  assert.equal(manifest.vocab.length, 65);
  assert.deepEqual(manifest.training, { dir: "training", trace: "training/trace.json" });
  const { flat, tensors } = decodeWeights(await loadBuffer(`${DATA_DIR}weights.f32.bin`), manifest);
  assert.equal(flat.length, 209729);
  const engine = await loadEngine(DATA_DIR, { loadJson, loadBuffer });
  for (const [name, view] of engine.paramViews) assert.deepEqual(tensors.get(name).data, view.data, name);
  assert.equal(tensors.get("blocks.0.ffwd.net.0.weight").data.buffer, flat.buffer, "nessuna copia: viste sullo stesso buffer");
});

test("weights.json: errori chiari per manifest sbagliati", () => {
  const { manifest } = synthetic();
  const withChange = (fn) => {
    const m = structuredClone(manifest);
    fn(m);
    return () => parseWeightsManifest(m);
  };
  throwsFormat(withChange((m) => (m.format_version = 2)), /format_version 2 non supportato/);
  throwsFormat(withChange((m) => (m.dtype = "float16")), /float32 little-endian/);
  throwsFormat(withChange((m) => (m.byte_order = "big")), /float32 little-endian/);
  throwsFormat(withChange((m) => (m.file = "../../etc/passwd")), /contiene ".."/);
  throwsFormat(withChange((m) => (m.file = "/abs/w.bin")), /percorso relativo/);
  throwsFormat(withChange((m) => (m.params = [])), /manca la lista dei parametri/);
  throwsFormat(withChange((m) => (m.params[3].name = m.params[2].name)), /compare due volte/);
  throwsFormat(withChange((m) => (m.params[0].shape = [11, -16])), /dimensione non valida/);
  throwsFormat(withChange((m) => (m.params[0].shape = [11.5, 16])), /dimensione non valida/);
  throwsFormat(withChange((m) => (m.params[0].shape = [1, 1, 1, 1, 1, 1, 1, 1, 1])), /9 dimensioni/);
  throwsFormat(withChange((m) => (m.params[0].numel = 7)), /numel di token_embedding_table.weight è 7/);
  throwsFormat(withChange((m) => (m.params[1].offset = 5)), /si sovrappone/);
  throwsFormat(withChange((m) => (m.params[1].offset = -1)), /offset .* non valido/);
  throwsFormat(withChange((m) => (m.params.at(-1).offset += 10)), /esce dal file/);
  throwsFormat(withChange((m) => (m.total_params += 5)), /sommano a/);
  throwsFormat(withChange((m) => (m.vocab = m.vocab.slice(1))), /10 voci ma vocab_size è 11/);
  throwsFormat(withChange((m) => (m.vocab[1] = m.vocab[0])), /duplicati/);
  throwsFormat(withChange((m) => (m.config.n_head = 0)), /config.n_head deve essere un intero positivo/);
  throwsFormat(withChange((m) => (m.params[0].name = "a\u0000b")), /caratteri di controllo/);
  throwsFormat(() => parseWeightsManifest([1, 2]), /oggetto JSON/);
  // Il limite di valori protegge la pagina da file enormi o sbagliati.
  throwsFormat(() => parseWeightsManifest(manifest, { limits: { maxParams: 100 } }), /limite/);
  // Il file binario deve avere esattamente total_params float.
  const parsed = parseWeightsManifest(manifest);
  throwsFormat(() => decodeWeights(new ArrayBuffer(parsed.totalParams * 4 - 4), parsed), /attesi .* byte/);
});

test("weights.json di un'architettura modificata: si legge (la compatibilità con il motore è a parte)", () => {
  const { manifest, bytes } = synthetic(SMALL, { extra: [{ name: "blocks.0.sa.rope.inv_freq", shape: [8] }] });
  const parsed = parseWeightsManifest(manifest);
  const { tensors } = decodeWeights(bytes, parsed);
  assert.ok(tensors.has("blocks.0.sa.rope.inv_freq"));
  assert.equal(parsed.config.n_layer, 2);
});

test("checkLayout: offset in qualunque ordine, ma senza buchi né sovrapposizioni", () => {
  const ok = checkLayout([{ name: "b", shape: [2], offset: 3 }, { name: "a", shape: [3], offset: 0 }], 5, "x");
  assert.equal(ok.total, 5);
  throwsFormat(() => checkLayout([{ name: "a", shape: [3], offset: 0 }, { name: "b", shape: [2], offset: 4 }], 6, "x"), /esce dal file|sommano/);
  throwsFormat(() => checkLayout([{ name: "a", shape: [3], offset: 0 }, { name: "b", shape: [2], offset: 2 }], 5, "x"), /si sovrappone/);
});

test("trace.json del corso: 11 snapshot in ordine, stesso layout dei pesi", { skip: !HAS_TRACE }, async () => {
  const trace = parseTrace(await loadJson(`${DATA_DIR}training/trace.json`));
  assert.deepEqual(trace.snapshots.map((s) => s.step), [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.equal(trace.totalParams, 209729);
  assert.equal(trace.layout.length, 94);
  assert.equal(trace.train_loss.length, 5000, "le metriche restano quelle del file");
  const manifest = parseWeightsManifest(await loadJson(`${DATA_DIR}weights.json`));
  assert.deepEqual(trace.layout, manifest.params);
});

test("trace.json: errori su dtype, step doppi e file degli snapshot", () => {
  const { manifest } = synthetic();
  const base = { format_version: 1, snapshot_dtype: "float16", byte_order: "little", params: manifest.params, total_params: manifest.total_params, snapshots: [{ step: 0, file: "snap_00000.f16.bin" }] };
  assert.equal(parseTrace(base).snapshots.length, 1);
  throwsFormat(() => parseTrace({ ...base, snapshot_dtype: "float32" }), /atteso float16/);
  throwsFormat(() => parseTrace({ ...base, format_version: 3 }), /format_version 3/);
  throwsFormat(() => parseTrace({ ...base, snapshots: [{ step: 0, file: "a.bin" }, { step: 0, file: "b.bin" }] }), /compare due volte/);
  throwsFormat(() => parseTrace({ ...base, snapshots: [{ step: -1, file: "a.bin" }] }), /step non valido/);
  throwsFormat(() => parseTrace({ ...base, snapshots: [{ step: 1, file: "../a.bin" }] }), /".."/);
  throwsFormat(() => parseTrace({ ...base, snapshots: [{ step: 1, file: "sub/a.bin" }] }), /accanto a trace.json/);
  const sorted = parseTrace({ ...base, snapshots: [{ step: 9, file: "b.bin" }, { step: 3, file: "a.bin" }] });
  assert.deepEqual(sorted.snapshots.map((s) => s.step), [3, 9]);
});

test("snapshot float16: conversione esatta e lunghezza controllata", () => {
  // 1, −2, 65504 (massimo), 2^-24 (subnormale più piccolo), +∞, NaN
  const values = decodeSnapshot(u16(0x3c00, 0xc000, 0x7bff, 0x0001, 0x7c00, 0x7e00).buffer, 6);
  assert.deepEqual(Array.from(values.slice(0, 5)), [1, -2, 65504, 2 ** -24, Infinity]);
  assert.ok(Number.isNaN(values[5]));
  throwsFormat(() => decodeSnapshot(new ArrayBuffer(10), 6, { file: "snap_00001.f16.bin" }), /snap_00001.f16.bin: snapshot float16: attesi 12 byte, ricevuti 10/);
});

test("run.json e benchmark.json: campi semplici, il resto nel JSON originale", () => {
  const info = parseRunInfo({ name: "GPT con GELU", description: "prova", created: "2026-11-01", notes: ["a", 3, "b"], engine_compatible: false, config: SMALL, extra: { x: 1 } });
  assert.equal(info.name, "GPT con GELU");
  assert.deepEqual(info.notes, ["a", "b"]);
  assert.equal(info.engineCompatible, false);
  assert.equal(info.raw.extra.x, 1);
  throwsFormat(() => parseRunInfo("ciao"), /oggetto JSON/);
  const bench = parseBenchmark({ val_loss: 1.82, perplexity: 6.17, text: "ciao", metrics: { top1: 0.45 } });
  assert.deepEqual(bench.metrics, { val_loss: 1.82, perplexity: 6.17, top1: 0.45 });
});

// ---------------------------------------------------------------- safetensors

test("safetensors: F32, F16 e BF16 in Float32Array, metadati e ordine dei byte", () => {
  const data = concat(f32(1.5, -2.25), u16(0x3c00, 0xc000, 0x7bff), u16(0x3f80, 0xc000, 0x3e20, 0x0000));
  const header = {
    __metadata__: { format: "pt", config: JSON.stringify(SMALL), vocab: JSON.stringify(SMALL_VOCAB) },
    "a.weight": { dtype: "F32", shape: [2], data_offsets: [0, 8] },
    "b.half": { dtype: "F16", shape: [3], data_offsets: [8, 14] },
    "c.bf16": { dtype: "BF16", shape: [2, 2], data_offsets: [14, 22] },
  };
  const parsed = parseSafetensors(safetensors(header, data));
  assert.deepEqual(Array.from(parsed.tensors.get("a.weight").data), [1.5, -2.25]);
  assert.deepEqual(Array.from(parsed.tensors.get("b.half").data), [1, -2, 65504]);
  assert.deepEqual(Array.from(parsed.tensors.get("c.bf16").data), [1, -2, 0.15625, 0]);
  assert.deepEqual(parsed.tensors.get("c.bf16").shape, [2, 2]);
  assert.equal(parsed.tensors.get("b.half").dtype, "float16");
  assert.deepEqual(parsed.order, ["a.weight", "b.half", "c.bf16"]);
  assert.equal(parsed.flat.length, 9);
  const meta = safetensorsInfo(parsed.metadata);
  assert.deepEqual(meta.config, SMALL);
  assert.deepEqual(meta.vocab, SMALL_VOCAB);
  assert.deepEqual(meta.other, { format: "pt" });
});

test("safetensors: intestazione di lunghezza dispari (dati non allineati) e tensori vuoti", () => {
  const header = { x: { dtype: "F32", shape: [3], data_offsets: [0, 12] }, vuoto: { dtype: "F32", shape: [0, 4], data_offsets: [12, 12] } };
  for (let pad = 0; pad < 4; pad++) {
    const parsed = parseSafetensors(safetensors(header, f32(0.1, 0.2, 0.3), { pad }));
    assert.deepEqual(Array.from(parsed.tensors.get("x").data), Array.from(new Float32Array([0.1, 0.2, 0.3])), `pad ${pad}`);
    assert.equal(parsed.tensors.get("vuoto").numel, 0);
  }
  // Anche da una vista che non parte dall'inizio del suo buffer (es. un Buffer di Node).
  const bytes = safetensors(header, f32(1, 2, 3), { pad: 1 });
  const shifted = new Uint8Array(bytes.length + 3);
  shifted.set(bytes, 3);
  assert.deepEqual(Array.from(parseSafetensors(shifted.subarray(3)).tensors.get("x").data), [1, 2, 3]);
});

test("safetensors: file non validi o pericolosi rifiutati con un messaggio chiaro", () => {
  const one = { x: { dtype: "F32", shape: [2], data_offsets: [0, 8] } };
  throwsFormat(() => parseSafetensors(new Uint8Array(5)), /almeno 8 byte/);
  throwsFormat(() => parseSafetensors(safetensors(one, f32(1, 2), { declaredLength: 999999 })), /il file ne ha solo/);
  throwsFormat(() => parseSafetensors(safetensors(one, f32(1, 2)), { limits: { maxSafetensorsHeaderBytes: 10 } }), /il limite è 10/);
  const huge = new Uint8Array(16);
  new DataView(huge.buffer).setUint32(4, 1, true); // lunghezza ≥ 2^32: non sta nei limiti
  throwsFormat(() => parseSafetensors(huge), /limite/);
  const badUtf8 = concat(new Uint8Array([4, 0, 0, 0, 0, 0, 0, 0]), new Uint8Array([0x7b, 0xff, 0xfe, 0x7d]));
  throwsFormat(() => parseSafetensors(badUtf8), /UTF-8/);
  throwsFormat(() => parseSafetensors(concat(new Uint8Array([3, 0, 0, 0, 0, 0, 0, 0]), new TextEncoder().encode("{x}"))), /JSON non valida/);
  throwsFormat(() => parseSafetensors(safetensors([1, 2])), /deve essere un oggetto/);
  throwsFormat(() => parseSafetensors(safetensors({ __metadata__: { n: 3 } })), /__metadata__.n deve essere una stringa/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "I64", shape: [1], data_offsets: [0, 8] } }, new Uint8Array(8))), /dtype I64 non supportato: sono supportati F32, F16 e BF16/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F64", shape: [1], data_offsets: [0, 8] } }, new Uint8Array(8))), /F64 non supportato/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "Q4_K", shape: [1], data_offsets: [0, 1] } }, new Uint8Array(1))), /dtype sconosciuto/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F32", shape: [2] } }, f32(1, 2))), /data_offsets/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F32", shape: [2], data_offsets: [8, 0] } }, f32(1, 2))), /l'inizio supera la fine/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F32", shape: [4], data_offsets: [0, 16] } }, f32(1, 2))), /escono dai dati/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F32", shape: [3], data_offsets: [0, 8] } }, f32(1, 2))), /ne richiede 12/);
  throwsFormat(() => parseSafetensors(safetensors({ x: { dtype: "F32", shape: [-2], data_offsets: [0, 8] } }, f32(1, 2))), /dimensione non valida/);
  const overlap = { a: { dtype: "F32", shape: [2], data_offsets: [0, 8] }, b: { dtype: "F32", shape: [2], data_offsets: [4, 12] } };
  throwsFormat(() => parseSafetensors(safetensors(overlap, f32(1, 2, 3))), /si sovrappone/);
  const hole = { a: { dtype: "F32", shape: [1], data_offsets: [0, 4] }, b: { dtype: "F32", shape: [1], data_offsets: [8, 12] } };
  throwsFormat(() => parseSafetensors(safetensors(hole, f32(1, 2, 3))), /buco nei dati: i byte 4-8/);
  throwsFormat(() => parseSafetensors(safetensors(one, f32(1, 2, 3))), /4 byte in più dopo l'ultimo tensore/);
  throwsFormat(() => parseSafetensors(safetensors(one, f32(1, 2)), { limits: { maxParams: 1 } }), /limite/);
});

test("safetensors: un tensore chiamato __proto__ resta un nome qualunque", () => {
  const bytes = concat(new Uint8Array(8), new TextEncoder().encode('{"__proto__":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'), f32(7));
  new DataView(bytes.buffer).setUint32(0, bytes.length - 12, true);
  const parsed = parseSafetensors(bytes);
  assert.deepEqual(Array.from(parsed.tensors.get("__proto__").data), [7]);
  assert.equal({}.dtype, undefined, "Object.prototype non è stato toccato");
});

// ---------------------------------------------------------------- riconoscimento dei file

test("recognizeFiles: la cartella data/ del corso è un run solo (training/ è la sua storia)", () => {
  const files = [
    sized("data/weights.json", 8000),
    sized("data/weights.f32.bin", 838916),
    sized("data/model.json", 338250),
    sized("data/training/trace.json", 165621),
    sized("data/training/snap_00000.f16.bin", 419458),
    sized("data/training/snap_05000.f16.bin", 419458),
    sized("data/.DS_Store", 6000),
  ];
  const { runs, ignored, totalBytes } = recognizeFiles(files);
  assert.equal(runs.length, 1);
  const [run] = runs;
  assert.equal(run.dir, "data");
  assert.equal(run.name, "data");
  assert.equal(run.manifest.path, "data/weights.json");
  assert.deepEqual(run.binaries.map((b) => b.path), ["data/weights.f32.bin"]);
  assert.equal(run.trace.path, "data/training/trace.json");
  assert.deepEqual(run.snapshots.map((s) => s.step), [0, 5000]);
  assert.deepEqual(ignored.map((i) => i.path), ["data/model.json"]);
  assert.match(ignored[0].reason, /scena del Transformer/);
  assert.equal(totalBytes, 8000 + 838916 + 165621 + 2 * 419458);
});

test("recognizeFiles: sottocartelle = più run, shard uniti, safetensors separati, solo storia del training", () => {
  const { runs, ignored } = recognizeFiles([
    sized("runs/a/weights.json", 10),
    sized("runs/a/weights.f32.bin", 40),
    sized("runs/a/benchmark.json", 10),
    sized("runs/b/run.json", 10),
    sized("runs/b/model-00001-of-00002.safetensors", 100),
    sized("runs/b/model-00002-of-00002.safetensors", 100),
    sized("runs/c/step1000.safetensors", 50),
    sized("runs/c/step2000.safetensors", 50),
    sized("runs/d/trace.json", 10),
    sized("runs/d/snap_00100.f16.bin", 20),
    sized("runs/e/model-00001-of-00003.safetensors", 100),
    sized("runs/leggimi.txt", 10),
    sized("runs/../segreto/weights.json", 10),
    sized("runs/orfano.bin", 10),
  ]);
  const byKey = new Map(runs.map((r) => [r.key, r]));
  assert.deepEqual([...byKey.keys()], ["runs/a", "runs/b", "runs/c", "runs/c/step2000", "runs/d", "runs/e"]);
  assert.equal(byKey.get("runs/a").benchmark.path, "runs/a/benchmark.json");
  assert.equal(byKey.get("runs/b").safetensors.length, 2, "gli shard formano un run");
  assert.equal(byKey.get("runs/b").runInfo.path, "runs/b/run.json");
  assert.equal(byKey.get("runs/c").name, "step1000");
  assert.equal(byKey.get("runs/c/step2000").safetensors[0].path, "runs/c/step2000.safetensors");
  assert.equal(byKey.get("runs/d").trace.path, "runs/d/trace.json");
  assert.match(byKey.get("runs/e").warnings[0], /mancano gli shard 2, 3 di 3/);
  const reasons = new Map(ignored.map((i) => [i.path, i.reason]));
  assert.match(reasons.get("runs/leggimi.txt"), /non riconosciuto/);
  assert.match(reasons.get("runs/../segreto/weights.json"), /".."/);
  assert.match(reasons.get("runs/orfano.bin"), /fuori da una cartella di run/);
});

test("recognizeFiles: selezione multipla senza cartelle (File veri di Node) e limiti di dimensione", () => {
  const files = [new File(["{}"], "weights.json"), new File([new Uint8Array(8)], "weights.f32.bin")];
  const { runs } = recognizeFiles(files);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].key, ".");
  assert.equal(filePath(files[0]), "weights.json");
  assert.equal(filePath({ name: "x.bin", webkitRelativePath: "", relativePath: "./a\\b//x.bin" }), "a/b/x.bin");
  const big = recognizeFiles([sized("r/weights.json", 10), sized("r/weights.f32.bin", 2000)], { limits: { maxBinaryBytes: 1000 } });
  assert.match(big.runs[0].errors[0], /r\/weights.f32.bin: 0,0 MB, il limite/);
  throwsFormat(() => recognizeFiles([sized("r/weights.json", 10), sized("r/weights.f32.bin", 2000)], { limits: { maxTotalBytes: 1000 } }), /limite per un caricamento/);
});

// ---------------------------------------------------------------- run del corso

test("run del corso: pesi, config, 11 snapshot a richiesta, motore compatibile", { skip: !HAS_TRACE }, async () => {
  const run = await loadBuiltinRun(fsLoaders);
  assert.equal(run.name, "GPT del corso");
  assert.equal(run.source, "builtin");
  assert.equal(run.totalParams, 209729);
  assert.equal(run.tensors.size, 94);
  assert.deepEqual(run.config, { vocab_size: 65, block_size: 32, n_embd: 64, n_head: 4, n_layer: 4, dropout: 0 });
  assert.equal(run.compatibleWithEngine, true, run.engine.issues.join("; "));
  assert.deepEqual(run.steps, [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]);
  assert.match(run.description, /209\.729 parametri, pesi finali dopo 5\.000 step, 11 snapshot/);
  assert.deepEqual(run.tensor("blocks.0.ffwd.net.0.weight").shape, [256, 64]);
  assert.equal(run.tensor("_orig_mod.lm_head.bias"), run.tensor("lm_head.bias"));

  // Lo snapshot 5000 sono i pesi finali in float16: differenza entro la precisione di un half.
  const last = await run.loadSnapshot(5000);
  const final = run.flat();
  let worst = 0;
  for (let i = 0; i < final.length; i++) worst = Math.max(worst, Math.abs(last.flat[i] - final[i]) / Math.max(1e-3, Math.abs(final[i])));
  assert.ok(worst < 1e-3, `errore relativo float16 ${worst}`);
  assert.equal(await run.loadSnapshot(5000), last, "in cache: lo stesso oggetto");
  const init = await run.loadSnapshot(0);
  assert.notDeepEqual(init.tensors.get("lm_head.weight").data, run.tensor("lm_head.weight").data);
  await assert.rejects(run.loadSnapshot(123), /snapshot dello step 123 non disponibile/);

  // Step intermedio: media pesata dei due snapshot vicini, dichiarata come non esatta.
  const mid = await run.weightsAt(75);
  const [a, b] = [await run.loadSnapshot(50), await run.loadSnapshot(100)];
  assert.equal(mid.exact, false);
  assert.deepEqual(mid.from, [50, 100]);
  assert.ok(Math.abs(mid.flat[1234] - (a.flat[1234] + b.flat[1234]) / 2) < 1e-6);
  const exact = await run.weightsAt(500);
  assert.equal(exact.exact, true);
  assert.equal(exact.flat, (await run.loadSnapshot(500)).flat);

  // Il motore creato dal run è quello di loadEngine (anche con gli snapshot del motore).
  const engine = run.createEngine();
  const reference = await loadEngine(DATA_DIR, { loadJson, loadBuffer });
  const ids = reference.encode("ROMEO:\nBut, soft!");
  assert.deepEqual(engine.forward(ids).logits, reference.forward(ids).logits);
  await engine.loadSnapshot(500);
  engine.setWeights(run.engineFlat((await run.loadSnapshot(1000)).tensors));
  assert.equal(engine.weightsSource, "custom");
});

test("snapshotBracket: step esatti, intermedi e fuori intervallo", () => {
  const steps = [0, 50, 100];
  assert.deepEqual(snapshotBracket(steps, 75), { lo: 50, hi: 100, t: 0.5, clamped: false });
  assert.deepEqual(snapshotBracket(steps, 50), { lo: 50, hi: 100, t: 0, clamped: false });
  assert.deepEqual(snapshotBracket(steps, 100), { lo: 100, hi: 100, t: 0, clamped: false });
  assert.deepEqual(snapshotBracket(steps, 500), { lo: 100, hi: 100, t: 0, clamped: true });
  assert.deepEqual(snapshotBracket(steps, -3), { lo: 0, hi: 0, t: 0, clamped: true });
  assert.equal(snapshotBracket([], 5), null);
});

// ---------------------------------------------------------------- run caricati

test("run caricato dalla cartella data/ (file veri): uguale al run del corso, snapshot dai file", { skip: !HAS_TRACE }, async () => {
  const read = async (rel) => fileLike(`data/${rel}`, new Uint8Array(await readFile(`${DATA_DIR}${rel}`)));
  const files = await Promise.all(["weights.json", "weights.f32.bin", "training/trace.json", "training/snap_00000.f16.bin", "training/snap_05000.f16.bin"].map(read));
  const { runs, errors } = await runsFromFiles(files);
  assert.deepEqual(errors, []);
  const [run] = runs;
  const builtin = await loadBuiltinRun(fsLoaders);
  assert.equal(run.source, "upload");
  assert.equal(run.name, "data");
  assert.ok(run.id.startsWith("upload:"));
  assert.equal(run.compatibleWithEngine, true);
  for (const [name, t] of builtin.tensors) assert.deepEqual(run.tensor(name).data, t.data, name);
  assert.deepEqual(run.steps, [0, 5000], "solo gli snapshot caricati");
  assert.equal(run.snapshots.length, 11, "le metriche di tutti gli snapshot restano");
  assert.match(run.meta.warnings.join(" "), /cita 9 snapshot senza file/);
  assert.deepEqual((await run.loadSnapshot(0)).flat, (await builtin.loadSnapshot(0)).flat);
  // Caricato due volte: stesso id (nel browser non si duplica).
  const again = await runsFromFiles(files);
  assert.equal(again.runs[0].id, run.id);
});

test("run caricati: architettura modificata, altra config, vocabolario, prefissi di torch.compile", async () => {
  // 1. Altra config (più piccola): il motore la esegue.
  const small = synthetic();
  let { runs } = await runsFromFiles([fileLike("piccolo/weights.json", JSON.stringify(small.manifest)), fileLike("piccolo/weights.f32.bin", small.bytes)]);
  assert.equal(runs[0].compatibleWithEngine, true, runs[0].engine.issues.join("; "));
  const engine = runs[0].createEngine();
  const direct = GPTEngine.fromBuffers(small.manifest, small.values.buffer);
  assert.deepEqual(engine.forward([2, 3, 4]).logits, direct.forward([2, 3, 4]).logits);

  // 2. Un bias in più sulla key (codice modificato): si carica ma il motore non lo esegue.
  const modified = synthetic(SMALL, { extra: [{ name: "blocks.0.sa.heads.0.key.bias", shape: [4] }] });
  ({ runs } = await runsFromFiles([fileLike("mod/weights.json", JSON.stringify(modified.manifest)), fileLike("mod/weights.f32.bin", modified.bytes), fileLike("mod/run.json", JSON.stringify({ name: "GPT con bias sulle key" }))]));
  assert.equal(runs[0].name, "GPT con bias sulle key");
  assert.equal(runs[0].compatibleWithEngine, false);
  assert.match(runs[0].engine.issues.join(" "), /architettura modificata\? \(blocks\.0\.sa\.heads\.0\.key\.bias\)/);
  assert.throws(() => runs[0].createEngine(), /il motore JS non può eseguire questo run/);

  // 3. run.json può dichiarare che il calcolo è diverso anche con le stesse forme.
  ({ runs } = await runsFromFiles([fileLike("gelu/weights.json", JSON.stringify(small.manifest)), fileLike("gelu/weights.f32.bin", small.bytes), fileLike("gelu/run.json", JSON.stringify({ engine_compatible: false }))]));
  assert.equal(runs[0].compatibleWithEngine, false);
  assert.match(runs[0].engine.issues[0], /engine_compatible: false/);

  // 4. Safetensors senza vocabolario, salvato da un modello compilato (prefisso _orig_mod.).
  const names = expectedParameters(SMALL);
  let at = 0;
  const header = {};
  const parts = [];
  for (const { name, shape } of names) {
    const numel = shape.reduce((a, b) => a * b, 1);
    header[`_orig_mod.${name}`] = { dtype: "F32", shape, data_offsets: [at, at + numel * 4] };
    parts.push(new Uint8Array(small.values.buffer, small.manifest.params.find((p) => p.name === name).offset * 4, numel * 4));
    at += numel * 4;
  }
  const file = fileLike("st/model.safetensors", safetensors(header, concat(...parts)));
  ({ runs } = await runsFromFiles([file]));
  assert.equal(runs[0].compatibleWithEngine, false, "senza vocabolario il motore non sa trasformare il testo");
  assert.match(runs[0].engine.issues.join(" "), /manca il vocabolario/);
  assert.ok(runs[0].tensors.has("blocks.1.ffwd.net.2.bias"), "prefisso tolto");
  assert.match(runs[0].meta.warnings.join(" "), /tolti i prefissi _orig_mod\./);
  assert.ok(naturalCompare([...runs[0].tensors.keys()][2], [...runs[0].tensors.keys()][3]) < 0, "ordine naturale");
  ({ runs } = await runsFromFiles([file], { defaultVocab: SMALL_VOCAB }));
  assert.equal(runs[0].compatibleWithEngine, true);
  assert.match(runs[0].meta.warnings.join(" "), /non contiene il vocabolario: si usa quello del GPT del corso/);
  assert.deepEqual(runs[0].createEngine().forward([1, 2]).logits, direct.forward([1, 2]).logits, "riordinati nel layout del motore");

  // 5. Un file rotto produce un errore per il suo run, gli altri si caricano.
  const result = await runsFromFiles([fileLike("ok/weights.json", JSON.stringify(small.manifest)), fileLike("ok/weights.f32.bin", small.bytes), fileLike("rotto/weights.json", "{non json"), fileLike("senzabin/weights.json", JSON.stringify(small.manifest))]);
  assert.equal(result.runs.length, 1);
  assert.deepEqual(result.errors.map((e) => e.run), ["rotto", "senzabin"]);
  assert.match(result.errors[0].message, /non è un JSON valido/);
  assert.match(result.errors[1].message, /manca il file dei pesi weights.f32.bin/);
});

test("run dalla sola storia del training: i pesi sono l'ultimo snapshot", async () => {
  const { manifest, values } = synthetic();
  const half = new Uint16Array(values.length).fill(0x3c00); // tutti 1.0 in float16
  const trace = { format_version: 1, snapshot_dtype: "float16", byte_order: "little", config: SMALL, vocab: SMALL_VOCAB, params: manifest.params, total_params: manifest.total_params, snapshots: [{ step: 0, file: "snap_00000.f16.bin" }, { step: 10, file: "snap_00010.f16.bin" }] };
  const { runs } = await runsFromFiles([fileLike("storia/trace.json", JSON.stringify(trace)), fileLike("storia/snap_00010.f16.bin", new Uint8Array(half.buffer))]);
  assert.equal(runs.length, 1);
  assert.equal(runs[0].tensor("lm_head.bias").data[0], 1);
  assert.equal(runs[0].compatibleWithEngine, true);
  assert.match(runs[0].meta.warnings.join(" "), /ultimo snapshot del training \(step 10, float16\)/);
});

test("engineCompatibility e inferGptConfig: i motivi sono detti", () => {
  const tensorsOf = (manifest) => new Map(manifest.params.map((p) => [p.name, { shape: p.shape }]));
  const ok = engineCompatibility(tensorsOf(synthetic().manifest), { vocab: SMALL_VOCAB, config: SMALL });
  assert.equal(ok.compatible, true);
  assert.deepEqual(ok.config, SMALL);
  const wrongConfig = engineCompatibility(tensorsOf(synthetic().manifest), { vocab: SMALL_VOCAB, config: { ...SMALL, n_layer: 3 } });
  assert.match(wrongConfig.issues[0], /config.n_layer è 3 ma le forme dei pesi dicono 2/);
  const issues = [];
  assert.equal(inferGptConfig(new Map([["wte.weight", { shape: [5, 4] }]]), issues), null);
  assert.match(issues[0], /token_embedding_table/);
  const gaps = new Map([...tensorsOf(synthetic().manifest)].filter(([n]) => !n.startsWith("blocks.0.")));
  const result = engineCompatibility(gaps, { vocab: SMALL_VOCAB });
  assert.equal(result.compatible, false);
  assert.match(result.issues[0], /numerati da 0 senza salti/);
  // n_embd 18 non è divisibile per 4 head: lo dice il motore stesso.
  const odd = synthetic({ ...SMALL, n_embd: 16 }).manifest.params.map((p) => [p.name, { shape: p.name === "token_embedding_table.weight" ? [11, 18] : p.name === "position_embedding_table.weight" ? [8, 18] : p.shape }]);
  assert.match(engineCompatibility(new Map(odd), { vocab: SMALL_VOCAB }).issues.join(" "), /divisibile/);
});

test("fingerprint: stabile, sensibile ai valori e ai nomi", () => {
  const a = new Float32Array([1, 2, 3]);
  assert.equal(fingerprint(a, ["x"]), fingerprint(new Float32Array([1, 2, 3]), ["x"]));
  assert.notEqual(fingerprint(a, ["x"]), fingerprint(new Float32Array([1, 2, 3.0001]), ["x"]));
  assert.notEqual(fingerprint(a, ["x"]), fingerprint(a, ["y"]));
});

// ---------------------------------------------------------------- statistiche

test("tensorStats: valori noti, percentili come numpy, NaN e infiniti contati a parte", () => {
  const s = tensorStats([1, 2, 3, 4]);
  assert.equal(s.mean, 2.5);
  assert.ok(Math.abs(s.std - Math.sqrt(1.25)) < 1e-12, "deviazione standard di popolazione");
  assert.equal(s.min, 1);
  assert.equal(s.max, 4);
  assert.ok(Math.abs(s.l2 - Math.sqrt(30)) < 1e-12);
  assert.deepEqual([s.percentiles[25], s.percentiles[50], s.percentiles[75]], [1.75, 2.5, 3.25]); // numpy.percentile
  assert.equal(s.histogram.counts.reduce((a, b) => a + b, 0), 4);
  const dirty = tensorStats(new Float32Array([NaN, 0.0005, -0.5, Infinity, 0.5, -Infinity]));
  assert.deepEqual([dirty.count, dirty.finite, dirty.nan, dirty.inf], [6, 3, 1, 2]);
  assert.equal(dirty.nearZero.count, 1);
  assert.ok(Math.abs(dirty.nearZero.fraction - 1 / 3) < 1e-12);
  assert.equal(dirty.absMax, 0.5);
  const empty = tensorStats([NaN]);
  assert.ok(Number.isNaN(empty.mean));
  assert.equal(empty.histogram, null);
  assert.deepEqual(percentiles([5, 1, 3], [0, 100]), { 0: 1, 100: 5 });
});

test("histogram e range: intervallo comune per sovrapporre due tensori", () => {
  const h = histogram([0, 0.5, 1, 1, 2], { bins: 4, min: 0, max: 1 });
  assert.deepEqual(Array.from(h.counts), [1, 0, 1, 3]);
  assert.equal(h.above, 1);
  assert.deepEqual(range([[1, -2], new Float32Array([5, NaN]), [Infinity]]), { min: -2, max: 5 });
  const flat = histogram([3, 3, 3], { bins: 5 });
  assert.equal(flat.counts[0], 3, "tutti uguali: un intervallo solo");
});

test("tensorStats sui pesi veri: std come gpt/recording.py (weight_std_by_group allo step 0)", { skip: !HAS_TRACE }, async () => {
  const run = await loadBuiltinRun(fsLoaders);
  const init = await run.loadSnapshot(0);
  const metrics = run.trace.metrics;
  assert.equal(metrics.steps[0], 0);
  const groupStd = (group) => {
    const names = run.trace.group_params[group];
    const all = new Float32Array(names.reduce((s, n) => s + init.tensors.get(n).numel, 0));
    let at = 0;
    for (const n of names) {
      all.set(init.tensors.get(n).data, at);
      at += init.tensors.get(n).numel;
    }
    return tensorStats(all, { histogram: false, percentiles: [] }).std;
  };
  // LayerNorm appena creata: γ = 1 e β = 0 → std 0,5 esatta (anche in float16).
  assert.equal(groupStd("blocks.0.layernorm"), metrics.weight_std_by_group["blocks.0.layernorm"][0]);
  // Embedding inizializzato con N(0, 1): coincide a meno dell'arrotondamento float16 dello snapshot.
  assert.ok(Math.abs(groupStd("token_embedding") - metrics.weight_std_by_group.token_embedding[0]) < 2e-3);
});

// ---------------------------------------------------------------- confronto

test("compareTensors: identici, scalati, opposti; nomi canonici; forme diverse riportate", () => {
  const map = (entries) => new Map(entries.map(([name, values, shape = [values.length]]) => [name, { name, shape, data: Float32Array.from(values) }]));
  const A = map([["x", [1, 2, 2]], ["y", [1, -1]], ["solo_a", [3]], ["z", [1, 2, 3, 4], [2, 2]]]);
  assert.equal(compareTensors(A, A).identical, true);
  const B = map([["_orig_mod.x", [2, 4, 4]], ["y", [-1, 1]], ["solo_b", [3]], ["z", [1, 2, 3, 4], [4]]]);
  const c = compareTensors(A, B, { groupOf: () => "g" });
  const x = c.matched.find((m) => m.name === "x");
  assert.equal(x.nameB, "_orig_mod.x");
  assert.equal(x.relChange, 1);
  assert.equal(x.cosine, 1);
  assert.equal(x.maxAbsDiff, 2);
  const y = c.matched.find((m) => m.name === "y");
  assert.ok(Math.abs(y.cosine + 1) < 1e-12);
  assert.equal(y.relChange, 2);
  assert.deepEqual(c.onlyA.map((t) => t.name), ["solo_a"]);
  assert.deepEqual(c.onlyB.map((t) => t.name), ["solo_b"]);
  assert.deepEqual(c.shapeMismatch, [{ name: "z", nameB: "z", shapeA: [2, 2], shapeB: [4] }]);
  assert.deepEqual(c.mostChanged.map((m) => m.name), ["y", "x"]);
  assert.equal(c.groups.length, 1);
  assert.equal(c.groups[0].params, 5);
  // Gruppo = tutti i pesi in fila: ‖A − B‖² = 9 + 8, ‖A‖² = 9 + 2.
  assert.ok(Math.abs(c.totals.relChange - Math.sqrt(17 / 11)) < 1e-12);
  assert.equal(c.identical, false);
  assert.deepEqual(Array.from(difference(new Float32Array([3, 1]), new Float32Array([1, 1]))), [2, 0]);
  assert.equal(tensorDiff(new Float32Array([0, 0]), new Float32Array([0, 0])).cosine, null, "coseno indefinito con un vettore nullo");
  assert.equal(tensorDiff(new Float32Array([1, NaN]), new Float32Array([1, 2])).skipped, 1);
});

test("compareTensors sul training vero: dall'inizializzazione ai pesi finali cambia tutto", { skip: !HAS_TRACE }, async () => {
  const run = await loadBuiltinRun(fsLoaders);
  const c = compareTensors(await run.loadSnapshot(0), run, { top: 5, trace: run.trace });
  assert.equal(c.matched.length, 94);
  assert.equal(c.onlyA.length + c.onlyB.length + c.shapeMismatch.length, 0);
  assert.ok(c.matched.every((m) => m.relChange > 0));
  assert.deepEqual(c.groups.map((g) => g.group), run.trace.groups, "gli stessi gruppi di gpt/recording.py, nello stesso ordine");
  for (let i = 1; i < c.mostChanged.length; i++) assert.ok(c.mostChanged[i - 1].relChange >= c.mostChanged[i].relChange);
  assert.equal(c.totals.params, 209729);
});

test("groupOf: le regole di gpt/recording.py per tutti i 94 parametri, e un ripiego per gli altri", { skip: !HAS_TRACE }, async () => {
  const trace = await loadJson(`${DATA_DIR}training/trace.json`);
  const fromTrace = groupsFromTrace(trace);
  assert.equal(fromTrace.size, 94);
  for (const [name, group] of fromTrace) assert.equal(groupOf(name), group, name);
  assert.equal(groupOf("transformer.h.3.attn.c_attn.weight"), "transformer.h.3.attn");
  assert.equal(groupOf("model.layers.0.mlp.up_proj.weight"), "model.layers.0.mlp");
  assert.equal(groupOf("transformer.wte.weight"), "transformer.wte");
  assert.equal(groupOf("constructor.weight"), "constructor", "niente proprietà ereditate di Object");
});

// ---------------------------------------------------------------- albero dei moduli

test("albero dei moduli del corso: struttura del codice, conteggi e quote", { skip: !HAS_DATA }, async () => {
  const spec = await loadJson(`${DATA_DIR}model.json`);
  const manifest = await loadJson(`${DATA_DIR}weights.json`);
  const tree = buildModuleTree(manifest.params, { snippets: spec.snippets, config: manifest.config });
  assert.equal(tree.label, "GPTLanguageModel");
  assert.equal(tree.architecture, "corso");
  assert.deepEqual(tree.unexplained, []);
  assert.deepEqual(tree.mismatched, []);
  assert.equal(tree.params, 209729);
  assert.equal(tree.tensors, 94);
  assert.deepEqual(tree.children.map((c) => c.key), ["token_embedding_table", "position_embedding_table", "blocks", "ln_f", "lm_head"]);
  const block = findNode(tree, "blocks.0");
  assert.equal(block.label, "[0]");
  assert.equal(block.pyPath, "blocks[0]");
  assert.equal(block.code.className, "Block");
  assert.equal(block.code.createdBy.text, "self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])");
  assert.equal(block.params, 49792);
  assert.ok(Math.abs(findNode(tree, "blocks").share - 199168 / 209729) < 1e-12);
  const key = findNode(tree, "blocks.0.sa.heads.0.key.weight");
  assert.equal(key.kind, "param");
  assert.deepEqual(key.shape, [16, 64]);
  assert.equal(key.pyPath, "blocks[0].sa.heads[0].key.weight");
  assert.deepEqual(key.code, {
    snippet: "head",
    file: "gpt/attention.py",
    line: 29,
    text: "self.key = nn.Linear(config.n_embd, self.head_size, bias=False)",
    comment: null,
    type: "nn.Linear",
    param: "weight",
    role: "matrice W (uscite × ingressi): y = x · Wᵀ + b",
  });
  assert.equal(key.codeMatch, true);
  assert.equal(key.shapeMatch, true);
  const head = findNode(tree, "blocks.0.sa.heads.0");
  assert.equal(head.code.buffers[0].name, "tril", "la maschera causale è un buffer, non un peso");
  assert.match(head.code.buffers[0].statement, /torch\.tril\(torch\.ones\(config\.block_size, config\.block_size\)\), persistent=False\)/);
  const net = findNode(tree, "blocks.0.ffwd.net");
  assert.deepEqual(net.code.items.map((i) => i.type), ["nn.Linear", "nn.ReLU", "nn.Linear", "nn.Dropout"]);
  assert.deepEqual(net.children.map((c) => c.label), ["[0]", "[2]"], "ReLU e Dropout non hanno pesi");
  assert.equal(findNode(tree, "blocks.0.ffwd.net.2.weight").code.text, "nn.Linear(hidden, config.n_embd),  # pesi (C, 4C) + bias (C)");
  assert.equal(findNode(tree, "lm_head.bias").code.role, "bias b: un numero per ogni uscita");
  assert.equal(findNode(tree, "nessuno"), null);
});

test("albero dei moduli: ogni riga annotata è davvero quella riga del file gpt/*.py", { skip: !HAS_DATA }, async () => {
  const spec = await loadJson(`${DATA_DIR}model.json`);
  const manifest = await loadJson(`${DATA_DIR}weights.json`);
  const tree = buildModuleTree(manifest.params, { snippets: spec.snippets, config: manifest.config });
  const files = new Map();
  let checked = 0;
  const check = (code) => {
    if (!files.has(code.file)) files.set(code.file, null);
    return code;
  };
  walkTree(tree, (n) => {
    if (n.code) check(n.code);
    if (n.code?.createdBy) check(n.code.createdBy);
  });
  for (const file of files.keys()) files.set(file, (await readFile(`${ROOT}${file}`, "utf8")).split("\n"));
  walkTree(tree, (n) => {
    for (const code of [n.code, n.code?.createdBy, ...(n.code?.items ?? []), ...(n.code?.buffers ?? [])].filter(Boolean)) {
      if (!code.line || !code.text) continue;
      const lines = files.get(code.file ?? n.code.file);
      assert.equal(lines[code.line - 1].trim(), code.text, `${n.path}: ${code.file ?? n.code.file}:${code.line}`);
      checked++;
    }
  });
  assert.ok(checked > 200, `righe controllate: ${checked}`);
});

test("albero dei moduli: architettura modificata o sconosciuta", { skip: !HAS_DATA }, async () => {
  const spec = await loadJson(`${DATA_DIR}model.json`);
  const params = [...expectedParameters(SMALL), { name: "blocks.0.sa.heads.0.key.bias", shape: [4] }, { name: "blocks.1.sa.rope.inv_freq", shape: [2] }];
  const tree = buildModuleTree(params, { snippets: spec.snippets, config: SMALL });
  assert.equal(tree.architecture, "modificata");
  assert.equal(findNode(tree, "blocks.0.sa.heads.0.key.bias").codeMatch, false, "il codice dice bias=False");
  assert.ok(tree.mismatched.includes("blocks.0.sa.heads.0.key.bias"));
  assert.deepEqual(tree.unexplained, ["blocks.1.sa.rope.inv_freq"]);
  const bare = buildModuleTree([{ name: "wte.weight", shape: [5, 4] }, { name: "h.0.attn.weight", shape: [4, 4] }]);
  assert.equal(bare.architecture, "sconosciuta");
  assert.equal(bare.label, "modello");
  assert.equal(findNode(bare, "h.0").label, "[0]");
  assert.equal(bare.params, 36);
  assert.throws(() => buildModuleTree([{ name: "a", shape: [1] }, { name: "a.b", shape: [1] }]), /in conflitto/);
  assert.equal(pythonPath("blocks.10.sa.heads.2.key.weight"), "blocks[10].sa.heads[2].key.weight");
});

test("lettore del codice Python: commenti, istruzioni su più righe, Sequential e comprensioni", () => {
  assert.deepEqual(splitComment('x = f("#non è un commento")  # sì'), { code: 'x = f("#non è un commento")', comment: "sì" });
  const source = parseModuleSource("demo", {
    file: "demo.py",
    start_line: 10,
    code: [
      "class Demo(nn.Module):",
      "    def __init__(self, c):",
      "        super().__init__()",
      "        self.a = nn.Linear(c.n, c.m, bias=False)  # pesi (m, n)",
      "        self.b = nn.Sequential(",
      "            nn.Linear(c.n, 4 * c.n),",
      "            nn.GELU(),",
      "        )",
      "        self.c = nn.ModuleList([Demo(c) for _ in range(3)])",
      '        self.register_buffer("mask", torch.ones(3))',
      "",
      "    def forward(self, x):",
      "        self.z = nn.Linear(1, 1)",
    ].join("\n"),
  });
  assert.equal(source.className, "Demo");
  assert.deepEqual([...source.attrs.keys()], ["a", "b", "c"], "solo dentro __init__");
  assert.equal(source.attrs.get("a").line, 13);
  assert.equal(source.attrs.get("a").comment, "pesi (m, n)");
  assert.equal(source.attrs.get("a").args, "c.n, c.m, bias=False");
  assert.deepEqual(source.attrs.get("b").items.map((i) => [i.callee, i.line]), [["nn.Linear", 15], ["nn.GELU", 16]]);
  assert.equal(source.attrs.get("c").element, "Demo");
  assert.equal(source.buffers[0].name, "mask");
});

// ---------------------------------------------------------------- salvataggio nel browser

test("store: un run caricato si serializza e torna identico (snapshot compresi)", async () => {
  const { manifest, bytes, values } = synthetic();
  const half = new Uint16Array(values.length).fill(0x3800); // 0.5
  const trace = { format_version: 1, snapshot_dtype: "float16", params: manifest.params, total_params: manifest.total_params, train_loss: [3, 2], snapshots: [{ step: 0, file: "snap_00000.f16.bin", loss: 3 }] };
  const { runs } = await runsFromFiles([
    fileLike("r/weights.json", JSON.stringify(manifest)),
    fileLike("r/weights.f32.bin", bytes),
    fileLike("r/trace.json", JSON.stringify(trace)),
    fileLike("r/snap_00000.f16.bin", new Uint8Array(half.buffer)),
    fileLike("r/benchmark.json", JSON.stringify({ val_loss: 2.5 })),
  ]);
  const run = runs[0];
  const { record, blobs } = await serializeRun(run, { now: 1234 });
  assert.equal(record.savedAt, 1234);
  assert.equal(blobs.length, 2);
  assert.equal(record.bytes, values.length * 4 + values.length * 2);
  const clone = structuredClone(record); // come lo salva IndexedDB
  const stored = new Map(blobs.map((b) => [b.key, b.data]));
  const back = deserializeRun(clone, stored.get(`${run.id}|weights`), async (key) => stored.get(key));
  assert.equal(back.id, run.id);
  assert.equal(back.compatibleWithEngine, true);
  for (const [name, t] of run.tensors) assert.deepEqual(back.tensor(name).data, t.data);
  assert.equal((await back.loadSnapshot(0)).flat[7], 0.5);
  assert.equal(back.benchmark.metrics.val_loss, 2.5);
  assert.deepEqual(back.trace.train_loss, [3, 2]);
});

test("store: il run del corso non si salva; senza IndexedDB tutto risponde senza eccezioni", { skip: !HAS_DATA }, async () => {
  const builtin = await loadBuiltinRun({ ...fsLoaders, trace: false });
  await assert.rejects(serializeRun(builtin), /già nei dati della web app/);
  const store = new RunStore({ indexedDB: null });
  assert.equal(store.available, false);
  assert.deepEqual(await store.list(), []);
  assert.equal(await store.load("x"), null);
  assert.equal(await store.remove("x"), false);
  const saved = await store.save(builtin);
  assert.equal(saved.ok, false);
  const { runs } = await runsFromFiles([fileLike("w/weights.json", JSON.stringify(synthetic().manifest)), fileLike("w/weights.f32.bin", synthetic().bytes)]);
  const result = await store.save(runs[0]);
  assert.deepEqual(result, { ok: false, reason: "il browser non ha salvato il run (IndexedDB non disponibile in questo browser)" });
  assert.equal(await store.estimate(), null);
});

// ---------------------------------------------------------------- formato dei numeri nelle celle

test("formatCell e formatPrecise: virgola, segno meno, infiniti, notazione esponenziale", () => {
  assert.equal(formatCell(0.1234, 2), "0,12");
  assert.equal(formatCell(-0.1234, 3), "−0,123");
  assert.equal(formatCell(-0.0004, 2), "−0,00", "il segno resta: era negativo");
  assert.equal(formatCell(-0, 2), "0,00");
  assert.equal(formatCell(3.14159, 4), "3,1416");
  assert.equal(formatCell(123456, 2), "1,2e5");
  assert.equal(formatCell(NaN, 2), "NaN");
  assert.equal(formatCell(-Infinity, 2), "−∞");
  assert.equal(formatPrecise(-0.012345678), "−0,01234568");
  assert.equal(formatPrecise(1.5), "1,5");
  assert.equal(formatPrecise(3e-7), "3e-7");
  assert.equal(formatPrecise(0), "0");
});

test("chooseCellFormat: niente numeri nelle celle piccole, poi più decimali man mano che crescono", () => {
  assert.equal(chooseCellFormat({ cellW: 20, cellH: 20, maxAbs: 0.5 }), null);
  assert.equal(chooseCellFormat({ cellW: 34, cellH: 34, maxAbs: 0.5 }), null);
  assert.equal(chooseCellFormat({ cellW: 40, cellH: 40, maxAbs: 0.5 }).decimals, 2);
  assert.equal(chooseCellFormat({ cellW: 60, cellH: 60, maxAbs: 0.5 }).decimals, 3);
  assert.equal(chooseCellFormat({ cellW: 110, cellH: 110, maxAbs: 0.5 }).decimals, 6);
  assert.equal(chooseCellFormat({ cellW: 110, cellH: 110, maxAbs: 0.5 }).showIndex, true);
  let last = 0;
  for (let s = 36; s <= 300; s += 2) {
    const f = chooseCellFormat({ cellW: s, cellH: s, maxAbs: 3.7 });
    if (!f) continue;
    assert.ok(f.decimals >= last, `decimali non decrescenti (${s} px)`);
    assert.ok(f.fontPx >= 10, `corpo leggibile (${s} px)`);
    last = f.decimals;
  }
  // Celle basse e larghe (le barre dei vettori del Transformer): limitate dall'altezza.
  const bar = chooseCellFormat({ cellW: 120, cellH: 30, maxAbs: 1 });
  assert.ok(bar.fontPx <= 30 * 0.42 + 1e-9);
  // Valori grandi: servono più caratteri, quindi meno decimali a parità di cella.
  assert.ok(chooseCellFormat({ cellW: 70, cellH: 70, maxAbs: 250 }).decimals < chooseCellFormat({ cellW: 70, cellH: 70, maxAbs: 0.5 }).decimals);
});

test("contrasto: nero o bianco con almeno 4,5:1 su qualunque colore", () => {
  assert.equal(textColorFor(22, 27, 34), "#ffffff", "cella quasi nera (peso ≈ 0): testo bianco");
  assert.equal(textColorFor(251, 146, 60), "#000000", "arancione pieno: testo nero");
  let worst = Infinity;
  for (let r = 0; r < 256; r += 15) for (let g = 0; g < 256; g += 15) for (let b = 0; b < 256; b += 15) worst = Math.min(worst, textContrastFor(r, g, b));
  assert.ok(worst >= 4.5, `contrasto minimo ${worst}`);
  assert.ok(Math.abs(contrastRatio(relativeLuminance(0, 0, 0), relativeLuminance(255, 255, 255)) - 21) < 1e-9);
});

test("finestre di celle, forme e quantizzazione", () => {
  const rect = { minX: -1, maxX: 1, minY: -2, maxY: 2 }; // 4 righe × 2 colonne da 1 unità
  assert.deepEqual(windowFromPoints([{ x: -0.5, y: 1.5 }, { x: 0.2, y: 0.5 }], rect, 4, 2), { r0: 0, r1: 2, c0: 0, c1: 2 });
  assert.equal(windowFromPoints([{ x: 5, y: 5 }, { x: 6, y: 6 }], rect, 4, 2), null);
  const big = { r0: 0, r1: 256, c0: 0, c1: 64 };
  const capped = capWindow(big, { row: 200, col: 60 }, 2500);
  assert.ok(windowCells(capped) <= 2500);
  assert.ok(windowContains(big, capped));
  assert.ok(capped.r0 <= 200 && capped.r1 > 200 && capped.c0 <= 60 && capped.c1 > 60, "la cella al centro resta dentro");
  const row = capWindow({ r0: 0, r1: 1, c0: 0, c1: 256 }, { row: 0, col: 250 }, 100);
  assert.deepEqual(row, { r0: 0, r1: 1, c0: 156, c1: 256 }, "una riga sola usa tutto il tetto");
  assert.deepEqual(expandWindow({ r0: 10, r1: 20, c0: 0, c1: 4 }, 0.25, 2, 30, 5), { r0: 7, r1: 23, c0: 0, c1: 5 });
  assert.deepEqual(matrixShape([256]), [1, 256]);
  assert.deepEqual(matrixShape([65, 64]), [65, 64]);
  assert.deepEqual(matrixShape([2, 3, 4]), [6, 4]);
  assert.deepEqual(matrixShape([]), [1, 1]);
  assert.deepEqual([quantizeSigned(0, 0.5), quantizeSigned(0.5, 0.5), quantizeSigned(-0.5, 0.5), quantizeSigned(9, 0.5), quantizeSigned(NaN, 1)], [128, 255, 1, 255, 128]);
  assert.equal(canonicalName("module._orig_mod.blocks.0.ln1.weight"), "blocks.0.ln1.weight");
});

// ---------------------------------------------------------------- NumericMatrix e attachNumbers (three.js)

// three.js sta in webapp/vendor (nel browser lo trova l'importmap): qui lo trova un hook di risoluzione.
const VENDOR = pathToFileURL(fileURLToPath(new URL("../vendor/three/", import.meta.url))).href;
register(
  "data:text/javascript," +
    encodeURIComponent(`export async function resolve(specifier, context, next) {
      if (specifier === "three") return { url: ${JSON.stringify(`${VENDOR}three.module.min.js`)}, shortCircuit: true };
      if (specifier.startsWith("three/addons/")) return { url: ${JSON.stringify(`${VENDOR}addons/`)} + specifier.slice(13), shortCircuit: true };
      return next(specifier, context);
    }`),
);
const THREE = await import("three");
const { NumericMatrix, NumbersGroup, attachNumbers, colormapLUT } = await import("../js/core/matrixNumbers.js");
const { CELL, matrixPlane, weightTexture } = await import("../js/core/visuals.js");

/** Canvas finto: registra i testi scritti (con colore e font) a ogni ridisegno. */
function fakeCanvas() {
  const canvas = { width: 0, height: 0, texts: [], draws: 0 };
  const ctx = {
    font: "10px monospace", fillStyle: "#000", strokeStyle: "", lineWidth: 1, textAlign: "", textBaseline: "", globalAlpha: 1,
    clearRect() {
      canvas.texts = [];
      canvas.draws++;
    },
    fillRect() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {},
    fillText(text, x, y) {
      canvas.texts.push({ text, x, y, color: this.fillStyle, font: this.font, alpha: this.globalAlpha });
    },
    measureText(text) {
      return { width: text.length * parseFloat(/([\d.]+)px/.exec(this.font)[1]) * 0.6 };
    },
  };
  canvas.getContext = () => ctx;
  return canvas;
}

const W = 1600;
const H = 900;
function scene3d({ maxOverlays = 4, maxCellsPerOverlay = 2500 } = {}) {
  const canvases = [];
  const group = new NumbersGroup({ maxOverlays, maxCellsPerOverlay, pixelRatio: 1, createCanvas: () => {
    const c = fakeCanvas();
    canvases.push(c);
    return c;
  } });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.05, 2000);
  const look = (x, y, z, target = [x, y, 0]) => {
    camera.position.set(x, y, z);
    camera.lookAt(...target);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    scene.updateMatrixWorld(true);
  };
  const centerHit = (objects) => {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2(0, 0), camera);
    return ray.intersectObjects(objects, false)[0];
  };
  return { group, scene, camera, look, canvases, centerHit };
}
const ffwd = Float32Array.from({ length: 256 * 64 }, (_, i) => Math.sin(i * 0.37) * 0.3 + (i % 7 === 0 ? 0.25 : 0));
const unpack = (text) => Number(text.replace("−", "-").replace(",", "."));

test("NumericMatrix 256×64: geometria come core/visuals.js, riga 0 in alto, cella sotto il mouse", () => {
  const { group, scene, look, centerHit } = scene3d();
  const m = new NumericMatrix({ rows: 256, cols: 64, data: ffwd, group });
  scene.add(m.object3d);
  const size = new THREE.Box3().setFromObject(m.mesh).getSize(new THREE.Vector3());
  assert.ok(Math.abs(size.x - 64 * CELL) < 1e-6 && Math.abs(size.y - 256 * CELL) < 1e-6, "le posizioni sono float32");
  assert.deepEqual([m.texture.image.width, m.texture.image.height], [64, 256]);
  // Riga 0 della matrice = ultima riga della DataTexture (che parte dal basso), colore della colormap.
  const lut = colormapLUT("diverging");
  const q = quantizeSigned(ffwd[0], m.range);
  const texel = Array.from(m.texture.image.data.slice((255 * 64) * 4, (255 * 64) * 4 + 3));
  assert.deepEqual(texel, Array.from(lut.slice(4 * q, 4 * q + 3)));
  // La cella al centro dello schermo.
  look(-0.5, 2.3, 3);
  const cell = m.cellAt(centerHit([m.mesh]));
  const expectedRow = Math.floor((256 * CELL / 2 - 2.3) / CELL);
  const expectedCol = Math.floor((-0.5 + 64 * CELL / 2) / CELL);
  assert.deepEqual([cell.row, cell.col, cell.value], [expectedRow, expectedCol, ffwd[expectedRow * 64 + expectedCol]]);
  assert.equal(m.cellAt(null), null);
  m.highlight(cell);
  assert.equal(m.target.highlightLine.visible, true);
  m.highlight(null);
  assert.equal(m.target.highlightLine.visible, false);
  m.dispose();
  assert.equal(group.targets.size, 0);
});

test("NumericMatrix: numeri solo da vicino, solo per le celle visibili, a riposo nessun lavoro", () => {
  const { group, scene, camera, look, canvases } = scene3d();
  const m = new NumericMatrix({ rows: 256, cols: 64, data: ffwd, group });
  scene.add(m.object3d);
  look(0.2, 1, 20);
  assert.equal(group.update(camera, H), true);
  assert.equal(m.showingNumbers, false, "da lontano solo i colori");
  look(0.2, 1, 1);
  group.update(camera, H);
  assert.equal(m.showingNumbers, true);
  const [shown] = group.shown();
  const canvas = canvases[0];
  const cells = windowCells(shown.window);
  assert.ok(cells <= 2500 && cells < 256 * 64, `celle disegnate ${cells}`);
  assert.equal(canvas.texts.length, cells, "un numero per cella della finestra");
  assert.equal(shown.decimals, 2);
  // I numeri sono i valori veri delle celle (arrotondati alla precisione scelta).
  const { r0, c0, c1 } = shown.window;
  const nc = c1 - c0;
  for (const k of [0, 17, cells - 1]) {
    const r = r0 + Math.floor(k / nc);
    const c = c0 + (k % nc);
    assert.equal(canvas.texts[k].text, formatCell(ffwd[r * 64 + c], 2));
    assert.ok(Math.abs(unpack(canvas.texts[k].text) - ffwd[r * 64 + c]) <= 0.005 + 1e-9);
  }
  // Ogni testo contrasta con il colore vero della sua cella.
  const lut = colormapLUT("diverging");
  for (let k = 0; k < cells; k += 37) {
    const r = r0 + Math.floor(k / nc);
    const c = c0 + (k % nc);
    const i = 4 * quantizeSigned(ffwd[r * 64 + c], m.range);
    const bg = relativeLuminance(lut[i], lut[i + 1], lut[i + 2]);
    const fg = canvas.texts[k].color === "#ffffff" ? 1 : 0;
    assert.ok(contrastRatio(bg, fg) >= 4.5, `contrasto alla cella ${r},${c}`);
  }
  // A riposo: update esce subito, nessun ridisegno.
  const draws = canvas.draws;
  for (let i = 0; i < 100; i++) assert.equal(group.update(camera, H), false);
  assert.equal(canvas.draws, draws);
  assert.equal(group.stats.idle, 100);
  // Un piccolo spostamento resta nel margine: si ricalcola ma non si ridisegna.
  look(0.22, 1.01, 1);
  assert.equal(group.update(camera, H), true);
  assert.equal(canvas.draws, draws);
  // Più vicino: più decimali, si ridisegna.
  look(0.22, 1.01, 0.6);
  group.update(camera, H);
  assert.equal(group.shown()[0].decimals, 3);
  assert.equal(canvas.draws, draws + 1);
  // Ancora più vicino: celle enormi, tutta la precisione e l'indice riga,colonna.
  look(0.22, 1.01, 0.3);
  group.update(camera, H);
  assert.equal(group.shown()[0].decimals, 6);
  assert.ok(canvas.texts.some((t) => /^\d+,\d+$/.test(t.text) && t.alpha < 1), "indici in piccolo");
  // Di spalle o nascosta: niente numeri.
  look(0.2, 1, -1, [0.2, 1, 0]);
  group.update(camera, H);
  assert.equal(m.showingNumbers, false);
  look(0.2, 1, 1);
  m.object3d.visible = false;
  group.update(camera, H);
  assert.equal(m.showingNumbers, false);
  // Tolta dalla scena (es. una vista ricostruita): non consuma il budget.
  m.object3d.visible = true;
  scene.remove(m.object3d);
  group.invalidate();
  group.update(camera, H);
  assert.equal(m.showingNumbers, false);
});

test("NumbersGroup: le matrici lontane si scartano senza scorrere i loro valori", () => {
  const { group, scene, camera, look } = scene3d();
  let reads = 0;
  const mesh = matrixPlane(weightTexture({ shape: [64, 64], data: Buffer.from(new Uint8Array(4096).fill(140)).toString("base64") }), 64, 64);
  scene.add(mesh);
  const handle = attachNumbers(mesh, { group, getValue: () => (reads++, 0.5) });
  look(0, 0, 30);
  for (let i = 0; i < 5; i++) {
    mesh.material.map.needsUpdate = true; // come durante il "training completo" del Transformer
    group.update(camera, H);
  }
  assert.equal(reads, 0);
  look(0, 0, 0.8);
  group.update(camera, H);
  assert.ok(reads > 0 && handle.showing());
});

test("NumericMatrix: setData (un altro step del training) ridisegna anche a camera ferma", () => {
  const { group, scene, camera, look, canvases } = scene3d();
  const m = new NumericMatrix({ rows: 65, cols: 64, data: ffwd.subarray(0, 65 * 64), group });
  scene.add(m.object3d);
  look(0.1, 0.1, 0.8);
  group.update(camera, H);
  const before = canvases[0].texts[0].text;
  const scale = m.range;
  const scaled = ffwd.subarray(0, 65 * 64).map((v) => v * 0.1 + 0.07);
  m.setData(scaled);
  assert.equal(m.range, scale, "la scala resta quella di prima: si vede che i pesi sono cambiati");
  assert.equal(group.update(camera, H), true);
  const [shown] = group.shown();
  const first = scaled[shown.window.r0 * 64 + shown.window.c0];
  assert.equal(canvases[0].texts[0].text, formatCell(first, shown.decimals));
  assert.notEqual(canvases[0].texts[0].text, before);
  m.setData(scaled, { range: "auto" });
  assert.ok(m.range < scale);
});

test("NumericMatrix: tensori 1D come una riga e tetto di celle anche su una riga sola", () => {
  const { group, scene, camera, look, canvases } = scene3d({ maxCellsPerOverlay: 40 });
  const bias = NumericMatrix.fromTensor({ shape: [256], data: ffwd.subarray(0, 256) }, { group, cell: 0.1 });
  assert.deepEqual([bias.rows, bias.cols], [1, 256]);
  scene.add(bias.object3d);
  look(0, 0, 2.2);
  group.update(camera, H);
  const [shown] = group.shown();
  assert.equal(shown.window.r1 - shown.window.r0, 1);
  assert.ok(windowCells(shown.window) <= 40);
  assert.equal(canvases[0].texts.length, windowCells(shown.window));
  assert.ok(shown.window.c0 <= 128 && shown.window.c1 > 128, "attorno al centro dello schermo");
});

test("NumbersGroup: budget per scena, vincono le matrici più grandi sullo schermo", () => {
  const { group, scene, camera, look } = scene3d({ maxOverlays: 2 });
  const mats = [0, 1, 2, 3, 4, 5].map((i) => {
    const m = new NumericMatrix({ rows: 16, cols: 16, data: ffwd.subarray(i * 256, i * 256 + 256), group, cell: i === 4 ? 0.06 : 0.04 });
    m.object3d.position.set((i % 3) * 0.7 - 0.7, i < 3 ? 0.4 : -0.4, i === 5 ? -0.6 : 0);
    scene.add(m.object3d);
    return m;
  });
  look(0, 0, 1.0); // celle da 0,04 → ~43 px: abbastanza grandi per i numeri
  group.update(camera, H);
  const showing = mats.filter((m) => m.showingNumbers);
  assert.equal(showing.length, 2, "al più maxOverlays matrici con i numeri");
  assert.ok(mats[4].showingNumbers, "la matrice con le celle più grandi");
  assert.equal(mats[5].showingNumbers, false, "quella più lontana resta senza");
  assert.equal(group.stats.overlays, 2);
  for (const m of mats) m.dispose();
  assert.equal(group.active.size, 0);
  group.dispose();
});

test("attachNumbers su una heatmap di core/visuals.js: colori letti dalla texture, ridisegno quando cambia", { skip: !HAS_DATA }, async () => {
  const spec = await loadJson(`${DATA_DIR}model.json`);
  const run = await loadBuiltinRun({ ...fsLoaders, trace: false });
  const name = "token_embedding_table.weight";
  const tensor = run.tensor(name);
  const mesh = matrixPlane(weightTexture(spec.tensors[name]), 65, 64);
  const { group, scene, camera, look, canvases, centerHit } = scene3d();
  scene.add(mesh);
  let values = tensor.data;
  const handle = attachNumbers(mesh, { getValue: (r, c) => values[r * 64 + c], group });
  assert.deepEqual([handle.target.rows, handle.target.cols], [65, 64], "righe e colonne dalla texture");
  look(0.3, 0.2, 0.9);
  group.update(camera, H);
  assert.equal(handle.showing(), true);
  const [shown] = group.shown();
  const nc = shown.window.c1 - shown.window.c0;
  const image = mesh.material.map.image;
  for (let k = 0; k < canvases[0].texts.length; k += 29) {
    const r = shown.window.r0 + Math.floor(k / nc);
    const c = shown.window.c0 + (k % nc);
    const o = ((65 - 1 - r) * 64 + c) * 4;
    assert.equal(canvases[0].texts[k].color, textColorFor(image.data[o], image.data[o + 1], image.data[o + 2]));
    assert.equal(canvases[0].texts[k].text, formatCell(tensor.data[r * 64 + c], shown.decimals));
  }
  const cell = handle.cellAt(centerHit([mesh]));
  assert.equal(cell.value, tensor.data[cell.index]);
  // Chi riscrive la heatmap (es. pesi di uno snapshot) segna la texture: i numeri seguono.
  assert.equal(group.update(camera, H), false);
  values = tensor.data.map((v) => -v);
  mesh.material.map.needsUpdate = true;
  assert.equal(group.update(camera, H), true);
  assert.equal(canvases[0].texts[0].text, formatCell(values[shown.window.r0 * 64 + shown.window.c0], shown.decimals));
  // Oppure, se cambiano solo i valori, refresh().
  values = tensor.data;
  handle.refresh();
  assert.equal(group.update(camera, H), true);
  handle.dispose();
  assert.equal(mesh.children.filter((c) => c.userData.numbersOverlay).length, 0, "overlay tolto dalla heatmap");
});

test("attachNumbers su una barra verticale (vettore (n, 1) scalato in x come nel Transformer)", () => {
  const n = 64;
  const bytes = Uint8Array.from({ length: n }, (_, i) => (i * 4) % 256);
  const mesh = matrixPlane(weightTexture({ shape: [n, 1], data: Buffer.from(bytes).toString("base64") }), n, 1);
  mesh.scale.x = 0.12 / CELL;
  const { group, scene, camera, look, canvases } = scene3d();
  scene.add(mesh);
  const values = Float32Array.from({ length: n }, (_, i) => (i - 32) / 40);
  const handle = attachNumbers(mesh, { getValue: (r) => values[r], group });
  look(0, 0.1, 0.7);
  group.update(camera, H);
  assert.equal(handle.showing(), true);
  const [shown] = group.shown();
  assert.deepEqual([shown.window.c0, shown.window.c1], [0, 1]);
  assert.equal(canvases[0].texts[0].text, formatCell(values[shown.window.r0], shown.decimals));
  // Le celle sono 3 volte più larghe che alte: il canvas ne rispetta le proporzioni.
  assert.ok(Math.abs(canvases[0].width / canvases[0].height - 3 / (shown.window.r1 - shown.window.r0)) < 0.05);
});
