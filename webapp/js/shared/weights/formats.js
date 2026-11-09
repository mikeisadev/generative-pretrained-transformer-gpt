/**
 * Formati dei pesi che la web app sa leggere (logica pura: niente DOM, niente three.js).
 *
 * - `weights.json` + `weights.f32.bin` (gpt/export.py): manifest con il layout
 *   `{name, shape, offset, numel}` (offset e numel in NUMERO DI FLOAT) e file float32 little-endian.
 * - `training/trace.json` + `snap_XXXXX.f16.bin` (gpt/recording.py): storia del training e
 *   snapshot dei pesi in float16 con lo stesso layout.
 * - `run.json` e `benchmark.json` (scripts/export_run.py, scripts/benchmark.py): descrizione del
 *   run e metriche, JSON liberi di cui si tengono solo i campi semplici.
 * - `*.safetensors` (safetensors.js, riesportato qui).
 *
 * A differenza di js/engine/weights.js (che pretende ESATTAMENTE l'architettura del corso, perché
 * deve eseguirla) qui il manifest può descrivere qualunque insieme di tensori: un run con
 * l'architettura modificata si carica, si guarda e si confronta anche se il motore non lo esegue
 * (vedi `engineCompatibility` in run.js). La conversione dei byte riusa quella del motore.
 */

import { decodeFloat16, decodeFloat32, toArrayBuffer } from "../../engine/weights.js";
import {
  WeightsFormatError,
  checkLayout,
  checkRelativePath,
  checkVocab,
  cleanConfig,
  isPlainObject,
  withLimits,
} from "./validate.js";

export { LIMITS, WeightsFormatError, canonicalName } from "./validate.js";
export { parseSafetensors, safetensorsInfo, SAFETENSORS_DTYPES } from "./safetensors.js";
export { recognizeFiles, readJsonFile, readBinaryFile, filePath } from "./files.js";

export const MANIFEST_FORMAT_VERSION = 1;
export const TRACE_FORMAT_VERSION = 1;

/**
 * Controlla un manifest `weights.json` e lo restituisce normalizzato.
 * @param {unknown} json contenuto già letto con JSON.parse
 * @param {{file?: string, limits?: object}} [options]
 * @returns {{formatVersion: number, file: string, source: string|null, trained: boolean|null,
 *   config: object|null, vocab: string[]|null, totalParams: number,
 *   params: {name: string, shape: number[], offset: number, numel: number}[],
 *   training: {dir: string, trace: string}|null, raw: object}}
 *   `raw` è il JSON originale (serve a GPTEngine.fromBuffers per i run del corso)
 */
export function parseWeightsManifest(json, { file = "weights.json", limits } = {}) {
  const lim = withLimits(limits);
  if (!isPlainObject(json)) throw new WeightsFormatError(file, "il manifest deve essere un oggetto JSON");
  if (json.format_version !== MANIFEST_FORMAT_VERSION) {
    throw new WeightsFormatError(file, `format_version ${JSON.stringify(json.format_version)} non supportato (atteso ${MANIFEST_FORMAT_VERSION})`);
  }
  if (json.dtype !== "float32" || json.byte_order !== "little") {
    throw new WeightsFormatError(file, `pesi attesi in float32 little-endian, trovato ${json.dtype}/${json.byte_order}`);
  }
  const binary = checkRelativePath(json.file, `${file} (campo "file")`);
  const config = cleanConfig(json.config, `${file} (config)`);
  const vocab = checkVocab(json.vocab, `${file} (vocab)`, config?.vocab_size ?? null);
  const { params, total } = checkLayout(json.params, json.total_params, file, lim);
  let training = null;
  if (json.training !== undefined && json.training !== null) {
    if (!isPlainObject(json.training)) throw new WeightsFormatError(file, "training deve essere {dir, trace} oppure null");
    training = {
      dir: checkRelativePath(json.training.dir, `${file} (training.dir)`),
      trace: checkRelativePath(json.training.trace, `${file} (training.trace)`),
    };
  }
  return {
    formatVersion: json.format_version,
    file: binary,
    source: typeof json.source === "string" ? json.source : null,
    trained: typeof json.trained === "boolean" ? json.trained : null,
    config,
    vocab,
    totalParams: total,
    params,
    training,
    raw: json,
  };
}

/**
 * Byte di `weights.f32.bin` → tensori. Nessuna copia: ogni `data` è una vista su `flat`.
 * @param {ArrayBuffer|ArrayBufferView} data byte del file binario
 * @param {ReturnType<typeof parseWeightsManifest>} manifest
 * @returns {{flat: Float32Array, tensors: Map<string, {name, shape, numel, offset, dtype: "float32", data: Float32Array}>}}
 */
export function decodeWeights(data, manifest, { file = manifest.file } = {}) {
  let flat;
  try {
    flat = decodeFloat32(data, manifest.totalParams);
  } catch (error) {
    throw new WeightsFormatError(file, error.message);
  }
  return { flat, tensors: tensorsFromLayout(flat, manifest.params, "float32") };
}

/** Tensori come viste su un Float32Array piatto, secondo un layout già controllato. */
export function tensorsFromLayout(flat, layout, dtype = "float32") {
  const tensors = new Map();
  for (const p of layout) {
    tensors.set(p.name, { name: p.name, shape: p.shape, numel: p.numel, offset: p.offset, dtype, data: flat.subarray(p.offset, p.offset + p.numel) });
  }
  return tensors;
}

/**
 * Controlla `trace.json` (gpt/recording.py) e restituisce la storia del training con gli
 * snapshot in ordine di step. I campi delle metriche (loss, lr, …) restano quelli del file.
 * @returns {object} il trace con in più `layout` (params controllati), `totalParams` e
 *   `snapshots` ordinati `[{step, file, ...metriche}]`
 */
export function parseTrace(json, { file = "trace.json", limits } = {}) {
  const lim = withLimits(limits);
  if (!isPlainObject(json)) throw new WeightsFormatError(file, "il trace deve essere un oggetto JSON");
  if (json.format_version !== TRACE_FORMAT_VERSION) {
    throw new WeightsFormatError(file, `format_version ${JSON.stringify(json.format_version)} non supportato (atteso ${TRACE_FORMAT_VERSION})`);
  }
  if (json.snapshot_dtype !== "float16") throw new WeightsFormatError(file, `snapshot in ${json.snapshot_dtype}, atteso float16`);
  if (json.byte_order !== undefined && json.byte_order !== "little") throw new WeightsFormatError(file, `byte_order ${json.byte_order}, atteso little`);
  const config = cleanConfig(json.config, `${file} (config)`);
  const vocab = checkVocab(json.vocab, `${file} (vocab)`, config?.vocab_size ?? null);
  const { params, total } = checkLayout(json.params, json.total_params, file, lim);
  const list = json.snapshots ?? [];
  if (!Array.isArray(list)) throw new WeightsFormatError(file, "snapshots deve essere una lista");
  const steps = new Set();
  const snapshots = list.map((s, i) => {
    const where = `${file} snapshots[${i}]`;
    if (!isPlainObject(s)) throw new WeightsFormatError(where, "ogni snapshot deve essere un oggetto {step, file, …}");
    if (!Number.isSafeInteger(s.step) || s.step < 0) throw new WeightsFormatError(where, `step non valido: ${JSON.stringify(s.step)}`);
    if (steps.has(s.step)) throw new WeightsFormatError(where, `lo step ${s.step} compare due volte`);
    steps.add(s.step);
    const name = checkRelativePath(s.file, `${where} (file)`);
    if (name.includes("/")) throw new WeightsFormatError(where, `il file dello snapshot deve stare accanto a trace.json, trovato ${JSON.stringify(s.file)}`);
    return { ...s, file: name };
  });
  snapshots.sort((a, b) => a.step - b.step);
  return { ...json, config, vocab, layout: params, totalParams: total, snapshots };
}

/**
 * Byte di uno snapshot `snap_XXXXX.f16.bin` → Float32Array di `totalParams` valori
 * (conversione esatta da float16, la stessa del motore).
 */
export function decodeSnapshot(data, totalParams, { file = "snapshot" } = {}) {
  try {
    return decodeFloat16(data, totalParams);
  } catch (error) {
    throw new WeightsFormatError(file, error.message);
  }
}

/** Stringa semplice di lunghezza limitata (o null): i testi dei file finiscono nell'interfaccia. */
function text(value, max = 2000) {
  return typeof value === "string" ? value.slice(0, max) : null;
}

/**
 * `run.json`: descrizione di un run esportato. Campi letti (tutti facoltativi):
 * `name`, `description`, `created` (o `date`), `source`, `config`, `vocab`, `notes` (stringa o
 * lista di stringhe), `modules` (albero con le forme, tenuto così com'è), `engine_compatible`
 * (false = "il calcolo è diverso dal GPT del corso anche se le forme coincidono").
 */
export function parseRunInfo(json, { file = "run.json" } = {}) {
  if (!isPlainObject(json)) throw new WeightsFormatError(file, "run.json deve essere un oggetto JSON");
  const notes = Array.isArray(json.notes) ? json.notes.filter((n) => typeof n === "string").map((n) => n.slice(0, 2000)) : text(json.notes) ? [text(json.notes)] : [];
  const config = cleanConfig(json.config, `${file} (config)`);
  return {
    name: text(json.name, 200),
    description: text(json.description),
    created: text(json.created ?? json.date ?? json.created_at, 100),
    source: text(json.source, 500),
    config,
    vocab: checkVocab(json.vocab, `${file} (vocab)`, config?.vocab_size ?? null),
    notes,
    modules: json.modules ?? json.module_tree ?? null,
    engineCompatible: typeof json.engine_compatible === "boolean" ? json.engine_compatible : null,
    raw: json,
  };
}

/**
 * `benchmark.json`: metriche calcolate in Python (scripts/benchmark.py). Si controlla solo che
 * sia un oggetto; `metrics` raccoglie i valori numerici di primo livello (e di `metrics`, se c'è)
 * per le tabelle di confronto. Il JSON originale resta in `raw`.
 */
export function parseBenchmark(json, { file = "benchmark.json" } = {}) {
  if (!isPlainObject(json)) throw new WeightsFormatError(file, "benchmark.json deve essere un oggetto JSON");
  const metrics = {};
  const collect = (obj, prefix = "") => {
    for (const [key, value] of Object.entries(obj)) {
      // format_version è la versione del file, non una misura: in una tabella di confronto sarebbe rumore
      if (key === "format_version" || key === "__proto__") continue;
      if (typeof value === "number" && Number.isFinite(value)) metrics[prefix + key] = value;
    }
  };
  collect(json);
  if (isPlainObject(json.metrics)) collect(json.metrics);
  return { metrics, raw: json };
}

export { toArrayBuffer };
