/**
 * Run = un insieme di pesi da guardare e confrontare: il GPT del corso (dati committati in
 * webapp/data/) oppure un run caricato dallo studente (cartella esportata da scripts/export_run.py,
 * safetensors, storia del training). Logica pura: il browser serve solo per fetch e File.
 *
 *   const run = await loadBuiltinRun();                       // "GPT del corso", snapshot a richiesta
 *   run.tensor("blocks.0.sa.heads.0.key.weight")              // {name, shape: [16, 64], numel, dtype, data}
 *   const s = await run.loadSnapshot(500);                    // {step, meta, flat, tensors}
 *   run.unloadSnapshots({ keep: [500] })                      // libera la memoria degli altri step decodificati
 *   const { runs, errors } = await runsFromFiles(input.files, { defaultVocab: run.vocab });
 *   if (runs[0].compatibleWithEngine) runs[0].createEngine().generate(…)
 *
 * `compatibleWithEngine`: il run ha ESATTAMENTE i nomi e le forme che js/engine/gpt.js sa
 * eseguire (expectedParameters della config ricavata dalle forme: qualsiasi vocab_size,
 * block_size, n_embd, n_head, n_layer con n_embd divisibile per n_head; feed-forward 4·n_embd,
 * LayerNorm pre-norm, ReLU, come gpt/*.py) e un vocabolario. Attenzione: forme uguali non
 * garantiscono lo stesso calcolo (es. GELU al posto di ReLU): per questo run.json può dichiarare
 * `engine_compatible: false`.
 */

import { fetchBuffer, fetchJson } from "../../core/binary.js";
import { GPTEngine, expectedParameters } from "../../engine/gpt.js";
import { assertSameLayout } from "../../engine/weights.js";
import { readBinaryFile, readJsonFile, recognizeFiles } from "./files.js";
import { decodeSnapshot, decodeWeights, parseBenchmark, parseRunInfo, parseTrace, parseWeightsManifest, tensorsFromLayout } from "./formats.js";
import { parseSafetensors, safetensorsInfo } from "./safetensors.js";
import { WeightsFormatError, canonicalName, checkVocab, cleanConfig, naturalCompare, withLimits } from "./validate.js";

export const BUILTIN_RUN_ID = "builtin:gpt-corso";
const CONFIG_KEYS = ["vocab_size", "block_size", "n_embd", "n_head", "n_layer"];
const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/**
 * @typedef {{name: string, shape: number[], numel: number, dtype: string, data: Float32Array, offset?: number}} Tensor
 * @typedef {{step: number, file: string|null, meta: object, available: boolean, layoutMatches: boolean,
 *   load: () => Promise<{step: number, meta: object, flat: Float32Array, tensors: Map<string, Tensor>, layout: object[]}>,
 *   unload: () => boolean, loaded: boolean, raw: () => Promise<ArrayBuffer>}} SnapshotEntry
 */

export class WeightRun {
  /**
   * Di solito non si crea a mano: usa loadBuiltinRun, runsFromFiles o RunStore.load.
   * @param {object} o
   * @param {string} o.id identificativo stabile (per il salvataggio nel browser)
   * @param {string} o.name nome da mostrare
   * @param {"builtin"|"upload"} o.source
   * @param {Map<string, Tensor>} o.tensors nell'ordine in cui mostrarli (di solito quello di named_parameters)
   * @param {Float32Array} [o.flat] tutti i valori, se i tensori sono viste su un unico array
   * @param {object} [o.config] config dichiarata (GPTConfig.to_dict()); se manca si ricava dalle forme
   * @param {string[]} [o.vocab] token id → testo
   * @param {object} [o.trace] trace.json controllato (parseTrace)
   * @param {SnapshotEntry[]} [o.snapshots]
   * @param {object} [o.benchmark] benchmark.json (parseBenchmark)
   * @param {object} [o.info] run.json (parseRunInfo)
   * @param {string[]} [o.defaultVocab] vocabolario da usare se il run non ne ha uno (stessa dimensione)
   * @param {object} [o.meta] altro: provenienza, avvisi (`warnings`), …
   */
  constructor({ id, name, description = "", source, tensors, flat = null, config = null, vocab = null, trace = null, snapshots = [], benchmark = null, info = null, defaultVocab = null, meta = {}, engineSource = null }) {
    if (!(tensors instanceof Map) || tensors.size === 0) throw new WeightsFormatError(name, "il run non contiene nessun tensore");
    this.id = id;
    this.name = name;
    this.description = description;
    this.source = source;
    this.tensors = tensors;
    this.trace = trace;
    this.snapshots = [...snapshots].sort((a, b) => a.step - b.step);
    this.benchmark = benchmark;
    this.info = info;
    this.meta = { warnings: [], ...meta };
    this.totalParams = [...tensors.values()].reduce((s, t) => s + t.numel, 0);
    /** Dettagli della compatibilità con il motore JS: config ricavata, problemi, tensori mancanti/in più. */
    this.engine = engineCompatibility(tensors, { config, vocab, declared: info?.engineCompatible ?? null, defaultVocab });
    this.compatibleWithEngine = this.engine.compatible;
    this.config = config ?? (this.engine.config ? { ...this.engine.config } : null);
    this.vocab = vocab ?? this.engine.vocab ?? null;
    if (!vocab && this.engine.vocabSource === "default") {
      this.meta.warnings.push("il run non contiene il vocabolario: si usa quello del GPT del corso (stessa dimensione)");
    }
    this._flat = flat;
    this._engineSource = engineSource; // run del corso: manifest e byte originali (anche gli snapshot del motore)
  }

  /** Parametri in ordine: `[{name, shape, numel, dtype}]` (senza i dati). */
  get params() {
    return [...this.tensors.values()].map(({ name, shape, numel, dtype }) => ({ name, shape, numel, dtype }));
  }

  /** Step degli snapshot di cui ci sono i pesi (quelli citati da trace.json ma mancanti restano fuori). */
  get steps() {
    return this.snapshots.filter((s) => s.available).map((s) => s.step);
  }

  /** Un tensore per nome (accetta anche i nomi con `module.`/`_orig_mod.`). */
  tensor(name) {
    return this.tensors.get(name) ?? this.tensors.get(canonicalName(name));
  }

  /**
   * Tutti i valori in un Float32Array. Se i tensori sono viste su un unico array (manifest,
   * safetensors) è quello, nell'ordine dei byte del file; altrimenti la concatenazione
   * nell'ordine di `tensors`, creata una volta sola.
   */
  flat() {
    this._flat ??= concatTensors(this.tensors);
    return this._flat;
  }

  snapshot(step) {
    return this.snapshots.find((s) => s.step === step);
  }

  /**
   * Toglie dalla cache gli snapshot decodificati, tranne quelli degli step in `keep` (es. lo step mostrato e
   * quello di confronto): per i run grandi, quando lo studente ha visitato molti step. → quanti ne ha tolti.
   * @param {{keep?: Iterable<number>}} [o]
   */
  unloadSnapshots({ keep = [] } = {}) {
    const kept = new Set(keep);
    let dropped = 0;
    for (const s of this.snapshots) if (!kept.has(s.step) && s.unload?.()) dropped++;
    return dropped;
  }

  /** Pesi dello snapshot allo step indicato (scaricati o letti una volta sola). */
  async loadSnapshot(step) {
    const entry = this.snapshot(step);
    if (!entry) throw new Error(`snapshot dello step ${step} non disponibile (step: ${this.steps.join(", ") || "nessuno"})`);
    if (!entry.available) throw new Error(`lo snapshot dello step ${step} è citato in trace.json ma il file ${entry.file} non c'è`);
    return entry.load();
  }

  /**
   * Pesi a uno step qualsiasi: lo snapshot se esiste, altrimenti (con `interpolate`) la media
   * pesata dei due snapshot vicini. Didattico: il training vero non va in linea retta tra due
   * snapshot, per questo il risultato dice `exact: false` e da quali step viene.
   * @returns {Promise<{step: number, exact: boolean, from: number[], t: number, flat: Float32Array, tensors: Map<string, Tensor>}>}
   */
  async weightsAt(step, { interpolate = true } = {}) {
    const b = snapshotBracket(this.steps, step);
    if (!b) throw new Error("questo run non ha snapshot del training");
    if (b.t === 0 || !interpolate) {
      const s = await this.loadSnapshot(b.t < 0.5 ? b.lo : b.hi);
      return { step: s.step, exact: s.step === step, from: [s.step], t: 0, flat: s.flat, tensors: s.tensors };
    }
    const [a, z] = await Promise.all([this.loadSnapshot(b.lo), this.loadSnapshot(b.hi)]);
    const flat = new Float32Array(a.flat.length);
    for (let i = 0; i < flat.length; i++) flat[i] = a.flat[i] + (z.flat[i] - a.flat[i]) * b.t;
    return { step, exact: false, from: [b.lo, b.hi], t: b.t, flat, tensors: tensorsFromLayout(flat, a.layout, "float16") };
  }

  /**
   * I valori di `tensors` (default: quelli del run) nell'ordine di expectedParameters, cioè il
   * layout di weights.f32.bin che il motore si aspetta: `engine.setWeights(run.engineFlat(s.tensors))`.
   */
  engineFlat(tensors = this.tensors) {
    if (!this.compatibleWithEngine) throw new Error(`il motore JS non può eseguire questo run: ${this.engine.issues.join("; ")}`);
    const expected = expectedParameters(this.engine.config);
    const flat = new Float32Array(expected.reduce((s, p) => s + p.shape.reduce((a, b) => a * b, 1), 0));
    let at = 0;
    for (const p of expected) {
      const t = tensors.get(p.name);
      if (!t) throw new Error(`manca il tensore ${p.name}`);
      flat.set(t.data, at);
      at += t.data.length;
    }
    return flat;
  }

  /** Un GPTEngine con i pesi di questo run (errore chiaro se il motore non lo sa eseguire). */
  createEngine() {
    if (!this.compatibleWithEngine) throw new Error(`il motore JS non può eseguire questo run: ${this.engine.issues.join("; ")}`);
    if (this._engineSource) {
      const { manifest, buffer, sources } = this._engineSource;
      return GPTEngine.fromBuffers(manifest, buffer, sources);
    }
    const flat = this.engineFlat();
    let offset = 0;
    const params = expectedParameters(this.engine.config).map(({ name, shape }) => {
      const numel = shape.reduce((a, b) => a * b, 1);
      const entry = { name, shape, offset, numel };
      offset += numel;
      return entry;
    });
    const manifest = { format_version: 1, dtype: "float32", byte_order: "little", file: "weights.f32.bin", source: this.name, trained: true, config: this.engine.config, vocab: this.vocab, total_params: offset, params, training: null };
    return GPTEngine.fromBuffers(manifest, flat.buffer);
  }

  /** Riepilogo leggero (per elenchi e per il salvataggio). */
  summary() {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      source: this.source,
      totalParams: this.totalParams,
      tensors: this.tensors.size,
      config: this.config,
      steps: this.steps,
      hasTrace: Boolean(this.trace),
      hasBenchmark: Boolean(this.benchmark),
      compatibleWithEngine: this.compatibleWithEngine,
      warnings: [...this.meta.warnings],
    };
  }
}

/**
 * Due snapshot attorno a `step`: `{lo, hi, t}` con t ∈ [0, 1] (0 = esattamente `lo`).
 * Fuori dall'intervallo si resta sul primo o sull'ultimo (`clamped: true`).
 * @param {number[]} steps in ordine crescente
 */
export function snapshotBracket(steps, step) {
  if (!steps.length) return null;
  const first = steps[0];
  const last = steps[steps.length - 1];
  if (step <= first) return { lo: first, hi: first, t: 0, clamped: step < first };
  if (step >= last) return { lo: last, hi: last, t: 0, clamped: step > last };
  let i = 0;
  while (steps[i + 1] <= step) i++;
  const lo = steps[i];
  const hi = steps[i + 1];
  return { lo, hi, t: lo === step ? 0 : (step - lo) / (hi - lo), clamped: false };
}

// ---------------------------------------------------------------- compatibilità con il motore

/**
 * Config del GPT del corso ricavata dalle forme dei pesi (o null, con i motivi in `issues`).
 * V e C da token_embedding_table (V, C), block_size da position_embedding_table, n_layer dai
 * `blocks.<i>`, n_head dalle `blocks.0.sa.heads.<h>`.
 */
export function inferGptConfig(tensors, issues = []) {
  const shapeOf = (n) => tensors.get(n)?.shape;
  const tok = shapeOf("token_embedding_table.weight");
  const pos = shapeOf("position_embedding_table.weight");
  if (!tok || tok.length !== 2) {
    issues.push("manca token_embedding_table.weight con forma (vocab_size, n_embd): non ha la struttura del GPT del corso");
    return null;
  }
  if (!pos || pos.length !== 2 || pos[1] !== tok[1]) {
    issues.push("manca position_embedding_table.weight con forma (block_size, n_embd)");
    return null;
  }
  const layers = new Set();
  const heads = new Set();
  for (const name of tensors.keys()) {
    const l = /^blocks\.(\d+)\./.exec(name);
    if (l) layers.add(Number(l[1]));
    const h = /^blocks\.0\.sa\.heads\.(\d+)\./.exec(name);
    if (h) heads.add(Number(h[1]));
  }
  const contiguous = (set) => set.size > 0 && Math.max(...set) === set.size - 1;
  if (!contiguous(layers)) {
    issues.push(layers.size ? "i blocchi blocks.<i> non sono numerati da 0 senza salti" : "nessun blocco transformer (blocks.0, blocks.1, …)");
    return null;
  }
  if (!contiguous(heads)) {
    issues.push(heads.size ? "le head blocks.0.sa.heads.<h> non sono numerate da 0 senza salti" : "nessuna head di attention in blocks.0.sa.heads");
    return null;
  }
  return { vocab_size: tok[0], block_size: pos[0], n_embd: tok[1], n_head: heads.size, n_layer: layers.size };
}

/**
 * Il motore JS sa eseguire questi pesi?
 * @param {Map<string, {shape: number[]}>} tensors
 * @param {{config?: object|null, vocab?: string[]|null, declared?: boolean|null, defaultVocab?: string[]|null}} [o]
 * @returns {{compatible: boolean, config: object|null, vocab: string[]|null, vocabSource: "run"|"default"|null,
 *   issues: string[], missing: string[], extra: string[], mismatched: {name: string, shape: number[], expected: number[]}[]}}
 */
export function engineCompatibility(tensors, { config = null, vocab = null, declared = null, defaultVocab = null } = {}) {
  const issues = [];
  const inferred = inferGptConfig(tensors, issues);
  let missing = [];
  let extra = [];
  let mismatched = [];
  if (inferred) {
    let expected = null;
    try {
      expected = expectedParameters(inferred);
    } catch (error) {
      issues.push(error.message);
    }
    if (expected) {
      const known = new Set(expected.map((p) => p.name));
      missing = expected.filter((p) => !tensors.has(p.name)).map((p) => p.name);
      extra = [...tensors.keys()].filter((n) => !known.has(n));
      mismatched = expected.filter((p) => tensors.has(p.name) && !sameShape(tensors.get(p.name).shape, p.shape)).map((p) => ({ name: p.name, shape: tensors.get(p.name).shape, expected: p.shape }));
      const few = (list) => list.slice(0, 3).join(", ") + (list.length > 3 ? ` e altri ${list.length - 3}` : "");
      if (missing.length) issues.push(`mancano ${missing.length} tensori che il motore usa (${few(missing)})`);
      if (extra.length) issues.push(`tensori che il motore non conosce, architettura modificata? (${few(extra)})`);
      if (mismatched.length) issues.push(`forme diverse da quelle attese: ${mismatched.slice(0, 3).map((m) => `${m.name} [${m.shape}] invece di [${m.expected}]`).join("; ")}`);
    }
    if (config) {
      for (const key of CONFIG_KEYS) {
        if (key in config && config[key] !== inferred[key]) issues.push(`config.${key} è ${config[key]} ma le forme dei pesi dicono ${inferred[key]}`);
      }
    }
  }
  let usedVocab = vocab ?? null;
  let vocabSource = vocab ? "run" : null;
  if (!usedVocab && inferred && defaultVocab?.length === inferred.vocab_size) {
    usedVocab = defaultVocab;
    vocabSource = "default";
  }
  if (inferred) {
    if (!usedVocab) issues.push("manca il vocabolario (vocab): senza non si può trasformare il testo in token");
    else if (usedVocab.length !== inferred.vocab_size) issues.push(`il vocabolario ha ${usedVocab.length} voci ma i pesi ne prevedono ${inferred.vocab_size}`);
  }
  if (declared === false) issues.push("run.json dichiara engine_compatible: false (il calcolo è diverso da quello del GPT del corso)");
  const fullConfig = inferred ? { ...inferred, dropout: typeof config?.dropout === "number" ? config.dropout : 0 } : null;
  return { compatible: issues.length === 0, config: fullConfig, vocab: usedVocab, vocabSource, issues, missing, extra, mismatched };
}

// ---------------------------------------------------------------- snapshot

/**
 * Voce di uno snapshot con caricamento pigro e in cache (un errore di rete non resta in cache).
 * `unload()` toglie dalla cache lo snapshot decodificato (4 byte per peso: con un run grande e tanti
 * snapshot la memoria cresce a ogni step visitato); chi ne tiene ancora un riferimento lo conserva, e il
 * prossimo `load()` lo rilegge. `loaded` dice se è in cache (anche se il caricamento è in corso).
 * @param {{step: number, file: string|null, meta: object, layout: object[], totalParams: number,
 *   layoutMatches: boolean, raw: (() => Promise<ArrayBuffer>)|null}} o
 * @returns {SnapshotEntry}
 */
export function snapshotEntry({ step, file, meta, layout, totalParams, layoutMatches, raw }) {
  let pending = null;
  const load = () => {
    if (!pending) {
      const request = Promise.resolve(raw()).then((buffer) => {
        const flat = decodeSnapshot(buffer, totalParams, { file: file ?? `snapshot ${step}` });
        return { step, meta, flat, tensors: tensorsFromLayout(flat, layout, "float16"), layout };
      });
      pending = request;
      request.catch(() => {
        if (pending === request) pending = null;
      });
    }
    return pending;
  };
  const unload = () => {
    const had = pending !== null;
    pending = null;
    return had;
  };
  return {
    step,
    file,
    meta,
    available: Boolean(raw),
    layoutMatches,
    load,
    unload,
    get loaded() {
      return pending !== null;
    },
    raw: raw ?? (() => Promise.reject(new Error(`snapshot ${step} non disponibile`))),
  };
}

/** Concatenazione dei valori di tutti i tensori, nell'ordine della Map. */
function concatTensors(tensors) {
  let total = 0;
  for (const t of tensors.values()) total += t.data.length;
  const flat = new Float32Array(total);
  let at = 0;
  for (const t of tensors.values()) {
    flat.set(t.data, at);
    at += t.data.length;
  }
  return flat;
}

/** I tensori dello snapshot hanno gli stessi nomi e forme dei pesi del run? */
function sameLayout(layout, tensors) {
  if (layout.length !== tensors.size) return false;
  return layout.every((p) => {
    const t = tensors.get(p.name);
    return t && sameShape(t.shape, p.shape);
  });
}

// ---------------------------------------------------------------- run del corso

const builtinCache = new Map();
/** Interi all'italiana con il separatore delle migliaia sempre presente (come fmt di core/dom.js). */
const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const joinUrl = (base, path) => (!base ? path : base.endsWith("/") ? base + path : `${base}/${path}`);

/**
 * Il run "GPT del corso" dai dati committati: weights.json + weights.f32.bin e, se c'è,
 * training/trace.json con gli 11 snapshot (scaricati solo quando servono).
 * Con i loader predefiniti (fetch) il run si carica una volta per pagina.
 * @param {{baseUrl?: string, loadJson?: (url: string) => Promise<any>, loadBuffer?: (url: string) => Promise<ArrayBuffer>, trace?: boolean}} [o]
 * @returns {Promise<WeightRun>}
 */
export function loadBuiltinRun({ baseUrl = "data/", loadJson = fetchJson, loadBuffer = fetchBuffer, trace = true } = {}) {
  const cacheable = loadJson === fetchJson && loadBuffer === fetchBuffer;
  const key = `${baseUrl}|${trace}`;
  if (cacheable && builtinCache.has(key)) return builtinCache.get(key);
  const pending = buildBuiltinRun({ baseUrl, loadJson, loadBuffer, trace });
  if (cacheable) {
    builtinCache.set(key, pending);
    pending.catch(() => builtinCache.delete(key));
  }
  return pending;
}

async function buildBuiltinRun({ baseUrl, loadJson, loadBuffer, trace }) {
  const rawManifest = await loadJson(joinUrl(baseUrl, "weights.json"));
  const manifest = parseWeightsManifest(rawManifest);
  const buffer = await loadBuffer(joinUrl(baseUrl, manifest.file));
  const { flat, tensors } = decodeWeights(buffer, manifest);
  let history = null;
  let snapshots = [];
  if (trace && manifest.training) {
    history = parseTrace(await loadJson(joinUrl(baseUrl, manifest.training.trace)), { file: manifest.training.trace });
    try {
      assertSameLayout(history.layout, manifest.params);
    } catch (error) {
      throw new WeightsFormatError(manifest.training.trace, error.message);
    }
    snapshots = history.snapshots.map((s) =>
      snapshotEntry({
        step: s.step,
        file: s.file,
        meta: s,
        layout: history.layout,
        totalParams: history.totalParams,
        layoutMatches: true,
        raw: () => loadBuffer(joinUrl(baseUrl, `${manifest.training.dir}/${s.file}`)),
      }),
    );
  }
  const steps = snapshots.map((s) => s.step);
  const maxIters = history?.train_config?.max_iters;
  const parts = [`il GPT di gpt/ (ricostruito dal notebook di Karpathy) addestrato su Tiny Shakespeare: ${INT.format(manifest.totalParams)} parametri`];
  if (Number.isInteger(maxIters)) parts.push(`pesi finali dopo ${INT.format(maxIters)} step`);
  if (steps.length) parts.push(`${steps.length} snapshot del training (step ${steps.join(", ")})`);
  return new WeightRun({
    id: BUILTIN_RUN_ID,
    name: "GPT del corso",
    description: `${parts.join(", ")}.`.replace(/^./, (c) => c.toUpperCase()),
    source: "builtin",
    tensors,
    flat,
    config: manifest.config,
    vocab: manifest.vocab,
    trace: history,
    snapshots,
    meta: { trained: manifest.trained, origin: manifest.source, device: history?.device ?? null, files: [manifest.file] },
    engineSource: { manifest: rawManifest, buffer, sources: { baseUrl, loadJson, loadBuffer } },
  });
}

// ---------------------------------------------------------------- run caricati

/**
 * Run dai file caricati (FileList di una cartella, selezione multipla, drag & drop).
 * Ogni run è indipendente: un file sbagliato produce un errore per quel run, gli altri si caricano.
 * @param {FileList|Iterable<object>} files
 * @param {{limits?: object, defaultVocab?: string[]}} [o] `defaultVocab`: vocabolario per i run che
 *   non lo contengono (es. un safetensors), usato solo se ha la dimensione giusta
 * @returns {Promise<{runs: WeightRun[], errors: {run: string, message: string}[], ignored: {path: string, reason: string}[]}>}
 * @throws {WeightsFormatError} se i file nel loro insieme superano il limite di dimensione
 */
export async function runsFromFiles(files, { limits, defaultVocab = null } = {}) {
  const lim = withLimits(limits);
  const recognized = recognizeFiles(files, { limits: lim });
  const runs = [];
  const errors = [];
  for (const group of recognized.runs) {
    const label = group.name ?? group.key;
    if (group.errors.length) {
      errors.push({ run: label, message: group.errors.join("; ") });
      continue;
    }
    try {
      runs.push(await runFromGroup(group, { limits: lim, defaultVocab, index: runs.length + 1 }));
    } catch (error) {
      errors.push({ run: label, message: error.message });
    }
  }
  return { runs, errors, ignored: recognized.ignored };
}

const dirOf = (path) => (path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "");
const baseOf = (path) => path.slice(path.lastIndexOf("/") + 1);

/** Un gruppo di recognizeFiles → WeightRun (legge e controlla i file). */
async function runFromGroup(group, { limits, defaultVocab, index }) {
  const warnings = [...group.warnings];
  const info = group.runInfo ? parseRunInfo(await readJsonFile(group.runInfo, { limits }), { file: group.runInfo.path }) : null;
  const benchmark = group.benchmark ? parseBenchmark(await readJsonFile(group.benchmark, { limits }), { file: group.benchmark.path }) : null;
  const history = group.trace ? parseTrace(await readJsonFile(group.trace, { limits }), { file: group.trace.path, limits }) : null;
  let config = info?.config ?? null;
  let vocab = info?.vocab ?? null;
  let tensors;
  let flat = null;
  const files = [];
  const formats = [];

  if (group.manifest) {
    const manifest = parseWeightsManifest(await readJsonFile(group.manifest, { limits }), { file: group.manifest.path, limits });
    const dir = dirOf(group.manifest.path);
    const wanted = dir ? `${dir}/${manifest.file}` : manifest.file;
    const bin = group.binaries.find((b) => b.path === wanted) ?? group.binaries.find((b) => baseOf(b.path) === baseOf(manifest.file));
    if (!bin) throw new WeightsFormatError(group.manifest.path, `manca il file dei pesi ${manifest.file} indicato nel campo "file"`);
    ({ flat, tensors } = decodeWeights(await readBinaryFile(bin, { limits }), manifest, { file: bin.path }));
    config ??= manifest.config;
    vocab ??= manifest.vocab;
    files.push(group.manifest.path, bin.path);
    formats.push("weights.json + float32");
    if (group.safetensors.length) warnings.push(`ignorati ${group.safetensors.length} .safetensors: in questa cartella i pesi sono quelli di weights.json`);
  } else if (group.safetensors.length) {
    tensors = new Map();
    for (const entry of group.safetensors) {
      const parsed = parseSafetensors(await readBinaryFile(entry, { limits }), { file: entry.path, limits });
      for (const [name, t] of parsed.tensors) {
        if (tensors.has(name)) throw new WeightsFormatError(entry.path, `il tensore ${name} compare in più shard`);
        tensors.set(name, t);
      }
      // Un file solo: i tensori sono già viste su un unico array, niente seconda copia dei pesi.
      if (group.safetensors.length === 1) flat = parsed.flat;
      const meta = safetensorsInfo(parsed.metadata);
      config ??= cleanConfig(meta.config, `${entry.path} (__metadata__.config)`);
      vocab ??= checkVocab(meta.vocab, `${entry.path} (__metadata__.vocab)`);
      files.push(entry.path);
      const kinds = new Set([...parsed.tensors.values()].map((t) => t.dtype));
      formats.push(`safetensors ${[...kinds].join("/") || "vuoto"}`);
    }
    // Nel file i tensori sono in ordine di byte (spesso alfabetico): l'ordine naturale è più leggibile.
    tensors = new Map([...tensors].sort((a, b) => naturalCompare(a[0], b[0])));
    if (group.binaries.length) warnings.push(`ignorati ${group.binaries.length} file .bin senza weights.json`);
  } else if (history) {
    // Solo la storia del training: i pesi del run sono quelli dell'ultimo snapshot presente.
    const last = [...history.snapshots].reverse().find((s) => findSnapshotFile(group, s.file));
    if (!last) throw new WeightsFormatError(group.trace.path, "nessun file di pesi: né weights.json, né .safetensors, né uno snapshot citato in trace.json");
    const entry = findSnapshotFile(group, last.file);
    flat = decodeSnapshot(await readBinaryFile(entry, { limits }), history.totalParams, { file: entry.path });
    tensors = tensorsFromLayout(flat, history.layout, "float16");
    files.push(entry.path);
    formats.push(`snapshot float16 dello step ${last.step}`);
    warnings.push(`pesi presi dall'ultimo snapshot del training (step ${last.step}, float16)`);
  } else {
    throw new WeightsFormatError(group.runInfo?.path ?? group.key, "nessun file di pesi nel run");
  }

  // Nomi canonici: un checkpoint di un modello compilato ha "_orig_mod." davanti a ogni nome.
  const renamed = canonicalize(tensors);
  if (renamed.changed) {
    tensors = renamed.tensors;
    if (history) history.layout = history.layout.map((p) => ({ ...p, name: canonicalName(p.name) }));
    warnings.push(`tolti i prefissi ${[...renamed.prefixes].join(" e ")} dai nomi dei tensori (aggiunti da torch.compile o DataParallel)`);
  } else if (renamed.conflict) {
    warnings.push("alcuni nomi hanno prefissi come module. o _orig_mod. ma toglierli creerebbe nomi doppi: lasciati così");
  }

  config ??= history?.config ?? null;
  vocab ??= history?.vocab ?? null;
  const matches = history ? sameLayout(history.layout, tensors) : false;
  const snapshots = history
    ? history.snapshots.map((s) => {
        const entry = findSnapshotFile(group, s.file, dirOf(group.trace.path));
        return snapshotEntry({
          step: s.step,
          file: s.file,
          meta: s,
          layout: history.layout,
          totalParams: history.totalParams,
          layoutMatches: matches,
          raw: entry ? () => readBinaryFile(entry, { limits }) : null,
        });
      })
    : [];
  if (history) {
    const missing = snapshots.filter((s) => !s.available).map((s) => s.step);
    if (missing.length) warnings.push(`trace.json cita ${missing.length} snapshot senza file (step ${missing.join(", ")}): restano solo le metriche`);
    if (snapshots.length && !snapshots[0].layoutMatches) warnings.push("gli snapshot di trace.json hanno tensori diversi dai pesi del run: si guardano a parte");
  }
  const extra = group.snapshots.filter((s) => !history?.snapshots.some((h) => h.file === baseOf(s.path)));
  if (extra.length) warnings.push(`ignorati ${extra.length} snapshot non citati in trace.json`);

  const name = info?.name ?? group.name ?? `Run caricato ${index}`;
  flat ??= concatTensors(tensors);
  return new WeightRun({
    id: `upload:${fingerprint(flat, [...tensors.keys()])}`,
    name,
    description: info?.description ?? "",
    source: "upload",
    tensors,
    flat,
    config,
    vocab,
    trace: history,
    snapshots,
    benchmark,
    info,
    defaultVocab,
    meta: { warnings, files, formats, created: info?.created ?? null, origin: info?.source ?? null, notes: info?.notes ?? [] },
  });
}

/** Il file di uno snapshot tra quelli caricati: prima nella cartella di trace.json, poi ovunque nel run. */
function findSnapshotFile(group, file, traceDir = null) {
  const same = group.snapshots.filter((s) => baseOf(s.path) === file);
  return same.find((s) => dirOf(s.path) === traceDir) ?? same[0] ?? null;
}

/** Toglie i prefissi degli involucri di PyTorch, se non crea nomi doppi. */
function canonicalize(tensors) {
  const prefixes = new Set();
  const out = new Map();
  for (const [name, t] of tensors) {
    const canon = canonicalName(name);
    if (canon !== name) prefixes.add(name.startsWith("_orig_mod.") ? "_orig_mod." : "module.");
    if (out.has(canon)) return { changed: false, conflict: true, tensors };
    out.set(canon, canon === name ? t : { ...t, name: canon });
  }
  return { changed: prefixes.size > 0, conflict: false, tensors: out, prefixes };
}

/**
 * Impronta dei pesi (FNV-1a a 32 bit sui bit dei float, al più ~1 milione di campioni, più i
 * nomi): lo stesso run caricato due volte ha lo stesso id e nel browser non si duplica.
 */
export function fingerprint(flat, names) {
  let h = 0x811c9dc5;
  const bits = new Uint32Array(flat.buffer, flat.byteOffset, flat.length);
  const stride = Math.max(1, Math.floor(bits.length / 1_000_000));
  for (let i = 0; i < bits.length; i += stride) {
    h ^= bits[i];
    h = Math.imul(h, 0x01000193);
  }
  for (const name of names) {
    for (let i = 0; i < name.length; i++) {
      h ^= name.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
  }
  return `${(h >>> 0).toString(16).padStart(8, "0")}-${flat.length.toString(36)}`;
}
