/**
 * Dati VERI del training registrato (webapp/data/training/, prodotti da scripts/record_training.py):
 * trace.json (loss per step, metriche, snapshot) e gli snapshot dei pesi in float16.
 *
 * Gli snapshot si caricano qui (non con engine.loadSnapshot) perché per l'animazione servono
 * più snapshot contemporaneamente (interpolazione tra due). Per i forward con i pesi di uno
 * snapshot usiamo un motore SEPARATO: il motore condiviso del tab (pesi addestrati) non viene
 * mai toccato, così la vista "Rete neurale" e le altre azioni non vedono pesi a metà training.
 */
import { decodeFloat16 } from "../../../engine/weights.js";
import { GPTEngine, mulberry32 } from "../../../engine/gpt.js";
import { fetchBuffer } from "../../../core/binary.js";
import { crossEntropy, makeBatch } from "./math.js";

/** Famiglie di gruppi di parametri: i blocchi si riassumono per tipo (4 blocchi → 1 serie). */
export const GROUP_KINDS = [
  { id: "token_embedding", label: "token embedding", match: (g) => g === "token_embedding" },
  { id: "position_embedding", label: "position embedding", match: (g) => g === "position_embedding" },
  { id: "attention", label: "attention (blocchi)", match: (g) => g.endsWith(".attention") },
  { id: "ffwd", label: "feed-forward (blocchi)", match: (g) => g.endsWith(".ffwd") },
  { id: "layernorm", label: "LayerNorm (blocchi)", match: (g) => g.endsWith(".layernorm") },
  { id: "ln_f", label: "LayerNorm finale", match: (g) => g === "ln_f" },
  { id: "lm_head", label: "lm_head", match: (g) => g === "lm_head" },
];

/**
 * Riassume una metrica per gruppo ({gruppo: valori[]}) in una serie per famiglia.
 * mode "rss": radice della somma dei quadrati (corretto per le norme: è la norma dell'unione
 * dei gruppi); mode "mean": media semplice (per rapporti e deviazioni standard).
 * @returns {{id, label, values: number[]}[]}
 */
export function summarizeByKind(byGroup, mode = "mean") {
  const groups = Object.keys(byGroup);
  return GROUP_KINDS.map((kind) => {
    const members = groups.filter(kind.match);
    if (!members.length) return null;
    const n = byGroup[members[0]].length;
    const values = new Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (const g of members) acc += mode === "rss" ? byGroup[g][i] ** 2 : byGroup[g][i];
      values[i] = mode === "rss" ? Math.sqrt(acc) : acc / members.length;
    }
    return { id: kind.id, label: kind.label, values };
  }).filter(Boolean);
}

/** Gruppo di parametri (nome di trace.groups) a cui appartiene una stazione del flusso. */
export function groupOfStation(id) {
  if (id === "tokens" || id === "embeddings") return "token_embedding";
  if (id === "embed_add") return "position_embedding";
  const m = /^blocks\.(\d+)\.(\w+)$/.exec(id);
  if (m) {
    const part = m[2];
    if (part === "ln1" || part === "ln2") return `blocks.${m[1]}.layernorm`;
    if (part === "heads" || part === "proj" || part === "add1") return `blocks.${m[1]}.attention`;
    return `blocks.${m[1]}.ffwd`;
  }
  if (id === "ln_f") return "ln_f";
  return "lm_head";
}

/** Indice del punto di metrics.steps più vicino (a sinistra) allo step dato. */
export function metricIndex(steps, step) {
  let lo = 0;
  let hi = steps.length - 1;
  if (step >= steps[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (steps[mid] <= step) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** Loss (cross-entropy media) di un batch di finestre {x, y} con i pesi attuali del motore. */
export function batchLoss(engine, windows) {
  const V = engine.config.vocab_size;
  let sum = 0;
  const perWindow = windows.map((w) => {
    const { logits } = engine.forward(w.x);
    const ce = crossEntropy(logits, w.y, V);
    sum += ce.mean;
    return { ...ce, logits };
  });
  return { mean: sum / windows.length, perWindow };
}

/** Batch come get_batch di gpt/data.py, da un testo tokenizzato e un seme. */
export function sampleBatch(ids, blockSize, batchSize, seed) {
  return makeBatch(ids, blockSize, batchSize, mulberry32(seed));
}

export class TrainingData {
  /** @param {() => Promise<import("../../../engine/gpt.js").GPTEngine>} getEngine motore condiviso del tab */
  constructor(getEngine, baseUrl = "data/") {
    this.getEngine = getEngine;
    this.baseUrl = baseUrl;
    this.snapshots = new Map(); // step → Promise<Float32Array>
    this.tracePromise = null;
    this.enginePromise = null;
  }

  /** trace.json (caricato una volta, tramite il motore condiviso che ne controlla il layout). */
  trace() {
    this.tracePromise ??= this.getEngine().then((engine) => engine.loadTrace());
    return this.tracePromise;
  }

  /** Pesi (float32) dello snapshot allo step dato. */
  async snapshot(step) {
    if (!this.snapshots.has(step)) {
      const pending = (async () => {
        const [trace, engine] = await Promise.all([this.trace(), this.getEngine()]);
        const entry = trace.snapshots.find((s) => s.step === step);
        if (!entry) throw new Error(`snapshot ${step} non presente in trace.json`);
        const dir = engine.manifest.training?.dir ?? "training";
        return decodeFloat16(await fetchBuffer(`${this.baseUrl}${dir}/${entry.file}`), engine.paramCount);
      })();
      pending.catch(() => this.snapshots.delete(step));
      this.snapshots.set(step, pending);
    }
    return this.snapshots.get(step);
  }

  /** Tutti gli snapshot (11 × ~0,8 MB in float32). */
  async allSnapshots() {
    const trace = await this.trace();
    const steps = trace.snapshots.map((s) => s.step);
    const values = await Promise.all(steps.map((s) => this.snapshot(s)));
    return new Map(steps.map((s, i) => [s, values[i]]));
  }

  /** Motore separato per i forward con i pesi del training (vedi commento in testa). */
  trainingEngine() {
    this.enginePromise ??= this.getEngine().then((main) => GPTEngine.fromBuffers(main.manifest, main.weights));
    return this.enginePromise;
  }

  /**
   * Viste per nome di tensore su un array di pesi qualsiasi (stesso layout di weights.f32.bin):
   * è il formato che SceneTextures.writeWeights si aspetta.
   */
  static views(engine, values) {
    const map = new Map();
    for (const [name, p] of engine.paramViews) map.set(name, values.subarray(p.offset, p.offset + p.numel));
    return map;
  }
}
