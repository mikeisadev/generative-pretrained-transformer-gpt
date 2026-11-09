/**
 * Il benchmark di un run nel Laboratorio (logica pura, testata): lettura prudente di benchmark.json
 * (anche rotto: è un file dello studente), i numeri a uno step del training e il confronto A/B con la
 * colonna "meglio", come gpt/benchmark.py (comparison_rows, comparison_warnings).
 *
 *   const bench = readBenchmark(run.benchmark?.raw ?? builtinJson);   // normalizzato, con `problems`
 *   const ref = referenceAt(bench, 500);                              // metriche dello snapshot 500
 *   comparisonRows(refA, refB)                                        // [{key, label, a, b, delta, better}]
 *
 * Convenzione del Laboratorio: delta = A − B (A è il run nella scena, il "dopo"; B il termine di
 * paragone, il "prima"). È lo stesso verso di `scripts/benchmark.py PRIMA DOPO`, che stampa dopo − prima.
 */

export const PROTOCOL_ID = "finestre-contigue-v1";

/** Limiti della lettura: un benchmark.json enorme o malevolo non deve bloccare la pagina. */
export const BENCH_LIMITS = Object.freeze({ text: 4000, name: 200, notes: 40, snapshots: 2000, bits: 1_000_000 });

const METRIC_KEYS = ["loss", "perplexity", "bpc", "top1", "top5", "entropy"];

/**
 * Le righe del confronto, nell'ordine di gpt/benchmark.py (_ROWS). `direction`: −1 più basso è meglio,
 * +1 più alto è meglio, 0 nessun verdetto (l'entropia non ha un verso; le velocità dipendono dalla
 * macchina; i parametri sono una scelta di progetto).
 */
export const METRIC_ROWS = Object.freeze([
  { key: "loss", label: "loss (nat)", format: "fixed4", direction: -1 },
  { key: "perplexity", label: "perplexity", format: "fixed3", direction: -1 },
  { key: "bpc", label: "bit/carattere", format: "fixed3", direction: -1 },
  { key: "top1", label: "top-1", format: "pct", direction: 1 },
  { key: "top5", label: "top-5", format: "pct", direction: 1 },
  { key: "entropy", label: "entropia (nat)", format: "fixed3", direction: 0 },
  { key: "params", label: "parametri", format: "int", direction: 0 },
  { key: "tokens_per_s", label: "token/s*", format: "speed", direction: 0, machine: true },
]);

const isPlain = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const finite = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const count = (v) => (Number.isSafeInteger(v) && v >= 0 ? v : null);
const text = (v, max = BENCH_LIMITS.text) => (typeof v === "string" ? v.slice(0, max) : null);
const field = (obj, key) => (isPlain(obj) && Object.hasOwn(obj, key) ? obj[key] : undefined);

/** Le metriche note di un oggetto (validation o estratto): solo numeri finiti, il resto null. */
function readMetrics(obj, problems, where) {
  const out = {};
  for (const key of METRIC_KEYS) {
    const v = field(obj, key);
    out[key] = finite(v);
    if (v !== undefined && out[key] === null) problems.push(`${where}.${key} non è un numero valido`);
  }
  for (const key of ["predictions", "params"]) out[key] = count(field(obj, key));
  for (const key of ["tokens_per_s", "eval_tokens_per_s", "duration_s"]) out[key] = finite(field(obj, key));
  // le quote top-1/top-5 stanno tra 0 e 1: fuori, il file è rotto (o in percentuale: non indovino)
  for (const key of ["top1", "top5"]) {
    if (out[key] !== null && (out[key] < 0 || out[key] > 1)) {
      problems.push(`${where}.${key} = ${out[key]} fuori da 0…1`);
      out[key] = null;
    }
  }
  return out;
}

function readCounts(obj) {
  if (!isPlain(obj)) return null;
  const out = {};
  for (const key of ["predictions", "windows", "chars", "top1", "top5"]) out[key] = count(field(obj, key));
  return Object.values(out).some((v) => v !== null) ? out : null;
}

function readExcerpt(obj, problems, where) {
  if (!isPlain(obj)) return null;
  const bits = field(obj, "surprisal_bits");
  let surprisal = null;
  if (Array.isArray(bits)) {
    if (bits.length > BENCH_LIMITS.bits) problems.push(`${where}.surprisal_bits troppo lungo: ignorato`);
    else if (bits.every((b) => typeof b === "number" && Number.isFinite(b))) surprisal = bits;
    else problems.push(`${where}.surprisal_bits contiene valori non numerici: ignorato`);
  }
  return {
    ...readMetrics(obj, problems, where),
    file: text(field(obj, "file"), 200),
    chars: count(field(obj, "chars")),
    sha256: text(field(obj, "sha256"), 64),
    counts: readCounts(field(obj, "counts")),
    surprisalBits: surprisal,
  };
}

/**
 * benchmark.json (o builtin.json) normalizzato. Non lancia mai: un campo rotto diventa null e finisce
 * in `problems` (in italiano). `ok` dice se c'è almeno qualcosa da mostrare.
 * @param {unknown} raw il JSON letto
 * @returns {{ok: boolean, problems: string[], name: string|null, created: string|null, source: string|null,
 *   protocol: {id: string|null, blockSize: number|null}, data: {sha256: string|null, chars: number|null},
 *   model: {params: number|null, vocabSize: number|null, weightsSha256: string|null, codeSha256: string|null},
 *   metrics: object, counts: object|null, excerpt: object|null, generation: object|null,
 *   snapshots: {step: number, metrics: object, excerpt: object|null}[], reference: object|null,
 *   environment: object|null, notes: string[]}}
 */
export function readBenchmark(raw) {
  const problems = [];
  if (!isPlain(raw)) {
    return { ok: false, problems: ["benchmark.json non è un oggetto JSON"], name: null, created: null, source: null, protocol: { id: null, blockSize: null }, data: { sha256: null, chars: null }, model: { params: null, vocabSize: null, weightsSha256: null, codeSha256: null }, metrics: readMetrics({}, [], "metrics"), counts: null, excerpt: null, generation: null, snapshots: [], reference: null, environment: null, notes: [] };
  }
  const protocol = field(raw, "protocol");
  const data = field(raw, "data");
  const model = field(raw, "model");
  // metriche in "metrics" (benchmark.py); per i file a mano anche al primo livello (come parseBenchmark)
  const metrics = readMetrics(isPlain(field(raw, "metrics")) ? raw.metrics : raw, problems, "metrics");
  const generation = field(raw, "generation");
  const snapshots = [];
  const seen = new Set();
  for (const [i, s] of (Array.isArray(field(raw, "snapshots")) ? raw.snapshots : []).slice(0, BENCH_LIMITS.snapshots).entries()) {
    const step = count(field(s, "step"));
    if (step === null || seen.has(step)) {
      problems.push(`snapshots[${i}]: step mancante o ripetuto`);
      continue;
    }
    seen.add(step);
    snapshots.push({ step, metrics: readMetrics(field(s, "metrics"), problems, `snapshots[${i}].metrics`), excerpt: readExcerpt(field(s, "excerpt"), problems, `snapshots[${i}].excerpt`) });
  }
  snapshots.sort((a, b) => a.step - b.step);
  const env = field(raw, "environment");
  const environment = isPlain(env)
    ? Object.fromEntries(["python", "torch", "numpy", "device", "system", "machine"].map((k) => [k, text(field(env, k), 80)]).concat([["threads", count(field(env, "threads"))]]))
    : null;
  const ref = field(raw, "reference");
  const bench = {
    ok: true,
    problems,
    name: text(field(raw, "name"), BENCH_LIMITS.name),
    created: text(field(raw, "created"), 100),
    source: text(field(raw, "source"), 500),
    protocol: { id: text(field(protocol, "id"), 100), blockSize: count(field(protocol, "block_size")) },
    data: { sha256: text(field(data, "sha256"), 64), chars: count(field(data, "chars")) },
    model: { params: count(field(model, "params")), vocabSize: count(field(model, "vocab_size")), weightsSha256: text(field(model, "weights_sha256"), 64), codeSha256: text(field(model, "code_sha256"), 64) },
    metrics: { ...metrics, params: metrics.params ?? count(field(model, "params")) },
    counts: readCounts(field(raw, "counts")),
    excerpt: readExcerpt(field(raw, "excerpt"), problems, "excerpt"),
    generation: isPlain(generation)
      ? {
          prompt: text(field(generation, "prompt"), 500),
          seed: Number.isSafeInteger(field(generation, "seed")) ? generation.seed : null,
          temperature: finite(field(generation, "temperature")),
          maxNewTokens: count(field(generation, "max_new_tokens")),
          sampled: text(field(generation, "sampled")),
          greedy: text(field(generation, "greedy")),
          tokensPerS: finite(field(generation, "tokens_per_s")),
        }
      : null,
    snapshots,
    reference: isPlain(ref) ? { uniformLoss: finite(field(ref, "uniform_loss")), uniformBpc: finite(field(ref, "uniform_bpc")) } : null,
    environment,
    notes: (Array.isArray(field(raw, "notes")) ? raw.notes : []).filter((n) => typeof n === "string").slice(0, BENCH_LIMITS.notes).map((n) => n.slice(0, 2000)),
  };
  if (bench.protocol.id && bench.protocol.id !== PROTOCOL_ID) problems.push(`protocollo ${bench.protocol.id}: il Laboratorio conosce ${PROTOCOL_ID}`);
  bench.ok = METRIC_KEYS.some((k) => bench.metrics[k] !== null) || Boolean(bench.excerpt) || snapshots.length > 0 || Boolean(bench.generation);
  if (!bench.ok) problems.push("nessuna metrica riconosciuta (loss, top1, …)");
  return bench;
}

/**
 * I numeri del Python per un insieme di pesi: i pesi finali (step null) o uno snapshot. null se il
 * benchmark non ha quello step (es. un passo interpolato, mai misurato).
 * @returns {{step: number|null, source: "final"|"snapshot", metrics: object, counts: object|null,
 *   excerpt: object|null, greedy: string|null, sampled: string|null, bench: object}|null}
 */
export function referenceAt(bench, step = null) {
  if (!bench?.ok) return null;
  if (step === null || step === undefined) {
    return { step: null, source: "final", metrics: bench.metrics, counts: bench.counts, excerpt: bench.excerpt, greedy: bench.generation?.greedy ?? null, sampled: bench.generation?.sampled ?? null, bench };
  }
  const s = bench.snapshots.find((x) => x.step === step);
  if (!s) return null;
  return { step, source: "snapshot", metrics: s.metrics, counts: null, excerpt: s.excerpt, greedy: null, sampled: null, bench };
}

/**
 * Le righe del confronto A/B: `{key, label, a, b, delta (A − B), better: "A" | "B" | "=" | null}`.
 * Le velocità non hanno mai un verdetto (dipendono dalla macchina), come in gpt/benchmark.py.
 */
export function comparisonRows(refA, refB) {
  return METRIC_ROWS.map(({ key, label, format, direction, machine }) => {
    const a = metricOf(refA, key);
    const b = metricOf(refB, key);
    const delta = a !== null && b !== null ? a - b : null;
    let better = null;
    if (delta !== null && direction) better = Math.abs(delta) < 1e-12 ? "=" : delta * direction > 0 ? "A" : "B";
    return { key, label, format, direction, machine: Boolean(machine), a, b, delta, better };
  });
}

function metricOf(ref, key) {
  if (!ref) return null;
  const v = ref.metrics?.[key];
  if (key === "params") return v ?? ref.bench?.model?.params ?? null;
  return typeof v === "number" ? v : null;
}

/**
 * Avvisi di confrontabilità tra due benchmark (come comparison_warnings): `[{level: "attenzione" | "nota", text}]`.
 */
export function comparisonWarnings(a, b) {
  const out = [];
  if (!a?.ok || !b?.ok) return out;
  if (a.protocol.id !== b.protocol.id) out.push({ level: "attenzione", text: `protocolli diversi (${a.protocol.id ?? "?"} contro ${b.protocol.id ?? "?"}): i numeri non sono confrontabili` });
  if (a.data.sha256 !== b.data.sha256) out.push({ level: "attenzione", text: "testi di validation diversi: i numeri non sono confrontabili" });
  if (a.protocol.blockSize !== b.protocol.blockSize) out.push({ level: "nota", text: `contesti diversi (block_size ${a.protocol.blockSize ?? "?"} contro ${b.protocol.blockSize ?? "?"}): un contesto più lungo aiuta a prevedere` });
  if (a.model.vocabSize !== b.model.vocabSize) out.push({ level: "attenzione", text: `vocabolari diversi (${a.model.vocabSize ?? "?"} contro ${b.model.vocabSize ?? "?"} token)` });
  if (a.model.weightsSha256 && a.model.weightsSha256 === b.model.weightsSha256) out.push({ level: "nota", text: "stessi pesi, bit per bit" });
  else if (a.model.codeSha256 && b.model.codeSha256 && a.model.codeSha256 !== b.model.codeSha256) out.push({ level: "nota", text: "sorgenti del modello diversi (code_sha256): è un confronto prima/dopo una modifica al codice" });
  const ea = a.environment ?? {};
  const eb = b.environment ?? {};
  if (["system", "machine", "threads", "torch", "device"].some((k) => ea[k] !== eb[k])) out.push({ level: "nota", text: "velocità misurate in condizioni diverse (macchina, thread o versione di torch): non confrontarle" });
  return out;
}

/** La riga dell'ambiente in cui il Python ha misurato (per dire "dichiarato, su un'altra macchina"). */
export function environmentText(bench) {
  const e = bench?.environment;
  if (!e) return "";
  const parts = [e.device, e.system, e.machine].filter(Boolean);
  if (e.threads) parts.push(`${e.threads} thread`);
  if (e.torch) parts.push(`torch ${e.torch}`);
  return parts.join(" · ");
}
