/**
 * Rappresentazione "Rete neurale" del Laboratorio: la rete VERA del run A, disegnata dal renderer del
 * nucleo (js/core/neuralNet/, formato networkSpec in ARCHITECTURE.md). Logica pura (niente DOM, niente
 * three.js): le specifiche si provano in Node.
 *
 * - Run compatibile con il motore JS: buildDepthSpec (js/shared/gpt/depthSpec.js, la stessa rete della vista
 *   "Orizzontale con profondità" del Transformer) con la traccia vera del motore su un prompt d'esempio, con i
 *   pesi del run allo step scelto:
 *   ogni sfera è un neurone con il valore calcolato, ogni arco di attention il peso vero.
 * - Altri run (FFN più larga, GELU dichiarata, moduli nuovi…): una specifica di SOLA STRUTTURA ricavata dai
 *   nomi e dalle forme dei pesi, con i pesi veri delle nn.Linear sui collegamenti; le attivazioni non si
 *   possono calcolare e lo si dice ("valori non disponibili: architettura diversa dal motore JS"): il
 *   renderer le simula con un seme e mostra "simulazione didattica".
 *
 * createNetworkSource(tab) collega la specifica al selettore delle rappresentazioni (vedi in fondo).
 */
import { ATTENTION_SLABS, buildDepthSpec, depthSections, shownChar } from "../../shared/gpt/depthSpec.js";

/** Il prompt d'esempio: l'inizio del prompt del benchmark ("ROMEO:\n", scripts/benchmark.py) e una battuta di Romeo. */
export const NETWORK_PROMPT = "ROMEO:\nBut, soft!";
/** Quota dei pesi più forti di ogni nn.Linear disegnati sulla fetta scelta (come nel tab Transformer). */
export const NETWORK_FRACTION = 0.02;
/** Neuroni per posizione al massimo nella rete di sola struttura (limite del nucleo: 4.096). */
const MAX_NEURONS = 4096;
/** Unità al massimo (limite del nucleo 250.000): con modelli grandi il prompt si accorcia. */
const MAX_UNITS = 240_000;

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });

/** I nomi delle posizioni in profondità: i caratteri del prompt o, senza vocabolario, "p0" … "p7". */
function depthNames(vocab, prompt, max) {
  const ids = encodeWith(vocab, prompt, max);
  return ids.length && Array.isArray(vocab) ? ids.map((id) => shownChar(vocab[id] ?? "?")) : Array.from({ length: Math.min(8, max) }, (_, i) => `p${i}`);
}

/** Il prompt in token del vocabolario del run (i caratteri che il vocabolario non ha si saltano). */
export function encodeWith(vocab, text, max = Infinity) {
  if (!Array.isArray(vocab) || !vocab.length) return [];
  const stoi = new Map(vocab.map((c, i) => [c, i]));
  const ids = [];
  for (const ch of String(text)) {
    const id = stoi.get(ch);
    if (id !== undefined) ids.push(id);
    if (ids.length >= max) break;
  }
  return ids.length ? ids : [0];
}

/**
 * Le misure di un run "come il GPT del corso" (stessi nomi), anche con forme diverse: V, T, C, H, L, F
 * (neuroni nascosti della feed-forward, letti dai pesi). null se mancano i pezzi del corso.
 */
export function courseDims(tensors) {
  const shape = (n) => tensors.get(n)?.shape;
  const tok = shape("token_embedding_table.weight");
  const pos = shape("position_embedding_table.weight");
  const head = shape("lm_head.weight");
  if (tok?.length !== 2 || pos?.length !== 2 || head?.length !== 2) return null;
  let L = 0;
  while (tensors.has(`blocks.${L}.ln1.weight`)) L++;
  let H = 0;
  while (tensors.has(`blocks.0.sa.heads.${H}.key.weight`)) H++;
  const fc1 = shape("blocks.0.ffwd.net.0.weight");
  const key = shape("blocks.0.sa.heads.0.key.weight");
  if (!L || !H || fc1?.length !== 2 || key?.length !== 2) return null;
  const C = tok[1];
  if (pos[1] !== C || key[1] !== C || key[0] * H !== C || fc1[1] !== C) return null;
  for (let l = 0; l < L; l++) {
    for (const n of ["sa.proj.weight", "ffwd.net.0.weight", "ffwd.net.2.weight", "ln2.weight"]) if (!tensors.has(`blocks.${l}.${n}`)) return null;
  }
  return { V: tok[0], T: pos[0], C, H, L, F: fc1[0], hs: key[0] };
}

const config = (d) => ({ vocab_size: d.V, block_size: d.T, n_embd: d.C, n_head: d.H, n_layer: d.L });

/** La rete vera (run compatibile): traccia del motore sul prompt, pesi del motore. */
export function trueSpec({ run, engine, label, prompt = NETWORK_PROMPT, fraction = NETWORK_FRACTION }) {
  const cfg = engine.config;
  const perToken = cfg.vocab_size + cfg.block_size + 3 * cfg.n_embd + cfg.n_layer * 14 * cfg.n_embd + cfg.n_embd + 2 * cfg.vocab_size;
  const T = Math.max(1, Math.min(cfg.block_size, Math.floor(MAX_UNITS / perToken)));
  const ids = encodeWith(engine.vocab, prompt, T);
  const { trace } = engine.forward(ids, { trace: true });
  const spec = buildDepthSpec({ config: cfg, vocab: engine.vocab, trace, param: (name) => engine.paramViews.get(name).data, fraction });
  const shown = ids.map((id) => shownChar(engine.vocab[id])).join("");
  return {
    ...spec,
    title: `Rete neurale vera di «${run.name}»`,
    intro: `I neuroni VERI del run (pesi: ${label}): ogni sfera ha il valore calcolato dal motore JS per il prompt d'esempio «${shown}» (${ids.length} token, uno dietro l'altro in profondità). Di lato leggi il flusso da sinistra a destra; girando la camera vedi l'attention che collega i token.`,
  };
}

/** Una rete di sola struttura: sezioni e pesi veri dai nomi e dalle forme; attivazioni simulate e dichiarate. */
export function structureSpec({ run, label, reason, prompt = NETWORK_PROMPT, fraction = NETWORK_FRACTION }) {
  const dims = courseDims(run.tensors);
  const head = dims ? courseLikeSpec(run, dims, { prompt, fraction }) : chainSpec(run, { prompt, fraction });
  const why = reason ? ` Il motivo: ${reason}.` : "";
  return {
    ...head,
    title: `Rete neurale di «${run.name}»: solo struttura`,
    intro: `Valori non disponibili: architettura diversa dal motore JS.${why} Le sezioni e le loro dimensioni vengono dai pesi del run (${label}); i collegamenti colorati sono i pesi VERI delle nn.Linear. Le attivazioni delle sfere e gli archi di attention sono una simulazione didattica: il browser non sa calcolare questo modello.`,
    note: `${head.note ?? ""} Le attivazioni simulate servono solo a vedere la forma della rete: non sono quelle del modello.`.trim(),
  };
}

/** Struttura del GPT del corso con le misure del run (es. una FFN più larga, un'altra attivazione). */
function courseLikeSpec(run, d, { prompt, fraction }) {
  const names = depthNames(run.vocab, prompt, Math.min(d.T, 24));
  const sections = depthSections({ ...config(d) }).map((s) => {
    const out = { ...s, depth: true };
    delete out.layer;
    if (s.id.endsWith(".hidden")) Object.assign(out, { neurons: Math.min(MAX_NEURONS, d.F), real: d.F > MAX_NEURONS ? d.F : undefined, sub: `${INT.format(d.F)} neuroni nascosti`, activation: "signed", text: `Feed-forward: ${INT.format(d.F)} neuroni nascosti per token (il GPT del corso ne ha 4 × n_embd = ${INT.format(4 * d.C)}). L'attivazione è quella del codice del run: il motore JS non la calcola.` });
    if (s.id.endsWith(".ffnOut")) out.text = `La seconda nn.Linear riporta ${INT.format(d.F)} numeri a ${INT.format(d.C)}, token per token.`;
    if (s.id === "tok_onehot" || s.id === "logits" || s.id === "probs") out.neuronNames = (run.vocab ?? []).map(shownChar);
    return out;
  });
  const links = [
    { from: "tok_onehot", to: "emb", pattern: "lookup", toSlab: 0, label: "lookup nella tabella dei token" },
    { from: "pos_onehot", to: "emb", pattern: "lookup", toSlab: 1, label: "lookup nella tabella delle posizioni" },
    { from: "emb", to: "x0", pattern: "identity", fromSlab: 0, label: "somma: embedding del token" },
    { from: "emb", to: "x0", pattern: "identity", fromSlab: 1, label: "somma: embedding della posizione" },
  ];
  const W = (name) => run.tensors.get(name);
  const explicit = (from, to, name, toIndex, fromIndex, text) => {
    const t = W(name);
    if (!t || t.shape.length !== 2) return;
    links.push({ from, to, pattern: "explicit", pairs: strongest(t, fraction, toIndex, fromIndex), label: `${text} (${name}, pesi veri)` });
  };
  let input = "x0";
  for (let l = 0; l < d.L; l++) {
    const p = `b${l}.`;
    const q = `blocks.${l}.`;
    links.push({ from: input, to: `${p}ln1`, pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
    ATTENTION_SLABS.forEach(({ name, part }, slab) => {
      if (part === null) return;
      for (let h = 0; h < d.H; h++) explicit(`${p}ln1`, `${p}attn`, `${q}sa.heads.${h}.${name}.weight`, (r) => slab * d.C + h * d.hs + r, (c) => c, `W${name[0]} della head ${h}`);
    });
    if (names.length > 1) links.push({ within: `${p}attn`, pattern: "attention", causal: true, heads: d.H, label: "attention causale: pesi simulati (il motore JS non esegue questo run)" });
    explicit(`${p}attn`, `${p}proj`, `${q}sa.proj.weight`, (r) => r, (c) => (ATTENTION_SLABS.length - 1) * d.C + c, "proiezione");
    links.push({ from: `${p}proj`, to: `${p}resid1`, pattern: "identity", label: "somma: uscita dell'attention" });
    links.push({ from: input, to: `${p}resid1`, pattern: "residual", label: "connessione residua (x + attention)" });
    links.push({ from: `${p}resid1`, to: `${p}ln2`, pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
    if (d.F <= MAX_NEURONS) {
      explicit(`${p}ln2`, `${p}hidden`, `${q}ffwd.net.0.weight`, (r) => r, (c) => c, "feed-forward, primo strato");
      explicit(`${p}hidden`, `${p}ffnOut`, `${q}ffwd.net.2.weight`, (r) => r, (c) => c, "feed-forward, secondo strato");
    }
    links.push({ from: `${p}ffnOut`, to: `${p}resid2`, pattern: "identity", label: "somma: uscita della feed-forward" });
    links.push({ from: `${p}resid1`, to: `${p}resid2`, pattern: "residual", label: "connessione residua (x + feed-forward)" });
    input = `${p}resid2`;
  }
  links.push({ from: input, to: "ln_f", pattern: "identity", label: "LayerNorm: ogni token per conto suo" });
  explicit("ln_f", "logits", "lm_head.weight", (r) => r, (c) => c, "lm_head");
  links.push({ from: "logits", to: "probs", pattern: "identity", label: "softmax" });
  return {
    scale: `1 sfera = 1 neurone del run: ${INT.format(sections.reduce((s, x) => s + x.neurons * (x.slabs ?? 1), 0))} neuroni per token × ${names.length} token.`,
    note: `I collegamenti colorati sono i pesi veri più forti (${(fraction * 100).toLocaleString("it-IT")}% di ogni matrice) sulla fetta scelta.`,
    seed: 3,
    axes: { x: "flusso: dall'ingresso all'uscita", y: `neuroni della sezione (${INT.format(d.C)} per il vettore x)`, z: `posizioni del prompt d'esempio (${names.length} token)` },
    depth: { count: names.length, label: "token", names },
    sections,
    links,
    groups: [
      { label: "Ingresso", from: "tok_onehot", to: "x0", text: "Il carattere e la sua posizione diventano il vettore x." },
      ...Array.from({ length: d.L }, (_, l) => ({ label: `Blocco ${l}`, from: `b${l}.ln1`, to: `b${l}.resid2`, text: `Blocco transformer ${l}: attention e feed-forward con le loro connessioni residue.` })),
      { label: "Uscita", from: "ln_f", to: "probs", text: "LayerNorm finale, punteggi e probabilità del prossimo carattere." },
    ],
    sources: [{ title: "Karpathy, ng-video-lecture (gpt.py)", url: "https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py", note: "Il modello del corso, da cui vengono i nomi dei moduli." }],
  };
}

/** Architettura sconosciuta: i moduli con i pesi in fila, nell'ordine dei parametri (= di __init__). */
function chainSpec(run, { prompt, fraction }) {
  const modules = new Map();
  for (const [name, t] of run.tensors) {
    const path = name.split(".").slice(0, -1).join(".") || name;
    if (!modules.has(path)) modules.set(path, []);
    modules.get(path).push(t);
  }
  const names = depthNames(run.vocab, prompt, 12);
  const sections = [];
  const links = [];
  let prev = null;
  for (const [path, params] of modules) {
    if (sections.length >= 60) break; // una rete leggibile: i primi 60 moduli
    const w = params.find((p) => p.shape.length === 2) ?? params[0];
    const out = w.shape.length === 2 ? w.shape[0] : w.shape[0] ?? 1;
    const neurons = Math.max(1, Math.min(MAX_NEURONS, out));
    const id = `m${sections.length}`;
    const role = /emb|wte|wpe/.test(path) ? "embedding" : /norm|ln/.test(path) ? "norm" : /attn|attention|heads|query|key|value/.test(path) ? "attention" : /lm_head|output/.test(path) ? "output" : "mlp";
    sections.push({ id, label: path.split(".").slice(-2).join("."), sub: `(${w.shape.join(", ")})`, role, neurons, depth: true, real: out > MAX_NEURONS ? out : undefined, text: `Il modulo ${path}: ${params.map((p) => `${p.name} (${p.shape.join(", ")})`).join(", ")}.` });
    if (prev) {
      const inDim = w.shape.length === 2 ? w.shape[1] : null;
      if (inDim === prev.neurons && w.shape[0] <= MAX_NEURONS) links.push({ from: prev.id, to: id, pattern: "explicit", pairs: strongest(w, fraction, (r) => r, (c) => c), label: `${w.name} (pesi veri)` });
      else links.push({ from: prev.id, to: id, pattern: "dense", density: 0.1, label: "collegamento di forma (pesi non disegnati: dimensioni non consecutive)" });
    }
    prev = { id, neurons };
  }
  return {
    scale: `1 sfera = 1 neurone d'uscita di ogni modulo (al più ${INT.format(MAX_NEURONS)}); ${INT.format(sections.length)} moduli in fila nell'ordine dei pesi.`,
    note: "Architettura sconosciuta: l'ordine è quello dei parametri, che di solito segue il codice (__init__) ma non è garantito che sia il passaggio in avanti.",
    seed: 5,
    axes: { y: "neuroni d'uscita del modulo", z: `posizioni del prompt d'esempio (${names.length} token)` },
    depth: { count: names.length, label: "token", names },
    sections,
    links,
  };
}

/** I pesi più forti (|w|) di una matrice (out × in) come coppie "explicit": ingresso → uscita, peso vero. */
export function strongest(t, fraction, toIndex, fromIndex) {
  const [rows, cols] = t.shape;
  const W = t.data;
  const k = Math.min(W.length, Math.max(1, Math.round(W.length * fraction)));
  // selezione dei k più forti senza ordinare tutto: soglia per approssimazioni successive (rapida e deterministica)
  const order = Array.from({ length: W.length }, (_, i) => i).sort((a, b) => Math.abs(W[b]) - Math.abs(W[a]) || a - b).slice(0, k);
  const from = new Int32Array(k);
  const to = new Int32Array(k);
  const weight = new Float32Array(k);
  order.forEach((e, s) => {
    const r = Math.floor(e / cols);
    from[s] = fromIndex(e - r * cols);
    to[s] = toIndex(r);
    weight[s] = W[e];
  });
  void rows;
  return { from, to, weight };
}

/** La specifica della rete per il run A allo step attuale (vera se il motore la esegue, altrimenti di struttura). */
export async function networkSpecFor(state) {
  const id = state.runA;
  const { engine, reason, view } = await state.engineFor(id, state.step);
  const run = state.run(id);
  const label = view?.label ?? (await state.weightsAt(id, state.step)).label;
  if (engine) return trueSpec({ run, engine, label });
  return structureSpec({ run, label, reason });
}

/**
 * Il collegamento con core/representations.js: la specifica si costruisce a ogni apertura della Rete
 * neurale (`cache: false`) per il run A e lo step di adesso. Se run A o step cambiano con la rete aperta,
 * `reps.refresh("network")` la rifà sulla stessa scena e con la stessa camera (niente livello WebGL nuovo),
 * poco dopo l'ultimo cambio: il cursore dello step può scorrere.
 */
export function createNetworkSource(tab, { build = networkSpecFor, delay = 450 } = {}) {
  let timer = 0;
  let reps = null;
  return {
    load: () => build(tab.state),
    /** Run o step cambiati: con la Rete neurale aperta la si rifà, quando lo studente si ferma. */
    invalidate() {
      if (reps?.current !== "network") return;
      clearTimeout(timer);
      timer = setTimeout(() => void reps.refresh("network"), delay);
    },
    /** Collega il selettore delle rappresentazioni (l'API di installRepresentations). */
    attach(r) {
      reps = r;
    },
    dispose: () => clearTimeout(timer),
  };
}
