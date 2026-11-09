/**
 * Vista "Pesi numerici": la logica pura (niente DOM, niente three.js; test in
 * tests/transformer-numbers.test.mjs).
 *
 *   sorgente dei pesi   pesi finali (float32) oppure uno step del training: lo snapshot vero più vicino
 *                       o, se lo studente lo chiede, la stima lineare tra i due vicini (dichiarata);
 *   etichette oneste    cosa è vero, cosa è arrotondato (float16) e cosa è stimato;
 *   assi delle tabelle  che cosa sono righe e colonne di ogni tensore (caratteri, dimensioni, head…);
 *   elenco dei tensori  raggruppati come il codice Python (GPTLanguageModel → blocks[i] → sa.heads[h]…);
 *   link diretti        tensor:<nome>[@<step>] (anche verso il Laboratorio pesi);
 *   "pesi che nascono"  il calendario dell'animazione sugli snapshot.
 */
import { pythonPath } from "../../../shared/weights/moduleTree.js";
import { snapshotBracket } from "../../../shared/weights/run.js";

// gli id della vista stanno nella registrazione leggera (numbers.js): una sola fonte
export { BIRTH_VIEW_ID, NUMBERS_VIEW_ID } from "./numbers.js";
/** Le carte di una head nella scena, dalla più vicina alla camera (architecture.js: z 0,25 · 0 · −0,25). */
export const CARD_KINDS = Object.freeze(["query", "key", "value"]);

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
/** Gli step si scrivono senza separatore delle migliaia: sono nomi (max_iters 5000, snap_05000), non quantità. */
const STEP = (n) => String(n);
const pct = (t) => `${Math.round(t * 100)}%`;
/** Spazio e a capo visibili: il GPT del corso lavora a caratteri e " " e "\n" sono token veri. */
export const shownChar = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c === "\t" ? "⇥" : c);

// ---------------------------------------------------------------- sorgente dei pesi

/** Lo snapshot più vicino a `step` (a metà strada vince il precedente). `steps` in ordine crescente. */
export function snapToSnapshot(step, steps) {
  if (!steps?.length) return null;
  let best = steps[0];
  for (const s of steps) if (Math.abs(s - step) < Math.abs(best - step)) best = s;
  return best;
}

/** Pesi finali del checkpoint (data/weights.f32.bin), gli stessi che usa il motore. */
export const FINAL_SOURCE = Object.freeze({ kind: "final" });

/**
 * La sorgente dei pesi per uno step scelto con il cursore del training.
 * Senza interpolazione lo step si aggancia allo snapshot più vicino (è l'unico dato vero); con
 * l'interpolazione uno step tra due snapshot diventa la media pesata dei due (una STIMA: il training
 * vero non va in linea retta tra due snapshot).
 * @param {number} step
 * @param {number[]} steps step degli snapshot, crescenti (0, 50, …, 5000)
 * @returns {{kind: "snapshot", step: number} | {kind: "interpolated", step: number, from: [number, number], t: number}}
 */
export function trainingSource(step, steps, { interpolate = false } = {}) {
  if (!steps?.length) throw new Error("nessuno snapshot del training");
  const s = clamp(Math.round(Number(step) || 0), steps[0], steps[steps.length - 1]);
  if (!interpolate) return { kind: "snapshot", step: snapToSnapshot(s, steps) };
  const b = snapshotBracket(steps, s);
  if (b.t === 0) return { kind: "snapshot", step: b.lo };
  return { kind: "interpolated", step: s, from: [b.lo, b.hi], t: b.t };
}

/** Due sorgenti uguali? (serve a non riscrivere 94 heatmap per lo stesso step). */
export function sameSource(a, b) {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === "final") return true;
  if (a.kind === "snapshot") return a.step === b.step;
  return a.from[0] === b.from[0] && a.from[1] === b.from[1] && Math.abs(a.t - b.t) < 1e-9;
}

/**
 * Lo snapshot è il peso finale arrotondato a float16? Per ogni valore la differenza deve stare entro
 * mezzo "passo" del float16 vicino a quel valore (2^(e − 11) per |w| in [2^e, 2^(e+1)), 2^−25 sotto
 * 2^−14, dove il float16 è subnormale). È così che si scopre, dai dati e non da una promessa, se lo
 * snapshot dello step finale è davvero il checkpoint usato dalla web app.
 * @param {Float32Array} final pesi finali float32
 * @param {Float32Array} snap snapshot (float16 già convertito in float32)
 * @returns {{matches: boolean, maxAbsDiff: number, count: number, worstIndex: number, beyond: number}}
 *   beyond = quanti valori escono dalla tolleranza dell'arrotondamento
 */
export function float16RoundingCheck(final, snap) {
  if (!final || !snap || final.length !== snap.length) return { matches: false, maxAbsDiff: NaN, count: final?.length ?? 0, worstIndex: -1, beyond: NaN };
  const bits = new Uint32Array(1);
  const view = new Float32Array(bits.buffer);
  let maxAbsDiff = 0;
  let worstIndex = -1;
  let beyond = 0;
  for (let i = 0; i < final.length; i++) {
    const w = final[i];
    const d = Math.abs(snap[i] - w);
    if (d > maxAbsDiff) {
      maxAbsDiff = d;
      worstIndex = i;
    }
    view[0] = w;
    const e = ((bits[0] >>> 23) & 0xff) - 127; // esponente binario di |w| (float32)
    const half = e < -14 ? 2 ** -25 : 2 ** (e - 11);
    if (!(d <= half * (1 + 1e-7))) beyond++;
  }
  return { matches: beyond === 0, maxAbsDiff, count: final.length, worstIndex, beyond };
}

/**
 * Etichetta onesta della sorgente: titolo (aria-valuetext del cursore, suggerimento), testo breve
 * (scheda, etichetta della cella), `tag` (il cartellino accanto al cursore nella barra dello stage: lo
 * studente vede quale step guarda e se è una stima anche con la sidebar chiusa) e frase (pannello).
 * `lastCheck` = risultato di float16RoundingCheck tra l'ultimo snapshot e i pesi finali (o null se
 * non ancora calcolato: allora non si promette niente).
 * @returns {{title: string, short: string, tag: string, detail: string, estimate: boolean}}
 */
export function sourceLabel(source, { steps = [], lastCheck = null, format = (v) => String(v) } = {}) {
  const last = steps[steps.length - 1];
  if (!source || source.kind === "final") {
    return {
      title: "Pesi finali (float32)",
      short: "Pesi finali",
      tag: "Pesi finali",
      detail: "I numeri esatti del checkpoint (data/weights.f32.bin): gli stessi che usa il motore della web app per generare testo. Float32: circa 7 cifre significative.",
      estimate: false,
    };
  }
  if (source.kind === "interpolated") {
    const [lo, hi] = source.from;
    return {
      title: `Step ${STEP(source.step)}: stima lineare tra ${STEP(lo)} e ${STEP(hi)}`,
      short: `≈ step ${STEP(source.step)} (stima)`,
      tag: `≈ ${STEP(source.step)} stima`,
      detail: `Non è un vero step del training: è la media pesata degli snapshot ${STEP(lo)} (${pct(1 - source.t)}) e ${STEP(hi)} (${pct(source.t)}). Il training vero non va in linea retta tra due snapshot: la stima serve solo a vedere il cambiamento in modo continuo.`,
      estimate: true,
    };
  }
  const step = source.step;
  const f16 = "Snapshot salvato in float16 (circa 3 cifre significative): le cifre oltre la terza vengono dall'arrotondamento a 16 bit.";
  if (step === last && lastCheck) {
    if (lastCheck.matches) {
      return {
        title: `Step ${STEP(step)} (float16: i pesi finali arrotondati)`,
        short: `Step ${STEP(step)} (pesi finali in float16)`,
        tag: `Step ${STEP(step)}`,
        detail: `Verificato sui file: ognuno dei ${INT.format(lastCheck.count)} valori è il peso finale arrotondato a 16 bit (scarto massimo ${format(lastCheck.maxAbsDiff)}). È l'ultimo snapshot del training che ha prodotto i pesi della web app.`,
        estimate: false,
      };
    }
    return {
      title: `Step ${STEP(step)} (float16)`,
      short: `Step ${STEP(step)}`,
      tag: `Step ${STEP(step)}`,
      detail: `Attenzione: questo snapshot NON coincide con i pesi finali del checkpoint (${INT.format(lastCheck.beyond)} valori su ${INT.format(lastCheck.count)} differiscono più dell'arrotondamento a 16 bit, scarto massimo ${format(lastCheck.maxAbsDiff)}). ${f16}`,
      estimate: false,
    };
  }
  if (step === steps[0]) {
    return {
      title: `Step ${STEP(step)}: pesi iniziali (float16)`,
      short: `Step ${STEP(step)} (inizio)`,
      tag: `Step ${STEP(step)}`,
      detail: `Prima del training: numeri casuali scelti dall'inizializzazione di PyTorch. Il modello non sa ancora niente. ${f16}`,
      estimate: false,
    };
  }
  return {
    title: `Step ${STEP(step)} (float16)`,
    short: `Step ${STEP(step)}`,
    tag: `Step ${STEP(step)}`,
    detail: `Snapshot vero del training, registrato allo step ${STEP(step)} di ${STEP(last)}. ${f16}`,
    estimate: false,
  };
}

// ---------------------------------------------------------------- righe e colonne di ogni tensore

const LAYER = String.raw`blocks\.(\d+)\.`;
const PATTERNS = [
  ["token_embedding", /^token_embedding_table\.weight$/],
  ["position_embedding", /^position_embedding_table\.weight$/],
  ["head", new RegExp(`^${LAYER}sa\\.heads\\.(\\d+)\\.(query|key|value)\\.weight$`)],
  ["proj", new RegExp(`^${LAYER}sa\\.proj\\.weight$`)],
  ["proj_bias", new RegExp(`^${LAYER}sa\\.proj\\.bias$`)],
  ["ln_gamma", new RegExp(`^(?:${LAYER}(ln1|ln2)|ln_f)\\.weight$`)],
  ["ln_beta", new RegExp(`^(?:${LAYER}(ln1|ln2)|ln_f)\\.bias$`)],
  ["ffn_in", new RegExp(`^${LAYER}ffwd\\.net\\.0\\.weight$`)],
  ["ffn_in_bias", new RegExp(`^${LAYER}ffwd\\.net\\.0\\.bias$`)],
  ["ffn_out", new RegExp(`^${LAYER}ffwd\\.net\\.2\\.weight$`)],
  ["ffn_out_bias", new RegExp(`^${LAYER}ffwd\\.net\\.2\\.bias$`)],
  ["lm_head", /^lm_head\.weight$/],
  ["lm_head_bias", /^lm_head\.bias$/],
];

/** Tipo di tensore del GPT del corso (null per un nome che il codice del corso non crea). */
export function tensorKind(name) {
  for (const [kind, re] of PATTERNS) if (re.test(name)) return kind;
  return null;
}

/** Carta di una head (query/key/value): { layer, head, kind } o null. */
export function headCard(name) {
  const m = /^blocks\.(\d+)\.sa\.heads\.(\d+)\.(query|key|value)\.weight$/.exec(name);
  return m ? { layer: Number(m[1]), head: Number(m[2]), kind: m[3] } : null;
}

/**
 * Nuove profondità (z) delle tre carte di una head quando una passa davanti: la scelta prende la z più
 * vicina alla camera, le altre due seguono nel loro ordine di prima (le posizioni restano le stesse tre).
 * @param {Record<"query"|"key"|"value", number>} original z originali delle carte
 * @param {"query"|"key"|"value"} chosen
 * @returns {Record<"query"|"key"|"value", number>}
 */
export function cardDepths(original, chosen) {
  const kinds = Object.keys(original);
  const zs = kinds.map((k) => original[k]).sort((a, b) => b - a); // dalla più vicina alla camera (+z)
  const order = [chosen, ...kinds.filter((k) => k !== chosen).sort((a, b) => original[b] - original[a])];
  return Object.fromEntries(order.map((k, i) => [k, zs[i]]));
}

/** Che cosa normalizza una LayerNorm: "ln1 del blocco 2", "ln_f". */
function normName(name) {
  const m = /^blocks\.(\d+)\.(ln1|ln2)\./.exec(name);
  return m ? `${m[2]} del blocco ${m[1]}` : "ln_f";
}

/**
 * Che cosa sono righe e colonne di un tensore, con le etichette da mostrare (le stesse nella tabella,
 * nel suggerimento della scena e nel link). Righe e colonne sono quelle della scena e di PyTorch: un
 * nn.Linear (out, in) ha una riga per uscita; i vettori (bias, γ, β) sono una colonna.
 * @param {string} name nome PyTorch del tensore
 * @param {number[]} shape
 * @param {{vocab?: string[], config?: {n_embd: number, n_head: number}}} [o]
 * @returns {{kind: string|null, rowTitle: string, colTitle: string, rowLabel: (i: number) => string, colLabel: (j: number) => string, meaning: string, vector: boolean}}
 */
export function tensorAxes(name, shape, { vocab = [], config = {} } = {}) {
  const kind = tensorKind(name);
  const vector = shape.length === 1;
  const C = config.n_embd ?? shape[shape.length - 1];
  const hs = config.n_head ? C / config.n_head : 16;
  const index = (i) => String(i);
  const char = (i) => `'${shownChar(vocab[i] ?? "?")}'`;
  const axes = (rowTitle, colTitle, meaning, rowLabel = index, colLabel = index) => ({ kind, rowTitle, colTitle: vector ? "" : colTitle, rowLabel, colLabel: vector ? index : colLabel, meaning, vector });
  switch (kind) {
    case "token_embedding":
      return axes("carattere", "dimensione dell'embedding", `Una riga per ognuno dei ${shape[0]} caratteri del vocabolario: la riga di 'a' è il vettore di ${shape[1]} numeri che rappresenta 'a' quando entra nel modello.`, char);
    case "position_embedding":
      return axes("posizione nel contesto", "dimensione dell'embedding", `Una riga per ognuna delle ${shape[0]} posizioni del contesto: si somma al vettore del carattere che sta in quella posizione.`);
    case "head": {
      const { layer, head, kind: part } = headCard(name);
      const letter = part[0];
      const role = { query: "cosa cerca il token", key: "cosa offre il token", value: "cosa passa il token" }[part];
      return axes(`dimensione della ${part} (uscita della head)`, "dimensione d'ingresso (x dopo ln1)", `Head ${head} del blocco ${layer}: ogni riga calcola una delle ${shape[0]} componenti della ${part} (${role}) come somma pesata dei ${shape[1]} numeri del token.`, (i) => `${letter}${i}`);
    }
    case "proj":
      return axes("dimensione d'uscita (verso il residuo)", "ingresso: le head concatenate", `Rimescola le uscite delle head: la colonna h1·5 è la componente 5 della head 1; ogni riga produce una delle ${shape[0]} dimensioni che si sommano al residuo.`, index, (j) => `h${Math.floor(j / hs)}·${j % hs}`);
    case "proj_bias":
      return axes("dimensione d'uscita", "", `Un numero per ognuna delle ${shape[0]} uscite della proiezione, sommato alla fine.`);
    case "ln_gamma":
      return axes("dimensione (γ: quanto la scala)", "", `γ della LayerNorm (${normName(name)}): dopo aver riportato il vettore a media 0 e varianza 1, la dimensione j viene moltiplicata per γ[j].`);
    case "ln_beta":
      return axes("dimensione (β: quanto la sposta)", "", `β della LayerNorm (${normName(name)}): si somma alla dimensione j dopo la moltiplicazione per γ.`);
    case "ffn_in":
      return axes("neurone nascosto", "dimensione d'ingresso (x dopo ln2)", `Primo strato del feed-forward: ognuno dei ${shape[0]} neuroni nascosti somma i ${shape[1]} numeri del token con i suoi pesi (una riga), poi la ReLU azzera i negativi.`);
    case "ffn_in_bias":
      return axes("neurone nascosto", "", `Il bias di ognuno dei ${shape[0]} neuroni nascosti: il punto di partenza della somma, prima della ReLU.`);
    case "ffn_out":
      return axes("dimensione d'uscita (verso il residuo)", "neurone nascosto", `Secondo strato del feed-forward: riporta i ${shape[1]} neuroni nascosti a ${shape[0]} numeri; la colonna j dice come il neurone j spinge ogni uscita.`);
    case "ffn_out_bias":
      return axes("dimensione d'uscita", "", `Il bias di ognuna delle ${shape[0]} uscite del feed-forward.`);
    case "lm_head":
      return axes("carattere in uscita (un logit per riga)", "dimensione d'ingresso (x dopo ln_f)", `Una riga per carattere: il punteggio (logit) di 'a' è la somma dei ${shape[1]} numeri finali del token pesati con la riga di 'a'.`, char);
    case "lm_head_bias":
      return axes("carattere in uscita", "", "Il bias del punteggio di ogni carattere: quanto il modello lo preferisce prima ancora di guardare il testo.", char);
    default:
      return axes("riga", "colonna", "Tensore senza un significato noto per il GPT del corso.");
  }
}

/** "riga 39 = carattere 'a' · colonna 3 = dimensione dell'embedding 3" (un vettore: solo la riga). */
export function cellMeaning(axes, row, col) {
  const rowText = axes.rowLabel(row);
  const r = `riga ${row} = ${axes.rowTitle}${rowText === String(row) ? ` ${row}` : ` ${rowText}`}`;
  if (axes.vector) return r;
  const colText = axes.colLabel(col);
  return `${r} · colonna ${col} = ${axes.colTitle}${colText === String(col) ? ` ${col}` : ` ${colText}`}`;
}

// ---------------------------------------------------------------- elenco dei tensori come il codice

/**
 * I parametri raggruppati come il codice Python che li crea: embedding (GPTLanguageModel), un gruppo per
 * blocco (blocks[i]: ln1, sa.heads[h].key/query/value, sa.proj, ln2, ffwd.net), l'uscita (ln_f, lm_head).
 * Ogni voce: nome PyTorch, percorso Python (blocks[0].sa.heads[2].key.weight), percorso dentro il
 * gruppo, forma e numero di valori. Ordine: quello di named_parameters.
 * @param {{name: string, shape: number[], numel: number}[]} params
 * @returns {{id: string, title: string, code: string, count: number, items: {name: string, path: string, local: string, shape: number[], numel: number}[]}[]}
 */
export function tensorGroups(params) {
  const groups = new Map();
  const group = (id, title, code) => {
    if (!groups.has(id)) groups.set(id, { id, title, code, count: 0, items: [] });
    return groups.get(id);
  };
  for (const p of params) {
    const path = pythonPath(p.name);
    const m = /^blocks\.(\d+)\./.exec(p.name);
    let g;
    if (m) g = group(`blocks.${m[1]}`, `blocks[${m[1]}] · Block`, "self.blocks = nn.Sequential(*[Block(config) …])");
    else if (/^(token|position)_embedding_table\./.test(p.name)) g = group("embedding", "Embedding", "GPTLanguageModel.__init__");
    else if (/^(ln_f|lm_head)\./.test(p.name)) g = group("output", "Uscita", "GPTLanguageModel.__init__");
    else g = group("other", "Altri parametri", "");
    const local = m ? path.replace(/^blocks\[\d+\]\./, "") : path;
    g.items.push({ name: p.name, path, local, shape: [...p.shape], numel: p.numel });
    g.count += p.numel;
  }
  // embedding prima, i blocchi in ordine, l'uscita in fondo (come il forward)
  const order = (id) => (id === "embedding" ? -1 : id === "output" ? 1e6 : id === "other" ? 1e6 + 1 : Number(id.split(".")[1]));
  return [...groups.values()].sort((a, b) => order(a.id) - order(b.id));
}

/** "(65, 64)", "(64,)". */
export const shapeText = (shape) => `(${shape.join(", ")}${shape.length === 1 ? "," : ""})`;

// ---------------------------------------------------------------- link diretti

/**
 * `tensor:<nome>[@<step>]` (il formato che usa anche il Laboratorio pesi). Lo step, se c'è, è un intero
 * nell'intervallo del training: uno step di snapshot è un dato vero, uno step intermedio si mostra come
 * stima lineare (dichiarata). Senza step: i pesi finali.
 * @param {string} id
 * @param {{names: Set<string>|string[], steps: number[]}} o
 * @returns {{name: string, step: number|null, interpolate: boolean}|null} null se il nome o lo step non esistono
 */
export function parseTensorTarget(id, { names, steps = [] }) {
  if (typeof id !== "string" || !id) return null;
  const known = names instanceof Set ? names : new Set(names);
  const at = id.lastIndexOf("@");
  const name = at < 0 ? id : id.slice(0, at);
  if (!known.has(name)) return null;
  if (at < 0) return { name, step: null, interpolate: false };
  const raw = id.slice(at + 1);
  if (!/^\d+$/.test(raw) || !steps.length) return null;
  const step = Number(raw);
  if (step < steps[0] || step > steps[steps.length - 1]) return null;
  return { name, step, interpolate: !steps.includes(step) };
}

/** Il contrario di parseTensorTarget: "blocks.0.ln1.weight@1000" (o solo il nome per i pesi finali). */
export const formatTensorTarget = (name, step = null) => (step === null || step === undefined ? name : `${name}@${step}`);

/** Destinazione del pulsante "Apri nel Laboratorio pesi" per il tensore e la sorgente scelti. */
export const labTarget = (name, source) => ({ kind: "tensor", id: formatTensorTarget(name, !source || source.kind === "final" ? null : source.step) });

// ---------------------------------------------------------------- "Guarda i pesi nascere"

/**
 * Calendario dell'animazione dal rumore iniziale ai pesi finali: un tratto di uguale durata per ogni
 * coppia di snapshot vicini (gli snapshot non sono equidistanti: 0, 50, 100, 250, …), poi una pausa
 * sull'ultimo. Senza interpolazione si resta su ogni snapshot per tutto il suo tratto (solo dati veri);
 * con l'interpolazione lo step scorre in modo continuo (stima tra due snapshot).
 * @param {number[]} steps
 * @param {{interpolate?: boolean, segment?: number, hold?: number}} [o] secondi per tratto e di pausa finale
 * @returns {{duration: number, segment: number, stepAt: (seconds: number) => number, timeOf: (step: number) => number}}
 */
export function birthSchedule(steps, { interpolate = false, segment = 1.1, hold = 1.2 } = {}) {
  const n = steps.length;
  const segments = Math.max(0, n - 1);
  const duration = segments * segment + hold;
  const stepAt = (seconds) => {
    if (!n) return 0;
    const t = Math.max(0, seconds);
    const i = Math.floor(t / segment);
    if (i >= segments) return steps[n - 1];
    if (!interpolate) return steps[i];
    const local = (t - i * segment) / segment;
    return Math.round(steps[i] + (steps[i + 1] - steps[i]) * local);
  };
  /** Il momento del calendario in cui si arriva allo step (per riprendere da dove si era fermi). */
  const timeOf = (step) => {
    if (n < 2 || step <= steps[0]) return 0;
    if (step >= steps[n - 1]) return segments * segment;
    let i = 0;
    while (steps[i + 1] <= step) i++;
    const local = interpolate ? (step - steps[i]) / (steps[i + 1] - steps[i]) : 0;
    return (i + local) * segment;
  };
  return { duration, segment, stepAt, timeOf };
}
