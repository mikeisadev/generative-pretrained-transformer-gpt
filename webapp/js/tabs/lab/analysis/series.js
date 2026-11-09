/**
 * Le serie dei grafici dalla storia del training (trace.json di gpt/recording.py) e dal benchmark
 * (logica pura, testata). Il trace di un run caricato è un dato NON fidato: parseTrace controlla solo
 * il layout dei pesi, le metriche restano quelle del file. Qui ogni lista si legge con cura: solo numeri
 * finiti, step interi, lunghezze diverse tagliate alla più corta, niente eccezioni su campi mancanti.
 *
 *   lossSeries(trace)            → { train: [[step, loss]…], val: […] }   (stime di estimate_loss)
 *   metricSeries(trace, "lr")    → [[step, valore]…]                       (ogni log_every step)
 *   groupFamilies(trace)         → famiglie di gruppi (attention dei blocchi, …) per i grafici per gruppo
 *   snapshotSeries(trace, "top1") → [[step, valore]…]                     (metriche degli snapshot)
 *   attentionEntropy(trace)      → heatmap (blocco·head × step)
 */

const MAX_POINTS = 200_000; // un trace enorme o malevolo non deve bloccare la pagina

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);
const list = (v) => (Array.isArray(v) ? v : []);

/** Coppie [x, y] da due liste parallele: solo x interi ≥ 0 e y finiti (con `positive`, y > 0). */
export function pairs(xs, ys, { positive = false } = {}) {
  const a = list(xs);
  const b = list(ys);
  const n = Math.min(a.length, b.length, MAX_POINTS);
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    if (!Number.isSafeInteger(x) || x < 0 || !isFiniteNumber(y) || (positive && y <= 0)) continue;
    out.push([x, y]);
  }
  return out;
}

/** Loss di train e di validation stimate da estimate_loss (evals): le due curve sovrapposte. */
export function lossSeries(trace) {
  const e = trace?.evals;
  return { train: pairs(e?.steps, e?.train), val: pairs(e?.steps, e?.val) };
}

/** La loss del batch di training a ogni step (train_loss: indice = step), rumorosa per natura. */
export function batchLossSeries(trace) {
  const values = list(trace?.train_loss).slice(0, MAX_POINTS);
  const out = [];
  for (let i = 0; i < values.length; i++) if (isFiniteNumber(values[i])) out.push([i, values[i]]);
  return out;
}

/** Una metrica di `metrics` (lr, grad_norm, update_ratio) contro metrics.steps. */
export function metricSeries(trace, key, { positive = false } = {}) {
  const m = trace?.metrics;
  if (!m || typeof m !== "object" || !Object.hasOwn(m, key)) return [];
  return pairs(m.steps, m[key], { positive });
}

/** Una metrica per gruppo (update_ratio_by_group, weight_std_by_group, grad_norm_by_group) per un gruppo. */
export function groupMetricSeries(trace, key, group, { positive = false } = {}) {
  const m = trace?.metrics;
  const byGroup = m && typeof m === "object" && Object.hasOwn(m, key) ? m[key] : null;
  if (!byGroup || typeof byGroup !== "object" || Array.isArray(byGroup) || !Object.hasOwn(byGroup, group)) return [];
  return pairs(m.steps, byGroup[group], { positive });
}

/** true se tutti i valori sono uguali (es. un learning rate costante: nessuno scheduler). */
export function isConstant(points) {
  return points.length > 1 && points.every((p) => p[1] === points[0][1]);
}

/** Lo step massimo del training (pesi finali), se il trace lo dice. */
export function finalStep(trace) {
  const iters = trace?.train_config?.max_iters;
  if (Number.isSafeInteger(iters) && iters >= 0) return iters;
  let last = null; // un ciclo, non Math.max(...): la lista viene da un file e può essere lunghissima
  for (const s of list(trace?.snapshots)) if (Number.isSafeInteger(s?.step) && s.step >= 0 && (last === null || s.step > last)) last = s.step;
  return last;
}

// ---------------------------------------------------------------- gruppi di pesi

const COURSE_FAMILY = { attention: "attention (head + proj)", ffwd: "feed-forward", layernorm: "le due LayerNorm" };
const OUTSIDE = { token_embedding: "embedding dei token", position_embedding: "embedding delle posizioni", ln_f: "ln_f", lm_head: "lm_head" };
const MAX_SERIES = 8; // la palette dei grafici ha 8 colori: oltre servirebbero grigi indistinguibili

/**
 * I gruppi del trace raccolti in famiglie, per grafici leggibili (al più 8 serie ciascuno): per il GPT
 * del corso "attention", "feed-forward", "LayerNorm" (una serie per blocco) e "fuori dai blocchi"
 * (embedding, ln_f, lm_head). Per nomi sconosciuti la famiglia è il nome con i numeri al posto di "*"
 * (transformer.h.*.attn), i gruppi senza numeri vanno in "fuori dai blocchi".
 * @returns {{id: string, label: string, members: {group: string, label: string}[], truncated: number}[]}
 */
export function groupFamilies(trace) {
  const groups = list(trace?.groups).filter((g) => typeof g === "string" && g.length <= 200);
  const families = new Map();
  const outside = { id: "outside", label: "fuori dai blocchi", members: [], truncated: 0 };
  for (const group of groups) {
    const course = /^blocks\.(\d+)\.(attention|ffwd|layernorm)$/.exec(group);
    if (course) {
      add(families, course[2], COURSE_FAMILY[course[2]], { group, label: `blocco ${course[1]}` });
      continue;
    }
    if (Object.hasOwn(OUTSIDE, group)) {
      outside.members.push({ group, label: OUTSIDE[group] });
      continue;
    }
    const numbers = group.match(/\d+/g);
    if (numbers) add(families, group.replace(/\d+/g, "*"), group.replace(/\d+/g, "*"), { group, label: numbers.length === 1 ? `blocco ${numbers[0]}` : numbers.join(".") });
    else outside.members.push({ group, label: group });
  }
  const out = [...families.values()];
  if (outside.members.length) out.push(outside);
  for (const f of out) {
    if (f.members.length > MAX_SERIES) {
      // primi e ultimi, così si vedono l'inizio e la fine della pila di blocchi
      const keep = [...f.members.slice(0, MAX_SERIES - 2), ...f.members.slice(-2)];
      f.truncated = f.members.length - keep.length;
      f.members = keep;
    }
  }
  return out;
}

function add(families, id, label, member) {
  if (!families.has(id)) families.set(id, { id, label, members: [], truncated: 0 });
  families.get(id).members.push(member);
}

// ---------------------------------------------------------------- snapshot

/** Una metrica degli snapshot (loss, top1, top5, entropy) per step, in ordine. */
export function snapshotSeries(trace, key) {
  const out = [];
  for (const s of list(trace?.snapshots)) {
    if (!s || typeof s !== "object" || !Object.hasOwn(s, key)) continue;
    if (Number.isSafeInteger(s.step) && s.step >= 0 && isFiniteNumber(s[key])) out.push([s.step, s[key]]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}

/**
 * L'entropia dell'attention di ogni head (snapshots[i].attention_entropy[blocco][head]) come heatmap:
 * una riga per (blocco, head), una colonna per snapshot. null se il trace non la ha.
 * @returns {{rows: number, cols: number, values: Float32Array, rowLabels: string[], colLabels: string[], steps: number[]}|null}
 */
export function attentionEntropy(trace) {
  const snaps = list(trace?.snapshots)
    .filter((s) => s && Number.isSafeInteger(s.step) && Array.isArray(s.attention_entropy))
    .sort((a, b) => a.step - b.step)
    .slice(0, 400);
  if (!snaps.length) return null;
  // blocchi e head al più 64 ciascuno: una heatmap più grande non si leggerebbe (e il file è dello studente)
  let layers = 0;
  let heads = 0;
  for (const s of snaps) {
    layers = Math.max(layers, Math.min(64, s.attention_entropy.length));
    for (const h of s.attention_entropy.slice(0, 64)) if (Array.isArray(h)) heads = Math.max(heads, Math.min(64, h.length));
  }
  if (!layers || !heads) return null;
  const rows = layers * heads;
  const cols = snaps.length;
  const values = new Float32Array(rows * cols).fill(NaN);
  snaps.forEach((s, c) => {
    for (let l = 0; l < layers; l++) {
      const row = s.attention_entropy[l];
      for (let h = 0; h < heads; h++) {
        const v = Array.isArray(row) ? row[h] : undefined;
        if (isFiniteNumber(v)) values[(l * heads + h) * cols + c] = v;
      }
    }
  });
  const rowLabels = [];
  for (let l = 0; l < layers; l++) for (let h = 0; h < heads; h++) rowLabels.push(`b${l}h${h}`); // corte: il margine delle etichette della heatmap è stretto
  return { rows, cols, values, rowLabels, colLabels: snaps.map((s) => String(s.step)), steps: snaps.map((s) => s.step) };
}

/**
 * La probabilità di un carattere dopo il prompt d'esempio del training, snapshot per snapshot
 * (snapshots[i].example_next_token_probs, calcolate durante il training con i pesi di quel momento).
 */
export function exampleProbSeries(trace, tokenId) {
  const out = [];
  if (!Number.isSafeInteger(tokenId) || tokenId < 0) return out;
  for (const s of list(trace?.snapshots)) {
    const p = Array.isArray(s?.example_next_token_probs) ? s.example_next_token_probs[tokenId] : undefined;
    if (Number.isSafeInteger(s?.step) && isFiniteNumber(p)) out.push([s.step, p]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}

/**
 * Media mobile (centrata a sinistra: il valore i è la media degli ultimi `window` fino a i), per
 * leggere una serie rumorosa come i bit di sorpresa carattere per carattere. Ignora i non finiti.
 * @returns {[number, number][]} coppie [posizione, media]
 */
export function movingAverage(values, window = 64) {
  const v = list(values).slice(0, MAX_POINTS);
  const w = Math.max(1, Math.floor(window));
  const finite = (x) => (isFiniteNumber(x) ? x : null);
  const out = [];
  let sum = 0;
  let count = 0;
  for (let i = 0; i < v.length; i++) {
    const x = finite(v[i]);
    if (x !== null) {
      sum += x;
      count++;
    }
    const old = i >= w ? finite(v[i - w]) : null; // esce dalla finestra
    if (old !== null) {
      sum -= old;
      count--;
    }
    if (count) out.push([i, sum / count]);
  }
  return out;
}

/**
 * Il punto di una serie più vicino a `x` (in una lista vuota null): quando A e B sono due momenti dello
 * stesso training la storia è una sola, e B diventa un punto su quella curva.
 */
export function pointAt(points, x) {
  if (!points?.length || !Number.isFinite(x)) return null;
  let best = points[0];
  for (const p of points) if (Math.abs(p[0] - x) < Math.abs(best[0] - x)) best = p;
  return best;
}

/**
 * Nomi delle serie stabili: "A · train", "B · validation". I colori dei grafici del nucleo si legano al
 * NOME della serie alla sua prima comparsa: nomi fissi = colori fissi, anche cambiando i run.
 */
export const seriesName = (side, what = "") => (what ? `${side} · ${what}` : side);

/** Le serie di A e di B, solo quelle con dei punti. */
export function abSeries(a, b, what = "") {
  const out = [];
  if (a?.length) out.push({ name: seriesName("A", what), points: a });
  if (b?.length) out.push({ name: seriesName("B", what), points: b });
  return out;
}
