/**
 * Confronto tra due insiemi di pesi: due run (es. il GPT del corso e una versione migliorata) o
 * due fasi dello stesso training (snapshot). Logica pura.
 *
 * Abbinamento: per nome esatto, poi per nome canonico (senza `module.`/`_orig_mod.`). Si
 * confrontano solo i tensori con la stessa forma; gli altri si riportano (`onlyA`, `onlyB`,
 * `shapeMismatch`): con architetture diverse si confronta tutto ciò che si può e si dice il resto.
 *
 * Misure per tensore (A è il riferimento, "prima"):
 * - cambiamento relativo ‖A − B‖ / ‖A‖ (0 = identici, 1 = la differenza è grande quanto A);
 * - similarità del coseno A·B / (‖A‖‖B‖) (1 = stessa direzione, 0 = scorrelati, −1 = opposti);
 * - differenza massima e media in valore assoluto, frazione di pesi cambiati oltre una tolleranza.
 */

import { grouper } from "./groups.js";
import { canonicalName } from "./validate.js";

const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

/** Accetta una Map di tensori, un WeightRun o uno snapshot caricato ({tensors}). */
function tensorsOf(source) {
  if (source instanceof Map) return source;
  if (source?.tensors instanceof Map) return source.tensors;
  throw new TypeError("servono i tensori: una Map nome → {shape, data}, un run o uno snapshot");
}

/**
 * Differenze tra due array della stessa lunghezza (accumuli in float64).
 * Le posizioni con NaN o ±Infinity in A o in B si saltano e si contano in `skipped`.
 */
export function tensorDiff(a, b, tolerance = 0) {
  if (a.length !== b.length) throw new RangeError(`lunghezze diverse: ${a.length} e ${b.length}`);
  let dot = 0;
  let aa = 0;
  let bb = 0;
  let dd = 0;
  let maxAbs = 0;
  let sumAbs = 0;
  let changed = 0;
  let skipped = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      skipped++;
      continue;
    }
    const d = y - x;
    const ad = d < 0 ? -d : d;
    dot += x * y;
    aa += x * x;
    bb += y * y;
    dd += d * d;
    sumAbs += ad;
    if (ad > maxAbs) maxAbs = ad;
    if (ad > tolerance) changed++;
  }
  const n = a.length - skipped;
  const normA = Math.sqrt(aa);
  const normB = Math.sqrt(bb);
  const diffNorm = Math.sqrt(dd);
  return {
    normA,
    normB,
    diffNorm,
    relChange: normA > 0 ? diffNorm / normA : diffNorm > 0 ? Infinity : 0,
    cosine: normA > 0 && normB > 0 ? Math.max(-1, Math.min(1, dot / (normA * normB))) : null,
    maxAbsDiff: maxAbs,
    meanAbsDiff: n ? sumAbs / n : 0,
    changed,
    changedFraction: n ? changed / n : 0,
    skipped,
    // per i riepiloghi di gruppo
    sums: { dot, aa, bb, dd },
  };
}

/**
 * Confronta due insiemi di tensori.
 * @param {Map|object} a riferimento (WeightRun, snapshot o Map nome → {shape, data})
 * @param {Map|object} b
 * @param {{top?: number, tolerance?: number, groupOf?: (name: string) => string, trace?: object}} [o]
 *   `tolerance`: un peso conta come "cambiato" se |B − A| la supera (predefinita 0: qualsiasi cambiamento);
 *   `trace`: per usare i gruppi di trace.json; `groupOf` li sostituisce del tutto
 * @returns {{matched: object[], onlyA: object[], onlyB: object[], shapeMismatch: object[],
 *   groups: object[], totals: object, mostChanged: object[], identical: boolean}}
 */
export function compareTensors(a, b, { top = 10, tolerance = 0, groupOf = null, trace = null } = {}) {
  const A = tensorsOf(a);
  const B = tensorsOf(b);
  const group = groupOf ?? grouper(trace ?? a?.trace ?? b?.trace ?? null);
  const byCanonical = new Map();
  for (const name of B.keys()) {
    const c = canonicalName(name);
    if (!byCanonical.has(c)) byCanonical.set(c, name);
  }
  const usedB = new Set();
  const matched = [];
  const onlyA = [];
  const shapeMismatch = [];
  for (const [name, ta] of A) {
    const nameB = B.has(name) && !usedB.has(name) ? name : byCanonical.get(canonicalName(name));
    if (nameB === undefined || usedB.has(nameB)) {
      onlyA.push({ name, shape: ta.shape, numel: ta.data.length });
      continue;
    }
    usedB.add(nameB);
    const tb = B.get(nameB);
    if (!sameShape(ta.shape, tb.shape)) {
      shapeMismatch.push({ name, nameB, shapeA: ta.shape, shapeB: tb.shape });
      continue;
    }
    const { sums, ...diff } = tensorDiff(ta.data, tb.data, tolerance);
    matched.push({ name, nameB: nameB === name ? null : nameB, shape: ta.shape, numel: ta.data.length, group: group(canonicalName(name)), ...diff, _sums: sums });
  }
  const onlyB = [...B.keys()].filter((n) => !usedB.has(n)).map((name) => ({ name, shape: B.get(name).shape, numel: B.get(name).data.length }));

  // Riepiloghi per gruppo e totali: si sommano i quadrati, non le misure (la norma di un gruppo
  // è la norma di tutti i suoi pesi messi in fila, come weight_std_by_group in gpt/recording.py).
  const groups = new Map();
  const total = { tensors: 0, params: 0, dot: 0, aa: 0, bb: 0, dd: 0, changed: 0 };
  for (const m of matched) {
    if (!groups.has(m.group)) groups.set(m.group, { group: m.group, tensors: 0, params: 0, dot: 0, aa: 0, bb: 0, dd: 0, changed: 0 });
    for (const acc of [groups.get(m.group), total]) {
      acc.tensors++;
      acc.params += m.numel;
      acc.dot += m._sums.dot;
      acc.aa += m._sums.aa;
      acc.bb += m._sums.bb;
      acc.dd += m._sums.dd;
      acc.changed += m.changed;
    }
    delete m._sums;
  }
  const summarize = ({ dot, aa, bb, dd, ...rest }) => {
    const normA = Math.sqrt(aa);
    const normB = Math.sqrt(bb);
    const diffNorm = Math.sqrt(dd);
    return {
      ...rest,
      normA,
      normB,
      diffNorm,
      relChange: normA > 0 ? diffNorm / normA : diffNorm > 0 ? Infinity : 0,
      cosine: normA > 0 && normB > 0 ? Math.max(-1, Math.min(1, dot / (normA * normB))) : null,
      changedFraction: rest.params ? rest.changed / rest.params : 0,
    };
  };
  const mostChanged = [...matched].sort((x, y) => y.relChange - x.relChange || (x.name < y.name ? -1 : 1)).slice(0, top);
  // Con un trace i gruppi seguono il suo ordine (quello dei grafici del training); altrimenti l'ordine dei tensori.
  const order = (trace ?? a?.trace ?? b?.trace)?.groups;
  const rank = Array.isArray(order) ? new Map(order.map((g, i) => [g, i])) : null;
  const groupList = [...groups.values()];
  if (rank) groupList.sort((x, y) => (rank.get(x.group) ?? Infinity) - (rank.get(y.group) ?? Infinity));
  return {
    matched,
    onlyA,
    onlyB,
    shapeMismatch,
    groups: groupList.map(summarize),
    totals: summarize(total),
    mostChanged,
    identical: onlyA.length === 0 && onlyB.length === 0 && shapeMismatch.length === 0 && matched.every((m) => m.diffNorm === 0 && m.skipped === 0),
  };
}

/** Differenza elemento per elemento `a − b` (es. la modalità "A − B" della scena). */
export function difference(a, b, out = new Float32Array(a.length)) {
  if (a.length !== b.length || out.length < a.length) throw new RangeError(`lunghezze diverse: ${a.length}, ${b.length}, uscita ${out.length}`);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}
