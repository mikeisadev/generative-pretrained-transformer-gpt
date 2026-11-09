/**
 * Il confronto A/B dei pesi per la tabella, la scena e gli istogrammi (logica pura, testata).
 *
 * Convenzione del Laboratorio (content.js): A è il run nella scena, B il termine di paragone. La
 * libreria (shared/weights/compare.js) misura il cambiamento relativo rispetto al suo PRIMO argomento,
 * il "prima": per avere ‖A − B‖ / ‖B‖ la si chiama con (B, A) e qui si rimettono i nomi al loro posto
 * (normA è sempre la norma di A, onlyA i tensori che ha solo A, e così via). Nessun calcolo è rifatto:
 * norme, coseni e differenze sono quelli della libreria, provati in tests/weights.test.mjs.
 *
 *   const c = compareAB(viewA, viewB, { trace });        // viste di LabState.weightsAt
 *   sortRows(c.rows, "relChange", "desc")                // i più cambiati in alto
 *   const d = differenceMap(c, viewA, viewB, buffers);   // nome → Float32Array (A − B) per la scena
 */
import { compareTensors, difference } from "../../../shared/weights/compare.js";

/**
 * Le colonne della tabella, nell'ordine; `dir` è il verso predefinito quando la si ordina. Le misure del
 * cambiamento vengono subito dopo il nome: nella sidebar stretta e sul telefono si leggono senza scorrere.
 */
export const COLUMNS = Object.freeze([
  { id: "name", label: "Tensore", dir: "asc", numeric: false },
  { id: "relChange", label: "Cambio rel.", dir: "desc", numeric: true },
  { id: "cosine", label: "Coseno", dir: "asc", numeric: true },
  { id: "diffNorm", label: "‖A − B‖", dir: "desc", numeric: true },
  { id: "maxAbsDiff", label: "max |Δ|", dir: "desc", numeric: true },
  { id: "normA", label: "‖A‖", dir: "desc", numeric: true },
  { id: "normB", label: "‖B‖", dir: "desc", numeric: true },
  { id: "shape", label: "Forma", dir: "desc", numeric: true },
]);

export const DEFAULT_SORT = Object.freeze({ key: "relChange", dir: "desc" });

const swapNorms = ({ normA, normB, ...rest }) => ({ ...rest, normA: normB, normB: normA });

/**
 * Confronta i pesi di A con quelli di B.
 * @param {{tensors: Map}} a i pesi di A (vista di LabState.weightsAt, WeightRun o snapshot)
 * @param {{tensors: Map}} b i pesi di B
 * @param {{trace?: object, tolerance?: number}} [o] `trace`: i gruppi di trace.json per i riepiloghi
 * @returns {{rows: object[], onlyA: object[], onlyB: object[], shapeMismatch: object[], groups: object[],
 *   totals: object, identical: boolean, matchedParams: number, paramsA: number, paramsB: number}}
 *   rows: [{name (di A), nameB (di B se diverso, es. con module.), shape, numel, group, normA, normB,
 *   diffNorm, relChange (‖A − B‖ / ‖B‖), cosine, maxAbsDiff, meanAbsDiff, changedFraction, skipped}]
 */
export function compareAB(a, b, { trace = null, tolerance = 0 } = {}) {
  const tensorsA = a?.tensors instanceof Map ? a.tensors : a;
  const tensorsB = b?.tensors instanceof Map ? b.tensors : b;
  // B per primo: la libreria fa il cambiamento relativo rispetto al primo argomento
  const c = compareTensors(tensorsB, tensorsA, { trace, tolerance, top: 0 });
  const rows = c.matched.map((m) => {
    const { name, nameB, ...rest } = swapNorms(m);
    // per la libreria `name` è il nome in B, `nameB` (se diverso) quello in A
    return { ...rest, name: nameB ?? name, nameB: nameB ? name : null };
  });
  const sum = (map) => [...map.values()].reduce((s, t) => s + (t.data?.length ?? t.numel ?? 0), 0);
  return {
    rows,
    onlyA: c.onlyB,
    onlyB: c.onlyA,
    shapeMismatch: c.shapeMismatch.map((s) => ({ name: s.nameB, nameB: s.name === s.nameB ? null : s.name, shapeA: s.shapeB, shapeB: s.shapeA })),
    groups: c.groups.map(swapNorms),
    totals: swapNorms(c.totals),
    identical: c.identical,
    matchedParams: rows.reduce((s, r) => s + r.numel, 0),
    paramsA: sum(tensorsA),
    paramsB: sum(tensorsB),
  };
}

const shapeSize = (s) => (Array.isArray(s) ? s.reduce((p, v) => p * v, 1) : 0);

/**
 * Righe ordinate (copia) per una colonna. A pari valore decide il nome, così l'ordine è stabile; i
 * valori mancanti (es. il coseno di un tensore tutto zero) e i cambi relativi infiniti (B tutto zero:
 * nessuna scala con cui dividere) vanno sempre in fondo, così in cima ci sono i cambiamenti misurabili.
 * @param {object[]} rows righe di compareAB
 * @param {string} key id di COLUMNS
 * @param {"asc"|"desc"} dir
 */
export function sortRows(rows, key = DEFAULT_SORT.key, dir = DEFAULT_SORT.dir) {
  const sign = dir === "asc" ? 1 : -1;
  const value = (r) => (key === "shape" ? shapeSize(r.shape) : r[key]);
  return [...rows].sort((x, y) => {
    if (key === "name") return sign * compareNames(x.name, y.name);
    const vx = value(x);
    const vy = value(y);
    const mx = typeof vx !== "number" || !Number.isFinite(vx);
    const my = typeof vy !== "number" || !Number.isFinite(vy);
    if (mx || my) return mx === my ? compareNames(x.name, y.name) : mx ? 1 : -1;
    return vx === vy ? compareNames(x.name, y.name) : sign * (vx < vy ? -1 : 1);
  });
}

/** Nomi in ordine "naturale": blocks.2 prima di blocks.10. */
function compareNames(a, b) {
  return String(a).localeCompare(String(b), "it", { numeric: true });
}

/** Il prossimo ordinamento quando si sceglie una colonna: la stessa inverte il verso, una nuova parte dal suo verso. */
export function nextSort(current, key) {
  const column = COLUMNS.find((c) => c.id === key);
  if (!column) return current;
  if (current?.key === key) return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  return { key, dir: column.dir };
}

/**
 * Che cosa si è abbinato e che cosa no, in italiano (per il riquadro "Abbinamento").
 * @returns {{matched: number, onlyA: number, onlyB: number, mismatched: number, lines: string[], complete: boolean}}
 */
export function matchSummary(c) {
  const lines = [];
  const few = (list, k = 3) => `${list.slice(0, k).join(", ")}${list.length > k ? ` e altri ${list.length - k}` : ""}`;
  if (c.shapeMismatch.length) {
    lines.push(`${c.shapeMismatch.length === 1 ? "1 tensore ha" : `${c.shapeMismatch.length} tensori hanno`} lo stesso nome ma una forma diversa e non si confronta cella per cella: ${few(c.shapeMismatch.map((s) => `${s.name} (${s.shapeA.join("×")} in A, ${s.shapeB.join("×")} in B)`))}.`);
  }
  if (c.onlyA.length) lines.push(`Solo in A (${c.onlyA.length}): ${few(c.onlyA.map((t) => t.name))}.`);
  if (c.onlyB.length) lines.push(`Solo in B (${c.onlyB.length}): ${few(c.onlyB.map((t) => t.name))}.`);
  const renamed = c.rows.filter((r) => r.nameB);
  if (renamed.length) lines.push(`${renamed.length === 1 ? "1 abbinato" : `${renamed.length} abbinati`} togliendo il prefisso module. o _orig_mod. (es. ${renamed[0].name} in A ↔ ${renamed[0].nameB} in B).`);
  return {
    matched: c.rows.length,
    onlyA: c.onlyA.length,
    onlyB: c.onlyB.length,
    mismatched: c.shapeMismatch.length,
    lines,
    complete: !c.onlyA.length && !c.onlyB.length && !c.shapeMismatch.length,
  };
}

/**
 * I valori della modalità differenza: per ogni tensore abbinato A − B, elemento per elemento.
 * `buffers` (Map nome → Float32Array) permette di riusare gli array di una volta precedente: scorrendo
 * gli step del training non si alloca a ogni passo (la scena riceve lo stesso array, con i valori nuovi).
 * @returns {{values: Map<string, Float32Array>, maxAbs: number, unmatched: string[]}}
 *   `unmatched`: i tensori di A senza un gemello della stessa forma (la scena li vela)
 */
export function differenceMap(c, a, b, buffers = new Map()) {
  const tensorsA = a?.tensors instanceof Map ? a.tensors : a;
  const tensorsB = b?.tensors instanceof Map ? b.tensors : b;
  const values = new Map();
  let maxAbs = 0;
  for (const r of c.rows) {
    const ta = tensorsA.get(r.name);
    const tb = tensorsB.get(r.nameB ?? r.name);
    if (!ta || !tb) continue;
    let out = buffers.get(r.name);
    if (!out || out.length !== ta.data.length) out = new Float32Array(ta.data.length);
    difference(ta.data, tb.data, out);
    for (let i = 0; i < out.length; i++) {
      const v = out[i] < 0 ? -out[i] : out[i];
      if (v > maxAbs && v !== Infinity) maxAbs = v; // NaN non vince mai il confronto
    }
    values.set(r.name, out);
  }
  const unmatched = [...c.onlyA.map((t) => t.name), ...c.shapeMismatch.map((s) => s.name)];
  return { values, maxAbs, unmatched };
}

/**
 * Le differenze sono solo l'arrotondamento a float16 di uno snapshot? Vero se uno dei due lati è
 * float16, l'altro float32 e ogni peso differisce al più di mezzo passo di float16 attorno al suo
 * valore (2⁻¹¹ relativo, più un minimo per i numeri piccolissimi).
 */
export function onlyHalfRounding(c, dtypeA, dtypeB) {
  if (!c?.rows.length || dtypeA === dtypeB || ![dtypeA, dtypeB].includes("float16")) return false;
  const HALF = 2 ** -11;
  return c.rows.every((r) => r.maxAbsDiff <= Math.max(HALF * Math.max(r.normA, r.normB), 6.2e-5) && r.relChange <= 2 * HALF);
}
