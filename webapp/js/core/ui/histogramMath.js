/**
 * Logica pura dell'istogramma dei valori di un tensore (Histogram, core/ui/histogram.js): intervalli
 * comuni a più serie sovrapposte (confronto A/B), conteggi, tacche degli assi, scala logaritmica dei
 * conteggi, testi dell'intervallo e del riassunto per gli screen reader. Niente DOM né canvas: test in
 * tests/tensor-ui.test.mjs.
 *
 * I conteggi vengono da shared/weights/stats.js (`histogram`, `range`): gli stessi intervalli del
 * Laboratorio pesi e delle statistiche, con NaN e ±Infinity esclusi e contati a parte.
 */
import { formatCell } from "../../shared/weights/display.js";
import { histogram, range } from "../../shared/weights/stats.js";
import { PALETTE, OVERFLOW_COLOR, decimalsForStep, niceTicks } from "./chartMath.js";

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const COMPACT = new Intl.NumberFormat("it-IT", { notation: "compact", maximumFractionDigits: 1 });
const PCT = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 1, minimumFractionDigits: 1 });

/** Colore della serie i-esima: slot fisso della palette dei grafici (mai un colore generato). */
export const seriesColor = (i) => PALETTE[i] ?? OVERFLOW_COLOR;

/** Conteggi interi all'italiana; compatti (123,5K, 1,2 Mln) sopra i 100.000 per stare nel margine dell'asse. */
export const countText = (n, { compact = false } = {}) => (compact && n >= 100000 ? COMPACT.format(n) : INT.format(n));

/** "1 valore", "1.234 valori". */
export const valuesText = (n) => `${INT.format(n)} ${n === 1 ? "valore" : "valori"}`;

/**
 * Modello dell'istogramma di una o più serie con GLI STESSI intervalli (così due tensori si
 * confrontano barra per barra). Un intervallo vuoto (tutti i valori uguali) si allarga attorno al
 * valore, così la barra unica sta al centro invece che sul bordo.
 * @param {{label?: string, data: ArrayLike<number>, color?: string}[]} series
 * @param {{bins?: number, logY?: boolean}} [o]
 * @returns {{empty: boolean, min: number, max: number, bins: number, binWidth: number, logY: boolean,
 *   series: {label: string, color: string, counts: Uint32Array, total: number, excluded: number}[],
 *   maxCount: number, xTicks: {step: number, ticks: number[]}, yTicks: number[], yTop: number}}
 */
export function histogramModel(series, { bins = 60, logY = false } = {}) {
  const list = (Array.isArray(series) ? series : [series]).filter((s) => s && s.data && typeof s.data.length === "number");
  const nb = Math.max(1, Math.min(1000, Math.floor(Number.isFinite(bins) ? bins : 60)));
  const r = range(list.map((s) => s.data));
  const empty = !Number.isFinite(r.min);
  let { min, max } = empty ? { min: 0, max: 1 } : r;
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 0.5;
    min -= pad;
    max += pad;
  }
  const out = list.map((s, i) => {
    const h = empty ? { counts: new Uint32Array(nb), total: 0 } : histogram(s.data, { bins: nb, min, max });
    return { label: s.label ?? `serie ${i + 1}`, color: s.color ?? seriesColor(i), counts: h.counts, total: h.total, excluded: s.data.length - h.total };
  });
  let maxCount = 0;
  for (const s of out) for (let b = 0; b < s.counts.length; b++) if (s.counts[b] > maxCount) maxCount = s.counts[b];
  const xTicks = niceTicks(min, max, { count: 5, expand: false });
  const y = logY ? logCountTicks(maxCount) : niceTicks(0, Math.max(1, maxCount), { count: 4, expand: true });
  return {
    empty: empty || out.every((s) => s.total === 0),
    min,
    max,
    bins: nb,
    binWidth: (max - min) / nb,
    logY,
    series: out,
    maxCount,
    xTicks: { step: xTicks.step, ticks: xTicks.ticks },
    yTicks: logY ? y : y.ticks,
    yTop: logY ? y[y.length - 1] : y.max,
  };
}

/** Tacche dei conteggi in scala logaritmica: 0, 1, 10, 100, … fino alla potenza di 10 che copre il massimo. */
export function logCountTicks(maxCount) {
  const ticks = [0, 1];
  while (ticks[ticks.length - 1] < maxCount) ticks.push(ticks[ticks.length - 1] * 10);
  return ticks;
}

/**
 * Altezza relativa (0…1) di un conteggio. In scala logaritmica si usa log(1 + n): lo zero resta a
 * zero e un intervallo con un solo valore si vede ancora (con log(n) sparirebbe).
 */
export function countHeight(count, model) {
  if (!(model.yTop > 0) || count <= 0) return 0;
  return model.logY ? Math.log10(1 + count) / Math.log10(1 + model.yTop) : count / model.yTop;
}

/** Indice dell'intervallo che contiene il valore x (i bordi esterni appartengono al primo e all'ultimo). */
export function binAt(model, x) {
  if (!Number.isFinite(x) || !(model.binWidth > 0)) return -1;
  if (x < model.min || x > model.max) return -1;
  return Math.min(model.bins - 1, Math.floor((x - model.min) / model.binWidth));
}

/**
 * Decimali dei bordi degli intervalli (da 2 a 6): quelli del passo, più uno se i bordi non cadono
 * esattamente su quei decimali (succede quasi sempre con i pesi: il minimo è un numero qualunque).
 */
export function edgeDecimals(model) {
  const d = decimalsForStep(model.binWidth);
  const exact = (v) => Math.abs(v * 10 ** d - Math.round(v * 10 ** d)) < 1e-6;
  return Math.min(6, Math.max(2, exact(model.min) && exact(model.binWidth) ? d : d + 1));
}

/** "[−0,051; −0,047)" (l'ultimo intervallo comprende il massimo: "]"). */
export function binLabel(model, index) {
  const d = edgeDecimals(model);
  const a = model.min + index * model.binWidth;
  const b = index === model.bins - 1 ? model.max : model.min + (index + 1) * model.binWidth;
  return `[${formatCell(a, d)}; ${formatCell(b, d)}${index === model.bins - 1 ? "]" : ")"}`;
}

/** Righe del suggerimento di un intervallo: valore prima (in evidenza), poi la serie. */
export function binRows(model, index) {
  return model.series.map((s) => {
    const n = s.counts[index] ?? 0;
    return { label: s.label, color: s.color, count: n, text: valuesText(n), share: s.total ? `${PCT.format((n / s.total) * 100)}%` : "–" };
  });
}

/**
 * Riassunto per gli screen reader (aria-label del grafico): quanti valori, da dove a dove, quanti
 * intervalli e dov'è il picco di ogni serie.
 */
export function summaryText(model, { label } = {}) {
  if (model.empty) return `${label ? `${label}: ` : ""}istogramma senza valori da mostrare.`;
  const d = edgeDecimals(model);
  const span = `da ${formatCell(model.min, d)} a ${formatCell(model.max, d)} in ${model.bins} intervalli larghi ${formatCell(model.binWidth, d)}`;
  const peak = (s) => {
    let best = 0;
    for (let b = 1; b < s.counts.length; b++) if (s.counts[b] > s.counts[best]) best = b;
    return `${valuesText(s.total)}, il picco in ${binLabel(model, best)} con ${INT.format(s.counts[best])}`;
  };
  const head = label ? `${label}. ` : "";
  const scale = model.logY ? " Conteggi in scala logaritmica." : "";
  if (model.series.length === 1) return `${head}Istogramma ${span}: ${peak(model.series[0])}.${scale}`;
  return `${head}Istogramma di ${model.series.length} serie sovrapposte con gli stessi intervalli, ${span}. ${model.series.map((s) => `${s.label}: ${peak(s)}`).join("; ")}.${scale}`;
}
