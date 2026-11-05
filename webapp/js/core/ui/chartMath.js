/**
 * Matematica dei grafici, separata dal disegno: scale, tick "belli", bin degli istogrammi,
 * decimazione delle serie lunghe, rampe di colore, formattazione all'italiana.
 *
 * Il modulo è PURO (nessun accesso a DOM o canvas): così si può testare con `node --test`
 * e ogni bug di calcolo emerge senza aprire il browser. Lo usa `charts.js`.
 */
import { fmt } from "../dom.js";

/* ------------------------------------------------------------------------------------------------
 * Palette e tema
 * ---------------------------------------------------------------------------------------------- */

/**
 * Colori categoriali (identità di una serie), in ordine FISSO: la serie n-esima prende sempre lo
 * slot n-esimo, mai un colore generato. Sono i passi "dark" della palette di riferimento della
 * skill dataviz, validati sullo sfondo #0f172a con il suo script: banda di luminosità, croma,
 * separazione per daltonici (ΔE adiacenti >= 8,4), contrasto >= 3:1. Per gli scatter (tutte le
 * coppie a confronto) solo i primi tre slot superano i controlli: oltre servono le etichette.
 */
export const PALETTE = Object.freeze(["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]);

/** Colore per le serie oltre l'ottava: un grigio neutro, mai un nono colore inventato. */
export const OVERFLOW_COLOR = "#64748b";

/** Colori dell'interfaccia dei grafici, coerenti con core.css (tema scuro). */
export const THEME = Object.freeze({
  surface: "#0f172a",
  text: "#e2e8f0",
  muted: "#94a3b8",
  grid: "#1e293b",
  axis: "#334155",
  accent: "#38bdf8",
  dimmed: "#475569", // barre non evidenziate quando una è in primo piano
});

/** Rampa sequenziale (una sola tinta, blu): il valore basso si confonde con lo sfondo scuro. */
export const SEQUENTIAL_STOPS = Object.freeze(["#0f2547", "#104281", "#1c5cab", "#3987e5", "#6da7ec", "#b7d3f6"]);

/** Rampa divergente blu ↔ rosso con punto medio grigio neutro ("zero" = niente colore). */
export const DIVERGING_STOPS = Object.freeze(["#b7d3f6", "#3987e5", "#1c4f8f", "#383835", "#9b3434", "#e66767", "#f6c1c1"]);

/* ------------------------------------------------------------------------------------------------
 * Numeri
 * ---------------------------------------------------------------------------------------------- */

/** true se v è un numero utilizzabile (scarta NaN, ±Infinity e non-numeri). */
export const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/** Minimo e massimo dei valori finiti; null se non ce n'è nessuno. */
export function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return min <= max ? { min, max } : null;
}

/** Conteggio, media, deviazione standard, minimo e massimo dei valori finiti (algoritmo di Welford). */
export function summarize(values) {
  let count = 0;
  let mean = 0;
  let m2 = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    count++;
    const delta = v - mean;
    mean += delta / count;
    m2 += delta * (v - mean);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!count) return { count: 0, mean: NaN, std: NaN, min: NaN, max: NaN };
  return { count, mean, std: Math.sqrt(m2 / count), min, max };
}

/** Cifre decimali sufficienti a distinguere tick distanti `step` (0,25 → 2; 50 → 0). */
export function decimalsForStep(step) {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  const exponent = Math.floor(Math.log10(step) + 1e-9);
  // 0,25 ha due cifre significative dopo la virgola: controlliamo se la mantissa è "tonda".
  const extra = Math.abs(step / 10 ** exponent - Math.round(step / 10 ** exponent)) > 1e-6 ? 1 : 0;
  return Math.max(0, -exponent + extra);
}

/**
 * Etichetta di un tick all'italiana: virgola decimale, punto delle migliaia; forma compatta
 * (10K, 2 Mln) quando l'asse arriva a 10.000, perché le cifre lunghe non entrano nel margine.
 */
export function formatTick(value, step, axisMax = Math.abs(value)) {
  if (Math.abs(value) < step * 1e-9) value = 0; // evita "-0" e 1e-17 dovuti agli arrotondamenti
  if (axisMax >= 10000) return value.toLocaleString("it-IT", { notation: "compact", maximumFractionDigits: 1 });
  return fmt(value, Math.min(decimalsForStep(step), 10));
}

/** Valore leggibile per tooltip ed etichette: 4 cifre significative, compatto sopra il milione. */
export function formatValue(value) {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs >= 1e6) return value.toLocaleString("it-IT", { notation: "compact", maximumFractionDigits: 2 });
  if (abs >= 1000) return fmt(value, 0);
  if (abs !== 0 && abs < 1e-4) return value.toLocaleString("it-IT", { notation: "scientific", maximumFractionDigits: 2 });
  return value.toLocaleString("it-IT", { maximumSignificantDigits: 4 });
}

/** Percentuale all'italiana (0,1234 → "12,3%"). */
export const formatPercent = (value, digits = 1) => `${fmt(value * 100, digits)}%`;

/** Rende visibili gli spazi nelle etichette (il GPT del corso lavora a caratteri: " " e "\n" contano). */
export function visibleLabel(label) {
  const text = String(label ?? "");
  if (text === " ") return "␣";
  if (text === "\n") return "⏎";
  if (text === "\t") return "⇥";
  return text.replace(/\n/g, "⏎");
}

/* ------------------------------------------------------------------------------------------------
 * Tick e scale
 * ---------------------------------------------------------------------------------------------- */

/** Passo "bello" (1, 2, 2,5 o 5 per una potenza di 10) che divide `span` in circa `count` parti. */
export function niceStep(span, count = 5) {
  if (!(span > 0) || !Number.isFinite(span)) return 1;
  const raw = span / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const mantissa = raw / power;
  const nice = mantissa <= 1 ? 1 : mantissa <= 2 ? 2 : mantissa <= 2.5 ? 2.5 : mantissa <= 5 ? 5 : 10;
  return nice * power;
}

/**
 * Tick arrotondati per un asse lineare.
 * @param {number} min @param {number} max estremi dei dati
 * @param {object} [o]
 * @param {number} [o.count=5] numero indicativo di intervalli
 * @param {boolean} [o.expand=true] allarga il dominio ai tick esterni (asse y); false = dominio esatto (asse x)
 * @returns {{min:number, max:number, step:number, ticks:number[]}}
 */
export function niceTicks(min, max, { count = 5, expand = true } = {}) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1, step: 0.2, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
  if (min > max) [min, max] = [max, min];
  if (min === max) {
    // Un solo valore: apriamo un intervallo attorno, altrimenti la scala dividerebbe per zero.
    const pad = min === 0 ? 1 : Math.abs(min) * 0.1;
    min -= pad;
    max += pad;
  }
  const step = niceStep(max - min, count);
  const first = expand ? Math.floor(min / step + 1e-9) * step : Math.ceil(min / step - 1e-9) * step;
  const last = expand ? Math.ceil(max / step - 1e-9) * step : Math.floor(max / step + 1e-9) * step;
  const ticks = [];
  for (let i = 0, v = first; v <= last + step * 1e-6 && i < 1000; i++, v = first + i * step) ticks.push(roundTo(v, step));
  return expand ? { min: ticks[0], max: ticks[ticks.length - 1], step, ticks } : { min, max, step, ticks };
}

/** Elimina gli errori di virgola mobile (0,30000000000000004 → 0,3) in base al passo. */
function roundTo(value, step) {
  const digits = Math.min(12, decimalsForStep(step) + 2);
  return Number(value.toFixed(digits));
}

/**
 * Tick per un asse logaritmico (solo valori > 0). Con meno di una decade usa 1…9 × 10^k,
 * con poche decadi 1, 2, 5 × 10^k, altrimenti le sole potenze di 10.
 * @returns {{min:number, max:number, ticks:number[]}} dominio allargato alle decadi o ai tick esterni
 */
export function logTicks(min, max) {
  if (!(min > 0) || !(max > 0) || !Number.isFinite(min) || !Number.isFinite(max)) return { min: 1, max: 10, ticks: [1, 10] };
  if (min > max) [min, max] = [max, min];
  if (min === max) {
    min /= 2;
    max *= 2;
  }
  const decades = Math.log10(max / min);
  const mantissas = decades <= 1.3 ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : decades <= 3.5 ? [1, 2, 5] : [1];
  const k0 = Math.floor(Math.log10(min));
  const k1 = Math.ceil(Math.log10(max));
  const every = Math.max(1, Math.ceil((k1 - k0) / 8)); // con tante decadi una potenza sì e una no
  const candidates = [];
  for (let k = k0; k <= k1; k++) {
    if (mantissas.length === 1 && (k - k0) % every) continue;
    for (const m of mantissas) candidates.push(Number((m * 10 ** k).toPrecision(12)));
  }
  // Dominio: dal tick appena sotto il minimo a quello appena sopra il massimo.
  const lower = candidates.filter((t) => t <= min * (1 + 1e-9)).pop() ?? candidates[0];
  const upper = candidates.find((t) => t >= max * (1 - 1e-9)) ?? candidates[candidates.length - 1];
  const ticks = candidates.filter((t) => t >= lower && t <= upper);
  return { min: lower, max: upper, ticks };
}

/** Scala lineare dominio → pixel, con `invert` (pixel → dominio). */
export function linearScale([d0, d1], [r0, r1]) {
  const span = d1 - d0 || 1;
  const k = (r1 - r0) / span;
  const scale = (v) => r0 + (v - d0) * k;
  scale.invert = (p) => d0 + (p - r0) / k;
  scale.domain = [d0, d1];
  return scale;
}

/** Scala logaritmica (dominio > 0): uguali rapporti → uguali distanze. */
export function logScale([d0, d1], [r0, r1]) {
  const l0 = Math.log10(d0);
  const inner = linearScale([l0, Math.log10(d1)], [r0, r1]);
  const scale = (v) => (v > 0 ? inner(Math.log10(v)) : NaN);
  scale.invert = (p) => 10 ** inner.invert(p);
  scale.domain = [d0, d1];
  return scale;
}

/**
 * Geometria di barre affiancate: banda per categoria e spessore della barra, che non supera
 * `maxBar` (barre sottili, il resto è aria) e lascia sempre almeno `gap` px tra due barre.
 */
export function bandLayout(n, length, { maxBar = 24, gap = 2 } = {}) {
  if (n <= 0 || length <= 0) return { band: 0, bar: 0, offset: 0 };
  const band = length / n;
  const bar = Math.max(1, Math.min(maxBar, band - gap, band * 0.72));
  return { band, bar, offset: (band - bar) / 2 };
}

/* ------------------------------------------------------------------------------------------------
 * Istogramma
 * ---------------------------------------------------------------------------------------------- */

/**
 * Divide i valori finiti in `bins` intervalli uguali [a, b); l'ultimo include il massimo.
 * @param {ArrayLike<number>} values
 * @param {object} [o] @param {number} [o.bins=40] @param {number} [o.min] @param {number} [o.max]
 * @returns {{edges: Float64Array, counts: Uint32Array, min:number, max:number, total:number, ignored:number} | null}
 *   `ignored` conta i valori scartati (NaN, Infinity o fuori da [min, max]); null se non resta nulla.
 */
export function histogram(values, { bins = 40, min, max } = {}) {
  const n = Math.max(1, Math.floor(Number.isFinite(bins) ? bins : 40));
  const range = extent(values);
  if (!range) return null;
  let lo = Number.isFinite(min) ? min : range.min;
  let hi = Number.isFinite(max) ? max : range.max;
  if (lo > hi) [lo, hi] = [hi, lo];
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const width = (hi - lo) / n;
  const counts = new Uint32Array(n);
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    counts[Math.min(n - 1, Math.floor((v - lo) / width))]++;
    total++;
  }
  const edges = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) edges[i] = lo + i * width;
  edges[n] = hi; // esatto, senza errore accumulato
  return { edges, counts, min: lo, max: hi, total, ignored: values.length - total };
}

/* ------------------------------------------------------------------------------------------------
 * Serie lunghe
 * ---------------------------------------------------------------------------------------------- */

/**
 * Decimazione min/max per colonna di pixel: di ogni colonna tiene il primo, il minimo, il massimo
 * e l'ultimo punto (nell'ordine in cui compaiono). La linea disegnata è identica a quella completa
 * — picchi compresi — ma con al più 4 punti per pixel invece di centinaia di migliaia.
 * Richiede x crescenti.
 * @param {ArrayLike<number>} xs @param {ArrayLike<number>} ys
 * @param {number} count punti validi in xs/ys
 * @param {number} x0 @param {number} x1 intervallo visibile (dominio)
 * @param {number} columns larghezza del grafico in pixel
 * @returns {Float64Array} coppie x, y consecutive
 */
export function decimateMinMax(xs, ys, count, x0, x1, columns) {
  const cols = Math.max(1, Math.floor(columns));
  const out = new Float64Array(Math.min(count, cols * 4 + 4) * 2);
  const k = cols / (x1 - x0 || 1);
  let n = 0;
  const push = (i) => {
    out[n++] = xs[i];
    out[n++] = ys[i];
  };
  let i = 0;
  while (i < count) {
    const column = Math.min(cols - 1, Math.max(0, Math.floor((xs[i] - x0) * k)));
    let first = i;
    let iMin = i;
    let iMax = i;
    let last = i;
    for (i++; i < count && Math.min(cols - 1, Math.max(0, Math.floor((xs[i] - x0) * k))) === column; i++) {
      if (ys[i] < ys[iMin]) iMin = i;
      if (ys[i] > ys[iMax]) iMax = i;
      last = i;
    }
    // Indici in ordine crescente e senza ripetizioni.
    const picks = [first, iMin, iMax, last].sort((a, b) => a - b);
    for (let p = 0; p < 4; p++) if (p === 0 || picks[p] !== picks[p - 1]) push(picks[p]);
  }
  return out.subarray(0, n);
}

/** Indice del punto con x più vicina a `x` in un array ordinato (ricerca binaria); -1 se vuoto. */
export function nearestIndexSorted(xs, count, x) {
  if (count <= 0) return -1;
  let lo = 0;
  let hi = count - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}

/** Primo e ultimo indice con x in [x0, x1] (array ordinato), allargati di un punto per non tagliare la linea. */
export function visibleRange(xs, count, x0, x1) {
  let lo = 0;
  let hi = count;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x0) lo = mid + 1;
    else hi = mid;
  }
  const start = Math.max(0, lo - 1);
  hi = count;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x1) lo = mid + 1;
    else hi = mid;
  }
  return [start, Math.min(count, lo + 1)];
}

/* ------------------------------------------------------------------------------------------------
 * Etichette
 * ---------------------------------------------------------------------------------------------- */

/**
 * Posizionamento "goloso" delle etichette: in ordine di priorità tiene un'etichetta solo se non
 * si sovrappone a quelle già tenute (meglio qualche etichetta in meno che un groviglio).
 * @param {{x:number, y:number, w:number, h:number}[]} boxes rettangoli già in ordine di priorità
 * @param {number} [padding=2]
 * @returns {boolean[]} quali tenere
 */
export function placeLabels(boxes, padding = 2) {
  const kept = [];
  return boxes.map((b) => {
    const clash = kept.some((k) => b.x < k.x + k.w + padding && k.x < b.x + b.w + padding && b.y < k.y + k.h + padding && k.y < b.y + b.h + padding);
    if (!clash) kept.push(b);
    return !clash;
  });
}

/* ------------------------------------------------------------------------------------------------
 * Colore
 * ---------------------------------------------------------------------------------------------- */

/** "#rrggbb" → [r, g, b] (0-255). */
export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`Colore non valido: ${hex}`);
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const toLinear = (c) => ((c /= 255) <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fromLinear = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** sRGB → OKLab: lo spazio in cui interpolare perché i passi della rampa sembrino uguali all'occhio. */
function rgbToOklab([r, g, b]) {
  const [lr, lg, lb] = [toLinear(r), toLinear(g), toLinear(b)];
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb);
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb);
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}

function oklabToRgb([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s];
  return rgb.map((c) => Math.round(Math.min(255, Math.max(0, fromLinear(c)))));
}

/**
 * Tabella di colori pre-calcolata (`size` passi) per una rampa: il disegno di una heatmap
 * grande legge un colore per cella, quindi l'interpolazione si fa una volta sola.
 * @returns {Uint8Array} size × 3 componenti RGB
 */
export function buildColorTable(stops, size = 256) {
  const labs = stops.map((hex) => rgbToOklab(hexToRgb(hex)));
  const table = new Uint8Array(size * 3);
  for (let i = 0; i < size; i++) {
    const t = (i / (size - 1)) * (labs.length - 1);
    const j = Math.min(labs.length - 2, Math.floor(t));
    const f = t - j;
    const rgb = oklabToRgb(labs[j].map((c, k) => c + (labs[j + 1][k] - c) * f));
    table.set(rgb, i * 3);
  }
  return table;
}

/**
 * Dominio di colore di una heatmap. Divergente: simmetrico attorno a 0 (lo zero cade sul grigio
 * neutro, così il segno si legge subito). Sequenziale: da min a max dei dati, o quelli forniti.
 */
export function colorDomain(values, { min, max, diverging = false } = {}) {
  const range = extent(values) ?? { min: 0, max: 1 };
  if (diverging) {
    const limit = Number.isFinite(min) || Number.isFinite(max) ? Math.max(Math.abs(min ?? 0), Math.abs(max ?? 0)) : Math.max(Math.abs(range.min), Math.abs(range.max));
    const m = limit || 1;
    return { min: -m, max: m };
  }
  const lo = Number.isFinite(min) ? min : range.min;
  const hi = Number.isFinite(max) ? max : range.max;
  return lo === hi ? { min: lo - 0.5, max: hi + 0.5 } : { min: lo, max: hi };
}

/** Posizione 0…1 di un valore nel dominio di colore (saturata agli estremi). */
export const colorPosition = (v, { min, max }) => Math.min(1, Math.max(0, (v - min) / (max - min || 1)));
