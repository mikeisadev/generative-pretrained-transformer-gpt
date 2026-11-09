/**
 * Logica pura della tabella dei numeri di un tensore (TensorTable, core/ui/tensorTable.js): forma della
 * tabella, finestra virtuale (quali righe e colonne disegnare), scorrimento "compresso" per i tensori
 * enormi, movimenti della tastiera, testi delle celle e delle statistiche. Niente DOM, niente three.js:
 * i test sono in tests/tensor-ui.test.mjs.
 *
 * Perché una finestra virtuale: un vettore di 65.536 valori sono 65.536 celle; il DOM ne regge qualche
 * centinaio senza rallentare. La tabella disegna solo le righe e le colonne visibili (più un margine)
 * e le riposiziona mentre si scorre, come una finestra che scivola su un foglio di calcolo enorme.
 */
import { MINUS, formatCell, formatPrecise } from "../../shared/weights/display.js";

/**
 * Lato massimo (px) dell'area di scorrimento. Oltre, i browser non sanno più posizionare gli elementi
 * (Chromium ~33 milioni di px, Firefox ~17 milioni): l'area resta di questa misura e la posizione si
 * "comprime" (vedi axisWindow). Con celle alte 24 px vale da 333.000 righe in su.
 */
export const MAX_SCROLL_PX = 8_000_000;

/** Decimali predefiniti delle celle: "−0,1234" sta in una cella stretta e basta a confrontare i pesi. */
export const CELL_DECIMALS = 4;

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Forma del tensore → [righe, colonne] della tabella: [n] → n righe × 1 colonna (un vettore si legge
 * dall'alto in basso, come le barre dei bias nella scena del Transformer); [r, c] così com'è (un
 * nn.Linear (out, in) ha una riga per uscita); oltre 2D le dimensioni iniziali in fila, come
 * `tensor.reshape(-1, ultima)`; [] (scalare) → 1 × 1.
 * @param {number[]} shape
 * @returns {[number, number]}
 */
export function tableShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return [1, 1];
  if (!shape.every((d) => Number.isInteger(d) && d >= 0)) throw new RangeError(`forma non valida: (${shape.join(", ")})`);
  if (shape.length === 1) return [shape[0], 1];
  const cols = shape[shape.length - 1];
  return [shape.slice(0, -1).reduce((a, b) => a * b, 1), cols];
}

/**
 * Finestra virtuale lungo un asse: `count` elementi da `size` px dentro una finestra di `viewport` px,
 * scorsa di `scroll` px. Se count × size supera `maxPx`, l'area di scorrimento è lunga `maxPx` e la
 * posizione virtuale cresce più in fretta dello scorrimento (fattore `scale`): ogni pixel di barra
 * vale più di un pixel di tabella, ma l'inizio e la fine restano raggiungibili e ogni cella esatta.
 * La cella i va disegnata a `i × size + shift` (coordinate dell'area di scorrimento).
 * @param {{count: number, size: number, viewport: number, scroll: number, overscan?: number, maxPx?: number}} o
 * @returns {{content: number, first: number, last: number, shift: number, scale: number, virtual: number}}
 *   content = lunghezza dell'area di scorrimento; first…last (escluso) = elementi da disegnare
 */
export function axisWindow({ count, size, viewport, scroll, overscan = 2, maxPx = MAX_SCROLL_PX }) {
  const n = Math.max(0, Math.floor(count));
  const total = n * size;
  const content = Math.min(total, maxPx);
  const view = Math.max(0, viewport);
  const range = Math.max(0, content - view); // quanto si può scorrere davvero
  const virtualRange = Math.max(0, total - view); // quanto c'è da scorrere nella tabella vera
  const s = clamp(scroll || 0, 0, range);
  const scale = range > 0 ? virtualRange / range : 1;
  const virtual = s * scale;
  if (!n || !(size > 0)) return { content, first: 0, last: 0, shift: 0, scale, virtual };
  const first = clamp(Math.floor(virtual / size) - overscan, 0, n);
  const last = clamp(Math.ceil((virtual + view) / size) + overscan, first, n);
  return { content, first, last, shift: s - virtual, scale, virtual };
}

/**
 * Scorrimento che porta in vista l'elemento `index` (spostandosi il meno possibile): `current` se è già
 * tutto visibile, altrimenti quello che lo mette sul bordo più vicino. Tiene conto della compressione.
 * @returns {number} il nuovo scrollTop / scrollLeft
 */
export function scrollToShow({ index, count, size, viewport, current, maxPx = MAX_SCROLL_PX }) {
  const total = Math.max(0, count) * size;
  const content = Math.min(total, maxPx);
  const view = Math.max(0, viewport);
  const range = Math.max(0, content - view);
  const virtualRange = Math.max(0, total - view);
  if (range <= 0) return 0;
  const scale = virtualRange / range;
  const virtual = clamp(current, 0, range) * scale;
  const start = clamp(index, 0, Math.max(0, count - 1)) * size;
  let target = virtual;
  if (start < virtual) target = start;
  else if (start + size > virtual + view) target = start + size - view;
  return clamp(target / scale, 0, range);
}

/** Tasti che la tabella usa per muovere la cella attiva (gli stessi di una griglia ARIA). */
export const GRID_KEYS = Object.freeze(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown", "Home", "End"]);

/**
 * Nuova cella attiva dopo un tasto, o null se il tasto non è della tabella.
 * Frecce: una cella; PagSu/PagGiù: una finestra di righe (`pageRows`); Home/Fine: inizio/fine della
 * riga (con Ctrl: prima/ultima cella della tabella). Con una colonna sola (un vettore) Home e Fine
 * vanno alla prima e all'ultima riga: in una riga di una cella non avrebbero niente da fare.
 * @param {{row: number, col: number}} at
 * @param {string} key
 * @param {{rows: number, cols: number, pageRows?: number, ctrl?: boolean}} o
 * @returns {{row: number, col: number}|null}
 */
export function moveActive(at, key, { rows, cols, pageRows = 10, ctrl = false }) {
  if (!GRID_KEYS.includes(key) || rows <= 0 || cols <= 0) return null;
  let { row, col } = at ?? { row: 0, col: 0 };
  const page = Math.max(1, Math.floor(pageRows));
  switch (key) {
    case "ArrowUp":
      row -= 1;
      break;
    case "ArrowDown":
      row += 1;
      break;
    case "ArrowLeft":
      col -= 1;
      break;
    case "ArrowRight":
      col += 1;
      break;
    case "PageUp":
      row -= page;
      break;
    case "PageDown":
      row += page;
      break;
    case "Home":
      if (ctrl || cols === 1) row = 0;
      col = 0;
      break;
    case "End":
      if (ctrl || cols === 1) row = rows - 1;
      col = cols - 1;
      break;
  }
  return { row: clamp(row, 0, rows - 1), col: clamp(col, 0, cols - 1) };
}

/** Testo di una cella: virgola decimale e segno "−" come nelle matrici della scena (display.js). */
export const cellText = (value, decimals = CELL_DECIMALS) => formatCell(value, decimals);

const SUPERSCRIPT = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

/**
 * Il valore "esatto" da leggere (scheda, suggerimento, annuncio della cella attiva): fino a 6 decimali
 * ("−0,012346"); i numeri minuscoli in notazione scientifica leggibile ("3,2 · 10⁻⁷") invece di
 * "0,000000", che farebbe credere a uno zero vero; quelli enormi con 7 cifre significative.
 */
export function preciseText(value) {
  if (!Number.isFinite(value)) return formatCell(value, 0);
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e-5 && abs < 1e4) return formatCell(value, 6);
  if (abs >= 1e4) return formatPrecise(value, 7);
  return scientificText(value, 2);
}

/** "5,278 · 10⁻⁵": notazione scientifica leggibile (`digits` cifre significative), virgola e segno "−". */
export function scientificText(value, digits = 4) {
  const [mantissa, exponent] = Math.abs(value).toExponential(Math.max(0, digits - 1)).split("e");
  const power = String(Number(exponent)).replace(/[-0-9]/g, (c) => SUPERSCRIPT[c]);
  return `${value < 0 ? MINUS : ""}${mantissa.replace(".", ",")} · 10${power}`;
}

/** Larghezza (in caratteri) del numero più largo della tabella: decide quanto è larga una colonna. */
export function widestText(maxAbs, format = cellText) {
  const m = Number.isFinite(maxAbs) ? maxAbs : 0;
  return Math.max(format(-m).length, format(m).length, format(0).length);
}

/** |valore| massimo dei valori finiti (per la scala dei colori e la larghezza delle colonne). */
export function finiteAbsMax(data, count = data.length) {
  let m = 0;
  for (let i = 0; i < count; i++) {
    const v = data[i];
    const a = v < 0 ? -v : v;
    if (a > m && a !== Infinity) m = a;
  }
  return m;
}

/**
 * Testo dell'intestazione di riga/colonna: l'etichetta del chiamante (es. il carattere di una riga
 * dell'embedding) o l'indice. Le etichette possono contenere spazi e a capo: si rendono visibili.
 */
export function headerText(index, label) {
  if (typeof label !== "function") return String(index);
  const text = label(index);
  return text === undefined || text === null || text === "" ? String(index) : String(text);
}

/**
 * Frase che descrive una cella (annuncio della cella attiva per gli screen reader, riga di stato):
 * "riga 39 ('a') · colonna 3 · −0,123457". L'etichetta si aggiunge solo se dice qualcosa in più
 * dell'indice.
 */
export function describeCell({ row, col, value }, { rowLabel, colLabel, rowTitle = "riga", colTitle = "colonna" } = {}) {
  const part = (title, index, label) => {
    const text = headerText(index, label);
    return text === String(index) ? `${title} ${index}` : `${title} ${index} (${text})`;
  };
  return `${part(rowTitle, row, rowLabel)} · ${part(colTitle, col, colLabel)} · ${preciseText(value)}`;
}

const INT = new Intl.NumberFormat("it-IT", { useGrouping: "always", maximumFractionDigits: 0 });
const PCT = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 1, minimumFractionDigits: 1 });

/**
 * Righe della scheda delle statistiche da `tensorStats(data, { nearZero: 0.01 })` di
 * shared/weights/stats.js: numero di valori, media, deviazione standard, minimo, massimo, media di
 * |w|, quota di |w| < 0,01, norma L2. Quattro cifre significative (formatPrecise): bastano a
 * confrontare due tensori e non fingono una precisione che un float16 non ha.
 * @returns {{id: string, label: string, value: string, title?: string}[]}
 */
export function statsRows(s) {
  // i numeri minuscoli (la media dei pesi è spesso ~10⁻⁵) in notazione scientifica leggibile, non "5,278e-5"
  const num = (v) => (!Number.isFinite(v) ? "–" : v !== 0 && Math.abs(v) < 1e-3 ? scientificText(v, 4) : formatPrecise(v, 4));
  const excluded = (s.nan ?? 0) + (s.inf ?? 0);
  const threshold = formatPrecise(s.nearZero?.threshold ?? 0.01, 3);
  const rows = [
    { id: "count", label: "valori", value: INT.format(s.count ?? 0) + (excluded ? ` (${INT.format(excluded)} non finiti esclusi)` : "") },
    { id: "mean", label: "media", value: num(s.mean) },
    { id: "std", label: "deviazione standard", value: num(s.std), title: "Quanto i valori si allontanano in media dalla media (di popolazione, come in gpt/recording.py)." },
    { id: "min", label: "minimo", value: num(s.min) },
    { id: "max", label: "massimo", value: num(s.max) },
    { id: "absMean", label: "media di |w|", value: num(s.absMean), title: "La grandezza tipica di un valore, senza segno." },
    {
      id: "nearZero",
      label: `|w| < ${threshold}`,
      value: Number.isFinite(s.nearZero?.fraction) ? `${PCT.format(s.nearZero.fraction * 100)}%` : "–",
      title: `Quota di valori quasi nulli: |w| sotto ${threshold}.`,
    },
    { id: "l2", label: "norma L2", value: num(s.l2), title: "Radice della somma dei quadrati: la lunghezza del tensore visto come un unico vettore." },
  ];
  return rows;
}
