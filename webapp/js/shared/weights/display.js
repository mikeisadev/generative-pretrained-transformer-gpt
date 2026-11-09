/**
 * Come si scrivono i numeri dei pesi dentro le celle di una matrice (logica pura: niente DOM,
 * niente three.js). La usa js/core/matrixNumbers.js; i test sono in tests/weights.test.mjs.
 *
 * - Formato: virgola decimale all'italiana come il resto dell'interfaccia (fmt di core/dom.js)
 *   e segno meno tipografico "−", più visibile del trattino nelle celle piccole.
 * - Precisione adattiva: più la cella è grande sullo schermo, più cifre decimali (da 2 a 6).
 *   Il corpo del testo è una frazione fissa della cella: zoomando cresce con lei senza
 *   ridisegnare, si ridisegna solo quando cambiano le cifre o la risoluzione.
 * - Contrasto: testo chiaro o scuro scelto con il rapporto di contrasto WCAG sul colore vero
 *   della cella (una cella arancione satura vuole il testo scuro, una quasi nera il testo chiaro).
 * - Finestre: solo le celle visibili, con un tetto, a una risoluzione scelta tra pochi gradini.
 *
 * In fondo, `shownChar`: come si scrive un carattere del vocabolario (righe e colonne delle tabelle, etichette).
 */

export const MINUS = "−";
export const MIN_DECIMALS = 2;
export const MAX_DECIMALS = 6;
/** Larghezza di una cifra in un font monospace, in em (DejaVu/Menlo/Consolas ≈ 0,6). */
export const DIGIT_EM = 0.62;
/** Sotto questo corpo (px sullo schermo) un numero non si legge: la cella resta solo colorata. */
export const MIN_FONT_PX = 10;
/** Frazione della larghezza della cella che il testo può occupare (il resto è margine). */
const WIDTH_FILL = 0.86;
/** Corpo massimo come frazione dell'altezza della cella (una riga di testo). */
const HEIGHT_FILL = 0.42;

/**
 * Un valore come testo da cella: `formatCell(-0.1234, 2)` → "−0,12".
 * Oltre 10.000 in valore assoluto si passa alla notazione esponenziale ("1,2e5").
 */
export function formatCell(value, decimals) {
  if (Number.isNaN(value)) return "NaN";
  if (value === Infinity) return "∞";
  if (value === -Infinity) return `${MINUS}∞`;
  const abs = Math.abs(value);
  let text;
  if (abs >= 1e4) text = abs.toExponential(Math.max(0, decimals - 1)).replace("e+", "e");
  else text = abs.toFixed(decimals);
  text = text.replace(".", ",");
  return value < 0 ? MINUS + text : text;
}

/**
 * Valore con tutta la precisione utile di un float32 (7 cifre significative), per il valore
 * esatto della cella sotto il mouse: `formatPrecise(-0.012345678)` → "−0,01234568".
 */
export function formatPrecise(value, digits = 7) {
  if (!Number.isFinite(value)) return formatCell(value, 0);
  if (value === 0) return "0";
  const abs = Math.abs(value);
  const exponential = !(abs >= 1e-4 && abs < 1e7);
  const [digitsPart, exponent = null] = (exponential ? abs.toExponential(digits - 1) : abs.toPrecision(digits)).split("e");
  let mantissa = digitsPart;
  if (mantissa.includes(".")) mantissa = mantissa.replace(/0+$/, "").replace(/\.$/, "");
  const text = mantissa.replace(".", ",") + (exponent === null ? "" : `e${exponent.replace("+", "")}`);
  return value < 0 ? MINUS + text : text;
}

/** Caratteri del numero più largo (con il segno) a una certa precisione. */
function widestChars(maxAbs, decimals) {
  return formatCell(-Math.abs(Number.isFinite(maxAbs) ? maxAbs : 0), decimals).length;
}

/**
 * Quante cifre e che corpo usare per celle grandi `cellW × cellH` pixel sullo schermo.
 * Regola: il massimo di decimali che sta nella cella con un corpo "comodo" (cresce piano con
 * la cella, da 10 a 15 px); se nemmeno 2 decimali stanno comodi ma stanno a 10 px, 2 decimali.
 * @param {{cellW: number, cellH: number, maxAbs?: number, minFont?: number}} o
 * @returns {{decimals: number, fontFrac: number, fontPx: number, showIndex: boolean}|null}
 *   `fontFrac` = corpo / altezza della cella; null = celle troppo piccole per i numeri
 */
export function chooseCellFormat({ cellW, cellH, maxAbs = 1, minFont = MIN_FONT_PX }) {
  if (!(cellW > 0 && cellH > 0)) return null;
  const heightLimit = cellH * HEIGHT_FILL;
  if (heightLimit < minFont) return null;
  const fit = (d) => Math.min(heightLimit, (cellW * WIDTH_FILL) / (widestChars(maxAbs, d) * DIGIT_EM));
  const comfortable = Math.min(15, Math.max(minFont, 9 + 0.08 * Math.min(cellW, cellH)));
  let decimals = null;
  for (let d = MAX_DECIMALS; d >= MIN_DECIMALS; d--) {
    if (fit(d) >= comfortable) {
      decimals = d;
      break;
    }
  }
  if (decimals === null) {
    if (fit(MIN_DECIMALS) < minFont) return null;
    decimals = MIN_DECIMALS;
  }
  const fontPx = fit(decimals);
  return { decimals, fontPx, fontFrac: fontPx / cellH, showIndex: Math.min(cellW, cellH) >= 110 };
}

// ---------------------------------------------------------------- contrasto

/** Luminanza relativa WCAG di un colore sRGB 0..255. */
export function relativeLuminance(r, g, b) {
  const lin = (c) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** Rapporto di contrasto WCAG tra due luminanze (1 … 21; il testo normale vuole almeno 4,5). */
export function contrastRatio(l1, l2) {
  const hi = Math.max(l1, l2);
  const lo = Math.min(l1, l2);
  return (hi + 0.05) / (lo + 0.05);
}

// Nero e bianco puri: con loro il contrasto migliore dei due è almeno 4,58:1 su QUALSIASI
// sfondo (livello AA del testo normale). Con un bianco o un nero appena attenuati i blu medi
// della colormap divergente scendevano a 4,2:1.
export const TEXT_DARK = "#000000";
export const TEXT_LIGHT = "#ffffff";
const L_DARK = 0;
const L_LIGHT = 1;

/** Colore del testo per una cella di colore (r, g, b): quello con il contrasto più alto. */
export function textColorFor(r, g, b) {
  const l = relativeLuminance(r, g, b);
  return contrastRatio(l, L_LIGHT) >= contrastRatio(l, L_DARK) ? TEXT_LIGHT : TEXT_DARK;
}

/** Contrasto ottenuto da textColorFor (per i test e per una legenda onesta). */
export function textContrastFor(r, g, b) {
  const l = relativeLuminance(r, g, b);
  return Math.max(contrastRatio(l, L_LIGHT), contrastRatio(l, L_DARK));
}

// ---------------------------------------------------------------- colori e forme

/**
 * Quantizzazione come gpt/export.py (`_quantize_signed`): byte 0..255 con 128 = zero,
 * simmetrica rispetto a `range` (il massimo in valore assoluto). È l'indice nella colormap.
 */
export function quantizeSigned(value, range) {
  if (!Number.isFinite(value)) return value === Infinity ? 255 : value === -Infinity ? 0 : 128;
  const q = Math.round((value / (range || 1)) * 127) + 128;
  return q < 0 ? 0 : q > 255 ? 255 : q;
}

/** Valori 0…range → byte 0..255 (colormap sequenziale, es. l'attention). */
export function quantizeUnsigned(value, range) {
  if (!Number.isFinite(value)) return value === Infinity ? 255 : 0;
  const q = Math.round((value / (range || 1)) * 255);
  return q < 0 ? 0 : q > 255 ? 255 : q;
}

/**
 * Forma di un tensore → (righe, colonne) da disegnare: 1D come una riga (bias, γ, β),
 * 2D così com'è (nn.Linear (out, in) alta `out`), oltre 2D le dimensioni iniziali in fila
 * come `tensor.reshape(-1, ultima)`.
 */
export function matrixShape(shape) {
  if (!Array.isArray(shape) || shape.length === 0) return [1, 1];
  if (shape.length === 1) return [1, shape[0]];
  const cols = shape[shape.length - 1];
  return [shape.slice(0, -1).reduce((a, b) => a * b, 1), cols];
}

// ---------------------------------------------------------------- finestre di celle

/**
 * Celle coperte da punti nel piano locale della matrice (x verso destra, y verso l'alto;
 * riga 0 in alto, come core/visuals.js). Restituisce `{r0, r1, c0, c1}` (r1, c1 esclusi),
 * già limitata alla matrice, o null se i punti cadono tutti fuori.
 * @param {{x: number, y: number}[]} points
 * @param {{minX: number, maxX: number, minY: number, maxY: number}} rect
 */
export function windowFromPoints(points, rect, rows, cols) {
  if (!points.length) return null;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const cw = (rect.maxX - rect.minX) / cols;
  const ch = (rect.maxY - rect.minY) / rows;
  const clamp = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);
  const c0 = clamp(Math.floor((minX - rect.minX) / cw), cols);
  const c1 = clamp(Math.ceil((maxX - rect.minX) / cw), cols);
  const r0 = clamp(Math.floor((rect.maxY - maxY) / ch), rows);
  const r1 = clamp(Math.ceil((rect.maxY - minY) / ch), rows);
  return r1 > r0 && c1 > c0 ? { r0, r1, c0, c1 } : null;
}

/** Finestra attorno a una cella (mezze ampiezze in celle), limitata alla matrice. */
export function windowAround(row, col, halfRows, halfCols, rows, cols) {
  const r0 = Math.max(0, Math.floor(row - halfRows));
  const r1 = Math.min(rows, Math.ceil(row + halfRows + 1));
  const c0 = Math.max(0, Math.floor(col - halfCols));
  const c1 = Math.min(cols, Math.ceil(col + halfCols + 1));
  return r1 > r0 && c1 > c0 ? { r0, r1, c0, c1 } : null;
}

export const windowCells = (w) => (w ? (w.r1 - w.r0) * (w.c1 - w.c0) : 0);

/** true se `outer` contiene tutta `inner`. */
export const windowContains = (outer, inner) => Boolean(outer && inner) && outer.r0 <= inner.r0 && outer.r1 >= inner.r1 && outer.c0 <= inner.c0 && outer.c1 >= inner.c1;

/** Allarga la finestra di una frazione per lato (almeno `minCells` celle), dentro la matrice. */
export function expandWindow(w, fraction, minCells, rows, cols) {
  const dr = Math.max(minCells, Math.ceil((w.r1 - w.r0) * fraction));
  const dc = Math.max(minCells, Math.ceil((w.c1 - w.c0) * fraction));
  return { r0: Math.max(0, w.r0 - dr), r1: Math.min(rows, w.r1 + dr), c0: Math.max(0, w.c0 - dc), c1: Math.min(cols, w.c1 + dc) };
}

/**
 * Riduce la finestra a non più di `maxCells` celle, centrata sulla cella `focus` (quella al
 * centro dello schermo) e con le stesse proporzioni. Il risultato resta dentro `w`.
 */
export function capWindow(w, focus, maxCells) {
  const h = w.r1 - w.r0;
  const width = w.c1 - w.c0;
  if (h * width <= maxCells) return { ...w };
  const k = Math.sqrt(maxCells / (h * width));
  let nh = Math.max(1, Math.floor(h * k));
  let nw = Math.max(1, Math.floor(width * k));
  // Una finestra stretta (es. una riga sola) non deve sprecare il tetto: si allarga l'altro lato.
  if (nh === 1) nw = Math.min(width, maxCells);
  if (nw === 1) nh = Math.min(h, maxCells);
  while (nh * nw > maxCells) nh > nw ? nh-- : nw--;
  const fr = Math.min(Math.max(focus.row, w.r0), w.r1 - 1);
  const fc = Math.min(Math.max(focus.col, w.c0), w.c1 - 1);
  const r0 = Math.min(Math.max(w.r0, Math.round(fr - nh / 2 + 0.5)), w.r1 - nh);
  const c0 = Math.min(Math.max(w.c0, Math.round(fc - nw / 2 + 0.5)), w.c1 - nw);
  return { r0, r1: r0 + nh, c0, c1: c0 + nw };
}

/** Gradini della risoluzione del testo (pixel del canvas per lato della cella). */
export const CELL_PX_STEPS = Object.freeze([24, 32, 48, 64, 96, 128]);

/**
 * Pixel del canvas per cella: il gradino che copre la cella sullo schermo (× pixel ratio),
 * con le proporzioni vere della cella (`aspect` = larghezza / altezza nel mondo), ridotto se il
 * canvas supererebbe `maxPixels` o `maxSide`.
 * @returns {{pw: number, ph: number}}
 */
export function canvasCellSize({ cellH, aspect = 1, pixelRatio = 1, rows, cols, maxPixels = 4_194_304, maxSide = 4096 }) {
  const wanted = cellH * pixelRatio;
  let ph = CELL_PX_STEPS.find((s) => s >= wanted) ?? CELL_PX_STEPS[CELL_PX_STEPS.length - 1];
  let pw = Math.max(1, Math.round(ph * aspect));
  const scale = Math.min(1, Math.sqrt(maxPixels / Math.max(1, pw * ph * rows * cols)), maxSide / Math.max(1, pw * cols), maxSide / Math.max(1, ph * rows));
  if (scale < 1) {
    ph = Math.max(8, Math.floor(ph * scale));
    pw = Math.max(8, Math.floor(pw * scale));
  }
  return { pw, ph };
}

// ---------------------------------------------------------------- caratteri del vocabolario

/**
 * Caratteri invisibili o che cambiano la direzione del testo: controlli C0 e C1, DEL, trattino morbido, spazi e
 * segni a larghezza zero (U+200B–U+200F, U+2060–U+206F, U+FEFF, U+180E), U+061C, separatori di riga e di
 * paragrafo (U+2028, U+2029) e i controlli bidirezionali (U+202A–U+202E, U+2066–U+2069).
 */
// eslint-disable-next-line no-control-regex
const INVISIBLE = /[\u0000-\u001f\u007f-\u009f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/g;

/**
 * Come si scrive un token del vocabolario in un'etichetta, una tabella o una frase: lo spazio «␣», l'a capo «↵»,
 * la tabulazione «⇥» (resterebbero invisibili); un altro carattere invisibile (INVISIBLE) come «U+202E». Il
 * vocabolario di un run caricato viene dai file dello studente: un U+202E nella frase «il carattere «…»» la
 * rovescerebbe. Gli altri caratteri restano come sono; null o undefined → "". Una sola regola per il Laboratorio
 * e per il Transformer (prima quattro copie, non tutte con la tabulazione).
 * @param {string|null|undefined} c
 */
export function shownChar(c) {
  if (c === " ") return "␣";
  if (c === "\n") return "↵";
  if (c === "\t") return "⇥";
  return String(c ?? "").replace(INVISIBLE, (ch) => `U+${ch.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`);
}
