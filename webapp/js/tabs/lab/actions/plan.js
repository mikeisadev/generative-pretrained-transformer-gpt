/**
 * Geometria delle azioni (logica pura: niente three.js, niente DOM; testata):
 * - dove mettere le matrici temporanee (attivazioni) accanto ai pesi veri della scena, allineate in modo
 *   che "riga per colonna" si veda: x sopra W (x[j] sopra la colonna j), il risultato a destra (y[d]
 *   accanto alla riga d), come nel classico schema della moltiplicazione tra matrici;
 * - come inquadrare un rettangolo del mondo nella parte dello stage davvero libera (tolti i comandi in
 *   alto, la lavagna e la barra del player in basso), così i numeri non finiscono sotto un pannello.
 *
 * Coordinate della scena del Laboratorio (scene/layout.js): x verso destra, y verso l'alto, z verso la
 * camera; una cella è CELL = 0,04; i vettori sono stirati ×3 sul lato sottile (VECTOR_STRETCH).
 */
import { padPlacement } from "../../../core/navMath.js";
import { CELL } from "../scene/layout.js";

/** Spazio tra un peso e la sua attivazione temporanea: lascia libera l'etichetta "weight (16, 64)" (0,105). */
export const LABEL_CLEARANCE = 0.13;
/** Spazio tra due carte vicine. */
export const CARD_GAP = 0.05;
/**
 * Il lato sottile delle carte-vettore temporanee (x, i prodotti, le uscite): ×2, più sottili dei vettori
 * della scena. Da vicino conta: una finestra di numeri leggibili è alta poche celle.
 */
export const TEMP_SIDE = CELL * 2;
/**
 * Quanto una carta sta davanti a ciò che copre. Le carte stanno appena sopra la scena (non molto più
 * avanti): con la camera vicina (a ~1 unità per leggere i numeri) una carta 0,1 più vicina sembrerebbe
 * più grande del 10% e x[j] non starebbe più sopra la colonna j di W.
 */
export const CARD_LIFT = 0.014;

/**
 * Rettangolo di una carta (x0, x1, y0, y1 nel mondo, celle cellW × cellH) dato l'angolo in alto a sinistra.
 * @returns {{x0: number, x1: number, y0: number, y1: number, rows: number, cols: number, cellW: number, cellH: number}}
 */
export function cardRect({ x0, y1, rows, cols, cellW = CELL, cellH = CELL }) {
  return { x0, x1: x0 + cols * cellW, y0: y1 - rows * cellH, y1, rows, cols, cellW, cellH };
}

/**
 * Una riga di `cols` celle sopra la matrice `m` (rettangolo delle sue celle), allineata alle sue colonne:
 * l'ingresso x di una nn.Linear (x[j] sopra la colonna j di W). `level` impila più righe (0 = la più vicina).
 */
export function rowAbove(m, { level = 0, rows = 1, cellH = TEMP_SIDE, gap = LABEL_CLEARANCE } = {}) {
  const y0 = m.y1 + gap + level * (rows * cellH + CARD_GAP);
  return cardRect({ x0: m.x0, y1: y0 + rows * cellH, rows, cols: m.cols, cellW: m.cellW ?? CELL, cellH });
}

/**
 * Una colonna di `rows` celle a destra di `m`, allineata alle sue righe: l'uscita y di una nn.Linear
 * (y[d] accanto alla riga d). `after` = rettangolo da scavalcare (la colonna della bias della scena).
 */
export function columnRight(m, { after = null, level = 0, cols = 1, cellW = TEMP_SIDE, gap = CARD_GAP * 1.6 } = {}) {
  const start = Math.max(m.x1, after?.x1 ?? -Infinity) + gap + level * (cols * cellW + CARD_GAP * 1.6);
  return cardRect({ x0: start, y1: m.y1, rows: m.rows, cols, cellW, cellH: m.cellH ?? CELL });
}

/** Una matrice a sinistra di `m` con le stesse righe (es. Q a sinistra dei punteggi). */
export function blockLeft(m, { cols, cellW = CELL, gap = CARD_GAP * 1.6 } = {}) {
  return cardRect({ x0: m.x0 - gap - cols * cellW, y1: m.y1, rows: m.rows, cols, cellW, cellH: m.cellH ?? CELL });
}

/** Una matrice sopra `m` con le stesse colonne (es. Kᵀ sopra i punteggi). */
export function blockAbove(m, { rows, cellH = CELL, gap = LABEL_CLEARANCE } = {}) {
  return cardRect({ x0: m.x0, y1: m.y1 + gap + rows * cellH, rows, cols: m.cols, cellW: m.cellW ?? CELL, cellH });
}

/** Una matrice a destra di `m` con le stesse righe (es. l'uscita dell'attention accanto ai pesi). */
export function blockRight(m, { cols, cellW = CELL, gap = CARD_GAP * 1.6 } = {}) {
  return cardRect({ x0: m.x1 + gap, y1: m.y1, rows: m.rows, cols, cellW, cellH: m.cellH ?? CELL });
}

/** Il sotto-rettangolo delle prime `rows` × `cols` celle di `m` (es. i punteggi T × T sopra il buffer tril 32 × 32). */
export function subRect(m, rows, cols) {
  const cellW = m.cellW ?? CELL;
  const cellH = m.cellH ?? CELL;
  return cardRect({ x0: m.x0, y1: m.y1, rows, cols, cellW, cellH });
}

/** Il centro della cella (i, j) di un rettangolo di celle (riga 0 in alto). */
export function cellCenter(r, i, j) {
  return { x: r.x0 + (j + 0.5) * (r.cellW ?? CELL), y: r.y1 - (i + 0.5) * (r.cellH ?? CELL) };
}

/** Il rettangolo delle celle [r0, r1) × [c0, c1) di `r` (per inquadrare una finestra di numeri). */
export function cellsRect(r, r0, r1, c0, c1) {
  const cw = r.cellW ?? CELL;
  const ch = r.cellH ?? CELL;
  return { x0: r.x0 + c0 * cw, x1: r.x0 + c1 * cw, y0: r.y1 - r1 * ch, y1: r.y1 - r0 * ch };
}

/** L'unione di rettangoli (quelli null si saltano). */
export function unionRect(...rects) {
  const list = rects.flat().filter(Boolean);
  if (!list.length) return null;
  return {
    x0: Math.min(...list.map((r) => r.x0)),
    x1: Math.max(...list.map((r) => r.x1)),
    y0: Math.min(...list.map((r) => r.y0)),
    y1: Math.max(...list.map((r) => r.y1)),
  };
}

/**
 * La z a cui appoggiare una carta sopra il rettangolo `rect`: appena davanti al contenuto più avanti della
 * scena che copre (contenitori, matrici, etichette: `boxes` = layout.boxes di scene/layout.js).
 */
export function zAbove(boxes, rect, { lift = CARD_LIFT, floor = 0 } = {}) {
  let z = floor;
  for (const b of boxes ?? []) {
    const r = b.matrix && (b.kind === "param" || b.kind === "buffer") ? { x0: Math.min(b.x0, b.matrix.x0), x1: Math.max(b.x1, b.matrix.x1), y0: Math.min(b.y0, b.matrix.y0), y1: Math.max(b.y1, b.matrix.y1) } : b;
    if (r.x1 < rect.x0 || r.x0 > rect.x1 || r.y1 < rect.y0 || r.y0 > rect.y1) continue;
    z = Math.max(z, (b.z ?? 0) + 0.006); // le etichette della scena stanno 0,004 davanti al loro contenitore
  }
  return z + lift;
}

/** Un rettangolo allargato di `dx`, `dy` per lato. */
export function padRect(r, dx, dy = dx) {
  return { x0: r.x0 - dx, x1: r.x1 + dx, y0: r.y0 - dy, y1: r.y1 + dy };
}

/**
 * Quante colonne di celle stanno nella parte libera dello stage con celle di almeno `minCellPx` pixel
 * (i numeri compaiono da ~36 px): la finestra da inquadrare per leggere i numeri.
 */
export function readableColumns(freeWidthPx, { minCellPx = 52, min = 5, max = 16, labelPx = 0, margin = 1.12 } = {}) {
  // le etichette a sinistra hanno una grandezza fissa sullo schermo: si tolgono i loro pixel
  const usable = (freeWidthPx - labelPx) / margin;
  return Math.max(min, Math.min(max, Math.floor(usable / minCellPx)));
}

/**
 * La posa della camera che mette il rettangolo `rect` (mondo, piano z) nella parte libera dello stage.
 * Lo stage è alto `height` px; in alto `top` px e in basso `bottom` px sono coperti (comandi, lavagna,
 * barra del player); la larghezza visibile `width` è già quella non coperta dai pannelli a destra (la
 * camera ha il "view offset": il centro orizzontale è già quello della parte visibile).
 *
 * @param {{rect: {x0: number, x1: number, y0: number, y1: number}, z?: number, fovDeg: number,
 *   width: number, height: number, top?: number, bottom?: number, margin?: number, maxScale?: number,
 *   minFree?: number}} o
 * @returns {{target: {x: number, y: number, z: number}, distance: number, scale: number, free: number}}
 *   `scale` = pixel per unità del mondo nel piano del rettangolo; la camera sta in target + direzione · distance
 */
export function framePose({ rect, z = 0, fovDeg, width, height, top = 0, bottom = 0, leftPx = 0, margin = 1.1, maxScale = 2600, minFree = 0.3 }) {
  const tanHalf = Math.tan((fovDeg * Math.PI) / 360);
  const H = Math.max(1, height);
  // se i pannelli coprono troppo (telefono in orizzontale), si restringono in proporzione: resta
  // sempre almeno `minFree` dell'altezza per la scena
  let t = Math.max(0, top);
  let b = Math.max(0, bottom);
  const room = H * (1 - minFree);
  if (t + b > room) {
    const k = room / (t + b);
    t *= k;
    b *= k;
  }
  const free = H - t - b;
  const w = Math.max(1e-6, rect.x1 - rect.x0);
  const h = Math.max(1e-6, rect.y1 - rect.y0);
  // `leftPx`: pixel da lasciare liberi a sinistra del rettangolo (le etichette delle righe, che hanno una
  // grandezza fissa sullo schermo): il rettangolo si stringe in quello che resta e si sposta a destra
  const left = Math.min(Math.max(0, leftPx), Math.max(1, width) * 0.5);
  const scale = Math.min(maxScale, free / (h * margin), Math.max(1, width - left) / (w * margin));
  const distance = H / (2 * scale * tanHalf);
  // il centro della parte libera è (t − b)/2 px sotto il centro del canvas: la camera guarda un punto
  // più in alto del centro del rettangolo, così il rettangolo scende nel mezzo della parte libera
  const offsetPx = (t - b) / 2;
  return {
    target: { x: (rect.x0 + rect.x1) / 2 - left / (2 * scale), y: (rect.y0 + rect.y1) / 2 + offsetPx / scale, z },
    distance,
    scale,
    free,
  };
}

// ---------------------------------------------------------------- lavagna, codice e pad (schermi larghi)
/** Il codice più basso che lascia il pad della navigazione nell'angolo sotto di lui (px). */
export const CODE_MIN_PX = 140;
/** Il codice non scende mai sotto questa altezza (intestazione e due righe), nemmeno con la lavagna alta. */
export const CODE_FLOOR_PX = 96;

/**
 * Dove stanno lavagna, codice e pad della navigazione su uno schermo largo (px relativi allo stage).
 *
 * Il pad del nucleo (core/ui/navPad.js) va nell'angolo in basso a destra se il codice non lo tocca,
 * altrimenti accanto al codice. Se la lavagna seguisse la posizione VERA del pad mentre il codice si
 * accorcia sopra la lavagna, i tre si inseguirebbero senza fine (lavagna stretta → più alta → codice più
 * corto → pad nell'angolo → lavagna larga → più bassa → codice più lungo → pad accanto al codice → …:
 * misurato a 1280 × 800, ~40 spostamenti del pad al secondo). Qui il posto del pad è quello nell'angolo,
 * calcolato dalla stessa funzione del nucleo (`padPlacement` di core/navMath.js, quella di navPad.js), e
 * il codice finisce sopra quel posto: il pad ci va e ci resta, e la lavagna si misura su di lui. Finestra
 * troppo bassa per un codice di CODE_MIN_PX sopra il pad: il codice resta di quell'altezza (costante), il
 * pad va dove lo mette il nucleo e la lavagna evita sia il pad vero sia il codice.
 *
 * @param {object} o
 * @param {number} o.sw larghezza dello stage   @param {number} o.sh altezza dello stage
 * @param {{left: number, right: number, top: number, bottom: number}|null} o.bar la barra del player
 * @param {object|null} [o.band] barra + chip "Segui l'azione" (il pad sta sopra entrambi); di solito = bar
 * @param {number|null} [o.chartsLeft] bordo sinistro del pannello dei grafici aperto (con i grafici aperti
 *   il nucleo non guarda il codice: conta il pad vero)
 * @param {object|null} [o.code] il pannello del codice visibile   @param {object|null} [o.pad] il pad VERO
 * @param {{w: number, h: number}} [o.padSize] il pad aperto (il nucleo lo piazza con questa misura)
 * @param {Function} [o.padBottomFn] non serve più (il posto lo calcola padPlacement del nucleo): accettato e ignorato
 * @returns {{left: number, maxW: number, codeMax: number|null, fallback: boolean, padSpot: object|null}}
 *   left e maxW della lavagna; codeMax = altezza massima del codice per il pad (null = nessun limite)
 */
export function deskLayout({ sw, sh, bar, band = bar, chartsLeft = null, code = null, pad = null, padSize = { w: 190, h: 146 } }) {
  let padSpot = null;
  let codeMax = null;
  let fallback = false;
  if (pad) {
    if (chartsLeft !== null) padSpot = pad;
    else {
      // dove il nucleo metterebbe il pad senza il codice (nell'angolo, sopra la barra e il chip se li tocca)
      const spot = padPlacement({ sw, sh, size: padSize, bar: band });
      const corner = { left: sw - spot.right - padSize.w, right: sw - spot.right, top: sh - spot.bottom - padSize.h, bottom: sh - spot.bottom };
      if (code && code.right > corner.left) {
        const fits = Math.floor(corner.top - 10 - code.top);
        if (fits >= CODE_MIN_PX) {
          codeMax = fits;
          padSpot = corner;
        } else {
          codeMax = CODE_MIN_PX;
          padSpot = pad;
          fallback = true;
        }
      } else padSpot = corner;
    }
  }
  const baseLeft = bar ? bar.left : 16;
  const barWidth = bar ? bar.right - bar.left : Math.min(720, sw - 32);
  let right = sw - 16;
  if (chartsLeft !== null) right = Math.min(right, chartsLeft - 16);
  // il pad conta se sta nella fascia della lavagna (sopra la barra del player)
  if (padSpot && (!band || padSpot.top < band.top - 4)) right = Math.min(right, padSpot.left - 10);
  if (fallback && code) right = Math.min(right, code.left - 10);
  let left = baseLeft;
  let width = Math.min(barWidth, right - baseLeft);
  // poco posto fra l'inizio della barra e il pad: la lavagna parte dal bordo sinistro dello stage
  if (right - baseLeft < 320 && baseLeft > 16) {
    left = 16;
    width = Math.min(720, right - 16);
  }
  return { left: Math.round(left), maxW: Math.round(Math.max(200, width)), codeMax, fallback, padSpot };
}

/** L'altezza massima del codice quando la lavagna gli sta sotto (si sovrappongono in orizzontale): 8 px sopra di lei. */
export function codeAboveBoard(codeMax, code, board) {
  if (!code || codeMax === null) return codeMax;
  let max = codeMax;
  if (board && board.right > code.left && board.left < code.right) max = Math.min(max, Math.floor(board.top - 8 - code.top));
  return Math.max(CODE_FLOOR_PX, max);
}
