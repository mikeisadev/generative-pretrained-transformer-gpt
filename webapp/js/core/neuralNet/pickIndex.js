/**
 * Indice sullo schermo per trovare in fretta il neurone sotto il puntatore (logica pura, testata con node).
 *
 * Con decine di migliaia di neuroni un raycast per ogni movimento del mouse costerebbe millisecondi.
 * Qui le posizioni si proiettano UNA volta dopo che la camera si è fermata (pochi ms per 60.000
 * neuroni) e si mettono in una griglia di celle di schermo (counting sort, niente oggetti): ogni
 * movimento del mouse guarda solo le celle vicine. Tra i neuroni abbastanza vicini al puntatore vince
 * quello più vicino alla camera (di lato è quello che si vede: la fetta davanti copre le altre).
 *
 *   const index = new ScreenIndex();
 *   index.build(positions, count, matrix, width, height, { pickRadius: (i) => r, pxPerUnit });
 *   index.query(x, y) → indice dell'unità o -1
 *   hoverUnit(label, unit, sectionIdOf) → l'unità da raccontare al passaggio del mouse (anche sotto un titolo) o -1
 */
export class ScreenIndex {
  constructor() {
    this.count = 0;
    this.capacity = 0;
    this.cell = 16;
    this.cols = 0;
    this.rows = 0;
    this.width = 0;
    this.height = 0;
    this.valid = false;
  }

  ensure(n) {
    if (n <= this.capacity) return;
    this.capacity = n;
    this.sx = new Float32Array(n);
    this.sy = new Float32Array(n);
    this.w = new Float32Array(n);
    this.r = new Float32Array(n);
    this.items = new Int32Array(n);
    this.cellOf = new Int32Array(n);
  }

  /**
   * @param {ArrayLike<number>} positions x, y, z di ogni unità (coordinate del modello)
   * @param {number} count
   * @param {ArrayLike<number>} m matrice 4×4 per colonne (come three.js: projection × view × model)
   * @param {number} width larghezza in pixel CSS
   * @param {number} height altezza in pixel CSS
   * @param {{ pickRadius?: (i: number) => number, pxPerUnit?: number, minPx?: number, maxPx?: number, cell?: number }} o
   *   pickRadius(i): raggio di presa in unità del mondo; pxPerUnit: pixel per unità a distanza 1
   *   (altezza / 2 × projection[5]); il raggio sullo schermo è r × pxPerUnit / w, tra minPx e maxPx
   */
  build(positions, count, m, width, height, { pickRadius = () => 0.1, pxPerUnit = height, minPx = 5, maxPx = 40, cell = 16 } = {}) {
    this.ensure(count);
    this.cell = cell;
    this.width = width;
    this.height = height;
    this.cols = Math.max(1, Math.ceil(width / cell));
    this.rows = Math.max(1, Math.ceil(height / cell));
    const cells = this.cols * this.rows;
    if (!this.cellStart || this.cellStart.length < cells + 1) this.cellStart = new Int32Array(cells + 1);
    const start = this.cellStart;
    start.fill(0, 0, cells + 1);
    const { sx, sy, w, r, cellOf } = this;
    let kept = 0;
    for (let i = 0; i < count; i++) {
      const x = positions[3 * i];
      const y = positions[3 * i + 1];
      const z = positions[3 * i + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      cellOf[i] = -1;
      if (cw <= 1e-6) continue; // dietro la camera
      const px = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / cw + 1) * 0.5 * width;
      const py = (1 - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw) * 0.5 * height;
      if (px < -maxPx || py < -maxPx || px > width + maxPx || py > height + maxPx) continue;
      sx[i] = px;
      sy[i] = py;
      w[i] = cw;
      r[i] = Math.min(maxPx, Math.max(minPx, (pickRadius(i) * pxPerUnit) / cw));
      const c = Math.min(this.cols - 1, Math.max(0, Math.floor(px / cell))) + this.cols * Math.min(this.rows - 1, Math.max(0, Math.floor(py / cell)));
      cellOf[i] = c;
      start[c + 1]++;
      kept++;
    }
    for (let c = 0; c < cells; c++) start[c + 1] += start[c];
    const fill = this.fillCursor && this.fillCursor.length >= cells ? this.fillCursor : (this.fillCursor = new Int32Array(cells));
    fill.set(start.subarray(0, cells));
    for (let i = 0; i < count; i++) if (cellOf[i] >= 0) this.items[fill[cellOf[i]]++] = i;
    this.count = count;
    this.kept = kept;
    this.valid = true;
    return kept;
  }

  /**
   * L'unità sotto il punto (x, y): tra quelle entro il loro raggio di presa, la più vicina alla
   * camera; a parità (stessa profondità) la più vicina al puntatore. `prefer(i)`: se qualche candidata
   * lo soddisfa (es. la fetta scelta), si sceglie solo tra quelle.
   * @returns {number} indice dell'unità o -1
   */
  query(x, y, { prefer = null, maxPx = 40 } = {}) {
    if (!this.valid) return -1;
    const { cell, cols, rows, sx, sy, w, r, items } = this;
    const c0 = Math.max(0, Math.floor((x - maxPx) / cell));
    const c1 = Math.min(cols - 1, Math.floor((x + maxPx) / cell));
    const r0 = Math.max(0, Math.floor((y - maxPx) / cell));
    const r1 = Math.min(rows - 1, Math.floor((y + maxPx) / cell));
    let best = -1;
    let bestW = Infinity;
    let bestD = Infinity;
    let bestPreferred = false;
    for (let row = r0; row <= r1; row++) {
      for (let col = c0; col <= c1; col++) {
        const c = col + row * cols;
        for (let k = this.cellStart[c]; k < this.cellStart[c + 1]; k++) {
          const i = items[k];
          const dx = sx[i] - x;
          const dy = sy[i] - y;
          const d2 = dx * dx + dy * dy;
          if (d2 > r[i] * r[i]) continue;
          const preferred = prefer ? Boolean(prefer(i)) : false;
          if (bestPreferred && !preferred) continue;
          const better = (preferred && !bestPreferred) || w[i] < bestW - 1e-4 || (Math.abs(w[i] - bestW) <= 1e-4 && d2 < bestD);
          if (better) {
            best = i;
            bestW = w[i];
            bestD = d2;
            bestPreferred = preferred;
          }
        }
      }
    }
    return best;
  }

  invalidate() {
    this.valid = false;
  }
}

/**
 * L'unità da raccontare nell'etichetta al passaggio del mouse (o -1), quando sotto il puntatore c'è anche un
 * titolo cliccabile. Il titolo di una sezione è più largo della sua colonna: di lato, con le sezioni strette e
 * vicine (le parti di una rete lunga, le lastre da lontano), passa sopra i neuroni delle sezioni accanto più
 * alte, che lì non davano nessuna etichetta (Audio a 1280 px: il titolo «Forma d'onda» copriva il 40% della
 * colonna di «Spettro del frame»). Su quei neuroni l'etichetta al passaggio ora c'è; il clic resta del titolo,
 * che è disegnato sopra (chi lo mostra lo dice nell'etichetta). Sopra i neuroni della sua sezione, fuori dai
 * neuroni e con i titoli dei gruppi, dei "× N" e dei token resta tutto come prima: niente etichetta.
 * @param {{ref?: {kind?: string, id?: string}}|null} label l'etichetta sotto il punto (LabelLayer.hit) o null
 * @param {number} unit l'unità sotto il punto entro il suo raggio di presa (ScreenIndex.query) o -1
 * @param {(unit: number) => string} sectionIdOf l'id della sezione di un'unità
 * @returns {number}
 */
export function hoverUnit(label, unit, sectionIdOf) {
  if (unit < 0 || !label) return unit;
  return label.ref?.kind === "section" && sectionIdOf(unit) !== label.ref.id ? unit : -1;
}
