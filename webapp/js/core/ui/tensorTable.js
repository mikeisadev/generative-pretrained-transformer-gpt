/**
 * Tabella dei numeri di un tensore, virtualizzata: regge una matrice 256 × 64, un embedding 65 × 64 o un
 * vettore di 65.536 valori scorrendo fluida, perché disegna solo le righe e le colonne visibili (più un
 * margine) e ricicla gli stessi elementi mentre si scorre. Come un foglio di calcolo: intestazioni di
 * riga e di colonna fisse, una cella per numero, colorata con la STESSA colormap delle heatmap della
 * scena (blu negativo, scuro ≈ 0, arancione positivo) e il testo nero o bianco scelto per contrasto
 * (≥ 4,5:1 su ogni colore, shared/weights/display.js).
 *
 *   const table = new TensorTable(container, {
 *     rowLabel: (i) => vocab[i], colLabel: (j) => `${j}`, caption: "Token embedding",
 *     onHover: (cell) => …, onSelect: (cell) => …,          // cell = { i, j, value } (onHover anche null)
 *   });
 *   table.setTensor({ name: "token_embedding_table.weight", shape: [65, 64], data });   // data non copiato
 *   table.refresh();                                          // stessi dati cambiati sul posto (un altro step)
 *   table.select(39, 3);                                      // cella attiva, portata in vista
 *   table.dispose();
 *
 * Tastiera (griglia ARIA, un solo punto di tabulazione): frecce, PagSu/PagGiù, Home/Fine (con Ctrl la
 * prima/ultima cella), Invio o Spazio sceglie la cella. La cella attiva è annunciata con
 * aria-activedescendant e un'etichetta con riga, colonna e valore preciso.
 * La logica pura (finestra virtuale, tastiera, testi) è in tensorTableMath.js.
 */
import { el } from "../dom.js";
import { colormapLUT } from "../matrixNumbers.js";
import { quantizeSigned, textColorFor } from "../../shared/weights/display.js";
import { tensorStats } from "../../shared/weights/stats.js";
import { MAX_SCROLL_PX, axisWindow, cellText, describeCell, finiteAbsMax, headerText, moveActive, scrollToShow, statsRows, tableShape, widestText } from "./tensorTableMath.js";

const CELL_H = 24; // px: una riga di testo da 12 px con aria intorno (come le righe di core.css)
const HEAD_H = 26;
const OVERSCAN_ROWS = 4;
const OVERSCAN_COLS = 2;
/** Quante etichette di riga guardare per decidere la larghezza dell'intestazione (le altre sono simili). */
const LABEL_SAMPLE = 2048;

let instances = 0;

/** I 256 colori di sfondo e di testo della colormap divergente delle heatmap (una volta per pagina). */
let palette = null;
function cellPalette() {
  if (palette) return palette;
  const lut = colormapLUT("diverging");
  const bg = new Array(256);
  const fg = new Array(256);
  for (let i = 0; i < 256; i++) {
    const r = lut[4 * i];
    const g = lut[4 * i + 1];
    const b = lut[4 * i + 2];
    bg[i] = `rgb(${r}, ${g}, ${b})`;
    fg[i] = textColorFor(r, g, b);
  }
  palette = { bg, fg };
  return palette;
}

export class TensorTable {
  /**
   * @param {HTMLElement} container dove aggiungere la tabella
   * @param {object} [o]
   * @param {(i: number) => string} [o.rowLabel] intestazione della riga i (es. il carattere di una riga dell'embedding)
   * @param {(j: number) => string} [o.colLabel] intestazione della colonna j
   * @param {string} [o.caption] titolo della tabella (anche il suo nome accessibile)
   * @param {(v: number) => string} [o.format] testo di una cella (predefinito: 4 decimali, virgola e "−")
   * @param {(cell: {i: number, j: number, value: number}|null) => void} [o.onHover] cella sotto il puntatore (null uscendo)
   * @param {(cell: {i: number, j: number, value: number}) => void} [o.onSelect] cella scelta dallo studente
   *   (clic, frecce, Invio); `select()` da codice non lo chiama
   */
  constructor(container, { rowLabel = null, colLabel = null, caption = "", format = null, onHover = null, onSelect = null } = {}) {
    if (!container?.append) throw new TypeError("TensorTable: serve un elemento contenitore");
    this.id = `tt${++instances}`;
    this.options = { rowLabel, colLabel, caption, onHover, onSelect };
    this.format = typeof format === "function" ? format : (v) => cellText(v);
    this.tensor = null;
    this.rows = 0;
    this.cols = 0;
    this.range = 1;
    this.active = null; // { row, col }
    this.hovered = null; // { row, col }
    this.version = 0; // cambia con i dati: le celle già scritte si riscrivono
    this.raf = 0;
    this.disposed = false;
    this.sizes = { cellW: 64, rowHeadW: 40, charW: 7.2 };
    this.rowPool = [];
    this.colPool = [];

    this.captionText = el("span", { className: "tt-caption-text" });
    this.captionName = el("span", { className: "tt-name" });
    this.captionEl = el("figcaption", { className: "tt-caption", id: `${this.id}-caption` }, this.captionText, this.captionName);
    this.axesEl = el("p", { className: "tt-axes", hidden: true });
    this.corner = el("div", { className: "tt-corner", role: "columnheader", ariaLabel: "riga \\ colonna" });
    this.corner.setAttribute("aria-colindex", "1");
    this.head = el("div", { className: "tt-head", role: "row" }, this.corner);
    this.head.setAttribute("aria-rowindex", "1");
    this.body = el("div", { className: "tt-body", role: "rowgroup" });
    this.probe = el("span", { className: "tt-probe", ariaHidden: "true" }, "0000000000");
    this.content = el("div", { className: "tt-content" }, this.head, this.body);
    this.grid = el("div", { className: "tt-grid", role: "grid", tabIndex: 0 }, this.content, this.probe);
    this.grid.setAttribute("aria-readonly", "true");
    this.grid.setAttribute("aria-labelledby", this.captionEl.id);
    this.status = el("p", { className: "tt-status", id: `${this.id}-status` });
    this.grid.setAttribute("aria-describedby", this.status.id);
    this.root = el("figure", { className: "tt" }, this.captionEl, this.axesEl, this.grid, this.status);
    container.append(this.root);

    this.onScroll = () => this.schedule();
    this.onKey = (e) => this.handleKey(e);
    this.onMove = (e) => this.handleMove(e);
    this.onLeave = () => this.setHover(null);
    this.onClick = (e) => this.handleClick(e);
    // Arrivando con Tab serve subito una cella attiva da annunciare; con un clic la sceglie il clic.
    this.onFocus = () => {
      if (this.tensor && !this.active && this.grid.matches(":focus-visible")) this.setActive(0, 0, { scroll: false, notify: false });
    };
    this.grid.addEventListener("scroll", this.onScroll, { passive: true });
    this.grid.addEventListener("keydown", this.onKey);
    this.grid.addEventListener("pointermove", this.onMove);
    this.grid.addEventListener("pointerleave", this.onLeave);
    this.grid.addEventListener("click", this.onClick);
    this.grid.addEventListener("focus", this.onFocus);
    // Solo eventi di layout (la tabella appare, il pannello cambia larghezza): mai lavoro per frame.
    this.resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => this.layout()) : null;
    this.resize?.observe(this.grid);
    this.renderStatus();
  }

  /**
   * Mostra un tensore. `data` è un riferimento (non viene copiato): se cambia sul posto basta refresh().
   * Campi facoltativi in più: `range` (|valore| del colore pieno; predefinito il massimo |valore|, come
   * NumericMatrix; per confrontare gli step del training conviene la scala dei pesi finali),
   * `rowTitle` / `colTitle` (cosa sono righe e colonne, es. "carattere" e "dimensione") e `caption`.
   * @param {{name?: string, shape: number[], data: ArrayLike<number>, range?: number, rowTitle?: string, colTitle?: string, caption?: string}} t
   */
  setTensor({ name = "", shape, data, range = null, rowTitle = "", colTitle = "", caption = null }) {
    if (this.disposed) return;
    const [rows, cols] = tableShape(shape ?? [data?.length ?? 0]);
    if (!data || data.length < rows * cols) throw new RangeError(`TensorTable: servono ${rows * cols} valori per la forma (${(shape ?? []).join(", ")}), ricevuti ${data?.length}`);
    this.tensor = { name: String(name ?? ""), shape: shape ? [...shape] : [data.length], data, rowTitle: rowTitle || "riga", colTitle: colTitle || "colonna", caption };
    this.rows = rows;
    this.cols = cols;
    const maxAbs = finiteAbsMax(data, rows * cols);
    this.range = range > 0 ? range : maxAbs || 1;
    this.maxAbs = maxAbs;
    this.active = null;
    this.hovered = null;
    this.version++;
    // Il nuovo tensore riparte dall'angolo in alto a sinistra.
    this.grid.scrollTop = 0;
    this.grid.scrollLeft = 0;
    this.renderCaption();
    this.grid.setAttribute("aria-rowcount", String(rows + 1));
    this.grid.setAttribute("aria-colcount", String(cols + 1));
    this.grid.removeAttribute("aria-activedescendant");
    this.measure();
    this.render();
    this.renderStatus();
  }

  /** Gli stessi dati sono cambiati sul posto (es. un altro step del training): ridisegna, la scala resta. */
  refresh() {
    if (!this.tensor || this.disposed) return;
    this.version++;
    this.maxAbs = finiteAbsMax(this.tensor.data, this.rows * this.cols);
    this.render();
    this.renderStatus();
  }

  /** Evidenzia la cella (i, j) come attiva e, con `scroll`, la porta in vista. Non chiama onSelect. */
  select(i, j, { scroll = true } = {}) {
    if (!this.tensor || this.disposed) return;
    this.setActive(i, j, { scroll, notify: false });
  }

  /** La cella attiva { i, j, value } o null. */
  get selected() {
    return this.active ? this.cellInfo(this.active.row, this.active.col) : null;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resize?.disconnect();
    this.grid.removeEventListener("scroll", this.onScroll);
    this.grid.removeEventListener("keydown", this.onKey);
    this.grid.removeEventListener("pointermove", this.onMove);
    this.grid.removeEventListener("pointerleave", this.onLeave);
    this.grid.removeEventListener("click", this.onClick);
    this.grid.removeEventListener("focus", this.onFocus);
    this.root.remove();
    this.rowPool.length = 0;
    this.colPool.length = 0;
    this.tensor = null;
  }

  // ---------------------------------------------------------------- interno: misure e disegno

  cellInfo(row, col) {
    return { i: row, j: col, value: this.tensor.data[row * this.cols + col] };
  }

  /** Larghezza delle colonne (il numero più largo) e delle intestazioni di riga (l'etichetta più lunga). */
  measure() {
    const w = this.probe.getBoundingClientRect().width;
    if (w > 0) this.sizes.charW = w / 10;
    const { charW } = this.sizes;
    const chars = widestText(this.maxAbs ?? 0, this.format);
    let labelChars = String(Math.max(0, this.rows - 1)).length;
    if (typeof this.options.rowLabel === "function" && this.rows) {
      const step = Math.max(1, Math.floor(this.rows / LABEL_SAMPLE));
      for (let i = 0; i < this.rows; i += step) labelChars = Math.max(labelChars, headerText(i, this.options.rowLabel).length);
      labelChars = Math.max(labelChars, headerText(this.rows - 1, this.options.rowLabel).length);
    }
    this.sizes.cellW = Math.max(44, Math.ceil(chars * charW + 14));
    this.sizes.rowHeadW = Math.min(180, Math.max(36, Math.ceil(labelChars * charW + 16)));
    this.root.style.setProperty("--tt-cell-w", `${this.sizes.cellW}px`);
    this.root.style.setProperty("--tt-rowhead-w", `${this.sizes.rowHeadW}px`);
  }

  /** Cambio di layout (la tabella diventa visibile, il pannello si allarga): rimisura e ridisegna. */
  layout() {
    if (!this.tensor || this.disposed) return;
    this.measure(); // nascosta (pannello chiuso) la tabella non si poteva misurare: ora sì
    this.render();
  }

  schedule() {
    if (this.raf || this.disposed) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.render();
    });
  }

  windows() {
    const { cellW, rowHeadW } = this.sizes;
    const vw = Math.max(0, this.grid.clientWidth - rowHeadW);
    const vh = Math.max(0, this.grid.clientHeight - HEAD_H);
    const wx = axisWindow({ count: this.cols, size: cellW, viewport: vw, scroll: this.grid.scrollLeft, overscan: OVERSCAN_COLS });
    const wy = axisWindow({ count: this.rows, size: CELL_H, viewport: vh, scroll: this.grid.scrollTop, overscan: OVERSCAN_ROWS });
    return { wx, wy, vw, vh };
  }

  /** Disegna la finestra visibile riciclando gli elementi (righe, celle, intestazioni di colonna). */
  render() {
    if (!this.tensor || this.disposed) return;
    const { cellW, rowHeadW } = this.sizes;
    // Prima le dimensioni dell'area di scorrimento: l'altezza visibile della griglia dipende da loro
    // (una matrice di 4 righe non ha bisogno di una finestra alta 300 px), poi la finestra virtuale.
    this.content.style.width = `${rowHeadW + Math.min(this.cols * cellW, MAX_SCROLL_PX)}px`;
    this.body.style.height = `${Math.min(this.rows * CELL_H, MAX_SCROLL_PX)}px`;
    const { wx, wy } = this.windows();
    const { bg, fg } = cellPalette();
    const { data } = this.tensor;
    const cols = this.cols;
    const range = this.range;
    const { rowLabel, colLabel } = this.options;
    const active = this.active;

    // intestazioni di colonna
    let k = 0;
    for (let j = wx.first; j < wx.last; j++, k++) {
      const head = this.colPool[k] ?? this.newColHead(k);
      if (head.j !== j) {
        head.j = j;
        head.el.textContent = headerText(j, colLabel);
        head.el.setAttribute("aria-colindex", String(j + 2));
        head.el.title = head.el.textContent;
      }
      head.el.style.left = `${rowHeadW + j * cellW + wx.shift}px`;
      head.el.classList.toggle("on", active?.col === j);
      if (!head.el.isConnected) this.head.append(head.el);
    }
    for (let m = k; m < this.colPool.length; m++) this.colPool[m].el.remove();

    // righe e celle
    let activeId = null;
    let r = 0;
    for (let i = wy.first; i < wy.last; i++, r++) {
      const row = this.rowPool[r] ?? this.newRow(r);
      if (row.i !== i) {
        row.i = i;
        row.el.setAttribute("aria-rowindex", String(i + 2));
        row.head.textContent = headerText(i, rowLabel);
        row.head.title = row.head.textContent;
      }
      row.el.style.top = `${i * CELL_H + wy.shift}px`;
      row.head.classList.toggle("on", active?.row === i);
      let c = 0;
      for (let j = wx.first; j < wx.last; j++, c++) {
        const cell = row.cells[c] ?? this.newCell(row, c);
        const el = cell.el;
        if (cell.i !== i || cell.j !== j || cell.version !== this.version) {
          const v = data[i * cols + j];
          const q = quantizeSigned(v, range);
          el.textContent = this.format(v);
          el.style.background = bg[q];
          el.style.color = fg[q];
          el.setAttribute("aria-colindex", String(j + 2));
          el.id = `${this.id}-${i}-${j}`;
          cell.i = i;
          cell.j = j;
          cell.version = this.version;
        }
        el.style.left = `${rowHeadW + j * cellW + wx.shift}px`;
        const isActive = active?.row === i && active?.col === j;
        el.classList.toggle("active", isActive);
        el.classList.toggle("hover", this.hovered?.row === i && this.hovered?.col === j);
        if (isActive) {
          activeId = el.id;
          el.setAttribute("aria-label", describeCell(this.cellInfoRC(i, j), this.describeOptions()));
          el.setAttribute("aria-selected", "true");
        } else if (el.hasAttribute("aria-label")) {
          el.removeAttribute("aria-label");
          el.removeAttribute("aria-selected");
        }
        if (!el.isConnected) row.el.append(el);
      }
      for (let m = c; m < row.cells.length; m++) row.cells[m].el.remove();
      if (!row.el.isConnected) this.body.append(row.el);
    }
    for (let m = r; m < this.rowPool.length; m++) this.rowPool[m].el.remove();
    if (activeId) this.grid.setAttribute("aria-activedescendant", activeId);
    else this.grid.removeAttribute("aria-activedescendant");
  }

  newColHead(k) {
    const head = { el: el("div", { className: "tt-colhead", role: "columnheader" }), j: -1 };
    this.colPool[k] = head;
    return head;
  }

  newRow(k) {
    const head = el("div", { className: "tt-rowhead", role: "rowheader" });
    head.setAttribute("aria-colindex", "1");
    const row = { el: el("div", { className: "tt-row", role: "row" }, head), head, cells: [], i: -1 };
    this.rowPool[k] = row;
    return row;
  }

  newCell(row, c) {
    const cell = { el: el("div", { className: "tt-cell", role: "gridcell" }), i: -1, j: -1, version: -1 };
    row.cells[c] = cell;
    return cell;
  }

  cellInfoRC(row, col) {
    return { row, col, value: this.tensor.data[row * this.cols + col] };
  }

  describeOptions() {
    const t = this.tensor;
    return { rowLabel: this.options.rowLabel, colLabel: this.options.colLabel, rowTitle: t.rowTitle, colTitle: t.colTitle };
  }

  renderCaption() {
    const t = this.tensor;
    const caption = t?.caption ?? this.options.caption ?? "";
    this.captionText.textContent = caption ? `${caption} ` : "";
    this.captionName.textContent = t ? `${t.name}${t.name ? " " : ""}(${t.shape.join(", ")}${t.shape.length === 1 ? "," : ""})` : "";
    const axes = t && (t.rowTitle !== "riga" || t.colTitle !== "colonna");
    this.axesEl.hidden = !axes;
    if (axes) this.axesEl.textContent = this.cols === 1 ? `Righe: ${t.rowTitle} (${this.rows}). Una colonna: il vettore si legge dall'alto in basso.` : `Righe: ${t.rowTitle} (${this.rows}) · colonne: ${t.colTitle} (${this.cols})`;
  }

  renderStatus() {
    if (!this.tensor) {
      this.status.textContent = "";
      return;
    }
    if (!this.active) {
      this.status.textContent = "Frecce, PagSu/PagGiù e Home/Fine per muoverti tra le celle; Invio sceglie la cella.";
      return;
    }
    this.status.textContent = describeCell(this.cellInfoRC(this.active.row, this.active.col), this.describeOptions());
  }

  // ---------------------------------------------------------------- interno: cella attiva, puntatore, tastiera

  setActive(row, col, { scroll = true, notify = false } = {}) {
    const r = Math.min(this.rows - 1, Math.max(0, Math.floor(row)));
    const c = Math.min(this.cols - 1, Math.max(0, Math.floor(col)));
    if (!(r >= 0 && c >= 0)) return;
    this.active = { row: r, col: c };
    if (scroll) {
      const { cellW, rowHeadW } = this.sizes;
      const vw = Math.max(0, this.grid.clientWidth - rowHeadW);
      const vh = Math.max(0, this.grid.clientHeight - HEAD_H);
      this.grid.scrollTop = scrollToShow({ index: r, count: this.rows, size: CELL_H, viewport: vh, current: this.grid.scrollTop });
      this.grid.scrollLeft = scrollToShow({ index: c, count: this.cols, size: cellW, viewport: vw, current: this.grid.scrollLeft });
    }
    this.render(); // subito: la cella attiva deve esistere per aria-activedescendant
    this.renderStatus();
    if (notify) this.options.onSelect?.(this.cellInfo(r, c));
  }

  setHover(cell) {
    const same = cell && this.hovered && cell.row === this.hovered.row && cell.col === this.hovered.col;
    if (same || (!cell && !this.hovered)) return;
    this.hovered = cell;
    for (const row of this.rowPool) for (const c of row.cells) c.el.classList.toggle("hover", Boolean(cell) && c.i === cell.row && c.j === cell.col && c.el.isConnected);
    this.options.onHover?.(cell ? this.cellInfo(cell.row, cell.col) : null);
  }

  /** Cella dell'elemento sotto il puntatore (le celle riciclate sanno quale cella mostrano). */
  cellFromEvent(e) {
    const target = e.target instanceof Element ? e.target.closest(".tt-cell") : null;
    if (!target) return null;
    for (const row of this.rowPool) for (const c of row.cells) if (c.el === target) return { row: c.i, col: c.j };
    return null;
  }

  handleMove(e) {
    if (!this.tensor) return;
    this.setHover(this.cellFromEvent(e));
  }

  handleClick(e) {
    if (!this.tensor) return;
    const cell = this.cellFromEvent(e);
    if (!cell) return;
    this.setActive(cell.row, cell.col, { scroll: false, notify: true });
    this.grid.focus({ preventScroll: true }); // dopo: il focus non deve scegliere la cella (0, 0)
  }

  handleKey(e) {
    if (!this.tensor || e.altKey) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault(); // niente pausa del player né clic su altro
      if (!this.active) this.setActive(0, 0, { scroll: true, notify: false });
      this.options.onSelect?.(this.cellInfo(this.active.row, this.active.col));
      return;
    }
    const pageRows = Math.max(1, Math.floor((this.grid.clientHeight - HEAD_H) / CELL_H) - 1);
    const next = moveActive(this.active ?? { row: 0, col: 0 }, e.key, { rows: this.rows, cols: this.cols, pageRows, ctrl: e.ctrlKey || e.metaKey });
    if (!next) return;
    e.preventDefault(); // le frecce sono della tabella: né la pagina né la scena 3D si muovono
    this.setActive(next.row, next.col, { scroll: true, notify: true });
  }
}

/**
 * Le statistiche di un tensore in una scheda compatta: numero di valori, media, deviazione standard,
 * minimo, massimo, media di |w|, quota di |w| < 0,01, norma L2 (calcolate da shared/weights/stats.js).
 * Una seconda chiamata sullo stesso contenitore sostituisce la scheda precedente.
 * @param {HTMLElement} container
 * @param {ArrayLike<number>} data
 * @param {{label?: string}} [o] titolo della scheda
 * @returns {HTMLElement}
 */
export function renderTensorStats(container, data, { label } = {}) {
  const stats = tensorStats(data, { nearZero: 0.01, percentiles: [], histogram: false });
  const items = statsRows(stats).map((row) => el("div", { className: "tt-stat", dataset: { stat: row.id } }, el("dt", { title: row.title ?? "" }, row.label), el("dd", {}, row.value)));
  const box = el("section", { className: "tt-stats", dataset: { tensorStats: "" } }, label ? el("h5", {}, label) : null, el("dl", {}, ...items));
  const previous = container.querySelector(":scope > [data-tensor-stats]");
  if (previous) previous.replaceWith(box);
  else container.append(box);
  return box;
}
