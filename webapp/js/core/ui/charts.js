/**
 * Pannello dei grafici: pulsante fluttuante "Grafici" + pannello laterale destro con sezioni e
 * card. Cinque tipi (line, bar, histogram, heatmap, scatter) disegnati su canvas HiDPI.
 *
 *   const section = ui.charts.section("training", "Training", "descrizione breve");
 *   const loss = section.chart("loss", { type: "line", title: "Loss", description: "…", yScale: "log" });
 *   loss.setData({ series: [{ name: "train", points: [[0, 4.4]] }] });
 *   loss.append("train", [100, 3.1]);   // ridisegno pigro: al più uno per frame
 *
 * Scelte di progetto:
 * - Il ridisegno è PIGRO: setData/append segnano il grafico come "sporco" e un solo
 *   requestAnimationFrame per pannello ridisegna i grafici sporchi. A pannello chiuso non si
 *   disegna nulla: i grafici si aggiornano quando il pannello si apre. Così append() chiamato
 *   centinaia di volte al secondo costa un push in un array.
 * - La matematica (scale, tick, bin, decimazione, colori) sta in chartMath.js, testata in Node.
 * - Palette e regole visive dalla skill dataviz: colori categoriali in ordine fisso e legati al
 *   nome della serie (non alla posizione), linee da 2 px, barre sottili, griglia tenue,
 *   tooltip su ogni grafico, legenda con 2 o più serie, tabella dei dati come alternativa.
 */
import { el } from "../dom.js";
import {
  OVERFLOW_COLOR,
  PALETTE,
  THEME,
  DIVERGING_STOPS,
  SEQUENTIAL_STOPS,
  bandLayout,
  buildColorTable,
  colorDomain,
  colorPosition,
  decimateMinMax,
  formatPercent,
  formatTick,
  formatValue,
  histogram,
  isFiniteNumber,
  linearScale,
  logScale,
  logTicks,
  nearestIndexSorted,
  niceTicks,
  placeLabels,
  summarize,
  visibleLabel,
  visibleRange,
} from "./chartMath.js";

const FONT = "11px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
const FONT_STRONG = "600 11px Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
const EMPTY_TEXT = "In attesa di dati";
const EMPTY_HEIGHT = 120;
const TABLE_MAX_ROWS = 200; // la tabella è un'alternativa leggibile, non un dump di 100.000 righe
const NARROW_QUERY = "(max-width: 899px)"; // sotto i 900 px il pannello copre tutto lo stage

/* ================================================================================================
 * Pannello
 * ============================================================================================== */

let panelCount = 0;

export class ChartsPanel {
  /**
   * @param {HTMLElement} parent contenitore del tab (il pulsante si posiziona sullo stage via core.css)
   * @param {object} [o] @param {string} [o.title="Grafici"]
   */
  constructor(parent, { title = "Grafici" } = {}) {
    ensureStylesheet();
    const id = `charts-panel-${++panelCount}`;
    this.sections = new Map();
    this.listeners = new Set();
    this.queue = new RenderQueue(() => this.isOpen);

    this.button = el(
      "button",
      { type: "button", className: "charts-fab", ariaExpanded: "false", onclick: () => this.toggle() },
      chartIcon(),
      el("span", {}, title),
    );
    this.button.setAttribute("aria-controls", id);
    this.empty = el("p", { className: "charts-empty" }, "Nessun grafico per questa vista.");
    this.body = el("div", { className: "charts-body" }, this.empty);
    this.root = el(
      "aside",
      { id, className: "charts-panel", ariaLabel: title, inert: true, onkeydown: (e) => e.key === "Escape" && this.close() },
      el(
        "header",
        { className: "charts-header" },
        el("h2", {}, title),
        el("button", { type: "button", className: "icon-button", ariaLabel: "Chiudi i grafici", onclick: () => this.close() }, "✕"),
      ),
      this.body,
    );
    applyThemeVariables(this.root);
    parent.append(this.button, this.root);

    // Un solo ResizeObserver per tutte le card: quando cambia la larghezza si ridisegna.
    this.resizeObserver = new ResizeObserver((entries) => entries.forEach((e) => e.target.chart?.invalidate()));
    // Attraversare i 900 px cambia coveredWidth(): la scena deve saperlo.
    this.narrow = window.matchMedia(NARROW_QUERY);
    this.onMedia = () => this.notify();
    this.narrow.addEventListener("change", this.onMedia);
  }

  /** Crea la sezione `id` o la riusa (aggiornando titolo e descrizione se forniti). */
  section(id, title = id, description = "") {
    let section = this.sections.get(id);
    if (!section) {
      section = new ChartSection(this, title, description);
      this.sections.set(id, section);
      this.body.append(section.root);
      this.empty.hidden = true;
    } else if (arguments.length > 1) {
      section.setText(title, description);
    }
    return section;
  }

  get isOpen() {
    return this.root.classList.contains("open");
  }

  open() {
    this.setOpen(true);
  }

  close() {
    this.setOpen(false);
  }

  toggle() {
    this.setOpen(!this.isOpen);
  }

  setOpen(open) {
    if (open === this.isOpen) return;
    this.root.classList.toggle("open", open);
    this.root.inert = !open;
    this.button.ariaExpanded = String(open);
    if (open) {
      this.queue.flush(); // i grafici aggiornati a pannello chiuso si disegnano adesso
      this.root.querySelector(".charts-header button")?.focus({ preventScroll: true });
    } else if (this.root.contains(document.activeElement)) {
      this.button.focus({ preventScroll: true });
    }
    this.notify();
  }

  /** Larghezza coperta sul lato destro dello stage: 0 se chiuso o su schermi sotto i 900 px. */
  coveredWidth() {
    if (!this.isOpen || this.narrow.matches) return 0;
    return this.root.offsetWidth;
  }

  /** `fn()` viene chiamata a ogni apertura/chiusura (e cambio di coveredWidth). Restituisce la funzione di rimozione. */
  onResize(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  notify() {
    for (const fn of this.listeners) fn();
  }

  /** Rimuove sezioni, osservatori e DOM (quando il tab viene distrutto). */
  destroy() {
    for (const section of this.sections.values()) section.clear();
    this.sections.clear();
    this.resizeObserver.disconnect();
    this.narrow.removeEventListener("change", this.onMedia);
    this.queue.cancel();
    this.button.remove();
    this.root.remove();
    this.listeners.clear();
  }
}

/** Gruppo di card con titolo e descrizione. */
class ChartSection {
  constructor(panel, title, description) {
    this.panel = panel;
    this.charts = new Map();
    this.title = el("h3", {}, title);
    this.description = el("p", { className: "charts-section-desc" }, description);
    this.description.hidden = !description;
    this.cards = el("div", { className: "charts-cards" });
    this.root = el("section", { className: "charts-section" }, this.title, this.description, this.cards);
  }

  setText(title, description) {
    this.title.textContent = title;
    this.description.textContent = description ?? "";
    this.description.hidden = !description;
  }

  /**
   * Crea il grafico `id` o lo riusa. Se il tipo cambia, la card viene ricreata.
   * @param {string} id
   * @param {ChartOptions} options
   * @returns {Chart}
   */
  chart(id, options = {}) {
    const existing = this.charts.get(id);
    if (existing && !existing.destroyed && existing.type === (options.type ?? existing.type)) {
      existing.setText(options.title, options.description);
      return existing;
    }
    const chart = new Chart(this.panel, options);
    if (existing) existing.root.replaceWith(chart.root);
    else this.cards.append(chart.root);
    existing?.destroy();
    this.charts.set(id, chart);
    return chart;
  }

  /** Svuota la sezione: rimuove tutte le sue card (i riferimenti vecchi diventano inerti). */
  clear() {
    for (const chart of this.charts.values()) chart.destroy();
    this.charts.clear();
    this.cards.replaceChildren();
  }
}

/**
 * Coda di ridisegno: un solo requestAnimationFrame per pannello, qualunque sia il numero di
 * append() nel frame. Se il pannello è chiuso i grafici restano in coda fino a flush().
 */
class RenderQueue {
  constructor(isVisible) {
    this.isVisible = isVisible;
    this.pending = new Set();
    this.frame = 0;
  }

  request(chart) {
    this.pending.add(chart);
    if (!this.frame && this.isVisible()) this.frame = requestAnimationFrame(() => this.run());
  }

  flush() {
    if (this.pending.size && !this.frame) this.frame = requestAnimationFrame(() => this.run());
  }

  run() {
    this.frame = 0;
    const charts = [...this.pending];
    this.pending.clear();
    for (const chart of charts) {
      try {
        chart.render();
      } catch (error) {
        // Un grafico con dati inattesi non deve bloccare gli altri né il tab.
        console.error(`Grafico "${chart.options.title ?? chart.type}":`, error);
      }
    }
  }

  cancel() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.pending.clear();
  }
}

/* ================================================================================================
 * Card di un grafico
 * ============================================================================================== */

/**
 * @typedef {object} ChartOptions
 * @property {"line"|"bar"|"histogram"|"heatmap"|"scatter"} type
 * @property {string} [title] @property {string} [description] "a cosa serve"
 * @property {string} [xLabel] @property {string} [yLabel]
 * @property {"linear"|"log"} [yScale="linear"] solo line
 * @property {number} [yMin] @property {number} [yMax] dominio y fisso (line, scatter)
 * @property {number} [height] altezza dell'area del grafico in px
 * @property {boolean} [percent] valori in 0…1 mostrati come percentuali (bar)
 * @property {(v:number)=>string} [format] formato dei valori in tooltip ed etichette
 * @property {string} [highlightLabel] voce di legenda della barra evidenziata (bar)
 */

class Chart {
  constructor(panel, options) {
    const Renderer = RENDERERS[options.type];
    if (!Renderer) throw new Error(`Tipo di grafico sconosciuto: "${options.type}". Tipi validi: ${Object.keys(RENDERERS).join(", ")}.`);
    this.panel = panel;
    this.type = options.type;
    this.options = options;
    this.colors = new Map(); // nome serie/gruppo → colore, stabile per tutta la vita del grafico
    this.renderer = new Renderer(this, options);
    this.pointer = null;
    this.destroyed = false;
    this.cache = { legend: null, note: null };
    this.tableDirty = true;

    this.titleEl = el("h4", { className: "chart-title" });
    this.descEl = el("p", { className: "chart-desc" });
    this.legendEl = el("div", { className: "chart-legend", role: "list" });
    this.canvas = el("canvas", { className: "chart-canvas", role: "img" });
    this.tooltip = el("div", { className: "chart-tooltip", hidden: true, ariaHidden: "true" }); // la tabella è l'alternativa accessibile
    this.plot = el("div", { className: "chart-plot" }, this.canvas, this.tooltip);
    this.plot.chart = this;
    this.noteEl = el("p", { className: "chart-note", hidden: true });
    this.tableBody = el("div", { className: "chart-table-body" });
    this.tableEl = el(
      "details",
      { className: "chart-table", ontoggle: () => this.tableEl.open && this.renderTable() },
      el("summary", {}, "Dati in tabella"),
      this.tableBody,
    );
    this.root = el("article", { className: `chart-card chart-${this.type}` }, el("header", {}, this.titleEl, this.descEl), this.legendEl, this.plot, this.noteEl, this.tableEl);
    this.setText(options.title ?? "", options.description ?? "");

    this.canvas.addEventListener("pointermove", (e) => this.setPointer(e));
    this.canvas.addEventListener("pointerdown", (e) => this.setPointer(e));
    this.canvas.addEventListener("pointerleave", () => this.setPointer(null));
    panel.resizeObserver.observe(this.plot);
    this.invalidate();
  }

  setText(title, description) {
    if (title !== undefined) {
      this.titleEl.textContent = title;
      this.options.title = title;
    }
    if (description !== undefined) {
      this.descEl.textContent = description;
      this.descEl.hidden = !description;
      this.options.description = description;
    }
    this.canvas.ariaLabel = [this.options.title, this.options.description].filter(Boolean).join(". ") || "Grafico";
  }

  /** Sostituisce i dati (formato per tipo: vedi ARCHITECTURE.md, "Grafici"). */
  setData(data) {
    if (this.destroyed) return;
    this.renderer.setData(data ?? {});
    this.invalidate(true);
  }

  /** Aggiunta incrementale: line → append(nomeSerie, [x, y] | [[x, y], …]); scatter → append({x, y, …} | [...]). */
  append(...args) {
    if (this.destroyed) return;
    if (!this.renderer.append) throw new Error(`append() non è disponibile per i grafici "${this.type}": usa setData().`);
    this.renderer.append(...args);
    this.invalidate(true);
  }

  /** Evidenzia un elemento: line/histogram {x}; bar {index}; heatmap {row, col}; scatter {index} o {label}. null toglie. */
  setHighlight(highlight) {
    if (this.destroyed) return;
    this.renderer.highlight = highlight ?? null;
    this.invalidate(true);
  }

  clear() {
    if (this.destroyed) return;
    this.renderer.clear();
    this.invalidate(true);
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.panel.resizeObserver.unobserve(this.plot);
    this.panel.queue.pending.delete(this);
    this.root.remove();
  }

  /** Colore categoriale della serie `name`: assegnato alla prima comparsa, poi fisso. */
  colorFor(name) {
    if (!this.colors.has(name)) this.colors.set(name, PALETTE[this.colors.size] ?? OVERFLOW_COLOR);
    return this.colors.get(name);
  }

  /** Formato dei valori (tooltip, etichette): option format > percent > automatico. */
  formatValue(v) {
    if (this.options.format) return this.options.format(v);
    return this.options.percent ? formatPercent(v) : formatValue(v);
  }

  invalidate(dataChanged = false) {
    if (dataChanged) this.tableDirty = true;
    if (!this.destroyed) this.panel.queue.request(this);
  }

  setPointer(event) {
    if (!event) this.pointer = null;
    else {
      const rect = this.canvas.getBoundingClientRect();
      this.pointer = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }
    this.invalidate();
  }

  /** Disegno vero e proprio: chiamato solo dalla RenderQueue, al più una volta per frame. */
  render() {
    if (this.destroyed) return;
    const width = Math.floor(this.plot.clientWidth);
    if (width <= 0) return; // card non ancora in pagina: il ResizeObserver ci richiamerà
    const empty = this.renderer.isEmpty();
    const height = Math.round(empty ? EMPTY_HEIGHT : (this.options.height ?? this.renderer.preferredHeight(width)));
    if (this.plot.style.height !== `${height}px`) this.plot.style.height = `${height}px`;

    const dpr = window.devicePixelRatio || 1;
    const [cw, ch] = [Math.round(width * dpr), Math.round(height * dpr)];
    if (this.canvas.width !== cw || this.canvas.height !== ch) {
      this.canvas.width = cw;
      this.canvas.height = ch;
      this.canvas.style.width = `${width}px`;
      this.canvas.style.height = `${height}px`;
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = FONT;
    ctx.textBaseline = "middle";

    const frame = { width, height };
    this.tableEl.hidden = empty; // niente tabella vuota da aprire
    if (empty) {
      drawEmpty(ctx, frame);
      this.showTooltip(null);
    } else {
      this.renderer.draw(ctx, frame);
      const hit = this.pointer ? this.renderer.hitTest(this.pointer.x, this.pointer.y) : null;
      if (hit) this.renderer.drawHover(ctx, hit);
      this.showTooltip(hit, frame);
    }
    this.updateLegend();
    this.updateNote();
    if (this.tableEl.open && this.tableDirty) this.renderTable();
  }

  updateLegend() {
    const items = this.renderer.legend();
    const key = JSON.stringify(items);
    if (key === this.cache.legend) return;
    this.cache.legend = key;
    this.legendEl.replaceChildren(...items.map(legendItem));
    this.legendEl.hidden = items.length === 0;
  }

  updateNote() {
    const note = this.renderer.isEmpty() ? "" : this.renderer.note();
    if (note === this.cache.note) return;
    this.cache.note = note;
    this.noteEl.textContent = note;
    this.noteEl.hidden = !note;
  }

  renderTable() {
    this.tableDirty = false;
    const table = this.renderer.isEmpty() ? null : this.renderer.table();
    if (!table) {
      this.tableBody.replaceChildren(el("p", { className: "chart-note" }, EMPTY_TEXT));
      return;
    }
    const head = el("tr", {}, ...table.columns.map((c) => el("th", { scope: "col" }, c)));
    const rows = table.rows.map((r) => el("tr", {}, ...r.map((cell) => el("td", {}, String(cell)))));
    this.tableBody.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows)), ...(table.note ? [el("p", { className: "chart-note" }, table.note)] : []));
  }

  /** Tooltip HTML (testo inserito con textContent: nomi e etichette sono dati, non HTML). */
  showTooltip(hit, frame) {
    if (!hit?.tooltip) {
      this.tooltip.hidden = true;
      return;
    }
    const { title, rows } = hit.tooltip;
    this.tooltip.replaceChildren(
      ...(title ? [el("div", { className: "chart-tooltip-title" }, title)] : []),
      ...rows.map((r) =>
        el(
          "div",
          { className: "chart-tooltip-row" },
          r.color ? el("span", { className: `chart-key chart-key-${r.key ?? "line"}`, style: { background: r.color } }) : null,
          el("strong", {}, r.value),
          r.name ? el("span", {}, r.name) : null,
        ),
      ),
    );
    this.tooltip.hidden = false;
    // Accanto al puntatore, girato a sinistra se uscirebbe dalla card.
    const { offsetWidth: tw, offsetHeight: th } = this.tooltip;
    const ax = hit.anchor?.x ?? this.pointer.x;
    const ay = hit.anchor?.y ?? this.pointer.y;
    const left = ax + 14 + tw > frame.width ? Math.max(0, ax - 14 - tw) : ax + 14;
    const top = Math.min(Math.max(0, ay - th / 2), Math.max(0, frame.height - th));
    this.tooltip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
}

/* ================================================================================================
 * Disegno condiviso: assi, griglia, stato vuoto
 * ============================================================================================== */

const MARGIN = { top: 22, right: 12, tickBand: 18, labelBand: 16 };

/** Rettangolo dell'area dati, lasciando spazio alle etichette dei tick y (misurate) e agli assi. */
function plotArea(ctx, { width, height }, yTickLabels, { right = MARGIN.right, xLabel = "", top = MARGIN.top } = {}) {
  ctx.font = FONT;
  const yw = yTickLabels.reduce((m, t) => Math.max(m, ctx.measureText(t).width), 0);
  const left = Math.ceil(yw) + 10;
  const bottom = height - MARGIN.tickBand - (xLabel ? MARGIN.labelBand : 0);
  return { left, top, right: width - right, bottom, width: width - right - left, height: bottom - top };
}

/** Griglia orizzontale tenue + etichette dei tick y + titolo dell'asse y (in alto a sinistra). */
function drawYAxis(ctx, plot, ticks, scale, labels, yLabel) {
  ctx.lineWidth = 1;
  ctx.font = FONT;
  ctx.textAlign = "right";
  ticks.forEach((t, i) => {
    const y = crisp(scale(t));
    ctx.strokeStyle = THEME.grid;
    line(ctx, plot.left, y, plot.right, y);
    ctx.fillStyle = THEME.muted;
    ctx.fillText(labels[i], plot.left - 6, y);
  });
  if (yLabel) {
    ctx.textAlign = "left";
    ctx.fillStyle = THEME.muted;
    ctx.fillText(yLabel, 0, 8);
  }
}

/** Linea di base, tick x e titolo dell'asse x (in basso a destra). */
function drawXAxis(ctx, plot, ticks, scale, format, xLabel, { grid = false } = {}) {
  ctx.lineWidth = 1;
  ctx.font = FONT;
  ctx.textAlign = "center";
  const boxes = ticks.map((t) => {
    const w = ctx.measureText(format(t)).width;
    return { x: scale(t) - w / 2, y: 0, w, h: 10 };
  });
  const keep = placeLabels(boxes, 6);
  ticks.forEach((t, i) => {
    const x = crisp(scale(t));
    if (grid) {
      ctx.strokeStyle = THEME.grid;
      line(ctx, x, plot.top, x, plot.bottom);
    }
    if (!keep[i]) return;
    ctx.fillStyle = THEME.muted;
    ctx.fillText(format(t), Math.min(Math.max(x, boxes[i].w / 2), plot.right + MARGIN.right - boxes[i].w / 2), plot.bottom + MARGIN.tickBand / 2 + 2);
  });
  ctx.strokeStyle = THEME.axis;
  line(ctx, plot.left, crisp(plot.bottom), plot.right, crisp(plot.bottom));
  if (xLabel) {
    ctx.textAlign = "right";
    ctx.fillStyle = THEME.muted;
    ctx.fillText(xLabel, plot.right, plot.bottom + MARGIN.tickBand + MARGIN.labelBand / 2 + 1);
  }
}

/** Linea verticale "siamo qui" con etichetta. */
function drawVerticalHighlight(ctx, plot, x, label) {
  if (!(x >= plot.left - 0.5 && x <= plot.right + 0.5)) return;
  // Colore del testo, non un colore di serie: la linea "siamo qui" non deve sembrare un dato.
  ctx.strokeStyle = THEME.text;
  ctx.globalAlpha = 0.75;
  ctx.lineWidth = 1.5;
  line(ctx, x, plot.top, x, plot.bottom);
  ctx.globalAlpha = 1;
  if (!label) return;
  ctx.font = FONT_STRONG;
  const w = ctx.measureText(label).width;
  const flip = x + 6 + w > plot.right;
  ctx.textAlign = flip ? "right" : "left";
  ctx.fillStyle = THEME.text;
  ctx.fillText(label, flip ? x - 5 : x + 5, plot.top + 7);
  ctx.font = FONT;
}

function drawEmpty(ctx, { width, height }) {
  ctx.fillStyle = THEME.muted;
  ctx.textAlign = "center";
  ctx.fillText(EMPTY_TEXT, width / 2, height / 2);
}

/** Pallino da 8 px con anello di 2 px del colore dello sfondo (resta leggibile sopra le linee). */
function dot(ctx, x, y, color, r = 4, ring = 2) {
  ctx.beginPath();
  ctx.arc(x, y, r + ring, 0, Math.PI * 2);
  ctx.fillStyle = THEME.surface;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

/** Barra con angoli arrotondati di 4 px solo all'estremità dei dati (la base resta squadrata). */
function bar(ctx, x, y0, w, y1, radius = 4) {
  const up = y1 <= y0;
  const r = Math.min(radius, w / 2, Math.abs(y1 - y0));
  ctx.beginPath();
  if (up) {
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1 + r);
    ctx.arcTo(x, y1, x + r, y1, r);
    ctx.arcTo(x + w, y1, x + w, y1 + r, r);
    ctx.lineTo(x + w, y0);
  } else {
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1 - r);
    ctx.arcTo(x, y1, x + r, y1, r);
    ctx.arcTo(x + w, y1, x + w, y1 - r, r);
    ctx.lineTo(x + w, y0);
  }
  ctx.closePath();
  ctx.fill();
}

const crisp = (v) => Math.round(v) + 0.5; // linee da 1 px nitide su qualunque devicePixelRatio intero
function line(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

/** Tronca un testo con "…" perché stia in `max` px. */
function fitText(ctx, text, max) {
  if (ctx.measureText(text).width <= max) return text;
  let s = text;
  while (s.length > 1 && ctx.measureText(`${s}…`).width > max) s = s.slice(0, -1);
  return `${s}…`;
}

function legendItem(item) {
  const key = el("span", { className: `chart-key chart-key-${item.key}` });
  if (item.gradient) key.style.background = item.gradient;
  else key.style.background = item.color;
  if (item.gradient) {
    return el("div", { className: "chart-legend-scale", role: "listitem" }, el("span", {}, item.min), key, el("span", {}, item.max), item.name ? el("span", { className: "chart-legend-name" }, item.name) : null);
  }
  return el("div", { className: "chart-legend-item", role: "listitem" }, key, el("span", {}, item.name));
}

/** Righe campionate a intervalli regolari (più l'ultima) per le tabelle di serie lunghe. */
function sampleIndices(count, max = TABLE_MAX_ROWS) {
  if (count <= max) return Array.from({ length: count }, (_, i) => i);
  const step = (count - 1) / (max - 1);
  return Array.from({ length: max }, (_, i) => Math.round(i * step));
}

/* ================================================================================================
 * Renderer per tipo. Interfaccia comune:
 *   setData(data), clear(), isEmpty(), preferredHeight(width), draw(ctx, frame),
 *   hitTest(x, y) → {tooltip, anchor, …} | null, drawHover(ctx, hit), legend(), note(), table()
 * ============================================================================================== */

class BaseRenderer {
  constructor(chart, options) {
    this.chart = chart;
    this.options = options;
    this.highlight = null;
    this.ignored = 0; // valori scartati perché non finiti: lo si dice nella nota
  }
  preferredHeight() {
    return 200;
  }
  legend() {
    return [];
  }
  note() {
    return this.ignoredNote();
  }
  ignoredNote(noun = { singular: "valore", plural: "valori" }) {
    if (!this.ignored) return "";
    return this.ignored === 1 ? `1 ${noun.singular} non valido (NaN o infinito) ignorato.` : `${formatValue(this.ignored)} ${noun.plural} non validi (NaN o infiniti) ignorati.`;
  }
  drawHover() {}
}

/* ---------------------------------------------------------------------------------- line ----- */

/** Serie con array tipizzati a crescita geometrica: append() costa O(1) ammortizzato. */
class Series {
  constructor(name, color) {
    this.name = name;
    this.color = color;
    this.xs = new Float64Array(64);
    this.ys = new Float64Array(64);
    this.count = 0;
    this.sorted = true; // x crescenti: abilita ricerca binaria e decimazione
    this.xMin = Infinity;
    this.xMax = -Infinity;
    this.yMin = Infinity;
    this.yMax = -Infinity;
    this.yMinPositive = Infinity; // per la scala log
  }

  push(x, y) {
    if (this.count === this.xs.length) {
      const grow = (a) => {
        const b = new Float64Array(a.length * 2);
        b.set(a);
        return b;
      };
      this.xs = grow(this.xs);
      this.ys = grow(this.ys);
    }
    if (this.count && x < this.xs[this.count - 1]) this.sorted = false;
    this.xs[this.count] = x;
    this.ys[this.count] = y;
    this.count++;
    if (x < this.xMin) this.xMin = x;
    if (x > this.xMax) this.xMax = x;
    if (y < this.yMin) this.yMin = y;
    if (y > this.yMax) this.yMax = y;
    if (y > 0 && y < this.yMinPositive) this.yMinPositive = y;
  }

  nearest(x) {
    if (this.sorted) return nearestIndexSorted(this.xs, this.count, x);
    let best = -1;
    for (let i = 0, d = Infinity; i < this.count; i++) {
      const di = Math.abs(this.xs[i] - x);
      if (di < d) [d, best] = [di, i];
    }
    return best;
  }
}

class LineRenderer extends BaseRenderer {
  constructor(chart, options) {
    super(chart, options);
    this.log = options.yScale === "log";
    this.series = new Map();
  }

  setData({ series = [] } = {}) {
    this.series.clear();
    this.ignored = 0;
    for (const s of series) {
      const target = this.ensure(s.name);
      for (const p of s.points ?? []) this.push(target, p);
    }
  }

  append(name, point) {
    const target = this.ensure(name);
    if (Array.isArray(point?.[0])) for (const p of point) this.push(target, p);
    else this.push(target, point);
  }

  ensure(name = "serie") {
    let s = this.series.get(name);
    if (!s) {
      s = new Series(name, this.chart.colorFor(name));
      this.series.set(name, s);
    }
    return s;
  }

  push(series, p) {
    const x = p?.[0];
    const y = p?.[1];
    if (isFiniteNumber(x) && isFiniteNumber(y) && (!this.log || y > 0)) series.push(x, y);
    else this.ignored++;
  }

  clear() {
    this.series.clear();
    this.ignored = 0;
  }

  isEmpty() {
    for (const s of this.series.values()) if (s.count) return false;
    return true;
  }

  preferredHeight() {
    return 210;
  }

  domains() {
    let [xMin, xMax, yMin, yMax] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const s of this.series.values()) {
      if (!s.count) continue;
      xMin = Math.min(xMin, s.xMin);
      xMax = Math.max(xMax, s.xMax);
      yMin = Math.min(yMin, this.log ? s.yMinPositive : s.yMin);
      yMax = Math.max(yMax, s.yMax);
    }
    if (isFiniteNumber(this.options.yMin)) yMin = this.options.yMin;
    if (isFiniteNumber(this.options.yMax)) yMax = this.options.yMax;
    return { xMin, xMax, yMin, yMax };
  }

  draw(ctx, frame) {
    const { xMin, xMax, yMin, yMax } = this.domains();
    const yAxis = this.log ? logTicks(yMin, yMax, { pad: 1.08 }) : niceTicks(yMin, yMax);
    const yLabels = yAxis.ticks.map((t) => (this.log ? formatValue(t) : formatTick(t, yAxis.step, Math.max(Math.abs(yAxis.min), Math.abs(yAxis.max)))));
    const visible = [...this.series.values()].filter((s) => s.count);
    const endLabels = visible.length <= 4 ? visible.map((s) => `${s.name} ${this.chart.formatValue(s.ys[s.count - 1])}`) : [];
    ctx.font = FONT;
    const right = endLabels.length ? Math.min(110, 14 + endLabels.reduce((m, t) => Math.max(m, ctx.measureText(t).width), 0)) : MARGIN.right;
    const plot = plotArea(ctx, frame, yLabels, { right, xLabel: this.options.xLabel });

    const xDomain = xMin === xMax ? niceTicks(xMin, xMax) : { min: xMin, max: xMax };
    const xAxis = niceTicks(xDomain.min, xDomain.max, { expand: false, count: Math.max(2, Math.floor(plot.width / 70)) });
    const xScale = linearScale([xDomain.min, xDomain.max], [plot.left, plot.right]);
    const yScale = (this.log ? logScale : linearScale)([yAxis.min, yAxis.max], [plot.bottom, plot.top]);
    this.frame = { plot, xScale, yScale };

    drawYAxis(ctx, plot, yAxis.ticks, yScale, yLabels, this.options.yLabel);
    drawXAxis(ctx, plot, xAxis.ticks, xScale, (t) => formatTick(t, xAxis.step, Math.abs(xDomain.max)), this.options.xLabel);

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.left - 6, plot.top - 6, plot.width + 12, plot.height + 12); // lascia respirare i pallini ai bordi
    ctx.clip();
    for (const s of visible) this.drawSeries(ctx, s, xDomain, plot, xScale, yScale);
    ctx.restore();

    if (isFiniteNumber(this.highlight?.x)) {
      drawVerticalHighlight(ctx, plot, xScale(this.highlight.x), `${this.options.xLabel ?? "x"} ${formatValue(this.highlight.x)}`);
    }
    this.drawEndLabels(ctx, visible, endLabels, plot, xScale, yScale);
  }

  drawSeries(ctx, s, xDomain, plot, xScale, yScale) {
    let xs = s.xs;
    let ys = s.ys;
    let start = 0;
    let end = s.count;
    let stride = 1;
    // Decimazione: con più di 2 punti per pixel si disegnano solo primo/min/max/ultimo per colonna.
    if (s.sorted && s.count > plot.width * 2) {
      [start, end] = visibleRange(s.xs, s.count, xDomain.min, xDomain.max);
      const pairs = decimateMinMax(s.xs.subarray(start, end), s.ys.subarray(start, end), end - start, xDomain.min, xDomain.max, plot.width);
      xs = pairs;
      ys = pairs.subarray(1);
      start = 0;
      end = pairs.length / 2;
      stride = 2;
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = start; i < end; i++) {
      const px = xScale(xs[i * stride]);
      const py = yScale(ys[i * stride]);
      if (i === start) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    const last = s.count - 1;
    dot(ctx, xScale(s.xs[last]), yScale(s.ys[last]), s.color); // pallino finale: "siamo arrivati qui"
  }

  /** Etichette dirette a fine linea (nome + ultimo valore), solo se non si sovrappongono. */
  drawEndLabels(ctx, visible, labels, plot, xScale, yScale) {
    if (!labels.length) return;
    ctx.font = FONT;
    const boxes = visible.map((s, i) => {
      const y = yScale(s.ys[s.count - 1]);
      return { x: plot.right + 8, y: y - 6, w: ctx.measureText(labels[i]).width, h: 12, cx: xScale(s.xs[s.count - 1]) };
    });
    const keep = placeLabels(boxes, 1);
    ctx.textAlign = "left";
    ctx.fillStyle = THEME.text;
    boxes.forEach((b, i) => {
      // Solo per le serie che arrivano al bordo destro: altrimenti l'etichetta si staccherebbe dalla linea.
      if (keep[i] && b.cx >= plot.right - 2) ctx.fillText(fitText(ctx, labels[i], b.w + 1), b.x, b.y + 6);
    });
  }

  hitTest(px, py) {
    const { plot, xScale, yScale } = this.frame ?? {};
    if (!plot || px < plot.left - 8 || px > plot.right + 8 || py < plot.top - 8 || py > plot.bottom + 8) return null;
    const x = xScale.invert(px);
    // Il mirino si aggancia al punto più vicino in x fra tutte le serie.
    let snap = null;
    for (const s of this.series.values()) {
      const i = s.nearest(x);
      if (i >= 0 && (snap === null || Math.abs(s.xs[i] - x) < Math.abs(snap - x))) snap = s.xs[i];
    }
    if (snap === null) return null;
    const points = [];
    for (const s of this.series.values()) {
      const i = s.nearest(snap);
      if (i >= 0) points.push({ s, x: s.xs[i], y: s.ys[i] });
    }
    const name = this.options.xLabel ?? "x";
    return {
      x: xScale(snap),
      points: points.map((p) => ({ ...p, px: xScale(p.x), py: yScale(p.y) })),
      anchor: { x: xScale(snap), y: py },
      tooltip: {
        title: `${name} ${formatValue(snap)}`,
        rows: points.map((p) => ({ color: p.s.color, key: "line", value: this.chart.formatValue(p.y), name: p.x === snap ? p.s.name : `${p.s.name} (${name} ${formatValue(p.x)})` })),
      },
    };
  }

  drawHover(ctx, hit) {
    const { plot } = this.frame;
    ctx.strokeStyle = THEME.muted;
    ctx.lineWidth = 1;
    line(ctx, crisp(hit.x), plot.top, crisp(hit.x), plot.bottom);
    for (const p of hit.points) dot(ctx, p.px, p.py, p.s.color);
  }

  legend() {
    const visible = [...this.series.values()];
    return visible.length >= 2 ? visible.map((s) => ({ key: "line", color: s.color, name: s.name })) : [];
  }

  note() {
    const total = [...this.series.values()].reduce((n, s) => n + s.count, 0);
    const parts = [];
    // La decimazione scatta già oltre 2 punti per pixel, ma la si spiega solo quando è vistosa.
    if (this.frame && total > this.frame.plot.width * 10) parts.push(`${formatValue(total)} punti: ogni pixel mostra il minimo e il massimo dei suoi punti.`);
    const ignored = this.ignoredNote({ singular: "punto", plural: "punti" });
    if (ignored) parts.push(ignored);
    return parts.join(" ");
  }

  table() {
    const name = this.options.xLabel ?? "x";
    const rows = [];
    let sampled = false;
    for (const s of this.series.values()) {
      const idx = sampleIndices(s.count, Math.floor(TABLE_MAX_ROWS / Math.max(1, this.series.size)));
      sampled ||= idx.length < s.count;
      for (const i of idx) rows.push([s.name, formatValue(s.xs[i]), this.chart.formatValue(s.ys[i])]);
    }
    return { columns: ["serie", name, this.options.yLabel ?? "y"], rows, note: sampled ? "Righe campionate a intervalli regolari (l'ultima è sempre inclusa)." : "" };
  }
}

/* ----------------------------------------------------------------------------------- bar ----- */

class BarRenderer extends BaseRenderer {
  constructor(chart, options) {
    super(chart, options);
    this.clear();
  }

  setData({ labels = [], values = [], highlight } = {}) {
    this.labels = Array.from(labels, (l) => visibleLabel(l));
    this.values = Array.from(values, (v) => (isFiniteNumber(v) ? v : NaN));
    this.ignored = this.values.filter((v) => Number.isNaN(v)).length;
    if (highlight !== undefined) this.highlight = isFiniteNumber(highlight) ? { index: highlight } : highlight;
  }

  clear() {
    this.labels = [];
    this.values = [];
    this.ignored = 0;
  }

  isEmpty() {
    return !this.values.some((v) => !Number.isNaN(v));
  }

  get highlightIndex() {
    return isFiniteNumber(this.highlight?.index) ? this.highlight.index : isFiniteNumber(this.highlight) ? this.highlight : -1;
  }

  draw(ctx, frame) {
    const finite = this.values.filter((v) => !Number.isNaN(v));
    // Le barre partono sempre da zero: una barra che non parte da zero esagera le differenze.
    const yAxis = niceTicks(Math.min(0, ...finite), Math.max(0, ...finite), { count: 4 });
    const percent = this.options.percent && !this.options.format;
    const yLabels = yAxis.ticks.map((t) => (percent ? formatPercent(t, Math.max(0, -Math.floor(Math.log10(yAxis.step * 100) + 1e-9))) : formatTick(t, yAxis.step, Math.max(Math.abs(yAxis.min), Math.abs(yAxis.max)))));
    const plot = plotArea(ctx, frame, yLabels, { xLabel: this.options.xLabel, top: MARGIN.top + 4 });
    const yScale = linearScale([yAxis.min, yAxis.max], [plot.bottom, plot.top]);
    const layout = bandLayout(this.values.length, plot.width, { maxBar: 24, gap: 2 });
    this.frame = { plot, yScale, layout };
    drawYAxis(ctx, plot, yAxis.ticks, yScale, yLabels, this.options.yLabel);

    const hi = this.highlightIndex;
    const y0 = yScale(0);
    this.values.forEach((v, i) => {
      if (Number.isNaN(v)) return;
      ctx.fillStyle = hi < 0 || i === hi ? PALETTE[0] : THEME.dimmed;
      bar(ctx, plot.left + i * layout.band + layout.offset, y0, layout.bar, yScale(v));
    });
    ctx.strokeStyle = THEME.axis;
    ctx.lineWidth = 1;
    line(ctx, plot.left, crisp(y0), plot.right, crisp(y0));
    this.drawCategoryLabels(ctx, plot, layout);
    this.drawValueLabels(ctx, plot, layout, yScale, hi);
    if (this.options.xLabel) {
      ctx.textAlign = "right";
      ctx.fillStyle = THEME.muted;
      ctx.fillText(this.options.xLabel, plot.right, plot.bottom + MARGIN.tickBand + MARGIN.labelBand / 2 + 1);
    }
  }

  drawCategoryLabels(ctx, plot, layout) {
    ctx.font = FONT;
    ctx.textAlign = "center";
    const texts = this.labels.map((l) => fitText(ctx, l, Math.max(layout.band - 2, 14)));
    const boxes = texts.map((t, i) => {
      const w = ctx.measureText(t).width;
      return { x: plot.left + (i + 0.5) * layout.band - w / 2, y: 0, w, h: 10 };
    });
    // Con molte barre si salta qualche etichetta, a passo costante (il tooltip le ha tutte).
    const every = Math.max(1, Math.ceil(boxes.reduce((m, b) => Math.max(m, b.w + 4), 0) / layout.band));
    ctx.fillStyle = THEME.muted;
    texts.forEach((t, i) => {
      if (i % every === 0) ctx.fillText(t, plot.left + (i + 0.5) * layout.band, plot.bottom + MARGIN.tickBand / 2 + 2);
    });
  }

  /** Valore in cima alle barre se c'è spazio; altrimenti solo per la barra evidenziata e la più alta. */
  drawValueLabels(ctx, plot, layout, yScale, hi) {
    const texts = this.values.map((v) => (Number.isNaN(v) ? "" : this.chart.formatValue(v)));
    ctx.font = FONT;
    const fitsAll = texts.every((t) => ctx.measureText(t).width + 4 <= layout.band);
    let maxIndex = -1;
    this.values.forEach((v, i) => {
      if (!Number.isNaN(v) && (maxIndex < 0 || v > this.values[maxIndex])) maxIndex = i;
    });
    ctx.textAlign = "center";
    texts.forEach((t, i) => {
      if (!t || !(fitsAll || i === hi || i === maxIndex)) return;
      const v = this.values[i];
      ctx.font = i === hi ? FONT_STRONG : FONT;
      ctx.fillStyle = i === hi || hi < 0 ? THEME.text : THEME.muted;
      const y = v >= 0 ? yScale(v) - 8 : yScale(v) + 8;
      ctx.fillText(t, Math.min(Math.max(plot.left + (i + 0.5) * layout.band, plot.left + 12), plot.right - 12), Math.max(plot.top - 14, Math.min(y, plot.bottom - 6)));
    });
    ctx.font = FONT;
  }

  hitTest(px, py) {
    const { plot, layout } = this.frame ?? {};
    if (!plot || !layout.band || px < plot.left || px > plot.right || py < plot.top - 16 || py > plot.bottom + MARGIN.tickBand) return null;
    const i = Math.min(this.values.length - 1, Math.floor((px - plot.left) / layout.band));
    const v = this.values[i];
    return {
      index: i,
      anchor: { x: plot.left + (i + 1) * layout.band - layout.offset, y: py },
      tooltip: { title: this.labels[i] ?? `#${i}`, rows: [{ value: Number.isNaN(v) ? "valore non valido" : this.chart.formatValue(v), color: i === this.highlightIndex || this.highlightIndex < 0 ? PALETTE[0] : THEME.dimmed, key: "rect" }] },
    };
  }

  drawHover(ctx, hit) {
    const { plot, layout } = this.frame;
    // La banda sotto il puntatore si illumina appena: si vede che la barra "risponde".
    ctx.fillStyle = "rgba(226, 232, 240, 0.06)";
    ctx.fillRect(plot.left + hit.index * layout.band, plot.top, layout.band, plot.height);
  }

  legend() {
    const hi = this.highlightIndex;
    if (hi < 0 || hi >= this.values.length) return [];
    return [
      { key: "rect", color: PALETTE[0], name: `${this.options.highlightLabel ?? "evidenziato"}: ${this.labels[hi] ?? hi}` },
      { key: "rect", color: THEME.dimmed, name: "altri" },
    ];
  }

  table() {
    return { columns: [this.options.xLabel ?? "etichetta", this.options.yLabel ?? "valore"], rows: this.values.map((v, i) => [this.labels[i] ?? i, Number.isNaN(v) ? "–" : this.chart.formatValue(v)]) };
  }
}

/* ----------------------------------------------------------------------------- histogram ----- */

class HistogramRenderer extends BaseRenderer {
  setData({ values = [], bins = 40, min, max } = {}) {
    this.hist = histogram(values, { bins, min, max });
    this.stats = summarize(values);
    this.ignored = this.hist?.ignored ?? values.length;
  }

  clear() {
    this.hist = null;
    this.ignored = 0;
  }

  isEmpty() {
    return !this.hist;
  }

  preferredHeight() {
    return 190;
  }

  draw(ctx, frame) {
    const { edges, counts, min, max } = this.hist;
    const yAxis = niceTicks(0, Math.max(1, ...counts), { count: 4 });
    const yLabels = yAxis.ticks.map((t) => formatTick(t, yAxis.step, yAxis.max));
    const plot = plotArea(ctx, frame, yLabels, { xLabel: this.options.xLabel });
    const xScale = linearScale([min, max], [plot.left, plot.right]);
    const yScale = linearScale([yAxis.min, yAxis.max], [plot.bottom, plot.top]);
    const xAxis = niceTicks(min, max, { expand: false, count: Math.max(2, Math.floor(plot.width / 70)) });
    this.frame = { plot, xScale, yScale };
    drawYAxis(ctx, plot, yAxis.ticks, yScale, yLabels, this.options.yLabel ?? "conteggio");

    // Barre contigue separate da uno spazio del colore dello sfondo (2 px se c'è posto, poi 1).
    const binWidth = plot.width / counts.length;
    const gap = binWidth >= 8 ? 2 : binWidth >= 3 ? 1 : 0;
    ctx.fillStyle = PALETTE[0];
    for (let i = 0; i < counts.length; i++) {
      if (!counts[i]) continue;
      const x0 = xScale(edges[i]);
      bar(ctx, x0 + gap / 2, plot.bottom, Math.max(0.5, xScale(edges[i + 1]) - x0 - gap), yScale(counts[i]), binWidth >= 8 ? 2 : 0);
    }
    drawXAxis(ctx, plot, xAxis.ticks, xScale, (t) => formatTick(t, xAxis.step, Math.max(Math.abs(min), Math.abs(max))), this.options.xLabel);
    if (isFiniteNumber(this.highlight?.x)) drawVerticalHighlight(ctx, plot, xScale(this.highlight.x), this.highlight.label ?? formatValue(this.highlight.x));
  }

  hitTest(px, py) {
    const { plot, xScale } = this.frame ?? {};
    if (!plot || px < plot.left || px > plot.right || py < plot.top || py > plot.bottom) return null;
    const { edges, counts, total } = this.hist;
    const i = Math.min(counts.length - 1, Math.max(0, Math.floor(((xScale.invert(px) - edges[0]) / (edges[edges.length - 1] - edges[0])) * counts.length)));
    const closing = i === counts.length - 1 ? "]" : ")";
    return {
      index: i,
      anchor: { x: xScale(edges[i + 1]), y: py },
      tooltip: {
        title: `da ${formatValue(edges[i])} a ${formatValue(edges[i + 1])} [${closing === "]" ? "incluso" : "escluso"}]`,
        rows: [{ color: PALETTE[0], key: "rect", value: formatValue(counts[i]), name: `valori (${formatPercent(total ? counts[i] / total : 0)})` }],
      },
    };
  }

  drawHover(ctx, hit) {
    const { plot, xScale } = this.frame;
    const { edges } = this.hist;
    ctx.fillStyle = "rgba(226, 232, 240, 0.08)";
    ctx.fillRect(xScale(edges[hit.index]), plot.top, xScale(edges[hit.index + 1]) - xScale(edges[hit.index]), plot.height);
  }

  note() {
    const s = this.stats;
    const summary = `n = ${formatValue(s.count)} · media ${formatValue(s.mean)} · dev. std ${formatValue(s.std)} · min ${formatValue(s.min)} · max ${formatValue(s.max)}`;
    return this.ignored ? `${summary}. Valori fuori scala o non validi esclusi dai bin: ${formatValue(this.ignored)}.` : summary;
  }

  table() {
    const { edges, counts } = this.hist;
    return { columns: ["da", "a", "conteggio"], rows: Array.from(counts, (c, i) => [formatValue(edges[i]), formatValue(edges[i + 1]), formatValue(c)]) };
  }
}

/* ------------------------------------------------------------------------------- heatmap ----- */

const COLOR_TABLES = { sequential: buildColorTable(SEQUENTIAL_STOPS), diverging: buildColorTable(DIVERGING_STOPS) };
const gradientCss = (stops) => `linear-gradient(to right, ${stops.join(", ")})`;

class HeatmapRenderer extends BaseRenderer {
  setData({ rows = 0, cols = 0, values = [], rowLabels, colLabels, min, max, diverging = false } = {}) {
    if (rows * cols > values.length) throw new RangeError(`heatmap: servono ${rows}×${cols} = ${rows * cols} valori, ne sono arrivati ${values.length}.`);
    this.rows = rows;
    this.cols = cols;
    this.values = values;
    this.rowLabels = rowLabels?.map(visibleLabel);
    this.colLabels = colLabels?.map(visibleLabel);
    this.diverging = diverging;
    this.domain = colorDomain(values.length > rows * cols ? values.slice(0, rows * cols) : values, { min, max, diverging });
    this.ignored = 0;
    for (let i = 0; i < rows * cols; i++) if (!Number.isFinite(values[i])) this.ignored++;
    this.image = null; // l'immagine delle celle si ricostruisce solo quando cambiano i dati
  }

  clear() {
    this.rows = 0;
    this.cols = 0;
    this.values = [];
    this.image = null;
    this.ignored = 0;
  }

  isEmpty() {
    return !(this.rows > 0 && this.cols > 0) || this.ignored === this.rows * this.cols;
  }

  margins() {
    const left = this.rowLabels ? 34 : 6;
    const bottom = (this.colLabels ? MARGIN.tickBand : 4) + (this.options.xLabel ? MARGIN.labelBand : 0);
    const top = this.options.yLabel ? MARGIN.top - 4 : 4;
    return { left, bottom, top };
  }

  preferredHeight(width) {
    const m = this.margins();
    const cell = (width - m.left - 4) / Math.max(1, this.cols);
    return Math.round(Math.min(340, Math.max(70, cell * Math.max(1, this.rows))) + m.top + m.bottom);
  }

  /** Celle → immagine cols × rows, scalata poi senza sfumature: veloce anche per matrici da 100.000 celle. */
  buildImage() {
    const table = COLOR_TABLES[this.diverging ? "diverging" : "sequential"];
    const canvas = document.createElement("canvas");
    canvas.width = this.cols;
    canvas.height = this.rows;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(this.cols, this.rows);
    const [sr, sg, sb] = [15, 23, 42]; // celle NaN: colore dello sfondo (cioè "vuote")
    for (let i = 0; i < this.rows * this.cols; i++) {
      const v = this.values[i];
      const o = i * 4;
      if (Number.isFinite(v)) {
        const k = Math.round(colorPosition(v, this.domain) * 255) * 3;
        image.data[o] = table[k];
        image.data[o + 1] = table[k + 1];
        image.data[o + 2] = table[k + 2];
      } else {
        image.data[o] = sr;
        image.data[o + 1] = sg;
        image.data[o + 2] = sb;
      }
      image.data[o + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return canvas;
  }

  draw(ctx, frame) {
    const m = this.margins();
    const plot = { left: m.left, top: m.top, right: frame.width - 4, bottom: frame.height - m.bottom };
    plot.width = plot.right - plot.left;
    plot.height = plot.bottom - plot.top;
    const cw = plot.width / this.cols;
    const ch = plot.height / this.rows;
    this.frame = { plot, cw, ch };
    this.image ??= this.buildImage();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.image, plot.left, plot.top, plot.width, plot.height);

    // Celle grandi: separate da una linea sottile del colore dello sfondo, come piastrelle.
    if (cw >= 10 && ch >= 10) {
      ctx.strokeStyle = THEME.surface;
      ctx.lineWidth = 1.5;
      for (let c = 1; c < this.cols; c++) line(ctx, plot.left + c * cw, plot.top, plot.left + c * cw, plot.bottom);
      for (let r = 1; r < this.rows; r++) line(ctx, plot.left, plot.top + r * ch, plot.right, plot.top + r * ch);
    }
    this.drawLabels(ctx, plot, cw, ch);
    if (this.highlight) this.outline(ctx, this.highlight, THEME.text, 2);
  }

  drawLabels(ctx, plot, cw, ch) {
    ctx.font = FONT;
    ctx.fillStyle = THEME.muted;
    if (this.rowLabels) {
      ctx.textAlign = "right";
      const every = Math.max(1, Math.ceil(12 / ch));
      for (let r = 0; r < this.rows; r += every) ctx.fillText(fitText(ctx, this.rowLabels[r] ?? "", plot.left - 6), plot.left - 5, plot.top + (r + 0.5) * ch);
    }
    if (this.colLabels) {
      ctx.textAlign = "center";
      const widest = this.colLabels.reduce((w, l) => Math.max(w, ctx.measureText(l).width), 0);
      const every = Math.max(1, Math.ceil((Math.min(widest, 40) + 4) / cw));
      for (let c = 0; c < this.cols; c += every) ctx.fillText(fitText(ctx, this.colLabels[c] ?? "", Math.max(cw * every - 2, 12)), plot.left + (c + 0.5) * cw, plot.bottom + MARGIN.tickBand / 2 + 2);
    }
    if (this.options.xLabel) {
      ctx.textAlign = "right";
      ctx.fillText(this.options.xLabel, plot.right, plot.bottom + (this.colLabels ? MARGIN.tickBand : 4) + MARGIN.labelBand / 2);
    }
    if (this.options.yLabel) {
      ctx.textAlign = "left";
      ctx.fillText(this.options.yLabel, 0, 8);
    }
  }

  /** Contorno di una cella, di una riga ({row}) o di una colonna ({col}). */
  outline(ctx, { row, col }, color, width) {
    const { plot, cw, ch } = this.frame;
    const hasRow = isFiniteNumber(row);
    const hasCol = isFiniteNumber(col);
    if (!hasRow && !hasCol) return;
    const x = hasCol ? plot.left + col * cw : plot.left;
    const y = hasRow ? plot.top + row * ch : plot.top;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.strokeRect(x + width / 2, y + width / 2, (hasCol ? cw : plot.width) - width, (hasRow ? ch : plot.height) - width);
  }

  hitTest(px, py) {
    const { plot, cw, ch } = this.frame ?? {};
    if (!plot || px < plot.left || px >= plot.right || py < plot.top || py >= plot.bottom) return null;
    const col = Math.min(this.cols - 1, Math.floor((px - plot.left) / cw));
    const row = Math.min(this.rows - 1, Math.floor((py - plot.top) / ch));
    const v = this.values[row * this.cols + col];
    const rowName = this.rowLabels?.[row] ?? row;
    const colName = this.colLabels?.[col] ?? col;
    return {
      row,
      col,
      anchor: { x: plot.left + (col + 1) * cw, y: plot.top + (row + 0.5) * ch },
      tooltip: { title: `${this.options.yLabel ?? "riga"} ${rowName} · ${this.options.xLabel ?? "colonna"} ${colName}`, rows: [{ value: Number.isFinite(v) ? this.chart.formatValue(v) : "valore non valido" }] },
    };
  }

  drawHover(ctx, hit) {
    this.outline(ctx, hit, THEME.accent, 1.5);
  }

  legend() {
    if (this.isEmpty()) return [];
    const stops = this.diverging ? DIVERGING_STOPS : SEQUENTIAL_STOPS;
    const f = (v) => this.chart.formatValue(v);
    return [{ key: "scale", gradient: gradientCss(stops), min: f(this.domain.min), max: f(this.domain.max), name: this.diverging ? "blu < 0, grigio = 0, rosso > 0" : "più chiaro = valore più alto" }];
  }

  table() {
    const maxCols = 24;
    const cols = Math.min(this.cols, maxCols);
    const rowIdx = sampleIndices(this.rows, 48);
    const columns = ["", ...Array.from({ length: cols }, (_, c) => this.colLabels?.[c] ?? c)];
    const rows = rowIdx.map((r) => [this.rowLabels?.[r] ?? r, ...Array.from({ length: cols }, (_, c) => formatValue(this.values[r * this.cols + c]))]);
    const cut = this.cols > maxCols || rowIdx.length < this.rows;
    return { columns, rows, note: cut ? `Matrice ${this.rows} × ${this.cols}: la tabella ne mostra una parte, il tooltip le mostra tutte.` : "" };
  }
}

/* ------------------------------------------------------------------------------- scatter ----- */

class ScatterRenderer extends BaseRenderer {
  constructor(chart, options) {
    super(chart, options);
    this.points = [];
  }

  setData({ points = [] } = {}) {
    this.points = [];
    this.ignored = 0;
    this.append(points);
  }

  append(points) {
    for (const p of Array.isArray(points) ? points : [points]) {
      if (isFiniteNumber(p?.x) && isFiniteNumber(p?.y)) {
        const group = p.group ?? "";
        this.points.push({ x: p.x, y: p.y, label: p.label === undefined ? undefined : visibleLabel(p.label), group, color: this.chart.colorFor(group) });
      } else this.ignored++;
    }
  }

  clear() {
    this.points = [];
    this.ignored = 0;
  }

  isEmpty() {
    return this.points.length === 0;
  }

  preferredHeight() {
    return 260;
  }

  draw(ctx, frame) {
    let [xMin, xMax, yMin, yMax] = [Infinity, -Infinity, Infinity, -Infinity];
    for (const p of this.points) {
      xMin = Math.min(xMin, p.x);
      xMax = Math.max(xMax, p.x);
      yMin = Math.min(yMin, p.y);
      yMax = Math.max(yMax, p.y);
    }
    if (isFiniteNumber(this.options.yMin)) yMin = this.options.yMin;
    if (isFiniteNumber(this.options.yMax)) yMax = this.options.yMax;
    const yAxis = niceTicks(yMin, yMax);
    const yLabels = yAxis.ticks.map((t) => formatTick(t, yAxis.step, Math.max(Math.abs(yAxis.min), Math.abs(yAxis.max))));
    const plot = plotArea(ctx, frame, yLabels, { xLabel: this.options.xLabel });
    const xAxis = niceTicks(xMin, xMax, { count: Math.max(2, Math.floor(plot.width / 70)) });
    const xScale = linearScale([xAxis.min, xAxis.max], [plot.left, plot.right]);
    const yScale = linearScale([yAxis.min, yAxis.max], [plot.bottom, plot.top]);
    this.frame = { plot, xScale, yScale };
    drawYAxis(ctx, plot, yAxis.ticks, yScale, yLabels, this.options.yLabel);
    drawXAxis(ctx, plot, xAxis.ticks, xScale, (t) => formatTick(t, xAxis.step, Math.max(Math.abs(xAxis.min), Math.abs(xAxis.max))), this.options.xLabel, { grid: true });

    // Con migliaia di punti pallini più piccoli e senza anello: resterebbe solo una macchia.
    const dense = this.points.length > 2000;
    for (const p of this.points) {
      p.px = xScale(p.x);
      p.py = yScale(p.y);
      if (dense) {
        ctx.fillStyle = p.color;
        ctx.fillRect(p.px - 1.5, p.py - 1.5, 3, 3);
      } else dot(ctx, p.px, p.py, p.color);
    }
    this.drawLabels(ctx, plot);
    const hi = this.highlighted();
    if (hi) {
      ctx.strokeStyle = THEME.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hi.px, hi.py, 8, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  /** Etichette accanto ai punti, solo dove non si sovrappongono (i punti non etichettati restano nel tooltip). */
  drawLabels(ctx, plot) {
    const labelled = this.points.filter((p) => p.label);
    if (!labelled.length || labelled.length > 400) return;
    ctx.font = FONT;
    ctx.textAlign = "left";
    const boxes = labelled.map((p) => ({ x: p.px + 7, y: p.py - 6, w: ctx.measureText(p.label).width, h: 12 }));
    // I pallini stessi sono ostacoli: un'etichetta non deve coprire un altro punto.
    const dots = this.points.length <= 2000 ? this.points.map((p) => ({ x: p.px - 5, y: p.py - 5, w: 10, h: 10 })) : [];
    const keep = placeLabels(boxes, 1, dots);
    ctx.fillStyle = THEME.text;
    labelled.forEach((p, i) => {
      if (keep[i] && boxes[i].x + boxes[i].w <= plot.right + MARGIN.right) ctx.fillText(p.label, boxes[i].x, p.py);
    });
  }

  highlighted() {
    const h = this.highlight;
    if (!h) return null;
    if (isFiniteNumber(h.index)) return this.points[h.index] ?? null;
    if (h.label !== undefined) return this.points.find((p) => p.label === visibleLabel(h.label)) ?? null;
    return null;
  }

  hitTest(px, py) {
    // Il punto più vicino entro 24 px: non serve centrare un pallino da 8 px.
    let best = null;
    let bestD = 24 * 24;
    for (const p of this.points) {
      const d = (p.px - px) ** 2 + (p.py - py) ** 2;
      if (d < bestD) [bestD, best] = [d, p];
    }
    if (!best) return null;
    const title = [best.label, best.group].filter(Boolean).join(" · ");
    return {
      point: best,
      anchor: { x: best.px, y: best.py },
      tooltip: { title, rows: [{ color: best.color, key: "dot", value: `${this.options.xLabel ?? "x"} ${formatValue(best.x)} · ${this.options.yLabel ?? "y"} ${formatValue(best.y)}` }] },
    };
  }

  drawHover(ctx, hit) {
    ctx.strokeStyle = THEME.accent;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(hit.point.px, hit.point.py, 7, 0, Math.PI * 2);
    ctx.stroke();
  }

  legend() {
    const groups = [...new Set(this.points.map((p) => p.group))];
    return groups.length >= 2 ? groups.map((g) => ({ key: "dot", color: this.chart.colorFor(g), name: g === "" ? "senza gruppo" : String(g) })) : [];
  }

  table() {
    const idx = sampleIndices(this.points.length);
    return {
      columns: ["etichetta", "gruppo", this.options.xLabel ?? "x", this.options.yLabel ?? "y"],
      rows: idx.map((i) => {
        const p = this.points[i];
        return [p.label ?? "", p.group, formatValue(p.x), formatValue(p.y)];
      }),
      note: idx.length < this.points.length ? `Mostrati ${idx.length} punti su ${formatValue(this.points.length)}.` : "",
    };
  }
}

const RENDERERS = { line: LineRenderer, bar: BarRenderer, histogram: HistogramRenderer, heatmap: HeatmapRenderer, scatter: ScatterRenderer };

/* ================================================================================================
 * Stili e icona
 * ============================================================================================== */

/** Inietta css/charts.css una sola volta (index.html non lo elenca: il modulo è autosufficiente). */
function ensureStylesheet() {
  if (document.querySelector("link[data-charts-css]")) return;
  // Risolto rispetto a questo file, così funziona anche da pagine in sottocartelle (tests/).
  const href = new URL("../../../css/charts.css", import.meta.url).href;
  document.head.append(el("link", { rel: "stylesheet", href, dataset: { chartsCss: "" } }));
}

/** I colori del tema arrivano al CSS da qui: una sola fonte di verità per canvas e HTML. */
function applyThemeVariables(root) {
  for (const [name, value] of Object.entries(THEME)) root.style.setProperty(`--chart-${name}`, value);
}

function chartIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("charts-fab-icon");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M2 13.5h12M3.5 11l3-4 3 2.5 4-6");
  svg.append(path);
  return svg;
}
